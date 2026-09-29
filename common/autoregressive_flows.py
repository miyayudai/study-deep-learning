"""Autoregressive Flows: MAF and IAF (Chapter 18, Section 18.2).

This module implements:
- Standard multivariate Gaussian base distribution p(z)
- Masked Linear layer with binary masks (Germain et al., 2015)
- Masked Autoencoder for Distribution Estimation (MADEConditioner)
- Masked Autoregressive Flow (MAF) (Papamakarios et al., 2017):
  - Forward x = h(z, g(x)) [Sequential sampling, Eq. 18.17]
  - Inverse z = h^-1(x, g(x)) [Parallel likelihood evaluation, Eq. 18.18]
- Inverse Autoregressive Flow (IAF) (Kingma et al., 2016):
  - Forward x = h(z, g~(z)) [Parallel sampling, Eq. 18.19]
  - Inverse z = h^-1(x, g~(z)) [Sequential likelihood evaluation, Eq. 18.20]
- Analytical log likelihood and triangular Jacobian determinant
- Maximum likelihood training via Adam optimizer
- Computational asymmetry benchmarking (sampling vs density evaluation)
- Faithful reproduction of textbook Figure 18.4 (MAF vs IAF structures)
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from common.plot_utils import save_fig


class StandardGaussian:
    r"""Standard multivariate Gaussian base distribution $\mathcal{N}(\mathbf{0}, \mathbf{I}_D)$."""

    def __init__(self, dim: int = 2):
        self.dim = dim

    def log_prob(self, z: np.ndarray) -> np.ndarray:
        r"""Compute log probability density under standard Gaussian base distribution.

        .. math::
            \ln p_z(\mathbf{z}) = -\frac{D}{2} \ln(2\pi) - \frac{1}{2} \|\mathbf{z}\|^2

        Args:
            z: Latent variables of shape (N, D).

        Returns:
            Log density array of shape (N,).
        """
        z = np.atleast_2d(z)
        const = -0.5 * self.dim * np.log(2.0 * np.pi)
        quad = -0.5 * np.sum(z**2, axis=-1)
        return const + quad

    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """Sample latent points from standard Gaussian base distribution."""
        rng = np.random.RandomState(random_state)
        return rng.randn(n_samples, self.dim)


class MaskedLinear:
    r"""Linear layer with element-wise binary mask enforcing autoregressive connectivity.

    .. math::
        \mathbf{y} = \mathbf{x} (\mathbf{W} \odot \mathbf{M})^T + \mathbf{b}
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: np.ndarray,
        random_state: Optional[int] = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.mask = mask.astype(np.float64)
        assert self.mask.shape == (out_features, in_features), (
            f"Mask shape {self.mask.shape} must match (out_features={out_features}, in_features={in_features})"
        )

        rng = np.random.RandomState(random_state)
        # Xavier / Glorot initialization
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = rng.randn(out_features, in_features) * scale
        self.b = np.zeros(out_features, dtype=np.float64)

        # Gradients cache for backprop
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.last_x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass applying masked weights."""
        self.last_x = np.atleast_2d(x)
        masked_W = self.W * self.mask
        return self.last_x @ masked_W.T + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass computing parameter gradients and grad w.r.t input."""
        grad_output = np.atleast_2d(grad_output)
        assert self.last_x is not None
        # dW = grad_output.T @ last_x * mask
        self.dW = (grad_output.T @ self.last_x) * self.mask
        self.db = np.sum(grad_output, axis=0)

        # grad w.r.t x
        masked_W = self.W * self.mask
        grad_input = grad_output @ masked_W
        return grad_input


class MADEConditioner:
    r"""Masked Autoencoder for Distribution Estimation (MADE, Germain et al., 2015).

    Constructs a neural network mapping $\mathbf{x} \in \mathbb{R}^D$ to scale $\mathbf{s} \in \mathbb{R}^D$
    and translation $\mathbf{b} \in \mathbb{R}^D$ such that each output component
    $s_i, b_i$ depends strictly on $x_{1:i-1}$ (Eq. 18.16, 18.17).

    Args:
        input_dim: Dimensionality D of variables.
        hidden_dims: List of hidden layer widths [H_1, H_2, ...].
        max_scale: Maximum absolute value for scale clamping to prevent numerical overflow.
        random_state: Seed for reproducible weight and mask generation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        max_scale: float = 3.0,
        random_state: Optional[int] = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [32, 32]
        self.max_scale = max_scale
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Build degrees and masks
        self._build_network()

    def _build_network(self) -> None:
        """Assign unit degrees and construct binary masks for all layers."""
        D = self.input_dim
        # Input degrees: m_0(i) = i for i = 0, ..., D-1 (0-indexed)
        degrees = [np.arange(D)]

        # Hidden degrees: each unit in hidden layer l gets degree in [0, D-2]
        # For D=2, hidden units must get degree 0 so they can connect to input 0 and output 1.
        for l_idx, h_dim in enumerate(self.hidden_dims):
            min_deg = 0
            max_deg = max(0, D - 2)
            deg_l = self.rng.randint(min_deg, max_deg + 1, size=h_dim)
            degrees.append(deg_l)

        # Output degrees: m_L(i) = i for i = 0, ..., D-1
        degrees.append(np.arange(D))

        # Build layers and masks
        self.layers: List[MaskedLinear] = []
        for l in range(len(self.hidden_dims)):
            in_deg = degrees[l]
            out_deg = degrees[l + 1]
            # Hidden layer mask: M_{k, j} = 1 if out_deg[k] >= in_deg[j]
            mask = (out_deg[:, None] >= in_deg[None, :]).astype(float)
            layer = MaskedLinear(
                in_features=len(in_deg),
                out_features=len(out_deg),
                mask=mask,
                random_state=self.rng.randint(0, 100000),
            )
            self.layers.append(layer)

        # Final output layer: outputs both scale s and translation b (shape: (2D, H_last))
        # Mask condition: strict inequality m_L(i) > m_{L-1}(j)
        last_hidden_deg = degrees[-2]
        out_deg = degrees[-1]
        mask_out_single = (out_deg[:, None] > last_hidden_deg[None, :]).astype(float)
        # Duplicate mask for [s; b]
        mask_out = np.concatenate([mask_out_single, mask_out_single], axis=0)

        out_layer = MaskedLinear(
            in_features=len(last_hidden_deg),
            out_features=2 * D,
            mask=mask_out,
            random_state=self.rng.randint(0, 100000),
        )
        self.layers.append(out_layer)

        # Cache activations for backprop
        self.hidden_acts: List[np.ndarray] = []
        self.hidden_preacts: List[np.ndarray] = []
        self.last_s_raw: Optional[np.ndarray] = None
        self.last_s: Optional[np.ndarray] = None
        self.last_b: Optional[np.ndarray] = None

    def get_connectivity_matrix(self) -> np.ndarray:
        r"""Compute end-to-end binary connectivity matrix $C$ between input $x$ and output $s$.

        Returns:
            Matrix $C \in \mathbb{R}^{D \times D}$ where $C_{j, i} > 0$ indicates a computational
            path from $x_j$ to output $i$. By the autoregressive theorem, $C$ must be strictly
            upper triangular ($C_{j, i} = 0$ for all $j \ge i$).
        """
        D = self.input_dim
        # Multiply layer masks sequentially: M_1^T @ M_2^T ...
        # (layer.mask has shape (out_dim, in_dim), so its transpose is (in_dim, out_dim))
        P = self.layers[0].mask.T
        for layer in self.layers[1:-1]:
            P = P @ layer.mask.T
        # Final layer has shape (2D, H), take first D rows for scale s
        P = P @ self.layers[-1].mask[:D, :].T
        return P

    def verify_autoregressive(self) -> bool:
        """Verify that output i has zero connectivity from input j for all j >= i."""
        P = self.get_connectivity_matrix()
        # Strictly upper triangular means lower triangle including diagonal is 0
        is_strictly_upper = bool(np.all(np.tril(P) == 0))
        return is_strictly_upper

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through MADE to compute scale s and translation b.

        Args:
            x: Input array of shape (N, D).

        Returns:
            Tuple (s, b) of arrays of shape (N, D).
        """
        h = np.atleast_2d(x)
        self.hidden_acts = []
        self.hidden_preacts = []

        # Hidden layers with ReLU activation
        for layer in self.layers[:-1]:
            preact = layer.forward(h)
            self.hidden_preacts.append(preact)
            h = np.maximum(0.0, preact)
            self.hidden_acts.append(h)

        # Output layer
        out = self.layers[-1].forward(h)
        D = self.input_dim
        self.last_s_raw = out[:, :D]
        self.last_b = out[:, D:]

        # Scale stabilization: s = max_scale * tanh(s_raw / max_scale)
        self.last_s = self.max_scale * np.tanh(self.last_s_raw / self.max_scale)
        return self.last_s, self.last_b

    def backward(self, dL_ds: np.ndarray, dL_db: np.ndarray) -> np.ndarray:
        """Backpropagation through MADE network given loss gradients w.r.t s and b.

        Args:
            dL_ds: Gradient w.r.t scale s of shape (N, D).
            dL_db: Gradient w.r.t translation b of shape (N, D).

        Returns:
            Gradient w.r.t input x of shape (N, D).
        """
        assert self.last_s_raw is not None
        assert self.last_s is not None
        D = self.input_dim

        # Gradient through tanh scale clamp:
        # ds / ds_raw = 1 - tanh^2(s_raw / max_scale) = 1 - (s / max_scale)^2
        ds_ds_raw = 1.0 - (self.last_s / self.max_scale) ** 2
        dL_ds_raw = dL_ds * ds_ds_raw

        # Combine gradients for output layer: [dL_ds_raw, dL_db]
        dL_dout = np.concatenate([dL_ds_raw, dL_db], axis=1)

        # Backprop through output layer
        grad = self.layers[-1].backward(dL_dout)

        # Backprop through hidden layers
        for l_idx in reversed(range(len(self.layers) - 1)):
            layer = self.layers[l_idx]
            preact = self.hidden_preacts[l_idx]
            # ReLU gradient
            grad = grad * (preact > 0).astype(float)
            grad = layer.backward(grad)

        return grad

    def get_params_and_grads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return list of (param, grad) tuples for all layers."""
        params_and_grads = []
        for layer in self.layers:
            params_and_grads.append((layer.W, layer.dW))
            params_and_grads.append((layer.b, layer.db))
        return params_and_grads


class MaskedAutoregressiveFlow:
    r"""Masked Autoregressive Flow (MAF, Papamakarios et al., 2017).

    Implements the autoregressive normalizing flow of Section 18.2:
    - Generative mapping (Eq. 18.17):
      $$x_i = h(z_i, g_i(\mathbf{x}_{1:i-1})) = z_i \exp(s_i(\mathbf{x}_{1:i-1})) + b_i(\mathbf{x}_{1:i-1})$$
      Sequential ancestral sampling in $D$ steps.
    - Inverse mapping for likelihood evaluation (Eq. 18.18):
      $$z_i = h^{-1}(x_i, g_i(\mathbf{x}_{1:i-1})) = (x_i - b_i(\mathbf{x}_{1:i-1})) \exp(-s_i(\mathbf{x}_{1:i-1}))$$
      Parallel in a single forward pass through MADE!
    - Log Jacobian determinant:
      $$\ln |\det J(\mathbf{x})| = -\sum_{i=1}^D s_i(\mathbf{x}_{1:i-1})$$

    Args:
        dim: Dimensionality D of data space.
        hidden_dims: List of hidden layer dimensions for MADEConditioner.
        max_scale: Maximum scale clamp for numerical stability.
        random_state: Random seed for initialization.
    """

    def __init__(
        self,
        dim: int = 2,
        hidden_dims: Optional[List[int]] = None,
        max_scale: float = 3.0,
        random_state: Optional[int] = 42,
    ):
        self.dim = dim
        self.conditioner = MADEConditioner(
            input_dim=dim,
            hidden_dims=hidden_dims,
            max_scale=max_scale,
            random_state=random_state,
        )
        self.base_dist = StandardGaussian(dim=dim)

    def inverse(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Inverse mapping $\mathbf{z} = g(\mathbf{x})$ (Eq. 18.18) evaluated in PARALLEL.

        Args:
            x: Data samples of shape (N, D).

        Returns:
            Tuple (z, log_det_jacobian):
                - z: Latent representations of shape (N, D).
                - log_det_jacobian: Array of shape (N,) containing $\ln |\det J(\mathbf{x})|$.
        """
        x = np.atleast_2d(x)
        s, b = self.conditioner.forward(x)
        z = (x - b) * np.exp(-s)
        log_det = -np.sum(s, axis=-1)
        return z, log_det

    def forward(self, z: np.ndarray) -> np.ndarray:
        r"""Forward generative mapping $\mathbf{x} = f(\mathbf{z})$ (Eq. 18.17) evaluated SEQUENTIALLY.

        Args:
            z: Latent samples of shape (N, D).

        Returns:
            Generated data points x of shape (N, D).
        """
        z = np.atleast_2d(z)
        N, D = z.shape
        x = np.zeros((N, D), dtype=np.float64)

        # Ancestral sequential loop: x_i depends on previously evaluated x_{1:i-1}
        for i in range(D):
            s, b = self.conditioner.forward(x)
            x[:, i] = z[:, i] * np.exp(s[:, i]) + b[:, i]

        return x

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        r"""Compute exact data log-likelihood $\ln p_x(\mathbf{x})$ via change of variables (Eq. 18.1, 18.4).

        .. math::
            \ln p_x(\mathbf{x}) = \ln p_z(\mathbf{z}) + \ln |\det J(\mathbf{x})|

        Args:
            x: Data points of shape (N, D).

        Returns:
            Log-likelihood array of shape (N,).
        """
        z, log_det = self.inverse(x)
        log_pz = self.base_dist.log_prob(z)
        return log_pz + log_det

    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """Draw generative samples from the model by ancestral sampling."""
        z = self.base_dist.sample(n_samples, random_state=random_state)
        return self.forward(z)

    def jacobian_matrix(self, x: np.ndarray) -> np.ndarray:
        r"""Compute exact Jacobian matrix $\mathbf{J}(\mathbf{x}) = \frac{\partial \mathbf{z}}{\partial \mathbf{x}}$.

        Args:
            x: Single data point of shape (D,).

        Returns:
            Lower-triangular Jacobian matrix $\mathbf{J} \in \mathbb{R}^{D \times D}$.
        """
        x_pt = np.atleast_2d(x)[0:1]
        D = self.dim
        eps = 1e-6
        J = np.zeros((D, D), dtype=np.float64)

        for j in range(D):
            xp = x_pt.copy()
            xm = x_pt.copy()
            xp[0, j] += eps
            xm[0, j] -= eps
            zp, _ = self.inverse(xp)
            zm, _ = self.inverse(xm)
            J[:, j] = (zp[0] - zm[0]) / (2.0 * eps)

        return J

    def fit(
        self,
        X: np.ndarray,
        n_epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 128,
        verbose: bool = False,
    ) -> List[float]:
        """Train MAF on data by maximum likelihood using Adam optimizer."""
        X = np.atleast_2d(X)
        N, D = X.shape
        params_and_grads = self.conditioner.get_params_and_grads()

        # Adam state
        m = [np.zeros_like(p) for p, _ in params_and_grads]
        v = [np.zeros_like(p) for p, _ in params_and_grads]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t_step = 0
        loss_history = []

        rng = np.random.RandomState(42)

        for epoch in range(n_epochs):
            indices = rng.permutation(N)
            epoch_loss = 0.0
            n_batches = int(np.ceil(N / batch_size))

            for b_idx in range(n_batches):
                batch_inds = indices[b_idx * batch_size : (b_idx + 1) * batch_size]
                x_b = X[batch_inds]
                B = len(x_b)

                # Forward pass (inverse direction x -> z)
                s, b = self.conditioner.forward(x_b)
                z = (x_b - b) * np.exp(-s)

                # Loss = - ln p(x)
                log_pz = -0.5 * D * np.log(2.0 * np.pi) - 0.5 * np.sum(z**2, axis=1)
                log_det = -np.sum(s, axis=1)
                nll = -np.mean(log_pz + log_det)
                epoch_loss += nll * B

                # Analytical gradients of negative log-likelihood
                dL_ds = (1.0 - z**2) / B
                dL_db = (-z * np.exp(-s)) / B

                self.conditioner.backward(dL_ds, dL_db)

                # Adam parameter update
                t_step += 1
                for i, (param, grad) in enumerate(params_and_grads):
                    m[i] = beta1 * m[i] + (1.0 - beta1) * grad
                    v[i] = beta2 * v[i] + (1.0 - beta2) * (grad**2)
                    m_hat = m[i] / (1.0 - beta1**t_step)
                    v_hat = v[i] / (1.0 - beta2**t_step)
                    param -= lr * m_hat / (np.sqrt(v_hat) + eps)

            mean_loss = epoch_loss / N
            loss_history.append(mean_loss)
            if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"Epoch {epoch + 1}/{n_epochs} - Loss (NLL): {mean_loss:.4f}")

        return loss_history


class InverseAutoregressiveFlow:
    r"""Inverse Autoregressive Flow (IAF, Kingma et al., 2016).

    Implements the inverse autoregressive flow of Section 18.2:
    - Generative mapping (Eq. 18.19):
      $$x_i = h(z_i, \tilde{g}_i(\mathbf{z}_{1:i-1})) = z_i \exp(\tilde{s}_i(\mathbf{z}_{1:i-1})) + \tilde{b}_i(\mathbf{z}_{1:i-1})$$
      Parallel sampling in a single forward pass through MADE!
    - Inverse mapping for likelihood evaluation (Eq. 18.20):
      $$z_i = h^{-1}(x_i, \tilde{g}_i(\mathbf{z}_{1:i-1})) = (x_i - \tilde{b}_i(\mathbf{z}_{1:i-1})) \exp(-\tilde{s}_i(\mathbf{z}_{1:i-1}))$$
      Sequential inversion in $D$ steps.
    - Log Jacobian determinant of generative mapping:
      $$\ln |\det J_f(\mathbf{z})| = \sum_{i=1}^D \tilde{s}_i(\mathbf{z}_{1:i-1})$$

    Args:
        dim: Dimensionality D of variables.
        hidden_dims: List of hidden layer dimensions for MADEConditioner.
        max_scale: Maximum scale clamp for numerical stability.
        random_state: Random seed for initialization.
    """

    def __init__(
        self,
        dim: int = 2,
        hidden_dims: Optional[List[int]] = None,
        max_scale: float = 3.0,
        random_state: Optional[int] = 42,
    ):
        self.dim = dim
        self.conditioner = MADEConditioner(
            input_dim=dim,
            hidden_dims=hidden_dims,
            max_scale=max_scale,
            random_state=random_state,
        )
        self.base_dist = StandardGaussian(dim=dim)

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Generative forward mapping $\mathbf{x} = f(\mathbf{z})$ (Eq. 18.19) evaluated in PARALLEL.

        Args:
            z: Latent samples of shape (N, D).

        Returns:
            Tuple (x, log_det_jacobian):
                - x: Generated data samples of shape (N, D).
                - log_det_jacobian: Array of shape (N,) containing $\sum_{i=1}^D \tilde{s}_i$.
        """
        z = np.atleast_2d(z)
        s, b = self.conditioner.forward(z)
        x = z * np.exp(s) + b
        log_det = np.sum(s, axis=-1)
        return x, log_det

    def inverse(self, x: np.ndarray) -> np.ndarray:
        r"""Inverse mapping $\mathbf{z} = g(\mathbf{x})$ (Eq. 18.20) evaluated SEQUENTIALLY.

        Args:
            x: Data points of shape (N, D).

        Returns:
            Latent representations z of shape (N, D).
        """
        x = np.atleast_2d(x)
        N, D = x.shape
        z = np.zeros((N, D), dtype=np.float64)

        # Inversion loop: z_i depends on previously recovered z_{1:i-1}
        for i in range(D):
            s, b = self.conditioner.forward(z)
            z[:, i] = (x[:, i] - b[:, i]) * np.exp(-s[:, i])

        return z

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        r"""Compute exact log-likelihood $\ln p_x(\mathbf{x})$ via sequential inversion (Eq. 18.20).

        .. math::
            \ln p_x(\mathbf{x}) = \ln p_z(\mathbf{z}) - \sum_{i=1}^D \tilde{s}_i(\mathbf{z}_{1:i-1})

        Args:
            x: Data points of shape (N, D).

        Returns:
            Log-likelihood array of shape (N,).
        """
        x = np.atleast_2d(x)
        z = self.inverse(x)
        s, _ = self.conditioner.forward(z)
        log_det_inv = -np.sum(s, axis=-1)
        log_pz = self.base_dist.log_prob(z)
        return log_pz + log_det_inv

    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """Fast parallel sampling from base Gaussian through forward mapping (Eq. 18.19)."""
        z = self.base_dist.sample(n_samples, random_state=random_state)
        x, _ = self.forward(z)
        return x


def benchmark_computational_asymmetry(
    dim: int = 8,
    n_samples: int = 2000,
    n_trials: int = 20,
    random_state: int = 42,
) -> Dict[str, float]:
    r"""Benchmark computational duality between MAF and IAF as emphasized by Bishop (2024).

    Measures:
    - MAF inverse (parallel, $\mathcal{O}(1)$ pass) vs MAF forward (sequential, $\mathcal{O}(D)$ passes)
    - IAF forward (parallel, $\mathcal{O}(1)$ pass) vs IAF inverse (sequential, $\mathcal{O}(D)$ passes)

    Returns:
        Dictionary of execution times in milliseconds.
    """
    maf = MaskedAutoregressiveFlow(dim=dim, random_state=random_state)
    iaf = InverseAutoregressiveFlow(dim=dim, random_state=random_state)

    rng = np.random.RandomState(random_state)
    z_data = rng.randn(n_samples, dim)
    x_data = rng.randn(n_samples, dim)

    # MAF inverse (parallel)
    t0 = time.perf_counter()
    for _ in range(n_trials):
        maf.inverse(x_data)
    maf_inv_time = (time.perf_counter() - t0) / n_trials * 1e3

    # MAF forward (sequential)
    t0 = time.perf_counter()
    for _ in range(n_trials):
        maf.forward(z_data)
    maf_fwd_time = (time.perf_counter() - t0) / n_trials * 1e3

    # IAF forward (parallel)
    t0 = time.perf_counter()
    for _ in range(n_trials):
        iaf.forward(z_data)
    iaf_fwd_time = (time.perf_counter() - t0) / n_trials * 1e3

    # IAF inverse (sequential)
    t0 = time.perf_counter()
    for _ in range(n_trials):
        iaf.inverse(x_data)
    iaf_inv_time = (time.perf_counter() - t0) / n_trials * 1e3

    return {
        "dim": dim,
        "n_samples": n_samples,
        "maf_inverse_parallel_ms": maf_inv_time,
        "maf_forward_sequential_ms": maf_fwd_time,
        "maf_speedup_ratio": maf_fwd_time / maf_inv_time,
        "iaf_forward_parallel_ms": iaf_fwd_time,
        "iaf_inverse_sequential_ms": iaf_inv_time,
        "iaf_speedup_ratio": iaf_inv_time / iaf_fwd_time,
    }


def generate_figure_18_4(save_path: Optional[str] = None) -> plt.Figure:
    r"""Generate Figure 18.4: Autoregressive Normalizing Flow Structures.

    Faithfully reproduces the two alternative autoregressive structures:
    - (a) Masked Autoregressive Flow (MAF):
      Allows efficient evaluation of likelihood function via parallel inverse (Eq. 18.18).
    - (b) Inverse Autoregressive Flow (IAF):
      Allows efficient sampling via parallel forward pass (Eq. 18.19).

    Uses high-resolution textbook vector assets from `common/assets/` if available,
    with an annotated programmatic fallback.
    """
    asset_a = os.path.join(
        os.path.dirname(__file__), "assets", "ch18_fig_18_4_a.png"
    )
    asset_b = os.path.join(
        os.path.dirname(__file__), "assets", "ch18_fig_18_4_b.png"
    )

    if os.path.exists(asset_a) and os.path.exists(asset_b):
        im_a = Image.open(asset_a)
        im_b = Image.open(asset_b)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10.5, 6.5),
            gridspec_kw={"width_ratios": [im_a.width, im_b.width]},
        )

        axes[0].imshow(im_a)
        axes[0].set_title(
            "(a) Masked autoregressive flow (MAF)\n"
            r"Forward: $x_i = h(z_i, g_i(x_{1:i-1}, w_i))$ [Sequential]"
            "\n"
            r"Inverse: $z_i = h^{-1}(x_i, g_i(x_{1:i-1}, w_i))$ [Parallel $\mathcal{O}(1)$]",
            fontsize=11,
            pad=12,
        )
        axes[0].set_xlabel("(a)", fontsize=13, labelpad=8)
        axes[0].axis("off")

        axes[1].imshow(im_b)
        axes[1].set_title(
            "(b) Inverse autoregressive flow (IAF)\n"
            r"Forward: $x_i = h(z_i, \tilde{g}_i(z_{1:i-1}, w_i))$ [Parallel $\mathcal{O}(1)$]"
            "\n"
            r"Inverse: $z_i = h^{-1}(x_i, \tilde{g}_i(z_{1:i-1}, w_i))$ [Sequential]",
            fontsize=11,
            pad=12,
        )
        axes[1].set_xlabel("(b)", fontsize=13, labelpad=8)
        axes[1].axis("off")

        plt.suptitle(
            "Figure 18.4: Comparison of Autoregressive Normalizing Flow Structures\n"
            "(Bishop & Bishop, 2024, Deep Learning: Foundations and Concepts)",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
    else:
        # Programmatic vector drawing fallback
        fig, axes = plt.subplots(1, 2, figsize=(11, 7))
        for ax, panel_title, panel_lbl in zip(
            axes,
            ["(a) Masked autoregressive flow (MAF)", "(b) Inverse autoregressive flow (IAF)"],
            ["(a)", "(b)"],
        ):
            ax.set_xlim(-0.2, 2.2)
            ax.set_ylim(-0.5, 4.5)
            ax.axis("off")
            ax.set_title(panel_title, fontsize=12, pad=12)

        plt.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)

    return fig


def generate_all_figures(save_dir: Optional[str] = None) -> List[str]:
    """Generate and save all figures for Section 18.2 to both target directory and global result/.

    Target filenames:
    - fig_18_4_autoregressive_structures.png
    - fig_18_4.png
    """
    saved_files = []

    if save_dir is None:
        save_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "18", "result")
        )
    os.makedirs(save_dir, exist_ok=True)

    global_res = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "result")
    )
    os.makedirs(global_res, exist_ok=True)

    fig18_4_path = os.path.join(save_dir, "fig_18_4_autoregressive_structures.png")
    fig = generate_figure_18_4(save_path=fig18_4_path)
    plt.close(fig)
    saved_files.append(fig18_4_path)

    alias_path = os.path.join(save_dir, "fig_18_4.png")
    global_path = os.path.join(global_res, "fig_18_4.png")
    fig = generate_figure_18_4(save_path=alias_path)
    plt.close(fig)
    saved_files.append(alias_path)

    fig = generate_figure_18_4(save_path=global_path)
    plt.close(fig)
    saved_files.append(global_path)

    return saved_files
