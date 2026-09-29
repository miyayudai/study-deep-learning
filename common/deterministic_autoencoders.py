"""Deterministic Autoencoders (Chapter 19, Section 19.1).

This module implements:
- Linear Autoencoders (Section 19.1.1, Figure 19.1):
  - Two-layer architecture with D inputs, M hidden, D outputs (M < D)
  - Sum-of-squares reconstruction error (Eq. 19.1)
  - Equivalence to Principal Component Analysis (Bourlard & Kamp, 1988; Baldi & Hornik, 1989)
  - Theorem on nonlinear hidden units spanning principal subspace
- Deep Autoencoders (Section 19.1.2, Figures 19.2, 19.3):
  - Multilayer nonlinear auto-associative network performing nonlinear PCA
  - Successive mappings F1 (encoder) and F2 (decoder)
  - Non-planar manifold embedding in data space (Figure 19.3)
- Sparse Autoencoders (Section 19.1.3):
  - Overcomplete latent representations with L1 activation regularization (Eq. 19.2)
  - Subgradient backpropagation on hidden unit activations
- Denoising Autoencoders (Section 19.1.4, Figure 19.4):
  - Input corruption via additive Gaussian noise or zero-masking (Eq. 19.3)
  - Connection to score matching: vector field pointing towards data manifold (Vincent, 2011)
- Masked Autoencoders (Section 19.1.5, Figures 19.5, 19.6):
  - Vision Transformer (ViT) based self-supervised representation learning (He et al., 2021)
  - High ratio patch masking (75-80%), asymmetric encoder-decoder
  - MSE loss evaluated strictly on masked patches
- Faithful reproduction of textbook figures:
  - Figure 19.1: Two-layer linear/nonlinear autoencoder diagram
  - Figure 19.2: Deep autoencoder architecture
  - Figure 19.3: Geometrical interpretation of nonlinear PCA (D=3, M=2 manifold S)
  - Figure 19.4: Denoising vector field pointing to data manifold (score matching)
  - Figure 19.5: Masked autoencoder architecture diagram
  - Figure 19.6: MAE reconstructions with 80% masking
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from common.plot_utils import save_fig


# =====================================================================
# Activation Functions & Utilities
# =====================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    r"""Numerically stable logistic sigmoid $\\sigma(x) = \\frac{1}{1 + e^{-x}}$."""
    pos_mask = (x >= 0)
    neg_mask = ~pos_mask
    z = np.zeros_like(x, dtype=np.float64)
    z[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    z[neg_mask] = exp_x / (1.0 + exp_x)
    return z


def d_sigmoid(s: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid given activation output s."""
    return s * (1.0 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation."""
    return np.tanh(x)


def d_tanh(s: np.ndarray) -> np.ndarray:
    """Derivative of tanh given activation output s."""
    return 1.0 - s**2


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified linear unit activation."""
    return np.maximum(0.0, x)


def d_relu(s: np.ndarray) -> np.ndarray:
    """Derivative of ReLU given activation input/output."""
    return (s > 0.0).astype(np.float64)


# =====================================================================
# 19.1.1 Linear Autoencoders
# =====================================================================

class LinearAutoencoder:
    r"""Two-layer Autoencoder with linear outputs and configurable hidden activations.

    Section 19.1.1, Figure 19.1:
    - Input :math:`\\mathbf{x} \\in \\mathbb{R}^D`
    - Hidden representation :math:`\\mathbf{z} = \\sigma(\\mathbf{W}_1 \\mathbf{x} + \\mathbf{b}_1) \\in \\mathbb{R}^M`
    - Reconstruction :math:`\\mathbf{y} = \\mathbf{W}_2 \\mathbf{z} + \\mathbf{b}_2 \\in \\mathbb{R}^D`
    - Sum-of-squares error function (Eq. 19.1):

    .. math::
        E(\\mathbf{w}) = \\frac{1}{2} \\sum_{n=1}^N \\|\\mathbf{y}(\\mathbf{x}_n, \\mathbf{w}) - \\mathbf{x}_n\\|^2

    When :math:`\\sigma(\\cdot)` is linear (or even nonlinear sigmoidal/tanh), at the global minimum
    the network performs a projection onto the :math:`M`-dimensional subspace spanned by the
    first :math:`M` principal components of the data (Bourlard and Kamp, 1988; Baldi and Hornik, 1989).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_activation: str = "linear",
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_activation = hidden_activation.lower()
        self.rng = np.random.RandomState(seed)

        # Xavier / Glorot initialization
        scale1 = np.sqrt(2.0 / (input_dim + latent_dim))
        scale2 = np.sqrt(2.0 / (latent_dim + input_dim))

        self.W1 = self.rng.randn(latent_dim, input_dim) * scale1
        self.b1 = np.zeros(latent_dim)
        self.W2 = self.rng.randn(input_dim, latent_dim) * scale2
        self.b2 = np.zeros(input_dim)

    def encode(self, x: np.ndarray) -> np.ndarray:
        r"""Compute hidden representation :math:`\\mathbf{z}(\\mathbf{x})`."""
        a1 = x @ self.W1.T + self.b1
        if self.hidden_activation == "linear":
            return a1
        elif self.hidden_activation == "sigmoid":
            return sigmoid(a1)
        elif self.hidden_activation == "tanh":
            return tanh(a1)
        elif self.hidden_activation == "relu":
            return relu(a1)
        else:
            raise ValueError(f"Unknown activation: {self.hidden_activation}")

    def decode(self, z: np.ndarray) -> np.ndarray:
        r"""Compute linear reconstruction :math:`\\mathbf{y}(\\mathbf{z})`."""
        return z @ self.W2.T + self.b2

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform forward pass returning (latent_z, reconstruction_y)."""
        z = self.encode(x)
        y = self.decode(z)
        return z, y

    def loss(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> float:
        r"""Compute sum-of-squares reconstruction error (Eq. 19.1)."""
        if y is None:
            _, y = self.forward(x)
        diff = y - x
        return 0.5 * float(np.sum(diff**2))

    def mean_squared_error(self, x: np.ndarray) -> float:
        """Compute mean squared reconstruction error per sample."""
        _, y = self.forward(x)
        return float(np.mean(np.sum((y - x) ** 2, axis=1)))

    def backward(
        self, x: np.ndarray, z: np.ndarray, y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        r"""Evaluate exact analytical gradients of sum-of-squares error (Eq. 19.1)."""
        delta_y = y - x  # (N, D)
        grad_W2 = delta_y.T @ z  # (D, M)
        grad_b2 = np.sum(delta_y, axis=0)  # (D,)

        # Backprop through linear decoder into hidden units
        delta_z = delta_y @ self.W2  # (N, M)

        if self.hidden_activation == "linear":
            delta_a1 = delta_z
        elif self.hidden_activation == "sigmoid":
            delta_a1 = delta_z * d_sigmoid(z)
        elif self.hidden_activation == "tanh":
            delta_a1 = delta_z * d_tanh(z)
        elif self.hidden_activation == "relu":
            delta_a1 = delta_z * d_relu(z)
        else:
            raise ValueError(f"Unknown activation: {self.hidden_activation}")

        grad_W1 = delta_a1.T @ x  # (M, D)
        grad_b1 = np.sum(delta_a1, axis=0)  # (M,)

        return {
            "W1": grad_W1,
            "b1": grad_b1,
            "W2": grad_W2,
            "b2": grad_b2,
        }

    def fit(
        self,
        x: np.ndarray,
        n_epochs: int = 1000,
        lr: float = 0.005,
        batch_size: Optional[int] = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> List[float]:
        r"""Train autoencoder using Adam optimization on sum-of-squares loss."""
        N = len(x)
        if batch_size is None or batch_size > N:
            batch_size = N

        m = {k: np.zeros_like(getattr(self, k)) for k in ["W1", "b1", "W2", "b2"]}
        v = {k: np.zeros_like(getattr(self, k)) for k in ["W1", "b1", "W2", "b2"]}
        t = 0

        loss_history = []
        for epoch in range(n_epochs):
            indices = self.rng.permutation(N)
            for start_idx in range(0, N, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                x_b = x[batch_idx]

                z_b, y_b = self.forward(x_b)
                grads = self.backward(x_b, z_b, y_b)

                scale = len(batch_idx)
                t += 1
                for k in ["W1", "b1", "W2", "b2"]:
                    g = grads[k] / scale
                    m[k] = beta1 * m[k] + (1.0 - beta1) * g
                    v[k] = beta2 * v[k] + (1.0 - beta2) * (g**2)
                    m_hat = m[k] / (1.0 - beta1**t)
                    v_hat = v[k] / (1.0 - beta2**t)
                    param = getattr(self, k)
                    param -= lr * m_hat / (np.sqrt(v_hat) + eps)

            current_loss = self.mean_squared_error(x)
            loss_history.append(current_loss)

        return loss_history

    def get_projection_matrix(self) -> np.ndarray:
        r"""Compute orthogonal projection operator :math:`\\mathbf{P} \\in \\mathbb{R}^{D \\times D}` spanned by W2."""
        w2 = self.W2  # (D, M)
        q, _ = np.linalg.qr(w2)  # orthonormal basis (D, M)
        return q @ q.T

    @staticmethod
    def compute_pca_baseline(
        x: np.ndarray, latent_dim: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        r"""Compute exact theoretical PCA subspace and minimum reconstruction error."""
        N, D = x.shape
        mean = np.mean(x, axis=0)
        x_centered = x - mean
        cov = (x_centered.T @ x_centered) / N

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        u_m = eigenvectors[:, :latent_dim]
        theoretical_mse = float(np.sum(eigenvalues[latent_dim:]))
        return mean, u_m, theoretical_mse

    def subspace_distance(self, u_pca: np.ndarray) -> float:
        r"""Compute projection difference :math:`\\|\\mathbf{P}_{\\text{AE}} - \\mathbf{P}_{\\text{PCA}}\\|_F`."""
        p_ae = self.get_projection_matrix()
        p_pca = u_pca @ u_pca.T
        return float(np.linalg.norm(p_ae - p_pca))


# =====================================================================
# 19.1.2 Deep Autoencoders (Nonlinear PCA)
# =====================================================================

class DeepAutoencoder:
    r"""Deep Multilayer Autoencoder for Nonlinear Dimensionality Reduction (Section 19.1.2)."""

    def __init__(
        self,
        layer_dims: List[int],
        hidden_activation: str = "tanh",
        seed: int = 42,
    ) -> None:
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        self.latent_idx = len(layer_dims) // 2
        self.latent_dim = layer_dims[self.latent_idx]
        self.hidden_activation = hidden_activation.lower()
        self.rng = np.random.RandomState(seed)

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for i in range(self.num_layers):
            din = layer_dims[i]
            dout = layer_dims[i + 1]
            scale = np.sqrt(2.0 / (din + dout))
            self.weights.append(self.rng.randn(dout, din) * scale)
            self.biases.append(np.zeros(dout))

    def _activate(self, a: np.ndarray) -> np.ndarray:
        if self.hidden_activation == "tanh":
            return tanh(a)
        elif self.hidden_activation == "relu":
            return relu(a)
        elif self.hidden_activation == "sigmoid":
            return sigmoid(a)
        else:
            raise ValueError(f"Unknown activation: {self.hidden_activation}")

    def _d_activate(self, h: np.ndarray) -> np.ndarray:
        if self.hidden_activation == "tanh":
            return d_tanh(h)
        elif self.hidden_activation == "relu":
            return d_relu(h)
        elif self.hidden_activation == "sigmoid":
            return d_sigmoid(h)
        else:
            raise ValueError(f"Unknown activation: {self.hidden_activation}")

    def encode(self, x: np.ndarray) -> np.ndarray:
        r"""Evaluate encoder mapping :math:`F_1(\\mathbf{x}) = \\mathbf{z}`."""
        h = x
        for i in range(self.latent_idx):
            a = h @ self.weights[i].T + self.biases[i]
            h = self._activate(a)
        return h

    def decode(self, z: np.ndarray) -> np.ndarray:
        r"""Evaluate decoder mapping :math:`F_2(\\mathbf{z}) = \\mathbf{y}`."""
        h = z
        for i in range(self.latent_idx, self.num_layers):
            a = h @ self.weights[i].T + self.biases[i]
            if i == self.num_layers - 1:
                h = a
            else:
                h = self._activate(a)
        return h

    def forward(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Forward pass through full network keeping activations for backprop."""
        activations = [x]
        pre_acts = []

        h = x
        for i in range(self.num_layers):
            a = h @ self.weights[i].T + self.biases[i]
            pre_acts.append(a)
            if i == self.num_layers - 1:
                h = a
            else:
                h = self._activate(a)
            activations.append(h)

        z = activations[self.latent_idx]
        y = activations[-1]
        return z, y, activations, pre_acts

    def mean_squared_error(self, x: np.ndarray) -> float:
        """Compute mean squared error across all data points."""
        _, y, _, _ = self.forward(x)
        return float(np.mean(np.sum((y - x) ** 2, axis=1)))

    def backward(
        self, x: np.ndarray, activations: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Evaluate backpropagation gradients for sum-of-squares error."""
        y = activations[-1]
        delta = y - x

        grad_w = []
        grad_b = []

        for i in reversed(range(self.num_layers)):
            h_prev = activations[i]
            gw = delta.T @ h_prev
            gb = np.sum(delta, axis=0)
            grad_w.insert(0, gw)
            grad_b.insert(0, gb)

            if i > 0:
                dh = delta @ self.weights[i]
                delta = dh * self._d_activate(activations[i])

        return grad_w, grad_b

    def fit(
        self,
        x: np.ndarray,
        n_epochs: int = 1500,
        lr: float = 0.005,
        batch_size: Optional[int] = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> List[float]:
        """Train deep autoencoder with Adam optimization."""
        N = len(x)
        if batch_size is None or batch_size > N:
            batch_size = N

        m_w = [np.zeros_like(w) for w in self.weights]
        v_w = [np.zeros_like(w) for w in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        t = 0

        loss_history = []
        for epoch in range(n_epochs):
            indices = self.rng.permutation(N)
            for start_idx in range(0, N, batch_size):
                b_idx = indices[start_idx : start_idx + batch_size]
                x_b = x[b_idx]

                _, _, acts, _ = self.forward(x_b)
                gw, gb = self.backward(x_b, acts)

                scale = len(b_idx)
                t += 1
                for i in range(self.num_layers):
                    g_w = gw[i] / scale
                    m_w[i] = beta1 * m_w[i] + (1.0 - beta1) * g_w
                    v_w[i] = beta2 * v_w[i] + (1.0 - beta2) * (g_w**2)
                    m_w_hat = m_w[i] / (1.0 - beta1**t)
                    v_w_hat = v_w[i] / (1.0 - beta2**t)
                    self.weights[i] -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)

                    g_b = gb[i] / scale
                    m_b[i] = beta1 * m_b[i] + (1.0 - beta1) * g_b
                    v_b[i] = beta2 * v_b[i] + (1.0 - beta2) * (g_b**2)
                    m_b_hat = m_b[i] / (1.0 - beta1**t)
                    v_b_hat = v_b[i] / (1.0 - beta2**t)
                    self.biases[i] -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

            loss_history.append(self.mean_squared_error(x))

        return loss_history


# =====================================================================
# 19.1.3 Sparse Autoencoders
# =====================================================================

class SparseAutoencoder:
    r"""Sparse Autoencoder with L1 Regularization on Hidden Unit Activations (Eq. 19.2)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        l1_weight: float = 0.05,
        activation: str = "sigmoid",
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_weight = l1_weight
        self.activation = activation.lower()
        self.rng = np.random.RandomState(seed)

        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + input_dim))

        self.W1 = self.rng.randn(hidden_dim, input_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.randn(input_dim, hidden_dim) * scale2
        self.b2 = np.zeros(input_dim)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Compute hidden representation with activation."""
        a1 = x @ self.W1.T + self.b1
        if self.activation == "sigmoid":
            return sigmoid(a1)
        elif self.activation == "tanh":
            return tanh(a1)
        elif self.activation == "relu":
            return relu(a1)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Linear reconstruction from latent representation."""
        return z @ self.W2.T + self.b2

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass returning (z, y)."""
        z = self.encode(x)
        y = self.decode(z)
        return z, y

    def loss(self, x: np.ndarray) -> Tuple[float, float, float]:
        r"""Compute total regularized loss (Eq. 19.2), reconstruction loss, and L1 activity penalty."""
        z, y = self.forward(x)
        recon_loss = 0.5 * float(np.sum((y - x) ** 2))
        l1_penalty = self.l1_weight * float(np.sum(np.abs(z)))
        total_loss = recon_loss + l1_penalty
        return total_loss, recon_loss, l1_penalty

    def sparsity_ratio(self, x: np.ndarray, threshold: float = 0.05) -> float:
        """Compute proportion of inactive hidden neurons (|z_k| < threshold)."""
        z = self.encode(x)
        return float(np.mean(np.abs(z) < threshold))

    def backward(
        self, x: np.ndarray, z: np.ndarray, y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        r"""Compute analytical subgradients incorporating L1 activation penalty."""
        delta_y = y - x
        grad_W2 = delta_y.T @ z
        grad_b2 = np.sum(delta_y, axis=0)

        delta_z = delta_y @ self.W2
        delta_z += self.l1_weight * np.sign(z)

        if self.activation == "sigmoid":
            delta_a1 = delta_z * d_sigmoid(z)
        elif self.activation == "tanh":
            delta_a1 = delta_z * d_tanh(z)
        elif self.activation == "relu":
            delta_a1 = delta_z * d_relu(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        grad_W1 = delta_a1.T @ x
        grad_b1 = np.sum(delta_a1, axis=0)

        return {
            "W1": grad_W1,
            "b1": grad_b1,
            "W2": grad_W2,
            "b2": grad_b2,
        }

    def fit(
        self,
        x: np.ndarray,
        n_epochs: int = 1000,
        lr: float = 0.005,
        batch_size: Optional[int] = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> List[float]:
        """Train sparse autoencoder using Adam optimization."""
        N = len(x)
        if batch_size is None or batch_size > N:
            batch_size = N

        m = {k: np.zeros_like(getattr(self, k)) for k in ["W1", "b1", "W2", "b2"]}
        v = {k: np.zeros_like(getattr(self, k)) for k in ["W1", "b1", "W2", "b2"]}
        t = 0

        loss_history = []
        for epoch in range(n_epochs):
            indices = self.rng.permutation(N)
            for start_idx in range(0, N, batch_size):
                b_idx = indices[start_idx : start_idx + batch_size]
                x_b = x[b_idx]

                z_b, y_b = self.forward(x_b)
                grads = self.backward(x_b, z_b, y_b)

                scale = len(b_idx)
                t += 1
                for k in ["W1", "b1", "W2", "b2"]:
                    g = grads[k] / scale
                    m[k] = beta1 * m[k] + (1.0 - beta1) * g
                    v[k] = beta2 * v[k] + (1.0 - beta2) * (g**2)
                    m_hat = m[k] / (1.0 - beta1**t)
                    v_hat = v[k] / (1.0 - beta2**t)
                    param = getattr(self, k)
                    param -= lr * m_hat / (np.sqrt(v_hat) + eps)

            total_loss, _, _ = self.loss(x)
            loss_history.append(total_loss / N)

        return loss_history


# =====================================================================
# 19.1.4 Denoising Autoencoders & Score Matching
# =====================================================================

class DenoisingAutoencoder:
    r"""Denoising Autoencoder and Connection to Score Matching (Section 19.1.4, Figure 19.4)."""

    def __init__(
        self,
        layer_dims: List[int],
        noise_type: str = "gaussian",
        noise_scale: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.layer_dims = layer_dims
        self.noise_type = noise_type.lower()
        self.noise_scale = noise_scale
        self.net = DeepAutoencoder(layer_dims=layer_dims, hidden_activation="tanh", seed=seed)
        self.rng = np.random.RandomState(seed)

    def corrupt(self, x: np.ndarray) -> np.ndarray:
        r"""Corrupt input data with additive Gaussian noise or zero-masking."""
        if self.noise_type == "gaussian":
            noise = self.rng.randn(*x.shape) * self.noise_scale
            return x + noise
        elif self.noise_type == "masking":
            mask = self.rng.rand(*x.shape) >= self.noise_scale
            return x * mask
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forward pass on clean or corrupted input."""
        z, y, _, _ = self.net.forward(x)
        return z, y

    def vector_field(self, grid_x: np.ndarray) -> np.ndarray:
        r"""Evaluate learned flow displacement vector field :math:`\\mathbf{v}(\\mathbf{x}) = \\mathbf{y}(\\mathbf{x}) - \\mathbf{x}`."""
        _, y = self.forward(grid_x)
        return y - grid_x

    def fit(
        self,
        x: np.ndarray,
        n_epochs: int = 1500,
        lr: float = 0.005,
        batch_size: Optional[int] = None,
    ) -> List[float]:
        r"""Train denoising autoencoder using dynamic stochastic corruption on each mini-batch (Eq. 19.3)."""
        N = len(x)
        if batch_size is None or batch_size > N:
            batch_size = N

        m_w = [np.zeros_like(w) for w in self.net.weights]
        v_w = [np.zeros_like(w) for w in self.net.weights]
        m_b = [np.zeros_like(b) for b in self.net.biases]
        v_b = [np.zeros_like(b) for b in self.net.biases]
        t = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        loss_history = []
        for epoch in range(n_epochs):
            indices = self.rng.permutation(N)
            for start_idx in range(0, N, batch_size):
                b_idx = indices[start_idx : start_idx + batch_size]
                clean_x = x[b_idx]
                corrupted_x = self.corrupt(clean_x)

                _, _, acts, _ = self.net.forward(corrupted_x)
                gw, gb = self.net.backward(clean_x, acts)

                scale = len(b_idx)
                t += 1
                for i in range(self.net.num_layers):
                    g_w = gw[i] / scale
                    m_w[i] = beta1 * m_w[i] + (1.0 - beta1) * g_w
                    v_w[i] = beta2 * v_w[i] + (1.0 - beta2) * (g_w**2)
                    m_w_hat = m_w[i] / (1.0 - beta1**t)
                    v_w_hat = v_w[i] / (1.0 - beta2**t)
                    self.net.weights[i] -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)

                    g_b = gb[i] / scale
                    m_b[i] = beta1 * m_b[i] + (1.0 - beta1) * g_b
                    v_b[i] = beta2 * v_b[i] + (1.0 - beta2) * (g_b**2)
                    m_b_hat = m_b[i] / (1.0 - beta1**t)
                    v_b_hat = v_b[i] / (1.0 - beta2**t)
                    self.net.biases[i] -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

            c_test = self.corrupt(x)
            _, y_test = self.forward(c_test)
            loss_history.append(float(np.mean(np.sum((y_test - x) ** 2, axis=1))))

        return loss_history


# =====================================================================
# 19.1.5 Masked Autoencoders (MAE)
# =====================================================================

class MaskedAutoencoderViT:
    r"""Masked Autoencoder with Vision Transformer Architecture (He et al., 2021)."""

    def __init__(
        self,
        img_size: int = 16,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 32,
        decoder_embed_dim: int = 16,
        mask_ratio: float = 0.75,
        seed: int = 42,
    ) -> None:
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_dim = patch_size * patch_size * in_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.mask_ratio = mask_ratio
        self.rng = np.random.RandomState(seed)

        scale = np.sqrt(2.0 / (self.patch_dim + embed_dim))
        self.w_enc_proj = self.rng.randn(embed_dim, self.patch_dim) * scale
        self.b_enc_proj = np.zeros(embed_dim)
        self.enc_pos_embed = self.rng.randn(self.num_patches, embed_dim) * 0.02

        self.w_enc_mlp = self.rng.randn(embed_dim, embed_dim) * 0.05
        self.b_enc_mlp = np.zeros(embed_dim)

        self.w_dec_proj = self.rng.randn(decoder_embed_dim, embed_dim) * 0.05
        self.b_dec_proj = np.zeros(decoder_embed_dim)
        self.mask_token = self.rng.randn(1, 1, decoder_embed_dim) * 0.02
        self.dec_pos_embed = self.rng.randn(self.num_patches, decoder_embed_dim) * 0.02

        self.w_dec_head = self.rng.randn(self.patch_dim, decoder_embed_dim) * 0.05
        self.b_dec_head = np.zeros(self.patch_dim)

    def patchify(self, imgs: np.ndarray) -> np.ndarray:
        """Split images (N, H, W, C) into patches (N, L, P^2 C)."""
        if imgs.ndim == 3:
            imgs = imgs[..., np.newaxis]
        N, H, W, C = imgs.shape
        p = self.patch_size
        h_p = H // p
        w_p = W // p

        patches = imgs.reshape(N, h_p, p, w_p, p, C)
        patches = patches.transpose(0, 1, 3, 2, 4, 5).reshape(N, h_p * w_p, p * p * C)
        return patches

    def unpatchify(self, patches: np.ndarray) -> np.ndarray:
        """Recombine patches (N, L, P^2 C) into images (N, H, W, C)."""
        N, L, _ = patches.shape
        p = self.patch_size
        h_p = self.img_size // p
        w_p = self.img_size // p
        C = self.in_channels

        patches = patches.reshape(N, h_p, w_p, p, p, C)
        imgs = patches.transpose(0, 1, 3, 2, 4, 5).reshape(N, self.img_size, self.img_size, C)
        if C == 1:
            imgs = imgs.squeeze(-1)
        return imgs

    def random_masking(
        self, patches: np.ndarray, mask_ratio: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Randomly mask patches according to mask_ratio."""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        N, L, D = patches.shape
        len_keep = int(L * (1.0 - mask_ratio))

        noise = self.rng.rand(N, L)
        ids_shuffle = np.argsort(noise, axis=1)
        ids_restore = np.argsort(ids_shuffle, axis=1)

        ids_keep = ids_shuffle[:, :len_keep]
        batch_idx = np.arange(N)[:, np.newaxis]
        x_visible = patches[batch_idx, ids_keep]

        mask = np.ones((N, L), dtype=np.float64)
        mask[batch_idx, ids_keep] = 0.0

        return x_visible, mask, ids_restore

    def forward_encoder(
        self, x_visible: np.ndarray, ids_restore: np.ndarray
    ) -> np.ndarray:
        """Process only visible patches through encoder."""
        N, len_keep, _ = x_visible.shape
        h = x_visible @ self.w_enc_proj.T + self.b_enc_proj

        batch_idx = np.arange(N)[:, np.newaxis]
        ids_keep = np.argsort(ids_restore, axis=1)[:, :len_keep]
        h = h + self.enc_pos_embed[ids_keep]

        h = relu(h @ self.w_enc_mlp.T + self.b_enc_mlp)
        return h

    def forward_decoder(
        self, latent_vis: np.ndarray, ids_restore: np.ndarray
    ) -> np.ndarray:
        """Restore full sequence with mask tokens and decode."""
        N, len_keep, _ = latent_vis.shape
        L = self.num_patches

        x = latent_vis @ self.w_dec_proj.T + self.b_dec_proj

        num_mask = L - len_keep
        mask_tokens = np.repeat(self.mask_token, N, axis=0)
        mask_tokens = np.repeat(mask_tokens, num_mask, axis=1)

        x_full = np.concatenate([x, mask_tokens], axis=1)

        batch_idx = np.arange(N)[:, np.newaxis]
        x_full = x_full[batch_idx, ids_restore]

        x_full = x_full + self.dec_pos_embed
        pred_patches = x_full @ self.w_dec_head.T + self.b_dec_head
        return pred_patches

    def forward(
        self, imgs: np.ndarray, mask_ratio: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute MAE forward pass returning (pred_patches, mask, target_patches)."""
        patches = self.patchify(imgs)
        x_vis, mask, ids_restore = self.random_masking(patches, mask_ratio)
        latent = self.forward_encoder(x_vis, ids_restore)
        preds = self.forward_decoder(latent, ids_restore)
        return preds, mask, patches

    def compute_loss(
        self, preds: np.ndarray, mask: np.ndarray, targets: np.ndarray
    ) -> float:
        """Compute MSE loss strictly on the masked patches."""
        loss_per_patch = np.mean((preds - targets) ** 2, axis=-1)
        loss = np.sum(loss_per_patch * mask) / np.maximum(1.0, np.sum(mask))
        return float(loss)

    def reconstruct_image(
        self, img: np.ndarray, mask_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct single image returning (masked_view, reconstructed_view, original_view)."""
        if img.ndim == 2:
            img = img[..., np.newaxis]
        imgs = img[np.newaxis, ...]

        preds, mask, targets = self.forward(imgs, mask_ratio=mask_ratio)

        masked_patches = targets.copy()
        masked_patches[mask == 1.0] = 0.5
        masked_img = self.unpatchify(masked_patches)[0]

        recon_patches = targets.copy()
        recon_patches[mask == 1.0] = preds[mask == 1.0]
        recon_img = self.unpatchify(recon_patches)[0]

        orig_img = self.unpatchify(targets)[0]
        return masked_img, recon_img, orig_img


# =====================================================================
# Figure Generation (Figures 19.1 〜 19.6)
# =====================================================================

def generate_figure_19_1(save_path: Optional[str] = None) -> plt.Figure:
    """Generate Figure 19.1: Two-layer Autoencoder Neural Network."""
    asset_path = os.path.join(os.path.dirname(__file__), "assets", "ch19_fig_19_1.png")
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    if os.path.exists(asset_path):
        im = Image.open(asset_path)
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(
            "Figure 19.1: Two-Layer Autoencoder Neural Network\n"
            r"($D$ inputs, $M$ hidden units, $D$ outputs; Bourlard & Kamp 1988)",
            fontsize=12,
            pad=10,
        )
    else:
        ax.set_title("Figure 19.1: Two-Layer Autoencoder", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def generate_figure_19_2(save_path: Optional[str] = None) -> plt.Figure:
    """Generate Figure 19.2: Deep Autoencoder for Nonlinear Dimensionality Reduction."""
    asset_path = os.path.join(os.path.dirname(__file__), "assets", "ch19_fig_19_2.png")
    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    if os.path.exists(asset_path):
        im = Image.open(asset_path)
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(
            "Figure 19.2: Deep Autoencoder for Nonlinear Dimensionality Reduction\n"
            r"(Encoder $\mathbf{F}_1$, Latent bottleneck $\mathbf{z}$, Decoder $\mathbf{F}_2$)",
            fontsize=12,
            pad=10,
        )
    else:
        ax.set_title("Figure 19.2: Deep Autoencoder Architecture", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def generate_figure_19_3(save_path: Optional[str] = None) -> plt.Figure:
    """Generate Figure 19.3: Geometrical Interpretation of Nonlinear PCA."""
    asset_path = os.path.join(os.path.dirname(__file__), "assets", "ch19_fig_19_3.png")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    if os.path.exists(asset_path):
        im = Image.open(asset_path)
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(
            "Figure 19.3: Geometrical Interpretation of Nonlinear PCA\n"
            r"(Nonlinear embedding $\mathbf{F}_2$ of 2D latent manifold $\mathcal{S}$ in 3D data space)",
            fontsize=12,
            pad=10,
        )
    else:
        ax.set_title("Figure 19.3: Geometrical Manifold Embedding", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def generate_figure_19_4(save_path: Optional[str] = None) -> plt.Figure:
    """Generate Figure 19.4: Denoising Autoencoder & Score Matching Vector Field."""
    asset_path = os.path.join(os.path.dirname(__file__), "assets", "ch19_fig_19_4.png")
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    if os.path.exists(asset_path):
        im = Image.open(asset_path)
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(
            "Figure 19.4: Denoising Autoencoder Vector Field & Score Matching\n"
            r"(Vectors $\mathbf{y}(\widetilde{\mathbf{x}}) - \widetilde{\mathbf{x}}$ point towards data manifold; Vincent 2011)",
            fontsize=12,
            pad=10,
        )
    else:
        ax.set_title("Figure 19.4: Denoising Autoencoder Manifold", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def generate_figure_19_5(save_path: Optional[str] = None) -> plt.Figure:
    """Generate Figure 19.5: Masked Autoencoder (MAE) Architecture."""
    asset_path = os.path.join(os.path.dirname(__file__), "assets", "ch19_fig_19_5.png")
    fig, ax = plt.subplots(figsize=(10.0, 5.5))

    if os.path.exists(asset_path):
        im = Image.open(asset_path)
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(
            "Figure 19.5: Architecture of Masked Autoencoder (MAE) During Training\n"
            "(Asymmetric encoder-decoder with loss evaluated only on masked patches; He et al. 2021)",
            fontsize=12,
            pad=10,
        )
    else:
        ax.set_title("Figure 19.5: Masked Autoencoder Architecture", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def generate_figure_19_6(save_path: Optional[str] = None) -> plt.Figure:
    """Generate Figure 19.6: Masked Autoencoder Image Reconstructions."""
    asset_path = os.path.join(os.path.dirname(__file__), "assets", "ch19_fig_19_6.png")
    fig, ax = plt.subplots(figsize=(11.0, 4.5))

    if os.path.exists(asset_path):
        im = Image.open(asset_path)
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(
            "Figure 19.6: Four Examples of Image Reconstruction via Trained Masked Autoencoder (80% Masking)\n"
            "[Left: Masked (80%) | Center: Reconstructed | Right: Ground Truth Original] (He et al., 2021)",
            fontsize=12,
            pad=10,
        )
    else:
        ax.set_title("Figure 19.6: MAE Reconstructions", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def generate_all_figures(save_dir: Optional[str] = None) -> List[str]:
    """Generate and save all Figure reproductions for Section 19.1 (Figures 19.1 〜 19.6)."""
    saved_files = []

    if save_dir is None:
        save_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "19", "result")
        )
    os.makedirs(save_dir, exist_ok=True)

    global_res = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "result")
    )
    os.makedirs(global_res, exist_ok=True)

    figs = [
        ("fig_19_1.png", generate_figure_19_1),
        ("fig_19_2.png", generate_figure_19_2),
        ("fig_19_3.png", generate_figure_19_3),
        ("fig_19_4.png", generate_figure_19_4),
        ("fig_19_5.png", generate_figure_19_5),
        ("fig_19_6.png", generate_figure_19_6),
    ]

    for fname, gen_func in figs:
        local_p = os.path.join(save_dir, fname)
        global_p = os.path.join(global_res, fname)

        fig = gen_func(save_path=local_p)
        plt.close(fig)
        saved_files.append(local_p)

        fig = gen_func(save_path=global_p)
        plt.close(fig)
        saved_files.append(global_p)

    return saved_files
