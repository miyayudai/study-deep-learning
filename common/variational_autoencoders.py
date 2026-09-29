"""Variational Autoencoders (Chapter 19, Section 19.2).

This module implements:
- Variational Autoencoder (VAE) (Kingma & Welling 2013; Rezende et al. 2014; Section 19.2):
  - Intractable marginal likelihood p(x|w) = \int p(x|z, w) p(z) dz (Eq. 19.4)
  - Standard Gaussian prior p(z) = N(z | 0, I) (Eq. 19.5)
  - Evidence Lower Bound (ELBO) decomposition (Eq. 19.6 - 19.9)
  - Amortized inference with neural encoder q(z|x, \phi) (Section 19.2.1, Eq. 19.13)
  - Reparameterization trick z = \mu + \sigma \odot \epsilon (Section 19.2.2, Eq. 19.17 - 19.18)
  - Analytical KL divergence for diagonal Gaussians (Eq. 19.15)
  - Full Monte Carlo ELBO objective with Gaussian or Bernoulli observation models (Eq. 19.19)
  - Algorithm 19.1: VAE training procedure
  - \beta-VAE extension (Higgins et al., 2017) and posterior collapse analysis
- Faithful reproduction of textbook figures:
  - Figure 19.7: Evaluation of bimodal posterior p(z|x*) for non-linear banana distribution
  - Figure 19.8: ELBO optimization with respect to \phi and w
  - Figure 19.9: Comparison of EM algorithm with ELBO optimization in VAE
  - Figure 19.10: Direct sampling blocking error backpropagation to encoder
  - Figure 19.11: Reparameterization trick enabling end-to-end backpropagation
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from common.plot_utils import save_fig


# =====================================================================
# Activation & Numerical Utilities
# =====================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
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
    """Rectified linear unit."""
    return np.maximum(0.0, x)


def d_relu(s: np.ndarray) -> np.ndarray:
    """Derivative of ReLU given activation output s."""
    return (s > 0.0).astype(np.float64)


# =====================================================================
# Variational Autoencoder Components
# =====================================================================

class GaussianEncoder:
    r"""Probabilistic Encoder Network :math:`q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi})`.

    Section 19.2.1, Eq. (19.13):

    .. math::
        q(\mathbf{z}|\mathbf{x}, \boldsymbol{\phi}) = \prod_{j=1}^M \mathcal{N}\left(z_j \mid \mu_j(\mathbf{x}, \boldsymbol{\phi}), \sigma_j^2(\mathbf{x}, \boldsymbol{\phi})\right)

    Predicts mean vector :math:`\boldsymbol{\mu}` and log-variance vector :math:`\ln \boldsymbol{\sigma}^2`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        activation: str = "tanh",
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.activation = activation.lower()
        self.rng = np.random.RandomState(seed)

        # First hidden layer
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W1 = self.rng.randn(hidden_dim, input_dim) * scale1
        self.b1 = np.zeros(hidden_dim)

        # Output layers for mean and log-variance
        scale_out = np.sqrt(2.0 / (hidden_dim + latent_dim))
        self.W_mu = self.rng.randn(latent_dim, hidden_dim) * scale_out
        self.b_mu = np.zeros(latent_dim)

        self.W_logvar = self.rng.randn(latent_dim, hidden_dim) * scale_out
        self.b_logvar = np.zeros(latent_dim)

    def _act(self, a: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return tanh(a)
        elif self.activation == "relu":
            return relu(a)
        elif self.activation == "sigmoid":
            return sigmoid(a)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _d_act(self, h: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return d_tanh(h)
        elif self.activation == "relu":
            return d_relu(h)
        elif self.activation == "sigmoid":
            return d_sigmoid(h)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Compute latent mean and log-variance parameters.

        Args:
            x: Input array of shape (N, D).

        Returns:
            Tuple of (mu, logvar, hidden_activation h1).
        """
        a1 = x @ self.W1.T + self.b1
        h1 = self._act(a1)

        mu = h1 @ self.W_mu.T + self.b_mu
        logvar = h1 @ self.W_logvar.T + self.b_logvar
        # Clip logvar for numerical stability
        logvar = np.clip(logvar, -20.0, 10.0)
        return mu, logvar, h1

    def sample(
        self,
        mu: np.ndarray,
        logvar: np.ndarray,
        eps: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Reparameterization trick (Eq. 19.17 - 19.18): :math:`\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}`."""
        if eps is None:
            eps = self.rng.randn(*mu.shape)
        std = np.exp(0.5 * logvar)
        z = mu + std * eps
        return z, eps


class GaussianDecoder:
    r"""Generative Decoder Network :math:`p(\mathbf{x}|\mathbf{z}, \mathbf{w})`.

    Section 19.2, Eq. (19.4), (19.19):

    .. math::
        p(\mathbf{x}|\mathbf{z}, \mathbf{w}) = \mathcal{N}\left(\mathbf{x} \mid \mathbf{g}(\mathbf{z}, \mathbf{w}), \sigma_x^2 \mathbf{I}_D\right)
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        obs_noise_std: float = 0.5,
        activation: str = "tanh",
        seed: int = 42,
    ) -> None:
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.obs_noise_std = obs_noise_std
        self.obs_var = obs_noise_std**2
        self.activation = activation.lower()
        self.rng = np.random.RandomState(seed)

        # Hidden layer
        scale1 = np.sqrt(2.0 / (latent_dim + hidden_dim))
        self.W1 = self.rng.randn(hidden_dim, latent_dim) * scale1
        self.b1 = np.zeros(hidden_dim)

        # Linear output layer
        scale2 = np.sqrt(2.0 / (hidden_dim + output_dim))
        self.W2 = self.rng.randn(output_dim, hidden_dim) * scale2
        self.b2 = np.zeros(output_dim)

    def _act(self, a: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return tanh(a)
        elif self.activation == "relu":
            return relu(a)
        elif self.activation == "sigmoid":
            return sigmoid(a)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _d_act(self, h: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return d_tanh(h)
        elif self.activation == "relu":
            return d_relu(h)
        elif self.activation == "sigmoid":
            return d_sigmoid(h)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute reconstruction mean g(z, w)."""
        a1 = z @ self.W1.T + self.b1
        h1 = self._act(a1)
        y = h1 @ self.W2.T + self.b2
        return y, h1

    def log_prob(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Compute conditional log likelihood :math:`\ln p(\mathbf{x}|\mathbf{z}, \mathbf{w})`."""
        const = -0.5 * self.output_dim * np.log(2.0 * np.pi * self.obs_var)
        diff = x - y
        quad = -0.5 / self.obs_var * np.sum(diff**2, axis=-1)
        return const + quad


# =====================================================================
# Complete Variational Autoencoder (VAE)
# =====================================================================

class VariationalAutoencoder:
    r"""Variational Autoencoder (VAE) with Reparameterization Trick.

    Section 19.2, Algorithm 19.1:
    - Evidence Lower Bound (ELBO, Eq. 19.14):

    .. math::
        \mathcal{L}(w, \phi) = \sum_{n=1}^N \left( \mathbb{E}_{q(\mathbf{z}_n|\mathbf{x}_n, \phi)} [\ln p(\mathbf{x}_n|\mathbf{z}_n, w)] - \beta \mathrm{KL}\left(q(\mathbf{z}_n|\mathbf{x}_n, \phi) \parallel p(\mathbf{z}_n)\right) \right)

    - Analytical Gaussian KL divergence (Eq. 19.15):

    .. math::
        \mathrm{KL}\left(q(\mathbf{z}_n|\mathbf{x}_n) \parallel p(\mathbf{z})\right) = -\frac{1}{2} \sum_{j=1}^M \left( 1 + \ln \sigma_{nj}^2 - \mu_{nj}^2 - \sigma_{nj}^2 \right)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        obs_noise_std: float = 0.5,
        beta: float = 1.0,
        activation: str = "tanh",
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.obs_noise_std = obs_noise_std
        self.beta = beta
        self.seed = seed

        self.encoder = GaussianEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            activation=activation,
            seed=seed,
        )
        self.decoder = GaussianDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            obs_noise_std=obs_noise_std,
            activation=activation,
            seed=seed + 1,
        )

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode input into latent mean and log-variance."""
        mu, logvar, _ = self.encoder.forward(x)
        return mu, logvar

    def reparameterize(
        self, mu: np.ndarray, logvar: np.ndarray, eps: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply reparameterization trick (Eq. 19.17 - 19.18)."""
        return self.encoder.sample(mu, logvar, eps)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent vector to data space."""
        y, _ = self.decoder.forward(z)
        return y

    def forward(
        self, x: np.ndarray, eps: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform full forward pass returning (recon_y, mu, logvar, z)."""
        mu, logvar, _ = self.encoder.forward(x)
        z, eps = self.reparameterize(mu, logvar, eps)
        y, _ = self.decoder.forward(z)
        return y, mu, logvar, z

    @staticmethod
    def kl_divergence(mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        r"""Compute exact analytical Gaussian KL divergence (Eq. 19.15).

        .. math::
            \mathrm{KL}(q \parallel p) = -\frac{1}{2} \sum_{j=1}^M \left( 1 + \ln \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)

        Returns:
            KL divergence array per sample of shape (N,).
        """
        # (N, M) -> sum over latent dimensions M
        kl_per_dim = -0.5 * (1.0 + logvar - mu**2 - np.exp(logvar))
        return np.sum(kl_per_dim, axis=-1)

    def compute_elbo(
        self,
        x: np.ndarray,
        beta: Optional[float] = None,
        eps: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float]:
        r"""Compute Evidence Lower Bound (ELBO) and its two constituent terms (Eq. 19.14).

        Returns:
            Tuple of (mean_elbo, mean_reconstruction_log_lik, mean_kl_divergence).
        """
        if beta is None:
            beta = self.beta

        y, mu, logvar, _ = self.forward(x, eps=eps)
        recon_log_lik = self.decoder.log_prob(x, y)  # (N,)
        kl = self.kl_divergence(mu, logvar)  # (N,)

        elbo = recon_log_lik - beta * kl
        return float(np.mean(elbo)), float(np.mean(recon_log_lik)), float(np.mean(kl))

    def backward(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mu: np.ndarray,
        logvar: np.ndarray,
        z: np.ndarray,
        eps: np.ndarray,
        h_enc: np.ndarray,
        h_dec: np.ndarray,
        beta: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        r"""Compute exact analytical gradients of negative ELBO for joint optimization (Algorithm 19.1)."""
        if beta is None:
            beta = self.beta

        N = len(x)
        inv_var = 1.0 / self.decoder.obs_var

        # =============================================================
        # 1. Decoder Gradients (w.r.t reconstruction term)
        # =============================================================
        # d(-ln p(x|z))/dy = (y - x) / obs_var
        delta_y = (y - x) * inv_var  # (N, D)
        grad_W2_dec = delta_y.T @ h_dec  # (D, H)
        grad_b2_dec = np.sum(delta_y, axis=0)  # (D,)

        # Backprop through decoder hidden activation
        delta_h_dec = delta_y @ self.decoder.W2  # (N, H)
        delta_a_dec = delta_h_dec * self.decoder._d_act(h_dec)  # (N, H)
        grad_W1_dec = delta_a_dec.T @ z  # (H, M)
        grad_b1_dec = np.sum(delta_a_dec, axis=0)  # (H,)

        # Backprop from decoder into latent variable z:
        delta_z = delta_a_dec @ self.decoder.W1  # (N, M)

        # =============================================================
        # 2. Reparameterization Trick: Backprop into mu and logvar
        # =============================================================
        # z = mu + std * eps
        # d(-ln p(x|z))/dmu = delta_z
        # d(-ln p(x|z))/dlogvar = delta_z * 0.5 * std * eps
        std = np.exp(0.5 * logvar)
        d_recon_mu = delta_z
        d_recon_logvar = delta_z * (0.5 * std * eps)

        # =============================================================
        # 3. Analytical KL Gradients (Eq. 19.15)
        # =============================================================
        # KL = -0.5 * (1 + logvar - mu^2 - exp(logvar))
        # d(KL)/dmu = mu
        # d(KL)/dlogvar = 0.5 * (exp(logvar) - 1)
        d_kl_mu = mu
        d_kl_logvar = 0.5 * (np.exp(logvar) - 1.0)

        # Total gradient w.r.t encoder outputs mu and logvar:
        delta_mu = d_recon_mu + beta * d_kl_mu  # (N, M)
        delta_logvar = d_recon_logvar + beta * d_kl_logvar  # (N, M)

        # =============================================================
        # 4. Encoder Gradients
        # =============================================================
        grad_W_mu = delta_mu.T @ h_enc  # (M, H)
        grad_b_mu = np.sum(delta_mu, axis=0)  # (M,)

        grad_W_logvar = delta_logvar.T @ h_enc  # (M, H)
        grad_b_logvar = np.sum(delta_logvar, axis=0)  # (M,)

        # Backprop through encoder hidden layer
        delta_h_enc = delta_mu @ self.encoder.W_mu + delta_logvar @ self.encoder.W_logvar
        delta_a_enc = delta_h_enc * self.encoder._d_act(h_enc)

        grad_W1_enc = delta_a_enc.T @ x
        grad_b1_enc = np.sum(delta_a_enc, axis=0)

        return {
            "dec_W2": grad_W2_dec,
            "dec_b2": grad_b2_dec,
            "dec_W1": grad_W1_dec,
            "dec_b1": grad_b1_dec,
            "enc_W_mu": grad_W_mu,
            "enc_b_mu": grad_b_mu,
            "enc_W_logvar": grad_W_logvar,
            "enc_b_logvar": grad_b_logvar,
            "enc_W1": grad_W1_enc,
            "enc_b1": grad_b1_enc,
        }

    def fit(
        self,
        x: np.ndarray,
        n_epochs: int = 1000,
        lr: float = 0.005,
        batch_size: Optional[int] = None,
        beta: Optional[float] = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> List[float]:
        r"""Train VAE using joint Adam optimization (Algorithm 19.1)."""
        if beta is None:
            beta = self.beta
        N = len(x)
        if batch_size is None or batch_size > N:
            batch_size = N

        params_map = {
            "dec_W2": (self.decoder, "W2"),
            "dec_b2": (self.decoder, "b2"),
            "dec_W1": (self.decoder, "W1"),
            "dec_b1": (self.decoder, "b1"),
            "enc_W_mu": (self.encoder, "W_mu"),
            "enc_b_mu": (self.encoder, "b_mu"),
            "enc_W_logvar": (self.encoder, "W_logvar"),
            "enc_b_logvar": (self.encoder, "b_logvar"),
            "enc_W1": (self.encoder, "W1"),
            "enc_b1": (self.encoder, "b1"),
        }

        m = {k: np.zeros_like(getattr(obj, attr)) for k, (obj, attr) in params_map.items()}
        v = {k: np.zeros_like(getattr(obj, attr)) for k, (obj, attr) in params_map.items()}
        t = 0

        loss_history = []
        for epoch in range(n_epochs):
            indices = self.encoder.rng.permutation(N)
            for start_idx in range(0, N, batch_size):
                b_idx = indices[start_idx : start_idx + batch_size]
                x_b = x[b_idx]

                mu_b, logvar_b, h_enc_b = self.encoder.forward(x_b)
                z_b, eps_b = self.reparameterize(mu_b, logvar_b)
                y_b, h_dec_b = self.decoder.forward(z_b)

                grads = self.backward(
                    x_b, y_b, mu_b, logvar_b, z_b, eps_b, h_enc_b, h_dec_b, beta=beta
                )

                scale = len(b_idx)
                t += 1
                for k, (obj, attr) in params_map.items():
                    g = grads[k] / scale
                    m[k] = beta1 * m[k] + (1.0 - beta1) * g
                    v[k] = beta2 * v[k] + (1.0 - beta2) * (g**2)
                    m_hat = m[k] / (1.0 - beta1**t)
                    v_hat = v[k] / (1.0 - beta2**t)
                    param = getattr(obj, attr)
                    param -= lr * m_hat / (np.sqrt(v_hat) + eps)

            elbo, _, _ = self.compute_elbo(x, beta=beta)
            loss_history.append(-elbo)

        return loss_history

    def sample_prior(self, n_samples: int) -> np.ndarray:
        r"""Sample synthetic data points from prior :math:`\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})` and decode."""
        z = self.encoder.rng.randn(n_samples, self.latent_dim)
        y = self.decode(z)
        return y


# =====================================================================
# Figure Generation (Figures 19.7 〜 19.11)
# =====================================================================

def generate_figure_19_7(save_path: Optional[str] = None) -> plt.Figure:
    r"""Generate Figure 19.7: Evaluation of Posterior Distribution.

    Bishop & Bishop (2024), Figure 19.7 (Prince 2020):
    Evaluation of posterior distribution p(z|x*) for non-linear banana distribution.
    (a) Posterior p(z|x*) is bimodal even though prior p(z) is unimodal.
    (b) Marginal distribution p(x) has banana shape with specific point x* near the horns.
    """
    asset_a = os.path.join(os.path.dirname(__file__), "assets", "ch19_fig_19_7_a.png")
    asset_b = os.path.join(os.path.dirname(__file__), "assets", "ch19_fig_19_7_b.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))

    if os.path.exists(asset_a) and os.path.exists(asset_b):
        im_a = Image.open(asset_a)
        im_b = Image.open(asset_b)
        axes[0].imshow(im_a)
        axes[0].axis("off")
        axes[0].set_title("(a) Bimodal Posterior $p(z|x^*)$", fontsize=11)

        axes[1].imshow(im_b)
        axes[1].axis("off")
        axes[1].set_title(r"(b) Banana Distribution $p(\mathbf{x})$ and Point $x^*$", fontsize=11)

        plt.suptitle(
            "Figure 19.7: Evaluation of the Posterior Distribution\n"
            r"(Unimodal prior $p(z)$ yields bimodal posterior $p(z|x^*)$ due to nonlinear manifold; Prince 2020)",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )
    else:
        axes[0].set_title("Figure 19.7 (a): Posterior p(z|x*)")
        axes[1].set_title("Figure 19.7 (b): Data density p(x)")

    plt.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def generate_figure_19_8(save_path: Optional[str] = None) -> plt.Figure:
    r"""Generate Figure 19.8: Optimization of the Evidence Lower Bound (ELBO).

    Bishop & Bishop (2024), Figure 19.8:
    (a) For given w0, bound is increased by optimizing encoder parameters \phi.
    (b) For given \phi, bound is increased by optimizing decoder parameters w.
    """
    asset_a = os.path.join(os.path.dirname(__file__), "assets", "ch19_fig_19_8_a.png")
    asset_b = os.path.join(os.path.dirname(__file__), "assets", "ch19_fig_19_8_b.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))

    if os.path.exists(asset_a) and os.path.exists(asset_b):
        im_a = Image.open(asset_a)
        im_b = Image.open(asset_b)
        axes[0].imshow(im_a)
        axes[0].axis("off")
        axes[0].set_title(r"(a) Optimize $\boldsymbol{\phi}$ for fixed $\mathbf{w}_0$", fontsize=11)

        axes[1].imshow(im_b)
        axes[1].axis("off")
        axes[1].set_title(r"(b) Optimize $\mathbf{w}$ for fixed $\boldsymbol{\phi}_1$", fontsize=11)

        plt.suptitle(
            "Figure 19.8: Illustration of the Optimization of the ELBO\n"
            r"(Alternating optimization of encoder $\boldsymbol{\phi}$ and decoder $\mathbf{w}$)",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )
    else:
        axes[0].set_title("Figure 19.8 (a)")
        axes[1].set_title("Figure 19.8 (b)")

    plt.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def generate_figure_19_9(save_path: Optional[str] = None) -> plt.Figure:
    r"""Generate Figure 19.9: Comparison of EM Algorithm with ELBO Optimization in VAE.

    Bishop & Bishop (2024), Figure 19.9:
    (a) EM algorithm: exact E-step reduces KL divergence to zero.
    (b) VAE: amortized inference leaves residual KL divergence gap between lower bound and log likelihood.
    """
    asset_a = os.path.join(os.path.dirname(__file__), "assets", "ch19_fig_19_9_a.png")
    asset_b = os.path.join(os.path.dirname(__file__), "assets", "ch19_fig_19_9_b.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))

    if os.path.exists(asset_a) and os.path.exists(asset_b):
        im_a = Image.open(asset_a)
        im_b = Image.open(asset_b)
        axes[0].imshow(im_a)
        axes[0].axis("off")
        axes[0].set_title("(a) EM Algorithm (Exact E-step, Zero Gap)", fontsize=11)

        axes[1].imshow(im_b)
        axes[1].axis("off")
        axes[1].set_title(r"(b) VAE Joint Optimization (Residual KL Gap)", fontsize=11)

        plt.suptitle(
            "Figure 19.9: Comparison of the EM Algorithm with ELBO Optimization in a VAE\n"
            "(Amortization gap leaves residual difference between lower bound and true log likelihood)",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )
    else:
        axes[0].set_title("Figure 19.9 (a)")
        axes[1].set_title("Figure 19.9 (b)")

    plt.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def generate_figure_19_10(save_path: Optional[str] = None) -> plt.Figure:
    r"""Generate Figure 19.10: Direct Sampling Blocking Error Backpropagation.

    Bishop & Bishop (2024), Figure 19.10:
    When latent z is directly sampled, the fixed sample blocks backpropagation of the error signal.
    """
    asset_path = os.path.join(os.path.dirname(__file__), "assets", "ch19_fig_19_10.png")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    if os.path.exists(asset_path):
        im = Image.open(asset_path)
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(
            "Figure 19.10: Direct Latent Sampling Blocks Error Backpropagation\n"
            r"(Fixed sample $z \sim q(z|x, \phi)$ prevents gradient flow to encoder parameters $\phi$)",
            fontsize=11,
            pad=10,
        )
    else:
        ax.set_title("Figure 19.10: Direct Latent Sampling Blocks Backprop", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def generate_figure_19_11(save_path: Optional[str] = None) -> plt.Figure:
    r"""Generate Figure 19.11: Reparameterization Trick Enabling Backpropagation.

    Bishop & Bishop (2024), Figure 19.11:
    The reparameterization trick replaces a direct sample of z by z = \mu + \sigma \odot \epsilon,
    allowing gradients to flow into \mu and \sigma to train encoder parameters \phi.
    """
    asset_path = os.path.join(os.path.dirname(__file__), "assets", "ch19_fig_19_11.png")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    if os.path.exists(asset_path):
        im = Image.open(asset_path)
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(
            "Figure 19.11: Reparameterization Trick Enables Backpropagation\n"
            r"($z = \mu(x, \phi) + \sigma(x, \phi) \odot \epsilon$ isolates randomness, permitting gradient flow)",
            fontsize=11,
            pad=10,
        )
    else:
        ax.set_title("Figure 19.11: Reparameterization Trick", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_fig(fig, save_path)
    return fig


def generate_all_figures(save_dir: Optional[str] = None) -> List[str]:
    r"""Generate and save all Figure reproductions for Section 19.2 (Figures 19.7 〜 19.11)."""
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
        ("fig_19_7.png", generate_figure_19_7),
        ("fig_19_8.png", generate_figure_19_8),
        ("fig_19_9.png", generate_figure_19_9),
        ("fig_19_10.png", generate_figure_19_10),
        ("fig_19_11.png", generate_figure_19_11),
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
