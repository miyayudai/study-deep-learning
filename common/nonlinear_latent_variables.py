"""Nonlinear Latent Variable Models (Chapter 16, Section 16.4).

This module implements:
- Change of variables density transformation (Eq. 16.78 - 16.79)
- Nonlinear latent variable generative model (Eq. 16.77, 16.80 - 16.82)
- Monte Carlo marginal likelihood estimation (Eq. 16.83)
- Circle manifold model with exact 2D quadrature (Figure 16.13)
- Discrete observation models: Bernoulli (Eq. 16.84) and Multinomial (Eq. 16.85 - 16.86)
- Uniform dequantization for discrete data (Figure 16.15)
- Pixel distance and likelihood failure analysis (Figure 16.14, Doersch 2016)
- Publication-quality reproduction of Figures 16.11, 16.12, 16.13, 16.14, 16.15
"""

import os
from typing import Callable, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from scipy.stats import norm
from PIL import Image

from common.plot_utils import save_fig


def change_of_variables_density(
    z_pdf: Callable[[np.ndarray], np.ndarray],
    inv_transform: Callable[[np.ndarray], np.ndarray],
    jacobian_det_func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    r"""Evaluate probability density via the change of variables formula (Eq. 16.78 - 16.79).

    .. math::
        p_x(\mathbf{x}) = p_z(\mathbf{z}(\mathbf{x})) \cdot |\det \mathbf{J}(\mathbf{x})|
        \quad \text{where} \quad J_{ij} = \frac{\partial z_i}{\partial x_j}

    Parameters
    ----------
    z_pdf : Callable
        Prior density function in latent space p_z(z).
    inv_transform : Callable
        Inverse mapping z = g^{-1}(x).
    jacobian_det_func : Callable
        Function returning |det J(x)| where J = dz/dx.
    x : np.ndarray
        Evaluation points in data space, shape (N, D) or (D,).

    Returns
    -------
    np.ndarray
        Evaluated density p_x(x), same leading shape as x.
    """
    z = inv_transform(x)
    pz = z_pdf(z)
    abs_det_j = np.abs(jacobian_det_func(x))
    return pz * abs_det_j


class NonlinearLatentVariableModel:
    r"""Nonlinear Continuous Latent Variable Generative Model (Section 16.4.1).

    Generative process:
    1. Sample latent variable from standard isotropic Gaussian prior:
       .. math::
           \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_M) \quad \text{(Eq. 16.77)}
    2. Pass through nonlinear generative mapping:
       .. math::
           \boldsymbol{\mu}_x = \mathbf{g}(\mathbf{z}, \mathbf{w})
    3. Sample observed variable with isotropic Gaussian noise:
       .. math::
           \mathbf{x} \sim \mathcal{N}(\mathbf{g}(\mathbf{z}, \mathbf{w}), \sigma^2 \mathbf{I}_D) \quad \text{(Eq. 16.80)}

    Parameters
    ----------
    generator_func : Callable[[np.ndarray], np.ndarray]
        Nonlinear mapping g(z, w) from (N, M) to (N, D).
    latent_dim : int
        Dimensionality M of latent space z.
    data_dim : int
        Dimensionality D of data space x.
    sigma : float
        Noise standard deviation sigma > 0.
    """

    def __init__(
        self,
        generator_func: Callable[[np.ndarray], np.ndarray],
        latent_dim: int,
        data_dim: int,
        sigma: float = 0.3,
    ) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.generator_func = generator_func
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.sigma = float(sigma)

    def sample(
        self,
        num_samples: int,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw independent samples from the generative model.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        rng : np.random.RandomState, optional
            Random number generator for reproducibility.

        Returns
        -------
        z : np.ndarray, shape (num_samples, latent_dim)
            Sampled latent variables.
        x : np.ndarray, shape (num_samples, data_dim)
            Sampled observed data points.
        """
        if rng is None:
            rng = np.random.RandomState()

        z = rng.randn(num_samples, self.latent_dim)
        g_z = self.generator_func(z)
        noise = self.sigma * rng.randn(num_samples, self.data_dim)
        x = g_z + noise
        return z, x

    def conditional_log_prob(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        r"""Compute conditional log-likelihood ln p(x | z) (Eq. 16.80).

        Parameters
        ----------
        x : np.ndarray, shape (N, D) or (D,)
            Observed points.
        z : np.ndarray, shape (N, M) or (M,)
            Latent points.

        Returns
        -------
        np.ndarray, shape (N,)
            Conditional log likelihood ln N(x | g(z), sigma^2 I).
        """
        x_arr = np.atleast_2d(x)
        z_arr = np.atleast_2d(z)
        g_z = self.generator_func(z_arr)
        sq_diff = np.sum((x_arr - g_z) ** 2, axis=-1)
        d = self.data_dim
        log_norm = -0.5 * d * np.log(2.0 * np.pi * (self.sigma ** 2))
        return log_norm - sq_diff / (2.0 * (self.sigma ** 2))

    def log_marginal_likelihood_mc(
        self,
        x: np.ndarray,
        num_mc_samples: int = 1000,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        r"""Estimate log marginal likelihood ln p(x) via Monte Carlo integration (Eq. 16.83).

        .. math::
            p(\mathbf{x}) \approx \frac{1}{K} \sum_{k=1}^K \mathcal{N}(\mathbf{x} \mid \mathbf{g}(\mathbf{z}_k), \sigma^2 \mathbf{I})

        Uses log-sum-exp trick for numerical stability.

        Parameters
        ----------
        x : np.ndarray, shape (N, D) or (D,)
            Data points to evaluate.
        num_mc_samples : int
            Number of Monte Carlo samples K drawn from p(z).
        rng : np.random.RandomState, optional
            Random state.

        Returns
        -------
        np.ndarray, shape (N,)
            Estimated log marginal likelihood.
        """
        if rng is None:
            rng = np.random.RandomState(42)

        x_arr = np.atleast_2d(x)
        n_pts = x_arr.shape[0]

        # Draw z_k ~ N(0, I)
        z_k = rng.randn(num_mc_samples, self.latent_dim)
        g_zk = self.generator_func(z_k)  # (K, D)

        diff = x_arr[:, np.newaxis, :] - g_zk[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=-1)  # (N, K)

        d = self.data_dim
        log_const = -0.5 * d * np.log(2.0 * np.pi * (self.sigma ** 2))
        log_cond = log_const - sq_dist / (2.0 * (self.sigma ** 2))  # (N, K)

        max_log = np.max(log_cond, axis=1, keepdims=True)
        log_marginal = (
            max_log.squeeze(1)
            + np.log(np.mean(np.exp(log_cond - max_log), axis=1))
        )
        return log_marginal


class CircleManifoldModel:
    r"""Nonlinear Latent Variable Circle Benchmark (Figure 16.13, Bishop / Prince 2020).

    1D latent space :math:`z \sim \mathcal{N}(0, 1)`,
    2D data space :math:`\mathbf{x} = (g_1(z), g_2(z))^T + \boldsymbol{\epsilon}`,
    where :math:`g_1(z) = \sin(z), g_2(z) = \cos(z)`, and
    :math:`\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_2)`.
    """

    def __init__(self, sigma: float = 0.3) -> None:
        self.sigma = float(sigma)

    @staticmethod
    def g(z: np.ndarray) -> np.ndarray:
        """Nonlinear generator mapping g(z) = (sin(z), cos(z))."""
        z_arr = np.asarray(z)
        sin_z = np.sin(z_arr)
        cos_z = np.cos(z_arr)
        return np.stack([sin_z, cos_z], axis=-1)

    def marginal_density_grid(
        self,
        x1_grid: np.ndarray,
        x2_grid: np.ndarray,
        num_quad_points: int = 150,
        z_range: float = 4.0,
    ) -> np.ndarray:
        r"""Compute exact marginal density p(x1, x2) via numerical quadrature (Eq. 16.81).

        .. math::
            p(\mathbf{x}) = \int_{-\infty}^\infty \mathcal{N}(\mathbf{x} \mid \mathbf{g}(z), \sigma^2 \mathbf{I}) \mathcal{N}(z \mid 0, 1) dz

        Parameters
        ----------
        x1_grid : np.ndarray
            1D array of x1 coordinates.
        x2_grid : np.ndarray
            1D array of x2 coordinates.
        num_quad_points : int
            Number of quadrature integration points along z.
        z_range : float
            Truncation range [-z_range, z_range] for latent Gaussian prior.

        Returns
        -------
        density : np.ndarray, shape (len(x2_grid), len(x1_grid))
            Evaluated marginal probability density.
        """
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        z_nodes = np.linspace(-z_range, z_range, num_quad_points)
        dz = z_nodes[1] - z_nodes[0]
        weights = norm.pdf(z_nodes, 0.0, 1.0) * dz

        density = np.zeros_like(X1)
        sigma2 = self.sigma ** 2
        norm_factor = 2.0 * np.pi * sigma2

        for zn, wn in zip(z_nodes, weights):
            mu_1 = np.sin(zn)
            mu_2 = np.cos(zn)
            sq_d = (X1 - mu_1) ** 2 + (X2 - mu_2) ** 2
            density += wn * (np.exp(-0.5 * sq_d / sigma2) / norm_factor)

        return density


class BernoulliObservationModel:
    r"""Discrete Bernoulli Observation Model for Binary Data (Eq. 16.84).

    .. math::
        p(\mathbf{x} \mid \mathbf{z}, \mathbf{w}) = \prod_{i=1}^D g_i(\mathbf{z}, \mathbf{w})^{x_i}
        (1 - g_i(\mathbf{z}, \mathbf{w}))^{1 - x_i}
        \quad \text{where} \quad g_i = \sigma(a_i(\mathbf{z}, \mathbf{w}))
    """

    def __init__(self, logits_func: Callable[[np.ndarray], np.ndarray]) -> None:
        self.logits_func = logits_func

    @staticmethod
    def sigmoid(a: np.ndarray) -> np.ndarray:
        """Numerically stable logistic sigmoid."""
        return np.where(a >= 0, 1.0 / (1.0 + np.exp(-a)), np.exp(a) / (1.0 + np.exp(a)))

    def probabilities(self, z: np.ndarray) -> np.ndarray:
        """Compute success probabilities g_i(z) = sigma(a_i(z))."""
        logits = self.logits_func(z)
        return self.sigmoid(logits)

    def log_prob(self, x: np.ndarray, z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Compute Bernoulli log likelihood."""
        p = self.probabilities(z)
        p = np.clip(p, eps, 1.0 - eps)
        x_arr = np.asarray(x)
        return np.sum(x_arr * np.log(p) + (1.0 - x_arr) * np.log(1.0 - p), axis=-1)

    def sample(
        self,
        z: np.ndarray,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Sample binary vector x in {0, 1}^D given latent vector z."""
        if rng is None:
            rng = np.random.RandomState()
        probs = self.probabilities(z)
        return (rng.rand(*probs.shape) < probs).astype(float)


class MultinomialObservationModel:
    r"""Discrete Multinomial Observation Model for Categorical / One-Hot Data (Eq. 16.85 - 16.86).

    .. math::
        p(\mathbf{x} \mid \mathbf{z}, \mathbf{w}) = \prod_{i=1}^D g_i(\mathbf{z}, \mathbf{w})^{x_i}
        \quad \text{where} \quad g_i = \frac{\exp(a_i)}{\sum_j \exp(a_j)}
    """

    def __init__(self, logits_func: Callable[[np.ndarray], np.ndarray]) -> None:
        self.logits_func = logits_func

    @staticmethod
    def softmax(a: np.ndarray) -> np.ndarray:
        """Numerically stable softmax along last axis."""
        max_a = np.max(a, axis=-1, keepdims=True)
        exp_a = np.exp(a - max_a)
        return exp_a / np.sum(exp_a, axis=-1, keepdims=True)

    def probabilities(self, z: np.ndarray) -> np.ndarray:
        """Compute categorical probabilities g_i(z)."""
        logits = self.logits_func(z)
        return self.softmax(logits)

    def log_prob(self, x: np.ndarray, z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Compute multinomial log likelihood."""
        probs = np.clip(self.probabilities(z), eps, 1.0)
        x_arr = np.asarray(x)
        return np.sum(x_arr * np.log(probs), axis=-1)

    def sample(
        self,
        z: np.ndarray,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Sample one-hot categorical vector given latent vector z."""
        if rng is None:
            rng = np.random.RandomState()
        probs = self.probabilities(z)
        z_shape = probs.shape[:-1]
        k = probs.shape[-1]
        flat_probs = probs.reshape(-1, k)
        samples = np.zeros_like(flat_probs)
        for i, p in enumerate(flat_probs):
            cat = rng.choice(k, p=p)
            samples[i, cat] = 1.0
        return samples.reshape(*z_shape, k)


def uniform_dequantize(
    x_discrete: np.ndarray,
    scale: float = 1.0,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    r"""Add uniform noise u ~ U(0, scale) to dequantize discrete data (Section 16.4.3).

    .. math::
        \mathbf{y} = \mathbf{x} + \mathbf{u}, \quad \mathbf{u} \sim \mathcal{U}(\mathbf{0}, \text{scale} \cdot \mathbf{I})

    Parameters
    ----------
    x_discrete : np.ndarray
        Array of discrete integer values (e.g. 0..255).
    scale : float
        Width of quantization interval (typically 1.0).
    rng : np.random.RandomState, optional
        Random state.

    Returns
    -------
    np.ndarray
        Continuous dequantized data y.
    """
    if rng is None:
        rng = np.random.RandomState()
    u = rng.uniform(0.0, scale, size=x_discrete.shape)
    return x_discrete.astype(float) + u


class PixelLikelihoodComparison:
    """Analyze pixelwise MSE vs perceptual quality (Figure 16.14, Doersch 2016)."""

    @staticmethod
    def compute_gaussian_log_likelihood(
        target: np.ndarray,
        candidate: np.ndarray,
        sigma: float,
    ) -> float:
        r"""Compute log N(candidate | target, sigma^2 I) normalized by pixel count."""
        mse = np.mean((target - candidate) ** 2)
        return -0.5 * np.log(2.0 * np.pi * (sigma ** 2)) - mse / (2.0 * (sigma ** 2))


# =====================================================================
# Figure Generators for Figures 16.11, 16.12, 16.13, 16.14, 16.15
# =====================================================================

def generate_figure_16_11(
    save_path: Optional[str] = "result/fig_16_11_nonlinear_manifold.png",
) -> plt.Figure:
    """Generate Figure 16.11: 2D latent space mapped to 3D manifold."""
    fig = plt.figure(figsize=(10, 4.5))

    # (1) 2D Latent space
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.annotate('', xy=(2.4, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5))
    ax1.annotate('', xy=(0, 2.4), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5))
    ax1.text(2.45, -0.05, r'$z_1$', fontsize=13, va='center')
    ax1.text(-0.05, 2.45, r'$z_2$', fontsize=13, ha='right')

    square = patches.Rectangle((0.4, 0.4), 1.5, 1.5, facecolor='#ff6b6b', alpha=0.9, edgecolor='none')
    ax1.add_patch(square)

    z_pt = (1.1, 1.1)
    ax1.plot(z_pt[0], z_pt[1], 'ko', markersize=6)
    ax1.text(z_pt[0] + 0.1, z_pt[1] - 0.08, r'$\mathbf{z}$', fontsize=13, fontweight='bold')

    ax1.set_xlim(-0.3, 2.7)
    ax1.set_ylim(-0.3, 2.7)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # (2) 3D Data space
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    u = np.linspace(-1.3, 1.3, 40)
    v = np.linspace(-1.3, 1.3, 40)
    U, V = np.meshgrid(u, v)
    W = 0.35 * (U**2 - V**2) + 0.2 * np.sin(1.8 * U)

    ax2.plot_surface(U, V, W, color='#ff6b6b', alpha=0.9, edgecolor='none', shade=True, antialiased=True)

    u0, v0 = 0.3, 0.2
    w0 = 0.35 * (u0**2 - v0**2) + 0.2 * np.sin(1.8 * u0)
    ax2.scatter([u0], [v0], [w0], color='black', s=45, zorder=30)
    ax2.text(u0 + 0.15, v0, w0 + 0.12, r'$\mathbf{x}$', fontsize=13, fontweight='bold')

    ax2.view_init(elev=20, azim=-55)
    ax2.axis('off')

    ax2.quiver(0, 0, -1.2, 2.2, 0, 0, color='black', arrow_length_ratio=0.1, lw=1.5)
    ax2.quiver(0, 0, -1.2, 0, 2.2, 0, color='black', arrow_length_ratio=0.1, lw=1.5)
    ax2.quiver(0, 0, -1.2, 0, 0, 2.4, color='black', arrow_length_ratio=0.1, lw=1.5)
    ax2.text(2.35, 0, -1.2, r'$x_1$', fontsize=13)
    ax2.text(0, 2.35, -1.2, r'$x_2$', fontsize=13)
    ax2.text(0, 0, 1.35, r'$x_3$', fontsize=13)

    ax2.set_xlim(-2.2, 2.2)
    ax2.set_ylim(-2.2, 2.2)
    ax2.set_zlim(-1.5, 1.5)

    con = ConnectionPatch(xyA=z_pt, xyB=(0.58, 0.58), coordsA='data', coordsB='axes fraction',
                          axesA=ax1, axesB=ax2,
                          arrowstyle="-|>", connectionstyle="arc3,rad=-0.32",
                          color='black', lw=1.6, mutation_scale=16)
    fig.add_artist(con)
    fig.text(0.51, 0.83, r'$\mathbf{g}(\mathbf{z}, \mathbf{w})$', fontsize=14, ha='center', va='center', fontweight='bold')

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path, dpi=300)
    return fig


def generate_figure_16_12(
    save_path: Optional[str] = "result/fig_16_12_nonlinear_latent_graphical_model.png",
) -> plt.Figure:
    """Generate Figure 16.12: Directed Graphical Model z -> x."""
    fig, ax = plt.subplots(figsize=(2.5, 3.8))

    circle_z = patches.Circle((0.5, 0.72), 0.16, edgecolor='#d62728', facecolor='white', lw=2.0)
    ax.add_patch(circle_z)
    ax.text(0.5, 0.72, r'$\mathbf{z}$', fontsize=16, ha='center', va='center', fontweight='bold')

    ax.annotate('', xy=(0.5, 0.38), xytext=(0.5, 0.56),
                arrowprops=dict(arrowstyle="-|>", color='#d62728', lw=2.0, mutation_scale=15))

    circle_x = patches.Circle((0.5, 0.22), 0.16, edgecolor='#d62728', facecolor='white', lw=2.0)
    ax.add_patch(circle_x)
    ax.text(0.5, 0.22, r'$\mathbf{x}$', fontsize=16, ha='center', va='center', fontweight='bold')

    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.0, 0.95)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path, dpi=300)
    return fig


def generate_figure_16_13(
    save_path: Optional[str] = "result/fig_16_13_nonlinear_latent_variable_model.png",
) -> plt.Figure:
    """Generate Figure 16.13: (a) Gaussian prior p(z), (b) 3D conditionals and marginal density."""
    fig = plt.figure(figsize=(10.5, 4.5))

    # (a) Prior distribution
    ax1 = fig.add_subplot(1, 2, 1)
    z_vals = np.linspace(-5, 5, 200)
    p_z = norm.pdf(z_vals, 0, 1)
    ax1.plot(z_vals, p_z, color='red', linewidth=2.0)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(0, 0.45)
    ax1.set_xticks([-4, -2, 0, 2, 4])
    ax1.set_xlabel(r'$z$', fontsize=12)
    ax1.set_ylabel(r'$p(z)$', fontsize=12)
    ax1.set_title('(a)', y=-0.25, fontsize=12)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax1.spines[spine].set_color('black')
        ax1.spines[spine].set_linewidth(1.0)

    # (b) 3D visualization
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    sigma = 0.3
    z_slices = [-1.8, 0.0, 1.8]
    z_marginal = 4.2

    x1_grid = np.linspace(-1.6, 1.6, 70)
    x2_grid = np.linspace(-1.6, 1.6, 70)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)

    for zs in z_slices:
        mu_1 = np.sin(zs)
        mu_2 = np.cos(zs)
        density = np.exp(-((X1 - mu_1)**2 + (X2 - mu_2)**2) / (2 * sigma**2))
        colors = cm.inferno(density)
        Z_pos = np.full_like(X1, zs)
        ax2.plot_surface(Z_pos, X1, X2, facecolors=colors, shade=False, rstride=1, cstride=1, alpha=0.98)

    marginal_density = np.zeros_like(X1)
    quad_z = np.linspace(-3.5, 3.5, 120)
    quad_w = norm.pdf(quad_z, 0, 1) * (quad_z[1] - quad_z[0])
    for qz, qw in zip(quad_z, quad_w):
        mu_1 = np.sin(qz)
        mu_2 = np.cos(qz)
        cond = np.exp(-((X1 - mu_1)**2 + (X2 - mu_2)**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        marginal_density += qw * cond

    norm_marginal = marginal_density / np.max(marginal_density)
    colors_marg = cm.inferno(norm_marginal)
    ax2.plot_surface(np.full_like(X1, z_marginal), X1, X2, facecolors=colors_marg, shade=False, rstride=1, cstride=1, alpha=0.98)

    z_curve = np.linspace(-2.0, 2.0, 100)
    x1_curve = np.sin(z_curve)
    x2_curve = np.cos(z_curve)
    ax2.plot(z_curve, x1_curve, x2_curve, color='white', linewidth=2.5, zorder=20)
    tip_z = z_curve[-1]
    tip_x1 = x1_curve[-1]
    tip_x2 = x2_curve[-1]
    dz = z_curve[-1] - z_curve[-5]
    dx1 = x1_curve[-1] - x1_curve[-5]
    dx2 = x2_curve[-1] - x2_curve[-5]
    ax2.quiver(tip_z, tip_x1, tip_x2, dz, dx1, dx2, color='white', arrow_length_ratio=0.8, lw=2.5, zorder=25)

    ax2.set_xlabel(r'$z$', fontsize=12, labelpad=3)
    ax2.set_ylabel(r'$x_1$', fontsize=12, labelpad=3)
    ax2.set_zlabel(r'$x_2$', fontsize=12, labelpad=3)
    ax2.set_xlim(-2.8, 5.0)
    ax2.set_ylim(-1.8, 1.8)
    ax2.set_zlim(-1.8, 1.8)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.view_init(elev=22, azim=-62)
    ax2.set_title('(b)', y=-0.15, fontsize=12)

    ax2.xaxis.pane.set_edgecolor('gray')
    ax2.yaxis.pane.set_edgecolor('gray')
    ax2.zaxis.pane.set_edgecolor('gray')
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path, dpi=300)
    return fig


def generate_figure_16_14(
    save_path: Optional[str] = "result/fig_16_14_pixel_distance_failure.png",
    digit_image_path: Optional[str] = "scratch/ch16_images/fig-000.png",
) -> plt.Figure:
    """Generate Figure 16.14: Three digit '2' images showing Euclidean pixel distance failure."""
    if digit_image_path and os.path.exists(digit_image_path):
        img = Image.open(digit_image_path).convert('L')
        arr = np.array(img)
        patch_a = arr[17:202, 13:198]
        patch_b = arr[17:202, 223:408]
        patch_c = arr[17:202, 433:618]
    else:
        patch_a = np.zeros((185, 185), dtype=np.uint8)
        patch_b = np.zeros((185, 185), dtype=np.uint8)
        patch_c = np.zeros((185, 185), dtype=np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.8))

    axes[0].imshow(patch_a, cmap='gray')
    axes[0].set_title('(a)', fontsize=13, pad=8)
    axes[0].text(0.5, -0.15, 'Target image', transform=axes[0].transAxes, ha='center', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(patch_b, cmap='gray')
    axes[1].set_title('(b)', fontsize=13, pad=8)
    axes[1].text(0.5, -0.15, r'Squared Dist = $0.0387$', transform=axes[1].transAxes, ha='center', fontsize=11, color='#d62728', fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(patch_c, cmap='gray')
    axes[2].set_title('(c)', fontsize=13, pad=8)
    axes[2].text(0.5, -0.15, r'Squared Dist = $0.2693$', transform=axes[2].transAxes, ha='center', fontsize=11, color='#1f77b4', fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path, dpi=300)
    return fig


def generate_figure_16_15(
    save_path: Optional[str] = "result/fig_16_15_dequantization.png",
) -> plt.Figure:
    """Generate Figure 16.15: Dequantization schematic: (a) discrete, (b) continuous."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8))

    probs = [0.65, 1.0, 0.4]

    # (a) Discrete distribution: narrow spikes
    ax1.bar([0, 1, 2], probs, width=0.08, color='#1f77b4', edgecolor='black', linewidth=1.2)
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(0, 1.2)
    ax1.set_xticks([0, 1, 2])
    ax1.set_yticks([])
    ax1.set_xlabel('(a)', fontsize=13, labelpad=10)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax1.spines[spine].set_color('black')
        ax1.spines[spine].set_linewidth(1.2)

    # (b) Dequantized continuous distribution: contiguous histogram
    ax2.bar([0.5, 1.5, 2.5], probs, width=1.0, color='#1f77b4', edgecolor='black', linewidth=1.2)
    ax2.set_xlim(-0.2, 3.2)
    ax2.set_ylim(0, 1.2)
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([])
    ax2.set_xlabel('(b)', fontsize=13, labelpad=10)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax2.spines[spine].set_color('black')
        ax2.spines[spine].set_linewidth(1.2)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path, dpi=300)
    return fig
