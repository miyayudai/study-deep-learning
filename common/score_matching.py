r"""Score Matching and Continuous Diffusion SDEs (Chapter 20, Section 20.3).

This module implements the mathematical foundations of score-based generative modeling
from Bishop & Bishop (2024), "Deep Learning: Foundations and Concepts":
- Score function definition and independence from partition function Z (Section 20.3.1, Eq. 20.21)
- Explicit and implicit score matching with trace of Jacobian (Hyvärinen 2005, Eq. 20.22, 20.23)
- Denoising score matching (Vincent 2011, Section 20.3.2, Eq. 20.24 - 20.26)
- Multiscale noise levels and annealed Langevin dynamics (Song & Ermon 2019, Section 20.3.3, Eq. 20.27, 20.28)
- Stochastic Differential Equations (SDEs), VP/VE SDEs, reverse SDE, and Probability Flow ODE (Section 20.3.4, Eq. 20.29 - 20.31)
"""

import os
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# =====================================================================
# Analytical Distributions & True Score Functions
# =====================================================================

class GaussianMixture2D:
    r"""2D Gaussian Mixture Model with analytical score function :math:`\nabla_{\mathbf{x}} \ln p(\mathbf{x})`."""

    def __init__(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray,
    ):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights /= np.sum(self.weights)
        self.means = np.asarray(means, dtype=np.float64)  # (K, 2)
        self.covs = np.asarray(covs, dtype=np.float64)    # (K, 2, 2)
        self.K = len(self.weights)

        self.inv_covs = np.array([np.linalg.inv(c) for c in self.covs])
        self.dets = np.array([np.linalg.det(c) for c in self.covs])

    def pdf(self, x: np.ndarray) -> np.ndarray:
        r"""Compute mixture probability density :math:`p(\mathbf{x}) = \sum_k \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \mathbf{\Sigma}_k)`."""
        x = np.atleast_2d(x)
        N = x.shape[0]
        p = np.zeros(N)
        for k in range(self.K):
            diff = x - self.means[k]
            quad = np.sum(diff @ self.inv_covs[k] * diff, axis=-1)
            norm_const = 2.0 * np.pi * np.sqrt(self.dets[k])
            p += self.weights[k] * np.exp(-0.5 * quad) / norm_const
        return p

    def score(self, x: np.ndarray) -> np.ndarray:
        r"""Compute exact analytical score function :math:`\nabla_{\mathbf{x}} \ln p(\mathbf{x})` (Eq. 20.21).

        .. math::
            \nabla_{\mathbf{x}} \ln p(\mathbf{x}) = \frac{\nabla_{\mathbf{x}} p(\mathbf{x})}{p(\mathbf{x})}
            = \sum_{k=1}^K \gamma_k(\mathbf{x}) \left( -\mathbf{\Sigma}_k^{-1}(\mathbf{x} - \boldsymbol{\mu}_k) \right)
        """
        x = np.atleast_2d(x)
        N = x.shape[0]
        densities_k = np.zeros((N, self.K))
        for k in range(self.K):
            diff = x - self.means[k]
            quad = np.sum(diff @ self.inv_covs[k] * diff, axis=-1)
            norm_const = 2.0 * np.pi * np.sqrt(self.dets[k])
            densities_k[:, k] = self.weights[k] * np.exp(-0.5 * quad) / norm_const

        p_total = np.sum(densities_k, axis=-1, keepdims=True) + 1e-12
        responsibilities = densities_k / p_total  # (N, K)

        score_vec = np.zeros((N, 2))
        for k in range(self.K):
            diff = x - self.means[k]
            grad_k = -diff @ self.inv_covs[k]  # (N, 2)
            score_vec += responsibilities[:, k:k+1] * grad_k

        return score_vec


# =====================================================================
# Score Matching Objectives (Explicit, Implicit, Denoising)
# =====================================================================

def implicit_score_matching_loss(
    score_model_fn: Callable[[np.ndarray], np.ndarray],
    jacobian_trace_fn: Callable[[np.ndarray], np.ndarray],
    x_samples: np.ndarray,
) -> float:
    r"""Compute implicit score matching loss (Hyvärinen 2005, Eq. 20.23).

    .. math::
        J(\mathbf{w}) = \frac{1}{N} \sum_{n=1}^N \left[ \frac{1}{2} \|\mathbf{s}(\mathbf{x}_n, \mathbf{w})\|^2 + \operatorname{Tr}(\nabla_{\mathbf{x}} \mathbf{s}(\mathbf{x}_n, \mathbf{w})) \right]
    """
    scores = score_model_fn(x_samples)
    norm_sq = np.sum(scores ** 2, axis=-1)
    traces = jacobian_trace_fn(x_samples)
    return float(np.mean(0.5 * norm_sq + traces))


def denoising_score_matching_loss(
    score_model_fn: Callable[[np.ndarray, float], np.ndarray],
    x_clean: np.ndarray,
    sigma: float,
    random_state: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    r"""Compute denoising score matching loss (Vincent 2011, Eq. 20.26).

    .. math::
        \tilde{\mathbf{x}} = \mathbf{x} + \sigma \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
        \nabla_{\tilde{\mathbf{x}}} \ln q(\tilde{\mathbf{x}} \mid \mathbf{x}) = -\frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma^2} = -\frac{\boldsymbol{\epsilon}}{\sigma} \\
        J_{\text{denoising}} = \frac{1}{2N} \sum_{n=1}^N \left\| \mathbf{s}(\tilde{\mathbf{x}}_n, \sigma) + \frac{\tilde{\mathbf{x}}_n - \mathbf{x}_n}{\sigma^2} \right\|^2
    """
    rng = np.random.RandomState(random_state)
    noise = rng.randn(*x_clean.shape)
    x_noisy = x_clean + sigma * noise

    target_score = -noise / sigma
    pred_score = score_model_fn(x_noisy, sigma)

    diff = pred_score - target_score
    loss = float(np.mean(0.5 * np.sum(diff ** 2, axis=-1)))
    return loss, x_noisy


# =====================================================================
# Multiscale Score Matching & Annealed Langevin Dynamics
# =====================================================================

class AnnealedLangevinDynamics:
    r"""Annealed Langevin Dynamics sampling algorithm (Song & Ermon 2019, Eq. 20.28).

    For each noise scale :math:`\sigma_1 > \sigma_2 > \dots > \sigma_L`:
    .. math::
        \mathbf{x}_{k+1} = \mathbf{x}_k + \frac{\alpha_i}{2} \mathbf{s}_\theta(\mathbf{x}_k, \sigma_i) + \sqrt{\alpha_i} \boldsymbol{\epsilon}_k
    """

    def __init__(
        self,
        score_fn: Callable[[np.ndarray, float], np.ndarray],
        sigmas: np.ndarray,
        n_steps_each: int = 100,
        step_lr: float = 2e-5,
    ):
        self.score_fn = score_fn
        self.sigmas = np.sort(sigmas)[::-1]  # descending order
        self.n_steps_each = n_steps_each
        self.step_lr = step_lr

    def sample(
        self,
        init_samples: np.ndarray,
        return_trajectory: bool = False,
        random_state: Optional[int] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        rng = np.random.RandomState(random_state)
        x = init_samples.copy()
        trajectory = [x.copy()] if return_trajectory else None

        sigma_min = self.sigmas[-1]
        for sigma in self.sigmas:
            alpha = self.step_lr * (sigma / sigma_min) ** 2
            alpha = min(alpha, 0.05)  # Safeguard against excessive step sizes
            for _ in range(self.n_steps_each):
                score = self.score_fn(x, sigma)
                # Clip extreme score magnitudes for numerical stability in low density regions
                score = np.clip(score, -50.0, 50.0)
                noise = rng.randn(*x.shape)
                x = x + 0.5 * alpha * score + np.sqrt(alpha) * noise
                if return_trajectory:
                    trajectory.append(x.copy())

        if return_trajectory:
            return x, np.array(trajectory)
        return x


# =====================================================================
# Continuous Diffusion SDE & Probability Flow ODE
# =====================================================================

class VPSDE:
    r"""Variance Preserving Stochastic Differential Equation (VP SDE, Song et al. 2021).

    Continuous-time limit of DDPM:
    .. math::
        d\mathbf{z} = -\frac{1}{2}\beta(t)\mathbf{z} \, dt + \sqrt{\beta(t)} \, d\mathbf{w} \tag{20.29}
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: float) -> float:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def drift(self, z: np.ndarray, t: float) -> np.ndarray:
        return -0.5 * self.beta(t) * z

    def diffusion(self, t: float) -> float:
        return np.sqrt(self.beta(t))

    def reverse_sde_step(
        self,
        z: np.ndarray,
        score: np.ndarray,
        t: float,
        dt: float,
        noise: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Simulate reverse-time SDE step (Anderson 1982, Eq. 20.30).

        .. math::
            d\mathbf{z} = \left[ \mathbf{f}(\mathbf{z}, t) - g(t)^2 \nabla_{\mathbf{z}} \ln p_t(\mathbf{z}) \right] dt + g(t) d\bar{\mathbf{w}}
        """
        beta_t = self.beta(t)
        # Reverse drift: f(z, t) - g^2 score
        reverse_drift = -0.5 * beta_t * z - beta_t * score
        diffusion_scale = np.sqrt(beta_t)

        if noise is None:
            noise = np.random.randn(*z.shape)

        # dt is positive step size backward in time
        return z - reverse_drift * dt + diffusion_scale * np.sqrt(dt) * noise

    def probability_flow_ode_step(
        self,
        z: np.ndarray,
        score: np.ndarray,
        t: float,
        dt: float,
    ) -> np.ndarray:
        r"""Simulate Probability Flow ODE step (Song et al. 2021, Eq. 20.31).

        .. math::
            \frac{d\mathbf{z}}{dt} = \mathbf{f}(\mathbf{z}, t) - \frac{1}{2} g(t)^2 \nabla_{\mathbf{z}} \ln p_t(\mathbf{z})
        """
        beta_t = self.beta(t)
        ode_drift = -0.5 * beta_t * z - 0.5 * beta_t * score
        return z - ode_drift * dt


# =====================================================================
# Visualizations for Section 20.3
# =====================================================================

def plot_score_vector_field_and_mixture(
    gmm: GaussianMixture2D,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    grid_size: int = 25,
) -> plt.Figure:
    r"""Plot probability contour and score vector field :math:`\nabla \ln p(\mathbf{x})`."""
    x = np.linspace(xlim[0], xlim[1], grid_size)
    y = np.linspace(ylim[0], ylim[1], grid_size)
    X, Y = np.meshgrid(x, y)
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)

    # Densities
    Z_pdf = gmm.pdf(pts).reshape(grid_size, grid_size)

    # Scores
    scores = gmm.score(pts)
    U = scores[:, 0].reshape(grid_size, grid_size)
    V = scores[:, 1].reshape(grid_size, grid_size)

    # Normalize vectors for visualization
    speed = np.sqrt(U ** 2 + V ** 2) + 1e-6
    U_norm = U / speed
    V_norm = V / speed

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    contour = ax.contourf(X, Y, Z_pdf, levels=20, cmap="viridis", alpha=0.8)
    plt.colorbar(contour, ax=ax, label=r"Density $p(\mathbf{x})$")

    ax.quiver(X, Y, U_norm, V_norm, color="white", alpha=0.85, pivot="mid")
    ax.set_title(r"Analytical Score Vector Field $\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \ln p(\mathbf{x})$ (Eq. 20.21)", fontsize=11)
    ax.set_xlabel(r"$x_1$", fontsize=11)
    ax.set_ylabel(r"$x_2$", fontsize=11)
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig
