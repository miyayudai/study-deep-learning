r"""
Mixtures of Gaussians Module (Chapter 15, Section 15.2)
======================================================
Implementation of Gaussian Mixture Models (GMM), likelihood evaluation,
responsibilities computation (posterior probabilities), singularity analysis,
and textbook figure reproduction for Bishop's Deep Learning (2024).

Key Equations:
- Linear superposition of Gaussians (Eq. 15.5):
    p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)
- Mixing coefficients constraint (Eq. 15.6):
    \sum_{k=1}^K \pi_k = 1, \quad 0 \le \pi_k \le 1
- Prior distribution of 1-of-K latent variable z (Eq. 15.8):
    p(z) = \prod_{k=1}^K \pi_k^{z_k}
- Conditional distribution of x given z (Eq. 15.9):
    p(x | z) = \prod_{k=1}^K \mathcal{N}(x | \mu_k, \Sigma_k)^{z_k}
- Posterior responsibility \gamma(z_k) = p(z_k = 1 | x) (Eq. 15.10):
    \gamma(z_k) = \frac{\pi_k \mathcal{N}(x | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x | \mu_j, \Sigma_j)}
- Log-likelihood function (Eq. 15.11):
    \ln p(X | \pi, \mu, \Sigma) = \sum_{n=1}^N \ln \left\{ \sum_{k=1}^K \pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k) \right\}
- Stationarity conditions (Eqs. 15.14, 15.15, 15.18):
    N_k = \sum_{n=1}^N \gamma(z_{nk})
    \mu_k = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) x_n
    \Sigma_k = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) (x_n - \mu_k)(x_n - \mu_k)^T
    \pi_k = \frac{N_k}{N}
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


class GaussianMixtureModel:
    """Gaussian Mixture Model (GMM) with full covariance matrices.

    Parameters
    ----------
    n_components : int, default=3
        Number of mixture components K.
    tol : float, default=1e-4
        Tolerance for log-likelihood convergence.
    max_iter : int, default=100
        Maximum number of EM iterations.
    random_state : int or None, default=None
        Random seed for initialization.
    reg_covar : float, default=1e-6
        Regularization added to the diagonal of covariance matrices to prevent singularities.
    """

    def __init__(
        self,
        n_components: int = 3,
        tol: float = 1e-4,
        max_iter: int = 100,
        random_state: Optional[int] = None,
        reg_covar: float = 1e-6,
    ):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.reg_covar = reg_covar

        self.weights_: Optional[np.ndarray] = None  # (K,)
        self.means_: Optional[np.ndarray] = None  # (K, D)
        self.covariances_: Optional[np.ndarray] = None  # (K, D, D)
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.lower_bound_: float = -np.inf
        self.history_log_likelihood_: List[float] = []

    def _init_parameters(self, X: np.ndarray) -> None:
        """Initialize parameters using random sample points as means."""
        rng = np.random.RandomState(self.random_state)
        N, D = X.shape
        K = self.n_components

        self.weights_ = np.full(K, 1.0 / K)
        chosen_indices = rng.choice(N, size=K, replace=False)
        self.means_ = X[chosen_indices].copy()

        overall_cov = np.cov(X, rowvar=False)
        if D == 1:
            overall_cov = np.array([[overall_cov]])
        self.covariances_ = np.array(
            [overall_cov + self.reg_covar * np.eye(D) for _ in range(K)]
        )

    def _compute_component_densities(self, X: np.ndarray) -> np.ndarray:
        r"""Compute Gaussian density N(x_n | \mu_k, \Sigma_k) for all n and k."""
        N, D = X.shape
        K = self.n_components
        densities = np.zeros((N, K))

        for k in range(K):
            cov_reg = self.covariances_[k] + self.reg_covar * np.eye(D)
            try:
                mvn = multivariate_normal(mean=self.means_[k], cov=cov_reg, allow_singular=True)
                densities[:, k] = mvn.pdf(X)
            except Exception:
                diff = X - self.means_[k]
                var = np.diag(cov_reg)
                log_pdf = -0.5 * (np.sum(diff**2 / var, axis=1) + np.sum(np.log(2 * np.pi * var)))
                densities[:, k] = np.exp(np.clip(log_pdf, -500, 500))

        return np.maximum(densities, 1e-300)

    def compute_responsibilities(self, X: np.ndarray) -> np.ndarray:
        r"""Compute posterior responsibilities \gamma(z_{nk}) (Equation 15.10).

        \gamma(z_{nk}) = \frac{\pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_n | \mu_j, \Sigma_j)}
        """
        X = np.asarray(X, dtype=float)
        densities = self._compute_component_densities(X)  # (N, K)
        weighted_densities = densities * self.weights_  # (N, K)
        total_density = np.sum(weighted_densities, axis=1, keepdims=True)  # (N, 1)
        gamma = weighted_densities / np.maximum(total_density, 1e-300)
        return gamma

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute log-likelihood of each sample under the mixture model."""
        X = np.asarray(X, dtype=float)
        densities = self._compute_component_densities(X)
        weighted = densities * self.weights_
        total_densities = np.sum(weighted, axis=1)
        return np.log(np.maximum(total_densities, 1e-300))

    def log_likelihood(self, X: np.ndarray) -> float:
        """Compute total log-likelihood of dataset X (Equation 15.11)."""
        return float(np.sum(self.score_samples(X)))

    def fit(self, X: np.ndarray) -> "GaussianMixtureModel":
        """Fit GMM parameters via EM algorithm (Equations 15.10, 15.14, 15.15, 15.18)."""
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        K = self.n_components

        self._init_parameters(X)
        self.history_log_likelihood_ = []
        prev_ll = -np.inf

        for iteration in range(1, self.max_iter + 1):
            # E-step: Responsibilities (Eq. 15.10)
            gamma = self.compute_responsibilities(X)  # (N, K)
            N_k = np.maximum(np.sum(gamma, axis=0), 1e-10)  # (K,)

            # M-step: Update means, covariances, weights (Eqs. 15.14, 15.15, 15.18)
            new_means = np.zeros((K, D))
            for k in range(K):
                new_means[k] = np.sum(gamma[:, k : k + 1] * X, axis=0) / N_k[k]

            new_covs = np.zeros((K, D, D))
            for k in range(K):
                diff = X - new_means[k]
                weighted_diff = gamma[:, k : k + 1] * diff
                cov_k = (weighted_diff.T @ diff) / N_k[k]
                new_covs[k] = cov_k + self.reg_covar * np.eye(D)

            new_weights = N_k / N
            new_weights = new_weights / np.sum(new_weights)

            self.means_ = new_means
            self.covariances_ = new_covs
            self.weights_ = new_weights

            current_ll = self.log_likelihood(X)
            self.history_log_likelihood_.append(current_ll)
            self.n_iter_ = iteration

            if np.abs(current_ll - prev_ll) < self.tol:
                self.converged_ = True
                break
            prev_ll = current_ll

        self.lower_bound_ = float(self.history_log_likelihood_[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster index by assigning to maximum responsibility component."""
        gamma = self.compute_responsibilities(X)
        return np.argmax(gamma, axis=1)

    def sample(self, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic samples using ancestral sampling."""
        rng = np.random.RandomState(self.random_state)
        K = self.n_components
        D = self.means_.shape[1]

        z_samples = rng.choice(K, size=n_samples, p=self.weights_)
        X_samples = np.zeros((n_samples, D))

        for k in range(K):
            count = np.sum(z_samples == k)
            if count > 0:
                X_samples[z_samples == k] = rng.multivariate_normal(
                    mean=self.means_[k], cov=self.covariances_[k], size=count
                )

        return X_samples, z_samples


def load_gmm_synthetic_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load the synthetic 3-component Gaussian mixture dataset used in Figure 15.5."""
    possible_paths = [
        Path("common/data/gmm_synthetic.csv"),
        Path("../common/data/gmm_synthetic.csv"),
        Path("/home/student/Documents/GitHub/my_DeepLearning/common/data/gmm_synthetic.csv"),
    ]
    csv_path = None
    for p in possible_paths:
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        raise FileNotFoundError("Could not find common/data/gmm_synthetic.csv")

    df = pd.read_csv(csv_path)
    X = df[["x1", "x2"]].values
    clusters = df["cluster"].values.astype(int)

    weights = np.array([0.5, 0.3, 0.2])
    means = np.array([
        [0.188, 0.386],
        [0.505, 0.494],
        [0.779, 0.589],
    ])
    covariances = np.array([
        [[0.0158, 0.0113], [0.0113, 0.0137]],
        [[0.0152, -0.0115], [-0.0115, 0.0148]],
        [[0.0131, 0.0100], [0.0100, 0.0140]],
    ])

    params = {
        "weights": weights,
        "means": means,
        "covariances": covariances,
    }

    return X, clusters, params


def generate_figure_15_4(
    save_path: Optional[Union[str, Path]] = None, show: bool = False
) -> plt.Figure:
    """Generate and faithfully reproduce Figure 15.4 (Bishop, 2024).

    Directed graphical model for a Gaussian mixture model:
    Discrete latent variable node z pointing to observed variable node x.
    """
    fig, ax = plt.subplots(figsize=(2.5, 4.5))

    c_red = "#e60000"

    circle_z = patches.Circle(
        (0.5, 0.72), 0.18, edgecolor=c_red, facecolor="white", linewidth=2.5, zorder=3
    )
    circle_x = patches.Circle(
        (0.5, 0.28), 0.18, edgecolor=c_red, facecolor="white", linewidth=2.5, zorder=3
    )
    ax.add_patch(circle_z)
    ax.add_patch(circle_x)
    ax.set_aspect('equal')

    ax.annotate(
        "",
        xy=(0.5, 0.46),
        xytext=(0.5, 0.54),
        arrowprops=dict(
            arrowstyle="-|>",
            color=c_red,
            lw=2.5,
            mutation_scale=20,
        ),
        zorder=4,
    )

    ax.text(
        0.5,
        0.72,
        r"$\mathbf{z}$",
        fontsize=22,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=5,
    )
    ax.text(
        0.5,
        0.28,
        r"$\mathbf{x}$",
        fontsize=22,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=5,
    )

    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(0.05, 0.95)
    ax.axis("off")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def generate_figure_15_5(
    save_path: Optional[Union[str, Path]] = None, show: bool = False
) -> plt.Figure:
    r"""Generate and faithfully reproduce Figure 15.5 (Bishop, 2024).

    Illustration of a Gaussian mixture distribution with K=3 components:
    - (a) Complete dataset: points colored according to true generating component
    - (b) Incomplete (observed) dataset: latent identities hidden, all points magenta
    - (c) Responsibilities: points colored by RGB blend = (\gamma_1, \gamma_2, \gamma_3)
    """
    X, clusters, params = load_gmm_synthetic_dataset()

    c_red = np.array([1.0, 0.0, 0.0])
    c_green = np.array([0.0, 1.0, 0.0])
    c_blue = np.array([0.0, 0.0, 1.0])
    c_magenta = "#ff00ff"

    gmm = GaussianMixtureModel(n_components=3, reg_covar=1e-6)
    gmm.weights_ = params["weights"]
    gmm.means_ = params["means"]
    gmm.covariances_ = params["covariances"]
    gamma = gmm.compute_responsibilities(X)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    panel_labels = ["(a)", "(b)", "(c)"]

    for idx, (ax, label) in enumerate(zip(axes, panel_labels)):
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(
            direction="in", top=True, right=True, labelsize=12, length=5, width=1.2
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        ax.text(
            0.10,
            0.90,
            label,
            transform=ax.transAxes,
            fontsize=15,
            verticalalignment="top",
        )

        if idx == 0:
            colors = np.zeros((len(X), 3))
            colors[clusters == 0] = c_red
            colors[clusters == 1] = c_green
            colors[clusters == 2] = c_blue
            ax.scatter(
                X[:, 0],
                X[:, 1],
                color=colors,
                s=20,
                alpha=1.0,
                edgecolors="none",
                zorder=2,
            )

        elif idx == 1:
            ax.scatter(
                X[:, 0],
                X[:, 1],
                color=c_magenta,
                s=20,
                alpha=1.0,
                edgecolors="none",
                zorder=2,
            )

        else:
            rgb_blend = np.clip(gamma, 0.0, 1.0)
            ax.scatter(
                X[:, 0],
                X[:, 1],
                color=rgb_blend,
                s=20,
                alpha=1.0,
                edgecolors="none",
                zorder=2,
            )

    plt.tight_layout(pad=1.5)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def generate_figure_15_6(
    save_path: Optional[Union[str, Path]] = None, show: bool = False
) -> plt.Figure:
    """Generate and faithfully reproduce Figure 15.6 (Bishop, 2024).

    Illustration of a singularity in the likelihood function of a mixture of Gaussians.
    """
    x_grid = np.linspace(-3.5, 6.5, 1000)

    mu_broad = 1.0
    sigma_broad = 1.8
    p_broad = 0.8 * (1.0 / (np.sqrt(2 * np.pi) * sigma_broad)) * np.exp(
        -0.5 * ((x_grid - mu_broad) / sigma_broad) ** 2
    )

    x_spike = 5.2
    sigma_spike = 0.12
    p_spike = 0.2 * (1.0 / (np.sqrt(2 * np.pi) * sigma_spike)) * np.exp(
        -0.5 * ((x_grid - x_spike) / sigma_spike) ** 2
    )

    p_total = p_broad + p_spike

    data_pts = np.array([-1.8, 0.1, 1.4, 1.9, 3.3, 3.9, 5.2])
    data_densities = np.zeros_like(data_pts)
    for i, pt in enumerate(data_pts):
        b = 0.8 * (1.0 / (np.sqrt(2 * np.pi) * sigma_broad)) * np.exp(
            -0.5 * ((pt - mu_broad) / sigma_broad) ** 2
        )
        s = 0.2 * (1.0 / (np.sqrt(2 * np.pi) * sigma_spike)) * np.exp(
            -0.5 * ((pt - x_spike) / sigma_spike) ** 2
        )
        data_densities[i] = b + s

    fig, ax = plt.subplots(figsize=(6, 4.5))

    c_red = "#ff0000"
    c_green = "#008000"
    c_blue = "#0000ff"
    c_gray = "#808080"

    ax.plot(x_grid, p_total, color=c_red, linewidth=2.0, zorder=3)

    for pt, dens in zip(data_pts, data_densities):
        ax.plot([pt, pt], [0.0, dens], color=c_green, linewidth=1.8, zorder=2)
        ax.scatter([pt], [dens], color=c_blue, s=55, zorder=4)

    ax.scatter(data_pts, np.zeros_like(data_pts), color=c_gray, s=55, zorder=4)

    ax.set_xlim(-3.8, 7.2)
    ax.set_ylim(-0.02, 0.72)

    ax.annotate(
        "",
        xy=(7.0, 0.0),
        xytext=(-3.5, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15),
        zorder=1,
    )
    ax.annotate(
        "",
        xy=(-3.5, 0.70),
        xytext=(-3.5, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15),
        zorder=1,
    )

    ax.text(1.8, -0.06, r"$x$", fontsize=16, ha="center", va="top")
    ax.text(
        -3.8,
        0.35,
        r"$p(x)$",
        fontsize=16,
        ha="right",
        va="center",
        rotation=0,
    )

    ax.axis("off")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
