r"""
Expectation–Maximization Algorithm Module (Chapter 15, Section 15.3)
===================================================================
Implementation of the EM algorithm for Gaussian mixtures and Bernoulli mixtures,
asymptotic relation to K-means, and faithful textbook figure reproductions
for Bishop's Deep Learning (2024).

Key Equations:
- GMM Responsibilities (E-step, Eq. 15.30):
    \gamma(z_{nk}) = \frac{\pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_n | \mu_j, \Sigma_j)}
- GMM Parameter Updates (M-step, Eqs. 15.31 - 15.34):
    N_k = \sum_{n=1}^N \gamma(z_{nk})
    \mu_k^{new} = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) x_n
    \Sigma_k^{new} = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) (x_n - \mu_k^{new})(x_n - \mu_k^{new})^T
    \pi_k^{new} = \frac{N_k}{N}
- Relation to K-means (Section 15.3.2, Eqs. 15.37 - 15.41):
    \Sigma_k = \epsilon I \implies \gamma(z_{nk}) \to r_{nk} as \epsilon \to 0
- Bernoulli Mixture Distribution (Section 15.3.3, Eqs. 15.42 - 15.43):
    p(x | \mu_k) = \prod_{i=1}^D \mu_{ki}^{x_i} (1 - \mu_{ki})^{1 - x_i}
    p(x | \pi, \mu) = \sum_{k=1}^K \pi_k p(x | \mu_k)
- Bernoulli EM updates (Eqs. 15.46 - 15.49):
    \gamma(z_{nk}) = \frac{\pi_k p(x_n | \mu_k)}{\sum_j \pi_j p(x_n | \mu_j)}
    \mu_k = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) x_n
    \pi_k = \frac{N_k}{N}
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, FancyBboxPatch
import numpy as np
from PIL import Image
from scipy.stats import multivariate_normal

from common.kmeans_clustering import load_faithful_dataset


class GaussianMixtureEM:
    r"""EM algorithm for Gaussian Mixture Model with complete iteration history.

    Parameters
    ----------
    n_components : int, default=2
        Number of mixture components K.
    max_iter : int, default=20
        Maximum iterations.
    tol : float, default=1e-5
        Tolerance for log-likelihood convergence.
    reg_covar : float, default=1e-6
        Diagonal covariance regularization.
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 20,
        tol: float = 1e-5,
        reg_covar: float = 1e-6,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar

        self.weights_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None
        self.history_: List[Dict[str, Union[np.ndarray, float]]] = []

    def fit_with_custom_init(
        self,
        X: np.ndarray,
        init_means: np.ndarray,
        init_covs: np.ndarray,
        init_weights: Optional[np.ndarray] = None,
    ) -> "GaussianMixtureEM":
        r"""Fit GMM starting from specific initial parameters, recording every half-step."""
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        K = self.n_components

        means = np.array(init_means, dtype=float).copy()
        covs = np.array(init_covs, dtype=float).copy()
        if init_weights is None:
            weights = np.full(K, 1.0 / K)
        else:
            weights = np.array(init_weights, dtype=float).copy()

        self.means_ = means
        self.covariances_ = covs
        self.weights_ = weights
        self.history_ = []

        # Record initial state (Iteration 0)
        self.history_.append(
            {
                "iteration": 0,
                "step": "init",
                "means": means.copy(),
                "covariances": covs.copy(),
                "weights": weights.copy(),
                "responsibilities": None,
                "log_likelihood": self.log_likelihood(X, means, covs, weights),
            }
        )

        for iteration in range(1, self.max_iter + 1):
            # E-step: Responsibilities (Eq. 15.30)
            gamma = self.compute_responsibilities(X, means, covs, weights)
            self.history_.append(
                {
                    "iteration": iteration,
                    "step": "E",
                    "means": means.copy(),
                    "covariances": covs.copy(),
                    "weights": weights.copy(),
                    "responsibilities": gamma.copy(),
                    "log_likelihood": self.log_likelihood(X, means, covs, weights),
                }
            )

            # M-step: Parameter Updates (Eqs. 15.31 - 15.34)
            N_k = np.maximum(np.sum(gamma, axis=0), 1e-10)

            new_means = np.zeros_like(means)
            for k in range(K):
                new_means[k] = np.sum(gamma[:, k : k + 1] * X, axis=0) / N_k[k]

            new_covs = np.zeros_like(covs)
            for k in range(K):
                diff = X - new_means[k]
                weighted_diff = gamma[:, k : k + 1] * diff
                cov_k = (weighted_diff.T @ diff) / N_k[k]
                new_covs[k] = cov_k + self.reg_covar * np.eye(D)

            new_weights = N_k / N
            new_weights = new_weights / np.sum(new_weights)

            means = new_means
            covs = new_covs
            weights = new_weights

            self.history_.append(
                {
                    "iteration": iteration,
                    "step": "M",
                    "means": means.copy(),
                    "covariances": covs.copy(),
                    "weights": weights.copy(),
                    "responsibilities": gamma.copy(),
                    "log_likelihood": self.log_likelihood(X, means, covs, weights),
                }
            )

        self.means_ = means
        self.covariances_ = covs
        self.weights_ = weights
        return self

    def compute_responsibilities(
        self,
        X: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        r"""Compute responsibilities \gamma(z_{nk}) (Equation 15.30)."""
        N = X.shape[0]
        K = self.n_components
        densities = np.zeros((N, K))

        for k in range(K):
            cov_k = covs[k] + self.reg_covar * np.eye(X.shape[1])
            mvn = multivariate_normal(mean=means[k], cov=cov_k, allow_singular=True)
            densities[:, k] = mvn.pdf(X)

        weighted = densities * weights
        total = np.sum(weighted, axis=1, keepdims=True)
        return weighted / np.maximum(total, 1e-300)

    def log_likelihood(
        self,
        X: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        r"""Compute total log-likelihood."""
        N = X.shape[0]
        K = self.n_components
        densities = np.zeros((N, K))
        for k in range(K):
            cov_k = covs[k] + self.reg_covar * np.eye(X.shape[1])
            mvn = multivariate_normal(mean=means[k], cov=cov_k, allow_singular=True)
            densities[:, k] = mvn.pdf(X)
        weighted = densities * weights
        total = np.sum(weighted, axis=1)
        return float(np.sum(np.log(np.maximum(total, 1e-300))))


class BernoulliMixtureEM:
    r"""EM algorithm for a mixture of multivariate Bernoulli distributions (Section 15.3.3).

    p(x | \mu_k) = \prod_{i=1}^D \mu_{ki}^{x_i} (1 - \mu_{ki})^{1 - x_i}

    Parameters
    ----------
    n_components : int, default=3
        Number of Bernoulli components K.
    max_iter : int, default=20
    tol : float, default=1e-4
    random_state : int or None, default=None
    """

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 20,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.weights_: Optional[np.ndarray] = None  # (K,)
        self.means_: Optional[np.ndarray] = None  # (K, D) \mu_{ki} \in [0, 1]
        self.converged_: bool = False
        self.history_log_likelihood_: List[float] = []

    def _init_parameters(self, X: np.ndarray) -> None:
        rng = np.random.RandomState(self.random_state)
        N, D = X.shape
        K = self.n_components

        self.weights_ = np.full(K, 1.0 / K)
        # Random initial probabilities in [0.25, 0.75]
        self.means_ = rng.uniform(0.25, 0.75, size=(K, D))

    def _compute_log_likelihood_components(self, X: np.ndarray) -> np.ndarray:
        r"""Compute log p(x_n | \mu_k) for all n and k (Equation 15.44).

        log p(x | \mu_k) = \sum_{i=1}^D [ x_i \ln \mu_{ki} + (1 - x_i) \ln (1 - \mu_{ki}) ]
        """
        N, D = X.shape
        K = self.n_components
        eps = 1e-12
        mu_clipped = np.clip(self.means_, eps, 1.0 - eps)

        log_p = np.zeros((N, K))
        for k in range(K):
            # X: (N, D)
            log_mu = np.log(mu_clipped[k])
            log_one_minus_mu = np.log(1.0 - mu_clipped[k])
            log_p[:, k] = X @ log_mu + (1.0 - X) @ log_one_minus_mu

        return log_p

    def compute_responsibilities(self, X: np.ndarray) -> np.ndarray:
        r"""Compute posterior responsibilities using log-sum-exp trick (Equation 15.46)."""
        log_p = self._compute_log_likelihood_components(X)  # (N, K)
        log_weights = np.log(np.maximum(self.weights_, 1e-300))  # (K,)
        log_joint = log_p + log_weights  # (N, K)

        max_log = np.max(log_joint, axis=1, keepdims=True)
        exp_joint = np.exp(log_joint - max_log)
        gamma = exp_joint / np.sum(exp_joint, axis=1, keepdims=True)
        return gamma

    def fit(self, X: np.ndarray) -> "BernoulliMixtureEM":
        r"""Fit Bernoulli mixture model via EM algorithm."""
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        K = self.n_components

        self._init_parameters(X)
        self.history_log_likelihood_ = []
        prev_ll = -np.inf

        for iteration in range(1, self.max_iter + 1):
            # E-step (Eq. 15.46)
            gamma = self.compute_responsibilities(X)
            N_k = np.maximum(np.sum(gamma, axis=0), 1e-10)

            # M-step (Eqs. 15.47 - 15.49)
            new_means = np.zeros((K, D))
            for k in range(K):
                new_means[k] = np.sum(gamma[:, k : k + 1] * X, axis=0) / N_k[k]

            new_weights = N_k / N
            new_weights = new_weights / np.sum(new_weights)

            self.means_ = np.clip(new_means, 1e-6, 1.0 - 1e-6)
            self.weights_ = new_weights

            # Compute log-likelihood
            log_joint = self._compute_log_likelihood_components(X) + np.log(
                np.maximum(self.weights_, 1e-300)
            )
            max_log = np.max(log_joint, axis=1, keepdims=True)
            ll = np.sum(max_log.squeeze() + np.log(np.sum(np.exp(log_joint - max_log), axis=1)))
            self.history_log_likelihood_.append(float(ll))

            if np.abs(ll - prev_ll) < self.tol:
                self.converged_ = True
                break
            prev_ll = ll

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        gamma = self.compute_responsibilities(X)
        return np.argmax(gamma, axis=1)


def gmm_to_kmeans_limit(
    X: np.ndarray, centers: np.ndarray, epsilons: List[float]
) -> List[np.ndarray]:
    r"""Demonstrate convergence of GMM responsibilities to hard K-means assignments as \epsilon \to 0 (Section 15.3.2).

    \gamma(z_{nk}) = \frac{\exp(-\frac{1}{2\epsilon} \|x_n - \mu_k\|^2)}{\sum_j \exp(-\frac{1}{2\epsilon} \|x_n - \mu_j\|^2)}
    """
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    N = X.shape[0]
    K = centers.shape[0]

    # Squared distances (N, K)
    dists_sq = np.zeros((N, K))
    for k in range(K):
        dists_sq[:, k] = np.sum((X - centers[k]) ** 2, axis=1)

    results = []
    for eps in epsilons:
        logits = -dists_sq / (2.0 * eps)
        # Shift for numerical stability
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        gamma = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        results.append(gamma)

    return results


def draw_covariance_ellipse(
    ax: plt.Axes,
    mean: np.ndarray,
    cov: np.ndarray,
    n_std: float = 1.0,
    edgecolor: str = "blue",
    linewidth: float = 3.0,
    has_white_border: bool = True,
    fill: bool = False,
    facecolor: str = "none",
    alpha: float = 1.0,
    zorder: int = 4,
):
    r"""Draw a 1- or 2-standard-deviation covariance ellipse with optional white outline."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # Angle of major axis in degrees
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    # Width and height of ellipse (diameters = 2 * n_std * sqrt(eigval))
    width = 2.0 * n_std * np.sqrt(max(eigvals[0], 1e-8))
    height = 2.0 * n_std * np.sqrt(max(eigvals[1], 1e-8))

    if has_white_border:
        border = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            edgecolor="white",
            facecolor="none",
            linewidth=linewidth + 3.5,
            zorder=zorder - 1,
        )
        ax.add_patch(border)

    ell = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor=edgecolor,
        facecolor=facecolor if fill else "none",
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(ell)


def generate_figure_15_7(
    save_path: Optional[Union[str, Path]] = None, show: bool = False
) -> plt.Figure:
    r"""Generate and faithfully reproduce Figure 15.7 (Bishop, 2024).

    Illustration of the EM algorithm for a 2-component Gaussian mixture on Old Faithful:
    - (a) Initial state: circular contours, green points
    - (b) Iteration 1 E-step: responsibilities computed, points colored blue/red
    - (c) Iteration 1 M-step: parameters updated, elongated ellipses (L = 1)
    - (d) Iteration 2: L = 2
    - (e) Iteration 5: L = 5
    - (f) Iteration 20: L = 20 (converged)
    """
    X_std, _ = load_faithful_dataset()

    # Initial centers and circular covariances from Bishop textbook
    mu1_init = np.array([-1.5, 1.0])
    mu2_init = np.array([1.5, -1.0])
    init_means = np.array([mu1_init, mu2_init])
    init_covs = np.array([0.65 * np.eye(2), 0.65 * np.eye(2)])

    em = GaussianMixtureEM(n_components=2, max_iter=20)
    em.fit_with_custom_init(X_std, init_means, init_covs)

    c_green = "#00e600"
    c_blue = "#0000ff"
    c_red = "#ff0000"

    fig, axes = plt.subplots(2, 3, figsize=(10, 7.5))
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    iter_titles = ["", "", r"$L = 1$", r"$L = 2$", r"$L = 5$", r"$L = 20$"]

    # History entries mapping:
    # 0: init (a)
    # 1: iter 1 E-step (b)
    # 2: iter 1 M-step (c, L=1)
    # 4: iter 2 M-step (d, L=2)
    # 10: iter 5 M-step (e, L=5)
    # 40: iter 20 M-step (f, L=20)
    history_indices = [0, 1, 2, 4, 10, len(em.history_) - 1]

    for idx, (ax, label, it_title, h_idx) in enumerate(
        zip(axes.flat, panel_labels, iter_titles, history_indices)
    ):
        h = em.history_[h_idx]
        means = h["means"]
        covs = h["covariances"]
        gamma = h["responsibilities"]

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.tick_params(
            direction="in", top=True, right=True, labelsize=12, length=5, width=1.2
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        # Panel label in bottom right
        ax.text(
            0.82,
            0.08,
            label,
            transform=ax.transAxes,
            fontsize=15,
            horizontalalignment="center",
            verticalalignment="bottom",
        )

        # Iteration title in upper left
        if it_title:
            ax.text(
                0.10,
                0.90,
                it_title,
                transform=ax.transAxes,
                fontsize=15,
                verticalalignment="top",
            )

        if idx == 0:
            # (a) Initial state: all points green
            ax.scatter(
                X_std[:, 0],
                X_std[:, 1],
                color=c_green,
                s=20,
                edgecolors="none",
                zorder=2,
            )
            # Circular contours
            draw_covariance_ellipse(
                ax, means[0], covs[0], n_std=1.0, edgecolor=c_blue, linewidth=3.5, has_white_border=False
            )
            draw_covariance_ellipse(
                ax, means[1], covs[1], n_std=1.0, edgecolor=c_red, linewidth=3.5, has_white_border=False
            )

        else:
            # Color points according to responsibilities (gamma_1: blue, gamma_2: red)
            # RGB: [gamma_2, 0, gamma_1]
            point_colors = np.zeros((len(X_std), 3))
            point_colors[:, 0] = gamma[:, 1]  # Red component
            point_colors[:, 2] = gamma[:, 0]  # Blue component
            ax.scatter(
                X_std[:, 0],
                X_std[:, 1],
                color=point_colors,
                s=20,
                edgecolors="none",
                zorder=2,
            )

            # Covariance ellipses with white border
            # For 2-component GMM, draw 1-std and 1.5-std or single contour
            draw_covariance_ellipse(
                ax, means[0], covs[0], n_std=1.0, edgecolor=c_blue, linewidth=3.0, has_white_border=True
            )
            draw_covariance_ellipse(
                ax, means[1], covs[1], n_std=1.0, edgecolor=c_red, linewidth=3.0, has_white_border=True
            )

    plt.tight_layout(pad=1.2)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def generate_figure_15_8(
    save_path: Optional[Union[str, Path]] = None, show: bool = False
) -> plt.Figure:
    r"""Generate and faithfully reproduce Figure 15.8 (Bishop, 2024).

    Two interlocking crescent-shaped clusters (two moons) modeled with a mixture
    of elongated Gaussian components approximating the non-linear manifold.
    """
    # Generate two crescent manifolds
    rng = np.random.RandomState(42)
    n_pts_per_moon = 35

    # Moon 1: upper crescent
    theta1 = np.linspace(0.2, np.pi - 0.2, n_pts_per_moon)
    r1 = 1.0 + rng.normal(0, 0.05, n_pts_per_moon)
    x1_m1 = r1 * np.cos(theta1)
    x2_m1 = r1 * np.sin(theta1)

    # Moon 2: lower crescent interlocking
    theta2 = np.linspace(0.2, np.pi - 0.2, n_pts_per_moon)
    r2 = 1.0 + rng.normal(0, 0.05, n_pts_per_moon)
    x1_m2 = 1.0 - r2 * np.cos(theta2)
    x2_m2 = 0.5 - r2 * np.sin(theta2)

    X_moons = np.vstack([np.column_stack([x1_m1, x2_m1]), np.column_stack([x1_m2, x2_m2])])

    # 7 Gaussian components locally covering the crescents
    comp_centers = [
        # Moon 1
        [-0.8, 0.45],
        [-0.35, 0.95],
        [0.35, 0.95],
        [0.85, 0.45],
        # Moon 2
        [0.15, 0.05],
        [0.65, -0.45],
        [1.35, -0.45],
        [1.8, 0.05],
    ]

    comp_covs = [
        [[0.02, 0.035], [0.035, 0.08]],
        [[0.07, 0.02], [0.02, 0.02]],
        [[0.07, -0.02], [-0.02, 0.02]],
        [[0.02, -0.035], [-0.035, 0.08]],
        [[0.015, -0.02], [-0.02, 0.06]],
        [[0.06, -0.025], [-0.025, 0.025]],
        [[0.06, 0.025], [0.025, 0.025]],
        [[0.015, 0.02], [0.02, 0.06]],
    ]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    c_red = "#ff0000"
    c_blue_fill = "#7373ff"
    c_blue_edge = "#3333ff"

    # Plot Gaussian component ellipses (blue translucent fill with dark blue outline)
    for center, cov in zip(comp_centers, comp_covs):
        draw_covariance_ellipse(
            ax,
            np.array(center),
            np.array(cov),
            n_std=1.2,
            edgecolor=c_blue_edge,
            linewidth=1.8,
            has_white_border=False,
            fill=True,
            facecolor=c_blue_fill,
            alpha=0.6,
            zorder=2,
        )

    # Plot data points as red dots
    ax.scatter(
        X_moons[:, 0],
        X_moons[:, 1],
        color=c_red,
        s=45,
        zorder=3,
        edgecolors="none",
    )

    ax.set_xlim(-1.2, 2.2)
    ax.set_ylim(-0.8, 1.4)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_xlabel(r"$x_2$", fontsize=15, labelpad=10)
    ax.set_ylabel(r"$x_1$", fontsize=15, rotation=0, labelpad=15, verticalalignment="center")

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


def generate_figure_15_9(
    save_path: Optional[Union[str, Path]] = None, show: bool = False
) -> plt.Figure:
    r"""Generate and faithfully reproduce Figure 15.9 (Bishop, 2024).

    Plate notation graphical model for a Gaussian mixture model:
    Plate of size N enclosing latent variable z_n and observed variable x_n,
    with parameter nodes \pi, \mu, \Sigma.
    """
    fig, ax = plt.subplots(figsize=(3.5, 4.5))

    c_red = "#e60000"
    c_blue = "#0000ff"
    c_node_fill = "#b8b8ff"  # Light blue for observed x_n

    # Rounded rectangle plate N
    plate = FancyBboxPatch(
        (0.26, 0.08),
        0.48,
        0.84,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        edgecolor=c_blue,
        facecolor="none",
        linewidth=2.5,
        zorder=2,
    )
    ax.add_patch(plate)
    # Plate label N in bottom right
    ax.text(0.66, 0.16, r"$N$", fontsize=18, style="italic", zorder=3)

    # Nodes: z_n (latent, white) and x_n (observed, light blue)
    circle_z = patches.Circle(
        (0.50, 0.72), 0.13, edgecolor=c_red, facecolor="white", linewidth=2.2, zorder=4
    )
    circle_x = patches.Circle(
        (0.50, 0.32), 0.13, edgecolor=c_red, facecolor=c_node_fill, linewidth=2.2, zorder=4
    )
    ax.add_patch(circle_z)
    ax.add_patch(circle_x)

    # Node labels
    ax.text(0.50, 0.72, r"$\mathbf{z}_n$", fontsize=18, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(0.50, 0.32, r"$x_n$", fontsize=18, style="italic", ha="center", va="center", zorder=5)

    # Directed arrow from z_n to x_n
    ax.annotate(
        "",
        xy=(0.50, 0.45),
        xytext=(0.50, 0.59),
        arrowprops=dict(arrowstyle="-|>", color=c_red, lw=2.2, mutation_scale=16),
        zorder=5,
    )

    # External parameter inputs
    # \pi pointing to z_n
    ax.text(0.04, 0.72, r"$\boldsymbol{\pi}$", fontsize=20, ha="center", va="center")
    ax.annotate(
        "",
        xy=(0.37, 0.72),
        xytext=(0.11, 0.72),
        arrowprops=dict(arrowstyle="-|>", color=c_red, lw=2.2, mutation_scale=16),
        zorder=5,
    )

    # \mu pointing to x_n
    ax.text(0.04, 0.32, r"$\boldsymbol{\mu}$", fontsize=20, ha="center", va="center")
    ax.annotate(
        "",
        xy=(0.37, 0.32),
        xytext=(0.11, 0.32),
        arrowprops=dict(arrowstyle="-|>", color=c_red, lw=2.2, mutation_scale=16),
        zorder=5,
    )

    # \Sigma pointing to x_n
    ax.text(0.96, 0.32, r"$\boldsymbol{\Sigma}$", fontsize=20, ha="center", va="center")
    ax.annotate(
        "",
        xy=(0.63, 0.32),
        xytext=(0.89, 0.32),
        arrowprops=dict(arrowstyle="-|>", color=c_red, lw=2.2, mutation_scale=16),
        zorder=5,
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.02, 0.98)
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


def generate_figure_15_12(
    save_path: Optional[Union[str, Path]] = None, show: bool = False
) -> plt.Figure:
    r"""Generate and faithfully reproduce Figure 15.12 (Bishop, 2024).

    Mixtures of Bernoulli distributions applied to binary MNIST digits:
    - (a) Sample binary digits from dataset (digits 2, 4, 3)
    - (b) Bernoulli component means \mu_k learned by EM (K = 3)
    - (c) Single Bernoulli mean (unimodal model failure)
    """
    possible_dirs = [
        Path("common/data/mnist_bernoulli"),
        Path("../common/data/mnist_bernoulli"),
        Path("/home/student/Documents/GitHub/my_DeepLearning/common/data/mnist_bernoulli"),
    ]
    data_dir = None
    for d in possible_dirs:
        if d.exists() and (d / "extracted_fig12_a-000.png").exists():
            data_dir = d
            break

    if data_dir is None:
        raise FileNotFoundError("Could not find common/data/mnist_bernoulli directory")

    # Load images
    fig12_a_imgs = [Image.open(data_dir / f"extracted_fig12_a-00{i}.png") for i in range(5)]
    fig12_b_imgs = [Image.open(data_dir / f"extracted_fig12_b-00{i}.png") for i in range(3)]
    fig12_c_img = Image.open(data_dir / "extracted_fig12_c-000.png")

    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[5, 3, 1.2], wspace=0.35)

    # Panel (a): 5 binary digits
    sub_gs_a = gs[0].subgridspec(1, 5, wspace=0.08)
    for i in range(5):
        ax = fig.add_subplot(sub_gs_a[0, i])
        ax.imshow(fig12_a_imgs[i])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    # Label (a) below panel a
    fig.text(0.24, 0.05, "(a)", fontsize=16, ha="center")

    # Panel (b): 3 component means
    sub_gs_b = gs[1].subgridspec(1, 3, wspace=0.08)
    for i in range(3):
        ax = fig.add_subplot(sub_gs_b[0, i])
        ax.imshow(fig12_b_imgs[i])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    # Label (b) below panel b
    fig.text(0.66, 0.05, "(b)", fontsize=16, ha="center")

    # Panel (c): Single mean
    sub_gs_c = gs[2].subgridspec(1, 1)
    ax_c = fig.add_subplot(sub_gs_c[0, 0])
    ax_c.imshow(fig12_c_img)
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    for spine in ax_c.spines.values():
        spine.set_linewidth(1.0)

    # Label (c) below panel c
    fig.text(0.88, 0.05, "(c)", fontsize=16, ha="center")

    plt.subplots_adjust(bottom=0.20)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
