r"""
Evidence Lower Bound Module (Chapter 15, Section 15.4)
======================================================
Comprehensive implementation of the Evidence Lower Bound (ELBO),
variational decomposition of log-likelihood, generalized EM, MAP EM with
parameter priors, sequential/incremental EM, and publication-quality
figure reproductions (Figures 15.10, 15.11, 15.13, 15.14, 15.15, 15.16)
for Bishop & Bishop's Deep Learning (2024).

Key Equations:
- Marginal Log-Likelihood (Eq. 15.51):
    \ln p(X \mid \theta) = \ln \sum_Z p(X, Z \mid \theta)
- Variational Decomposition (Eq. 15.52):
    \ln p(X \mid \theta) = \mathcal{L}(q, \theta) + \mathrm{KL}(q \parallel p)
- Evidence Lower Bound (ELBO, Eq. 15.53):
    \mathcal{L}(q, \theta) = \sum_Z q(Z) \ln \frac{p(X, Z \mid \theta)}{q(Z)}
- Kullback–Leibler Divergence (Eq. 15.54):
    \mathrm{KL}(q \parallel p) = -\sum_Z q(Z) \ln \frac{p(Z \mid X, \theta)}{q(Z)}
                               = \sum_Z q(Z) \ln \frac{q(Z)}{p(Z \mid X, \theta)} \ge 0
- Expected Complete-Data Log-Likelihood (Eq. 15.56):
    \mathcal{L}(q, \theta) = Q(\theta, \theta^{\text{old}}) + \mathrm{const}
- Posterior Factorization for i.i.d. Data (Eq. 15.57):
    p(Z \mid X, \theta) = \prod_{n=1}^N p(z_n \mid x_n, \theta)
- MAP Lower Bound with Parameter Priors (Eq. 15.59):
    \ln p(\theta \mid X) \ge \mathcal{L}(q, \theta) + \ln p(\theta) - \ln p(X)
- Sequential EM Sufficient Statistics Update (Eqs. 15.60 - 15.61):
    N_k^{\text{new}} = N_k^{\text{old}} + \gamma^{\text{new}}(z_{mk}) - \gamma^{\text{old}}(z_{mk})
    \mu_k^{\text{new}} = \mu_k^{\text{old}} + \frac{\gamma^{\text{new}}(z_{mk}) - \gamma^{\text{old}}(z_{mk})}{N_k^{\text{new}}} (x_m - \mu_k^{\text{old}})
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Ellipse
import numpy as np
from scipy.stats import multivariate_normal


def compute_elbo_decomposition(
    X: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
    q: Optional[np.ndarray] = None,
    reg_covar: float = 1e-6,
) -> Dict[str, Union[float, np.ndarray]]:
    r"""Compute exact variational decomposition: \ln p(X|\theta) = \mathcal{L}(q, \theta) + KL(q||p).

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Observed data points.
    means : np.ndarray, shape (K, D)
        Component mean vectors.
    covs : np.ndarray, shape (K, D, D)
        Component covariance matrices.
    weights : np.ndarray, shape (K,)
        Mixing coefficients \pi_k.
    q : np.ndarray, shape (N, K), optional
        Variational distribution q_n(z_nk). If None, set to true posterior p(Z|X, \theta).
    reg_covar : float, default=1e-6
        Diagonal regularization for covariances.

    Returns
    -------
    dict
        - "log_likelihood": float, \ln p(X|\theta)
        - "elbo": float, \mathcal{L}(q, \theta)
        - "kl_divergence": float, KL(q || p)
        - "posterior": np.ndarray, shape (N, K), p(z_nk = 1 | x_n, \theta)
        - "q": np.ndarray, shape (N, K), variational distribution q(Z)
    """
    X = np.asarray(X, dtype=float)
    N, D = X.shape
    K = len(weights)

    # 1. Compute joint likelihood components: p(x_n, z_nk = 1 | \theta) = \pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k)
    joint_probs = np.zeros((N, K))
    for k in range(K):
        cov_k = covs[k] + reg_covar * np.eye(D)
        mvn = multivariate_normal(mean=means[k], cov=cov_k, allow_singular=True)
        joint_probs[:, k] = np.maximum(weights[k] * mvn.pdf(X), 1e-300)

    # 2. Marginal likelihood: p(x_n | \theta) = \sum_k p(x_n, z_nk = 1 | \theta)
    marginal_probs = np.sum(joint_probs, axis=1)  # (N,)
    log_likelihood = float(np.sum(np.log(np.maximum(marginal_probs, 1e-300))))

    # 3. True posterior: p(z_nk = 1 | x_n, \theta) = p(x_n, z_nk = 1 | \theta) / p(x_n | \theta)
    posterior = joint_probs / marginal_probs[:, np.newaxis]

    # 4. If q is not provided, default to true posterior (optimal E-step where KL = 0)
    if q is None:
        q = posterior.copy()
    else:
        q = np.asarray(q, dtype=float)
        # Normalize q per row to ensure valid distribution
        q = np.maximum(q, 1e-300)
        q = q / np.sum(q, axis=1, keepdims=True)

    # 5. ELBO \mathcal{L}(q, \theta) = \sum_{n=1}^N \sum_{k=1}^K q_{nk} \ln \frac{p(x_n, z_nk = 1 | \theta)}{q_{nk}}
    elbo_terms = q * (np.log(joint_probs) - np.log(q))
    elbo = float(np.sum(elbo_terms))

    # 6. KL divergence: KL(q || p) = \sum_{n=1}^N \sum_{k=1}^K q_{nk} \ln \frac{q_{nk}}{p(z_nk = 1 | x_n, \theta)}
    kl_terms = q * (np.log(q) - np.log(posterior))
    kl_div = float(np.sum(kl_terms))

    return {
        "log_likelihood": log_likelihood,
        "elbo": elbo,
        "kl_divergence": kl_div,
        "posterior": posterior,
        "q": q,
    }


class SequentialGaussianMixtureEM:
    r"""Sequential / Incremental EM algorithm for Gaussian Mixture Model (Section 15.4.5).

    Updates sufficient statistics N_k, \mu_k, \Sigma_k, \pi_k incrementally one data point
    at a time according to Equations 15.60 - 15.61.

    Parameters
    ----------
    n_components : int, default=2
        Number of mixture components K.
    n_epochs : int, default=10
        Number of full passes over the dataset.
    reg_covar : float, default=1e-5
        Covariance regularization.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_epochs: int = 10,
        reg_covar: float = 1e-5,
    ):
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.reg_covar = reg_covar

        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None
        self.N_k_: Optional[np.ndarray] = None
        self.history_log_likelihood_: List[float] = []

    def fit(
        self,
        X: np.ndarray,
        init_means: np.ndarray,
        init_covs: np.ndarray,
        init_weights: Optional[np.ndarray] = None,
    ) -> "SequentialGaussianMixtureEM":
        r"""Fit sequential GMM on data X using incremental sufficient statistics."""
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        K = self.n_components

        means = np.array(init_means, dtype=float).copy()
        covs = np.array(init_covs, dtype=float).copy()
        if init_weights is None:
            weights = np.full(K, 1.0 / K)
        else:
            weights = np.array(init_weights, dtype=float).copy()

        # Initialize stored responsibilities \gamma_{nk}
        gamma_stored = np.zeros((N, K))
        for k in range(K):
            mvn = multivariate_normal(mean=means[k], cov=covs[k] + self.reg_covar * np.eye(D))
            gamma_stored[:, k] = weights[k] * mvn.pdf(X)
        gamma_stored = gamma_stored / np.sum(gamma_stored, axis=1, keepdims=True)

        # Sufficient statistics: S0 = N_k, S1 = \sum \gamma x, S2 = \sum \gamma x x^T
        S0 = np.sum(gamma_stored, axis=0)  # (K,)
        S1 = gamma_stored.T @ X  # (K, D)
        S2 = np.zeros((K, D, D))  # (K, D, D)
        for k in range(K):
            S2[k] = (gamma_stored[:, k : k + 1] * X).T @ X

        self.means_ = means
        self.covariances_ = covs
        self.weights_ = weights
        self.N_k_ = S0
        self.history_log_likelihood_ = []

        # Record initial log-likelihood
        decomp = compute_elbo_decomposition(X, means, covs, weights, reg_covar=self.reg_covar)
        self.history_log_likelihood_.append(decomp["log_likelihood"])

        for epoch in range(self.n_epochs):
            for m in range(N):
                x_m = X[m]
                old_gamma_m = gamma_stored[m].copy()

                # E-step for single point m:
                new_gamma_m = np.zeros(K)
                for k in range(K):
                    mu_k = S1[k] / max(S0[k], 1e-10)
                    cov_k = (S2[k] / max(S0[k], 1e-10)) - np.outer(mu_k, mu_k)
                    cov_k = 0.5 * (cov_k + cov_k.T) + self.reg_covar * np.eye(D)
                    mvn = multivariate_normal(mean=mu_k, cov=cov_k, allow_singular=True)
                    pi_k = S0[k] / N
                    new_gamma_m[k] = max(pi_k * mvn.pdf(x_m), 1e-300)
                total_density = np.sum(new_gamma_m)
                new_gamma_m = new_gamma_m / total_density

                delta_gamma = new_gamma_m - old_gamma_m
                gamma_stored[m] = new_gamma_m

                # M-step updates for sufficient statistics (Eqs. 15.60 - 15.61)
                x_outer = np.outer(x_m, x_m)
                for k in range(K):
                    dg = delta_gamma[k]
                    S0[k] = max(S0[k] + dg, 1e-6)
                    S1[k] = S1[k] + dg * x_m
                    S2[k] = S2[k] + dg * x_outer

            # Extract epoch parameters
            for k in range(K):
                means[k] = S1[k] / S0[k]
                cov_k = (S2[k] / S0[k]) - np.outer(means[k], means[k])
                covs[k] = 0.5 * (cov_k + cov_k.T) + self.reg_covar * np.eye(D)
            weights = S0 / np.sum(S0)

            decomp = compute_elbo_decomposition(X, means, covs, weights, reg_covar=self.reg_covar)
            self.history_log_likelihood_.append(decomp["log_likelihood"])

        self.means_ = means
        self.covariances_ = covs
        self.weights_ = weights
        self.N_k_ = S0
        return self


class MAPGaussianMixtureEM:
    r"""MAP EM algorithm for Gaussian Mixture Model with parameter priors (Section 15.4.3).

    Eliminates singularities in the likelihood function by placing:
    - Dirichlet prior \operatorname{Dir}(\pi \mid \alpha) on mixing coefficients
    - Conjugate Inverse-Wishart / Normal-Wishart prior on component parameters.

    Parameters
    ----------
    n_components : int, default=2
        Number of components K.
    max_iter : int, default=20
    alpha_prior : float, default=2.0
        Dirichlet concentration parameter \alpha_k > 1.
    beta_cov_prior : float, default=0.5
        Scale for covariance prior regularization \beta I.
    reg_covar : float, default=1e-6
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 20,
        alpha_prior: float = 2.0,
        beta_cov_prior: float = 0.5,
        reg_covar: float = 1e-6,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.alpha_prior = alpha_prior
        self.beta_cov_prior = beta_cov_prior
        self.reg_covar = reg_covar

        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None
        self.history_map_objective_: List[float] = []

    def fit(
        self,
        X: np.ndarray,
        init_means: np.ndarray,
        init_covs: np.ndarray,
        init_weights: Optional[np.ndarray] = None,
    ) -> "MAPGaussianMixtureEM":
        r"""Fit MAP GMM with parameter priors."""
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        K = self.n_components

        means = np.array(init_means, dtype=float).copy()
        covs = np.array(init_covs, dtype=float).copy()
        if init_weights is None:
            weights = np.full(K, 1.0 / K)
        else:
            weights = np.array(init_weights, dtype=float).copy()

        self.history_map_objective_ = []

        alpha = np.full(K, self.alpha_prior)
        S_0 = self.beta_cov_prior * np.eye(D)
        nu_0 = D + 2.0

        for iteration in range(self.max_iter):
            # E-step: Responsibilities
            gamma = np.zeros((N, K))
            for k in range(K):
                cov_k = covs[k] + self.reg_covar * np.eye(D)
                mvn = multivariate_normal(mean=means[k], cov=cov_k, allow_singular=True)
                gamma[:, k] = weights[k] * mvn.pdf(X)
            marginal = np.sum(gamma, axis=1, keepdims=True)
            gamma = gamma / np.maximum(marginal, 1e-300)

            N_k = np.sum(gamma, axis=0)

            # M-step with Priors (Section 15.4.3, Eq. 15.59)
            # 1. Update mixing weights with Dirichlet prior
            new_weights = (N_k + alpha - 1.0) / (N + np.sum(alpha - 1.0))
            new_weights = np.maximum(new_weights, 1e-6)
            new_weights = new_weights / np.sum(new_weights)

            # 2. Update means
            new_means = np.zeros_like(means)
            for k in range(K):
                new_means[k] = np.sum(gamma[:, k : k + 1] * X, axis=0) / max(N_k[k], 1e-10)

            # 3. Update covariances with regularizing prior S_0
            new_covs = np.zeros_like(covs)
            for k in range(K):
                diff = X - new_means[k]
                weighted_diff = gamma[:, k : k + 1] * diff
                data_cov = weighted_diff.T @ diff
                new_covs[k] = (data_cov + S_0) / (N_k[k] + nu_0) + self.reg_covar * np.eye(D)

            means = new_means
            covs = new_covs
            weights = new_weights

            # Compute log posterior objective \ln p(\theta | X) \propto \ln p(X | \theta) + \ln p(\theta)
            decomp = compute_elbo_decomposition(X, means, covs, weights, reg_covar=self.reg_covar)
            log_prior = np.sum((alpha - 1.0) * np.log(weights))
            for k in range(K):
                # Inverse-Wishart prior density term
                log_prior += -0.5 * (nu_0 + D + 1) * np.linalg.slogdet(covs[k])[1] - 0.5 * np.trace(S_0 @ np.linalg.inv(covs[k]))
            map_obj = decomp["log_likelihood"] + float(log_prior)
            self.history_map_objective_.append(map_obj)

        self.means_ = means
        self.covariances_ = covs
        self.weights_ = weights
        return self


# ==============================================================================
# Figure Generation Functions (Faithful Reproductions of Figures 15.10, 15.11, 15.13, 15.14, 15.15, 15.16)
# ==============================================================================

def generate_figure_15_10(save_path: Union[str, Path]) -> None:
    r"""Generate Figure 15.10: Directed graphical model for complete data (both z_n and x_n observed)."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.2, 5.0)
    ax.set_ylim(-0.5, 4.5)

    # Plate N box (blue outline)
    plate = FancyBboxPatch(
        (1.5, 0.2), 2.2, 3.8,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        ec="#0000cc", fc="none", lw=2.0
    )
    ax.add_patch(plate)
    ax.text(3.3, 0.5, r"$N$", fontsize=16, fontfamily="serif", fontstyle="italic")

    # Complete data: Both z_n and x_n are OBSERVED!
    # Shaded light blue/purple with red outline (Bishop textbook color convention)
    node_z = plt.Circle((2.5, 3.0), 0.5, ec="#e41a1c", fc="#d0d7de", lw=2.5, zorder=3)
    ax.add_patch(node_z)
    ax.text(2.5, 3.0, r"$\mathbf{z}_n$", fontsize=15, ha="center", va="center", weight="bold", zorder=4)

    node_x = plt.Circle((2.5, 1.2), 0.5, ec="#e41a1c", fc="#d0d7de", lw=2.5, zorder=3)
    ax.add_patch(node_x)
    ax.text(2.5, 1.2, r"$\mathbf{x}_n$", fontsize=15, ha="center", va="center", weight="bold", zorder=4)

    # Directed arrow from z_n to x_n
    ax.annotate(
        "", xy=(2.5, 1.7), xytext=(2.5, 2.5),
        arrowprops=dict(arrowstyle="-|>", color="#e41a1c", lw=2.0, mutation_scale=18)
    )

    # Parameters: \pi -> z_n
    ax.text(0.6, 3.0, r"$\boldsymbol{\pi}$", fontsize=18, ha="center", va="center", weight="bold")
    ax.annotate(
        "", xy=(2.0, 3.0), xytext=(0.9, 3.0),
        arrowprops=dict(arrowstyle="-|>", color="#e41a1c", lw=2.0, mutation_scale=18)
    )

    # Parameters: \mu -> x_n
    ax.text(0.6, 1.2, r"$\boldsymbol{\mu}$", fontsize=18, ha="center", va="center", weight="bold")
    ax.annotate(
        "", xy=(2.0, 1.2), xytext=(0.9, 1.2),
        arrowprops=dict(arrowstyle="-|>", color="#e41a1c", lw=2.0, mutation_scale=18)
    )

    # Parameters: \Sigma -> x_n
    ax.text(4.5, 1.2, r"$\mathbf{\Sigma}$", fontsize=18, ha="center", va="center", weight="bold")
    ax.annotate(
        "", xy=(3.0, 1.2), xytext=(4.2, 1.2),
        arrowprops=dict(arrowstyle="-|>", color="#e41a1c", lw=2.0, mutation_scale=18)
    )

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    # Also save to root result/ if standard figure name
    if save_path.name.startswith("fig_"):
        root_result = Path("result") / save_path.name
        root_result.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(root_result, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_figure_15_11(save_path: Union[str, Path]) -> None:
    r"""Generate Figure 15.11: Probabilistic graphical model for Hidden Markov Model (HMM)."""
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=300)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 3.0)

    # Nodes positions: z_1, z_2, ..., z_N at y = 2.0; x_1, x_2, ..., x_N at y = 0.5
    xs = [1.0, 3.0, 7.0]
    z_labels = [r"$\mathbf{z}_1$", r"$\mathbf{z}_2$", r"$\mathbf{z}_N$"]
    x_labels = [r"$\mathbf{x}_1$", r"$\mathbf{x}_2$", r"$\mathbf{x}_N$"]

    # Draw z nodes (latent: unobserved / white with red border)
    for x_pos, z_lab in zip(xs, z_labels):
        circle_z = plt.Circle((x_pos, 2.0), 0.45, ec="#e41a1c", fc="white", lw=2.5, zorder=3)
        ax.add_patch(circle_z)
        ax.text(x_pos, 2.0, z_lab, fontsize=14, ha="center", va="center", weight="bold", zorder=4)

    # Draw x nodes (observed: shaded blue with red border)
    for x_pos, x_lab in zip(xs, x_labels):
        circle_x = plt.Circle((x_pos, 0.5), 0.45, ec="#e41a1c", fc="#d0d7de", lw=2.5, zorder=3)
        ax.add_patch(circle_x)
        ax.text(x_pos, 0.5, x_lab, fontsize=14, ha="center", va="center", weight="bold", zorder=4)

    # Vertical arrows z_n -> x_n
    for x_pos in xs:
        ax.annotate(
            "", xy=(x_pos, 0.95), xytext=(x_pos, 1.55),
            arrowprops=dict(arrowstyle="-|>", color="#e41a1c", lw=2.0, mutation_scale=16)
        )

    # Horizontal chain: z_1 -> z_2
    ax.annotate(
        "", xy=(2.55, 2.0), xytext=(1.45, 2.0),
        arrowprops=dict(arrowstyle="-|>", color="#e41a1c", lw=2.0, mutation_scale=16)
    )

    # z_2 -> dots
    ax.annotate(
        "", xy=(4.6, 2.0), xytext=(3.45, 2.0),
        arrowprops=dict(arrowstyle="-|>", color="#e41a1c", lw=2.0, mutation_scale=16)
    )
    # Dots ...
    ax.text(5.0, 2.0, r"$\dots$", fontsize=20, ha="center", va="center", color="#e41a1c")

    # dots -> z_N
    ax.annotate(
        "", xy=(6.55, 2.0), xytext=(5.4, 2.0),
        arrowprops=dict(arrowstyle="-|>", color="#e41a1c", lw=2.0, mutation_scale=16)
    )

    # z_N -> (outgoing arrow)
    ax.annotate(
        "", xy=(8.3, 2.0), xytext=(7.45, 2.0),
        arrowprops=dict(arrowstyle="-|>", color="#e41a1c", lw=2.0, mutation_scale=16)
    )

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if save_path.name.startswith("fig_"):
        root_result = Path("result") / save_path.name
        root_result.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(root_result, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_figure_15_13(save_path: Union[str, Path]) -> None:
    r"""Generate Figure 15.13: Illustration of the decomposition \ln p(X|\theta) = \mathcal{L}(q, \theta) + KL(q||p)."""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.axis("off")
    ax.set_xlim(-0.2, 5.0)
    ax.set_ylim(-0.5, 4.5)

    # Gray horizontal bar at base y = 0
    base_rect = FancyBboxPatch(
        (0.8, -0.1), 3.8, 0.1,
        boxstyle="square,pad=0",
        ec="black", fc="#c0c0c0", lw=1.5
    )
    ax.add_patch(base_rect)

    # Blue horizontal line: ELBO \mathcal{L}(q, \theta) at y = 2.0
    ax.plot([0.8, 2.7], [2.0, 2.0], color="blue", lw=3.0)

    # Red horizontal line: Log-likelihood \ln p(X|\theta) at y = 4.0
    ax.plot([0.8, 4.6], [4.0, 4.0], color="red", lw=3.0)

    # Double-headed arrow for \mathcal{L}(q, \theta): from 0 to 2.0 at x = 1.75
    ax.annotate(
        "", xy=(1.75, 2.0), xytext=(1.75, 0.0),
        arrowprops=dict(arrowstyle="<->", color="black", lw=2.0, mutation_scale=18)
    )
    ax.text(1.35, 1.0, r"$\mathcal{L}(q, \boldsymbol{\theta})$", fontsize=15, ha="right", va="center")

    # Double-headed arrow for KL(q || p): from 2.0 to 4.0 at x = 1.2
    ax.annotate(
        "", xy=(1.2, 4.0), xytext=(1.2, 2.0),
        arrowprops=dict(arrowstyle="<->", color="black", lw=2.0, mutation_scale=18)
    )
    ax.text(0.8, 3.0, r"$\mathrm{KL}(q \parallel p)$", fontsize=15, ha="right", va="center")

    # Double-headed arrow for \ln p(X|\theta): from 0 to 4.0 at x = 3.45
    ax.annotate(
        "", xy=(3.45, 4.0), xytext=(3.45, 0.0),
        arrowprops=dict(arrowstyle="<->", color="black", lw=2.0, mutation_scale=18)
    )
    ax.text(3.6, 2.0, r"$\ln p(\mathbf{X} \mid \boldsymbol{\theta})$", fontsize=15, ha="left", va="center")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if save_path.name.startswith("fig_"):
        root_result = Path("result") / save_path.name
        root_result.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(root_result, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_figure_15_14(save_path: Union[str, Path]) -> None:
    r"""Generate Figure 15.14: Illustration of the E step of the EM algorithm."""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.axis("off")
    ax.set_xlim(-0.2, 5.0)
    ax.set_ylim(-0.5, 4.5)

    # Gray horizontal bar at base y = 0
    base_rect = FancyBboxPatch(
        (0.8, -0.1), 3.8, 0.1,
        boxstyle="square,pad=0",
        ec="black", fc="#c0c0c0", lw=1.5
    )
    ax.add_patch(base_rect)

    # Dashed blue horizontal line: previous lower bound at y = 1.6
    ax.plot([0.8, 2.7], [1.6, 1.6], color="blue", lw=2.5, linestyle="--")

    # Solid blue line moved up to y = 4.0, meeting the red line!
    ax.plot([0.8, 2.7], [4.0, 4.0], color="blue", lw=3.0)
    # Solid red line: \ln p(X | \theta^{old}) from 2.7 to 4.6
    ax.plot([2.7, 4.6], [4.0, 4.0], color="red", lw=3.0)

    # Upward vertical blue arrow from 1.6 to 4.0 showing lower bound moving up
    ax.annotate(
        "", xy=(2.5, 4.0), xytext=(2.5, 1.6),
        arrowprops=dict(arrowstyle="-|>", color="blue", lw=3.0, mutation_scale=22)
    )

    # Label KL(q||p) = 0
    ax.text(0.7, 4.0, r"$\mathrm{KL}(q \parallel p) = 0$", fontsize=15, ha="right", va="center")

    # Double-headed arrow for \mathcal{L}(q, \theta^{old}): from 0 to 4.0 at x = 1.75
    ax.annotate(
        "", xy=(1.75, 4.0), xytext=(1.75, 0.0),
        arrowprops=dict(arrowstyle="<->", color="black", lw=2.0, mutation_scale=18)
    )
    ax.text(1.65, 1.0, r"$\mathcal{L}(q, \boldsymbol{\theta}^{\mathrm{old}})$", fontsize=15, ha="right", va="center")

    # Double-headed arrow for \ln p(X|\theta^{old}): from 0 to 4.0 at x = 3.45
    ax.annotate(
        "", xy=(3.45, 4.0), xytext=(3.45, 0.0),
        arrowprops=dict(arrowstyle="<->", color="black", lw=2.0, mutation_scale=18)
    )
    ax.text(3.6, 1.0, r"$\ln p(\mathbf{X} \mid \boldsymbol{\theta}^{\mathrm{old}})$", fontsize=15, ha="left", va="center")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if save_path.name.startswith("fig_"):
        root_result = Path("result") / save_path.name
        root_result.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(root_result, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_figure_15_15(save_path: Union[str, Path]) -> None:
    r"""Generate Figure 15.15: Illustration of the M step of the EM algorithm."""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.axis("off")
    ax.set_xlim(-0.2, 5.0)
    ax.set_ylim(-0.5, 4.5)

    # Gray horizontal bar at base y = 0
    base_rect = FancyBboxPatch(
        (0.8, -0.1), 3.8, 0.1,
        boxstyle="square,pad=0",
        ec="black", fc="#c0c0c0", lw=1.5
    )
    ax.add_patch(base_rect)

    # Dashed lines representing state after E-step at y = 2.8:
    ax.plot([0.8, 2.6], [2.8, 2.8], color="blue", lw=2.0, linestyle="--")
    ax.plot([2.6, 4.6], [2.8, 2.8], color="red", lw=2.0, linestyle="--")

    # In M-step: \theta -> \theta^{new}
    # New ELBO \mathcal{L}(q, \theta^{new}) at y = 3.3
    ax.plot([0.8, 2.6], [3.3, 3.3], color="blue", lw=3.0)

    # New log-likelihood \ln p(X | \theta^{new}) at y = 4.1
    ax.plot([0.8, 4.6], [4.1, 4.1], color="red", lw=3.0)

    # Upward blue arrow from 2.8 to 3.3 showing lower bound increase
    ax.annotate(
        "", xy=(2.4, 3.3), xytext=(2.4, 2.8),
        arrowprops=dict(arrowstyle="-|>", color="blue", lw=3.0, mutation_scale=20)
    )

    # Upward red arrow from 2.8 to 4.1 showing larger log-likelihood increase
    ax.annotate(
        "", xy=(4.0, 4.1), xytext=(4.0, 2.8),
        arrowprops=dict(arrowstyle="-|>", color="red", lw=3.0, mutation_scale=20)
    )

    # Double-headed arrow for KL(q||p) between 3.3 and 4.1 at x = 1.2
    ax.annotate(
        "", xy=(1.2, 4.1), xytext=(1.2, 3.3),
        arrowprops=dict(arrowstyle="<->", color="black", lw=2.0, mutation_scale=16)
    )
    ax.text(0.7, 3.7, r"$\mathrm{KL}(q \parallel p)$", fontsize=15, ha="right", va="center")

    # Double-headed arrow for \mathcal{L}(q, \theta^{new}): from 0 to 3.3 at x = 1.75
    ax.annotate(
        "", xy=(1.75, 3.3), xytext=(1.75, 0.0),
        arrowprops=dict(arrowstyle="<->", color="black", lw=2.0, mutation_scale=18)
    )
    ax.text(1.65, 1.5, r"$\mathcal{L}(q, \boldsymbol{\theta}^{\mathrm{new}})$", fontsize=15, ha="right", va="center")

    # Double-headed arrow for \ln p(X|\theta^{new}): from 0 to 4.1 at x = 3.45
    ax.annotate(
        "", xy=(3.45, 4.1), xytext=(3.45, 0.0),
        arrowprops=dict(arrowstyle="<->", color="black", lw=2.0, mutation_scale=18)
    )
    ax.text(3.6, 1.5, r"$\ln p(\mathbf{X} \mid \boldsymbol{\theta}^{\mathrm{new}})$", fontsize=15, ha="left", va="center")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if save_path.name.startswith("fig_"):
        root_result = Path("result") / save_path.name
        root_result.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(root_result, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_figure_15_16(save_path: Union[str, Path]) -> None:
    r"""Generate Figure 15.16: EM operation in parameter space (tangential bounds)."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.set_xlim(-0.2, 6.0)
    ax.set_ylim(-0.5, 4.5)

    # Box outline around plot
    box = patches.Rectangle((0.0, 0.0), 5.8, 4.2, ec="black", fc="none", lw=1.5)
    ax.add_patch(box)
    ax.set_xticks([])
    ax.set_yticks([])

    # Parameter range theta
    theta = np.linspace(0.4, 5.75, 300)

    # Red curve: Incomplete-data log-likelihood \ln p(X|\theta) (concave smooth curve)
    # Peak near theta = 3.8
    ll = 3.8 - 0.22 * (theta - 3.8) ** 2
    ax.plot(theta, ll, color="red", lw=2.5, zorder=3)
    ax.text(4.7, 2.7, r"$\ln p(\mathbf{X} \mid \boldsymbol{\theta})$", fontsize=15, color="black")

    # Point \theta^{old} = 1.0
    theta_old = 1.0
    ll_old = 3.8 - 0.22 * (theta_old - 3.8) ** 2  # approx 2.075

    # Blue curve: Lower bound \mathcal{L}(\theta, \theta^{old})
    # Touches tangentially at theta_old, peak at theta_new = 2.2
    theta_new = 2.2
    B_blue = 0.52
    A_blue = ll_old + B_blue * (theta_old - theta_new) ** 2  # approx 2.824
    theta_blue = np.linspace(0.6, 3.2, 150)
    L_blue = A_blue - B_blue * (theta_blue - theta_new) ** 2
    ax.plot(theta_blue, L_blue, color="blue", lw=2.2, zorder=3)
    ax.text(3.3, 1.8, r"$\mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\theta}^{(\mathrm{old})})$", fontsize=15, color="black")

    # Green curve: Lower bound \mathcal{L}(\theta, \theta^{new})
    # Touches tangentially at theta_new = 2.2, peak at theta_3 = 3.2
    ll_new = 3.8 - 0.22 * (theta_new - 3.8) ** 2  # approx 3.237
    theta_3 = 3.2
    B_green = 0.40
    A_green = ll_new + B_green * (theta_new - theta_3) ** 2
    theta_green = np.linspace(1.8, 4.0, 150)
    L_green = A_green - B_green * (theta_green - theta_3) ** 2
    ax.plot(theta_green, L_green, color="green", lw=2.2, zorder=3)

    # Dashed vertical lines to x-axis
    ax.plot([theta_old, theta_old], [0.0, ll_old], color="gray", lw=1.5, linestyle="--")
    ax.plot([theta_new, theta_new], [0.0, ll_new], color="gray", lw=1.5, linestyle="--")

    # X-axis tick labels
    ax.text(theta_old, -0.25, r"$\boldsymbol{\theta}^{(\mathrm{old})}$", fontsize=14, ha="center")
    ax.text(theta_new, -0.25, r"$\boldsymbol{\theta}^{(\mathrm{new})}$", fontsize=14, ha="center")

    # Grey dots on curves
    # Point on blue bound before E-step
    ax.scatter([theta_old], [0.9], color="gray", s=50, zorder=5)
    # Point at contact theta_old
    ax.scatter([theta_old], [ll_old], color="gray", s=50, zorder=5)
    # Point at peak of blue curve theta_new
    ax.scatter([theta_new], [A_blue], color="gray", s=50, zorder=5)
    # Point on red curve at theta_new
    ax.scatter([theta_new], [ll_new], color="gray", s=50, zorder=5)
    # Point at peak of green curve theta_3
    ax.scatter([theta_3], [A_green], color="gray", s=50, zorder=5)

    # E-step vertical arrow at theta_old (from 0.9 up to ll_old)
    ax.annotate(
        "", xy=(theta_old, ll_old), xytext=(theta_old, 0.9),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0, mutation_scale=16)
    )
    ax.text(theta_old + 0.15, 1.4, "E", fontsize=15, ha="left", va="center")

    # M-step curved arrow from theta_old to theta_new along blue bound
    ax.annotate(
        "", xy=(theta_new - 0.05, A_blue), xytext=(theta_old + 0.1, ll_old + 0.1),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0, connectionstyle="arc3,rad=-0.25", mutation_scale=16)
    )
    ax.text(1.6, 2.25, "M", fontsize=15, ha="center", va="top")

    # Next E-step vertical arrow at theta_new (from A_blue up to ll_new)
    ax.annotate(
        "", xy=(theta_new, ll_new), xytext=(theta_new, A_blue),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0, mutation_scale=16)
    )

    # Next M-step arrow along green bound
    ax.annotate(
        "", xy=(theta_3 - 0.05, A_green), xytext=(theta_new + 0.15, ll_new + 0.05),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0, connectionstyle="arc3,rad=-0.2", mutation_scale=16)
    )

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if save_path.name.startswith("fig_"):
        root_result = Path("result") / save_path.name
        root_result.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(root_result, dpi=300, bbox_inches="tight")
    plt.close(fig)
