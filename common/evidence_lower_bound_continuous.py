"""
common/evidence_lower_bound_continuous.py
=========================================
Core implementations for Chapter 16 Section 16.3: Evidence Lower Bound.
Christopher M. Bishop & Hugh Bishop (2024). Deep Learning: Foundations and Concepts.

Mathematical models:
- EM for Probabilistic PCA (Section 16.3.1)
  - Complete-data log-likelihood                     [Eq. 16.64]
  - Expected complete log-likelihood                 [Eq. 16.65]
  - E-step:
    E[z_n] = M^-1 W^T (x_n - x_bar)                  [Eq. 16.66]
    E[z_n z_n^T] = sigma^2 M^-1 + E[z_n] E[z_n]^T    [Eq. 16.67]
  - M-step:
    W_new = [sum (x_n - x_bar) E[z_n]^T] [sum E[z_n z_n^T]]^-1 [Eq. 16.68]
    sigma_new^2 = 1/(ND) sum { ||x_n - x_bar||^2 - 2 E[z_n]^T W_new^T (x_n - x_bar)
                               + Tr(E[z_n z_n^T] W_new^T W_new) } [Eq. 16.69]
- EM for Standard PCA in zero-noise limit sigma^2 -> 0 (Section 16.3.2)
  - E-step: Omega = (W_old^T W_old)^-1 W_old^T X_tilde^T [Eq. 16.70]
  - M-step: W_new = X_tilde^T Omega^T (Omega Omega^T)^-1 [Eq. 16.71]
- EM for Factor Analysis (Section 16.3.3)
  - G = (I + W^T Psi^-1 W)^-1                        [Eq. 16.74]
  - E[z_n] = G W^T Psi^-1 (x_n - x_bar)              [Eq. 16.72]
  - E[z_n z_n^T] = G + E[z_n] E[z_n]^T               [Eq. 16.73]
  - W_new = [sum (x_n - x_bar) E[z_n]^T] [sum E[z_n z_n^T]]^-1 [Eq. 16.75]
  - Psi_new = diag(S - 1/N W_new sum E[z_n] (x_n - x_bar)^T)    [Eq. 16.76]
- Figure 16.10 Generator: EM algorithm for PCA on synthetic data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from common.plot_utils import setup_style


def _save_figure(fig: plt.Figure, base_name: str, save_path: Optional[Union[str, Path]] = None) -> None:
    """Save figure to requested path and standard result/ locations."""
    repo_root = Path(__file__).resolve().parent.parent
    ch_result_dir = repo_root / "16" / "result"
    global_result_dir = repo_root / "result"
    ch_result_dir.mkdir(parents=True, exist_ok=True)
    global_result_dir.mkdir(parents=True, exist_ok=True)

    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=300, bbox_inches="tight")
        print(f"Figure saved successfully to: {save_p}")
        if save_p.name.startswith("fig_"):
            fig.savefig(ch_result_dir / f"{base_name}.png", dpi=300, bbox_inches="tight")
            fig.savefig(global_result_dir / f"{base_name}.png", dpi=300, bbox_inches="tight")
    else:
        p1 = ch_result_dir / f"{base_name}.png"
        p2 = global_result_dir / f"{base_name}.png"
        fig.savefig(p1, dpi=300, bbox_inches="tight")
        fig.savefig(p2, dpi=300, bbox_inches="tight")
        print(f"Figure saved successfully to: {p1} and {p2}")


class ProbabilisticPCA_EM:
    r"""EM algorithm for Probabilistic Principal Component Analysis.

    Maximizes the Evidence Lower Bound (ELBO) / log-likelihood iteratively.
    Avoids explicit evaluation of the full D x D covariance matrix when M << D.

    Parameters
    ----------
    n_components : int
        Dimension M of the latent space.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Tolerance for log-likelihood convergence.
    random_state : Optional[int]
        Random seed for initializing W.
    """

    def __init__(self, n_components: int = 1, max_iter: int = 100, tol: float = 1e-4, random_state: Optional[int] = 42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.mu_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None
        self.sigma2_: Optional[float] = None
        self.history_: List[Dict[str, Union[float, np.ndarray]]] = []

    def fit(self, X: np.ndarray) -> "ProbabilisticPCA_EM":
        """Fit PPCA model via EM algorithm."""
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        M = self.n_components

        self.mu_ = np.mean(X, axis=0)
        Xc = X - self.mu_  # (N, D)

        # Initialize W and sigma^2
        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.standard_normal((D, M))
        # Initial sigma2 from residual sample variance
        self.sigma2_ = max(float(np.var(Xc)) / 2.0, 1e-4)

        prev_ll = -np.inf
        self.history_ = []

        for it in range(self.max_iter):
            # 1. E-step (Eq. 16.66, 16.67)
            M_mat = self.W_.T @ self.W_ + self.sigma2_ * np.eye(M)  # (M, M)
            M_inv = np.linalg.inv(M_mat)

            # E[z_n] = M^-1 W^T (x_n - mu)
            Ez = (Xc @ self.W_) @ M_inv  # (N, M)

            # E[z_n z_n^T] = sigma^2 M^-1 + E[z_n] E[z_n]^T
            # Sum over N: N * sigma^2 M_inv + Ez.T @ Ez
            sum_Ezz = N * self.sigma2_ * M_inv + Ez.T @ Ez  # (M, M)

            # 2. M-step (Eq. 16.68, 16.69)
            # W_new = (sum_n (x_n - mu) E[z_n]^T) (sum_n E[z_n z_n^T])^-1
            sum_x_Ez = Xc.T @ Ez  # (D, M)
            W_new = sum_x_Ez @ np.linalg.inv(sum_Ezz)

            # sigma_new^2 = 1/(ND) sum { ||x_n - mu||^2 - 2 E[z_n]^T W_new^T (x_n - mu) + Tr(E[z_n z_n^T] W_new^T W_new) }
            sq_norm_sum = np.sum(Xc**2)
            cross_term = 2.0 * np.sum(Ez * (Xc @ W_new))
            quad_term = np.trace(sum_Ezz @ (W_new.T @ W_new))

            sigma2_new = (sq_norm_sum - cross_term + quad_term) / (N * D)
            self.sigma2_ = max(float(sigma2_new), 1e-10)
            self.W_ = W_new

            # 3. Compute log-likelihood (marginal Gaussian)
            C = self.W_ @ self.W_.T + self.sigma2_ * np.eye(D)
            sign, log_det_C = np.linalg.slogdet(C)
            if sign > 0:
                C_inv = np.linalg.inv(C)
                ll = -0.5 * N * (D * np.log(2.0 * np.pi) + log_det_C) - 0.5 * np.sum((Xc @ C_inv) * Xc)
            else:
                ll = prev_ll

            self.history_.append({
                "iteration": it + 1,
                "log_likelihood": ll,
                "sigma2": self.sigma2_,
            })

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute posterior latent mean E[z | x]."""
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.mu_
        M_mat = self.W_.T @ self.W_ + self.sigma2_ * np.eye(self.n_components)
        M_inv = np.linalg.inv(M_mat)
        return (Xc @ self.W_) @ M_inv

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct data points from posterior latent means."""
        Ez = self.transform(X)
        return Ez @ self.W_.T + self.mu_


class StandardPCA_EM:
    r"""EM algorithm for Standard (Deterministic) PCA (Roweis 1998, Section 16.3.2).

    Obtained by taking the zero-noise limit sigma^2 -> 0.
    E-step: orthogonal projection of data onto current principal subspace.
    M-step: update principal subspace to minimize squared reconstruction error.

    Formulas:
      E-step: Omega = (W^T W)^-1 W^T X_tilde^T          [Eq. 16.70]
      M-step: W_new = X_tilde^T Omega^T (Omega Omega^T)^-1 [Eq. 16.71]

    Parameters
    ----------
    n_components : int
        Number of principal components M.
    max_iter : int
        Maximum iterations.
    tol : float
        Tolerance for subspace convergence.
    """

    def __init__(self, n_components: int = 1, max_iter: int = 100, tol: float = 1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

        self.mu_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None
        self.history_: List[Dict[str, np.ndarray]] = []

    def fit(self, X: np.ndarray, W_init: Optional[np.ndarray] = None) -> "StandardPCA_EM":
        """Fit standard PCA via Roweis's EM algorithm."""
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        M = self.n_components

        self.mu_ = np.mean(X, axis=0)
        Xc = X - self.mu_  # (N, D) = X_tilde

        if W_init is not None:
            W = np.asarray(W_init, dtype=np.float64).reshape(D, M)
        else:
            rng = np.random.default_rng(42)
            W = rng.standard_normal((D, M))
            W, _ = np.linalg.qr(W)

        self.history_ = [{"W": W.copy()}]

        for it in range(self.max_iter):
            # E-step (Eq. 16.70): Omega = (W^T W)^-1 W^T X_tilde^T  (M, N)
            Omega = np.linalg.inv(W.T @ W) @ W.T @ Xc.T

            # M-step (Eq. 16.71): W_new = X_tilde^T Omega^T (Omega Omega^T)^-1 (D, M)
            W_new = Xc.T @ Omega.T @ np.linalg.inv(Omega @ Omega.T)

            self.history_.append({"W": W_new.copy(), "Omega": Omega.copy()})

            # Check convergence via subspace alignment
            # Angle between subspaces
            if np.allclose(W, W_new, atol=self.tol):
                W = W_new
                break
            W = W_new

        self.W_ = W
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Orthogonal projection of X onto the learned subspace."""
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.mu_
        # (N, D) @ (D, M) @ (M, M) -> (N, M)
        return (Xc @ self.W_) @ np.linalg.inv(self.W_.T @ self.W_)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct points from subspace projection."""
        Z = self.transform(X)
        return Z @ self.W_.T + self.mu_


class FactorAnalysis_EM:
    r"""EM algorithm for Factor Analysis (Section 16.3.3).

    E-step:
      G = (I + W^T Psi^-1 W)^-1                           [Eq. 16.74]
      E[z_n] = G W^T Psi^-1 (x_n - x_bar)                 [Eq. 16.72]
      E[z_n z_n^T] = G + E[z_n] E[z_n]^T                  [Eq. 16.73]
    M-step:
      W_new = [sum (x_n - x_bar) E[z_n]^T] [sum E[z_n z_n^T]]^-1 [Eq. 16.75]
      Psi_new = diag(S - 1/N W_new sum E[z_n] (x_n - x_bar)^T)    [Eq. 16.76]

    Parameters
    ----------
    n_components : int
        Number of factors M.
    max_iter : int
        Maximum iterations.
    tol : float
        Tolerance for log-likelihood convergence.
    """

    def __init__(self, n_components: int = 1, max_iter: int = 100, tol: float = 1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

        self.mu_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None
        self.psi_: Optional[np.ndarray] = None
        self.history_: List[float] = []

    def fit(self, X: np.ndarray) -> "FactorAnalysis_EM":
        """Fit Factor Analysis model via EM algorithm."""
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        M = self.n_components

        self.mu_ = np.mean(X, axis=0)
        Xc = X - self.mu_
        S = (Xc.T @ Xc) / N

        # Initialize W with scaled SVD
        eigvals, eigvecs = np.linalg.eigh(S)
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        init_noise = max(float(np.mean(eigvals[M:])), 1e-4)
        scale = np.sqrt(np.maximum(eigvals[:M] - init_noise, 1e-4))
        self.W_ = eigvecs[:, :M] * scale[None, :]
        self.psi_ = np.full(D, init_noise)

        prev_ll = -np.inf
        self.history_ = []

        for it in range(self.max_iter):
            # E-step (Eq. 16.72 - 16.74)
            psi_inv = 1.0 / np.maximum(self.psi_, 1e-8)
            W_psi_inv = self.W_ * psi_inv[:, None]  # (D, M)
            G = np.linalg.inv(np.eye(M) + self.W_.T @ W_psi_inv)  # (M, M)

            Ez = (Xc @ W_psi_inv) @ G  # (N, M)
            sum_Ezz = N * G + Ez.T @ Ez  # (M, M)

            # M-step (Eq. 16.75, 16.76)
            W_new = (Xc.T @ Ez) @ np.linalg.inv(sum_Ezz)

            # Psi_new = diag(S - 1/N W_new (sum Ez (x_n - mu)^T))
            diag_term = np.diag(W_new @ (Ez.T @ Xc / N))
            psi_new = np.maximum(np.diag(S) - diag_term, 1e-6)

            self.W_ = W_new
            self.psi_ = psi_new

            # Marginal log-likelihood
            C = self.W_ @ self.W_.T + np.diag(self.psi_)
            sign, log_det_C = np.linalg.slogdet(C)
            if sign > 0:
                C_inv = np.linalg.inv(C)
                ll = -0.5 * N * (D * np.log(2.0 * np.pi) + log_det_C + np.trace(C_inv @ S))
            else:
                ll = prev_ll

            self.history_.append(ll)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute factor scores E[z | x]."""
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.mu_
        psi_inv = 1.0 / np.maximum(self.psi_, 1e-8)
        W_psi_inv = self.W_ * psi_inv[:, None]
        G = np.linalg.inv(np.eye(self.n_components) + self.W_.T @ W_psi_inv)
        return (Xc @ W_psi_inv) @ G


# =============================================================================
# Faithful Reproduction of Figure 16.10
# =============================================================================

# Exact 10 synthetic points from Bishop (2024) Figure 16.10
FIG16_10_DATA = np.array([
    [-0.25, -0.58],
    [ 1.14, -0.15],
    [-0.87, -0.77],
    [ 1.84,  2.06],
    [ 0.07, -0.81],
    [-0.58,  0.34],
    [ 0.37,  1.29],
    [-0.16, -0.18],
    [-0.46, -0.20],
    [-1.13, -1.00]
])


def generate_figure_16_10(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    r"""Figure 16.10: Synthetic data illustrating the EM algorithm for PCA.

    (a) Green data points with true principal components (dashed lines scaled by sqrt(lambda)).
    (b) Initial principal subspace (red) with latent projections Z W^T (cyan) and orthogonal error lines (blue).
    (c) After 1st M step: W updated, Z held fixed.
    (d) After successive E step: Z updated to orthogonal projections onto new W.
    (e) After 2nd M step: W updated again.
    (f) Converged solution: principal subspace aligned with dominant eigenvector.
    """
    setup_style()
    X = FIG16_10_DATA.copy()
    N, D = X.shape
    mu = np.mean(X, axis=0)
    Xc = X - mu

    # True principal components (for Fig 16.10a)
    S = (Xc.T @ Xc) / N
    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    u1, u2 = eigvecs[:, 0], eigvecs[:, 1]
    s1, s2 = np.sqrt(eigvals[0]), np.sqrt(eigvals[1])

    # Initial W: direction (1, -0.5)
    w_dir = np.array([1.0, -0.5])
    w_dir /= np.linalg.norm(w_dir)
    W_init = w_dir.reshape(D, 1)

    # Initial E-step for configuration (b)
    Omega_init = np.linalg.inv(W_init.T @ W_init) @ W_init.T @ Xc.T  # (1, N)
    proj_b = (W_init @ Omega_init).T + mu  # (N, D)

    # Step 1: M-step (W updated, Omega held fixed) -> panel (c)
    W_step1 = Xc.T @ Omega_init.T @ np.linalg.inv(Omega_init @ Omega_init.T)

    # Step 1: E-step (Omega updated with W_step1) -> panel (d)
    Omega_step1 = np.linalg.inv(W_step1.T @ W_step1) @ W_step1.T @ Xc.T
    proj_d = (W_step1 @ Omega_step1).T + mu

    # Step 2: M-step -> panel (e)
    W_step2 = Xc.T @ Omega_step1.T @ np.linalg.inv(Omega_step1 @ Omega_step1.T)

    # Converged solution -> panel (f)
    W_conv = W_step2.copy()
    for _ in range(25):
        Omega = np.linalg.inv(W_conv.T @ W_conv) @ W_conv.T @ Xc.T
        W_conv = Xc.T @ Omega.T @ np.linalg.inv(Omega @ Omega.T)
    Omega_conv = np.linalg.inv(W_conv.T @ W_conv) @ W_conv.T @ Xc.T
    proj_f = (W_conv @ Omega_conv).T + mu

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), dpi=300)
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for ax, letter in zip(axes.ravel(), letters):
        ax.set_xlim(-2.8, 2.8)
        ax.set_ylim(-2.8, 2.8)
        ax.set_aspect("equal")
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.text(-2.3, 2.0, letter, fontsize=12, va="center")
        # Green data points
        ax.scatter(X[:, 0], X[:, 1], color="#00ff00", edgecolors="#00aa00", s=55, zorder=5)

    # (a) True PCA eigenvectors
    ax_a = axes[0, 0]
    p1 = mu - u1 * s1
    p2 = mu + u1 * s1
    ax_a.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", lw=1.8)
    q1 = mu - u2 * s2
    q2 = mu + u2 * s2
    ax_a.plot([q1[0], q2[0]], [q1[1], q2[1]], "k--", lw=1.8)

    def draw_subspace(ax, w_vec, color="red", lw=2.5):
        w_u = w_vec.ravel() / np.linalg.norm(w_vec)
        p_start = mu - 3.8 * w_u
        p_end = mu + 3.8 * w_u
        ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color=color, lw=lw, zorder=3)

    def draw_projections(ax, projs):
        for i in range(N):
            ax.plot([X[i, 0], projs[i, 0]], [X[i, 1], projs[i, 1]], color="#0000ff", lw=1.4, zorder=4)
        ax.scatter(projs[:, 0], projs[:, 1], color="#00ffff", edgecolors="#0088aa", s=50, zorder=6)

    # (b) Initial W and Z
    draw_subspace(axes[0, 1], W_init)
    draw_projections(axes[0, 1], proj_b)

    # (c) After one M step
    draw_subspace(axes[0, 2], W_step1)
    draw_projections(axes[0, 2], proj_b)

    # (d) After successive E step
    draw_subspace(axes[1, 0], W_step1)
    draw_projections(axes[1, 0], proj_d)

    # (e) After second M step
    draw_subspace(axes[1, 1], W_step2)
    draw_projections(axes[1, 1], proj_d)

    # (f) Converged solution
    draw_subspace(axes[1, 2], W_conv)
    draw_projections(axes[1, 2], proj_f)

    plt.tight_layout()
    _save_figure(fig, "fig_16_10_em_pca", save_path)
    return fig
