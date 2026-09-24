"""
common/probabilistic_latent_variables.py
========================================
Core implementations for Chapter 16 Section 16.2: Probabilistic Latent Variables.
Christopher M. Bishop & Hugh Bishop (2024). Deep Learning: Foundations and Concepts.

Mathematical models:
- Probabilistic Principal Component Analysis (PPCA, Tipping & Bishop 1999, Roweis 1998)
  - Prior: p(z) = N(z | 0, I)                                  [Eq. 16.31]
  - Conditional: p(x | z) = N(x | W z + mu, sigma^2 I)         [Eq. 16.32]
  - Generative: x = W z + mu + eps, eps ~ N(0, sigma^2 I)      [Eq. 16.33]
  - Marginal: p(x) = N(x | mu, C), C = W W^T + sigma^2 I       [Eq. 16.35, 16.36]
  - Woodbury inverse: C^-1 = sigma^-2 I - sigma^-2 W M^-1 W^T  [Eq. 16.41]
    where M = W^T W + sigma^2 I                                [Eq. 16.42]
  - Posterior: p(z | x) = N(z | M^-1 W^T (x - mu), sigma^2 M^-1) [Eq. 16.43]
  - Closed-form MLE:
    W_ML = U_M (L_M - sigma^2 I)^(1/2) R                       [Eq. 16.46]
    sigma_ML^2 = 1/(D - M) sum_{i=M+1}^D lambda_i              [Eq. 16.47]
  - Degrees of freedom: D M + 1 - M(M - 1)/2                   [Eq. 16.52]
- Factor Analysis: p(x | z) = N(x | W z + mu, Psi), Psi diagonal [Eq. 16.53, 16.54]
- Independent Component Analysis (ICA, non-Gaussian p(z))       [Eq. 16.55, 16.56]
- Kalman Filter / Linear Dynamical System                       [Figure 16.9]
- Figure Generators: Figure 16.7, Figure 16.8, Figure 16.9.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, ConnectionPatch, Ellipse, FancyBboxPatch
from matplotlib.path import Path as MplPath

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


class ProbabilisticPCA:
    r"""Probabilistic Principal Component Analysis (PPCA).

    Formulated as a linear-Gaussian latent variable model:
      p(z) = N(z | 0, I)
      p(x | z) = N(x | W z + mu, sigma^2 I)
      p(x) = N(x | mu, C),  C = W W^T + sigma^2 I

    Closed-form maximum likelihood estimation (Tipping & Bishop, 1999):
      W_ML = U_M (L_M - sigma^2 I)^(1/2) R
      sigma_ML^2 = \frac{1}{D - M} \sum_{i=M+1}^D \lambda_i

    Parameters
    ----------
    n_components : int
        Dimension M of the latent space (principal subspace). Must satisfy 1 <= M < D.
    R : Optional[np.ndarray]
        Arbitrary M x M orthogonal rotation matrix in latent space. Default: identity I_M.
    """

    def __init__(self, n_components: int = 1, R: Optional[np.ndarray] = None):
        if n_components < 1:
            raise ValueError(f"n_components must be at least 1, got {n_components}")
        self.n_components = n_components
        self.R = R

        # Estimated parameters
        self.mu_: Optional[np.ndarray] = None  # (D,)
        self.W_: Optional[np.ndarray] = None  # (D, M)
        self.sigma2_: Optional[float] = None  # scalar
        self.C_: Optional[np.ndarray] = None  # (D, D)
        self.M_: Optional[np.ndarray] = None  # (M, M) = W^T W + sigma^2 I
        self.M_inv_: Optional[np.ndarray] = None  # (M, M)
        self.eigenvalues_: Optional[np.ndarray] = None  # (D,)
        self.eigenvectors_: Optional[np.ndarray] = None  # (D, D)
        self.n_features_in_: Optional[int] = None
        self.n_samples_seen_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "ProbabilisticPCA":
        """Fit the PPCA model to data X using exact closed-form MLE.

        Parameters
        ----------
        X : np.ndarray of shape (N, D)
            Observed data points.

        Returns
        -------
        self : ProbabilisticPCA
        """
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        self.n_samples_seen_ = N
        self.n_features_in_ = D
        M = self.n_components

        if M > D:
            raise ValueError(
                f"n_components M={M} cannot exceed data dimension D={D}."
            )

        # 1. Sample mean (Eq. 16.1)
        self.mu_ = np.mean(X, axis=0)
        Xc = X - self.mu_

        # 2. Sample covariance (Eq. 16.3)
        S = (Xc.T @ Xc) / N

        # 3. Eigendecomposition of S
        eigvals, eigvecs = np.linalg.eigh(S)
        # Sort in descending order
        sort_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_idx]
        eigvecs = eigvecs[:, sort_idx]

        self.eigenvalues_ = eigvals
        self.eigenvectors_ = eigvecs

        # 4. Maximum likelihood noise variance sigma_ML^2 (Eq. 16.47)
        # Average of discarded eigenvalues from M+1 to D
        if M < D:
            discarded_eigvals = eigvals[M:]
            self.sigma2_ = float(np.mean(discarded_eigvals))
            if self.sigma2_ < 1e-15:
                self.sigma2_ = 1e-15
        else:
            # M == D: No reduction in dimensionality (Eq. 16.48)
            self.sigma2_ = 1e-15

        # 5. Top M eigenvalues and eigenvectors
        U_M = eigvecs[:, :M]  # (D, M)
        L_M = eigvals[:M]  # (M,)

        # Scaling factor: sqrt(lambda_i - sigma^2)
        # Clamp to 0 if lambda_i < sigma^2 due to float rounding
        scale_diag = np.sqrt(np.maximum(L_M - self.sigma2_, 0.0))

        # 6. Orthogonal rotation R in latent space (default: I_M)
        if self.R is not None:
            R = np.asarray(self.R, dtype=np.float64)
            if R.shape != (M, M):
                raise ValueError(f"Rotation matrix R must have shape ({M}, {M}), got {R.shape}")
            # Verify orthogonality
            if not np.allclose(R @ R.T, np.eye(M), atol=1e-6):
                raise ValueError("Matrix R must be orthogonal (R @ R.T == I)")
        else:
            R = np.eye(M)

        # W_ML = U_M (L_M - sigma^2 I)^(1/2) R (Eq. 16.46)
        self.W_ = (U_M * scale_diag[None, :]) @ R  # (D, M)

        # 7. Precompute M = W^T W + sigma^2 I and M^-1 (Eq. 16.42)
        self.M_ = self.W_.T @ self.W_ + self.sigma2_ * np.eye(M)
        self.M_inv_ = np.linalg.inv(self.M_)

        # 8. Marginal covariance C = W W^T + sigma^2 I (Eq. 16.36)
        self.C_ = self.W_ @ self.W_.T + self.sigma2_ * np.eye(D)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        r"""Compute the posterior mean E[z | x] for each observation.

        From Eq. (16.43) and Eq. (16.49):
          E[z | x] = M^-1 W^T (x - mu)

        For row matrix X (N, D):
          E[Z | X] = (X - mu) W M^-1

        Parameters
        ----------
        X : np.ndarray of shape (N, D)
            Data to project into latent space.

        Returns
        -------
        Z_mean : np.ndarray of shape (N, M)
            Posterior latent mean coordinates.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.mu_
        # (N, D) @ (D, M) @ (M, M) -> (N, M)
        return (Xc @ self.W_) @ self.M_inv_

    def posterior_covariance(self) -> np.ndarray:
        r"""Compute the posterior covariance cov[z | x] = sigma^2 M^-1.

        From Eq. (16.43):
          cov[z | x] = sigma^2 M^-1
        Note that this is independent of the data point x.

        Returns
        -------
        cov_z : np.ndarray of shape (M, M)
        """
        self._check_is_fitted()
        return self.sigma2_ * self.M_inv_

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        r"""Project data to latent space and reconstruct back to data space.

        From Eq. (16.50):
          \hat{x} = W E[z | x] + mu

        Parameters
        ----------
        X : np.ndarray of shape (N, D)
            Observed points.

        Returns
        -------
        X_recon : np.ndarray of shape (N, D)
            Reconstructed points in data space.
        """
        self._check_is_fitted()
        Z_mean = self.transform(X)
        return Z_mean @ self.W_.T + self.mu_

    def sample(self, n_samples: int = 100, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        r"""Generate synthetic observations from the generative model.

        From Eq. (16.33):
          z ~ N(0, I_M)
          eps ~ N(0, sigma^2 I_D)
          x = W z + mu + eps

        Parameters
        ----------
        n_samples : int
            Number of points to generate.
        seed : Optional[int]
            Random seed for reproducibility.

        Returns
        -------
        X_samples : np.ndarray of shape (n_samples, D)
            Generated observations in data space.
        Z_samples : np.ndarray of shape (n_samples, M)
            Corresponding latent space samples.
        """
        self._check_is_fitted()
        rng = np.random.default_rng(seed)
        M = self.n_components
        D = self.n_features_in_

        Z = rng.standard_normal(size=(n_samples, M))
        eps = rng.normal(0.0, np.sqrt(self.sigma2_), size=(n_samples, D))
        X = Z @ self.W_.T + self.mu_ + eps
        return X, Z

    def log_likelihood(self, X: Optional[np.ndarray] = None) -> float:
        r"""Compute the total log likelihood ln p(X | mu, W, sigma^2).

        Evaluated efficiently using Woodbury inverse and Sylvester's determinant:
          ln p(X) = -ND/2 ln(2 pi) - N/2 ln|C| - 1/2 sum_{n=1}^N (x_n - mu)^T C^-1 (x_n - mu)
                  = -N/2 [ D ln(2 pi) + ln|C| + Tr(C^-1 S) ]   [Eq. 16.44, 16.45]

        Parameters
        ----------
        X : Optional[np.ndarray] of shape (N, D)
            Data matrix. If None, uses training log-likelihood.

        Returns
        -------
        total_ll : float
        """
        self._check_is_fitted()
        if X is None:
            N = self.n_samples_seen_
            D = self.n_features_in_
            M = self.n_components
            tr_C_inv_S = float(D)
            log_det_C = float(np.sum(np.log(self.eigenvalues_[:M])) + (D - M) * np.log(self.sigma2_))
            ll = -0.5 * N * (D * np.log(2.0 * np.pi) + log_det_C + tr_C_inv_S)
            return ll

        X = np.asarray(X, dtype=np.float64)
        scores = self.score_samples(X)
        return float(np.sum(scores))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        r"""Compute per-sample log likelihood ln p(x_n | mu, W, sigma^2).

        Using Woodbury inverse (Eq. 16.41):
          (x - mu)^T C^-1 (x - mu) = sigma^-2 ||x - mu||^2 - sigma^-2 (x - mu)^T W M^-1 W^T (x - mu)

        Parameters
        ----------
        X : np.ndarray of shape (N, D)

        Returns
        -------
        log_prob : np.ndarray of shape (N,)
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        M = self.n_components
        Xc = X - self.mu_

        # 1. Quadratic term via Woodbury identity
        sq_norm = np.sum(Xc**2, axis=1)  # (N,)
        proj = Xc @ self.W_
        quad_woodbury = np.sum((proj @ self.M_inv_) * proj, axis=1)  # (N,)
        mahalanobis = (sq_norm - quad_woodbury) / self.sigma2_

        # 2. Log determinant: ln|C| = (D - M) ln(sigma^2) + ln|M|
        sign, log_det_M = np.linalg.slogdet(self.M_)
        log_det_C = (D - M) * np.log(self.sigma2_) + log_det_M

        log_prob = -0.5 * (D * np.log(2.0 * np.pi) + log_det_C + mahalanobis)
        return log_prob

    def degrees_of_freedom(self) -> int:
        r"""Calculate the number of independent parameters in the covariance matrix C.

        From Eq. (16.52):
          N_params = D M + 1 - M(M - 1) / 2

        Returns
        -------
        dof : int
        """
        self._check_is_fitted()
        D = self.n_features_in_
        M = self.n_components
        return int(D * M + 1 - M * (M - 1) // 2)

    def _check_is_fitted(self) -> None:
        if self.W_ is None or self.mu_ is None:
            raise RuntimeError("This ProbabilisticPCA instance is not fitted yet. Call 'fit' first.")


class FactorAnalysisModel:
    r"""Factor Analysis (FA) with diagonal noise covariance.

    Linear-Gaussian model:
      p(z) = N(z | 0, I)
      p(x | z) = N(x | W z + mu, Psi)
      p(x) = N(x | mu, C),  C = W W^T + Psi

    where Psi = diag(psi_1^2, ..., psi_D^2) (uniquenesses).

    Parameters
    ----------
    n_components : int
        Number of latent factors M.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on log-likelihood.
    """

    def __init__(self, n_components: int = 1, max_iter: int = 100, tol: float = 1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

        self.mu_: Optional[np.ndarray] = None
        self.W_: Optional[np.ndarray] = None
        self.psi_: Optional[np.ndarray] = None  # (D,) diagonal elements
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray) -> "FactorAnalysisModel":
        """Fit Factor Analysis model via EM algorithm."""
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        M = self.n_components

        self.mu_ = np.mean(X, axis=0)
        Xc = X - self.mu_
        S = (Xc.T @ Xc) / N

        # Initialize W and psi with PPCA solution
        eigvals, eigvecs = np.linalg.eigh(S)
        sort_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_idx]
        eigvecs = eigvecs[:, sort_idx]

        sigma2_init = max(float(np.mean(eigvals[M:])), 1e-4)
        scale = np.sqrt(np.maximum(eigvals[:M] - sigma2_init, 1e-4))
        self.W_ = eigvecs[:, :M] * scale[None, :]
        self.psi_ = np.full(D, sigma2_init)

        prev_ll = -np.inf
        for it in range(self.max_iter):
            self.n_iter_ = it + 1
            # E-step
            psi_inv = 1.0 / np.maximum(self.psi_, 1e-8)  # (D,)
            W_psi_inv = self.W_ * psi_inv[:, None]  # (D, M)
            G = np.linalg.inv(np.eye(M) + self.W_.T @ W_psi_inv)  # (M, M)

            Ez = (Xc @ W_psi_inv) @ G  # (N, M)
            Ezz = N * G + Ez.T @ Ez  # (M, M)

            # M-step
            W_new = (Xc.T @ Ez) @ np.linalg.inv(Ezz)

            diag_reconstructed = np.diag(W_new @ (Ez.T @ Xc / N))
            psi_new = np.maximum(np.diag(S) - diag_reconstructed, 1e-5)

            self.W_ = W_new
            self.psi_ = psi_new

            # Check log-likelihood
            C = self.W_ @ self.W_.T + np.diag(self.psi_)
            sign, log_det_C = np.linalg.slogdet(C)
            if sign <= 0:
                continue
            C_inv = np.linalg.inv(C)
            ll = -0.5 * N * (D * np.log(2.0 * np.pi) + log_det_C + np.trace(C_inv @ S))
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute posterior factor scores E[z | x]."""
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.mu_
        psi_inv = 1.0 / np.maximum(self.psi_, 1e-8)
        W_psi_inv = self.W_ * psi_inv[:, None]
        G = np.linalg.inv(np.eye(self.n_components) + self.W_.T @ W_psi_inv)
        return (Xc @ W_psi_inv) @ G


class FastICA2D:
    r"""2-Dimensional Independent Component Analysis (FastICA).

    Recovers independent non-Gaussian sources from linear mixture:
      x = A s  <==>  s = W x
    where sources s_1, s_2 have non-Gaussian distributions (Eq. 16.55).

    Contrast function: g(u) = tanh(u), g'(u) = 1 - tanh^2(u).
    """

    def __init__(self, max_iter: int = 200, tol: float = 1e-5, random_state: Optional[int] = 42):
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.unmixing_matrix_: Optional[np.ndarray] = None  # (2, 2)
        self.whitening_matrix_: Optional[np.ndarray] = None  # (2, 2)
        self.mean_: Optional[np.ndarray] = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit ICA model and return recovered independent components."""
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        if D != 2:
            raise ValueError("FastICA2D only supports 2D observations.")

        # 1. Center
        self.mean_ = np.mean(X, axis=0)
        Xc = X - self.mean_

        # 2. Whiten (Sphering)
        cov = (Xc.T @ Xc) / N
        eigvals, eigvecs = np.linalg.eigh(cov)
        inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-10)))
        V = eigvecs @ inv_sqrt @ eigvecs.T  # (2, 2)
        self.whitening_matrix_ = V
        X_white = Xc @ V.T  # (N, 2)

        rng = np.random.default_rng(self.random_state)
        w1 = rng.standard_normal(2)
        w1 /= np.linalg.norm(w1)

        # FastICA fixed point iteration for 1st component
        for _ in range(self.max_iter):
            u = X_white @ w1
            g_u = np.tanh(u)
            gp_u = 1.0 - g_u**2

            w1_new = np.mean(X_white * g_u[:, None], axis=0) - np.mean(gp_u) * w1
            w1_new /= np.linalg.norm(w1_new)

            if abs(abs(np.dot(w1, w1_new)) - 1.0) < self.tol:
                w1 = w1_new
                break
            w1 = w1_new

        w2 = np.array([-w1[1], w1[0]])
        W_rot = np.vstack([w1, w2])  # (2, 2)

        self.unmixing_matrix_ = W_rot @ V
        S = Xc @ self.unmixing_matrix_.T
        return S


class KalmanFilter1D:
    r"""1-Dimensional Kalman Filter (Linear Dynamical System).

    State space model (Figure 16.9):
      Transition:  z_n = A z_{n-1} + w_n,  w_n ~ N(0, Gamma)
      Observation: x_n = C z_n + v_n,      v_n ~ N(0, Sigma)
    """

    def __init__(self, A: float = 1.0, C: float = 1.0, Gamma: float = 0.1, Sigma: float = 0.5):
        self.A = A
        self.C = C
        self.Gamma = Gamma
        self.Sigma = Sigma

    def filter(self, observations: np.ndarray, init_mean: float = 0.0, init_var: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman filtering forward pass over sequential observations."""
        N = len(observations)
        means = np.zeros(N)
        variances = np.zeros(N)

        z_hat = init_mean
        P = init_var

        for n in range(N):
            # 1. Predict
            z_pred = self.A * z_hat
            P_pred = self.A**2 * P + self.Gamma

            # 2. Update (Kalman gain)
            K = P_pred * self.C / (self.C**2 * P_pred + self.Sigma)
            x_n = observations[n]
            z_hat = z_pred + K * (x_n - self.C * z_pred)
            P = (1.0 - K * self.C) * P_pred

            means[n] = z_hat
            variances[n] = P

        return means, variances


# =============================================================================
# Faithful Figure Generators (Bishop & Bishop 2024, Chapter 16)
# =============================================================================

def draw_curly_brace(
    ax: plt.Axes,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    k: float = 0.18,
    text: Optional[str] = None,
    text_offset: float = 0.22,
    fontsize: int = 13,
) -> None:
    """Draw a smooth curly brace between p1 and p2 on the side indicated by normal."""
    p1_arr = np.asarray(p1, dtype=float)
    p2_arr = np.asarray(p2, dtype=float)
    v = p2_arr - p1_arr
    d = np.linalg.norm(v)
    if d < 1e-8:
        return
    u = v / d
    n = np.array([u[1], -u[0]])
    mid = (p1_arr + p2_arr) / 2.0
    tip = mid + k * 1.6 * n

    verts = [
        p1_arr,
        p1_arr + 0.1 * v + k * n,
        mid - 0.05 * v + k * n,
        tip,
        mid + 0.05 * v + k * n,
        p2_arr - 0.1 * v + k * n,
        p2_arr,
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
    ]
    path = MplPath(verts, codes)
    patch = patches.PathPatch(path, facecolor="none", edgecolor="black", lw=1.2)
    ax.add_patch(patch)

    if text:
        text_pos = tip + text_offset * n
        ax.text(text_pos[0], text_pos[1], text, fontsize=fontsize, ha="center", va="top")


def generate_figure_16_7(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    r"""Figure 16.7: Generative view of probabilistic PCA model.

    - Left: Latent space z with prior distribution p(z) and sample point \hat{z}.
    - Centre: Data space (x_1, x_2) with conditional distribution p(x | \hat{z})
              having mean w \hat{z} + mu and isotropic covariance sigma^2 I.
    - Right: Data space showing marginal distribution p(x) with elliptical contours.
    """
    setup_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.5, 3.8), dpi=300)

    # 1. Left: Prior p(z) = N(z | 0, 1)
    z_vals = np.linspace(-3.6, 3.6, 300)
    pz = np.exp(-0.5 * z_vals**2) / np.sqrt(2.0 * np.pi)

    ax1.plot(z_vals, pz, color="#b000b0", lw=2.2)

    ax1.annotate(
        "",
        xy=(3.8, 0),
        xytext=(-3.6, 0),
        arrowprops=dict(facecolor="black", edgecolor="black", arrowstyle="->", lw=1.3),
    )
    ax1.plot([0, 0], [-0.015, 0.015], color="black", lw=1.5)

    z_hat = 1.0
    ax1.plot(z_hat, 0, "ko", markersize=4.5)
    ax1.text(z_hat, -0.045, r"$\widehat{z}$", fontsize=13, ha="center", va="top")
    ax1.text(3.7, -0.045, r"$z$", fontsize=13, ha="center", va="top")
    ax1.text(-1.8, 0.22, r"$p(z)$", fontsize=13, ha="right", va="center")
    ax1.set_xlim(-3.8, 4.2)
    ax1.set_ylim(-0.08, 0.45)
    ax1.axis("off")

    # 2. Middle: Conditional distribution p(x | \hat{z})
    mu = np.array([2.0, 2.0])
    w = np.array([1.5, 1.0])
    w_norm = np.linalg.norm(w)

    ax2.annotate(
        "",
        xy=(5.2, 0),
        xytext=(0, 0),
        arrowprops=dict(facecolor="black", edgecolor="black", arrowstyle="->", lw=1.3),
    )
    ax2.annotate(
        "",
        xy=(0, 4.6),
        xytext=(0, 0),
        arrowprops=dict(facecolor="black", edgecolor="black", arrowstyle="->", lw=1.3),
    )
    ax2.text(5.0, -0.3, r"$x_1$", fontsize=13, ha="center")
    ax2.text(-0.3, 4.4, r"$x_2$", fontsize=13, ha="center")

    t = np.linspace(-1.3, 1.7, 100)
    line_pts = mu[:, None] + np.outer(w, t)
    ax2.plot(line_pts[0], line_pts[1], color="#0044cc", lw=2.5)

    ax2.plot(mu[0], mu[1], "ko", markersize=4.5)
    ax2.text(mu[0] - 0.15, mu[1] + 0.1, r"$\boldsymbol{\mu}$", fontsize=13, ha="right", va="bottom")

    w_u = w / w_norm
    n_up = np.array([-w_u[1], w_u[0]])
    w_arrow_start = mu + 1.1 * w + 0.3 * n_up
    w_arrow_end = w_arrow_start + 0.6 * w
    ax2.annotate(
        "",
        xy=w_arrow_end,
        xytext=w_arrow_start,
        arrowprops=dict(
            facecolor="black", edgecolor="black", arrowstyle="->", lw=1.5, mutation_scale=12
        ),
    )
    ax2.text(
        (w_arrow_start[0] + w_arrow_end[0]) / 2,
        (w_arrow_start[1] + w_arrow_end[1]) / 2 + 0.15,
        r"$\mathbf{w}$",
        fontsize=13,
        ha="center",
        va="bottom",
    )

    x_mean = mu + z_hat * w
    ax2.plot(x_mean[0], x_mean[1], "ko", markersize=4.5)

    c1 = Circle((x_mean[0], x_mean[1]), 0.32, color="red", fill=False, lw=1.8)
    c2 = Circle((x_mean[0], x_mean[1]), 0.64, color="red", fill=False, lw=1.8)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.text(x_mean[0] - 0.45, x_mean[1] + 0.72, r"$p(\mathbf{x} \mid \widehat{z})$", fontsize=13, ha="center")

    draw_curly_brace(
        ax2,
        (mu[0], mu[1]),
        (x_mean[0], x_mean[1]),
        k=0.18,
        text=r"$\widehat{z}|\mathbf{w}|$",
        text_offset=0.22,
        fontsize=13,
    )

    ax2.set_xlim(-0.3, 5.4)
    ax2.set_ylim(-0.4, 4.8)
    ax2.axis("off")

    # 3. Right: Marginal distribution p(x)
    ax3.annotate(
        "",
        xy=(5.2, 0),
        xytext=(0, 0),
        arrowprops=dict(facecolor="black", edgecolor="black", arrowstyle="->", lw=1.3),
    )
    ax3.annotate(
        "",
        xy=(0, 4.6),
        xytext=(0, 0),
        arrowprops=dict(facecolor="black", edgecolor="black", arrowstyle="->", lw=1.3),
    )
    ax3.text(5.0, -0.3, r"$x_1$", fontsize=13, ha="center")
    ax3.text(-0.3, 4.4, r"$x_2$", fontsize=13, ha="center")

    ax3.plot(mu[0], mu[1], "ko", markersize=4.5)
    ax3.text(mu[0] - 0.15, mu[1] + 0.15, r"$\boldsymbol{\mu}$", fontsize=13, ha="right", va="bottom")

    angle_deg = np.degrees(np.arctan2(w[1], w[0]))
    e1 = Ellipse(mu, width=1.4, height=0.62, angle=angle_deg, color="#00dd00", fill=False, lw=2.0)
    e2 = Ellipse(mu, width=2.8, height=1.24, angle=angle_deg, color="#00dd00", fill=False, lw=2.0)
    ax3.add_patch(e1)
    ax3.add_patch(e2)
    ax3.text(mu[0] + 0.75, mu[1] - 0.8, r"$p(\mathbf{x})$", fontsize=13, ha="left")

    ax3.set_xlim(-0.3, 5.4)
    ax3.set_ylim(-0.4, 4.8)
    ax3.axis("off")

    cp = ConnectionPatch(
        xyA=(z_hat, 0),
        xyB=(x_mean[0], x_mean[1]),
        coordsA="data",
        coordsB="data",
        axesA=ax1,
        axesB=ax2,
        arrowstyle="->",
        connectionstyle="arc3,rad=-0.32",
        linestyle="--",
        color="black",
        lw=1.4,
        mutation_scale=12,
    )
    fig.add_artist(cp)

    fig.subplots_adjust(wspace=0.15)
    _save_figure(fig, "fig_16_7_ppca_generative_model", save_path)
    return fig


def generate_figure_16_8(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    r"""Figure 16.8: Directed graphical model for Probabilistic PCA.

    - Latent variable z_n in plate N.
    - Observed variable x_n in plate N.
    - Parameters: W, mu, sigma^2.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(2.8, 3.8), dpi=300)

    plate = FancyBboxPatch(
        (0.2, 0.1),
        1.6,
        3.6,
        boxstyle="round,pad=0.1,rounding_size=0.3",
        fc="none",
        ec="#0000ff",
        lw=2.2,
    )
    ax.add_patch(plate)
    ax.text(1.5, 0.4, r"$N$", fontsize=16, fontstyle="italic")

    z_center = (1.0, 3.0)
    c_z = Circle(z_center, 0.42, fc="white", ec="red", lw=2.2)
    ax.add_patch(c_z)
    ax.text(z_center[0], z_center[1], r"$\mathbf{z}_n$", fontsize=15, ha="center", va="center")

    x_center = (1.0, 1.4)
    c_x = Circle(x_center, 0.42, fc="#b8b8ff", ec="red", lw=2.2)
    ax.add_patch(c_x)
    ax.text(x_center[0], x_center[1], r"$\mathbf{x}_n$", fontsize=15, ha="center", va="center")

    ax.annotate(
        "",
        xy=(1.0, 1.84),
        xytext=(1.0, 2.56),
        arrowprops=dict(facecolor="red", edgecolor="red", arrowstyle="->", lw=2.0, mutation_scale=15),
    )

    ax.annotate(
        "",
        xy=(0.56, 1.4),
        xytext=(-0.1, 1.4),
        arrowprops=dict(facecolor="red", edgecolor="red", arrowstyle="->", lw=2.2, mutation_scale=15),
    )
    ax.text(-0.2, 1.4, r"$\boldsymbol{\mu}$", fontsize=16, ha="right", va="center")

    ax.annotate(
        "",
        xy=(1.44, 1.4),
        xytext=(2.1, 1.4),
        arrowprops=dict(facecolor="red", edgecolor="red", arrowstyle="->", lw=2.2, mutation_scale=15),
    )
    ax.text(2.2, 1.4, r"$\mathbf{W}$", fontsize=16, ha="left", va="center")

    ax.annotate(
        "",
        xy=(0.7, 1.7),
        xytext=(-0.1, 2.6),
        arrowprops=dict(facecolor="red", edgecolor="red", arrowstyle="->", lw=2.2, mutation_scale=15),
    )
    ax.text(-0.2, 2.8, r"$\sigma^2$", fontsize=16, ha="right", va="center")

    ax.set_xlim(-0.8, 2.8)
    ax.set_ylim(0.0, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")

    _save_figure(fig, "fig_16_8_ppca_graphical_model", save_path)
    return fig


def generate_figure_16_9(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    r"""Figure 16.9: Probabilistic graphical model for Kalman filter (Linear Dynamical System).

    Chain of continuous latent states: z_1 -> z_2 -> ... -> z_N.
    Emissions to continuous observations: z_n -> x_n.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 2.8), dpi=300)

    nodes_z = [(1.0, 2.0, r"$\mathbf{z}_1$"), (3.0, 2.0, r"$\mathbf{z}_2$"), (6.5, 2.0, r"$\mathbf{z}_N$")]
    nodes_x = [(1.0, 0.7, r"$\mathbf{x}_1$"), (3.0, 0.7, r"$\mathbf{x}_2$"), (6.5, 0.7, r"$\mathbf{x}_N$")]

    r = 0.42
    for x, y, label in nodes_z:
        c = Circle((x, y), r, fc="white", ec="red", lw=2.2)
        ax.add_patch(c)
        ax.text(x, y, label, fontsize=15, ha="center", va="center")

    for x, y, label in nodes_x:
        c = Circle((x, y), r, fc="#b8b8ff", ec="red", lw=2.2)
        ax.add_patch(c)
        ax.text(x, y, label, fontsize=15, ha="center", va="center")

    for i in range(3):
        ax.annotate(
            "",
            xy=(nodes_x[i][0], nodes_x[i][1] + r + 0.02),
            xytext=(nodes_z[i][0], nodes_z[i][1] - r - 0.02),
            arrowprops=dict(facecolor="red", edgecolor="red", arrowstyle="->", lw=2.0, mutation_scale=15),
        )

    ax.annotate(
        "",
        xy=(3.0 - r - 0.02, 2.0),
        xytext=(1.0 + r + 0.02, 2.0),
        arrowprops=dict(facecolor="red", edgecolor="red", arrowstyle="->", lw=2.0, mutation_scale=15),
    )

    ax.annotate(
        "",
        xy=(4.3, 2.0),
        xytext=(3.0 + r + 0.02, 2.0),
        arrowprops=dict(facecolor="red", edgecolor="red", arrowstyle="->", lw=2.0, mutation_scale=15),
    )
    ax.text(4.75, 2.0, r"$\cdots$", fontsize=16, ha="center", va="center", color="red")
    ax.annotate(
        "",
        xy=(6.5 - r - 0.02, 2.0),
        xytext=(5.2, 2.0),
        arrowprops=dict(facecolor="red", edgecolor="red", arrowstyle="->", lw=2.0, mutation_scale=15),
    )

    ax.annotate(
        "",
        xy=(7.7, 2.0),
        xytext=(6.5 + r + 0.02, 2.0),
        arrowprops=dict(facecolor="red", edgecolor="red", arrowstyle="->", lw=2.0, mutation_scale=15),
    )

    ax.set_xlim(0.2, 8.2)
    ax.set_ylim(0.0, 2.8)
    ax.set_aspect("equal")
    ax.axis("off")

    _save_figure(fig, "fig_16_9_kalman_graphical_model", save_path)
    return fig
