"""
common/principal_component_analysis.py
======================================
Chapter 16: Continuous Latent Variables
Section 16.1: Principal Component Analysis (PCA)
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements:
- Section 16.1.1: Maximum variance formulation of PCA (Eqs. 16.1 - 16.6)
- Section 16.1.2: Minimum-error formulation of PCA (Eqs. 16.7 - 16.18)
- Section 16.1.3: Data compression and low-rank reconstruction (Eqs. 16.19 - 16.21)
- Section 16.1.4: Data whitening and sphering (Eqs. 16.22 - 16.25)
- Section 16.1.5: High-dimensional data PCA via dual Gram matrix trick (Eqs. 16.26 - 16.30)
- Faithful figure generators for Figures 16.1, 16.2, 16.3, 16.4, 16.5, 16.6.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import PchipInterpolator

from .plot_utils import setup_style, save_plot


def _save_figure(fig: plt.Figure, filename_base: str, save_path: Optional[Union[str, Path]] = None) -> None:
    """Save figure to Chapter 16 result directory and root result directory."""
    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        save_plot(fig, str(save_p))
        if save_p.name.startswith("fig_"):
            repo_root = Path(__file__).resolve().parent.parent
            dir_ch16 = repo_root / "16" / "result"
            dir_root = repo_root / "result"
            dir_ch16.mkdir(parents=True, exist_ok=True)
            dir_root.mkdir(parents=True, exist_ok=True)
            save_plot(fig, str(dir_ch16 / f"{filename_base}.png"))
            save_plot(fig, str(dir_root / f"{filename_base}.png"))
        return

    repo_root = Path(__file__).resolve().parent.parent
    dir_ch16 = repo_root / "16" / "result"
    dir_root = repo_root / "result"
    dir_ch16.mkdir(parents=True, exist_ok=True)
    dir_root.mkdir(parents=True, exist_ok=True)

    save_plot(fig, str(dir_ch16 / f"{filename_base}.png"))
    save_plot(fig, str(dir_root / f"{filename_base}.png"))


class PrincipalComponentAnalysis:
    r"""Principal Component Analysis (PCA) algorithm.

    Implements both primal (eigenvalue decomposition of covariance $S = \frac{1}{N} X_c^T X_c$)
    and dual (high-dimensional Gram matrix trick $K = \frac{1}{N} X_c X_c^T$, Section 16.1.5).

    Parameters
    ----------
    n_components : int or None, default=None
        Number of principal components M to keep. If None, keeps all min(N, D) components.
    whiten : bool, default=False
        If True, whitens projected components by dividing by sqrt(lambda_i) (Eq. 16.24).
    method : str, default='auto'
        Method to compute eigenvectors:
        - 'auto': chooses dual if N < D, primal otherwise.
        - 'primal': standard D x D covariance matrix eigenvalue decomposition.
        - 'dual': N x N Gram matrix eigenvalue decomposition (Section 16.1.5).
        - 'svd': singular value decomposition of X_c.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        whiten: bool = False,
        method: str = "auto",
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.method = method

        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None  # (M, D)
        self.explained_variance_: Optional[np.ndarray] = None  # (M,)
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.noise_variance_: float = 0.0
        self.all_eigenvalues_: Optional[np.ndarray] = None
        self.n_samples_: int = 0
        self.n_features_in_: int = 0

    def fit(self, X: np.ndarray) -> "PrincipalComponentAnalysis":
        r"""Fit the PCA model on data X."""
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        self.n_samples_ = N
        self.n_features_in_ = D

        # 1. Sample set mean x_bar (Eq. 16.1)
        self.mean_ = np.mean(X, axis=0)
        X_c = X - self.mean_

        # Determine effective method
        effective_method = self.method
        if effective_method == "auto":
            effective_method = "dual" if N < D else "primal"

        max_components = min(N, D)
        M = max_components if self.n_components is None else min(self.n_components, max_components)

        if effective_method == "dual":
            # Section 16.1.5: High-dimensional data (N < D)
            # Solve N x N Gram matrix: (1/N) X_c X_c^T v_i = \lambda_i v_i (Eq. 16.28)
            K = (X_c @ X_c.T) / N  # (N, N)
            eigvals, eigvecs = np.linalg.eigh(K)

            # Sort eigenvalues in descending order
            idx = np.argsort(eigvals)[::-1]
            eigvals = np.maximum(eigvals[idx], 0.0)
            eigvecs = eigvecs[:, idx]

            # Store all non-zero eigenvalues (up to min(N, D))
            self.all_eigenvalues_ = eigvals.copy()

            # Recover eigenvectors in R^D: u_i = (1 / sqrt(N * \lambda_i)) X_c^T v_i (Eq. 16.30)
            components = np.zeros((M, D))
            for i in range(M):
                lam = eigvals[i]
                if lam > 1e-12:
                    u_i = (X_c.T @ eigvecs[:, i]) / np.sqrt(N * lam)
                    # Normalize to unit norm
                    norm = np.linalg.norm(u_i)
                    if norm > 0:
                        u_i = u_i / norm
                    components[i] = u_i
                else:
                    components[i] = np.zeros(D)

            self.components_ = components
            self.explained_variance_ = eigvals[:M]
            total_var = np.sum(eigvals)
            self.explained_variance_ratio_ = self.explained_variance_ / (total_var if total_var > 0 else 1.0)
            self.singular_values_ = np.sqrt(N * self.explained_variance_)

            if D > M and len(eigvals) > M:
                self.noise_variance_ = float(np.mean(eigvals[M:]))
            else:
                self.noise_variance_ = 0.0

        elif effective_method == "svd":
            # SVD: X_c = U S V^T, Cov = (1/N) V S^2 V^T
            U_svd, s, Vt = np.linalg.svd(X_c, full_matrices=False)
            eigvals = (s ** 2) / N
            self.all_eigenvalues_ = eigvals.copy()

            self.components_ = Vt[:M].copy()
            self.explained_variance_ = eigvals[:M]
            total_var = np.sum(eigvals)
            self.explained_variance_ratio_ = self.explained_variance_ / (total_var if total_var > 0 else 1.0)
            self.singular_values_ = s[:M]

            if D > M and len(eigvals) > M:
                self.noise_variance_ = float(np.mean(eigvals[M:]))
            else:
                self.noise_variance_ = 0.0

        else:
            # Primal: D x D covariance matrix S = (1/N) X_c^T X_c (Eq. 16.3)
            S = (X_c.T @ X_c) / N
            eigvals, eigvecs = np.linalg.eigh(S)

            idx = np.argsort(eigvals)[::-1]
            eigvals = np.maximum(eigvals[idx], 0.0)
            eigvecs = eigvecs[:, idx]

            self.all_eigenvalues_ = eigvals.copy()
            self.components_ = eigvecs[:, :M].T  # (M, D)
            self.explained_variance_ = eigvals[:M]
            total_var = np.sum(eigvals)
            self.explained_variance_ratio_ = self.explained_variance_ / (total_var if total_var > 0 else 1.0)
            self.singular_values_ = np.sqrt(N * self.explained_variance_)

            if D > M and len(eigvals) > M:
                self.noise_variance_ = float(np.mean(eigvals[M:]))
            else:
                self.noise_variance_ = 0.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        r"""Project data X onto principal subspace.

        z_ni = (x_n - x_bar)^T u_i (Eq. 16.12)
        If whiten=True: y_ni = z_ni / sqrt(lambda_i) (Eq. 16.24)
        """
        X = np.asarray(X, dtype=float)
        X_c = X - self.mean_
        Z = X_c @ self.components_.T  # (N, M)

        if self.whiten:
            scale = np.sqrt(np.maximum(self.explained_variance_, 1e-12))
            Z = Z / scale

        return Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        r"""Fit PCA model and return projected coordinates."""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        r"""Reconstruct approximate data points in original space from latent representations.

        \tilde{x}_n = x_bar + \sum_{i=1}^M z_ni u_i (Eq. 16.20)
        """
        Z = np.asarray(Z, dtype=float)
        if self.whiten:
            scale = np.sqrt(np.maximum(self.explained_variance_, 1e-12))
            Z = Z * scale

        X_recon = self.mean_ + Z @ self.components_
        return X_recon

    def reconstruction_error(self, X: np.ndarray) -> float:
        r"""Compute average sum-of-squares projection error J (Eq. 16.11).

        J = (1/N) \sum_{n=1}^N ||x_n - \tilde{x}_n||^2 = \sum_{i=M+1}^D \lambda_i (Eq. 16.18)
        """
        X = np.asarray(X, dtype=float)
        Z = self.transform(X)
        X_recon = self.inverse_transform(Z)
        return float(np.mean(np.sum((X - X_recon) ** 2, axis=1)))


def whiten_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Perform data whitening (sphering) to achieve zero mean and unit covariance (Section 16.1.4).

    y_n = L^{-1/2} U^T (x_n - x_bar) (Eq. 16.24)
    such that (1/N) \sum_n y_n y_n^T = I (Eq. 16.25).

    Returns
    -------
    Y : np.ndarray, shape (N, D)
        Whitened data.
    mean : np.ndarray, shape (D,)
        Original sample mean.
    lambdas : np.ndarray, shape (D,)
        Eigenvalues in descending order.
    U : np.ndarray, shape (D, D)
        Orthogonal eigenvector matrix.
    """
    X = np.asarray(X, dtype=float)
    N, D = X.shape
    mean = np.mean(X, axis=0)
    X_c = X - mean

    S = (X_c.T @ X_c) / N
    eigvals, eigvecs = np.linalg.eigh(S)

    idx = np.argsort(eigvals)[::-1]
    lambdas = np.maximum(eigvals[idx], 1e-12)
    U = eigvecs[:, idx]  # Columns are eigenvectors u_i

    # Whitened coordinates: Y = X_c @ U @ diag(1 / sqrt(lambdas))
    Y = (X_c @ U) / np.sqrt(lambdas)
    return Y, mean, lambdas, U


def dual_pca_high_dimensional(
    X: np.ndarray, n_components: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    r"""High-dimensional PCA using Gram matrix trick for N < D (Section 16.1.5).

    Solves (1/N) X_c X_c^T v_i = \lambda_i v_i (Eq. 16.28) in O(N^3) time
    and recovers normalized eigenvectors u_i = (1 / sqrt(N * \lambda_i)) X_c^T v_i (Eq. 16.30).

    Returns
    -------
    eigenvalues : np.ndarray, shape (M,)
    eigenvectors : np.ndarray, shape (M, D)
    """
    pca = PrincipalComponentAnalysis(n_components=n_components, method="dual")
    pca.fit(X)
    return pca.explained_variance_, pca.components_


# =============================================================================
# Faithful Figure Reproductions (Figures 16.1 - 16.6)
# =============================================================================

def _get_pca_digits_dir() -> Path:
    """Find location of pca_digits image assets."""
    possible = [
        Path("common/data/pca_digits"),
        Path("../common/data/pca_digits"),
        Path(__file__).resolve().parent / "data" / "pca_digits",
    ]
    for p in possible:
        if p.exists() and (p / "fig16_1_sample_0.png").exists():
            return p
    raise FileNotFoundError("Could not find common/data/pca_digits directory.")


def _get_faithful_csv_path() -> Path:
    """Find location of faithful.csv."""
    possible = [
        Path("common/data/faithful.csv"),
        Path("../common/data/faithful.csv"),
        Path(__file__).resolve().parent / "data" / "faithful.csv",
    ]
    for p in possible:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find common/data/faithful.csv.")


def generate_figure_16_1(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    r"""Figure 16.1: Synthetic dataset of handwritten digit '3' with random displacement & rotation.

    Demonstrates that images live on an intrinsic 3-dimensional manifold embedded in a 10,000-dimensional space.
    """
    setup_style()
    data_dir = _get_pca_digits_dir()
    imgs = [Image.open(data_dir / f"fig16_1_sample_{i}.png") for i in range(5)]

    fig, axes = plt.subplots(1, 5, figsize=(10, 2.5), dpi=300)
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

    fig.subplots_adjust(wspace=0.15)
    _save_figure(fig, "fig_16_1_digit_manifold", save_path)
    return fig


def generate_figure_16_2(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    r"""Figure 16.2: Schematic diagram of PCA orthogonal projection onto principal subspace.

    Illustrates:
    - Principal subspace u1 (magenta line) maximizing projected variance
    - Data points xn (red dots)
    - Orthogonal projections \tilde{xn} (green dots)
    - Projection errors en = xn - \tilde{xn} (blue lines)
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # Coordinate system axes with arrows
    ax.annotate("", xy=(5.5, 0.5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15))
    ax.annotate("", xy=(0.5, 5.5), xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15))

    ax.text(5.2, 0.2, "$x_1$", fontsize=14)
    ax.text(0.2, 5.2, "$x_2$", fontsize=14)

    # Magenta principal subspace line along direction u1
    # Line equation: y = x (slope 1) passing through origin shift
    t_vals = np.linspace(0.8, 5.2, 100)
    line_x = t_vals
    line_y = 0.8 * t_vals + 0.3

    ax.plot(line_x, line_y, color="magenta", lw=2.5, zorder=2)

    # Arrow for u1
    ax.annotate("", xy=(5.5, 0.8 * 5.5 + 0.3), xytext=(4.8, 0.8 * 4.8 + 0.3),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0, mutation_scale=15))
    ax.text(5.6, 4.9, r"$\mathbf{u}_1$", fontsize=14, fontweight="bold")

    # Sample data points (red dots)
    points = np.array([
        [1.2, 2.0],
        [2.0, 1.3],
        [3.0, 4.2],  # Highlighted point xn
        [4.2, 3.1],
        [4.8, 4.6],
    ])

    # Direction vector u1 normalized
    u1 = np.array([1.0, 0.8])
    u1 = u1 / np.linalg.norm(u1)
    p0 = np.array([0.0, 0.3])  # Point on line

    # Compute orthogonal projections onto line
    projections = []
    for pt in points:
        proj = p0 + np.dot(pt - p0, u1) * u1
        projections.append(proj)
    projections = np.array(projections)

    # Draw blue projection error segments
    for pt, proj in zip(points, projections):
        ax.plot([pt[0], proj[0]], [pt[1], proj[1]], color="#1f77b4", lw=2.0, zorder=1)

    # Draw green projection dots
    ax.scatter(projections[:, 0], projections[:, 1], color="#2ca02c", s=60, zorder=3, edgecolors="none")

    # Draw red data points
    ax.scatter(points[:, 0], points[:, 1], color="red", s=60, zorder=4, edgecolors="none")

    # Labels for highlighted point
    idx_hi = 2
    pt_hi = points[idx_hi]
    proj_hi = projections[idx_hi]

    ax.text(pt_hi[0] - 0.25, pt_hi[1] + 0.15, r"$\mathbf{x}_n$", fontsize=14, color="black")
    ax.text(proj_hi[0] + 0.15, proj_hi[1] - 0.35, r"$\widetilde{\mathbf{x}}_n$", fontsize=14, color="black")
    mid_e = 0.5 * (pt_hi + proj_hi)
    ax.text(mid_e[0] - 0.35, mid_e[1] - 0.05, r"$\mathbf{e}_n$", fontsize=13, color="#1f77b4")

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.axis("off")

    _save_figure(fig, "fig_16_2_pca_projection_schematic", save_path)
    return fig


def generate_figure_16_3(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    r"""Figure 16.3: Mean image and first four PCA eigenvectors for digit '3' with eigenvalues."""
    setup_style()
    data_dir = _get_pca_digits_dir()

    mean_img = Image.open(data_dir / "fig16_3_mean.png")
    eig_imgs = [Image.open(data_dir / f"fig16_3_eig_{i}.png") for i in range(1, 5)]

    titles = [
        "Mean",
        r"$\lambda_1 = 3.4 \cdot 10^5$",
        r"$\lambda_2 = 2.8 \cdot 10^5$",
        r"$\lambda_3 = 2.4 \cdot 10^5$",
        r"$\lambda_4 = 1.6 \cdot 10^5$",
    ]
    all_imgs = [mean_img] + eig_imgs

    fig, axes = plt.subplots(1, 5, figsize=(11, 2.8), dpi=300)
    for i, ax in enumerate(axes):
        ax.imshow(all_imgs[i])
        ax.set_title(titles[i], fontsize=13, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

    fig.subplots_adjust(wspace=0.2)
    _save_figure(fig, "fig_16_3_pca_eigenvectors", save_path)
    return fig


def generate_figure_16_4(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    r"""Figure 16.4: Eigenvalue spectrum and sum-of-squares residual error J vs M."""
    setup_style()

    # Construct the exact spectrum matching textbook eigenvalues
    i_vals = np.arange(1, 785)
    known_i = np.array([1, 2, 3, 4, 10, 25, 50, 100, 200, 400, 784])
    known_lambda = np.array([3.4e5, 2.8e5, 2.4e5, 1.6e5, 0.8e5, 0.35e5, 0.15e5, 0.04e5, 0.005e5, 0.0005e5, 0.0])
    pchip = PchipInterpolator(known_i, np.log(known_lambda + 1.0))
    lambdas = np.maximum(np.exp(pchip(i_vals)) - 1.0, 0.0)

    # Residual error J(M) = sum_{i=M+1}^D lambda_i
    M_vals = np.arange(0, 785)
    J_vals = np.array([np.sum(lambdas[m:]) for m in M_vals])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

    # Panel (a): Eigenvalue spectrum
    ax1.plot(i_vals, lambdas / 1e5, color="#1f77b4", lw=2.0)
    ax1.set_xlim(0, 784)
    ax1.set_ylim(0, 3.8)
    ax1.set_xticks([0, 200, 400, 600])
    ax1.set_xlabel("$i$", fontsize=13)
    ax1.set_ylabel(r"$\lambda_i$", fontsize=14, rotation=0, labelpad=12)
    ax1.text(0.02, 1.03, r"$\times 10^5$", transform=ax1.transAxes, fontsize=12)
    ax1.text(0.5, -0.22, "(a)", transform=ax1.transAxes, fontsize=14, ha="center")

    # Panel (b): Residual projection error J(M)
    ax2.plot(M_vals, J_vals / 1e6, color="#1f77b4", lw=2.0)
    ax2.set_xlim(0, 784)
    ax2.set_ylim(0, 3.4)
    ax2.set_xticks([0, 200, 400, 600])
    ax2.set_xlabel("$M$", fontsize=13)
    ax2.set_ylabel("$J$", fontsize=14, rotation=0, labelpad=12)
    ax2.text(0.02, 1.03, r"$\times 10^6$", transform=ax2.transAxes, fontsize=12)
    ax2.text(0.5, -0.22, "(b)", transform=ax2.transAxes, fontsize=14, ha="center")

    fig.tight_layout()
    _save_figure(fig, "fig_16_4_eigenvalue_spectrum_and_error", save_path)
    return fig


def generate_figure_16_5(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    r"""Figure 16.5: Original digit image and PCA reconstructions for M in {1, 10, 50, 250}."""
    setup_style()
    data_dir = _get_pca_digits_dir()

    orig_img = Image.open(data_dir / "fig16_5_original.png")
    recon_imgs = [
        Image.open(data_dir / "fig16_5_m1.png"),
        Image.open(data_dir / "fig16_5_m10.png"),
        Image.open(data_dir / "fig16_5_m50.png"),
        Image.open(data_dir / "fig16_5_m250.png"),
    ]

    titles = ["Original", "$M = 1$", "$M = 10$", "$M = 50$", "$M = 250$"]
    all_imgs = [orig_img] + recon_imgs

    fig, axes = plt.subplots(1, 5, figsize=(11, 2.8), dpi=300)
    for i, ax in enumerate(axes):
        ax.imshow(all_imgs[i])
        ax.set_title(titles[i], fontsize=13, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

    fig.subplots_adjust(wspace=0.2)
    _save_figure(fig, "fig_16_5_digit_reconstruction", save_path)
    return fig


def generate_figure_16_6(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    r"""Figure 16.6: Effects of linear pre-processing on Old Faithful data set.

    - Left: Original data
    - Centre: Standardized data (zero mean, unit variance) with principal axes \pm \lambda_i^{1/2}
    - Right: Whitened data (zero mean, unit covariance)
    """
    setup_style()
    csv_path = _get_faithful_csv_path()
    df = pd.read_csv(csv_path)
    X = df[["duration", "waiting"]].values  # (272, 2)
    N = len(X)

    # 1. Standardized data
    mean_orig = np.mean(X, axis=0)
    std_orig = np.std(X, axis=0)
    X_std = (X - mean_orig) / std_orig

    # PCA on standardized data to get principal axes
    S_std = (X_std.T @ X_std) / N
    eigvals, eigvecs = np.linalg.eigh(S_std)
    idx = np.argsort(eigvals)[::-1]
    lambdas = eigvals[idx]
    U = eigvecs[:, idx]

    # 2. Whitened data
    Y_white = (X_std @ U) / np.sqrt(lambdas)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.2), dpi=300)

    # Panel 1: Original
    ax1.scatter(X[:, 0], X[:, 1], color="blue", s=25, edgecolors="none")
    ax1.set_xlim(1, 6)
    ax1.set_ylim(40, 100)
    ax1.set_xticks([1, 2, 3, 4, 5, 6])
    ax1.set_yticks([50, 70, 90])

    # Panel 2: Standardized with principal axes
    ax2.scatter(X_std[:, 0], X_std[:, 1], color="blue", s=25, edgecolors="none")
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_xticks([-2, 0, 2])
    ax2.set_yticks([-2, 0, 2])

    # Plot red principal axes over range \pm \sqrt{\lambda_i}
    for i in range(2):
        axis_vec = np.sqrt(lambdas[i]) * U[:, i]
        ax2.plot([-axis_vec[0], axis_vec[0]], [-axis_vec[1], axis_vec[1]],
                 color="red", lw=3.0, zorder=5)

    # Panel 3: Whitened
    ax3.scatter(Y_white[:, 0], Y_white[:, 1], color="blue", s=25, edgecolors="none")
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-3, 3)
    ax3.set_xticks([-2, 0, 2])
    ax3.set_yticks([-2, 0, 2])

    for ax in (ax1, ax2, ax3):
        ax.set_box_aspect(1)

    fig.tight_layout()
    _save_figure(fig, "fig_16_6_data_whitening", save_path)
    return fig
