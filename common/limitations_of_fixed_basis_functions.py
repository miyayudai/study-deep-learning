"""Chapter 6: Deep Neural Networks
Section 6.1: Limitations of Fixed Basis Functions

This module implements the mathematical formulations, algorithms, and figure reproduction
functions for Section 6.1 of:
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.

Contents:
- Mathematical Formulations:
  * Polynomial feature count and combinatorial growth O(D^M) (Eq 6.2, 6.3)
  * Hypersphere volume in D-dimensions and shell fraction 1 - (1 - eps)^D (Eq 6.4, 6.5)
  * High-dimensional Gaussian radial density p(r) and mode r_peak = sigma * sqrt(D - 1) (Eq 6.58)
  * Data-dependent radial basis functions (RBF) phi_n(x) = exp(-||x - x_n||^2 / s^2) (Eq 6.6)
- Algorithms & Models:
  * GridCellClassifier: Regular hypercube partitioning classifier illustrating the curse of dimensionality
  * RadialBasisModel: Kernel-based linear model with data-centered basis functions
- Figure Reproductions:
  * Figure 6.1: Iris dataset (sepal length vs sepal width) with test point
  * Figure 6.2: Grid-based classification on Iris data showing empty cells and majority voting
  * Figure 6.3: Curse of dimensionality (exponential growth of cells for D=1, 2, 3)
  * Figure 6.4: Hypersphere volume fraction in shell 1 - (1 - eps)^D vs eps for D=1, 2, 5, 20
  * Figure 6.5: Gaussian radial probability density p(r) vs r for D=1, 2, 20
  * Figure 6.6: Advantage of high-dimensional space (2D separable vs 1D overlapping projection)
  * Figure 6.7: Handwritten digit '5' varying across 3 degrees of freedom on a 3D manifold
  * Figure 6.8: Natural 64x64 images exhibiting spatial correlation vs Uniform random noise images
"""

import math
import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
from scipy import special
from sklearn.datasets import load_iris

from common.plot_utils import save_plot, setup_style


def _get_project_root() -> str:
    """Return the absolute path to the repository root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _save_figure(fig: plt.Figure, filename: str, filepath: Optional[str] = None, save_both: bool = True) -> Tuple[str, Optional[str]]:
    """Save figure to target filepath, and/or both 6/result and root result directory."""
    if filepath:
        save_plot(fig, filepath)
    root = _get_project_root()
    path_ch6 = os.path.join(root, "6", "result", filename)
    path_root = os.path.join(root, "result", filename)
    if save_both or not filepath:
        if not filepath or os.path.abspath(filepath) != os.path.abspath(path_ch6):
            save_plot(fig, path_ch6)
        if not filepath or os.path.abspath(filepath) != os.path.abspath(path_root):
            save_plot(fig, path_root)
        return path_ch6, path_root
    return filepath, None


# ==============================================================================
# 1. Mathematical Formulations & Utilities
# ==============================================================================

def polynomial_feature_dimension(D: int, M: int) -> int:
    """Calculate the number of independent coefficients for a full degree-M polynomial in D variables.

    Formula: binom(D + M, M) = (D + M)! / (D! * M!).
    Eq (6.2), (6.3).
    """
    return math.comb(D + M, M)


def hypersphere_volume_shell_fraction(
    D: Union[int, np.ndarray],
    epsilon: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Calculate fraction of hypersphere volume in radius [1 - epsilon, 1] in D dimensions.

    Formula:
        (V_D(1) - V_D(1 - epsilon)) / V_D(1) = 1 - (1 - epsilon)^D
    Eq (6.5).
    """
    eps = np.asarray(epsilon, dtype=np.float64)
    eps = np.clip(eps, 0.0, 1.0)
    return 1.0 - (1.0 - eps) ** D


def gaussian_radial_density(
    r: np.ndarray,
    D: int,
    sigma: float = 0.5
) -> np.ndarray:
    """Calculate the probability density p(r) with respect to radius r of a D-dimensional Gaussian.

    Formula:
        S_D = 2 * pi^(D/2) / Gamma(D/2)
        p(r) = S_D * r^(D-1) * (2*pi*sigma^2)^(-D/2) * exp(-r^2 / (2*sigma^2))
             = [2^(1 - D/2) / (sigma^D * Gamma(D/2))] * r^(D-1) * exp(-r^2 / (2*sigma^2))
    Eq (6.58).
    """
    r_arr = np.asarray(r, dtype=np.float64)
    # Special handling for D=1 to avoid 0^0 issues at r=0
    if D == 1:
        coeff = math.sqrt(2.0 / math.pi) / sigma
        return coeff * np.exp(-r_arr**2 / (2.0 * sigma**2))

    log_coeff = (1.0 - D / 2.0) * math.log(2.0) - D * math.log(sigma) - special.gammaln(D / 2.0)
    coeff = math.exp(log_coeff)
    return coeff * (r_arr ** (D - 1)) * np.exp(-r_arr**2 / (2.0 * sigma**2))


def gaussian_radial_mode(D: int, sigma: float = 0.5) -> float:
    """Theoretical mode (peak location) of the radial Gaussian density p(r).

    d ln p(r) / dr = (D - 1)/r - r / sigma^2 = 0
    => r_peak = sigma * sqrt(D - 1) for D >= 1 (0 for D=1).
    """
    if D <= 1:
        return 0.0
    return float(sigma * math.sqrt(D - 1))


def rbf_basis_function(x: np.ndarray, center: np.ndarray, s: float) -> np.ndarray:
    """Radial basis function: phi_n(x) = exp(-||x - center||^2 / s^2).

    Eq (6.6).
    """
    diff = x - center
    if diff.ndim == 1:
        dist_sq = np.sum(diff ** 2)
    else:
        dist_sq = np.sum(diff ** 2, axis=-1)
    return np.exp(-dist_sq / (s ** 2))


# ==============================================================================
# 2. Grid-Based Classifier (Curse of Dimensionality Demonstration)
# ==============================================================================

class GridCellClassifier:
    """A regular grid cell classifier dividing input space into hypercubes.

    Used to demonstrate the curse of dimensionality and the limitation of fixed
    problem-independent partitioning (Section 6.1.1, Figure 6.2).
    """

    def __init__(self, num_bins_per_dim: int = 4):
        self.num_bins_per_dim = num_bins_per_dim
        self.bin_edges: List[np.ndarray] = []
        self.cell_majority: Dict[Tuple[int, ...], int] = {}
        self.cell_counts: Dict[Tuple[int, ...], List[int]] = {}
        self.classes: np.ndarray = np.array([])
        self.num_empty_cells: int = 0
        self.total_cells: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, bounds: Optional[List[Tuple[float, float]]] = None):
        """Fit the grid classifier by recording point counts in each regular grid cell."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.classes = np.unique(y)
        K = len(self.classes)
        D = X.shape[1]

        self.total_cells = self.num_bins_per_dim ** D
        self.bin_edges = []

        for d in range(D):
            if bounds is not None and d < len(bounds):
                low, high = bounds[d]
            else:
                low, high = X[:, d].min() - 0.1, X[:, d].max() + 0.1
            edges = np.linspace(low, high, self.num_bins_per_dim + 1)
            self.bin_edges.append(edges)

        self.cell_counts = {}
        self.cell_majority = {}

        # Determine cell index for each training sample
        indices = np.zeros((X.shape[0], D), dtype=np.int64)
        for d in range(D):
            edges = self.bin_edges[d]
            # Digitize to [1, num_bins_per_dim]
            idx = np.digitize(X[:, d], edges) - 1
            idx = np.clip(idx, 0, self.num_bins_per_dim - 1)
            indices[:, d] = idx

        # Count class representatives per cell
        for i in range(X.shape[0]):
            key = tuple(indices[i])
            if key not in self.cell_counts:
                self.cell_counts[key] = [0] * K
            c_idx = np.where(self.classes == y[i])[0][0]
            self.cell_counts[key][c_idx] += 1

        # Determine majority
        for key, counts in self.cell_counts.items():
            maj_c_idx = int(np.argmax(counts))
            self.cell_majority[key] = int(self.classes[maj_c_idx])

        self.num_empty_cells = self.total_cells - len(self.cell_counts)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class for test points. Returns -1 if the test point falls into an empty cell."""
        X = np.asarray(X, dtype=np.float64)
        D = X.shape[1]
        preds = np.full(X.shape[0], -1, dtype=np.int64)

        for i in range(X.shape[0]):
            key = []
            valid = True
            for d in range(D):
                edges = self.bin_edges[d]
                idx = np.digitize(X[i, d], edges) - 1
                if idx < 0 or idx >= self.num_bins_per_dim:
                    valid = False
                    break
                key.append(idx)
            if valid and tuple(key) in self.cell_majority:
                preds[i] = self.cell_majority[tuple(key)]
        return preds


# ==============================================================================
# 3. Radial Basis Model (Data-Dependent Basis Functions)
# ==============================================================================

class RadialBasisModel:
    """Linear model with data-centered Radial Basis Functions (Section 6.1.4, Eq 6.6).

    phi_n(x) = exp(-||x - x_n||^2 / s^2)
    y(x) = sum_n w_n phi_n(x) + w_0
    """

    def __init__(self, s: float = 1.0, reg: float = 1e-3):
        self.s = s
        self.reg = reg
        self.centers: Optional[np.ndarray] = None
        self.w: Optional[np.ndarray] = None

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Transform inputs into RBF feature matrix with bias."""
        N = X.shape[0]
        M = self.centers.shape[0]
        Phi = np.zeros((N, M + 1), dtype=np.float64)
        Phi[:, 0] = 1.0  # Bias unit

        for m in range(M):
            diff = X - self.centers[m]
            dist_sq = np.sum(diff ** 2, axis=1)
            Phi[:, m + 1] = np.exp(-dist_sq / (self.s ** 2))
        return Phi

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit weights using regularized least squares."""
        self.centers = X.copy()
        Phi = self._transform(X)
        P = Phi.shape[1]
        # Ridge regression: w = (Phi^T Phi + reg * I)^(-1) Phi^T y
        reg_matrix = self.reg * np.eye(P)
        reg_matrix[0, 0] = 0.0  # Do not regularize bias
        self.w = np.linalg.solve(Phi.T @ Phi + reg_matrix, Phi.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous target values."""
        Phi = self._transform(X)
        return Phi @ self.w


# ==============================================================================
# 4. Figure Reproductions (Figures 6.1 to 6.8)
# ==============================================================================

def generate_figure_6_1(result_dirs: Optional[List[str]] = None) -> plt.Figure:
    """Reproduce Figure 6.1: Iris data (sepal length vs sepal width) with test point x."""
    setup_style()
    iris = load_iris()
    X_iris = iris.data[:, :2]  # sepal length, sepal width
    y_iris = iris.target

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    colors = ['red', 'green', 'blue']
    for cls_idx in range(3):
        mask = (y_iris == cls_idx)
        ax.scatter(X_iris[mask, 0], X_iris[mask, 1], c=colors[cls_idx], s=42, edgecolors='none', zorder=3)

    # Test point x at (6.15, 3.15) near cluster intersections
    test_point = np.array([6.15, 3.15])
    ax.scatter(test_point[0], test_point[1], marker='x', c='black', s=90, lw=2.2, zorder=5)

    ax.set_xlim(4.1, 8.1)
    ax.set_ylim(1.9, 4.5)
    ax.set_xlabel("sepal length", fontsize=12)
    ax.set_ylabel("sepal width", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)

    plt.tight_layout()

    filename = "fig_6_1_iris_data.png"
    if result_dirs:
        for r_dir in result_dirs:
            save_plot(fig, os.path.join(r_dir, filename))
    else:
        _save_figure(fig, filename)

    return fig


def generate_figure_6_2(result_dirs: Optional[List[str]] = None) -> plt.Figure:
    """Reproduce Figure 6.2: Grid-based classification on Iris data showing majority voting."""
    setup_style()
    iris = load_iris()
    X_iris = iris.data[:, :2]
    y_iris = iris.target

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    x_bins = np.linspace(4.1, 8.1, 5)
    y_bins = np.linspace(1.9, 4.5, 5)

    # Shading colors matching Bishop Figure 6.2
    bg_colors = {0: '#ffb3b3', 1: '#a1d99b', 2: '#9ecae1'}

    for i in range(4):
        for j in range(4):
            bx0, bx1 = x_bins[i], x_bins[i + 1]
            by0, by1 = y_bins[j], y_bins[j + 1]

            in_cell = (X_iris[:, 0] >= bx0) & (X_iris[:, 0] < bx1) & \
                      (X_iris[:, 1] >= by0) & (X_iris[:, 1] < by1)

            if np.any(in_cell):
                counts = [np.sum(y_iris[in_cell] == c) for c in range(3)]
                maj_c = int(np.argmax(counts))
                rect = patches.Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
                                         facecolor=bg_colors[maj_c], edgecolor='black', lw=0.9, alpha=0.9, zorder=1)
                ax.add_patch(rect)
            else:
                rect = patches.Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
                                         facecolor='white', edgecolor='black', lw=0.9, zorder=1)
                ax.add_patch(rect)

    colors = ['red', 'green', 'blue']
    for cls_idx in range(3):
        mask = (y_iris == cls_idx)
        ax.scatter(X_iris[mask, 0], X_iris[mask, 1], c=colors[cls_idx], s=42, edgecolors='none', zorder=3)

    # Test point in cell (col 3, row 3)
    test_point = np.array([6.15, 3.15])
    ax.scatter(test_point[0], test_point[1], marker='x', c='black', s=90, lw=2.2, zorder=5)

    ax.set_xlim(4.1, 8.1)
    ax.set_ylim(1.9, 4.5)
    ax.set_xlabel("sepal length", fontsize=12)
    ax.set_ylabel("sepal width", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)

    plt.tight_layout()

    filename = "fig_6_2_grid_classifier.png"
    if result_dirs:
        for r_dir in result_dirs:
            save_plot(fig, os.path.join(r_dir, filename))
    else:
        _save_figure(fig, filename)

    return fig


def generate_figure_6_3(result_dirs: Optional[List[str]] = None) -> plt.Figure:
    """Reproduce Figure 6.3: Curse of dimensionality (exponential growth of cells for D=1, 2, 3)."""
    setup_style()
    fig = plt.figure(figsize=(9, 4))

    # Subplot 1: D=1
    ax1 = fig.add_axes([0.04, 0.2, 0.22, 0.6])
    ax1.set_xlim(-0.2, 3.8)
    ax1.set_ylim(-1, 1)
    ax1.axis("off")
    ax1.annotate("", xy=(3.6, 0), xytext=(-0.1, 0),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=12))
    ax1.text(3.5, -0.35, "$x_1$", fontsize=13, ha="center")
    for x_t in [0.0, 1.0, 2.0, 3.0]:
        ax1.plot([x_t, x_t], [-0.18, 0.18], color="red", lw=2.0)
    ax1.text(1.5, -0.7, "$D = 1$", fontsize=14, ha="center")

    # Subplot 2: D=2
    ax2 = fig.add_axes([0.32, 0.2, 0.25, 0.6])
    ax2.set_xlim(-0.5, 4.0)
    ax2.set_ylim(-0.5, 4.0)
    ax2.axis("off")
    ax2.annotate("", xy=(3.8, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=12))
    ax2.annotate("", xy=(0, 3.8), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=12))
    ax2.text(3.7, -0.35, "$x_1$", fontsize=13, ha="center")
    ax2.text(-0.35, 3.7, "$x_2$", fontsize=13, va="center")
    for i in range(4):
        ax2.plot([0, 3], [i, i], color="red", lw=1.8)
        ax2.plot([i, i], [0, 3], color="red", lw=1.8)
    ax2.text(1.5, -0.7, "$D = 2$", fontsize=14, ha="center")

    # Subplot 3: D=3
    ax3 = fig.add_axes([0.62, 0.15, 0.35, 0.75])
    ax3.set_xlim(-2.2, 4.2)
    ax3.set_ylim(-1.8, 4.5)
    ax3.axis("off")

    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    e3 = np.array([-0.6, -0.45])

    def proj(p):
        return p[0] * e1 + p[1] * e2 + p[2] * e3

    cubes = [
        (0, 0, 0), (1, 0, 0), (2, 0, 0),
        (0, 1, 0), (1, 1, 0),
        (0, 2, 0),
        (0, 0, 1), (1, 0, 1),
        (0, 1, 1),
        (0, 0, 2)
    ]

    # Coordinate arrows through origin
    ax3.annotate("", xy=proj([3.8, 0, 0]), xytext=proj([3.0, 0, 0]),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=12))
    ax3.plot([proj([0, 0, 0])[0], proj([3.0, 0, 0])[0]],
             [proj([0, 0, 0])[1], proj([3.0, 0, 0])[1]],
             color="black", lw=1.5, ls=":")

    ax3.annotate("", xy=proj([0, 3.8, 0]), xytext=proj([0, 3.0, 0]),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=12))
    ax3.plot([proj([0, 0, 0])[0], proj([0, 3.0, 0])[0]],
             [proj([0, 0, 0])[1], proj([0, 3.0, 0])[1]],
             color="black", lw=1.5, ls=":")

    ax3.annotate("", xy=proj([0, 0, 3.8]), xytext=proj([0, 0, 3.0]),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=12))
    ax3.plot([proj([0, 0, 0])[0], proj([0, 0, 3.0])[0]],
             [proj([0, 0, 0])[1], proj([0, 0, 3.0])[1]],
             color="black", lw=1.5, ls=":")

    ax3.text(proj([3.8, 0, 0])[0] - 0.1, proj([3.8, 0, 0])[1] - 0.25, "$x_1$", fontsize=13)
    ax3.text(proj([0, 3.8, 0])[0] - 0.3, proj([0, 3.8, 0])[1] - 0.1, "$x_2$", fontsize=13)
    ax3.text(proj([0, 0, 3.8])[0] + 0.1, proj([0, 0, 3.8])[1] - 0.25, "$x_3$", fontsize=13)

    edges = set()
    for cx, cy, cz in cubes:
        for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            for ox in (0, 1) if dx == 0 else (0,):
                for oy in (0, 1) if dy == 0 else (0,):
                    for oz in (0, 1) if dz == 0 else (0,):
                        pA = (cx + ox, cy + oy, cz + oz)
                        pB = (cx + ox + dx, cy + oy + dy, cz + oz + dz)
                        edges.add(tuple(sorted([pA, pB])))

    for pA, pB in sorted(edges):
        pA_2d = proj(pA)
        pB_2d = proj(pB)
        is_internal = (pA[0] == 0 and pB[0] == 0 and pA[2] == 0 and pB[2] == 0) or \
                      (pA[1] == 0 and pB[1] == 0 and pA[2] == 0 and pB[2] == 0) or \
                      (pA[0] == 0 and pB[0] == 0 and pA[1] == 0 and pB[1] == 0) or \
                      (pA[2] == 0 and pB[2] == 0 and min(pA[0], pB[0]) < 2 and min(pA[1], pB[1]) < 2)

        ls = "--" if is_internal else "-"
        ax3.plot([pA_2d[0], pB_2d[0]], [pA_2d[1], pB_2d[1]], color="red", lw=1.5, ls=ls)

    ax3.text(0.5, -1.5, "$D = 3$", fontsize=14, ha="center")

    filename = "fig_6_3_curse_of_dimensionality.png"
    if result_dirs:
        for r_dir in result_dirs:
            save_plot(fig, os.path.join(r_dir, filename))
    else:
        _save_figure(fig, filename)

    return fig


def generate_figure_6_4(result_dirs: Optional[List[str]] = None) -> plt.Figure:
    """Reproduce Figure 6.4: Volume fraction of hypersphere shell 1 - (1 - eps)^D vs eps."""
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    eps = np.linspace(0, 1, 300)

    for D, pos in [(1, (0.7, 0.55)), (2, (0.55, 0.7)), (5, (0.35, 0.85)), (20, (0.15, 0.92))]:
        frac = hypersphere_volume_shell_fraction(D, eps)
        ax.plot(eps, frac, "b-", lw=2.0)
        ax.text(pos[0], pos[1], f"$D = {D}$", fontsize=11, color="blue", ha="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$\epsilon$", fontsize=13)
    ax.set_ylabel("volume fraction", fontsize=12)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    plt.tight_layout()

    filename = "fig_6_4_hypersphere_volume_fraction.png"
    if result_dirs:
        for r_dir in result_dirs:
            save_plot(fig, os.path.join(r_dir, filename))
    else:
        _save_figure(fig, filename)

    return fig


def generate_figure_6_5(result_dirs: Optional[List[str]] = None) -> plt.Figure:
    """Reproduce Figure 6.5: Gaussian radial probability density p(r) vs r for D=1, 2, 20."""
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    r = np.linspace(0, 4, 300)
    sigma = 0.5

    # D = 1
    p1 = gaussian_radial_density(r, D=1, sigma=sigma)
    ax.plot(r, p1, "r-", lw=2.2)
    ax.text(0.6, 1.5, "$D = 1$", color="red", fontsize=12)

    # D = 2
    p2 = gaussian_radial_density(r, D=2, sigma=sigma)
    ax.plot(r, p2, color="#00dd00", lw=2.2)
    ax.text(1.1, 1.25, "$D = 2$", color="#00dd00", fontsize=12)

    # D = 20
    p20 = gaussian_radial_density(r, D=20, sigma=sigma)
    ax.plot(r, p20, "b-", lw=2.2)
    ax.text(2.6, 1.05, "$D = 20$", color="blue", fontsize=12)

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2)
    ax.set_xlabel("$r$", fontsize=13)
    ax.set_ylabel("$p(r)$", fontsize=13)
    ax.set_xticks([0, 2, 4])
    ax.set_yticks([0, 1, 2])

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    plt.tight_layout()

    filename = "fig_6_5_gaussian_radial_density.png"
    if result_dirs:
        for r_dir in result_dirs:
            save_plot(fig, os.path.join(r_dir, filename))
    else:
        _save_figure(fig, filename)

    return fig


def generate_figure_6_6(result_dirs: Optional[List[str]] = None) -> plt.Figure:
    """Reproduce Figure 6.6: Advantage of higher-dimensional space: 2D separable vs 1D overlapping projection."""
    setup_style()
    np.random.seed(42)
    N1, N2 = 12, 12

    # Calibrate points to match Bishop Figure 6.6
    x1_green = np.array([-0.9, -0.7, -0.65, -0.3, 0.05, 0.1, 0.35, 0.45, 0.7, 0.85, 0.95, 1.05])
    x2_green = np.array([1.2, 1.3, 1.0, 1.1, 0.4, 0.1, -0.1, -0.12, -0.35, -0.5, 0.2, -0.2])

    x1_red = np.array([-1.0, -0.95, -0.75, -0.65, -0.4, -0.3, -0.2, 0.1, 0.2, 0.35, 0.75, 0.7])
    x2_red = np.array([0.1, -0.1, -0.4, -0.7, -0.6, -0.9, -0.75, -1.2, -1.7, -1.8, -1.6, -2.1])

    fig = plt.figure(figsize=(8.5, 4.2))

    # (a) 2D scatter plot
    ax_a = fig.add_axes([0.08, 0.2, 0.4, 0.75])
    ax_a.scatter(x1_green, x2_green, c="green", s=45, zorder=3)
    ax_a.scatter(x1_red, x2_red, c="red", s=45, zorder=3)
    ax_a.plot([-1.8, 1.8], [1.8, -1.8], "gray", linestyle="--", lw=1.5, zorder=2)
    ax_a.set_xlim(-2.0, 2.0)
    ax_a.set_ylim(-2.4, 1.8)
    ax_a.set_xlabel("$x_1$", fontsize=12)
    ax_a.set_ylabel("$x_2$", fontsize=12)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.text(0, -3.0, "(a)", fontsize=13, ha="center")

    for spine in ax_a.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    # (b) 1D projection onto horizontal axis
    ax_b = fig.add_axes([0.55, 0.2, 0.4, 0.75])
    ax_b.set_xlim(-2.0, 2.0)
    ax_b.set_ylim(-2.4, 1.8)
    ax_b.axis("off")

    ax_b.annotate("", xy=(1.8, -0.3), xytext=(-1.8, -0.3),
                  arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=12))
    ax_b.text(1.9, -0.3, "$x_1$", fontsize=12, va="center")

    ax_b.scatter(x1_green, np.full_like(x1_green, -0.3), c="green", s=45, zorder=3)
    ax_b.scatter(x1_red, np.full_like(x1_red, -0.3), c="red", s=45, zorder=3)
    ax_b.text(0, -3.0, "(b)", fontsize=13, ha="center")

    filename = "fig_6_6_curse_and_blessing_dimensionality.png"
    if result_dirs:
        for r_dir in result_dirs:
            save_plot(fig, os.path.join(r_dir, filename))
    else:
        _save_figure(fig, filename)

    return fig


def generate_figure_6_7(result_dirs: Optional[List[str]] = None) -> plt.Figure:
    """Reproduce Figure 6.7: Handwritten digit '5' varying on a 3-dimensional manifold."""
    setup_style()
    verts_base = np.array([
        [0.35, 0.52], [-0.3, 0.5],
        [-0.35, 0.05],
        [0.45, 0.15], [0.4, -0.5], [-0.35, -0.4]
    ])
    codes_base = [
        Path.MOVETO, Path.LINETO,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4
    ]

    fig, axes = plt.subplots(2, 4, figsize=(6, 3))
    shifts_rotations = [
        (-0.15, 0.1, -12),
        (0.05, -0.12, 6),
        (0.15, 0.12, -8),
        (0.18, -0.05, 15),
        (0.08, 0.05, -6),
        (-0.12, -0.15, 10),
        (0.12, -0.08, -14),
        (-0.02, 0.02, 2),
    ]

    for idx, ax in enumerate(axes.flat):
        dx, dy, angle_deg = shifts_rotations[idx]
        rad = math.radians(angle_deg)
        R = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])

        verts_trans = (verts_base @ R.T) + np.array([dx, dy])

        path = Path(verts_trans, codes_base)
        patch = patches.PathPatch(path, facecolor="none", edgecolor="#cc0000", lw=3.2, capstyle="round")
        ax.add_patch(patch)

        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

    plt.tight_layout()

    filename = "fig_6_7_handwritten_digit_manifold.png"
    if result_dirs:
        for r_dir in result_dirs:
            save_plot(fig, os.path.join(r_dir, filename))
    else:
        _save_figure(fig, filename)

    return fig


def generate_figure_6_8(result_dirs: Optional[List[str]] = None) -> plt.Figure:
    """Reproduce Figure 6.8: Natural 64x64 images vs Uniform random noise images."""
    setup_style()
    root = _get_project_root()
    npz_path = os.path.join(root, "data", "ch6_natural_images.npz")

    if os.path.exists(npz_path):
        data = np.load(npz_path)
        c1, c2, c3 = data["img1"], data["img2"], data["img3"]
    else:
        # Fallback synthetic textures
        X, Y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
        r_spot = np.sqrt(X ** 2 + Y ** 2)
        c1 = (np.clip(np.stack([0.85 - 0.4 * np.exp(-r_spot**2 / 0.2),
                                0.70 - 0.4 * np.exp(-r_spot**2 / 0.2),
                                0.60 - 0.35 * np.exp(-r_spot**2 / 0.2)], axis=-1), 0, 1) * 255).astype(np.uint8)
        c2 = (np.clip(np.stack([0.5 + 0.3 * np.cos(5*X), 0.5 + 0.3 * np.cos(5*X), 0.5 + 0.3 * np.cos(5*X)], axis=-1), 0, 1) * 255).astype(np.uint8)
        c3 = (np.clip(np.stack([0.7 + 0.2 * np.sin(3*Y), 0.5 + 0.2 * np.sin(3*Y), 0.3 + 0.1 * np.sin(3*Y)], axis=-1), 0, 1) * 255).astype(np.uint8)

    np.random.seed(42)
    noise1 = np.random.uniform(0, 1, (64, 64, 3))
    noise2 = np.random.uniform(0, 1, (64, 64, 3))
    noise3 = np.random.uniform(0, 1, (64, 64, 3))

    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.5))
    axes[0, 0].imshow(c1)
    axes[0, 1].imshow(c2)
    axes[0, 2].imshow(c3)

    axes[1, 0].imshow(noise1)
    axes[1, 1].imshow(noise2)
    axes[1, 2].imshow(noise3)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("black")
            s.set_linewidth(1.0)

    plt.tight_layout()

    filename = "fig_6_8_natural_vs_random_images.png"
    if result_dirs:
        for r_dir in result_dirs:
            save_plot(fig, os.path.join(r_dir, filename))
    else:
        _save_figure(fig, filename)

    return fig


def generate_all_section_6_1_figures(result_dirs: Optional[List[str]] = None) -> Dict[str, plt.Figure]:
    """Generate and save all 8 figures (6.1 through 6.8) for Section 6.1."""
    figs = {
        "fig_6_1": generate_figure_6_1(result_dirs),
        "fig_6_2": generate_figure_6_2(result_dirs),
        "fig_6_3": generate_figure_6_3(result_dirs),
        "fig_6_4": generate_figure_6_4(result_dirs),
        "fig_6_5": generate_figure_6_5(result_dirs),
        "fig_6_6": generate_figure_6_6(result_dirs),
        "fig_6_7": generate_figure_6_7(result_dirs),
        "fig_6_8": generate_figure_6_8(result_dirs),
    }
    return figs
