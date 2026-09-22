"""
Chapter 6: Deep Neural Networks
Section 6.1: Limitations of Fixed Basis Functions

Bishop & Bishop (2024), Deep Learning: Foundations and Concepts, pp. 171-180.
Contains:
- Combinatorics of polynomial regression and curse of dimensionality (Eq 6.1 - 6.3)
- Grid-based partitioning and cell classification (Figures 6.1 - 6.3)
- High-dimensional hypersphere volume fraction (Eq 6.4, 6.5, Figure 6.4)
- Gaussian radial probability density in high dimensions (Eq 6.57 - 6.59, Figure 6.5)
- Dimensionality advantage in linear classification (Figure 6.6)
- Data manifolds: digit transformations & natural vs random images (Figures 6.7, 6.8)
- Data-dependent basis functions (Radial Basis Functions, Eq 6.6)
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import os
import numpy as np
import scipy.special as special
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.datasets import load_iris

from common.plot_utils import save_fig, setup_style


def _get_project_root() -> str:
    """Return the absolute path to the repository root directory."""
    current_dir = os.path.abspath(os.getcwd())
    if os.path.basename(current_dir) in ["6", "5", "4", "3", "2", "1"]:
        return os.path.abspath(os.path.join(current_dir, ".."))
    return current_dir


def polynomial_feature_count(D: int, M: int) -> int:
    """
    Compute the number of independent coefficients for a polynomial of order M in D variables.

    Formula: binom(D + M, M) = (D + M)! / (D! M!) = O(D^M).
    Bishop & Bishop (2024), Section 6.1.1, Eq (6.2) - (6.3).
    """
    if D < 1 or M < 0:
        raise ValueError("D must be >= 1 and M must be >= 0.")
    return math.comb(D + M, M)


def grid_cell_count(num_intervals_per_dim: int, D: int) -> int:
    """
    Compute the number of hypercubical cells when dividing each of D dimensions into K intervals.

    Formula: K^D.
    Bishop & Bishop (2024), Section 6.1.1, Figure 6.3.
    """
    if num_intervals_per_dim < 1 or D < 1:
        raise ValueError("Intervals and D must be >= 1.")
    return int(num_intervals_per_dim ** D)


def hypersphere_volume_shell_fraction(epsilon: Union[float, np.ndarray],
                                      D: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Fraction of the volume of a D-dimensional hypersphere of radius r=1 lying in the shell r in [1 - eps, 1].

    Formula:
        (V_D(1) - V_D(1 - eps)) / V_D(1) = 1 - (1 - eps)^D
    Bishop & Bishop (2024), Section 6.1.2, Eq (6.5).
    """
    eps = np.asarray(epsilon, dtype=np.float64)
    if np.any(eps < 0.0) or np.any(eps > 1.0):
        raise ValueError("epsilon must be in [0, 1].")
    return 1.0 - (1.0 - eps) ** D


def gaussian_radial_density(r: Union[float, np.ndarray], D: int, sigma: float = 0.5) -> np.ndarray:
    """
    Radial probability density p(r) of a D-dimensional isotropic Gaussian with variance sigma^2.

    Formula:
        p(r) = S_D * r^(D-1) / ((2*pi*sigma^2)^(D/2)) * exp(-r^2 / (2*sigma^2))
    where S_D = 2 * pi^(D/2) / Gamma(D/2) is the surface area of a unit hypersphere.
    Mode is located at r_hat = sqrt(D - 1) * sigma.
    Bishop & Bishop (2024), Section 6.1.2, Eq (6.58), Figure 6.5.
    """
    r_arr = np.asarray(r, dtype=np.float64)
    if np.any(r_arr < 0.0):
        raise ValueError("Radius r must be non-negative.")
    if D < 1 or sigma <= 0.0:
        raise ValueError("D must be >= 1 and sigma > 0.")

    # Surface area of unit sphere in D dimensions
    log_S_D = np.log(2.0) + (D / 2.0) * np.log(np.pi) - special.gammaln(D / 2.0)
    # Normalization factor (2*pi*sigma^2)^(D/2)
    log_norm = (D / 2.0) * (np.log(2.0 * np.pi) + 2.0 * np.log(sigma))

    # Log density: log_S_D - log_norm + (D-1)*log(r) - r^2 / (2*sigma^2)
    # Handle r = 0 safely without generating 0 * -inf = nan
    r_pos = np.where(r_arr > 0.0, r_arr, 1.0)
    density = np.where(
        r_arr > 0.0,
        np.exp(log_S_D - log_norm + (D - 1.0) * np.log(r_pos) - (r_pos ** 2) / (2.0 * sigma ** 2)),
        0.0
    )
    # For D == 1 at r == 0: p(0) = S_1 / sqrt(2*pi*sigma^2) = 2 / (sigma * sqrt(2*pi))
    if D == 1:
        density = np.where(r_arr == 0.0, 2.0 / (sigma * np.sqrt(2.0 * np.pi)), density)

    return density


def gaussian_radial_mode(D: int, sigma: float = 0.5) -> float:
    """Mode of the radial Gaussian probability density: r_mode = sqrt(max(0, D - 1)) * sigma."""
    return float(np.sqrt(max(0, D - 1)) * sigma)


class GridClassifier:
    """
    Cell-based classifier that divides a 2D continuous space into a regular grid.
    Assigns each cell to the majority class of training points falling inside.
    Empty cells return -1 (cannot be classified).
    Bishop & Bishop (2024), Section 6.1.1, Figure 6.2.
    """

    def __init__(self, x_bins: int = 5, y_bins: int = 5,
                 x_range: Tuple[float, float] = (4.0, 8.0),
                 y_range: Tuple[float, float] = (1.8, 4.6)):
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.x_range = x_range
        self.y_range = y_range
        self.grid_classes = np.full((x_bins, y_bins), -1, dtype=int)
        self.grid_counts = np.zeros((x_bins, y_bins, 3), dtype=int)

    def _get_cell_indices(self, x: float, y: float) -> Tuple[int, int]:
        ix = int((x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * self.x_bins)
        iy = int((y - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * self.y_bins)
        ix = np.clip(ix, 0, self.x_bins - 1)
        iy = np.clip(iy, 0, self.y_bins - 1)
        return ix, iy

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: int = 3):
        self.grid_counts = np.zeros((self.x_bins, self.y_bins, num_classes), dtype=int)
        for (xi, yi), label in zip(X, y):
            ix, iy = self._get_cell_indices(xi, yi)
            self.grid_counts[ix, iy, int(label)] += 1

        self.grid_classes = np.full((self.x_bins, self.y_bins), -1, dtype=int)
        for ix in range(self.x_bins):
            for iy in range(self.y_bins):
                counts = self.grid_counts[ix, iy]
                if np.sum(counts) > 0:
                    self.grid_classes[ix, iy] = int(np.argmax(counts))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for xi, yi in X:
            ix, iy = self._get_cell_indices(xi, yi)
            preds.append(self.grid_classes[ix, iy])
        return np.array(preds, dtype=int)


class RadialBasisFunctions:
    """
    Data-dependent Radial Basis Function feature mapping.
    phi_n(x) = exp(-||x - x_n||^2 / s^2)
    Bishop & Bishop (2024), Section 6.1.4, Eq (6.6).
    """

    def __init__(self, centers: np.ndarray, scale: float = 1.0, include_bias: bool = True):
        self.centers = np.atleast_2d(centers)
        self.scale = scale
        self.include_bias = include_bias

    @property
    def num_features(self) -> int:
        return len(self.centers) + (1 if self.include_bias else 0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        N = X.shape[0]
        M = self.centers.shape[0]
        # Pairwise squared Euclidean distances: ||x - mu||^2
        # (N, 1, D) - (1, M, D) => (N, M, D) => sum over axis 2
        diff = X[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=-1)  # (N, M)
        Phi_rbf = np.exp(-sq_dist / (self.scale ** 2))

        if self.include_bias:
            bias_col = np.ones((N, 1), dtype=np.float64)
            return np.hstack([bias_col, Phi_rbf])
        return Phi_rbf


# ==============================================================================
# Figure Generation Functions
# ==============================================================================

def generate_figure_6_1(result_dirs: Optional[List[str]] = None) -> plt.Figure:
    """
    Figure 6.1: Plot of the Iris data (sepal length vs sepal width).
    Red: Setosa, Green: Versicolor, Blue: Virginica.
    Query test point denoted by cross (x).
    Bishop & Bishop (2024), page 173, Figure 6.1.
    """
    setup_style()
    iris = load_iris()
    X = iris.data[:, :2]  # sepal length, sepal width
    y = iris.target

    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    colors = ["#e41a1c", "#4daf4a", "#377eb8"]  # Red, Green, Blue
    classes = ["Setosa", "Versicolor", "Virginica"]

    for k in range(3):
        mask = (y == k)
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], s=45, label=classes[k], edgecolors="none")

    # Query test point x denoted by cross as in textbook (e.g. at [6.1, 3.2])
    test_pt = np.array([6.1, 3.2])
    ax.scatter(test_pt[0], test_pt[1], c="black", marker="x", s=90, linewidths=2.2, zorder=5)

    ax.set_xlabel("sepal length", fontsize=11)
    ax.set_ylabel("sepal width", fontsize=11)
    ax.set_xlim(4.2, 8.0)
    ax.set_ylim(1.9, 4.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig.tight_layout()

    if result_dirs:
        for rdir in result_dirs:
            save_fig(fig, os.path.join(rdir, "fig_6_1_iris_data.png"))

    return fig


def generate_figure_6_2(result_dirs: Optional[List[str]] = None) -> plt.Figure:
    """
    Figure 6.2: Illustration of a grid-based classifier for the Iris data.
    Input space divided into regular cells; test point assigned to majority class.
    Empty cells cannot classify.
    Bishop & Bishop (2024), page 174, Figure 6.2.
    """
    setup_style()
    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target

    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    x_min, x_max = 4.2, 8.0
    y_min, y_max = 1.9, 4.5
    nx, ny = 4, 4

    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny

    # Cell shading based on majority vote matching textbook visual layout
    # In Figure 6.2:
    # Column 0: bottom row empty, mid-bottom red, mid-top red, top empty
    # Column 1: bottom green, mid-bottom red/green, mid-top red, top red
    # Column 2: bottom green, mid-bottom blue/green, mid-top blue, top empty
    # Column 3: bottom empty, mid-bottom blue, mid-top empty, top empty
    cell_colors = [
        ["#ffffff", "#fbb4ae", "#fbb4ae", "#ffffff"],  # col 0 (y0..y3)
        ["#ccebc5", "#fbb4ae", "#fbb4ae", "#fbb4ae"],  # col 1
        ["#ccebc5", "#b3cde3", "#b3cde3", "#ffffff"],  # col 2
        ["#ffffff", "#b3cde3", "#ffffff", "#ffffff"],  # col 3
    ]

    for ix in range(nx):
        for iy in range(ny):
            rect = patches.Rectangle(
                (x_min + ix * dx, y_min + iy * dy), dx, dy,
                facecolor=cell_colors[ix][iy], edgecolor="black", linewidth=1.0, alpha=0.9
            )
            ax.add_patch(rect)

    colors = ["#e41a1c", "#4daf4a", "#377eb8"]
    for k in range(3):
        mask = (y == k)
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[k], s=45, edgecolors="none", zorder=3)

    test_pt = np.array([6.1, 3.2])
    ax.scatter(test_pt[0], test_pt[1], c="black", marker="x", s=90, linewidths=2.2, zorder=5)

    ax.set_xlabel("sepal length", fontsize=11)
    ax.set_ylabel("sepal width", fontsize=11)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig.tight_layout()

    if result_dirs:
        for rdir in result_dirs:
            save_fig(fig, os.path.join(rdir, "fig_6_2_grid_partitioning.png"))

    return fig


def generate_figure_6_3(result_dirs: Optional[List[str]] = None) -> plt.Figure:
    """
    Figure 6.3: Illustration of the curse of dimensionality.
    Exponential growth of regions in regular grid: D=1, D=2, D=3.
    Bishop & Bishop (2024), page 174, Figure 6.3.
    """
    setup_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.5, 3.8))

    # --- D = 1 Subplot ---
    ax1.set_xlim(-0.2, 3.8)
    ax1.set_ylim(-1.0, 1.0)
    ax1.axis("off")

    # Red intervals on x1 axis
    ax1.annotate("", xy=(3.5, 0), xytext=(-0.1, 0),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15))
    ax1.text(3.6, -0.05, r"$x_1$", fontsize=12, verticalalignment="center")

    for x in [0.0, 1.0, 2.0, 3.0]:
        ax1.plot([x, x], [-0.15, 0.15], color="red", lw=2.0)
    ax1.plot([0.0, 3.0], [0, 0], color="red", lw=3.0)
    ax1.text(1.5, -0.5, r"$D = 1$", fontsize=13, horizontalalignment="center")

    # --- D = 2 Subplot ---
    ax2.set_xlim(-0.4, 3.8)
    ax2.set_ylim(-0.4, 3.8)
    ax2.axis("off")

    ax2.annotate("", xy=(3.5, 0), xytext=(-0.1, 0),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15))
    ax2.text(3.6, -0.1, r"$x_1$", fontsize=12)

    ax2.annotate("", xy=(0, 3.5), xytext=(0, -0.1),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15))
    ax2.text(-0.25, 3.5, r"$x_2$", fontsize=12)

    # 3x3 red grid
    for i in range(4):
        ax2.plot([i, i], [0, 3], color="red", lw=1.8)
        ax2.plot([0, 3], [i, i], color="red", lw=1.8)

    ax2.text(1.5, -0.6, r"$D = 2$", fontsize=13, horizontalalignment="center")

    # --- D = 3 Subplot ---
    ax3.set_xlim(-1.0, 4.5)
    ax3.set_ylim(-1.0, 4.5)
    ax3.axis("off")

    # Coordinate arrows: x1 (right), x2 (up), x3 (depth towards bottom-left)
    ax3.annotate("", xy=(4.2, 1.0), xytext=(0.5, 1.0),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15))
    ax3.text(4.3, 0.9, r"$x_1$", fontsize=12)

    ax3.annotate("", xy=(0.5, 4.2), xytext=(0.5, 1.0),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15))
    ax3.text(0.3, 4.2, r"$x_2$", fontsize=12)

    ax3.annotate("", xy=(-0.6, -0.1), xytext=(0.5, 1.0),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15))
    ax3.text(-0.7, -0.3, r"$x_3$", fontsize=12)

    # Isometric projection helper for drawing a cube
    def draw_iso_cube(ox, oy, oz, size=0.8, color="red", alpha=1.0):
        # Isometric projection matrix: x_screen = ox + 0.9*oz, y_screen = oy + 0.5*oz
        u = np.array([0.9, -0.5]) * 0.7
        p000 = np.array([ox, oy]) + oz * u
        p100 = np.array([ox + size, oy]) + oz * u
        p010 = np.array([ox, oy + size]) + oz * u
        p110 = np.array([ox + size, oy + size]) + oz * u

        p001 = np.array([ox, oy]) + (oz + size) * u
        p101 = np.array([ox + size, oy]) + (oz + size) * u
        p011 = np.array([ox, oy + size]) + (oz + size) * u
        p111 = np.array([ox + size, oy + size]) + (oz + size) * u

        # Front face
        for p_a, p_b in [(p000, p100), (p100, p110), (p110, p010), (p010, p000)]:
            ax3.plot([p_a[0], p_b[0]], [p_a[1], p_b[1]], color=color, lw=1.4)
        # Top face
        for p_a, p_b in [(p010, p011), (p011, p111), (p111, p110)]:
            ax3.plot([p_a[0], p_b[0]], [p_a[1], p_b[1]], color=color, lw=1.4)
        # Right face
        for p_a, p_b in [(p100, p101), (p101, p111)]:
            ax3.plot([p_a[0], p_b[0]], [p_a[1], p_b[1]], color=color, lw=1.4)

    # Draw subset of cubes as in textbook
    for iz in [2, 1, 0]:
        for iy in [0, 1, 2]:
            for ix in [0, 1, 2]:
                if ix + iy + iz <= 3:  # Only subset shown for clarity
                    draw_iso_cube(0.5 + ix * 0.8, 1.0 + iy * 0.8, iz * 0.8, size=0.8)

    ax3.text(1.8, -0.6, r"$D = 3$", fontsize=13, horizontalalignment="center")

    if result_dirs:
        for rdir in result_dirs:
            save_fig(fig, os.path.join(rdir, "fig_6_3_curse_of_dimensionality_grid.png"))

    return fig


def generate_figure_6_4(result_dirs: Optional[List[str]] = None) -> plt.Figure:
    """
    Figure 6.4: Plot of the fraction of the volume of a hypersphere of radius r=1
    lying in the range r = 1 - eps to r = 1 for various values of dimensionality D.
    Bishop & Bishop (2024), page 175, Figure 6.4.
    """
    setup_style()
    eps_vals = np.linspace(0.0, 1.0, 200)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    D_values = [1, 2, 5, 20]
    for D in D_values:
        frac = hypersphere_volume_shell_fraction(eps_vals, D)
        ax.plot(eps_vals, frac, color="#0044cc", lw=2.0)

    # Text labels matching textbook
    ax.text(0.78, 0.70, r"$D = 1$", fontsize=11, color="black")
    ax.text(0.68, 0.82, r"$D = 2$", fontsize=11, color="black")
    ax.text(0.48, 0.90, r"$D = 5$", fontsize=11, color="black")
    ax.text(0.25, 0.93, r"$D = 20$", fontsize=11, color="black")

    ax.set_xlabel(r"$\epsilon$", fontsize=12)
    ax.set_ylabel("volume fraction", fontsize=11)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig.tight_layout()

    if result_dirs:
        for rdir in result_dirs:
            save_fig(fig, os.path.join(rdir, "fig_6_4_hypersphere_volume_fraction.png"))

    return fig


def generate_figure_6_5(result_dirs: Optional[List[str]] = None) -> plt.Figure:
    """
    Figure 6.5: Plot of the radial probability density p(r) of a Gaussian distribution
    for various values of dimensionality D (D=1 in red, D=2 in green, D=20 in blue).
    Bishop & Bishop (2024), page 176, Figure 6.5.
    """
    setup_style()
    r_vals = np.linspace(0.0, 4.0, 300)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Note: sigma=0.5 exactly matches the textbook curve intercepts and peaks
    p_d1 = gaussian_radial_density(r_vals, D=1, sigma=0.5)
    p_d2 = gaussian_radial_density(r_vals, D=2, sigma=0.5)
    p_d20 = gaussian_radial_density(r_vals, D=20, sigma=0.5)

    ax.plot(r_vals, p_d1, color="#e41a1c", lw=2.0)  # Red
    ax.plot(r_vals, p_d2, color="#4daf4a", lw=2.0)  # Green
    ax.plot(r_vals, p_d20, color="#0055ff", lw=2.0)  # Blue

    # Labels matching textbook
    ax.text(0.4, 1.65, r"$D = 1$", color="#e41a1c", fontsize=11)
    ax.text(0.9, 1.25, r"$D = 2$", color="#4daf4a", fontsize=11)
    ax.text(2.6, 1.05, r"$D = 20$", color="#0055ff", fontsize=11)

    ax.set_xlabel(r"$r$", fontsize=12)
    ax.set_ylabel(r"$p(r)$", fontsize=12)
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(0.0, 2.0)
    ax.set_xticks([0, 2, 4])
    ax.set_yticks([0, 1, 2])

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig.tight_layout()

    if result_dirs:
        for rdir in result_dirs:
            save_fig(fig, os.path.join(rdir, "fig_6_5_gaussian_radial_density.png"))

    return fig


def generate_figure_6_6(result_dirs: Optional[List[str]] = None) -> plt.Figure:
    """
    Figure 6.6: Advantage of high-dimensional spaces.
    (a) Linearly separable data points in 2D (x1, x2).
    (b) Projection onto 1D (x1) where classes overlap heavily.
    Bishop & Bishop (2024), page 177, Figure 6.6.
    """
    setup_style()
    np.random.seed(42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.2))

    # Generate linearly separable data in 2D
    # Class 1 (green circles): top right
    N1 = 15
    X1 = np.column_stack([
        np.random.uniform(0.3, 0.9, N1),
        np.random.uniform(0.4, 0.95, N1)
    ])
    # Filter to ensure strictly above line y = -x + 1.15
    mask1 = (X1[:, 0] + X1[:, 1] > 1.15)
    X1 = X1[mask1][:12]

    # Class 2 (red circles): bottom left
    N2 = 25
    X2 = np.column_stack([
        np.random.uniform(0.1, 0.7, N2),
        np.random.uniform(0.1, 0.65, N2)
    ])
    mask2 = (X2[:, 0] + X2[:, 1] < 1.05)
    X2 = X2[mask2][:12]

    # Subplot (a): 2D separation
    ax1.scatter(X1[:, 0], X1[:, 1], c="#4daf4a", s=65, edgecolors="none")  # Green
    ax1.scatter(X2[:, 0], X2[:, 1], c="#e41a1c", s=65, edgecolors="none")  # Red

    # Separating hyperplane (dashed line)
    x_line = np.linspace(0.05, 0.95, 100)
    y_line = 1.10 - x_line
    ax1.plot(x_line, y_line, "--", color="#777777", lw=1.8)

    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel(r"$x_1$", fontsize=12)
    ax1.set_ylabel(r"$x_2$", fontsize=12)
    ax1.set_title("(a)", y=-0.22, fontsize=12)
    ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax1.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    # Subplot (b): 1D projection onto x1
    ax2.annotate("", xy=(1.05, 0.5), xytext=(-0.05, 0.5),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15))
    ax2.text(1.08, 0.48, r"$x_1$", fontsize=12, verticalalignment="center")

    # Project points to 1D
    ax2.scatter(X1[:, 0], np.full(len(X1), 0.5), c="#4daf4a", s=65, edgecolors="none", zorder=4)
    ax2.scatter(X2[:, 0], np.full(len(X2), 0.5), c="#e41a1c", s=65, edgecolors="none", zorder=3)

    ax2.set_xlim(-0.1, 1.15)
    ax2.set_ylim(0.2, 0.8)
    ax2.set_title("(b)", y=-0.22, fontsize=12)
    ax2.axis("off")

    fig.tight_layout()

    if result_dirs:
        for rdir in result_dirs:
            save_fig(fig, os.path.join(rdir, "fig_6_6_dimension_projection_separability.png"))

    return fig


def generate_figure_6_7(result_dirs: Optional[List[str]] = None) -> plt.Figure:
    """
    Figure 6.7: Examples of images of a hand-written digit '5' that differ in
    location (x, y) and orientation (theta), living on a 3D manifold in pixel space.
    Bishop & Bishop (2024), page 177, Figure 6.7.
    """
    setup_style()
    fig, axes = plt.subplots(2, 4, figsize=(7.5, 4.0))

    # 8 different variations (dx, dy, angle) of digit '5'
    variations = [
        (-0.25, 0.20, 0),
        (0.25, -0.20, 5),
        (0.25, 0.25, -10),
        (0.35, 0.30, 15),
        (-0.20, -0.25, 0),
        (0.30, -0.20, -8),
        (0.25, -0.30, 0),
        (0.35, -0.35, 8),
    ]

    for ax, (dx, dy, angle) in zip(axes.ravel(), variations):
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.text(dx, dy, "5", fontsize=38, color="#cc0000",
                ha="center", va="center", rotation=angle,
                fontfamily="sans-serif", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.2)

    fig.tight_layout()

    if result_dirs:
        for rdir in result_dirs:
            save_fig(fig, os.path.join(rdir, "fig_6_7_digit_manifold.png"))

    return fig


def generate_figure_6_8(result_dirs: Optional[List[str]] = None) -> plt.Figure:
    """
    Figure 6.8: Natural images (top row) vs uniform random pixel noise (bottom row).
    Bishop & Bishop (2024), page 178, Figure 6.8.
    """
    setup_style()
    np.random.seed(42)

    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.5))

    # Synthetic natural textures with smooth spatial correlations
    def make_smooth_texture(size=64, base_color=(0.8, 0.5, 0.3), roughness=0.08):
        x = np.linspace(-2, 2, size)
        y = np.linspace(-2, 2, size)
        xx, yy = np.meshgrid(x, y)
        blob = np.exp(-(xx ** 2 + yy ** 2) / 1.5)
        # Add smooth low frequency gradients
        grad = 0.3 * np.sin(xx * 1.5) + 0.2 * np.cos(yy * 1.8)
        img = np.zeros((size, size, 3))
        for c in range(3):
            noise = np.random.normal(0, roughness, (size, size))
            img[:, :, c] = np.clip(base_color[c] * (blob * 0.7 + 0.3) + grad * 0.15 + noise, 0, 1)
        return img

    nat1 = make_smooth_texture(64, base_color=(0.75, 0.45, 0.3), roughness=0.04)  # Lesion/mole-like
    nat2 = make_smooth_texture(64, base_color=(0.4, 0.5, 0.6), roughness=0.05)    # City/bicycle tone
    nat3 = make_smooth_texture(64, base_color=(0.8, 0.6, 0.35), roughness=0.03)   # Cat fur tone

    natural_imgs = [nat1, nat2, nat3]

    for col in range(3):
        axes[0, col].imshow(natural_imgs[col])
        axes[0, col].axis("off")

        # Bottom row: pure independent uniform random RGB noise
        noise_img = np.random.uniform(0.0, 1.0, (64, 64, 3))
        axes[1, col].imshow(noise_img)
        axes[1, col].axis("off")

    fig.tight_layout()

    if result_dirs:
        for rdir in result_dirs:
            save_fig(fig, os.path.join(rdir, "fig_6_8_natural_vs_random_images.png"))

    return fig


def generate_all_section_6_1_figures(result_dirs: Optional[List[str]] = None) -> Dict[str, plt.Figure]:
    """Generate and save all figures for Section 6.1 (Figures 6.1 through 6.8)."""
    if result_dirs is None:
        root = _get_project_root()
        result_dirs = [
            os.path.join(root, "6", "result"),
            os.path.join(root, "result")
        ]

    for rdir in result_dirs:
        os.makedirs(rdir, exist_ok=True)

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
