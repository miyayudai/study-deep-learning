"""
Linear Models for Regression (Chapter 4: Single-layer Networks: Regression).
Covers:
- Basis Functions: Polynomial, Gaussian, Sigmoidal (Section 4.1.1, Eq 4.4 - 4.6)
- Maximum Likelihood Linear Regression & Pseudo-inverse (Section 4.1.2 - 4.1.3, Eq 4.14 - 4.20)
- Least Squares Geometry & Orthogonal Projection (Section 4.1.4, Figure 4.3)
- Sequential Learning / LMS Algorithm (Section 4.1.5, Eq 4.21 - 4.22)
- Regularized Least Squares / Ridge Regression (Section 4.1.6, Eq 4.26 - 4.27)
- Multiple Output Linear Regression (Section 4.1.7, Eq 4.28 - 4.32, Figure 4.4)
- Faithful reproduction of Figures 4.1, 4.2, 4.3, and 4.4.
"""
import os
from typing import Optional, Union, Tuple, List, Callable
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Polygon

from common.plot_utils import setup_style


# =====================================================================
# 1. Basis Functions (Section 4.1.1)
# =====================================================================

class PolynomialBasis:
    """Polynomial basis functions: phi_j(x) = x^j for j = 0, ..., M-1."""
    def __init__(self, degree: int):
        self.degree = int(degree)
        self.n_features = self.degree + 1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        # Shape: (N, degree + 1)
        return np.column_stack([x_arr ** j for j in range(self.degree + 1)])


class GaussianBasis:
    """
    Gaussian basis functions (Eq 4.4):
        phi_j(x) = exp( - (x - mu_j)^2 / (2 * s^2) )
    Includes dummy bias basis phi_0(x) = 1.
    """
    def __init__(self, centers: Union[List[float], np.ndarray], s: float, include_bias: bool = True):
        self.centers = np.asarray(centers, dtype=np.float64).ravel()
        if s <= 0:
            raise ValueError("Scale parameter s must be positive")
        self.s = float(s)
        self.include_bias = include_bias
        self.n_features = len(self.centers) + (1 if include_bias else 0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        # (N, M_centers)
        diff = (x_arr[:, None] - self.centers[None, :]) / self.s
        phi = np.exp(-0.5 * diff ** 2)
        if self.include_bias:
            bias = np.ones((len(x_arr), 1), dtype=np.float64)
            return np.hstack([bias, phi])
        return phi


class SigmoidalBasis:
    """
    Sigmoidal basis functions (Eq 4.5 - 4.6):
        phi_j(x) = sigma( (x - mu_j) / s )
        sigma(a) = 1 / (1 + exp(-a))
    Includes dummy bias basis phi_0(x) = 1.
    """
    def __init__(self, centers: Union[List[float], np.ndarray], s: float, include_bias: bool = True):
        self.centers = np.asarray(centers, dtype=np.float64).ravel()
        if s <= 0:
            raise ValueError("Scale parameter s must be positive")
        self.s = float(s)
        self.include_bias = include_bias
        self.n_features = len(self.centers) + (1 if include_bias else 0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        a = (x_arr[:, None] - self.centers[None, :]) / self.s
        phi = 1.0 / (1.0 + np.exp(-np.clip(a, -100, 100)))
        if self.include_bias:
            bias = np.ones((len(x_arr), 1), dtype=np.float64)
            return np.hstack([bias, phi])
        return phi


# =====================================================================
# 2. Linear Regression Models (Section 4.1.2 - 4.1.7)
# =====================================================================

class LinearRegression:
    """
    Linear Regression Model (Section 4.1.2 - 4.1.3):
        y(x, w) = w^T phi(x)
        p(t | x, w, sigma^2) = N(t | w^T phi(x), sigma^2)
        w_ML = (Phi^T Phi)^(-1) Phi^T t = Phi^dagger t
        sigma_ML^2 = (1 / N) sum_{n=1}^N { t_n - w_ML^T phi(x_n) }^2
    """
    def __init__(self, basis_func: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.basis_func = basis_func
        self.w: Optional[np.ndarray] = None
        self.sigma2: Optional[float] = None
        self.Phi: Optional[np.ndarray] = None
        self.N: int = 0
        self.M: int = 0

    def fit(self, X: np.ndarray, t: np.ndarray) -> "LinearRegression":
        """
        Fit weights using closed-form Maximum Likelihood / Normal Equations (Eq 4.14).
        Uses Moore-Penrose pseudo-inverse (Eq 4.16) for robust solution.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        t_arr = np.asarray(t, dtype=np.float64).ravel()
        self.N = len(t_arr)

        if self.basis_func is not None:
            self.Phi = self.basis_func(X_arr)
        else:
            # Add bias column if not already present
            if X_arr.ndim == 1:
                X_arr = X_arr[:, None]
            self.Phi = np.hstack([np.ones((self.N, 1)), X_arr])

        self.M = self.Phi.shape[1]

        # Compute w_ML = pinv(Phi) @ t (Eq 4.14, 4.16)
        self.w = la.pinv(self.Phi) @ t_arr

        # Residual variance sigma_ML^2 (Eq 4.20)
        residuals = t_arr - (self.Phi @ self.w)
        self.sigma2 = float(np.mean(residuals ** 2))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute point prediction y(x, w) = Phi @ w."""
        if self.w is None:
            raise ValueError("Model must be fitted before calling predict")
        X_arr = np.asarray(X, dtype=np.float64)
        if self.basis_func is not None:
            Phi_query = self.basis_func(X_arr)
        else:
            if X_arr.ndim == 1:
                X_arr = X_arr[:, None]
            Phi_query = np.hstack([np.ones((len(X_arr), 1)), X_arr])
        return Phi_query @ self.w

    def log_likelihood(self, X: np.ndarray, t: np.ndarray) -> float:
        """Compute Gaussian log-likelihood ln p(t | X, w, sigma^2) (Eq 4.10)."""
        t_arr = np.asarray(t, dtype=np.float64).ravel()
        y_pred = self.predict(X)
        ED = 0.5 * np.sum((t_arr - y_pred) ** 2)
        N = len(t_arr)
        return float(- 0.5 * N * np.log(self.sigma2) - 0.5 * N * np.log(2.0 * np.pi) - (1.0 / self.sigma2) * ED)


class SequentialLinearRegression:
    """
    Sequential / Online Linear Regression via LMS Algorithm (Section 4.1.5, Eq 4.21 - 4.22):
        w^(tau + 1) = w^(tau) + eta * (t_n - w^(tau)^T phi_n) * phi_n
    """
    def __init__(self, n_features: int, learning_rate: float = 0.01, w_init: Optional[np.ndarray] = None):
        self.n_features = n_features
        self.learning_rate = float(learning_rate)
        if w_init is not None:
            self.w = np.asarray(w_init, dtype=np.float64).copy()
        else:
            self.w = np.zeros(n_features, dtype=np.float64)
        self.trajectory = [self.w.copy()]

    def update(self, phi_n: np.ndarray, t_n: float) -> np.ndarray:
        """Perform one LMS update step with single sample (phi_n, t_n)."""
        phi_n = np.asarray(phi_n, dtype=np.float64).ravel()
        error = float(t_n - np.dot(self.w, phi_n))
        self.w += self.learning_rate * error * phi_n
        self.trajectory.append(self.w.copy())
        return self.w


class RidgeRegression:
    """
    Regularized Least Squares / Ridge Regression (Section 4.1.6, Eq 4.26 - 4.27):
        Total Error: E_D(w) + 0.5 * lambda * w^T w
        Solution: w = (lambda * I + Phi^T Phi)^(-1) Phi^T t
    """
    def __init__(self, alpha: float = 1.0, basis_func: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        if alpha < 0:
            raise ValueError("Regularization coefficient alpha must be non-negative")
        self.alpha = float(alpha)
        self.basis_func = basis_func
        self.w: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, t: np.ndarray) -> "RidgeRegression":
        X_arr = np.asarray(X, dtype=np.float64)
        t_arr = np.asarray(t, dtype=np.float64).ravel()
        N = len(t_arr)

        if self.basis_func is not None:
            Phi = self.basis_func(X_arr)
        else:
            if X_arr.ndim == 1:
                X_arr = X_arr[:, None]
            Phi = np.hstack([np.ones((N, 1)), X_arr])

        M = Phi.shape[1]
        reg_matrix = self.alpha * np.eye(M)
        # Solve (alpha * I + Phi^T Phi) w = Phi^T t
        self.w = la.solve(reg_matrix + Phi.T @ Phi, Phi.T @ t_arr, assume_a='pos')
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise ValueError("Model must be fitted before predict")
        X_arr = np.asarray(X, dtype=np.float64)
        if self.basis_func is not None:
            Phi_query = self.basis_func(X_arr)
        else:
            if X_arr.ndim == 1:
                X_arr = X_arr[:, None]
            Phi_query = np.hstack([np.ones((len(X_arr), 1)), X_arr])
        return Phi_query @ self.w


class MultipleOutputLinearRegression:
    """
    Multiple Output Linear Regression (Section 4.1.7, Eq 4.28 - 4.32):
        y(x, W) = W^T phi(x)
        W_ML = (Phi^T Phi)^(-1) Phi^T T = Phi^dagger T
    """
    def __init__(self, basis_func: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.basis_func = basis_func
        self.W: Optional[np.ndarray] = None
        self.sigma2: Optional[float] = None

    def fit(self, X: np.ndarray, T: np.ndarray) -> "MultipleOutputLinearRegression":
        X_arr = np.asarray(X, dtype=np.float64)
        T_arr = np.asarray(T, dtype=np.float64)
        if T_arr.ndim == 1:
            T_arr = T_arr[:, None]
        N, K = T_arr.shape

        if self.basis_func is not None:
            Phi = self.basis_func(X_arr)
        else:
            if X_arr.ndim == 1:
                X_arr = X_arr[:, None]
            Phi = np.hstack([np.ones((N, 1)), X_arr])

        # W_ML = Phi^dagger @ T (Eq 4.31)
        self.W = la.pinv(Phi) @ T_arr
        residuals = T_arr - (Phi @ self.W)
        self.sigma2 = float(np.mean(residuals ** 2))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise ValueError("Model must be fitted before predict")
        X_arr = np.asarray(X, dtype=np.float64)
        if self.basis_func is not None:
            Phi_query = self.basis_func(X_arr)
        else:
            if X_arr.ndim == 1:
                X_arr = X_arr[:, None]
            Phi_query = np.hstack([np.ones((len(X_arr), 1)), X_arr])
        return Phi_query @ self.W


# =====================================================================
# 3. Figure Reproduction Plotters (Figures 4.1 - 4.4)
# =====================================================================

def plot_figure_4_1_network_diagram(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Faithfully reproduce Figure 4.1 from Bishop & Bishop (2024), page 113:
    Single-layer linear regression neural network diagram.
    """
    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=300)

    # Node positions
    # Input nodes on the left: x = 0.2
    y_nodes = [0.2, 0.45, 0.75]
    node_radius = 0.055

    # Draw bottom solid bias node phi_0(x)
    c0 = Circle((0.2, y_nodes[0]), node_radius, facecolor='#0044FF', edgecolor='#001188', linewidth=1.5, zorder=3)
    ax.add_patch(c0)
    ax.text(0.11, y_nodes[0], r"$\phi_0(\mathbf{x})$", fontsize=11, ha='right', va='center')

    # Draw phi_1(x)
    c1 = Circle((0.2, y_nodes[1]), node_radius, facecolor='#D0E0FF', edgecolor='#0044FF', linewidth=1.5, zorder=3)
    ax.add_patch(c1)
    ax.text(0.11, y_nodes[1], r"$\phi_1(\mathbf{x})$", fontsize=11, ha='right', va='center')

    # Draw vertical dots
    ax.text(0.2, 0.60, r"$\vdots$", fontsize=14, ha='center', va='center')

    # Draw phi_{M-1}(x)
    c_m = Circle((0.2, y_nodes[2]), node_radius, facecolor='#D0E0FF', edgecolor='#0044FF', linewidth=1.5, zorder=3)
    ax.add_patch(c_m)
    ax.text(0.11, y_nodes[2], r"$\phi_{M-1}(\mathbf{x})$", fontsize=11, ha='right', va='center')

    # Output node on the right: x = 0.75, y = 0.45
    c_out = Circle((0.75, 0.45), node_radius, facecolor='#D0E0FF', edgecolor='#0044FF', linewidth=1.5, zorder=3)
    ax.add_patch(c_out)
    ax.text(0.75, 0.58, r"$y(\mathbf{x}, \mathbf{w})$", fontsize=11, ha='center', va='bottom')

    # Connecting arrows
    arrow_props = dict(arrowstyle='->', color='black', lw=1.2, mutation_scale=12)

    # From phi_0 to output
    ax.annotate('', xy=(0.75 - node_radius, 0.45 - 0.01), xytext=(0.2 + node_radius, y_nodes[0] + 0.01),
                arrowprops=arrow_props, zorder=2)
    ax.text(0.50, 0.30, r"$w_0$", fontsize=10.5, ha='center', va='center')

    # From phi_1 to output
    ax.annotate('', xy=(0.75 - node_radius, 0.45), xytext=(0.2 + node_radius, y_nodes[1]),
                arrowprops=arrow_props, zorder=2)
    ax.text(0.50, 0.49, r"$w_1$", fontsize=10.5, ha='center', va='bottom')

    # From phi_{M-1} to output
    ax.annotate('', xy=(0.75 - node_radius, 0.45 + 0.01), xytext=(0.2 + node_radius, y_nodes[2] - 0.01),
                arrowprops=arrow_props, zorder=2)
    ax.text(0.50, 0.65, r"$w_{M-1}$", fontsize=10.5, ha='center', va='center')

    ax.set_xlim(-0.05, 0.95)
    ax.set_ylim(0.05, 0.85)
    ax.axis('off')

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 4.1 saved to: {p}")
    if show:
        plt.show()
    return fig, ax


def plot_figure_4_2_basis_functions(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Faithfully reproduce Figure 4.2 from Bishop & Bishop (2024), page 114:
    Examples of basis functions:
    Left: Polynomials, Centre: Gaussians, Right: Sigmoidals.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.0, 3.2), dpi=300)
    x = np.linspace(-1, 1, 300)

    # Palette of colors matching textbook style
    colors = ['#00E000', '#0044FF', '#E00000', '#00C0C0', '#D000D0', '#E0A000', '#707070', '#0080FF', '#900000']

    # 1. Polynomials: x^j for j = 1, ..., 9
    for j, c in enumerate(colors[:8], start=1):
        ax1.plot(x, x ** j, color=c, linewidth=1.3)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1.05)
    ax1.set_xticks([-1, 0, 1])
    ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax1.tick_params(direction='in', top=True, right=True)
    for s in ax1.spines.values():
        s.set_linewidth(0.8)

    # 2. Gaussian basis functions: mu_j in [-1, 1], s = 0.2
    centers = np.linspace(-1, 1, 9)
    for mu, c in zip(centers, colors[:9]):
        y_gauss = np.exp(-0.5 * ((x - mu) / 0.2) ** 2)
        ax2.plot(x, y_gauss, color=c, linewidth=1.3)
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks([-1, 0, 1])
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax2.tick_params(direction='in', top=True, right=True)
    for s in ax2.spines.values():
        s.set_linewidth(0.8)

    # 3. Sigmoidal basis functions: mu_j in [-0.9, 0.9], s = 0.1
    centers_sig = np.linspace(-0.85, 0.85, 9)
    for mu, c in zip(centers_sig, colors[:9]):
        y_sig = 1.0 / (1.0 + np.exp(-(x - mu) / 0.1))
        ax3.plot(x, y_sig, color=c, linewidth=1.3)
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(0, 1.05)
    ax3.set_xticks([-1, 0, 1])
    ax3.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax3.tick_params(direction='in', top=True, right=True)
    for s in ax3.spines.values():
        s.set_linewidth(0.8)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 4.2 saved to: {p}")
    if show:
        plt.show()
    return fig, np.array([ax1, ax2, ax3])


def plot_figure_4_3_least_squares_geometry(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Faithfully reproduce Figure 4.3 from Bishop & Bishop (2024), page 117:
    Geometrical interpretation of the least-squares solution in N-dimensional space.
    Target vector t, basis vectors phi_1, phi_2 spanning subspace S,
    orthogonal projection y = Phi w_ML and error vector t - y.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)

    # Subspace S represented as a tilted red parallelogram
    origin = np.array([0.48, 0.18])
    poly_pts = np.array([
        [0.08, 0.65],
        [0.55, 0.85],
        [0.72, 0.25],
        [0.25, 0.05]
    ])
    subspace_patch = Polygon(poly_pts, closed=True, edgecolor='#E00000', facecolor='none', linewidth=1.5, zorder=1)
    ax.add_patch(subspace_patch)
    ax.text(0.20, 0.73, r"$\mathcal{S}$", fontsize=15, color='black', zorder=2)

    # Vectors inside the plane: phi_1 and phi_2
    phi1_end = np.array([0.22, 0.55])
    phi2_end = np.array([0.62, 0.52])
    y_end = np.array([0.40, 0.53]) # y lies in S

    # Target vector t pointing out of plane
    t_end = np.array([0.70, 0.80])

    # Arrow plotting helper
    def draw_arrow(start, end, col, lw=1.5):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=col, lw=lw, mutation_scale=14), zorder=3)

    # Draw phi_1 (red arrow in plane)
    draw_arrow(origin, phi1_end, '#E00000')
    ax.text(phi1_end[0] + 0.03, phi1_end[1] - 0.01, r"$\boldsymbol{\phi}_1$", fontsize=13, zorder=4)

    # Draw phi_2 (red arrow in plane)
    draw_arrow(origin, phi2_end, '#E00000')
    ax.text(phi2_end[0] - 0.04, phi2_end[1] + 0.03, r"$\boldsymbol{\phi}_2$", fontsize=13, zorder=4)

    # Draw y = Phi w_ML (blue arrow in plane)
    draw_arrow(origin, y_end, '#0000EE', lw=2.0)
    ax.text(y_end[0] - 0.05, y_end[1] + 0.02, r"$\mathbf{y}$", fontsize=13, color='black', zorder=4)

    # Draw target vector t (black arrow)
    draw_arrow(origin, t_end, 'black', lw=1.8)
    ax.text(t_end[0] + 0.02, t_end[1] - 0.04, r"$\mathbf{t}$", fontsize=13, zorder=4)

    # Draw error vector t - y (green line perpendicular to S)
    ax.plot([y_end[0], t_end[0]], [y_end[1], t_end[1]], color='#00C000', linewidth=1.8, zorder=3)

    # Draw right-angle symbol at y
    # Vector along y -> t
    v_perp = (t_end - y_end) / np.linalg.norm(t_end - y_end) * 0.035
    # Vector in plane along origin -> y
    v_in = (origin - y_end) / np.linalg.norm(origin - y_end) * 0.035
    corner1 = y_end + v_perp
    corner2 = y_end + v_perp + v_in
    corner3 = y_end + v_in
    ax.plot([corner1[0], corner2[0], corner3[0]], [corner1[1], corner2[1], corner3[1]],
            color='black', linewidth=1.0, zorder=3)

    ax.set_xlim(-0.02, 0.85)
    ax.set_ylim(-0.02, 0.95)
    ax.axis('off')

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 4.3 saved to: {p}")
    if show:
        plt.show()
    return fig, ax


def plot_figure_4_4_multiple_outputs_diagram(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Faithfully reproduce Figure 4.4 from Bishop & Bishop (2024), page 119:
    Representation of linear regression as a neural network with multiple outputs.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=300)

    # Input nodes: x = 0.2
    y_in = [0.2, 0.45, 0.75]
    node_radius = 0.055

    # Bottom solid bias node phi_0(x)
    c0 = Circle((0.2, y_in[0]), node_radius, facecolor='#0044FF', edgecolor='#001188', linewidth=1.5, zorder=3)
    ax.add_patch(c0)
    ax.text(0.11, y_in[0], r"$\phi_0(\mathbf{x})$", fontsize=11, ha='right', va='center')

    # phi_1(x)
    c1 = Circle((0.2, y_in[1]), node_radius, facecolor='#D0E0FF', edgecolor='#0044FF', linewidth=1.5, zorder=3)
    ax.add_patch(c1)
    ax.text(0.11, y_in[1], r"$\phi_1(\mathbf{x})$", fontsize=11, ha='right', va='center')

    # Vertical dots
    ax.text(0.2, 0.60, r"$\vdots$", fontsize=14, ha='center', va='center')

    # phi_{M-1}(x)
    c_m = Circle((0.2, y_in[2]), node_radius, facecolor='#D0E0FF', edgecolor='#0044FF', linewidth=1.5, zorder=3)
    ax.add_patch(c_m)
    ax.text(0.11, y_in[2], r"$\phi_{M-1}(\mathbf{x})$", fontsize=11, ha='right', va='center')

    # Output nodes: x = 0.75
    y_out = [0.35, 0.65]

    # y_1(x, w)
    c_out1 = Circle((0.75, y_out[0]), node_radius, facecolor='#D0E0FF', edgecolor='#0044FF', linewidth=1.5, zorder=3)
    ax.add_patch(c_out1)
    ax.text(0.83, y_out[0], r"$y_1(\mathbf{x}, \mathbf{w})$", fontsize=11, ha='left', va='center')

    # Vertical dots
    ax.text(0.75, 0.50, r"$\vdots$", fontsize=14, ha='center', va='center')

    # y_K(x, w)
    c_out2 = Circle((0.75, y_out[1]), node_radius, facecolor='#D0E0FF', edgecolor='#0044FF', linewidth=1.5, zorder=3)
    ax.add_patch(c_out2)
    ax.text(0.83, y_out[1], r"$y_K(\mathbf{x}, \mathbf{w})$", fontsize=11, ha='left', va='center')

    # Connecting arrows from each input node to each output node
    arrow_props = dict(arrowstyle='->', color='black', lw=1.1, mutation_scale=11)
    for yi in y_in:
        for yo in y_out:
            ax.annotate('', xy=(0.75 - node_radius, yo), xytext=(0.2 + node_radius, yi),
                        arrowprops=arrow_props, zorder=2)

    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(0.05, 0.85)
    ax.axis('off')

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 4.4 saved to: {p}")
    if show:
        plt.show()
    return fig, ax
