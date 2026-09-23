"""Chapter 7: Gradient Descent
Section 7.1: Error Surfaces

This module implements mathematical models, local quadratic approximations,
and figure reproduction functions for Section 7.1 of:
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.

Contents:
- Mathematical Formulations:
  * Small weight step change: Delta E ~= Delta w^T grad E (Eq 7.1)
  * Stationary points condition: grad E(w) = 0 (Eq 7.2)
  * Local quadratic Taylor expansion around w_hat (Eq 7.3 - 7.5)
  * Local gradient approximation: grad E(w) = b + H(w - w_hat) (Eq 7.6)
  * Expansion around minimum w*: E(w) = E(w*) + 0.5 * (w - w*)^T H (w - w*) (Eq 7.7)
  * Hessian eigendecomposition: H u_i = lambda_i u_i (Eq 7.8, 7.9)
  * Coordinate transformation to eigen-basis: w - w* = sum_i xi_i u_i (Eq 7.10)
  * Error function in decoupled eigen-coordinates: E(w) = E(w*) + 0.5 * sum_i lambda_i xi_i^2 (Eq 7.11)
  * Positive definiteness condition: v^T H v > 0 for all v != 0 (Eq 7.12 - 7.14)
- Classes & Utilities:
  * LocalQuadraticApproximation: Evaluates Taylor expansion, gradient, and eigen-basis transformation
  * classify_stationary_point: Identifies minimum, maximum, saddle point, or degenerate flat direction
  * numerical_gradient & numerical_hessian: Finite-difference calculus verifiers
- Figure Reproductions:
  * Figure 7.1: Geometrical view of the error function E(w) as a surface over weight space
  * Figure 7.2: Contours of constant error as ellipses aligned with eigenvectors of Hessian
"""

import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.interpolate import CubicSpline

from common.plot_utils import save_plot, setup_style


def _get_project_root() -> str:
    """Return the absolute path to the repository root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _save_figure(fig: plt.Figure, filename: str, filepath: Optional[str] = None, save_both: bool = True) -> Tuple[str, Optional[str]]:
    """Save figure to target filepath, and/or both 7/result and root result directory."""
    if filepath:
        save_plot(fig, filepath)
    root = _get_project_root()
    path_ch7 = os.path.join(root, "7", "result", filename)
    path_root = os.path.join(root, "result", filename)
    if save_both or not filepath:
        if not filepath or os.path.abspath(filepath) != os.path.abspath(path_ch7):
            save_plot(fig, path_ch7)
        if not filepath or os.path.abspath(filepath) != os.path.abspath(path_root):
            save_plot(fig, path_root)
        return path_ch7, path_root
    return filepath, None


# ---------------------------------------------------------------------------
# Mathematical Utilities: Local Quadratic Approximation & Stationary Points
# ---------------------------------------------------------------------------

class LocalQuadraticApproximation:
    """Local quadratic Taylor approximation of an error surface E(w).
    
    E(w) ~= E(w_hat) + (w - w_hat)^T b + 0.5 * (w - w_hat)^T H (w - w_hat)  (Eq 7.3)
    where:
      b = grad E(w_hat)  (Eq 7.4)
      H = grad grad E(w_hat)  (Eq 7.5)
    """

    def __init__(
        self,
        center: np.ndarray,
        value: float,
        gradient: np.ndarray,
        hessian: np.ndarray,
    ):
        self.center = np.asarray(center, dtype=np.float64)
        self.value = float(value)
        self.gradient_b = np.asarray(gradient, dtype=np.float64)
        self.hessian_H = np.asarray(hessian, dtype=np.float64)
        self.dim = len(self.center)

        # Symmetrize Hessian for numerical robustness: H = 0.5 * (H + H^T)
        self.hessian_H = 0.5 * (self.hessian_H + self.hessian_H.T)

        # Eigendecomposition: H u_i = lambda_i u_i (Eq 7.8)
        eigenvals, eigenvecs = np.linalg.eigh(self.hessian_H)
        # Sort in ascending order
        sort_idx = np.argsort(eigenvals)
        self.eigenvalues = eigenvals[sort_idx]
        self.eigenvectors = eigenvecs[:, sort_idx]  # Columns are eigenvectors u_i

    def evaluate(self, w: np.ndarray) -> Union[float, np.ndarray]:
        """Evaluate quadratic approximation at point(s) w."""
        w_arr = np.asarray(w, dtype=np.float64)
        if w_arr.ndim == 1:
            diff = w_arr - self.center
            quad = 0.5 * diff.T @ self.hessian_H @ diff
            lin = diff.T @ self.gradient_b
            return float(self.value + lin + quad)
        else:
            diff = w_arr - self.center[None, :]
            lin = diff @ self.gradient_b
            quad = 0.5 * np.sum((diff @ self.hessian_H) * diff, axis=1)
            return self.value + lin + quad

    def gradient(self, w: np.ndarray) -> np.ndarray:
        """Evaluate local gradient approximation grad E(w) = b + H(w - w_hat) (Eq 7.6)."""
        w_arr = np.asarray(w, dtype=np.float64)
        if w_arr.ndim == 1:
            return self.gradient_b + self.hessian_H @ (w_arr - self.center)
        else:
            diff = w_arr - self.center[None, :]
            return self.gradient_b[None, :] + diff @ self.hessian_H

    def to_eigen_coordinates(self, w: np.ndarray) -> np.ndarray:
        """Transform difference w - w_hat to eigen-coordinates xi = U^T (w - w_hat) (Eq 7.10)."""
        w_arr = np.asarray(w, dtype=np.float64)
        if w_arr.ndim == 1:
            return self.eigenvectors.T @ (w_arr - self.center)
        else:
            diff = w_arr - self.center[None, :]
            return diff @ self.eigenvectors

    def from_eigen_coordinates(self, xi: np.ndarray) -> np.ndarray:
        """Reconstruct weight vector w = w_hat + U xi from eigen-coordinates xi (Eq 7.10)."""
        xi_arr = np.asarray(xi, dtype=np.float64)
        if xi_arr.ndim == 1:
            return self.center + self.eigenvectors @ xi_arr
        else:
            return self.center[None, :] + xi_arr @ self.eigenvectors.T

    def evaluate_in_eigen_coordinates(self, xi: np.ndarray) -> Union[float, np.ndarray]:
        """Evaluate error in decoupled eigen-coordinates E(w) = E(w*) + 0.5 * sum_i lambda_i xi_i^2 (Eq 7.11).
        
        Assumes gradient b = 0 at stationary point w*.
        """
        xi_arr = np.asarray(xi, dtype=np.float64)
        if xi_arr.ndim == 1:
            return float(self.value + 0.5 * np.sum(self.eigenvalues * (xi_arr ** 2)))
        else:
            return self.value + 0.5 * np.sum(self.eigenvalues[None, :] * (xi_arr ** 2), axis=1)

    def contour_semi_axes(self, delta_E: float) -> np.ndarray:
        """Calculate lengths of ellipse semi-axes for constant error contour E(w) = E(w*) + delta_E.
        
        0.5 * lambda_i * xi_i^2 = delta_E  =>  xi_i = sqrt(2 * delta_E / lambda_i) = sqrt(2 * delta_E) * lambda_i^(-1/2) (Figure 7.2).
        Requires positive eigenvalues.
        """
        if np.any(self.eigenvalues <= 0):
            raise ValueError("All eigenvalues must be positive for closed elliptical contours.")
        return np.sqrt(2.0 * delta_E / self.eigenvalues)

    def is_positive_definite(self, tol: float = 1e-8) -> bool:
        """Check if Hessian H is positive definite (Eq 7.12 - 7.14)."""
        return bool(np.all(self.eigenvalues > tol))


def classify_stationary_point(eigenvalues: np.ndarray, tol: float = 1e-8) -> str:
    """Classify stationary point based on Hessian eigenvalues (Section 7.1.1).
    
    - 'minimum': all eigenvalues > tol
    - 'maximum': all eigenvalues < -tol
    - 'saddle_point': both positive and negative eigenvalues exist
    - 'degenerate': one or more eigenvalues have |lambda| <= tol
    """
    evals = np.asarray(eigenvalues, dtype=np.float64)
    pos = evals > tol
    neg = evals < -tol
    zero = np.abs(evals) <= tol

    if np.any(zero):
        return "degenerate"
    if np.all(pos):
        return "minimum"
    if np.all(neg):
        return "maximum"
    if np.any(pos) and np.any(neg):
        return "saddle_point"
    return "unclassified"


def numerical_gradient(func: Callable[[np.ndarray], float], w: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute numerical gradient of func at w via central differences."""
    w = np.asarray(w, dtype=np.float64)
    grad = np.zeros_like(w)
    for i in range(len(w)):
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[i] += eps
        w_minus[i] -= eps
        grad[i] = (func(w_plus) - func(w_minus)) / (2.0 * eps)
    return grad


def numerical_hessian(func: Callable[[np.ndarray], float], w: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Compute numerical Hessian matrix of func at w via central differences."""
    w = np.asarray(w, dtype=np.float64)
    n = len(w)
    H = np.zeros((n, n))
    f0 = func(w)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                w_p = w.copy()
                w_m = w.copy()
                w_p[i] += eps
                w_m[i] -= eps
                H[i, i] = (func(w_p) - 2.0 * f0 + func(w_m)) / (eps ** 2)
            else:
                w_pp = w.copy()
                w_pm = w.copy()
                w_mp = w.copy()
                w_mm = w.copy()
                w_pp[i] += eps
                w_pp[j] += eps
                w_pm[i] += eps
                w_pm[j] -= eps
                w_mp[i] -= eps
                w_mp[j] += eps
                w_mm[i] -= eps
                w_mm[j] -= eps
                H[i, j] = (func(w_pp) - func(w_pm) - func(w_mp) + func(w_mm)) / (4.0 * eps ** 2)
                H[j, i] = H[i, j]
    return H


# ---------------------------------------------------------------------------
# Figure 7.1: Geometrical View of Error Function over Weight Space
# ---------------------------------------------------------------------------

def generate_figure_7_1(
    filepath: Optional[str] = None,
    save_both: bool = True
) -> plt.Figure:
    """Generate and save Figure 7.1 (Bishop & Bishop 2024, p. 211).
    
    Geometrical view of the error function E(w) as a surface sitting over
    weight space (w1, w2). Point wA is a local minimum, wB is the global
    minimum so that E(wA) > E(wB), and at point wC the local gradient is grad E.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.set_xlim(0.0, 5.5)
    ax.set_ylim(-0.2, 5.8)
    ax.set_aspect("equal")
    ax.axis("off")

    # Origin in 2D projection
    O = np.array([1.2, 1.8])

    # Axes
    # E(w) axis: vertical up
    ax.annotate("", xy=(O[0], 5.6), xytext=(O[0], O[1] - 0.1),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=12))
    ax.text(O[0] - 0.15, 5.6, r"$E(\mathbf{w})$", ha="right", va="center", fontsize=13)

    # w1 axis: horizontal right
    ax.annotate("", xy=(5.2, O[1]), xytext=(O[0] - 0.1, O[1]),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=12))
    ax.text(5.2, O[1] - 0.25, r"$w_1$", ha="center", va="top", fontsize=13)

    # w2 axis: down-left
    w2_end = O + np.array([-0.9, -0.9])
    ax.annotate("", xy=(w2_end[0], w2_end[1]), xytext=(O[0], O[1]),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=12))
    ax.text(w2_end[0] - 0.05, w2_end[1] - 0.2, r"$w_2$", ha="right", va="top", fontsize=13)

    # Ground plane points (projections)
    p_wA = np.array([2.3, 1.4])
    p_wB = np.array([3.4, 1.25])
    p_wC = np.array([4.1, 1.1])

    # Rim of the bowl (horizontal ellipse)
    rim_cx = 2.9
    rim_cy = 4.6
    a_rim = 1.35
    b_rim = 0.35

    rim = Ellipse((rim_cx, rim_cy), 2 * a_rim, 2 * b_rim, angle=0,
                  edgecolor="red", facecolor="none", lw=2.2, zorder=4)
    ax.add_patch(rim)

    # Left and right tangent points on rim
    pt_left = np.array([rim_cx - a_rim, rim_cy])
    pt_right = np.array([rim_cx + a_rim, rim_cy])

    # Surface points
    s_wA = np.array([p_wA[0], 2.2])
    s_wB = np.array([p_wB[0], 1.7])
    s_saddle = np.array([2.85, 2.35])
    s_wC = np.array([p_wC[0], 3.8])

    # Smooth curve through: pt_left -> (1.75, 3.2) -> s_wA -> s_saddle -> s_wB -> s_wC -> pt_right
    ctrl_pts = np.array([
        pt_left,
        [1.75, 3.2],
        s_wA,
        s_saddle,
        s_wB,
        s_wC,
        pt_right
    ])

    distances = np.zeros(len(ctrl_pts))
    for i in range(1, len(ctrl_pts)):
        distances[i] = distances[i-1] + np.linalg.norm(ctrl_pts[i] - ctrl_pts[i-1])

    cs_x = CubicSpline(distances, ctrl_pts[:, 0], bc_type=((1, 0.0), (1, 0.0)))
    cs_y = CubicSpline(distances, ctrl_pts[:, 1], bc_type=((1, -2.5), (1, 2.5)))

    d_eval = np.linspace(0, distances[-1], 300)
    curve_x = cs_x(d_eval)
    curve_y = cs_y(d_eval)
    ax.plot(curve_x, curve_y, color="red", lw=2.2, zorder=5)

    # Vertical dashed line at wA up to rim level
    ax.plot([p_wA[0], p_wA[0]], [s_wA[1], rim_cy + 0.4], color="black", linestyle="--", lw=1.0, zorder=2)

    # Blue drop lines from surface to ground plane
    # 1. At wA
    ax.plot([s_wA[0], p_wA[0]], [s_wA[1], p_wA[1]], color="blue", lw=2.0, zorder=6)
    ax.scatter([s_wA[0]], [s_wA[1]], color="black", s=25, zorder=7)
    ax.scatter([p_wA[0]], [p_wA[1]], color="black", s=25, zorder=7)
    ax.text(p_wA[0] - 0.25, p_wA[1] - 0.15, r"$\mathbf{w}_A$", ha="center", va="top", fontsize=12)

    # 2. At wB
    ax.plot([s_wB[0], p_wB[0]], [s_wB[1], p_wB[1]], color="blue", lw=2.0, zorder=6)
    ax.scatter([s_wB[0]], [s_wB[1]], color="black", s=25, zorder=7)
    ax.scatter([p_wB[0]], [p_wB[1]], color="black", s=25, zorder=7)
    ax.text(p_wB[0] - 0.15, p_wB[1] - 0.15, r"$\mathbf{w}_B$", ha="center", va="top", fontsize=12)

    # 3. At wC
    ax.plot([s_wC[0], p_wC[0]], [s_wC[1], p_wC[1]], color="blue", lw=2.0, zorder=6)
    ax.scatter([s_wC[0]], [s_wC[1]], color="black", s=25, zorder=7)
    ax.scatter([p_wC[0]], [p_wC[1]], color="black", s=25, zorder=7)
    ax.text(p_wC[0] + 0.25, p_wC[1] - 0.15, r"$\mathbf{w}_C$", ha="left", va="top", fontsize=12)

    # Green gradient arrow at wC pointing forward-right
    grad_end = p_wC + np.array([0.45, -0.45])
    ax.annotate("", xy=(grad_end[0], grad_end[1]), xytext=(p_wC[0], p_wC[1]),
                arrowprops=dict(arrowstyle="-|>", color="green", lw=2.0, mutation_scale=14), zorder=8)
    ax.text(grad_end[0] + 0.1, grad_end[1] - 0.1, r"$\nabla E$", color="green", ha="left", va="top", fontsize=13)

    plt.tight_layout()
    _save_figure(fig, "fig_7_1_error_surface.png", filepath=filepath, save_both=save_both)
    return fig


# ---------------------------------------------------------------------------
# Figure 7.2: Contours of Constant Error and Hessian Eigenvectors
# ---------------------------------------------------------------------------

def generate_figure_7_2(
    filepath: Optional[str] = None,
    save_both: bool = True
) -> plt.Figure:
    """Generate and save Figure 7.2 (Bishop & Bishop 2024, p. 213).
    
    Contours of constant error as ellipses aligned with the eigenvectors
    u1, u2 of the Hessian matrix, with semi-axis lengths inversely
    proportional to the square roots of eigenvalues lambda_1^(-1/2), lambda_2^(-1/2).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.set_xlim(0.0, 5.5)
    ax.set_ylim(0.0, 5.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Axes origin
    O = np.array([0.8, 0.8])

    # w1 axis: horizontal right
    ax.annotate("", xy=(5.2, O[1]), xytext=(O[0], O[1]),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=12))
    ax.text(5.2, O[1] - 0.25, r"$w_1$", ha="center", va="top", fontsize=13)

    # w2 axis: vertical up
    ax.annotate("", xy=(O[0], 4.6), xytext=(O[0], O[1]),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=12))
    ax.text(O[0] - 0.25, 4.6, r"$w_2$", ha="right", va="center", fontsize=13)

    # Center of ellipse: w*
    w_star = np.array([3.1, 2.7])
    ax.scatter([w_star[0]], [w_star[1]], color="black", s=25, zorder=5)
    ax.text(w_star[0] - 0.15, w_star[1] - 0.15, r"$\mathbf{w}^\star$", ha="right", va="top", fontsize=13)

    # Ellipse parameters
    angle_deg = 20.0
    angle_rad = np.radians(angle_deg)
    semi_major = 1.7
    semi_minor = 0.65

    # Ellipse contour in red
    ellipse = Ellipse(w_star, 2 * semi_major, 2 * semi_minor, angle=angle_deg,
                      edgecolor="red", facecolor="none", lw=2.2, zorder=3)
    ax.add_patch(ellipse)

    # Eigenvector u1 along major axis
    u1_dir = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    u1_end = w_star + semi_major * u1_dir
    ax.annotate("", xy=(u1_end[0], u1_end[1]), xytext=(w_star[0], w_star[1]),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=12), zorder=4)
    ax.text(u1_end[0] + 0.15, u1_end[1], r"$\mathbf{u}_1$", ha="left", va="center", fontsize=13)

    # Eigenvector u2 along minor axis (perpendicular)
    u2_dir = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    u2_end = w_star + semi_minor * u2_dir
    ax.annotate("", xy=(u2_end[0], u2_end[1]), xytext=(w_star[0], w_star[1]),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=12), zorder=4)
    ax.text(u2_end[0] - 0.15, u2_end[1] + 0.15, r"$\mathbf{u}_2$", ha="right", va="bottom", fontsize=13)

    # Length annotations:
    # 1. lambda_1^(-1/2) parallel to u1 (shifted downward)
    offset_u1 = - 0.45 * u2_dir
    p1_start = w_star + offset_u1
    p1_end = w_star + semi_major * u1_dir + offset_u1
    ax.annotate("", xy=(p1_end[0], p1_end[1]), xytext=(p1_start[0], p1_start[1]),
                arrowprops=dict(arrowstyle="<|-|>", color="black", lw=1.2, mutation_scale=10), zorder=4)
    ax.text(0.5 * (p1_start[0] + p1_end[0]) + 0.4, 0.5 * (p1_start[1] + p1_end[1]) - 0.25,
            r"$\lambda_1^{-1/2}$", ha="center", va="top", fontsize=12)

    # 2. lambda_2^(-1/2) along minor axis across left contour edge
    p2_mid = w_star - semi_major * 0.95 * u1_dir
    p2_start = p2_mid - 0.35 * u2_dir
    p2_end = p2_mid + 0.5 * u2_dir
    ax.annotate("", xy=(p2_end[0], p2_end[1]), xytext=(p2_start[0], p2_start[1]),
                arrowprops=dict(arrowstyle="<|-|>", color="black", lw=1.2, mutation_scale=10), zorder=4)
    ax.text(p2_start[0] - 0.15, 0.5 * (p2_start[1] + p2_end[1]),
            r"$\lambda_2^{-1/2}$", ha="right", va="center", fontsize=12)

    plt.tight_layout()
    _save_figure(fig, "fig_7_2_error_contours.png", filepath=filepath, save_both=save_both)
    return fig
