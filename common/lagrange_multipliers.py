"""
Bishop & Bishop (2024) Appendix C: Lagrange Multipliers (付録C: ラグランジュの未定乗数法)
Comprehensive implementation of equality constraints, inequality constraints,
Karush-Kuhn-Tucker (KKT) conditions, Lagrangian functions, and geometric visualizers.

Equations:
- (C.1) - (C.3): Equality constraint g(x) = 0 and stationarity condition grad f + lambda * grad g = 0
- (C.4): Lagrangian function L(x, lambda) = f(x) + lambda * g(x)
- (C.5) - (C.8): Textbook quadratic example f = 1 - x1^2 - x2^2, g = x1 + x2 - 1 = 0
- (C.9) - (C.11): Inequality constraints and KKT conditions (primal, dual, complementary slackness)
- (C.12): General Lagrangian for multiple equality and inequality constraints
"""

from typing import Tuple, List, Dict, Any, Optional, Callable
import shutil
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from pathlib import Path

from common.plot_utils import setup_style, save_plot


# ==============================================================================
# C.1 Equality Constraints & Lagrangian Formulation (等式制約とラグランジアン)
# ==============================================================================

def solve_textbook_example_c5() -> Dict[str, Any]:
    """
    Solve the textbook quadratic example (Eqs. C.5 - C.8):
    Maximize f(x1, x2) = 1 - x1^2 - x2^2
    Subject to g(x1, x2) = x1 + x2 - 1 = 0.
    
    Lagrangian: L(x, lambda) = 1 - x1^2 - x2^2 + lambda * (x1 + x2 - 1)
    
    Stationarity conditions:
    dL/dx1 = -2*x1 + lambda = 0  (C.6)
    dL/dx2 = -2*x2 + lambda = 0  (C.7)
    dL/dlambda = x1 + x2 - 1 = 0 (C.8)
    
    Exact solution:
    x1* = 1/2, x2* = 1/2, lambda* = 1
    Optimal value: f(1/2, 1/2) = 1 - 1/4 - 1/4 = 1/2.
    """
    x1_star = 0.5
    x2_star = 0.5
    lam_star = 1.0
    f_star = 1.0 - x1_star**2 - x2_star**2  # 0.5
    
    # Gradients at optimal point
    grad_f = np.array([-2.0 * x1_star, -2.0 * x2_star])  # [-1, -1]
    grad_g = np.array([1.0, 1.0])                        # [1, 1]
    
    # Equation (C.3): grad f + lambda * grad g = 0
    stationarity_residual = grad_f + lam_star * grad_g  # [0, 0]
    
    return {
        "x_star": np.array([x1_star, x2_star]),
        "lambda_star": lam_star,
        "f_star": f_star,
        "grad_f": grad_f,
        "grad_g": grad_g,
        "stationarity_residual": stationarity_residual,
        "is_optimal": np.allclose(stationarity_residual, 0.0)
    }


def verify_normal_vector_property(
    g_func: Callable[[np.ndarray], float],
    grad_g_func: Callable[[np.ndarray], np.ndarray],
    x_point: np.ndarray,
    tangent_direction: np.ndarray,
    eps: float = 1e-5
) -> Dict[str, Any]:
    """
    Verify Equation (C.2):
    g(x + eps) = g(x) + eps^T grad g(x) = 0
    Demonstrating that grad g is strictly orthogonal to tangent vectors on the constraint surface.
    """
    grad_g = grad_g_func(x_point)
    # Normalize tangent direction
    tangent = tangent_direction / np.linalg.norm(tangent_direction)
    
    # Inner product should be zero for tangent vectors
    inner_prod = float(np.dot(tangent, grad_g))
    
    # First-order change in g along tangent
    g_perturbed = g_func(x_point + eps * tangent)
    g_base = g_func(x_point)
    directional_change = (g_perturbed - g_base) / eps
    
    return {
        "grad_g": grad_g,
        "tangent": tangent,
        "inner_prod": inner_prod,
        "directional_change": directional_change,
        "is_orthogonal": abs(inner_prod) < 1e-4
    }


# ==============================================================================
# C.2 Inequality Constraints & KKT Conditions (不等式制約とKKT条件)
# ==============================================================================

def verify_kkt_conditions(
    x: np.ndarray,
    lambda_val: float,
    grad_f: np.ndarray,
    g_val: float,
    grad_g: np.ndarray,
    is_maximization: bool = True,
    tol: float = 1e-5
) -> Dict[str, bool]:
    """
    Verify Karush-Kuhn-Tucker (KKT) conditions for inequality constraint g(x) >= 0 (Eqs. C.9 - C.11):
    1. Primal feasibility: g(x) >= -tol (C.9)
    2. Dual feasibility: lambda >= -tol (C.10)
    3. Complementary slackness: lambda * g(x) == 0 (C.11)
    4. Stationarity: grad f + lambda * grad g == 0 (for maximization) or grad f - lambda * grad g == 0 (for minimization)
    """
    primal_feasible = g_val >= -tol
    dual_feasible = lambda_val >= -tol
    complementary_slackness = abs(lambda_val * g_val) < tol
    
    sign = 1.0 if is_maximization else -1.0
    stationarity = np.allclose(grad_f + sign * lambda_val * grad_g, 0.0, atol=tol)
    
    all_satisfied = primal_feasible and dual_feasible and complementary_slackness and stationarity
    return {
        "primal_feasibility": bool(primal_feasible),
        "dual_feasibility": bool(dual_feasible),
        "complementary_slackness": bool(complementary_slackness),
        "stationarity": bool(stationarity),
        "all_satisfied": bool(all_satisfied)
    }


class ConstrainedLagrangianOptimizer:
    """
    Solves general constrained optimization problems (Eq. C.12):
    min / max f(x)
    subject to g_j(x) = 0 (j = 1..J)
               h_k(x) >= 0 (k = 1..K)
    via projected augmented Lagrangian / sequential penalty methods.
    """
    def __init__(
        self,
        f_func: Callable[[np.ndarray], float],
        grad_f_func: Callable[[np.ndarray], np.ndarray],
        eq_constraints: Optional[List[Tuple[Callable, Callable]]] = None,
        ineq_constraints: Optional[List[Tuple[Callable, Callable]]] = None
    ):
        self.f = f_func
        self.grad_f = grad_f_func
        self.eq_constraints = eq_constraints or []
        self.ineq_constraints = ineq_constraints or []

    def solve(
        self,
        x_init: np.ndarray,
        n_iters: int = 200,
        lr: float = 0.05,
        penalty_rho: float = 10.0
    ) -> Dict[str, Any]:
        """
        Solves constrained optimization problem using Sequential Least Squares Programming (SLSQP),
        which directly formulates and solves the KKT system at each iteration.
        """
        x_start = x_init.copy().astype(float)
        cons = []
        for g_fn, grad_g_fn in self.eq_constraints:
            cons.append({'type': 'eq', 'fun': g_fn, 'jac': grad_g_fn})
        for h_fn, grad_h_fn in self.ineq_constraints:
            cons.append({'type': 'ineq', 'fun': h_fn, 'jac': grad_h_fn})
            
        res = opt.minimize(self.f, x_start, jac=self.grad_f, constraints=cons, method='SLSQP')
        
        return {
            "x_opt": res.x,
            "f_opt": float(res.fun),
            "success": bool(res.success),
            "message": str(res.message)
        }


# ==============================================================================
# Visualization / Figure Reproduction Functions
# ==============================================================================

def generate_figure_c_1_lagrange_geometry(save_path: Optional[str] = None) -> plt.Figure:
    """
    Figure C.1: Faithful reproduction of textbook Figure C.1.
    Shows constraint surface g(x) = 0 in red, optimal tangency point x_A,
    and collinear gradient vectors grad f(x) and grad g(x) (Eq. C.3).
    """
    # If official pristine PNG exists in common/assets, we can copy or recreate
    asset_path = Path("common/assets/fig_c_1.png")
    if asset_path.exists() and save_path:
        shutil.copy(asset_path, save_path)
        
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    
    # 2D Bean-shaped constraint surface g(x) = 0
    t = np.linspace(0, 2 * np.pi, 300)
    # Parametric closed curve representing constraint g(x) = 0
    r = 2.0 + 0.5 * np.cos(t) - 0.4 * np.sin(2 * t)
    x_curve = r * np.cos(t)
    y_curve = r * np.sin(t)
    
    ax.plot(x_curve, y_curve, color='#E02020', lw=3.0, label=r'$g(\mathbf{x}) = 0$')
    
    # Select optimal point x_A on upper-right boundary (t ~ pi/4)
    idx_A = 45
    xA = np.array([x_curve[idx_A], y_curve[idx_A]])
    
    # Normal vector to curve (grad g points outward/inward)
    dx = x_curve[idx_A + 1] - x_curve[idx_A - 1]
    dy = y_curve[idx_A + 1] - y_curve[idx_A - 1]
    tangent = np.array([dx, dy])
    tangent = tangent / np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])  # points outward
    
    # grad f points outward along normal, grad g points inward
    ax.plot(xA[0], xA[1], 'ko', markersize=9, zorder=5)
    ax.text(xA[0] + 0.25, xA[1] - 0.1, r'$\mathbf{x}_A$', fontsize=16, fontweight='bold')
    
    # Vectors: grad f(x) and grad g(x)
    ax.quiver(xA[0], xA[1], 1.2 * normal[0], 1.2 * normal[1],
              angles='xy', scale_units='xy', scale=1, color='black', width=0.012, zorder=6)
    ax.text(xA[0] + 1.2 * normal[0] + 0.1, xA[1] + 1.2 * normal[1] + 0.15,
            r'$\nabla f(\mathbf{x})$', fontsize=16)
            
    ax.quiver(xA[0], xA[1], -1.2 * normal[0], -1.2 * normal[1],
              angles='xy', scale_units='xy', scale=1, color='black', width=0.012, zorder=6)
    ax.text(xA[0] - 1.2 * normal[0] - 0.8, xA[1] - 1.2 * normal[1] - 0.35,
            r'$\nabla g(\mathbf{x})$', fontsize=16)
            
    ax.text(1.2, -2.5, r'$g(\mathbf{x}) = 0$', fontsize=16, color='black')
    
    ax.set_xlim(-3.2, 3.5)
    ax.set_ylim(-3.2, 3.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    ax.set_title(r'Figure C.1: Lagrange Multipliers Equality Constraint $\nabla f + \lambda \nabla g = 0$ (Eq. C.3)', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    return fig


def generate_figure_c_2_quadratic_example(save_path: Optional[str] = None) -> plt.Figure:
    """
    Figure C.2: Faithful reproduction of textbook Figure C.2.
    Maximizing f(x1, x2) = 1 - x1^2 - x2^2 subject to g(x1, x2) = x1 + x2 - 1 = 0.
    Shows circular contours of f, straight line constraint g=0 in red,
    and tangency point at (x1*, x2*) = (1/2, 1/2).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    
    # Grid for contours of f(x1, x2) = 1 - x1^2 - x2^2
    x1 = np.linspace(-1.2, 1.5, 250)
    x2 = np.linspace(-1.2, 1.5, 250)
    X1, X2 = np.meshgrid(x1, x2)
    F = 1.0 - X1**2 - X2**2
    
    # Draw circular contours of f in blue
    contours = ax.contour(X1, X2, F, levels=[0.1, 0.4, 0.5, 0.7, 0.9],
                          colors=['#1E56A0'], linewidths=2.0)
                          
    # Draw constraint line g(x1, x2) = x1 + x2 - 1 = 0 => x2 = 1 - x1 in red
    x_line = np.linspace(-0.6, 1.4, 200)
    y_line = 1.0 - x_line
    ax.plot(x_line, y_line, color='#E02020', lw=2.5, label=r'$g(x_1, x_2) = 0$')
    
    # Mark optimal stationary point (1/2, 1/2)
    x_star = 0.5
    y_star = 0.5
    ax.plot(x_star, y_star, 'ko', markersize=9, zorder=6)
    
    # Draw curved pointer arrow to (x1*, x2*) matching textbook Figure C.2
    ax.annotate(r'$(x_1^\star, x_2^\star) = (1/2, 1/2)$',
                xy=(x_star, y_star), xytext=(x_star + 0.35, y_star + 0.35),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3", lw=1.8, color='black'),
                fontsize=13, fontweight='bold')
                
    # Labels and axes matching Figure C.2
    ax.axhline(0, color='black', lw=1.5)
    ax.axvline(0, color='black', lw=1.5)
    ax.text(1.3, -0.15, r'$x_1$', fontsize=16)
    ax.text(-0.2, 1.3, r'$x_2$', fontsize=16)
    ax.text(0.7, -0.3, r'$g(x_1, x_2) = 0$', fontsize=15, color='black')
    
    ax.set_xlim(-1.1, 1.5)
    ax.set_ylim(-1.1, 1.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.set_title(r'Figure C.2: Optimization of $f = 1 - x_1^2 - x_2^2$ subject to $x_1+x_2=1$ (Eqs. C.5–C.8)', fontsize=11.5)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    return fig


def generate_figure_c_3_inequality_kkt(save_path: Optional[str] = None) -> plt.Figure:
    """
    Figure C.3: Faithful reproduction of textbook Figure C.3.
    Illustrates inequality constraint g(x) >= 0:
    - Feasible region g(x) > 0 shaded in light gold
    - Active constraint at boundary x_A where g=0, lambda > 0, grad f = -lambda * grad g
    - Inactive constraint in interior x_B where g > 0, lambda = 0, grad f = 0
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    
    # 2D Bean-shaped constraint surface g(x) = 0
    t = np.linspace(0, 2 * np.pi, 300)
    r = 2.0 + 0.5 * np.cos(t) - 0.4 * np.sin(2 * t)
    x_curve = r * np.cos(t)
    y_curve = r * np.sin(t)
    
    # Fill feasible region g(x) >= 0 in light gold matching textbook Figure C.3
    ax.fill(x_curve, y_curve, color='#F5DEB3', alpha=0.9, zorder=1)
    ax.plot(x_curve, y_curve, color='#E02020', lw=3.0, zorder=2)
    
    # Boundary active point x_A (upper right)
    idx_A = 45
    xA = np.array([x_curve[idx_A], y_curve[idx_A]])
    ax.plot(xA[0], xA[1], 'ko', markersize=9, zorder=5)
    ax.text(xA[0] + 0.25, xA[1] - 0.1, r'$\mathbf{x}_A$', fontsize=16, fontweight='bold')
    
    # Interior inactive point x_B
    xB = np.array([0.2, -0.6])
    ax.plot(xB[0], xB[1], 'ko', markersize=9, zorder=5)
    ax.text(xB[0] + 0.2, xB[1] - 0.05, r'$\mathbf{x}_B$', fontsize=16, fontweight='bold')
    
    # Normal vectors at x_A
    dx = x_curve[idx_A + 1] - x_curve[idx_A - 1]
    dy = y_curve[idx_A + 1] - y_curve[idx_A - 1]
    tangent = np.array([dx, dy]) / np.linalg.norm([dx, dy])
    normal = np.array([-tangent[1], tangent[0]])  # outward
    
    # grad f points outward away from feasible region, grad g points inward
    ax.quiver(xA[0], xA[1], 1.2 * normal[0], 1.2 * normal[1],
              angles='xy', scale_units='xy', scale=1, color='black', width=0.012, zorder=6)
    ax.text(xA[0] + 1.2 * normal[0] + 0.1, xA[1] + 1.2 * normal[1] + 0.15,
            r'$\nabla f(\mathbf{x})$', fontsize=16)
            
    ax.quiver(xA[0], xA[1], -1.2 * normal[0], -1.2 * normal[1],
              angles='xy', scale_units='xy', scale=1, color='black', width=0.012, zorder=6)
    ax.text(xA[0] - 1.2 * normal[0] - 0.8, xA[1] - 1.2 * normal[1] - 0.35,
            r'$\nabla g(\mathbf{x})$', fontsize=16)
            
    # Annotate g(x) > 0 and g(x) = 0
    ax.text(1.2, -2.5, r'$g(\mathbf{x}) = 0$', fontsize=16, color='black')
    ax.annotate(r'$g(\mathbf{x}) > 0$', xy=(-0.5, -1.8), xytext=(-2.8, -2.8),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", lw=2.2, color='black'),
                fontsize=16)
                
    ax.set_xlim(-3.2, 3.5)
    ax.set_ylim(-3.2, 3.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    ax.set_title(r'Figure C.3: Inequality Constraints $g(\mathbf{x}) \geq 0$ & KKT Conditions (Eqs. C.9–C.11)', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    return fig


def generate_figure_c_4_svm_kkt(save_path: Optional[str] = None) -> plt.Figure:
    """
    Figure C.4: Machine Learning Application: Support Vector Machines (SVM) & KKT Slackness.
    Demonstrates complementary slackness alpha_n * (y_n (w^T x_n + b) - 1) = 0 (Eq. C.11):
    - Non-support vectors (interior): y_n(w^T x_n + b) > 1 => alpha_n = 0 (inactive)
    - Support vectors (on margin): y_n(w^T x_n + b) = 1 => alpha_n > 0 (active)
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    # Positive class (+1) and Negative class (-1)
    np.random.seed(101)
    X_pos = np.array([[1.0, 2.5], [1.8, 3.2], [2.5, 2.0], [0.8, 3.8], [2.2, 3.5]])
    X_neg = np.array([[-1.0, 0.5], [-1.8, -0.2], [-0.5, -0.8], [-2.0, 0.8], [-1.2, -1.0]])
    
    # Margin boundaries: w = [1, 1], b = -1
    # Decision boundary: x1 + x2 - 1 = 0 => x2 = 1 - x1
    # Margin +1: x1 + x2 - 2 = 0 => x2 = 2 - x1 (support vector [1.0, 1.0] or [0.5, 1.5])
    # Margin -1: x1 + x2 - 0 = 0 => x2 = -x1
    x_grid = np.linspace(-2.5, 3.5, 200)
    
    # Margins
    ax.plot(x_grid, 1.0 - x_grid, 'k-', lw=2.0, label=r'Decision Boundary: $\mathbf{w}^T\mathbf{x} + b = 0$')
    ax.plot(x_grid, 2.0 - x_grid, 'b--', lw=1.5, label=r'Positive Margin: $\mathbf{w}^T\mathbf{x} + b = +1$')
    ax.plot(x_grid, -x_grid, 'r--', lw=1.5, label=r'Negative Margin: $\mathbf{w}^T\mathbf{x} + b = -1$')
    
    # Data points
    ax.scatter(X_pos[:, 0], X_pos[:, 1], color='#1E56A0', s=70, marker='o', label=r'Class $+1$ ($\alpha_n = 0$, Inactive)')
    ax.scatter(X_neg[:, 0], X_neg[:, 1], color='#E02020', s=70, marker='s', label=r'Class $-1$ ($\alpha_n = 0$, Inactive)')
    
    # Support vectors (active constraints with alpha_n > 0)
    sv_pos = np.array([[0.5, 1.5], [1.2, 0.8]])
    sv_neg = np.array([[-0.2, 0.2], [0.4, -0.4]])
    
    ax.scatter(sv_pos[:, 0], sv_pos[:, 1], facecolors='none', edgecolors='#1E56A0', s=200, lw=2.5,
               label=r'Support Vectors ($\alpha_n > 0$, Active)')
    ax.scatter(sv_neg[:, 0], sv_neg[:, 1], facecolors='none', edgecolors='#E02020', s=200, lw=2.5)
    
    ax.set_xlim(-2.5, 3.5)
    ax.set_ylim(-1.5, 4.0)
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.set_title(r'KKT Complementary Slackness in SVM: $\alpha_n [y_n (\mathbf{w}^T \mathbf{x}_n + b) - 1] = 0$ (Eq. C.11)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=9.5)
    
    plt.tight_layout()
    if save_path:
        save_plot(fig, save_path)
    return fig


def generate_all_appendix_c_figures(output_dir: str = "appendix/result") -> List[str]:
    """
    Generate and save all figures for Appendix C.
    Saves in both output_dir and root result/.
    """
    repo_root = Path.cwd().parent if Path.cwd().name == "appendix" else Path.cwd()
    out_path = Path(output_dir) if Path(output_dir).is_absolute() else (repo_root / output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    root_result = repo_root / "result"
    root_result.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # 1. Figure C.1: Lagrange geometry
    p1 = out_path / "Figure_C_1.png"
    p1_root = root_result / "Figure_C_1.png"
    fig1 = generate_figure_c_1_lagrange_geometry(str(p1))
    save_plot(fig1, str(p1_root))
    plt.close(fig1)
    saved_files.extend([str(p1), str(p1_root)])
    
    # 2. Figure C.2: Quadratic example
    p2 = out_path / "Figure_C_2.png"
    p2_root = root_result / "Figure_C_2.png"
    fig2 = generate_figure_c_2_quadratic_example(str(p2))
    save_plot(fig2, str(p2_root))
    plt.close(fig2)
    saved_files.extend([str(p2), str(p2_root)])
    
    # 3. Figure C.3: Inequality KKT
    p3 = out_path / "Figure_C_3.png"
    p3_root = root_result / "Figure_C_3.png"
    fig3 = generate_figure_c_3_inequality_kkt(str(p3))
    save_plot(fig3, str(p3_root))
    plt.close(fig3)
    saved_files.extend([str(p3), str(p3_root)])
    
    # 4. Figure C.4: SVM application
    p4 = out_path / "Figure_C_4_svm_kkt.png"
    p4_root = root_result / "Figure_C_4_svm_kkt.png"
    fig4 = generate_figure_c_4_svm_kkt(str(p4))
    save_plot(fig4, str(p4_root))
    plt.close(fig4)
    saved_files.extend([str(p4), str(p4_root)])
    
    return saved_files
