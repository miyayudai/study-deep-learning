"""
Bishop & Bishop (2024) Appendix B: Calculus of Variations (付録B: 変分法)
Comprehensive implementation of functionals, functional derivatives,
Euler-Lagrange equations, shortest path geodesics, and maximum entropy principles.

Equations:
- (B.1) - (B.2): Ordinary and partial derivative Taylor expansions
- (B.3) - (B.4): Functional derivative definition and stationarity condition
- (B.5) - (B.7): Variation of integral functionals and integration by parts
- (B.8): Euler-Lagrange equation
- (B.9) - (B.10): Example functional G = y^2 + (y')^2 and differential equation y - y'' = 0
"""

from typing import Tuple, List, Dict, Any, Optional, Callable
import shutil
import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from pathlib import Path

from common.plot_utils import setup_style, save_plot


# ==============================================================================
# B.1 Functionals and Functional Derivatives (汎関数と変分導関数)
# ==============================================================================

def functional_derivative_finite_difference(
    F_functional: Callable[[np.ndarray], float],
    y_base: np.ndarray,
    eta: np.ndarray,
    dx: float,
    eps: float = 1e-5
) -> Dict[str, Any]:
    """
    Verify Equation (B.3):
    F[y(x) + eps * eta(x)] = F[y(x)] + eps * int (delta F / delta y(x)) eta(x) dx + O(eps^2)
    
    Computes numerical directional derivative of functional F along perturbation eta(x).
    """
    F0 = F_functional(y_base)
    F_plus = F_functional(y_base + eps * eta)
    F_minus = F_functional(y_base - eps * eta)
    
    # Numerical gateaux derivative: dF/deps |_{eps=0}
    dF_deps_num = (F_plus - F_minus) / (2.0 * eps)
    
    return {
        "F0": F0,
        "F_plus": F_plus,
        "F_minus": F_minus,
        "numerical_directional_derivative": dF_deps_num
    }


# ==============================================================================
# B.2 Euler-Lagrange Differential Equations (オイラー・ラグランジュ方程式)
# ==============================================================================

def solve_euler_lagrange_example_b10(
    x: np.ndarray,
    y0: float = 1.0,
    y1: float = 2.0
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Exact analytical solution to Equation (B.10):
    y(x) - y''(x) = 0
    with boundary conditions y(x[0]) = y0, y(x[-1]) = y1.
    
    General solution:
    y(x) = C_1 * exp(x) + C_2 * exp(-x)
    """
    x0, x1 = x[0], x[-1]
    
    # Linear system for coefficients [C1, C2]:
    # C1 * exp(x0) + C2 * exp(-x0) = y0
    # C1 * exp(x1) + C2 * exp(-x1) = y1
    M = np.array([
        [np.exp(x0), np.exp(-x0)],
        [np.exp(x1), np.exp(-x1)]
    ])
    rhs = np.array([y0, y1])
    C1, C2 = np.linalg.solve(M, rhs)
    
    y_exact = C1 * np.exp(x) + C2 * np.exp(-x)
    return y_exact, (float(C1), float(C2))


def trapz_compat(y: np.ndarray, dx: float) -> float:
    """Trapezoidal integration compatible with both NumPy 1.x and 2.x."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, dx=dx))
    elif hasattr(np, "trapz"):
        return float(np.trapz(y, dx=dx))
    return float(integrate.trapezoid(y, dx=dx))


def evaluate_example_functional_b9(y: np.ndarray, dx: float) -> float:
    """
    Evaluate functional F[y] = int (y(x)^2 + (y'(x))^2) dx (Eq. B.9)
    using trapezoidal numerical integration.
    """
    dy_dx = np.gradient(y, dx)
    integrand = y**2 + dy_dx**2
    return trapz_compat(integrand, dx=dx)


def optimize_functional_discrete(
    x: np.ndarray,
    y0: float,
    y1: float,
    n_steps: int = 2500,
    lr: float = 1.0
) -> Tuple[np.ndarray, List[float]]:
    """
    Numerically optimize functional F[y] = int (y^2 + (y')^2) dx (Eq. B.9)
    by performing coordinate relaxation on the interior values of y(x).
    Setting dF/dy_i = 0 gives the stationary condition:
    y_i = (y_{i+1} + y_{i-1}) / (2 + dx^2)
    Demonstrating that discrete relaxation converges precisely to the
    analytical Euler-Lagrange solution y - y'' = 0 (Eq. B.10).
    """
    N = len(x)
    dx = x[1] - x[0]
    
    # Initialize with a straight line between (x[0], y0) and (x[-1], y1)
    y = np.linspace(y0, y1, N)
    
    loss_history = []
    denom = 2.0 + dx**2
    for step in range(n_steps):
        # Update interior points towards stationary point
        y_target = (y[2:] + y[:-2]) / denom
        y[1:-1] = (1.0 - lr) * y[1:-1] + lr * y_target
        
        if step % 50 == 0 or step == n_steps - 1:
            loss = evaluate_example_functional_b9(y, dx)
            loss_history.append(loss)
            
    return y, loss_history


# ==============================================================================
# B.3 Shortest Path Geodesic Problem (最短経路問題)
# ==============================================================================

def compute_curve_length(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute arc length of 2D curve (x, y):
    L[y] = int sqrt(1 + (y'(x))^2) dx
    """
    dx = x[1] - x[0]
    dy_dx = np.gradient(y, dx)
    integrand = np.sqrt(1.0 + dy_dx**2)
    return trapz_compat(integrand, dx=dx)


def generate_perturbed_curves(
    x: np.ndarray,
    y_straight: np.ndarray,
    amplitudes: List[float]
) -> List[Tuple[float, np.ndarray, float]]:
    """
    Generate curves y(x) = y_straight(x) + amp * sin(pi * (x - x0) / (x1 - x0))
    which satisfy boundary conditions (amp * sin = 0 at boundaries).
    Evaluates their arc lengths, confirming that amp=0 (straight line) minimizes length.
    """
    x0, x1 = x[0], x[-1]
    results = []
    for amp in amplitudes:
        perturbation = amp * np.sin(np.pi * (x - x0) / (x1 - x0))
        y_perturbed = y_straight + perturbation
        length = compute_curve_length(x, y_perturbed)
        results.append((amp, y_perturbed, length))
    return results


# ==============================================================================
# B.4 Maximum Entropy Principle (最大エントロピー原理)
# ==============================================================================

def compare_maximum_entropy_distributions(variance: float = 1.0) -> Dict[str, Any]:
    """
    Demonstrate that under fixed variance sigma^2, the distribution maximizing
    continuous differential entropy H[p] = - int p(x) ln p(x) dx is the Gaussian.
    
    Theoretical continuous differential entropies:
    - Gaussian: H = 0.5 * ln(2 * pi * e * sigma^2)
    - Uniform [-a, a] (var = a^2 / 3 = sigma^2 -> a = sqrt(3)*sigma): H = ln(2*a) = ln(2 * sqrt(3) * sigma)
    - Laplace (var = 2 * b^2 = sigma^2 -> b = sigma / sqrt(2)): H = ln(2 * e * b) = ln(sqrt(2) * e * sigma)
    - Triangular [-a, a] (var = a^2 / 6 = sigma^2 -> a = sqrt(6)*sigma): H = ln(a) + 0.5 = ln(sqrt(6)*sigma) + 0.5
    """
    sigma = np.sqrt(variance)
    e = np.e
    
    # 1. Gaussian: N(0, sigma^2)
    h_gauss = 0.5 * np.log(2 * np.pi * e * variance)
    
    # 2. Uniform with variance = sigma^2
    # Var(Uniform[-a, a]) = (2a)^2 / 12 = a^2 / 3 = sigma^2 => a = sqrt(3) * sigma
    a_unif = np.sqrt(3.0) * sigma
    h_unif = np.log(2.0 * a_unif)
    
    # 3. Laplace with variance = sigma^2
    # Var(Laplace(b)) = 2 * b^2 = sigma^2 => b = sigma / sqrt(2)
    b_laplace = sigma / np.sqrt(2.0)
    h_laplace = np.log(2.0 * e * b_laplace)
    
    # 4. Triangular with variance = sigma^2
    # Var(Triangular[-a, a]) = a^2 / 6 = sigma^2 => a = sqrt(6) * sigma
    a_tri = np.sqrt(6.0) * sigma
    h_tri = np.log(a_tri) + 0.5
    
    results = {
        "Gaussian": float(h_gauss),
        "Laplace": float(h_laplace),
        "Uniform": float(h_unif),
        "Triangular": float(h_tri)
    }
    
    # Verify Gaussian is strictly the maximum
    max_dist = max(results, key=results.get)
    return {
        "entropies": results,
        "is_gaussian_maximum": max_dist == "Gaussian",
        "margin_over_second": float(h_gauss - sorted(results.values())[-2])
    }


# ==============================================================================
# Visualization / Figure Reproduction Functions
# ==============================================================================

def generate_figure_b_1_functional_derivative(save_path: Optional[str] = None) -> plt.Figure:
    """
    Figure B.1: Faithful reproduction of textbook Figure B.1.
    Shows function y(x) in red and perturbed function y(x) + eps * eta(x) in blue,
    illustrating how a variation eps * eta(x) induces functional derivative delta F / delta y(x).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    x = np.linspace(0.5, 9.5, 300)
    
    # Red curve: y(x)
    # A gentle S-like sigmoid curve similar to textbook Figure B.1
    y_base = 1.0 + 3.0 / (1.0 + np.exp(-0.8 * (x - 5.0)))
    
    # Blue curve: y(x) + eps * eta(x)
    # Crosses the red curve in the middle
    y_perturbed = 0.5 + 4.0 / (1.0 + np.exp(-1.4 * (x - 4.5)))
    
    ax.plot(x, y_base, color='#E02020', lw=2.5, label=r'$y(x)$')
    ax.plot(x, y_perturbed, color='#1E56A0', lw=2.5, label=r'$y(x) + \epsilon \eta(x)$')
    
    # Annotate labels matching textbook Figure B.1
    ax.text(1.5, 2.5, r'$y(x)$', fontsize=16, color='black')
    ax.text(3.5, 1.1, r'$y(x) + \epsilon \eta(x)$', fontsize=16, color='black')
    
    # Configure axes with arrows matching textbook style
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_xlabel(r'$x$', fontsize=14, loc='right')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    
    ax.set_title(r'Figure B.1: Variation of a Function $y(x) \to y(x) + \epsilon \eta(x)$ (Eq. B.3)', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    return fig


def generate_figure_b_2_euler_lagrange_solution(save_path: Optional[str] = None) -> plt.Figure:
    """
    Figure B.2: Euler-Lagrange analytical solution vs numerical discrete optimization
    for functional G = y^2 + (y')^2 (Eqs. B.9 - B.10).
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    
    x = np.linspace(0.0, 1.0, 50)
    y0, y1 = 1.0, 2.5
    
    # 1. Exact analytical solution (Eq. B.10)
    y_exact, (C1, C2) = solve_euler_lagrange_example_b10(x, y0, y1)
    
    # 2. Numerical gradient descent on functional
    y_opt, loss_hist = optimize_functional_discrete(x, y0, y1, n_steps=2000, lr=0.04)
    
    # Initial straight line
    y_init = np.linspace(y0, y1, len(x))
    
    # Left subplot: Function curves
    ax1 = axes[0]
    ax1.plot(x, y_init, 'k--', lw=1.5, alpha=0.7, label=r'Initial $y_{\mathrm{init}}(x)$')
    el_label = r'Analytical E-L: $y - y^{\prime\prime} = 0$' + f'\n$({C1:.2f}e^x + {C2:.2f}e^{{-x}})$'
    ax1.plot(x, y_exact, color='#E02020', lw=2.2, label=el_label)
    ax1.set_xlabel(r'$x$', fontsize=12)
    ax1.set_ylabel(r'$y(x)$', fontsize=12)
    ax1.set_title(r'Stationary Path for $\mathcal{F}[y] = \int (y^2 + (y^\prime)^2) dx$ (Eq. B.9)', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=10)
    
    # Right subplot: Optimization loss convergence
    ax2 = axes[1]
    steps = np.arange(len(loss_hist)) * 50
    f_exact = evaluate_example_functional_b9(y_exact, x[1] - x[0])
    ax2.plot(steps, loss_hist, color='#2E7D32', lw=2.0, label=r'Numerical $\mathcal{F}[y_k]$')
    ax2.axhline(f_exact, color='#E02020', ls='--', lw=1.8, label=rf'Exact E-L Minimum: $\mathcal{{F}}^* = {f_exact:.4f}$')
    ax2.set_xlabel('Iteration Step', fontsize=12)
    ax2.set_ylabel(r'Functional Value $\mathcal{F}[y]$', fontsize=12)
    ax2.set_title(r'Convergence of Functional to Minimum (Eq. B.10)', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=10)
    
    fig.suptitle(r'Appendix B: Euler–Lagrange Differential Equation Verification (Eqs. B.8–B.10)', fontsize=13)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    return fig


def generate_figure_b_3_maximum_entropy(save_path: Optional[str] = None) -> plt.Figure:
    """
    Figure B.3: Principle of Maximum Entropy.
    Under fixed variance sigma^2, the Gaussian distribution strictly maximizes
    differential entropy H[p] = - int p(x) ln p(x) dx.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    
    var = 1.0
    sigma = np.sqrt(var)
    res = compare_maximum_entropy_distributions(variance=var)
    entropies = res["entropies"]
    
    # Subplot 1: Probability density functions with matching variance = 1
    ax1 = axes[0]
    x = np.linspace(-3.5, 3.5, 400)
    
    # Gaussian
    p_gauss = stats.norm.pdf(x, 0, sigma)
    ax1.plot(x, p_gauss, color='#E02020', lw=2.5, label=rf'Gaussian $\mathcal{{N}}(0, 1)$ ($H={entropies["Gaussian"]:.3f}$)')
    
    # Laplace
    b_lap = sigma / np.sqrt(2.0)
    p_lap = stats.laplace.pdf(x, loc=0, scale=b_lap)
    ax1.plot(x, p_lap, color='#1E56A0', lw=2.0, label=rf'Laplace ($H={entropies["Laplace"]:.3f}$)')
    
    # Uniform
    a_unif = np.sqrt(3.0) * sigma
    p_unif = stats.uniform.pdf(x, loc=-a_unif, scale=2*a_unif)
    ax1.plot(x, p_unif, color='#2E7D32', lw=2.0, ls='-.', label=rf'Uniform ($H={entropies["Uniform"]:.3f}$)')
    
    ax1.set_xlabel(r'$x$', fontsize=12)
    ax1.set_ylabel(r'$p(x)$', fontsize=12)
    ax1.set_title(r'Distributions with Fixed Variance $\sigma^2 = 1$', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=9.5)
    
    # Subplot 2: Bar chart of differential entropies
    ax2 = axes[1]
    names = list(entropies.keys())
    values = [entropies[k] for k in names]
    colors = ['#E02020', '#1E56A0', '#2E7D32', '#F57C00']
    
    bars = ax2.bar(names, values, color=colors, alpha=0.85, edgecolor='black')
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
                 
    ax2.set_ylabel(r'Differential Entropy $\mathrm{H}[p]$ [nats]', fontsize=12)
    ax2.set_ylim(0, 1.7)
    ax2.set_title(r'Entropy Comparison: Gaussian Maximizes $\mathrm{H}[p]$', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6, axis='y')
    
    fig.suptitle(r'Appendix B: Variational Optimization & Maximum Entropy Principle', fontsize=13)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    return fig


def generate_figure_b_4_shortest_path(save_path: Optional[str] = None) -> plt.Figure:
    """
    Figure B.4: Geodesic Shortest Path Problem.
    Euler-Lagrange equation for curve length L[y] = int sqrt(1 + (y')^2) dx gives
    y''(x) = 0 => straight line y(x) = m*x + c.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    
    x = np.linspace(0.0, 3.0, 300)
    y0, y1 = 0.5, 2.0
    y_straight = np.linspace(y0, y1, len(x))
    l_straight = compute_curve_length(x, y_straight)
    
    # Perturbations
    amplitudes = [-0.8, -0.4, 0.0, 0.4, 0.8]
    colors = ['#7B1FA2', '#1E56A0', '#E02020', '#2E7D32', '#F57C00']
    
    ax1 = axes[0]
    amp_evals = np.linspace(-1.2, 1.2, 40)
    res_curves = generate_perturbed_curves(x, y_straight, list(amp_evals))
    
    for amp, col in zip(amplitudes, colors):
        label = rf'Straight Line: $L = {l_straight:.3f}$' if amp == 0 else rf'Perturbed ($\alpha={amp:+.1f}$)'
        ls = '-' if amp == 0 else '--'
        lw = 2.5 if amp == 0 else 1.5
        _, y_p, _ = generate_perturbed_curves(x, y_straight, [amp])[0]
        ax1.plot(x, y_p, color=col, lw=lw, ls=ls, label=label)
        
    ax1.scatter([x[0], x[-1]], [y0, y1], color='black', s=50, zorder=5)
    ax1.set_xlabel(r'$x$', fontsize=12)
    ax1.set_ylabel(r'$y(x)$', fontsize=12)
    ax1.set_title(r'Paths Connecting Fixed Points $(x_0, y_0)$ to $(x_1, y_1)$', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=9.5)
    
    # Right subplot: Length vs amplitude (quadratic bowl)
    ax2 = axes[1]
    amps = [r[0] for r in res_curves]
    lengths = [r[2] for r in res_curves]
    
    ax2.plot(amps, lengths, 'o-', color='#1E56A0', lw=2.0)
    ax2.axvline(0, color='#E02020', ls='--', lw=1.8, label=rf'Minimum at $\alpha=0$ ($L^*={l_straight:.3f}$)')
    ax2.scatter([0], [l_straight], color='#E02020', s=70, zorder=5)
    
    ax2.set_xlabel(r'Perturbation Amplitude $\alpha$', fontsize=12)
    ax2.set_ylabel(r'Path Arc Length $L[y]$', fontsize=12)
    ax2.set_title(r'Path Length Functional $\mathcal{L}[y] = \int \sqrt{1 + (y^\prime)^2} dx$', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper center', fontsize=10)
    
    fig.suptitle(r'Appendix B: Geodesic Shortest Path from Euler–Lagrange Equation (Eq. B.8)', fontsize=13)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    return fig


def generate_all_appendix_b_figures(output_dir: str = "appendix/result") -> List[str]:
    """
    Generate and save all figures for Appendix B.
    Saves in both output_dir and root result/.
    """
    repo_root = Path.cwd().parent if Path.cwd().name == "appendix" else Path.cwd()
    out_path = Path(output_dir) if Path(output_dir).is_absolute() else (repo_root / output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    root_result = repo_root / "result"
    root_result.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # 1. Figure B.1: Functional derivative
    p1 = out_path / "figB_1_functional_derivative.png"
    p1_root = root_result / "figB_1_functional_derivative.png"
    fig1 = generate_figure_b_1_functional_derivative(str(p1))
    save_plot(fig1, str(p1_root))
    plt.close(fig1)
    saved_files.extend([str(p1), str(p1_root)])
    
    # 2. Figure B.2: Euler-Lagrange solution
    p2 = out_path / "figB_2_euler_lagrange_solution.png"
    p2_root = root_result / "figB_2_euler_lagrange_solution.png"
    fig2 = generate_figure_b_2_euler_lagrange_solution(str(p2))
    save_plot(fig2, str(p2_root))
    plt.close(fig2)
    saved_files.extend([str(p2), str(p2_root)])
    
    # 3. Figure B.3: Maximum entropy
    p3 = out_path / "figB_3_maximum_entropy.png"
    p3_root = root_result / "figB_3_maximum_entropy.png"
    fig3 = generate_figure_b_3_maximum_entropy(str(p3))
    save_plot(fig3, str(p3_root))
    plt.close(fig3)
    saved_files.extend([str(p3), str(p3_root)])
    
    # 4. Figure B.4: Shortest path
    p4 = out_path / "figB_4_shortest_path.png"
    p4_root = root_result / "figB_4_shortest_path.png"
    fig4 = generate_figure_b_4_shortest_path(str(p4))
    save_plot(fig4, str(p4_root))
    plt.close(fig4)
    saved_files.extend([str(p4), str(p4_root)])
    
    return saved_files
