"""
Decision Theory for Regression (Chapter 4: Single-layer Networks: Regression, Section 4.2).

Covers:
- Expected Loss and Decomposition (Eq 4.34 - 4.39)
- Calculus of Variations derivation of optimal regression function f*(x) = E[t|x]
- Minkowski / L_q loss functions (Eq 4.40): Mean (q=2), Median (q=1), Mode (q->0)
- Multimodal conditional distributions and failure modes of squared loss
- Faithful reproduction of Figures 4.5 and 4.6
"""
import os
from typing import Optional, List, Tuple, Callable, Dict, Union
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as opt
import matplotlib.pyplot as plt

from common.plot_utils import setup_style
from common.linear_models import MinkowskiLoss


def minkowski_loss(diff: Union[float, np.ndarray], q: float) -> Union[float, np.ndarray]:
    """
    Compute the Minkowski loss L_q = |diff|^q = |f(x) - t|^q (Eq 4.40).
    """
    if q <= 0:
        raise ValueError("q must be positive")
    return np.abs(diff) ** q


def expected_squared_loss_decomposition(
    f_func: Callable[[np.ndarray], np.ndarray],
    cond_mean_func: Callable[[np.ndarray], np.ndarray],
    cond_var_func: Callable[[np.ndarray], np.ndarray],
    p_x_func: Callable[[np.ndarray], np.ndarray],
    x_range: Tuple[float, float] = (-1.0, 1.0)
) -> Dict[str, float]:
    r"""
    Compute the analytical decomposition of expected squared loss (Eq 4.39):
        E[L] = \int {f(x) - E[t|x]}^2 p(x) dx + \int var[t|x] p(x) dx
             = Model Error + Irreducible Noise
    """
    # 1. Model error: \int {f(x) - E[t|x]}^2 p(x) dx
    model_err, _ = integrate.quad(
        lambda x: ((f_func(np.array([x]))[0] - cond_mean_func(np.array([x]))[0]) ** 2) * p_x_func(np.array([x]))[0],
        x_range[0], x_range[1]
    )

    # 2. Irreducible noise: \int var[t|x] p(x) dx
    noise_err, _ = integrate.quad(
        lambda x: cond_var_func(np.array([x]))[0] * p_x_func(np.array([x]))[0],
        x_range[0], x_range[1]
    )

    total_loss = model_err + noise_err
    return {
        "model_error": float(model_err),
        "irreducible_noise": float(noise_err),
        "expected_loss": float(total_loss)
    }


def optimal_minkowski_point_prediction(
    t_samples: np.ndarray,
    q: float,
    init_guess: Optional[float] = None
) -> float:
    r"""
    Find the optimal scalar prediction f that minimizes expected Minkowski loss:
        \min_f (1/N) \sum_{n=1}^N |f - t_n|^q
    For q=2: sample mean
    For q=1: sample median
    For small q (e.g. 0.1): sample mode approximation
    """
    t_arr = np.asarray(t_samples, dtype=np.float64).ravel()
    if len(t_arr) == 0:
        raise ValueError("t_samples must be non-empty")

    if np.isclose(q, 2.0):
        return float(np.mean(t_arr))
    elif np.isclose(q, 1.0):
        return float(np.median(t_arr))

    if init_guess is None:
        init_guess = float(np.median(t_arr))

    # For general q, use numerical optimization with robust loss
    res = opt.minimize_scalar(
        lambda f: np.mean(np.abs(f - t_arr) ** q),
        bracket=(np.min(t_arr), init_guess, np.max(t_arr)),
        method='brent'
    )
    return float(res.x)


def plot_figure_4_5_regression_function(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Faithfully reproduce Figure 4.5 from Bishop & Bishop (2024), page 121:
    The regression function f*(x), which minimizes expected squared loss,
    is given by the mean of the conditional distribution p(t|x).
    """
    fig, ax = plt.subplots(figsize=(5.0, 4.5), dpi=300)

    # Hide box spines
    for s in ax.spines.values():
        s.set_visible(False)

    # Draw coordinate axes with black arrows starting at (0, 0)
    arrow_kw = dict(arrowstyle='->', color='black', lw=1.3, mutation_scale=14)
    ax.annotate('', xy=(1.05, 0.0), xytext=(0.0, 0.0), arrowprops=arrow_kw)
    ax.annotate('', xy=(0.0, 1.05), xytext=(0.0, 0.0), arrowprops=arrow_kw)

    ax.text(1.08, 0.0, r'$x$', fontsize=12, ha='left', va='center')
    ax.text(0.0, 1.08, r'$t$', fontsize=12, ha='center', va='bottom')

    # Red regression curve f*(x)
    x = np.linspace(0.0, 0.96, 300)
    x_inf = 0.48
    f_star = 0.54 + 2.4 * ((x - x_inf) ** 3) + 0.28 * (x - x_inf)
    ax.plot(x, f_star, color='red', lw=1.8, zorder=3)
    ax.text(0.92, f_star[-1] + 0.05, r'$f^\star(x)$', color='black', fontsize=11, ha='center')

    # Slicing line at x0 = 0.65
    x0 = 0.65
    mu_t = 0.54 + 2.4 * ((x0 - x_inf) ** 3) + 0.28 * (x0 - x_inf)

    # Vertical grey-blue line
    ax.plot([x0, x0], [0, 0.96], color='#5A7288', lw=1.2, zorder=2)

    # Rotated Gaussian bell curve centered at mu_t
    t_vals = np.linspace(0.15, 0.95, 400)
    sigma = 0.070
    density = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * ((t_vals - mu_t) / sigma) ** 2)
    scale_x = 0.018
    x_density = x0 + scale_x * density

    ax.plot(x_density, t_vals, color='#0044EE', lw=1.6, zorder=4)

    # Label p(t | x0, w, sigma^2)
    ax.text(x0 + 0.015, mu_t - 0.16, r'$p(t \mid x_0, \mathbf{w}, \sigma^2)$', fontsize=10.5, color='black')

    ax.set_xlim(-0.02, 1.15)
    ax.set_ylim(-0.02, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 4.5 saved to: {p}")
    if show:
        plt.show()
    return fig, ax


def plot_figure_4_6_minkowski_loss(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Faithfully reproduce Figure 4.6 from Bishop & Bishop (2024), page 123:
    Plots of the quantity L_q = |f - t|^q for various values of q:
    q = 0.3, q = 1, q = 2, q = 10.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.5), dpi=300)
    q_values = [0.3, 1, 2, 10]
    diff = np.linspace(-2, 2, 501) # ensures 0 is an exact grid point

    for ax, q in zip(axes.flat, q_values):
        loss = np.abs(diff) ** q
        ax.plot(diff, loss, color='red', lw=1.5)
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 2.05)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xlabel(r'$f - t$', fontsize=11)
        if q == 1:
            ax.set_ylabel(r'$|f - t|^1$', fontsize=11)
        elif q == 0.3:
            ax.set_ylabel(r'$|f - t|^{0.3}$', fontsize=11)
        else:
            ax.set_ylabel(rf'$|f - t|^{{{q}}}$', fontsize=11)
        ax.text(0.0, 1.7, rf'$q = {q}$', fontsize=11, ha='center', va='center')
        ax.tick_params(direction='out', top=False, right=False)
        for s in ax.spines.values():
            s.set_linewidth(0.8)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 4.6 saved to: {p}")
    if show:
        plt.show()
    return fig, axes


# =====================================================================
# Classification Decision Theory (Chapter 5, Section 5.2) Re-exports
# =====================================================================
from common.classification_decision_theory import (
    optimal_decision_rule_misclassification,
    compute_expected_loss,
    optimal_decision_rule_expected_loss,
    decision_rule_with_reject,
    compensate_for_class_priors,
    ConfusionMatrix2Class,
    compute_roc_curve,
    plot_figure_5_5_joint_probabilities,
    plot_figure_5_6_loss_matrix,
    plot_figure_5_7_reject_option,
    plot_figure_5_8_class_densities_posteriors,
    plot_figure_5_9_confusion_matrix,
    plot_figure_5_10_roc_regions,
    plot_figure_5_11_roc_curve,
    generate_all_section_5_2_figures,
)

