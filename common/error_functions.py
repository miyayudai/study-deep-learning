"""
Chapter 6: Deep Neural Networks
Section 6.4: Error Functions

Bishop & Bishop (2024), Deep Learning: Foundations and Concepts, pp. 194-198.

Covers:
- 6.4.1 Regression: Gaussian conditional distribution, sum-of-squares error,
  maximum likelihood noise variance estimation, canonical identity link (Eq 6.23 - 6.31)
- 6.4.2 Binary classification: Bernoulli conditional distribution, logistic sigmoid,
  binary cross-entropy, multiple independent binary classifications, label noise model (Eq 6.32 - 6.35)
- 6.4.3 Multiclass classification: 1-of-K coding, softmax activation, multi-class cross-entropy,
  weight-space translation invariance (Eq 6.36 - 6.37)
- Canonical error gradient unification: dE / da_k = y_k - t_k (Eq 6.31) across all canonical pairings
"""

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

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
# 1. Regression Error Functions & Noise Estimation (Section 6.4.1)
# ==============================================================================

def sum_of_squares_loss(y: np.ndarray, t: np.ndarray) -> float:
    """Compute sum-of-squares error function (Eq 6.26 / Eq 6.29).
    
    E(w) = 0.5 * sum_{n=1}^N ||y(x_n, w) - t_n||^2
    
    Args:
        y: Predictions array of shape (N,) or (N, K).
        t: Target values array of shape (N,) or (N, K).
        
    Returns:
        Scalar sum-of-squares error value.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    diff = y_arr - t_arr
    return float(0.5 * np.sum(diff ** 2))


def mean_squared_error(y: np.ndarray, t: np.ndarray) -> float:
    """Compute mean squared error: MSE = (1 / N) * sum_{n=1}^N ||y_n - t_n||^2."""
    y_arr = np.asarray(y, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    return float(np.mean(np.sum((y_arr - t_arr) ** 2, axis=-1 if y_arr.ndim > 1 else 0)))


def gaussian_nll_loss(y: np.ndarray, t: np.ndarray, sigma_sq: float) -> float:
    """Compute full Gaussian negative log likelihood error function (Eq 6.25).
    
    E(w, sigma^2) = (1 / (2 * sigma^2)) * sum_{n=1}^N (y_n - t_n)^2
                    + (N / 2) * ln(sigma^2) + (N / 2) * ln(2 * pi)
                    
    Args:
        y: Predictions (N,).
        t: Targets (N,).
        sigma_sq: Noise variance sigma^2 > 0.
        
    Returns:
        Full negative log likelihood.
    """
    if sigma_sq <= 0:
        raise ValueError("Noise variance sigma_sq must be strictly positive.")
    y_arr = np.asarray(y, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    N = len(y_arr)
    sse = np.sum((y_arr - t_arr) ** 2)
    nll = (0.5 / sigma_sq) * sse + 0.5 * N * np.log(sigma_sq) + 0.5 * N * np.log(2 * np.pi)
    return float(nll)


def estimate_noise_variance(y: np.ndarray, t: np.ndarray) -> float:
    """Compute Maximum Likelihood estimator for Gaussian noise variance sigma^{*2} (Eq 6.27 / Eq 6.30).
    
    Single target (Eq 6.27):
        sigma^{*2} = (1 / N) * sum_{n=1}^N (y_n - t_n)^2
        
    Multiple targets (Eq 6.30):
        sigma^{*2} = (1 / (N * K)) * sum_{n=1}^N ||y_n - t_n||^2
        
    Args:
        y: Predictions (N,) or (N, K).
        t: Targets (N,) or (N, K).
        
    Returns:
        Estimated scalar noise variance sigma^{*2}.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    N = y_arr.shape[0]
    K = y_arr.shape[1] if y_arr.ndim > 1 else 1
    total_squared_error = np.sum((y_arr - t_arr) ** 2)
    return float(total_squared_error / (N * K))


# ==============================================================================
# 2. Binary Classification & Label Noise (Section 6.4.2)
# ==============================================================================

def binary_cross_entropy_loss(y: np.ndarray, t: np.ndarray, eps: float = 1e-15) -> float:
    """Compute binary cross-entropy error function (Eq 6.33).
    
    E(w) = - sum_{n=1}^N [ t_n * ln(y_n) + (1 - t_n) * ln(1 - y_n) ]
    
    Args:
        y: Predicted probabilities in [0, 1], shape (N,).
        t: Binary target labels in {0, 1}, shape (N,).
        eps: Small constant for numerical stability.
        
    Returns:
        Scalar binary cross-entropy error.
    """
    y_arr = np.clip(np.asarray(y, dtype=np.float64), eps, 1.0 - eps)
    t_arr = np.asarray(t, dtype=np.float64)
    loss = -np.sum(t_arr * np.log(y_arr) + (1.0 - t_arr) * np.log(1.0 - y_arr))
    return float(loss)


def binary_cross_entropy_with_logits(a: np.ndarray, t: np.ndarray) -> float:
    """Compute binary cross-entropy directly from pre-activations (logits) a.
    
    Uses stable formulation:
        E(a, t) = sum_n [ max(a_n, 0) - a_n * t_n + ln(1 + exp(-|a_n|)) ]
    """
    a_arr = np.asarray(a, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    loss = np.sum(np.maximum(a_arr, 0.0) - a_arr * t_arr + np.log1p(np.exp(-np.abs(a_arr))))
    return float(loss)


def multilabel_cross_entropy_loss(y: np.ndarray, t: np.ndarray, eps: float = 1e-15) -> float:
    """Compute cross-entropy for K independent binary classifications (Eq 6.35).
    
    E(w) = - sum_{n=1}^N sum_{k=1}^K [ t_{nk} * ln(y_{nk}) + (1 - t_{nk}) * ln(1 - y_{nk}) ]
    
    Args:
        y: Predicted probabilities, shape (N, K).
        t: Binary targets, shape (N, K).
        eps: Epsilon for log stability.
        
    Returns:
        Scalar total cross-entropy.
    """
    y_arr = np.clip(np.asarray(y, dtype=np.float64), eps, 1.0 - eps)
    t_arr = np.asarray(t, dtype=np.float64)
    loss = -np.sum(t_arr * np.log(y_arr) + (1.0 - t_arr) * np.log(1.0 - y_arr))
    return float(loss)


def label_noise_bernoulli(y: np.ndarray, flip_prob: float = 0.05) -> np.ndarray:
    """Model label noise with label flip probability epsilon (Opper and Winther, 2000; Bishop p. 196-197).
    
    p(t=1 | x) = (1 - epsilon) * y + epsilon * (1 - y) = (1 - 2 * epsilon) * y + epsilon
    
    Args:
        y: Clean model predicted probability p(C_1 | x).
        flip_prob: Label corruption probability epsilon in [0, 0.5).
        
    Returns:
        Adjusted conditional probability p(t=1 | x, epsilon).
    """
    if not (0.0 <= flip_prob < 0.5):
        raise ValueError("flip_prob must be in [0.0, 0.5).")
    y_arr = np.asarray(y, dtype=np.float64)
    return (1.0 - 2.0 * flip_prob) * y_arr + flip_prob


# ==============================================================================
# 3. Multiclass Classification & Softmax (Section 6.4.3)
# ==============================================================================

def softmax(a: np.ndarray) -> np.ndarray:
    """Compute softmax activation function (Eq 6.37) with numerical stability.
    
    y_k = exp(a_k) / sum_j exp(a_j)
    
    Invariance property: softmax(a + c) == softmax(a) for any scalar constant c.
    
    Args:
        a: Pre-activations array of shape (..., K).
        
    Returns:
        Softmax probability array of same shape, summing to 1 along last axis.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    max_a = np.max(a_arr, axis=-1, keepdims=True)
    exp_a = np.exp(a_arr - max_a)
    return exp_a / np.sum(exp_a, axis=-1, keepdims=True)


def multiclass_cross_entropy_loss(y: np.ndarray, t: np.ndarray, eps: float = 1e-15) -> float:
    """Compute multiclass cross-entropy error function (Eq 6.36).
    
    E(w) = - sum_{n=1}^N sum_{k=1}^K t_{nk} * ln(y_{nk})
    
    Args:
        y: Predicted probabilities of shape (N, K).
        t: One-hot encoded target labels of shape (N, K).
        eps: Epsilon for log stability.
        
    Returns:
        Scalar cross-entropy loss.
    """
    y_arr = np.clip(np.asarray(y, dtype=np.float64), eps, 1.0)
    t_arr = np.asarray(t, dtype=np.float64)
    return float(-np.sum(t_arr * np.log(y_arr)))


def categorical_cross_entropy_with_logits(a: np.ndarray, t: np.ndarray) -> float:
    """Compute multiclass cross-entropy directly from pre-activations (logits) a
    using log-sum-exp for optimal numerical stability.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    max_a = np.max(a_arr, axis=-1, keepdims=True)
    log_sum_exp = max_a + np.log(np.sum(np.exp(a_arr - max_a), axis=-1, keepdims=True))
    log_probs = a_arr - log_sum_exp
    return float(-np.sum(t_arr * log_probs))


# ==============================================================================
# 4. Canonical Pre-activation Gradient Unification (Eq 6.31)
# ==============================================================================

def canonical_preactivation_gradient(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Universal canonical error gradient with respect to output pre-activation a_k (Eq 6.31):
    
    dE / da_k = y_k - t_k
    
    This exact relationship holds identically across:
    1. Linear regression with sum-of-squares error: y = a, E = 0.5 (y - t)^2
    2. Binary classification with cross-entropy error: y = sigma(a), E = -[t ln y + (1-t) ln(1-y)]
    3. Multiclass classification with cross-entropy error: y = softmax(a), E = -sum t_k ln y_k
    
    Args:
        y: Output predictions.
        t: Target labels.
        
    Returns:
        Gradient dE / da = y - t.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    return y_arr - t_arr


def compare_bce_vs_mse_gradient(a_grid: np.ndarray, target: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compare gradient magnitude of binary cross-entropy vs sum-of-squares with respect to logit a.
    
    For target t = 1:
    - Cross-entropy: dE / da = y - 1 = sigma(a) - 1.
      When a -> -inf (strongly wrong prediction), dE / da -> -1 (constant strong gradient, no saturation!).
    - Sum-of-squares: dE / da = (y - 1) * y' = (sigma(a) - 1) * sigma(a) * (1 - sigma(a)).
      When a -> -inf, sigma(a) -> 0, so dE / da -> 0 (severe vanishing gradient / saturation!).
      
    This explains why Simard et al. (2003) observed significantly faster training with cross-entropy.
    """
    sig = 1.0 / (1.0 + np.exp(-np.clip(a_grid, -30.0, 30.0)))
    grad_bce = sig - target
    grad_mse = (sig - target) * sig * (1.0 - sig)
    return grad_bce, grad_mse


# ==============================================================================
# 5. Publication-Quality Figure Generator (Figure 6 Section 6.4)
# ==============================================================================

def generate_figure_6_error_functions(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """Generate and save publication-quality 4-panel overview of Section 6.4 Error Functions:
    
    (a) Regression: Gaussian conditional distribution p(t|x) & sum-of-squares loss.
    (b) Binary Classification: Cross-entropy vs MSE gradients showing saturation immunity.
    (c) Multiclass Classification: Softmax probabilities and shift invariance a -> a + c.
    (d) Canonical Gradient Unification: Universal dE/da = y - t across all 3 canonical pairings.
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # -------------------------------------------------------------------------
    # Panel (a): Regression - Gaussian Conditional Distribution
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    x_reg = np.linspace(-3, 3, 300)
    # Target function y(x) = sin(x)
    y_reg = np.sin(x_reg)
    sigma = 0.4
    
    ax.plot(x_reg, y_reg, 'r-', lw=2.2, label='$y(x, w)$ (Network Mean)')
    ax.fill_between(x_reg, y_reg - 2 * sigma, y_reg + 2 * sigma, color='#f28b82', alpha=0.25, label='$\\pm 2\\sigma$ Noise Envelope')
    
    # Draw vertical Gaussian slices at two points x = -1.0 and x = 1.5
    for x_slice in [-1.0, 1.5]:
        mean_slice = np.sin(x_slice)
        t_vals = np.linspace(mean_slice - 3 * sigma, mean_slice + 3 * sigma, 100)
        p_vals = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((t_vals - mean_slice) / sigma) ** 2)
        # Scale for visualization
        ax.plot(x_slice + p_vals * 0.4, t_vals, 'b-', lw=1.5)
        ax.plot([x_slice, x_slice], [t_vals[0], t_vals[-1]], 'k--', alpha=0.4)
        
    ax.set_title('(a) Regression: $p(t|x, w) = \\mathcal{N}(t|y(x, w), \\sigma^2)$ (Eq 6.23)')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    ax.set_ylim(-2.2, 2.2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # -------------------------------------------------------------------------
    # Panel (b): Binary Classification - BCE vs MSE Gradient Dynamics
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    a_vals = np.linspace(-6, 4, 300)
    grad_bce, grad_mse = compare_bce_vs_mse_gradient(a_vals, target=1.0)
    
    ax.plot(a_vals, np.abs(grad_bce), color='#1a73e8', lw=2.4, label='Cross-Entropy $|\\partial E / \\partial a|$ (No vanishing)')
    ax.plot(a_vals, np.abs(grad_mse), color='#ea4335', lw=2.4, linestyle='--', label='Sum-of-Squares $|\\partial E / \\partial a|$ (Vanishes!)')
    
    # Mark saturation zone
    ax.axvspan(-6, -2, color='#fce8e6', alpha=0.5, label='Saturation Zone (Wrong prediction)')
    ax.annotate('Vanishing gradient\\nfor large error!', xy=(-3.5, 0.05), xytext=(-5.0, 0.4),
                arrowprops=dict(arrowstyle='->', color='#ea4335', lw=1.5), fontsize=10.5, color='#ea4335')
    ax.annotate('Constant driving force\\nmaintains learning', xy=(-4.0, 0.98), xytext=(-2.5, 0.75),
                arrowprops=dict(arrowstyle='->', color='#1a73e8', lw=1.5), fontsize=10.5, color='#1a73e8')
                
    ax.set_title('(b) Binary: Cross-Entropy vs MSE Gradient ($t=1$)')
    ax.set_xlabel('Pre-activation $a$')
    ax.set_ylabel('Absolute Gradient $|\\partial E / \\partial a|$')
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center right', fontsize=10)
    
    # -------------------------------------------------------------------------
    # Panel (c): Multiclass Classification - Softmax Translation Invariance
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    base_logits = np.array([2.0, 1.0, -1.0])
    shifts = np.linspace(-5, 5, 50)
    probs_class0 = []
    probs_class1 = []
    probs_class2 = []
    
    for c in shifts:
        p = softmax(base_logits + c)
        probs_class0.append(p[0])
        probs_class1.append(p[1])
        probs_class2.append(p[2])
        
    ax.plot(shifts, probs_class0, color='#0f9d58', lw=2.2, label='$y_1 = \\sigma(\\mathbf{a}+c)_1 \\approx 0.705$')
    ax.plot(shifts, probs_class1, color='#f4b400', lw=2.2, label='$y_2 = \\sigma(\\mathbf{a}+c)_2 \\approx 0.259$')
    ax.plot(shifts, probs_class2, color='#4285f4', lw=2.2, label='$y_3 = \\sigma(\\mathbf{a}+c)_3 \\approx 0.035$')
    
    ax.set_title('(c) Multiclass: Softmax Shift Invariance $\\mathbf{a} \\to \\mathbf{a} + c$ (Eq 6.37)')
    ax.set_xlabel('Constant Shift $c$ added to all logits')
    ax.set_ylabel('Output Probability $y_k$')
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # -------------------------------------------------------------------------
    # Panel (d): Canonical Gradient Unification Summary
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.axis('off')
    
    # Draw summary comparison table
    table_data = [
        ["Problem Type", "Output Activation", "Error Function $E(\\mathbf{w})$", "Pre-activation Gradient"],
        ["Regression", "Identity: $y_k = a_k$", "Sum-of-squares (Eq 6.26)", "$\\partial E / \\partial a_k = y_k - t_k$"],
        ["Binary", "Sigmoid: $y = \\sigma(a)$", "Cross-entropy (Eq 6.33)", "$\\partial E / \\partial a = y - t$"],
        ["Indep. Binary", "Sigmoid: $y_k = \\sigma(a_k)$", "Multi-label CE (Eq 6.35)", "$\\partial E / \\partial a_k = y_k - t_k$"],
        ["Multiclass", "Softmax: $y_k = \\frac{e^{a_k}}{\\sum e^{a_j}}$", "Multiclass CE (Eq 6.36)", "$\\partial E / \\partial a_k = y_k - t_k$"]
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 2.5)
    
    # Style header row
    for col_idx in range(4):
        cell = table[(0, col_idx)]
        cell.set_facecolor('#1a237e')
        cell.set_text_props(color='white', weight='bold')
        
    for row_idx in range(1, 5):
        bg = '#e8eaf6' if row_idx % 2 == 1 else '#ffffff'
        for col_idx in range(4):
            table[(row_idx, col_idx)].set_facecolor(bg)
            
    ax.set_title('(d) Canonical Matching Pairing & Universal Gradient (Eq 6.31)', pad=20)
    
    plt.tight_layout()
    _save_figure(fig, "fig_6_error_functions.png", filepath=filepath, save_both=save_both)
    return fig


if __name__ == "__main__":
    generate_figure_6_error_functions()
    print("Section 6.4 figures generated successfully.")
