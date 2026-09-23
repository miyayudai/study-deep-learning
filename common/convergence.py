"""Chapter 7: Gradient Descent
Section 7.3: Convergence

This module implements mathematical models, convergence diagnostics, momentum,
learning rate schedules, and adaptive optimizers (AdaGrad, RMSProp, Adam) for
Section 7.3 of:
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.

Contents:
- Convergence in Quadratic Error Surfaces:
  * Decoupled coordinate dynamics: alpha_i^(tau) = (1 - eta * lambda_i)^tau alpha_i^(0) (Eq 7.24 - 7.29)
  * Maximum stable learning rate: eta < 2 / lambda_max
  * Convergence rate along dominant slow axis: 1 - 2 * lambda_min / lambda_max (Eq 7.30)
  * Condition number of the Hessian: kappa = lambda_max / lambda_min
- Section 7.3.1: Momentum
  * Momentum update: Delta w^(tau-1) = -eta * grad E(w^(tau-1)) + mu * Delta w^(tau-2) (Eq 7.31)
  * Effective learning rate in low curvature: eta / (1 - mu) (Eq 7.32, 7.33)
  * Cancellation in high curvature regions (Figure 7.5)
  * Algorithm 7.3: Stochastic gradient descent with momentum
  * Nesterov momentum: Delta w^(tau-1) = -eta * grad E(w^(tau-1) + mu * Delta w^(tau-2)) + mu * Delta w^(tau-2) (Eq 7.34)
- Section 7.3.2: Learning Rate Schedules
  * Linear decay: eta(tau) = (1 - tau/K) eta_0 + (tau/K) eta_K (Eq 7.36)
  * Power law decay: eta(tau) = eta_0 / (1 + tau/s)^c (Eq 7.37)
  * Exponential decay: eta(tau) = eta_0 * c^(tau/s) (Eq 7.38)
- Section 7.3.3: RMSProp and Adam
  * AdaGrad: r_i^(tau) = r_i^(tau-1) + g_i^2 (Eq 7.39, 7.40)
  * RMSProp: r_i^(tau) = beta * r_i^(tau-1) + (1 - beta) * g_i^2 (Eq 7.41, 7.42)
  * Adam: first and second moments with bias correction (Eq 7.43 - 7.47, Algorithm 7.4)
- Figure Reproductions:
  * Figure 7.3: Fixed-step gradient descent in a long valley
  * Figure 7.4: Linear convergence down low curvature with effective learning rate increase
  * Figure 7.5: Oscillatory steps across high curvature with momentum cancellation
  * Figure 7.6: Gradient descent with momentum in a long valley
"""

from dataclasses import dataclass, field
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from common.plot_utils import save_plot, setup_style


def _get_project_root() -> str:
    """Return the absolute path to the repository root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _save_figure(
    fig: plt.Figure,
    filename: str,
    filepath: Optional[str] = None,
    save_both: bool = True,
) -> Tuple[str, Optional[str]]:
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


# ===========================================================================
# Mathematical Convergence Analysis in Quadratic Error Surface
# ===========================================================================

def max_stable_learning_rate(eigenvalues: Union[List[float], np.ndarray]) -> float:
    """Calculate the maximum stable learning rate for gradient descent.
    
    Convergence requires |1 - eta * lambda_i| < 1 for all i, which implies
    0 < eta < 2 / lambda_max. (Bishop & Bishop 2024, p. 219)
    
    Args:
        eigenvalues: Eigenvalues lambda of the Hessian matrix.
        
    Returns:
        Upper bound 2 / lambda_max.
    """
    ev = np.asarray(eigenvalues, dtype=np.float64)
    if np.any(ev <= 0):
        raise ValueError("All eigenvalues must be positive for a local minimum")
    return float(2.0 / np.max(ev))


def condition_number(eigenvalues: Union[List[float], np.ndarray]) -> float:
    """Calculate the condition number of the Hessian matrix.
    
    Condition number kappa = lambda_max / lambda_min.
    A large condition number corresponds to highly elongated elliptical contours
    (a steep narrow valley). (Bishop & Bishop 2024, p. 220)
    
    Args:
        eigenvalues: Eigenvalues of the Hessian matrix.
        
    Returns:
        kappa = lambda_max / lambda_min.
    """
    ev = np.asarray(eigenvalues, dtype=np.float64)
    if np.any(ev <= 0):
        raise ValueError("All eigenvalues must be positive")
    return float(np.max(ev) / np.min(ev))


def convergence_rate_dominant_axis(
    lambda_min: float,
    lambda_max: float,
) -> float:
    """Calculate the convergence factor along the slowest direction under maximum stable learning rate.
    
    When eta = 2 / lambda_max, the distance along the smallest eigenvalue axis
    decays per step by the factor:
        |1 - 2 * lambda_min / lambda_max| = 1 - 2 / kappa (Eq 7.30)
    
    Args:
        lambda_min: Smallest eigenvalue of the Hessian.
        lambda_max: Largest eigenvalue of the Hessian.
        
    Returns:
        Decay factor per step (close to 1 when condition number is large).
    """
    if lambda_min <= 0 or lambda_max <= 0 or lambda_min > lambda_max:
        raise ValueError("Invalid eigenvalues: must satisfy 0 < lambda_min <= lambda_max")
    return float(1.0 - 2.0 * lambda_min / lambda_max)


def eigen_distance_evolution(
    lambda_val: float,
    lr: float,
    steps: int,
    alpha_0: float = 1.0,
) -> np.ndarray:
    """Compute independent evolution of distance alpha_i^(tau) along eigenvector u_i (Eq 7.29).
    
    alpha_i^(tau) = (1 - eta * lambda_i)^tau * alpha_i^(0)
    
    Args:
        lambda_val: Eigenvalue lambda_i.
        lr: Learning rate eta.
        steps: Number of steps T.
        alpha_0: Initial distance alpha_i^(0).
        
    Returns:
        Array of distance values for tau = 0, ..., steps.
    """
    taus = np.arange(steps + 1)
    factor = 1.0 - lr * lambda_val
    return alpha_0 * (factor ** taus)


# ===========================================================================
# 7.3.1 Momentum & Nesterov Accelerated Gradient
# ===========================================================================

def effective_momentum_learning_rate(lr: float, momentum: float) -> float:
    """Calculate effective learning rate in a region of low curvature (Eq 7.33).
    
    When gradient is roughly constant:
        Delta w = -eta * grad E * (1 + mu + mu^2 + ...) = - (eta / (1 - mu)) * grad E
    
    Args:
        lr: Learning rate eta > 0.
        momentum: Momentum parameter 0 <= mu < 1.
        
    Returns:
        Effective learning rate eta / (1 - mu).
    """
    if not (0.0 <= momentum < 1.0):
        raise ValueError("Momentum parameter mu must be in [0, 1)")
    return lr / (1.0 - momentum)


@dataclass
class ConvergenceTrajectory:
    """Container for optimizer trajectory, loss, and step counts."""
    weights: np.ndarray  # Shape: (steps + 1, W)
    errors: np.ndarray   # Shape: (steps + 1,)
    steps: int
    converged: bool
    algorithm_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def gradient_descent_momentum(
    w_init: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    error_fn: Optional[Callable[[np.ndarray], float]] = None,
    lr: float = 0.01,
    momentum: float = 0.9,
    nesterov: bool = False,
    max_steps: int = 100,
    tol: float = 1e-6,
) -> ConvergenceTrajectory:
    """Gradient descent with classical momentum (Eq 7.31) or Nesterov momentum (Eq 7.34).
    
    Classical Momentum:
        Delta w^(tau-1) = -eta * grad E(w^(tau-1)) + mu * Delta w^(tau-2)
        w^(tau) = w^(tau-1) + Delta w^(tau-1)
        
    Nesterov Momentum:
        Delta w^(tau-1) = -eta * grad E(w^(tau-1) + mu * Delta w^(tau-2)) + mu * Delta w^(tau-2)
        w^(tau) = w^(tau-1) + Delta w^(tau-1)
        
    Args:
        w_init: Initial weight vector w0.
        grad_fn: Function returning gradient vector grad E(w).
        error_fn: Optional scalar error evaluation E(w).
        lr: Learning rate eta.
        momentum: Momentum coefficient mu in [0, 1).
        nesterov: If True, uses Nesterov accelerated gradient (Eq 7.34).
        max_steps: Maximum iterations.
        tol: Gradient norm tolerance for early stopping.
        
    Returns:
        ConvergenceTrajectory containing weights, error history, and status.
    """
    w = np.array(w_init, dtype=np.float64, copy=True)
    delta_w = np.zeros_like(w)
    
    weights_hist = [w.copy()]
    errors_hist = [error_fn(w) if error_fn is not None else float("nan")]
    
    converged = False
    
    for step in range(1, max_steps + 1):
        if nesterov:
            # Gradient evaluated at look-ahead position w + mu * delta_w
            w_lookahead = w + momentum * delta_w
            grad = grad_fn(w_lookahead)
        else:
            grad = grad_fn(w)
            
        if np.linalg.norm(grad) < tol:
            converged = True
            break
            
        delta_w = -lr * grad + momentum * delta_w
        w += delta_w
        
        weights_hist.append(w.copy())
        errors_hist.append(error_fn(w) if error_fn is not None else float("nan"))
        
    algo_name = "Nesterov Momentum (Eq 7.34)" if nesterov else "Classical Momentum (Eq 7.31)"
    return ConvergenceTrajectory(
        weights=np.array(weights_hist),
        errors=np.array(errors_hist),
        steps=len(weights_hist) - 1,
        converged=converged,
        algorithm_name=algo_name,
        metadata={"lr": lr, "momentum": momentum, "nesterov": nesterov},
    )


def sgd_momentum_algorithm_7_3(
    w_init: np.ndarray,
    grad_minibatch_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_samples: int,
    batch_size: int = 32,
    error_fn: Optional[Callable[[np.ndarray], float]] = None,
    lr: float = 0.01,
    momentum: float = 0.9,
    max_epochs: int = 10,
    shuffle: bool = True,
    random_state: Optional[int] = None,
    record_every_steps: Optional[int] = None,
) -> ConvergenceTrajectory:
    """Stochastic Gradient Descent with Momentum (Algorithm 7.3).
    
    Input: Training set indexed by n in {1, ..., N}
           Batch size B
           Error function per mini-batch E_{n:n+B-1}(w)
           Learning rate parameter eta
           Momentum parameter mu
           Initial weight vector w
    Output: Final weight vector w
    
    n <- 1
    Delta w <- 0
    repeat
        Delta w <- -eta * grad E_{n:n+B-1}(w) + mu * Delta w
        w <- w + Delta w
        n <- n + B
        if n > N then
            shuffle data
            n <- 1
        end if
    until convergence
    return w
    """
    rng = np.random.default_rng(random_state)
    w = np.array(w_init, dtype=np.float64, copy=True)
    delta_w = np.zeros_like(w)
    
    weights_hist = [w.copy()]
    errors_hist = [error_fn(w) if error_fn is not None else float("nan")]
    
    total_steps = 0
    num_batches_per_epoch = int(np.ceil(n_samples / batch_size))
    step_record_interval = record_every_steps if record_every_steps is not None else num_batches_per_epoch
    
    for epoch in range(max_epochs):
        indices = np.arange(n_samples)
        if shuffle:
            rng.shuffle(indices)
            
        for start_idx in range(0, n_samples, batch_size):
            batch_idx = indices[start_idx : min(start_idx + batch_size, n_samples)]
            grad_b = grad_minibatch_fn(w, batch_idx)
            
            delta_w = -lr * grad_b + momentum * delta_w
            w += delta_w
            total_steps += 1
            
            if total_steps % step_record_interval == 0:
                weights_hist.append(w.copy())
                errors_hist.append(error_fn(w) if error_fn is not None else float("nan"))
                
    if total_steps % step_record_interval != 0:
        weights_hist.append(w.copy())
        errors_hist.append(error_fn(w) if error_fn is not None else float("nan"))
        
    return ConvergenceTrajectory(
        weights=np.array(weights_hist),
        errors=np.array(errors_hist),
        steps=total_steps,
        converged=False,
        algorithm_name="SGD with Momentum (Algorithm 7.3)",
        metadata={"lr": lr, "momentum": momentum, "batch_size": batch_size},
    )


# ===========================================================================
# 7.3.2 Learning Rate Schedules
# ===========================================================================

class LinearLRScheduler:
    """Linear learning rate decay schedule (Eq 7.36).
    
    eta(tau) = (1 - tau / K) * eta_0 + (tau / K) * eta_K for tau <= K,
    and eta(tau) = eta_K for tau > K.
    """
    def __init__(self, eta_0: float, eta_K: float, K: int):
        self.eta_0 = eta_0
        self.eta_K = eta_K
        self.K = K
        
    def __call__(self, tau: int) -> float:
        if tau >= self.K:
            return float(self.eta_K)
        return float((1.0 - tau / self.K) * self.eta_0 + (tau / self.K) * self.eta_K)


class PowerLawLRScheduler:
    """Power-law learning rate decay schedule (Eq 7.37).
    
    eta(tau) = eta_0 / (1 + tau / s)^c
    """
    def __init__(self, eta_0: float, s: float, c: float = 1.0):
        self.eta_0 = eta_0
        self.s = s
        self.c = c
        
    def __call__(self, tau: int) -> float:
        return float(self.eta_0 / ((1.0 + tau / self.s) ** self.c))


class ExponentialLRScheduler:
    """Exponential learning rate decay schedule (Eq 7.38).
    
    eta(tau) = eta_0 * c^(tau / s)  (with 0 < c < 1)
    """
    def __init__(self, eta_0: float, s: float, c: float):
        self.eta_0 = eta_0
        self.s = s
        self.c = c
        
    def __call__(self, tau: int) -> float:
        return float(self.eta_0 * (self.c ** (tau / self.s)))


# ===========================================================================
# 7.3.3 Adaptive Optimizers: AdaGrad, RMSProp, Adam
# ===========================================================================

def adagrad_optimizer(
    w_init: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    error_fn: Optional[Callable[[np.ndarray], float]] = None,
    lr: float = 0.1,
    delta: float = 1e-8,
    max_steps: int = 100,
    tol: float = 1e-6,
) -> ConvergenceTrajectory:
    """AdaGrad optimizer (Duchi et al., 2011; Eq 7.39, 7.40).
    
    r_i^(tau) = r_i^(tau-1) + (grad E)_i^2
    w_i^(tau) = w_i^(tau-1) - (eta / sqrt(r_i^(tau) + delta)) * (grad E)_i
    """
    w = np.array(w_init, dtype=np.float64, copy=True)
    r = np.zeros_like(w)
    
    weights_hist = [w.copy()]
    errors_hist = [error_fn(w) if error_fn is not None else float("nan")]
    converged = False
    
    for step in range(1, max_steps + 1):
        grad = grad_fn(w)
        if np.linalg.norm(grad) < tol:
            converged = True
            break
            
        r += grad ** 2
        w -= (lr / (np.sqrt(r) + delta)) * grad
        
        weights_hist.append(w.copy())
        errors_hist.append(error_fn(w) if error_fn is not None else float("nan"))
        
    return ConvergenceTrajectory(
        weights=np.array(weights_hist),
        errors=np.array(errors_hist),
        steps=len(weights_hist) - 1,
        converged=converged,
        algorithm_name="AdaGrad (Eq 7.39, 7.40)",
        metadata={"lr": lr, "delta": delta},
    )


def rmsprop_optimizer(
    w_init: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    error_fn: Optional[Callable[[np.ndarray], float]] = None,
    lr: float = 0.01,
    beta: float = 0.9,
    delta: float = 1e-8,
    max_steps: int = 100,
    tol: float = 1e-6,
) -> ConvergenceTrajectory:
    """RMSProp optimizer (Hinton, 2012; Eq 7.41, 7.42).
    
    r_i^(tau) = beta * r_i^(tau-1) + (1 - beta) * (grad E)_i^2
    w_i^(tau) = w_i^(tau-1) - (eta / sqrt(r_i^(tau) + delta)) * (grad E)_i
    """
    w = np.array(w_init, dtype=np.float64, copy=True)
    r = np.zeros_like(w)
    
    weights_hist = [w.copy()]
    errors_hist = [error_fn(w) if error_fn is not None else float("nan")]
    converged = False
    
    for step in range(1, max_steps + 1):
        grad = grad_fn(w)
        if np.linalg.norm(grad) < tol:
            converged = True
            break
            
        r = beta * r + (1.0 - beta) * (grad ** 2)
        w -= (lr / (np.sqrt(r) + delta)) * grad
        
        weights_hist.append(w.copy())
        errors_hist.append(error_fn(w) if error_fn is not None else float("nan"))
        
    return ConvergenceTrajectory(
        weights=np.array(weights_hist),
        errors=np.array(errors_hist),
        steps=len(weights_hist) - 1,
        converged=converged,
        algorithm_name="RMSProp (Eq 7.41, 7.42)",
        metadata={"lr": lr, "beta": beta, "delta": delta},
    )


def adam_optimizer(
    w_init: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    error_fn: Optional[Callable[[np.ndarray], float]] = None,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    delta: float = 1e-8,
    bias_correction: bool = True,
    max_steps: int = 100,
    tol: float = 1e-6,
) -> ConvergenceTrajectory:
    """Adam optimizer (Kingma & Ba, 2014; Algorithm 7.4, Eq 7.43 - 7.47).
    
    s_i^(tau) = beta1 * s_i^(tau-1) + (1 - beta1) * g_i
    r_i^(tau) = beta2 * r_i^(tau-1) + (1 - beta2) * g_i^2
    s_hat_i^(tau) = s_i^(tau) / (1 - beta1^tau)
    r_hat_i^(tau) = r_i^(tau) / (1 - beta2^tau)
    w_i^(tau) = w_i^(tau-1) - eta * (s_hat_i / (sqrt(r_hat_i) + delta))
    """
    w = np.array(w_init, dtype=np.float64, copy=True)
    s = np.zeros_like(w)
    r = np.zeros_like(w)
    
    weights_hist = [w.copy()]
    errors_hist = [error_fn(w) if error_fn is not None else float("nan")]
    converged = False
    
    for step in range(1, max_steps + 1):
        grad = grad_fn(w)
        if np.linalg.norm(grad) < tol:
            converged = True
            break
            
        s = beta1 * s + (1.0 - beta1) * grad
        r = beta2 * r + (1.0 - beta2) * (grad ** 2)
        
        if bias_correction:
            s_hat = s / (1.0 - beta1 ** step)
            r_hat = r / (1.0 - beta2 ** step)
        else:
            s_hat = s
            r_hat = r
            
        delta_w = -lr * (s_hat / (np.sqrt(r_hat) + delta))
        w += delta_w
        
        weights_hist.append(w.copy())
        errors_hist.append(error_fn(w) if error_fn is not None else float("nan"))
        
    return ConvergenceTrajectory(
        weights=np.array(weights_hist),
        errors=np.array(errors_hist),
        steps=len(weights_hist) - 1,
        converged=converged,
        algorithm_name="Adam Optimization (Algorithm 7.4)",
        metadata={"lr": lr, "beta1": beta1, "beta2": beta2, "delta": delta},
    )


# ===========================================================================
# Figure Reproductions (Figures 7.3, 7.4, 7.5, 7.6)
# ===========================================================================

def generate_figure_7_3(
    filepath: Optional[str] = None,
    save_both: bool = True,
) -> Tuple[str, Optional[str]]:
    """Reproduce Figure 7.3: Fixed-step gradient descent in a long valley.
    
    Schematic illustration of fixed-step gradient descent for an error function
    that has substantially different curvatures along different directions.
    The error surface E has the form of a long valley, depicted by ellipses.
    Negative gradient vector does not point towards minimum, oscillating across
    the valley and leading to very slow progress along u1 towards the minimum.
    (Bishop & Bishop 2024, p. 218)
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    
    alphas = [0.12, 0.22, 0.35, 0.50, 0.65]
    widths = [8.0, 6.5, 5.0, 3.5, 2.0]
    heights = [2.6, 2.1, 1.6, 1.1, 0.6]
    coral = (0.95, 0.45, 0.40)
    
    for w, h, a in zip(widths, heights, alphas):
        ell = Ellipse(xy=(0, 0), width=w, height=h, facecolor=coral, alpha=a, edgecolor="none")
        ax.add_patch(ell)
        
    # Draw axes u1 and u2 from center
    ax.annotate("", xy=(2.5, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(2.7, -0.05, r"$\mathbf{u}_1$", fontsize=13, va="center")
    
    ax.annotate("", xy=(0, 1.2), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(0, 1.35, r"$\mathbf{u}_2$", fontsize=13, ha="center")
    
    # Zig-zag trajectory on the left oscillating across the valley
    pts = [
        [-3.2, 0.55],
        [-2.7, -0.50],
        [-2.3, 0.45],
        [-1.9, -0.40],
        [-1.5, 0.35],
        [-1.2, -0.28],
    ]
    pts = np.array(pts)
    for i in range(len(pts) - 1):
        ax.annotate("", xy=(pts[i+1, 0], pts[i+1, 1]), xytext=(pts[i, 0], pts[i, 1]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.8, shrinkA=0, shrinkB=0))
        
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    
    fig.tight_layout()
    filename = "fig_7_3_valley_oscillation.png"
    return _save_figure(fig, filename, filepath, save_both=save_both)


def generate_figure_7_4(
    filepath: Optional[str] = None,
    save_both: bool = True,
) -> Tuple[str, Optional[str]]:
    """Reproduce Figure 7.4: Linear convergence down low curvature surface.
    
    With a fixed learning rate parameter, gradient descent down a surface with
    low curvature leads to successively smaller steps corresponding to linear
    convergence. In such a situation, the effect of a momentum term is like an
    increase in the effective learning rate parameter. (Bishop & Bishop 2024, p. 220)
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    
    w = np.linspace(0.5, 4.5, 200)
    E = 4.0 - 0.9 * w + 0.08 * (w - 0.5) ** 2
    ax.plot(w, E, color="red", lw=2)
    
    w_pts = [1.1, 2.0, 2.7, 3.3]
    E_pts = [4.0 - 0.9 * wp + 0.08 * (wp - 0.5) ** 2 for wp in w_pts]
    
    y_min = 0.5
    for wp, Ep in zip(w_pts, E_pts):
        ax.plot([wp, wp], [y_min, Ep], "k--", lw=1, alpha=0.6)
        ax.plot(wp, Ep, "o", color="gray", markersize=6)
        ax.plot(wp, y_min, "o", color="gray", markersize=5)
        
    for i in range(len(w_pts) - 1):
        ax.annotate("", xy=(w_pts[i+1], E_pts[i+1] + 0.15), xytext=(w_pts[i], E_pts[i] + 0.15),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        mid_w = 0.5 * (w_pts[i] + w_pts[i+1])
        mid_E = 0.5 * (E_pts[i] + E_pts[i+1]) + 0.35
        ax.text(mid_w, mid_E, rf"$\Delta w^{{({i+1})}}$", fontsize=11, ha="center")
        
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(0.5, 4.0)
    ax.set_xlabel("w", fontsize=12)
    ax.set_ylabel("E", fontsize=12, rotation=0, labelpad=15)
    
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)
        
    fig.tight_layout()
    filename = "fig_7_4_linear_convergence.png"
    return _save_figure(fig, filename, filepath, save_both=save_both)


def generate_figure_7_5(
    filepath: Optional[str] = None,
    save_both: bool = True,
) -> Tuple[str, Optional[str]]:
    """Reproduce Figure 7.5: Oscillatory steps across high curvature surface.
    
    For a situation in which successive steps of gradient descent are oscillatory,
    a momentum term has little influence on the effective value of the learning
    rate parameter because successive momentum terms cancel. (Bishop & Bishop 2024, p. 221)
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    
    w = np.linspace(-1.5, 1.5, 200)
    E = 2.0 * (w ** 2) + 0.5
    ax.plot(w, E, color="red", lw=2)
    
    pts_w = [-1.15, 0.95, -0.75, 0.55]
    pts_E = [2.0 * (wp ** 2) + 0.5 for wp in pts_w]
    
    for i in range(len(pts_w) - 1):
        w_start, E_start = pts_w[i], pts_E[i]
        w_end = pts_w[i+1]
        
        ax.annotate("", xy=(w_end, E_start), xytext=(w_start, E_start),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        ax.plot(w_start, E_start, "o", color="gray", markersize=5)
        ax.plot([w_end, w_end], [E_start, pts_E[i+1]], "k--", lw=1, alpha=0.6)
        ax.plot(w_end, pts_E[i+1], "o", color="gray", markersize=5)
        
        mid_w = 0.5 * (w_start + w_end)
        ax.text(mid_w, E_start + 0.2, rf"$\Delta w^{{({i+1})}}$", fontsize=11, ha="center")
        
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(0.0, 5.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)
        
    fig.tight_layout()
    filename = "fig_7_5_oscillatory_cancellation.png"
    return _save_figure(fig, filename, filepath, save_both=save_both)


def generate_figure_7_6(
    filepath: Optional[str] = None,
    save_both: bool = True,
) -> Tuple[str, Optional[str]]:
    """Reproduce Figure 7.6: Gradient descent with momentum in a long valley.
    
    Illustration of the effect of adding a momentum term to the gradient descent
    algorithm, showing more rapid progress along the valley of the error function,
    compared with the unmodified gradient descent in Figure 7.3.
    (Bishop & Bishop 2024, p. 222)
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    
    alphas = [0.12, 0.22, 0.35, 0.50, 0.65]
    widths = [8.0, 6.5, 5.0, 3.5, 2.0]
    heights = [2.6, 2.1, 1.6, 1.1, 0.6]
    coral = (0.95, 0.45, 0.40)
    
    for w, h, a in zip(widths, heights, alphas):
        ell = Ellipse(xy=(0, 0), width=w, height=h, facecolor=coral, alpha=a, edgecolor="none")
        ax.add_patch(ell)
        
    # Draw axes u1 and u2 from center
    ax.annotate("", xy=(2.5, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(2.7, -0.05, r"$\mathbf{u}_1$", fontsize=13, va="center")
    
    ax.annotate("", xy=(0, 1.2), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(0, 1.35, r"$\mathbf{u}_2$", fontsize=13, ha="center")
    
    # Trajectory with momentum: starts at same point (-3.2, 0.55),
    # takes one step down-right, then curves and accelerates smoothly along u1 towards origin!
    pts = [
        [-3.2, 0.55],
        [-2.7, -0.30],
        [-2.1, 0.08],
        [-1.4, 0.02],
        [-0.8, 0.0],
        [-0.3, 0.0],
    ]
    pts = np.array(pts)
    for i in range(len(pts) - 1):
        ax.annotate("", xy=(pts[i+1, 0], pts[i+1, 1]), xytext=(pts[i, 0], pts[i, 1]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.8, shrinkA=0, shrinkB=0))
        
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    
    fig.tight_layout()
    filename = "fig_7_6_momentum_acceleration.png"
    return _save_figure(fig, filename, filepath, save_both=save_both)
