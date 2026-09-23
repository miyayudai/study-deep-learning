"""Chapter 7: Gradient Descent
Section 7.2: Gradient Descent Optimization

This module implements algorithms, statistical analyses, and initialization
strategies for Section 7.2 of:
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.

Contents:
- General Update Form:
  w^(tau) = w^(tau-1) + Delta w^(tau-1) (Eq 7.15)
- Section 7.2.1: Use of Gradient Information
  * Number of independent parameters in quadratic: W(W + 3) / 2
  * Computational effort without gradients: O(W^3)
  * Computational effort with error backpropagation: O(W^2)
- Section 7.2.2: Batch Gradient Descent
  * Batch update: w^(tau) = w^(tau-1) - eta * grad E(w^(tau-1)) (Eq 7.16)
- Section 7.2.3: Stochastic Gradient Descent (SGD)
  * Sum of per-sample errors: E(w) = sum_n E_n(w) (Eq 7.17)
  * Sequential update: w^(tau) = w^(tau-1) - eta * grad E_n(w^(tau-1)) (Eq 7.18)
  * Algorithm 7.1: Stochastic gradient descent
  * Redundancy handling and local minima escape
- Section 7.2.4: Mini-batches
  * Diminishing returns: error in mean scales as sigma / sqrt(B)
  * Hardware power-of-two considerations
  * Algorithm 7.2: Mini-batch stochastic gradient descent
- Section 7.2.5: Parameter Initialization
  * Symmetry breaking problem
  * He initialization for ReLU: epsilon = sqrt(2 / M) (Eq 7.19 - 7.23)
  * Glorot/Xavier initialization for tanh/linear
  * Variance propagation across deep layers
"""

from dataclasses import dataclass, field
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
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
# 7.2.1 Use of Gradient Information: Parameter & Complexity Counting
# ===========================================================================

def count_independent_quadratic_parameters(w_dim: int) -> int:
    """Calculate the number of independent parameters specifying a local quadratic error surface.
    
    In Eq (7.3), E(w) ~= E(w_hat) + b^T(w - w_hat) + 0.5 * (w - w_hat)^T H (w - w_hat),
    the linear vector b has W independent components, and the symmetric Hessian matrix H
    has W(W + 1) / 2 independent components.
    Total = W + W(W + 1)/2 = W(W + 3)/2. (Bishop & Bishop 2024, p. 214; Exercise 7.7)
    
    Args:
        w_dim: Dimensionality of weight vector W (number of parameters).
        
    Returns:
        Number of independent parameters W(W + 3) // 2.
    """
    if w_dim < 1:
        raise ValueError("w_dim must be positive")
    return w_dim * (w_dim + 3) // 2


def gradient_vs_function_eval_effort(w_dim: int) -> Dict[str, int]:
    """Return theoretical computational complexity comparison for finding a quadratic minimum.
    
    - Without gradient info: requires O(W^2) function evaluations, each taking O(W) steps,
      leading to O(W^3) total operations.
    - With error backpropagation: each gradient evaluation takes O(W) steps and provides
      W pieces of information. The minimum can be located in O(W) gradient evaluations,
      totaling O(W^2) steps. (Bishop & Bishop 2024, p. 214)
    
    Args:
        w_dim: Dimensionality of weight vector W.
        
    Returns:
        Dictionary with asymptotic operation counts.
    """
    return {
        "params_count": w_dim,
        "independent_pieces_of_info": count_independent_quadratic_parameters(w_dim),
        "no_grad_evals": w_dim ** 2,
        "no_grad_total_steps": w_dim ** 3,
        "with_grad_evals": w_dim,
        "with_grad_total_steps": w_dim ** 2,
    }


# ===========================================================================
# 7.2.2 - 7.2.4 Optimization Algorithms Data Structures & Implementations
# ===========================================================================

@dataclass
class OptimizationHistory:
    """Record of optimization trajectory and diagnostic metrics."""
    weights: np.ndarray  # Shape: (num_records, W)
    errors: np.ndarray  # Shape: (num_records,)
    grad_norms: np.ndarray  # Shape: (num_records,)
    iterations: int
    epochs: int
    converged: bool
    algorithm_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def batch_gradient_descent(
    w_init: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    error_fn: Optional[Callable[[np.ndarray], float]] = None,
    lr: float = 0.01,
    max_epochs: int = 100,
    tol: float = 1e-6,
    record_every: int = 1,
) -> OptimizationHistory:
    """Batch Gradient Descent (Eq 7.16).
    
    Updates the weight vector in the direction of the greatest rate of decrease
    evaluated over the entire dataset:
        w^(tau) = w^(tau-1) - eta * grad E(w^(tau-1))
        
    Args:
        w_init: Initial parameter vector, shape (W,).
        grad_fn: Function returning gradient of full training set grad E(w), shape (W,).
        error_fn: Optional function returning scalar full training set error E(w).
        lr: Learning rate eta > 0.
        max_epochs: Maximum number of batch updates.
        tol: Convergence tolerance on ||grad E(w)||.
        record_every: Frequency of trajectory snapshots.
        
    Returns:
        OptimizationHistory containing parameter trajectory and errors.
    """
    w = np.array(w_init, dtype=np.float64, copy=True)
    weights_hist = [w.copy()]
    errors_hist = [error_fn(w) if error_fn is not None else float("nan")]
    
    initial_grad = grad_fn(w)
    grad_norms_hist = [float(np.linalg.norm(initial_grad))]
    
    converged = False
    
    for epoch in range(1, max_epochs + 1):
        grad = grad_fn(w)
        norm_grad = float(np.linalg.norm(grad))
        
        if norm_grad < tol:
            converged = True
            weights_hist.append(w.copy())
            err = error_fn(w) if error_fn is not None else float("nan")
            errors_hist.append(err)
            grad_norms_hist.append(norm_grad)
            break
            
        w -= lr * grad
        
        if epoch % record_every == 0 or epoch == max_epochs:
            weights_hist.append(w.copy())
            err = error_fn(w) if error_fn is not None else float("nan")
            errors_hist.append(err)
            grad_norms_hist.append(norm_grad)
            
    return OptimizationHistory(
        weights=np.array(weights_hist),
        errors=np.array(errors_hist),
        grad_norms=np.array(grad_norms_hist),
        iterations=len(weights_hist) - 1,
        epochs=len(weights_hist) - 1,
        converged=converged,
        algorithm_name="Batch Gradient Descent",
        metadata={"lr": lr},
    )


def stochastic_gradient_descent(
    w_init: np.ndarray,
    grad_sample_fn: Callable[[np.ndarray, int], np.ndarray],
    n_samples: int,
    error_fn: Optional[Callable[[np.ndarray], float]] = None,
    lr: float = 0.01,
    max_epochs: int = 10,
    shuffle: bool = False,
    random_state: Optional[int] = None,
    record_every_steps: Optional[int] = None,
) -> OptimizationHistory:
    """Stochastic Gradient Descent (Algorithm 7.1, Eq 7.18).
    
    Updates the weight vector sequentially for one data point at a time:
        w <- w - eta * grad E_n(w)
        n <- n + 1 (mod N)
        
    Args:
        w_init: Initial parameter vector, shape (W,).
        grad_sample_fn: Function returning gradient for sample n: grad E_n(w).
        n_samples: Total number of data samples N.
        error_fn: Optional full dataset error evaluation E(w).
        lr: Learning rate eta > 0.
        max_epochs: Number of complete passes (epochs) through the data.
        shuffle: If True, randomly shuffle indices at each epoch.
        random_state: Seed for random shuffling.
        record_every_steps: How often to record weights (default: once per epoch).
        
    Returns:
        OptimizationHistory with trajectory snapshots.
    """
    rng = np.random.default_rng(random_state)
    w = np.array(w_init, dtype=np.float64, copy=True)
    
    weights_hist = [w.copy()]
    errors_hist = [error_fn(w) if error_fn is not None else float("nan")]
    grad_norms_hist = [0.0]
    
    total_steps = 0
    step_record_interval = record_every_steps if record_every_steps is not None else n_samples
    
    indices = np.arange(n_samples)
    for epoch in range(max_epochs):
        if shuffle:
            rng.shuffle(indices)
            
        for n in indices:
            grad_n = grad_sample_fn(w, int(n))
            w -= lr * grad_n
            total_steps += 1
            
            if total_steps % step_record_interval == 0:
                weights_hist.append(w.copy())
                err = error_fn(w) if error_fn is not None else float("nan")
                errors_hist.append(err)
                grad_norms_hist.append(float(np.linalg.norm(grad_n)))
                
    if total_steps % step_record_interval != 0:
        weights_hist.append(w.copy())
        err = error_fn(w) if error_fn is not None else float("nan")
        errors_hist.append(err)
        grad_norms_hist.append(0.0)
        
    return OptimizationHistory(
        weights=np.array(weights_hist),
        errors=np.array(errors_hist),
        grad_norms=np.array(grad_norms_hist),
        iterations=total_steps,
        epochs=max_epochs,
        converged=False,
        algorithm_name="Stochastic Gradient Descent (Algorithm 7.1)",
        metadata={"lr": lr, "n_samples": n_samples, "shuffle": shuffle},
    )


def minibatch_gradient_descent(
    w_init: np.ndarray,
    grad_minibatch_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_samples: int,
    batch_size: int = 32,
    error_fn: Optional[Callable[[np.ndarray], float]] = None,
    lr: float = 0.01,
    max_epochs: int = 10,
    shuffle: bool = True,
    random_state: Optional[int] = None,
    record_every_steps: Optional[int] = None,
) -> OptimizationHistory:
    """Mini-batch Stochastic Gradient Descent (Algorithm 7.2).
    
    Updates the weight vector using mini-batches of size B:
        w <- w - eta * grad E_{n:n+B-1}(w)
        n <- n + B
        if n > N:
            shuffle data
            n <- 1
            
    Args:
        w_init: Initial parameter vector, shape (W,).
        grad_minibatch_fn: Function accepting (w, batch_indices) and returning
            the gradient evaluated over that mini-batch.
        n_samples: Total number of data samples N.
        batch_size: Size of mini-batch B (powers of two e.g. 32, 64, 128).
        error_fn: Optional full dataset error evaluation E(w).
        lr: Learning rate eta > 0.
        max_epochs: Number of training epochs.
        shuffle: If True, shuffle dataset at the start of each epoch.
        random_state: Seed for random shuffling.
        record_every_steps: Frequency of trajectory snapshots (default: once per epoch).
        
    Returns:
        OptimizationHistory containing trajectory snapshots.
    """
    rng = np.random.default_rng(random_state)
    w = np.array(w_init, dtype=np.float64, copy=True)
    
    weights_hist = [w.copy()]
    errors_hist = [error_fn(w) if error_fn is not None else float("nan")]
    grad_norms_hist = [0.0]
    
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
            w -= lr * grad_b
            total_steps += 1
            
            if total_steps % step_record_interval == 0:
                weights_hist.append(w.copy())
                err = error_fn(w) if error_fn is not None else float("nan")
                errors_hist.append(err)
                grad_norms_hist.append(float(np.linalg.norm(grad_b)))
                
    if total_steps % step_record_interval != 0:
        weights_hist.append(w.copy())
        err = error_fn(w) if error_fn is not None else float("nan")
        errors_hist.append(err)
        grad_norms_hist.append(0.0)
        
    return OptimizationHistory(
        weights=np.array(weights_hist),
        errors=np.array(errors_hist),
        grad_norms=np.array(grad_norms_hist),
        iterations=total_steps,
        epochs=max_epochs,
        converged=False,
        algorithm_name="Mini-batch SGD (Algorithm 7.2)",
        metadata={"lr": lr, "batch_size": batch_size, "shuffle": shuffle},
    )


# ===========================================================================
# 7.2.4 Mini-batch Statistical Noise: Diminishing Returns Analysis
# ===========================================================================

def compute_gradient_noise_vs_batch_size(
    grad_sample_fn: Callable[[np.ndarray, int], np.ndarray],
    w: np.ndarray,
    n_samples: int,
    batch_sizes: List[int],
    num_trials: int = 100,
    random_state: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Verify that sample mean gradient estimation error scales as sigma / sqrt(B).
    
    According to Section 7.2.4 (Bishop & Bishop 2024, p. 216; Exercise 7.8):
    the standard deviation of the error in computing the mean from B samples is:
        sigma_mean = sigma / sqrt(B)
    where sigma is the standard deviation of individual gradient components.
    This gives diminishing returns: increasing batch size by 100x only reduces
    estimation error by 10x.
    
    Args:
        grad_sample_fn: Function returning gradient for sample n.
        w: Weight vector at which gradient is evaluated.
        n_samples: Total number of samples N.
        batch_sizes: List of batch sizes B to evaluate.
        num_trials: Number of random batch draws per batch size.
        random_state: Random seed.
        
    Returns:
        Dictionary containing:
        - 'batch_sizes': array of batch sizes B
        - 'empirical_std': standard deviation of gradient estimate across trials
        - 'theoretical_std': scaled 1 / sqrt(B) theoretical curve
        - 'single_sample_sigma': empirical standard deviation of single sample gradients
    """
    rng = np.random.default_rng(random_state)
    
    # First, compute all individual sample gradients
    all_grads = np.array([grad_sample_fn(w, i) for i in range(n_samples)])  # (N, W)
    true_mean_grad = np.mean(all_grads, axis=0)  # (W,)
    single_sample_sigma = float(np.mean(np.std(all_grads, axis=0)))
    
    empirical_stds = []
    
    for B in batch_sizes:
        batch_means = []
        for _ in range(num_trials):
            sampled_idx = rng.choice(n_samples, size=B, replace=False)
            batch_mean = np.mean(all_grads[sampled_idx], axis=0)
            batch_means.append(batch_mean)
        batch_means = np.array(batch_means)  # (num_trials, W)
        
        # Mean component-wise standard deviation
        std_est = float(np.mean(np.std(batch_means, axis=0)))
        empirical_stds.append(std_est)
        
    empirical_stds = np.array(empirical_stds)
    # Theoretical curve: sigma / sqrt(B)
    theoretical_stds = single_sample_sigma / np.sqrt(np.array(batch_sizes))
    
    return {
        "batch_sizes": np.array(batch_sizes),
        "empirical_std": empirical_stds,
        "theoretical_std": theoretical_stds,
        "single_sample_sigma": np.array(single_sample_sigma),
        "true_mean_grad": true_mean_grad,
    }


# ===========================================================================
# 7.2.5 Parameter Initialization: He, Glorot, and Symmetry Breaking
# ===========================================================================

def he_normal_init(
    fan_in: int,
    fan_out: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """He normal initialization for ReLU activation networks (Eq 7.23).
    
    Weights are sampled from Gaussian N(0, epsilon^2) with:
        epsilon = sqrt(2 / M) = sqrt(2 / fan_in)
    where M is the number of inputs to the unit.
    (He et al., 2015; Bishop & Bishop 2024, p. 218)
    
    Args:
        fan_in: Number of input units M.
        fan_out: Number of output units.
        rng: NumPy random generator.
        
    Returns:
        Weight matrix of shape (fan_out, fan_in).
    """
    if rng is None:
        rng = np.random.default_rng()
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(loc=0.0, scale=std, size=(fan_out, fan_in))


def he_uniform_init(
    fan_in: int,
    fan_out: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """He uniform initialization for ReLU networks.
    
    Sampled from Uniform[-limit, limit] with limit = sqrt(6 / fan_in).
    Variance is limit^2 / 3 = 2 / fan_in.
    
    Args:
        fan_in: Number of input units.
        fan_out: Number of output units.
        rng: NumPy random generator.
        
    Returns:
        Weight matrix of shape (fan_out, fan_in).
    """
    if rng is None:
        rng = np.random.default_rng()
    limit = np.sqrt(6.0 / fan_in)
    return rng.uniform(low=-limit, high=limit, size=(fan_out, fan_in))


def glorot_normal_init(
    fan_in: int,
    fan_out: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Glorot (Xavier) normal initialization for tanh/linear activations.
    
    Sampled from N(0, epsilon^2) with:
        epsilon = sqrt(2 / (fan_in + fan_out))
    (Glorot & Bengio, 2010; Bishop & Bishop 2024, p. 216)
    
    Args:
        fan_in: Number of input units.
        fan_out: Number of output units.
        rng: NumPy random generator.
        
    Returns:
        Weight matrix of shape (fan_out, fan_in).
    """
    if rng is None:
        rng = np.random.default_rng()
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(loc=0.0, scale=std, size=(fan_out, fan_in))


def glorot_uniform_init(
    fan_in: int,
    fan_out: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Glorot (Xavier) uniform initialization.
    
    Sampled from Uniform[-limit, limit] with limit = sqrt(6 / (fan_in + fan_out)).
    
    Args:
        fan_in: Number of input units.
        fan_out: Number of output units.
        rng: NumPy random generator.
        
    Returns:
        Weight matrix of shape (fan_out, fan_in).
    """
    if rng is None:
        rng = np.random.default_rng()
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(low=-limit, high=limit, size=(fan_out, fan_in))


def zero_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Zero initialization (illustrates failure of symmetry breaking).
    
    Args:
        fan_in: Number of input units.
        fan_out: Number of output units.
        
    Returns:
        Weight matrix of shape (fan_out, fan_in) filled with zeros.
    """
    return np.zeros((fan_out, fan_in), dtype=np.float64)


def verify_symmetry_breaking(
    hidden_units: int = 4,
    input_dim: int = 3,
    steps: int = 5,
    lr: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Demonstrate why zero initialization fails to break symmetry.
    
    As explained in Section 7.2.5 (Bishop & Bishop 2024, p. 216):
    "If the parameters were all initialized with the same value, for example
    if they were all set to zero, the parameters of these units would all be
    updated in unison and the units would each compute the same function and
    hence be redundant."
    
    Args:
        hidden_units: Number of hidden units M in single hidden layer network.
        input_dim: Dimensionality of inputs D.
        steps: Number of gradient descent steps.
        lr: Learning rate.
        seed: Random seed for training data.
        
    Returns:
        Dictionary containing weight evolution for zero vs random initialization.
    """
    rng = np.random.default_rng(seed)
    
    # Synthetic dataset
    X = rng.normal(size=(10, input_dim))
    y = rng.normal(size=(10, 1))
    
    # 1. Zero initialization
    W1_zero = np.zeros((hidden_units, input_dim))
    W2_zero = np.zeros((1, hidden_units))
    
    w1_zero_trajectory = [W1_zero.copy()]
    w1_cur = W1_zero.copy()
    w2_cur = W2_zero.copy()
    
    for _ in range(steps):
        # Forward: z = ReLU(X @ W1.T), y_pred = z @ W2.T
        z = np.maximum(0, X @ w1_cur.T)
        y_pred = z @ w2_cur.T
        err = y_pred - y
        
        # Backward gradients
        # dE/dW2 = err.T @ z
        dW2 = err.T @ z
        # dE/dz = err @ W2
        dz = err @ w2_cur
        da = dz * (X @ w1_cur.T > 0)
        dW1 = da.T @ X
        
        w1_cur -= lr * dW1
        w2_cur -= lr * dW2
        w1_zero_trajectory.append(w1_cur.copy())
        
    # Check if rows of W1 are identical for zero initialization
    zero_rows_identical = bool(
        np.allclose(w1_cur[0], w1_cur[1]) and np.allclose(w1_cur[1], w1_cur[2])
    )
    
    # 2. Random He initialization
    W1_rand = he_normal_init(input_dim, hidden_units, rng=rng)
    W2_rand = he_normal_init(hidden_units, 1, rng=rng)
    w1_rand_trajectory = [W1_rand.copy()]
    w1_r_cur = W1_rand.copy()
    w2_r_cur = W2_rand.copy()
    
    for _ in range(steps):
        z = np.maximum(0, X @ w1_r_cur.T)
        y_pred = z @ w2_r_cur.T
        err = y_pred - y
        
        dW2 = err.T @ z
        dz = err @ w2_r_cur
        da = dz * (X @ w1_r_cur.T > 0)
        dW1 = da.T @ X
        
        w1_r_cur -= lr * dW1
        w2_r_cur -= lr * dW2
        w1_rand_trajectory.append(w1_r_cur.copy())
        
    rand_rows_distinct = bool(
        not np.allclose(w1_r_cur[0], w1_r_cur[1]) and not np.allclose(w1_r_cur[1], w1_r_cur[2])
    )
    
    return {
        "zero_final_w1": w1_cur,
        "zero_rows_identical": zero_rows_identical,
        "rand_final_w1": w1_r_cur,
        "rand_rows_distinct": rand_rows_distinct,
    }


def simulate_variance_propagation(
    depth: int = 20,
    width: int = 128,
    init_type: str = "he",
    activation: str = "relu",
    num_samples: int = 2000,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Simulate variance of signals propagating through a deep neural network.
    
    Verifies Eq (7.21) E[a_i^(l)] = 0 and Eq (7.22) var[z_j^(l)] = (M / 2) * epsilon^2 * lambda^2.
    Under He initialization epsilon = sqrt(2 / M), the factor (M / 2) * epsilon^2 = 1,
    so variance is preserved across arbitrary depths: var[z^(l)] ~= var[z^(0)].
    Under standard normal initialization epsilon = 1, variance grows exponentially by (M/2)^l.
    Under Xavier initialization epsilon = sqrt(1 / M), variance decays exponentially by (1/2)^l.
    
    Args:
        depth: Number of layers L.
        width: Number of units M in each layer.
        init_type: 'he', 'xavier', or 'standard_normal'.
        activation: 'relu' or 'linear'.
        num_samples: Number of random input vectors.
        seed: Random seed.
        
    Returns:
        Dictionary with 'pre_act_means', 'pre_act_vars', 'post_act_vars' per layer.
    """
    rng = np.random.default_rng(seed)
    
    # Input has mean 0, variance 1
    z = rng.normal(loc=0.0, scale=1.0, size=(num_samples, width))
    
    pre_act_means = [float(np.mean(z))]
    pre_act_vars = [float(np.var(z))]
    post_act_vars = [float(np.var(z))]
    
    for _ in range(depth):
        if init_type == "he":
            W = he_normal_init(width, width, rng=rng)
        elif init_type == "xavier":
            # Glorot / Xavier variance = 1 / width for tanh/linear
            std = np.sqrt(1.0 / width)
            W = rng.normal(0.0, std, size=(width, width))
        elif init_type == "standard_normal":
            W = rng.normal(0.0, 1.0, size=(width, width))
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
            
        # Linear pre-activation: a = z @ W.T (Eq 7.19)
        a = z @ W.T
        pre_act_means.append(float(np.mean(a)))
        pre_act_vars.append(float(np.var(a)))
        
        # Post-activation z = ReLU(a) (Eq 7.20)
        if activation == "relu":
            z = np.maximum(0.0, a)
        elif activation == "linear":
            z = a
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        post_act_vars.append(float(np.var(z)))
        
    return {
        "pre_act_means": np.array(pre_act_means),
        "pre_act_vars": np.array(pre_act_vars),
        "post_act_vars": np.array(post_act_vars),
        "layers": np.arange(depth + 1),
    }


# ===========================================================================
# Figure & Visualization Generators
# ===========================================================================

def generate_minibatch_noise_figure(
    filepath: Optional[str] = None,
    save_both: bool = True,
) -> Tuple[str, Optional[str]]:
    """Generate and save Figure illustrating diminishing returns in mini-batch gradient estimation.
    
    Compares empirical standard error of the gradient estimate vs theoretical
    curve sigma / sqrt(B) (Section 7.2.4, Bishop & Bishop 2024, p. 216).
    """
    setup_style()
    rng = np.random.default_rng(42)
    
    # Create synthetic dataset with N=1000 samples, W=5 dimensions
    N = 1000
    W = 5
    X = rng.normal(size=(N, W))
    y = X @ rng.normal(size=W) + rng.normal(scale=0.5, size=N)
    w_eval = np.zeros(W)
    
    # Per-sample gradient of squared error: grad E_n(w) = (x_n^T w - y_n) * x_n
    def grad_sample_fn(w: np.ndarray, n: int) -> np.ndarray:
        return (np.dot(X[n], w) - y[n]) * X[n]
        
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    res = compute_gradient_noise_vs_batch_size(
        grad_sample_fn, w_eval, N, batch_sizes, num_trials=100, random_state=42
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Linear scale plot
    ax1.plot(res["batch_sizes"], res["empirical_std"], "o", color="#1f77b4", label="Empirical std error")
    ax1.plot(res["batch_sizes"], res["theoretical_std"], "--", color="#d62728", label=r"Theoretical $\sigma / \sqrt{B}$")
    ax1.set_xlabel("Mini-batch size $B$", fontsize=12)
    ax1.set_ylabel(r"Gradient Estimation Error ($\mathrm{std}$)", fontsize=12)
    ax1.set_title("Diminishing Returns with Batch Size", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Log-Log scale plot showing slope -1/2
    ax2.loglog(res["batch_sizes"], res["empirical_std"], "o", color="#1f77b4", label="Empirical std error")
    ax2.loglog(res["batch_sizes"], res["theoretical_std"], "--", color="#d62728", label=r"Slope $-1/2$ ($\propto B^{-1/2}$)")
    ax2.set_xlabel("Mini-batch size $B$ (log scale)", fontsize=12)
    ax2.set_ylabel("Estimation Error (log scale)", fontsize=12)
    ax2.set_title(r"Log-Log Scaling $\mathcal{O}(B^{-1/2})$", fontsize=13)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=11)
    
    fig.tight_layout()
    filename = "fig_7_2_minibatch_noise.png"
    return _save_figure(fig, filename, filepath, save_both=save_both)


def generate_variance_propagation_figure(
    filepath: Optional[str] = None,
    save_both: bool = True,
) -> Tuple[str, Optional[str]]:
    """Generate and save Figure illustrating signal variance propagation in deep ReLU networks.
    
    Compares He initialization (Eq 7.23, preserved variance) with standard normal
    (exponential explosion) and Xavier initialization (exponential decay).
    """
    setup_style()
    depth = 20
    width = 64
    
    res_he = simulate_variance_propagation(depth=depth, width=width, init_type="he", activation="relu")
    res_xavier = simulate_variance_propagation(depth=depth, width=width, init_type="xavier", activation="relu")
    res_std = simulate_variance_propagation(depth=depth, width=width, init_type="standard_normal", activation="relu")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Semilog plot of post-activation variance
    ax1.semilogy(res_he["layers"], res_he["post_act_vars"], "o-", color="#2ca02c", label=r"He Normal ($\sigma = \sqrt{2/M}$)")
    ax1.semilogy(res_xavier["layers"], res_xavier["post_act_vars"], "s-", color="#1f77b4", label=r"Xavier / Glorot ($\sigma = \sqrt{1/M}$)")
    ax1.semilogy(res_std["layers"][:8], res_std["post_act_vars"][:8], "^-", color="#d62728", label=r"Standard Normal ($\sigma = 1$)")
    ax1.set_xlabel("Layer index $l$", fontsize=12)
    ax1.set_ylabel(r"$\mathrm{var}[z^{(l)}]$ (log scale)", fontsize=12)
    ax1.set_title("Post-Activation Variance vs Network Depth", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Pre-activation mean verification (E[a] ~= 0)
    ax2.plot(res_he["layers"], res_he["pre_act_means"], "o-", color="#2ca02c", label=r"He $\mathbb{E}[a^{(l)}]$")
    ax2.axhline(0.0, linestyle="--", color="black", alpha=0.6, label=r"Theoretical $\mathbb{E}[a] = 0$")
    ax2.set_xlabel("Layer index $l$", fontsize=12)
    ax2.set_ylabel(r"Pre-activation Mean $\mathbb{E}[a^{(l)}]$", fontsize=12)
    ax2.set_title(r"Preservation of Zero Mean (Eq 7.21)", fontsize=13)
    ax2.set_ylim(-0.5, 0.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    fig.tight_layout()
    filename = "fig_7_2_variance_propagation.png"
    return _save_figure(fig, filename, filepath, save_both=save_both)


def generate_gd_comparison_figure(
    filepath: Optional[str] = None,
    save_both: bool = True,
) -> Tuple[str, Optional[str]]:
    """Generate and save Figure comparing Batch GD, SGD, and Mini-batch GD trajectories and loss.
    
    Visualizes trajectories on a 2D synthetic least-squares problem along with
    their convergence loss curves.
    """
    setup_style()
    rng = np.random.default_rng(42)
    
    N = 200
    w_true = np.array([1.5, -0.8])
    X = rng.normal(size=(N, 2))
    # Add correlation to create elliptical contours
    X[:, 1] = 0.5 * X[:, 0] + 0.8 * X[:, 1]
    y = X @ w_true + rng.normal(scale=0.2, size=N)
    
    def error_fn(w: np.ndarray) -> float:
        err = X @ w - y
        return 0.5 * float(np.mean(err ** 2))
        
    def batch_grad_fn(w: np.ndarray) -> np.ndarray:
        return (X.T @ (X @ w - y)) / N
        
    def sample_grad_fn(w: np.ndarray, n: int) -> np.ndarray:
        return (np.dot(X[n], w) - y[n]) * X[n]
        
    def minibatch_grad_fn(w: np.ndarray, batch_idx: np.ndarray) -> np.ndarray:
        X_b = X[batch_idx]
        y_b = y[batch_idx]
        return (X_b.T @ (X_b @ w - y_b)) / len(batch_idx)
        
    w0 = np.array([-1.5, 1.5])
    
    # 1. Batch GD
    hist_batch = batch_gradient_descent(
        w0, batch_grad_fn, error_fn=error_fn, lr=0.1, max_epochs=40
    )
    # 2. SGD
    hist_sgd = stochastic_gradient_descent(
        w0, sample_grad_fn, n_samples=N, error_fn=error_fn, lr=0.02, max_epochs=5,
        shuffle=True, random_state=42, record_every_steps=10
    )
    # 3. MiniBatch GD (B=16)
    hist_mb = minibatch_gradient_descent(
        w0, minibatch_grad_fn, n_samples=N, batch_size=16, error_fn=error_fn,
        lr=0.08, max_epochs=10, shuffle=True, random_state=42, record_every_steps=2
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Contour plot with trajectories
    w1_grid = np.linspace(-2.0, 2.5, 100)
    w2_grid = np.linspace(-1.5, 2.0, 100)
    W1, W2 = np.meshgrid(w1_grid, w2_grid)
    Z = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            Z[i, j] = error_fn(np.array([W1[i, j], W2[i, j]]))
            
    contours = ax1.contour(W1, W2, Z, levels=20, cmap="viridis", alpha=0.6)
    ax1.plot(hist_batch.weights[:, 0], hist_batch.weights[:, 1], "o-", color="#1f77b4", label="Batch GD", markersize=3)
    ax1.plot(hist_sgd.weights[:, 0], hist_sgd.weights[:, 1], ".-", color="#d62728", alpha=0.7, label="SGD (Alg 7.1)", markersize=2)
    ax1.plot(hist_mb.weights[:, 0], hist_mb.weights[:, 1], ".-", color="#2ca02c", label="Mini-batch GD (Alg 7.2)", markersize=3)
    ax1.plot(w_true[0], w_true[1], "k*", markersize=12, label=r"Optimal $\mathbf{w}^\star$")
    ax1.plot(w0[0], w0[1], "ks", markersize=8, label=r"Start $\mathbf{w}^{(0)}$")
    ax1.set_xlabel("$w_1$", fontsize=12)
    ax1.set_ylabel("$w_2$", fontsize=12)
    ax1.set_title("Optimization Trajectories in Parameter Space", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Loss convergence curves vs iteration step
    ax2.plot(hist_batch.errors, "o-", color="#1f77b4", label="Batch GD", markersize=3)
    ax2.plot(hist_sgd.errors, ".-", color="#d62728", alpha=0.7, label="SGD", markersize=2)
    ax2.plot(hist_mb.errors, ".-", color="#2ca02c", label="Mini-batch GD", markersize=3)
    ax2.set_xlabel("Recorded Steps", fontsize=12)
    ax2.set_ylabel(r"Error $E(\mathbf{w})$", fontsize=12)
    ax2.set_title("Error Convergence Comparison", fontsize=13)
    ax2.set_yscale("log")
    ax2.legend(fontsize=10)
    ax2.grid(True, which="both", alpha=0.3)
    
    fig.tight_layout()
    filename = "fig_7_2_gd_trajectories.png"
    return _save_figure(fig, filename, filepath, save_both=save_both)
