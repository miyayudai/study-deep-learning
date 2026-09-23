"""Chapter 7: Gradient Descent
Exercises 7.1 to 7.14 helper functions, theoretical proofs, and numerical verifications.

Bishop & Bishop (2024), Deep Learning: Foundations and Concepts, pp. 230-232.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize_scalar


# ===========================================================================
# Exercise 7.1: Quadratic Error in Eigen-Basis Coordinates (Eq 7.11)
# ===========================================================================

def exercise_7_1_verify_eigen_quadratic_form(
    H: np.ndarray,
    w: np.ndarray,
    w_star: np.ndarray,
    E_star: float = 0.0,
) -> Dict[str, Any]:
    """Exercise 7.1: Verify decoupled eigen-basis error representation (Eq 7.11).
    
    E(w) = E(w*) + 0.5 * (w - w*)^T H (w - w*)
         = E(w*) + 0.5 * sum_i lambda_i xi_i^2
    where xi_i = u_i^T (w - w*) are coordinates in the orthonormal eigen-basis {u_i}.
    """
    H_sym = 0.5 * (H + H.T)
    evals, evecs = np.linalg.eigh(H_sym)
    diff = w - w_star
    
    # Direct matrix quadratic form
    quad_direct = float(0.5 * diff.T @ H_sym @ diff)
    error_direct = E_star + quad_direct
    
    # Decoupled eigen-coordinate sum
    xi = evecs.T @ diff  # xi_i = u_i^T (w - w*)
    quad_eigen = float(0.5 * np.sum(evals * (xi ** 2)))
    error_eigen = E_star + quad_eigen
    
    return {
        "error_direct": error_direct,
        "error_eigen": error_eigen,
        "eigenvalues": evals,
        "eigen_coords_xi": xi,
        "is_equivalent": bool(np.isclose(error_direct, error_eigen, atol=1e-8)),
    }


# ===========================================================================
# Exercise 7.2: Positive Definite Hessian and Positive Eigenvalues (Eq 7.14)
# ===========================================================================

def exercise_7_2_positive_definite_eigenvalues(H: np.ndarray) -> Dict[str, Any]:
    """Exercise 7.2: Verify H is positive definite iff all eigenvalues lambda_i > 0.
    
    For arbitrary v = sum_i c_i u_i:
        v^T H v = sum_i c_i^2 lambda_i (Eq 7.14)
    If all lambda_i > 0, then for any v != 0 (at least one c_i != 0), v^T H v > 0.
    Conversely, choosing v = u_i gives u_i^T H u_i = lambda_i, which must be > 0.
    """
    H_sym = 0.5 * (H + H.T)
    evals, evecs = np.linalg.eigh(H_sym)
    all_positive = bool(np.all(evals > 0))
    
    # Test with random non-zero vectors
    rng = np.random.default_rng(42)
    V = rng.normal(size=(20, H.shape[0]))
    quadratic_forms = np.array([float(v.T @ H_sym @ v) for v in V])
    all_quadratics_positive = bool(np.all(quadratic_forms > 0))
    
    return {
        "eigenvalues": evals,
        "all_eigenvalues_positive": all_positive,
        "empirical_v_quadratics_positive": all_quadratics_positive,
        "is_positive_definite": all_positive,
    }


# ===========================================================================
# Exercise 7.3: Local Minimum Necessary and Sufficient Conditions
# ===========================================================================

def exercise_7_3_local_minimum_conditions(
    grad: np.ndarray,
    H: np.ndarray,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """Exercise 7.3: Check necessary and sufficient condition for local minimum.
    
    Condition: grad E(w*) = 0 and H(w*) is positive definite (all lambda_i > 0).
    """
    grad_norm = float(np.linalg.norm(grad))
    is_stationary = bool(grad_norm < tol)
    
    evals = np.linalg.eigvalsh(0.5 * (H + H.T))
    is_positive_definite = bool(np.all(evals > tol))
    is_local_minimum = bool(is_stationary and is_positive_definite)
    
    return {
        "grad_norm": grad_norm,
        "is_stationary": is_stationary,
        "eigenvalues": evals,
        "is_positive_definite": is_positive_definite,
        "is_local_minimum": is_local_minimum,
    }


# ===========================================================================
# Exercise 7.4: Linear Regression Hessian, Trace, and Determinant (Eq 7.61, 7.62)
# ===========================================================================

def exercise_7_4_linear_regression_hessian(x: np.ndarray) -> Dict[str, Any]:
    """Exercise 7.4: Derive and analyze 2x2 Hessian for linear regression.
    
    Model: y = wx + b. Error: E(w, b) = 0.5 * sum_n (wx_n + b - t_n)^2.
    First derivatives:
        del E / del w = sum_n (wx_n + b - t_n) x_n
        del E / del b = sum_n (wx_n + b - t_n)
    Second derivatives (Hessian elements):
        del^2 E / del w^2 = sum_n x_n^2
        del^2 E / del b^2 = N
        del^2 E / del w del b = sum_n x_n
        
    H = [[sum x_n^2, sum x_n], [sum x_n, N]]
    Trace(H) = sum x_n^2 + N > 0.
    Det(H) = N * sum x_n^2 - (sum x_n)^2 = N * sum (x_n - x_bar)^2 >= 0.
    Strictly positive if x has at least 2 distinct values (variance > 0).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    N = len(x_arr)
    sum_x2 = float(np.sum(x_arr ** 2))
    sum_x = float(np.sum(x_arr))
    
    H = np.array([[sum_x2, sum_x], [sum_x, float(N)]])
    trace_val = float(np.trace(H))
    det_val = float(np.linalg.det(H))
    
    # Theoretical variance-based determinant: N * sum (x_n - x_bar)^2
    x_bar = np.mean(x_arr)
    det_theoretical = float(N * np.sum((x_arr - x_bar) ** 2))
    
    evals = np.linalg.eigvalsh(H)
    
    return {
        "Hessian": H,
        "trace": trace_val,
        "determinant": det_val,
        "det_theoretical": det_theoretical,
        "eigenvalues": evals,
        "trace_positive": bool(trace_val > 0),
        "det_positive": bool(det_val > 0),
        "is_minimum": bool(np.all(evals > 0)),
    }


# ===========================================================================
# Exercise 7.5: Logistic Classification Hessian, Trace, and Determinant (Eq 7.63, 7.64)
# ===========================================================================

def exercise_7_5_logistic_regression_hessian(
    x: np.ndarray,
    w: float,
    b: float,
) -> Dict[str, Any]:
    """Exercise 7.5: Derive and analyze 2x2 Hessian for logistic classification.
    
    Model: y = sigma(wx + b). Cross-entropy: E = -sum_n [t_n ln y_n + (1 - t_n) ln(1 - y_n)].
    First derivatives:
        del E / del w = sum_n (y_n - t_n) x_n
        del E / del b = sum_n (y_n - t_n)
    Second derivatives with r_n = y_n(1 - y_n) > 0:
        del^2 E / del w^2 = sum_n r_n x_n^2
        del^2 E / del b^2 = sum_n r_n
        del^2 E / del w del b = sum_n r_n x_n
        
    H = [[sum r_n x_n^2, sum r_n x_n], [sum r_n x_n, sum r_n]]
    Trace(H) = sum r_n (x_n^2 + 1) > 0.
    Det(H) = (sum r_n)(sum r_n x_n^2) - (sum r_n x_n)^2 > 0 by Cauchy-Schwarz inequality.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    a = w * x_arr + b
    y = 1.0 / (1.0 + np.exp(-np.clip(a, -30.0, 30.0)))
    r = y * (1.0 - y)  # r_n in (0, 0.25]
    
    H00 = float(np.sum(r * (x_arr ** 2)))
    H01 = float(np.sum(r * x_arr))
    H11 = float(np.sum(r))
    H = np.array([[H00, H01], [H01, H11]])
    
    trace_val = float(np.trace(H))
    det_val = float(np.linalg.det(H))
    
    evals = np.linalg.eigvalsh(H)
    return {
        "Hessian": H,
        "trace": trace_val,
        "determinant": det_val,
        "eigenvalues": evals,
        "trace_positive": bool(trace_val > 0),
        "det_positive": bool(det_val > 0),
        "is_minimum": bool(np.all(evals > 0)),
    }


# ===========================================================================
# Exercise 7.6: Elliptical Error Contours and Semi-Axis Lengths (Figure 7.2)
# ===========================================================================

def exercise_7_6_ellipse_axes_and_lengths(
    H: np.ndarray,
    delta_E: float = 1.0,
) -> Dict[str, Any]:
    """Exercise 7.6: Contours of constant error as axis-aligned ellipses in eigen-basis.
    
    In eigen-coordinates xi_i, the level set E(w) - E* = delta_E satisfies:
        0.5 * sum_i lambda_i xi_i^2 = delta_E
        sum_i xi_i^2 / (2 * delta_E / lambda_i) = 1
    Semi-axis lengths along eigenvectors u_i:
        r_i = sqrt(2 * delta_E / lambda_i) proportional to lambda_i^(-1/2).
    """
    H_sym = 0.5 * (H + H.T)
    evals, evecs = np.linalg.eigh(H_sym)
    if np.any(evals <= 0):
        raise ValueError("Hessian must be positive definite for bounded elliptical contours")
        
    semi_axis_lengths = np.sqrt(2.0 * delta_E / evals)
    return {
        "eigenvalues": evals,
        "eigenvectors": evecs,
        "semi_axis_lengths": semi_axis_lengths,
        "proportional_to_inv_sqrt": np.allclose(
            semi_axis_lengths / np.sqrt(2.0 * delta_E), 1.0 / np.sqrt(evals)
        ),
    }


# ===========================================================================
# Exercise 7.7: Quadratic Error Function Parameter Counting
# ===========================================================================

def exercise_7_7_quadratic_independent_parameters(W: int) -> Dict[str, int]:
    """Exercise 7.7: Count independent elements in quadratic approximation (Eq 7.3).
    
    E(w) ~= E(w_hat) + b^T (w - w_hat) + 0.5 * (w - w_hat)^T H (w - w_hat)
    - Vector b: W independent elements
    - Symmetric matrix H: W(W + 1) / 2 independent elements
    Total = W + W(W + 1) / 2 = W(W + 3) / 2.
    """
    if W < 1:
        raise ValueError("Parameter dimensionality W must be >= 1")
    linear_terms = W
    quadratic_terms = W * (W + 1) // 2
    total = linear_terms + quadratic_terms
    formula_total = W * (W + 3) // 2
    return {
        "W": W,
        "linear_elements": linear_terms,
        "hessian_elements": quadratic_terms,
        "total_independent_elements": total,
        "formula_W_W_plus_3_over_2": formula_total,
        "matches_formula": int(total == formula_total),
    }


# ===========================================================================
# Exercise 7.8: Sample Mean Error and Diminishing Returns (Eq 7.65)
# ===========================================================================

def exercise_7_8_sample_mean_variance(
    N: int,
    sigma: float,
    num_experiments: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Exercise 7.8: Expectation of squared error E[(x_bar - mu)^2] = sigma^2 / N.
    
    RMS error = sigma / sqrt(N).
    """
    rng = np.random.default_rng(seed)
    mu = 5.0
    # Generate samples of size (num_experiments, N)
    samples = rng.normal(loc=mu, scale=sigma, size=(num_experiments, N))
    sample_means = np.mean(samples, axis=1)
    
    empirical_mse = float(np.mean((sample_means - mu) ** 2))
    theoretical_mse = float((sigma ** 2) / N)
    empirical_rms = float(np.sqrt(empirical_mse))
    theoretical_rms = float(sigma / np.sqrt(N))
    
    return {
        "N": N,
        "sigma": sigma,
        "empirical_mse": empirical_mse,
        "theoretical_mse": theoretical_mse,
        "empirical_rms": empirical_rms,
        "theoretical_rms": theoretical_rms,
        "relative_error": float(abs(empirical_mse - theoretical_mse) / theoretical_mse),
    }


# ===========================================================================
# Exercise 7.9: He Initialization and ReLU Variance Propagation (Eq 7.21 - 7.23)
# ===========================================================================

def exercise_7_9_relu_variance_propagation(
    M: int,
    input_variance: float = 1.0,
    weight_variance: Optional[float] = None,
    num_samples: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Exercise 7.9: Derive and verify E[a_i] = 0 and E[(z_j)^2] = (M/2) * epsilon^2 * lambda^2.
    
    He initialization condition: (M / 2) * epsilon^2 = 1 => epsilon = sqrt(2 / M).
    """
    rng = np.random.default_rng(seed)
    epsilon = np.sqrt(2.0 / M) if weight_variance is None else np.sqrt(weight_variance)
    
    # Input z has mean 0, variance lambda^2
    z = rng.normal(0.0, np.sqrt(input_variance), size=(num_samples, M))
    W = rng.normal(0.0, epsilon, size=(num_samples, M))
    
    # a_n = sum_j w_{nj} z_{nj} (Eq 7.19)
    a = np.sum(z * W, axis=1)
    # z_out = ReLU(a) (Eq 7.20)
    z_out = np.maximum(0.0, a)
    
    theoretical_a_mean = 0.0
    theoretical_z_second_moment = (M / 2.0) * (epsilon ** 2) * input_variance
    
    empirical_a_mean = float(np.mean(a))
    empirical_z_second_moment = float(np.mean(z_out ** 2))
    
    return {
        "M": M,
        "epsilon": float(epsilon),
        "he_epsilon_formula": float(np.sqrt(2.0 / M)),
        "empirical_a_mean": empirical_a_mean,
        "theoretical_a_mean": theoretical_a_mean,
        "empirical_z_var": empirical_z_second_moment,
        "theoretical_z_var": theoretical_z_second_moment,
        "empirical_z_second_moment": empirical_z_second_moment,
        "theoretical_z_second_moment": theoretical_z_second_moment,
        "variance_preserved": bool(np.isclose(empirical_z_second_moment, input_variance, rtol=0.10)),
    }


# ===========================================================================
# Exercise 7.10: Decoupled Eigen-Update Derivation (Eq 7.24 - 7.27)
# ===========================================================================

def exercise_7_10_eigen_update_derivation(
    H: np.ndarray,
    w: np.ndarray,
    w_star: np.ndarray,
    lr: float = 0.1,
) -> Dict[str, Any]:
    """Exercise 7.10: Verify Delta alpha_i = -eta * lambda_i * alpha_i (Eq 7.26).
    
    Using orthonormality u_i^T u_j = delta_ij:
        grad E = sum_i alpha_i lambda_i u_i (Eq 7.24)
        Delta w = sum_i Delta alpha_i u_i (Eq 7.25)
        Delta w = -eta grad E => Delta alpha_i = -eta lambda_i alpha_i.
    """
    H_sym = 0.5 * (H + H.T)
    evals, evecs = np.linalg.eigh(H_sym)
    
    # alpha_old = evecs.T @ (w - w_star)
    alpha_old = evecs.T @ (w - w_star)
    
    # Gradient in normal space: grad E = H (w - w_star)
    grad = H_sym @ (w - w_star)
    # Weight update: w_new = w - lr * grad
    w_new = w - lr * grad
    alpha_new = evecs.T @ (w_new - w_star)
    
    # Theoretical decoupled formula: alpha_new_theory = (1 - lr * lambda_i) * alpha_old
    alpha_new_theory = (1.0 - lr * evals) * alpha_old
    delta_alpha_theory = -lr * evals * alpha_old
    delta_alpha_actual = alpha_new - alpha_old
    
    return {
        "alpha_old": alpha_old,
        "alpha_new_actual": alpha_new,
        "alpha_new_theory": alpha_new_theory,
        "delta_alpha_actual": delta_alpha_actual,
        "delta_alpha_theory": delta_alpha_theory,
        "is_exact_match": bool(np.allclose(alpha_new, alpha_new_theory, atol=1e-8)),
    }


# ===========================================================================
# Exercise 7.11: Nesterov Momentum First-Order Equivalence (Eq 7.34 vs 7.31)
# ===========================================================================

def exercise_7_11_nesterov_momentum_equivalence(
    w: np.ndarray,
    delta_w_prev: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    hessian_fn: Callable[[np.ndarray], np.ndarray],
    lr: float = 0.01,
    mu: float = 0.9,
) -> Dict[str, Any]:
    """Exercise 7.11: Show Nesterov momentum is equivalent to standard momentum to first order.
    
    Standard Momentum:
        Delta w_std = -eta * grad E(w) + mu * Delta w_prev (Eq 7.31)
    Nesterov Momentum:
        Delta w_nag = -eta * grad E(w + mu * Delta w_prev) + mu * Delta w_prev (Eq 7.34)
    Taylor expansion:
        grad E(w + mu * Delta w_prev) ~= grad E(w) + mu * H * Delta w_prev
        Delta w_nag ~= -eta * grad E(w) + mu * Delta w_prev - eta * mu * H * Delta w_prev
    Difference = -eta * mu * H * Delta w_prev = O(eta * mu).
    """
    grad_current = grad_fn(w)
    delta_w_std = -lr * grad_current + mu * delta_w_prev
    
    grad_lookahead = grad_fn(w + mu * delta_w_prev)
    delta_w_nag = -lr * grad_lookahead + mu * delta_w_prev
    
    diff_actual = delta_w_nag - delta_w_std
    H = hessian_fn(w)
    diff_first_order_theory = -lr * mu * (H @ delta_w_prev)
    
    return {
        "delta_w_std": delta_w_std,
        "delta_w_nag": delta_w_nag,
        "diff_actual_norm": float(np.linalg.norm(diff_actual)),
        "diff_theory_norm": float(np.linalg.norm(diff_first_order_theory)),
        "is_close_first_order": bool(np.allclose(diff_actual, diff_first_order_theory, atol=1e-3)),
    }


# ===========================================================================
# Exercise 7.12: Exponentially Weighted Moving Average Bias Correction (Eq 7.66 - 7.68)
# ===========================================================================

def exercise_7_12_ema_bias_correction(
    beta: float,
    num_steps: int = 20,
    true_mean: float = 5.0,
) -> Dict[str, Any]:
    """Exercise 7.12: Derive and verify EMA bias correction factor 1 / (1 - beta^n).
    
    mu_n = beta * mu_{n-1} + (1 - beta) * x_n with mu_0 = 0.
    Unfolding gives mu_n = (1 - beta) * sum_{k=1}^n beta^{n-k} x_k.
    If E[x_k] = mu, then E[mu_n] = (1 - beta) * mu * sum_{k=1}^n beta^{n-k}
                                  = (1 - beta) * mu * (1 - beta^n) / (1 - beta)
                                  = (1 - beta^n) * mu.
    Therefore, mu_hat_n = mu_n / (1 - beta^n) satisfies E[mu_hat_n] = mu for all n.
    """
    steps = np.arange(1, num_steps + 1)
    
    # Constant input x_k = true_mean to inspect expectation directly
    mu_raw = np.zeros(num_steps)
    val = 0.0
    for i in range(num_steps):
        val = beta * val + (1.0 - beta) * true_mean
        mu_raw[i] = val
        
    bias_correction_factors = 1.0 / (1.0 - (beta ** steps))
    mu_corrected = mu_raw * bias_correction_factors
    
    return {
        "steps": steps,
        "mu_raw": mu_raw,
        "mu_corrected": mu_corrected,
        "true_mean": true_mean,
        "bias_corrected_is_exact": bool(np.allclose(mu_corrected, true_mean)),
    }


# ===========================================================================
# Exercise 7.13: Line Search Orthogonality Condition (Eq 7.69)
# ===========================================================================

def exercise_7_13_line_search_orthogonality(
    error_fn: Callable[[np.ndarray], float],
    grad_fn: Callable[[np.ndarray], np.ndarray],
    w: np.ndarray,
    d: np.ndarray,
) -> Dict[str, Any]:
    """Exercise 7.13: Show that gradient at line search minimum is orthogonal to search direction d.
    
    Let f(lambda) = E(w + lambda * d).
    At minimum lambda*:
        df / d lambda = d^T grad E(w + lambda* * d) = 0.
    """
    def f_lambda(lam: float) -> float:
        return error_fn(w + lam * d)
        
    res = minimize_scalar(f_lambda, bracket=(0.0, 1.0))
    lambda_star = float(res.x)
    
    w_new = w + lambda_star * d
    grad_new = grad_fn(w_new)
    directional_derivative = float(np.dot(d, grad_new))
    
    return {
        "lambda_star": lambda_star,
        "w_new": w_new,
        "grad_new": grad_new,
        "directional_derivative": directional_derivative,
        "is_orthogonal": bool(np.isclose(directional_derivative, 0.0, atol=1e-5)),
    }


# ===========================================================================
# Exercise 7.14: Renormalized Input Zero Mean and Unit Variance (Eq 7.50)
# ===========================================================================

def exercise_7_14_standardization_moments(x: np.ndarray) -> Dict[str, Any]:
    """Exercise 7.14: Show that x_tilde_n = (x_n - mu) / sigma has zero mean and unit variance.
    
    Mean:
        (1/N) * sum_n x_tilde_n = (1/N) * sum_n (x_n - mu) / sigma
                                = (1/sigma) * [ (1/N) * sum_n x_n - mu ]
                                = (1/sigma) * [ mu - mu ] = 0.
    Variance:
        (1/N) * sum_n x_tilde_n^2 = (1/N) * sum_n (x_n - mu)^2 / sigma^2
                                  = (1 / sigma^2) * [ (1/N) * sum_n (x_n - mu)^2 ]
                                  = sigma^2 / sigma^2 = 1.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    N = len(x_arr)
    mu = float(np.mean(x_arr))
    var = float(np.var(x_arr))
    sigma = float(np.sqrt(var))
    
    x_tilde = (x_arr - mu) / sigma
    mean_tilde = float(np.mean(x_tilde))
    var_tilde = float(np.var(x_tilde))
    
    return {
        "original_mean": mu,
        "original_var": var,
        "normalized_mean": mean_tilde,
        "normalized_var": var_tilde,
        "is_zero_mean": bool(np.isclose(mean_tilde, 0.0, atol=1e-7)),
        "is_unit_variance": bool(np.isclose(var_tilde, 1.0, atol=1e-7)),
    }
