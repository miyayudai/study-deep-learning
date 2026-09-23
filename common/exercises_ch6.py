"""Chapter 6: Deep Neural Networks
Exercises 6.1 to 6.21 helper functions and numerical verification.

Bishop & Bishop (2024), Deep Learning: Foundations and Concepts, pp. 204-207.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import integrate, special


# ---------------------------------------------------------------------------
# Exercise 6.1: Hypersphere Surface Area S_D and Volume V_D
# ---------------------------------------------------------------------------

def exercise_6_1_hypersphere_surface(D: int) -> float:
    """Exercise 6.1: Surface area S_D of a unit hypersphere in D dimensions (Eq 6.53).
    
    S_D = 2 * pi^(D/2) / Gamma(D/2)
    """
    return float(2.0 * (np.pi ** (D / 2.0)) / special.gamma(D / 2.0))


def exercise_6_1_hypersphere_volume(D: int) -> float:
    """Exercise 6.1: Volume V_D of a unit hypersphere in D dimensions (Eq 6.54).
    
    V_D = S_D / D = pi^(D/2) / Gamma(D/2 + 1)
    """
    return float(exercise_6_1_hypersphere_surface(D) / D)


def exercise_6_1_verify_polar_integral(D: int) -> Tuple[float, float]:
    """Exercise 6.1: Verify polar coordinate transformation (Eq 6.51).
    
    LHS: prod_{i=1}^D int_{-inf}^inf exp(-x_i^2) dx_i = pi^(D/2)
    RHS: S_D * int_0^inf exp(-r^2) * r^(D-1) dr
    """
    lhs = float(np.pi ** (D / 2.0))
    # Radial integral: int_0^inf exp(-r^2) * r^(D-1) dr = 0.5 * Gamma(D/2)
    integral_val, _ = integrate.quad(lambda r: np.exp(-r ** 2) * (r ** (D - 1)), 0.0, np.inf)
    rhs = float(exercise_6_1_hypersphere_surface(D) * integral_val)
    return lhs, rhs


# ---------------------------------------------------------------------------
# Exercise 6.2: Ratio of Volume of Hypersphere to Hypercube and Long Spikes
# ---------------------------------------------------------------------------

def exercise_6_2_sphere_to_cube_volume_ratio(D: int) -> float:
    """Exercise 6.2: Volume ratio of hypersphere of radius a to hypercube of side 2a (Eq 6.55).
    
    Ratio = V_D * a^D / (2a)^D = pi^(D/2) / (D * 2^(D-1) * Gamma(D/2))
    """
    num = np.pi ** (D / 2.0)
    denom = D * (2.0 ** (D - 1)) * special.gamma(D / 2.0)
    return float(num / denom)


def exercise_6_2_stirling_ratio_approximation(D: int) -> float:
    """Exercise 6.2: Stirling approximation of volume ratio as D -> infty."""
    # V_D / 2^D = pi^(D/2) / (2^D * Gamma(D/2 + 1))
    # With Gamma(x+1) ~ sqrt(2*pi) * e^(-x) * x^(x+0.5), where x = D/2:
    x = D / 2.0
    gamma_approx = np.sqrt(2.0 * np.pi) * np.exp(-x) * (x ** (x + 0.5))
    ratio_approx = (np.pi ** x) / ((2.0 ** D) * gamma_approx)
    return float(ratio_approx)


def exercise_6_2_corner_distance_ratio(D: int) -> float:
    """Exercise 6.2: Ratio of center-to-corner distance to center-to-face distance in hypercube."""
    # Center to corner is sqrt(D)*a, center to face is a. Ratio is sqrt(D).
    return float(np.sqrt(D))


# ---------------------------------------------------------------------------
# Exercise 6.3: Radial Density of High-Dimensional Gaussian and Shell Concentration
# ---------------------------------------------------------------------------

def exercise_6_3_gaussian_radial_density(r: Union[float, np.ndarray], D: int, sigma: float = 1.0) -> Union[float, np.ndarray]:
    """Exercise 6.3: Radial probability density p(r) of D-dimensional Gaussian (Eq 6.58).
    
    p(r) = S_D * r^(D-1) / (2 * pi * sigma^2)^(D/2) * exp(-r^2 / (2 * sigma^2))
    """
    r_arr = np.asarray(r, dtype=np.float64)
    S_D = exercise_6_1_hypersphere_surface(D)
    norm_const = (2.0 * np.pi * (sigma ** 2)) ** (D / 2.0)
    val = (S_D * (r_arr ** (D - 1)) / norm_const) * np.exp(- (r_arr ** 2) / (2.0 * (sigma ** 2)))
    if np.ndim(r) == 0:
        return float(val)
    return val


def exercise_6_3_radial_mode(D: int, sigma: float = 1.0) -> float:
    """Exercise 6.3: Exact stationary point r_hat = sigma * sqrt(D - 1)."""
    if D <= 1:
        return 0.0
    return float(sigma * np.sqrt(D - 1))


def exercise_6_3_density_ratio_origin_to_mode(D: int) -> float:
    """Exercise 6.3: Ratio of Cartesian probability density p(x=0) to p(x with ||x||=r_hat).
    
    p(0) / p(r_hat) = exp(r_hat^2 / (2 * sigma^2)) = exp((D - 1) / 2) ~ exp(D / 2).
    """
    return float(np.exp((D - 1.0) / 2.0))


# ---------------------------------------------------------------------------
# Exercise 6.4: Sigmoid and Tanh Hidden Units Equivalence
# ---------------------------------------------------------------------------

def exercise_6_4_convert_sigmoid_to_tanh_weights(
    W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exercise 6.4: Convert 2-layer MLP weights from logistic sigmoid to tanh activation.
    
    Given:
      y = W2 @ sigmoid(W1 @ x + b1) + b2
    Using sigmoid(a) = 0.5 + 0.5 * tanh(a / 2):
      W1_tilde = 0.5 * W1
      b1_tilde = 0.5 * b1
      W2_tilde = 0.5 * W2
      b2_tilde = b2 + 0.5 * W2 @ 1_M
    """
    W1_tilde = 0.5 * W1
    b1_tilde = 0.5 * b1
    W2_tilde = 0.5 * W2
    M = W2.shape[1]
    b2_tilde = b2 + 0.5 * (W2 @ np.ones(M))
    return W1_tilde, b1_tilde, W2_tilde, b2_tilde


def exercise_6_4_eval_networks(
    X: np.ndarray,
    W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray,
    W1_tilde: np.ndarray, b1_tilde: np.ndarray, W2_tilde: np.ndarray, b2_tilde: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Exercise 6.4: Evaluate both sigmoid and tanh networks and compute maximum absolute difference."""
    # Sigmoid network
    a_sig = X @ W1.T + b1
    h_sig = 1.0 / (1.0 + np.exp(-a_sig))
    y_sig = h_sig @ W2.T + b2

    # Tanh network
    a_tanh = X @ W1_tilde.T + b1_tilde
    h_tanh = np.tanh(a_tanh)
    y_tanh = h_tanh @ W2_tilde.T + b2_tilde

    max_diff = float(np.max(np.abs(y_sig - y_tanh)))
    return y_sig, y_tanh, max_diff


# ---------------------------------------------------------------------------
# Exercise 6.5: Swish Activation Function and ReLU Limit
# ---------------------------------------------------------------------------

def exercise_6_5_swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Exercise 6.5: Swish activation function h(x) = x * sigmoid(beta * x) (Eq 6.61)."""
    x = np.asarray(x, dtype=np.float64)
    # Numerically stable sigmoid:
    sig = special.expit(beta * x)
    return x * sig


def exercise_6_5_swish_deriv(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Exercise 6.5: First derivative of Swish activation function.
    
    h'(x) = sigmoid(beta * x) + beta * x * sigmoid(beta * x) * (1 - sigmoid(beta * x))
          = beta * h(x) + sigmoid(beta * x) * (1 - beta * h(x))
    """
    x = np.asarray(x, dtype=np.float64)
    sig = special.expit(beta * x)
    return sig + beta * x * sig * (1.0 - sig)


def exercise_6_5_relu(x: np.ndarray) -> np.ndarray:
    """Exercise 6.5: Standard ReLU function max(0, x)."""
    return np.maximum(0.0, np.asarray(x, dtype=np.float64))


# ---------------------------------------------------------------------------
# Exercise 6.6: Derivative of Tanh Activation Function
# ---------------------------------------------------------------------------

def exercise_6_6_tanh_deriv(a: np.ndarray) -> np.ndarray:
    """Exercise 6.6: Derivative of tanh(a) expressed in terms of the function value itself:
    
    d tanh(a) / da = 1 - tanh^2(a)
    """
    y = np.tanh(a)
    return 1.0 - y ** 2


# ---------------------------------------------------------------------------
# Exercise 6.7: Softplus Activation Function Properties
# ---------------------------------------------------------------------------

def exercise_6_7_softplus(a: np.ndarray) -> np.ndarray:
    """Exercise 6.7: Softplus activation function zeta(a) = ln(1 + exp(a)) (Eq 6.16)."""
    a = np.asarray(a, dtype=np.float64)
    # Using np.log1p(np.exp(a)) with clipping for numerical stability
    return np.log1p(np.exp(-np.abs(a))) + np.maximum(a, 0.0)


def exercise_6_7_softplus_inv(y: np.ndarray) -> np.ndarray:
    """Exercise 6.7: Inverse softplus function zeta^(-1)(y) = ln(exp(y) - 1) (Eq 6.65)."""
    y = np.asarray(y, dtype=np.float64)
    return np.log(np.expm1(y))


def exercise_6_7_verify_properties(a: np.ndarray) -> Dict[str, bool]:
    """Exercise 6.7: Verify the four softplus properties (Eq 6.62 - 6.65)."""
    a = np.asarray(a, dtype=np.float64)
    zeta_a = exercise_6_7_softplus(a)
    zeta_minus_a = exercise_6_7_softplus(-a)
    sig_a = special.expit(a)

    # 1. zeta(a) - zeta(-a) = a
    prop1 = bool(np.allclose(zeta_a - zeta_minus_a, a, atol=1e-10))

    # 2. ln sigma(a) = -zeta(-a)
    prop2 = bool(np.allclose(np.log(sig_a), -zeta_minus_a, atol=1e-10))

    # 3. d zeta(a) / da = sigma(a) (verified via finite differences)
    eps = 1e-6
    deriv_num = (exercise_6_7_softplus(a + eps) - exercise_6_7_softplus(a - eps)) / (2.0 * eps)
    prop3 = bool(np.allclose(deriv_num, sig_a, atol=1e-5))

    # 4. zeta^(-1)(zeta(a)) = a (for positive zeta(a))
    inv_zeta = exercise_6_7_softplus_inv(zeta_a)
    prop4 = bool(np.allclose(inv_zeta, a, atol=1e-7))

    return {
        "zeta(a) - zeta(-a) == a": prop1,
        "ln sigma(a) == -zeta(-a)": prop2,
        "d zeta(a) / da == sigma(a)": prop3,
        "zeta^(-1)(zeta(a)) == a": prop4,
    }


# ---------------------------------------------------------------------------
# Exercise 6.8: Variance Estimation in Single-Output Regression
# ---------------------------------------------------------------------------

def exercise_6_8_mle_variance(residuals: np.ndarray) -> float:
    """Exercise 6.8: MLE noise variance for single-output regression (Eq 6.27).
    
    sigma^2 = (1/N) * sum_n (y_n - t_n)^2
    """
    residuals = np.asarray(residuals, dtype=np.float64).ravel()
    N = len(residuals)
    return float(np.sum(residuals ** 2) / N)


# ---------------------------------------------------------------------------
# Exercise 6.9: Multi-output Regression with Isotropic Gaussian Noise
# ---------------------------------------------------------------------------

def exercise_6_9_mle_variance_multioutput(residuals: np.ndarray) -> float:
    """Exercise 6.9: MLE noise variance for multi-output regression (Eq 6.30).
    
    sigma^2 = (1 / (N * K)) * sum_n ||y_n - t_n||^2
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    N, K = residuals.shape
    return float(np.sum(residuals ** 2) / (N * K))


# ---------------------------------------------------------------------------
# Exercise 6.10: Regression with Arbitrary Full Covariance Matrix Sigma
# ---------------------------------------------------------------------------

def exercise_6_10_mle_covariance(residuals: np.ndarray) -> np.ndarray:
    """Exercise 6.10: MLE full covariance matrix Sigma for multi-output Gaussian noise.
    
    Sigma = (1/N) * sum_n (y_n - t_n)(y_n - t_n)^T
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    N = residuals.shape[0]
    return (residuals.T @ residuals) / N


def exercise_6_10_mahalanobis_loss(residuals: np.ndarray, Sigma: np.ndarray) -> float:
    """Exercise 6.10: Error function for multi-output regression with covariance Sigma."""
    residuals = np.asarray(residuals, dtype=np.float64)
    Sigma_inv = np.linalg.inv(Sigma)
    # sum_n (y_n - t_n)^T Sigma^(-1) (y_n - t_n)
    quad_form = np.sum((residuals @ Sigma_inv) * residuals)
    return float(0.5 * quad_form)


# ---------------------------------------------------------------------------
# Exercise 6.11: Binary Classification with Label Noise epsilon
# ---------------------------------------------------------------------------

def exercise_6_11_robust_loss(y: np.ndarray, t: np.ndarray, eps: float) -> float:
    """Exercise 6.11: Negative log likelihood error function with label flip noise eps.
    
    q_n = (1 - eps) * y_n + eps * (1 - y_n) = eps + (1 - 2*eps) * y_n
    E = - sum_n { t_n * ln(q_n) + (1 - t_n) * ln(1 - q_n) }
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    q = eps + (1.0 - 2.0 * eps) * y
    q = np.clip(q, 1e-15, 1.0 - 1e-15)
    loss = - np.sum(t * np.log(q) + (1.0 - t) * np.log(1.0 - q))
    return float(loss)


def exercise_6_11_robust_gradient_preactivation(y: np.ndarray, t: np.ndarray, eps: float) -> np.ndarray:
    """Exercise 6.11: Derivative of robust error function w.r.t pre-activation a (where y = sigmoid(a))."""
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    q = eps + (1.0 - 2.0 * eps) * y
    dq_dy = 1.0 - 2.0 * eps
    dy_da = y * (1.0 - y)
    dE_dq = - (t / q - (1.0 - t) / (1.0 - q))
    return dE_dq * dq_dy * dy_da


# ---------------------------------------------------------------------------
# Exercise 6.12: Binary Classification with Targets in {-1, +1} and Tanh Output
# ---------------------------------------------------------------------------

def exercise_6_12_tanh_loss(y: np.ndarray, t: np.ndarray) -> float:
    """Exercise 6.12: Error function for t in {-1, +1} and y in [-1, +1].
    
    E = - sum_n ln((1 + t_n * y_n) / 2)
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    prob = (1.0 + t * y) / 2.0
    prob = np.clip(prob, 1e-15, 1.0)
    return float(- np.sum(np.log(prob)))


def exercise_6_12_tanh_loss_from_a(a: np.ndarray, t: np.ndarray) -> float:
    """Exercise 6.12: Error function written directly in terms of pre-activations a.
    
    Using (1 + t * tanh(a))/2 = sigmoid(2 * t * a) = 1 / (1 + exp(-2 * t * a)):
    E = sum_n ln(1 + exp(-2 * t_n * a_n))
    """
    a = np.asarray(a, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    # Numerically stable softplus(-2 * t * a)
    z = - 2.0 * t * a
    return float(np.sum(np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)))


def exercise_6_12_tanh_gradient_a(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Exercise 6.12: Derivative dE_n / da_n = y_n - t_n for tanh activation."""
    return np.asarray(y, dtype=np.float64) - np.asarray(t, dtype=np.float64)


# ---------------------------------------------------------------------------
# Exercise 6.13: Multi-Class Likelihood Maximization gives Cross-Entropy
# ---------------------------------------------------------------------------

def exercise_6_13_multiclass_cross_entropy(y: np.ndarray, t: np.ndarray) -> float:
    """Exercise 6.13: Multi-class cross-entropy error function (Eq 6.36).
    
    E = - sum_n sum_k t_nk * ln(y_nk)
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    y_clipped = np.clip(y, 1e-15, 1.0)
    return float(- np.sum(t * np.log(y_clipped)))


# ---------------------------------------------------------------------------
# Exercise 6.14: Pre-activation Gradient for Logistic Sigmoid Output
# ---------------------------------------------------------------------------

def exercise_6_14_verify_sigmoid_cross_entropy_grad(a: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exercise 6.14: Verify dE_n / da_k = y_k - t_k (Eq 6.31) for independent sigmoid units."""
    a = np.asarray(a, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    y = special.expit(a)
    grad_analytic = y - t

    eps = 1e-6
    grad_num = np.zeros_like(a)
    for k in range(len(a)):
        a_plus = a.copy()
        a_minus = a.copy()
        a_plus[k] += eps
        a_minus[k] -= eps
        y_p = special.expit(a_plus)
        y_m = special.expit(a_minus)
        loss_p = - np.sum(t * np.log(np.clip(y_p, 1e-15, 1.0)) + (1.0 - t) * np.log(np.clip(1.0 - y_p, 1e-15, 1.0)))
        loss_m = - np.sum(t * np.log(np.clip(y_m, 1e-15, 1.0)) + (1.0 - t) * np.log(np.clip(1.0 - y_m, 1e-15, 1.0)))
        grad_num[k] = (loss_p - loss_m) / (2.0 * eps)

    return grad_analytic, grad_num


# ---------------------------------------------------------------------------
# Exercise 6.15: Pre-activation Gradient for Softmax Multi-Class Output
# ---------------------------------------------------------------------------

def exercise_6_15_verify_softmax_cross_entropy_grad(a: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exercise 6.15: Verify dE_n / da_k = y_k - t_k (Eq 6.31) for softmax outputs."""
    a = np.asarray(a, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    
    # Stable softmax
    shifted = a - np.max(a)
    exp_a = np.exp(shifted)
    y = exp_a / np.sum(exp_a)
    grad_analytic = y - t

    def loss_fn(act):
        s = act - np.max(act)
        ea = np.exp(s)
        probs = ea / np.sum(ea)
        return - np.sum(t * np.log(np.clip(probs, 1e-15, 1.0)))

    eps = 1e-6
    grad_num = np.zeros_like(a)
    for k in range(len(a)):
        a_p = a.copy()
        a_m = a.copy()
        a_p[k] += eps
        a_m[k] -= eps
        grad_num[k] = (loss_fn(a_p) - loss_fn(a_m)) / (2.0 * eps)

    return grad_analytic, grad_num


# ---------------------------------------------------------------------------
# Exercise 6.16: Planar Two-Link Manipulator Forward Kinematics
# ---------------------------------------------------------------------------

def exercise_6_16_forward_kinematics(
    theta1: Union[float, np.ndarray],
    theta2: Union[float, np.ndarray],
    L1: float = 0.8,
    L2: float = 0.2
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Exercise 6.16: Forward kinematics of two-link planar robot arm.
    
    x1 = L1 * cos(theta1) + L2 * cos(theta1 + theta2)
    x2 = L1 * sin(theta1) + L2 * sin(theta1 + theta2)
    """
    t1 = np.asarray(theta1, dtype=np.float64)
    t2 = np.asarray(theta2, dtype=np.float64)
    x1 = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    x2 = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    if np.ndim(theta1) == 0 and np.ndim(theta2) == 0:
        return float(x1), float(x2)
    return x1, x2


# ---------------------------------------------------------------------------
# Exercise 6.17: Mixture Density Network Responsibilities gamma_nk
# ---------------------------------------------------------------------------

def exercise_6_17_compute_responsibilities(
    pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Exercise 6.17: Compute posterior responsibilities gamma_nk = p(k | t_n) (Eq 6.44).
    
    pi: (K,)
    mu: (K, L)
    sigma: (K,)
    t: (L,)
    """
    pi = np.asarray(pi, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    K = len(pi)
    L = len(t) if t.ndim > 0 else 1
    t_vec = np.atleast_1d(t)

    # Log components
    log_probs = np.zeros(K)
    for k in range(K):
        diff = t_vec - mu[k]
        sq_dist = np.sum(diff ** 2)
        log_comp = - 0.5 * L * np.log(2.0 * np.pi) - L * np.log(sigma[k]) - 0.5 * sq_dist / (sigma[k] ** 2)
        log_probs[k] = np.log(np.maximum(pi[k], 1e-15)) + log_comp

    max_log = np.max(log_probs)
    probs = np.exp(log_probs - max_log)
    return probs / np.sum(probs)


# ---------------------------------------------------------------------------
# Exercise 6.18: MDN Derivative w.r.t Mixing Coefficient Pre-activations
# ---------------------------------------------------------------------------

def exercise_6_18_verify_grad_pi(
    a_pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray, t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Exercise 6.18: Verify dE_n / da_k^pi = pi_k - gamma_nk (Eq 6.45)."""
    a_pi = np.asarray(a_pi, dtype=np.float64)
    s = a_pi - np.max(a_pi)
    exp_s = np.exp(s)
    pi = exp_s / np.sum(exp_s)
    gamma = exercise_6_17_compute_responsibilities(pi, mu, sigma, t)
    grad_analytic = pi - gamma

    def nll(act_pi):
        shift = act_pi - np.max(act_pi)
        e = np.exp(shift)
        p = e / np.sum(e)
        t_vec = np.atleast_1d(t)
        L = len(t_vec)
        comp = np.array([
            (2.0 * np.pi * (sigma[k] ** 2)) ** (-0.5 * L) * np.exp(-0.5 * np.sum((t_vec - mu[k]) ** 2) / (sigma[k] ** 2))
            for k in range(len(p))
        ])
        return - np.log(np.sum(p * comp))

    eps = 1e-6
    grad_num = np.zeros_like(a_pi)
    for k in range(len(a_pi)):
        ap = a_pi.copy()
        am = a_pi.copy()
        ap[k] += eps
        am[k] -= eps
        grad_num[k] = (nll(ap) - nll(am)) / (2.0 * eps)

    return grad_analytic, grad_num


# ---------------------------------------------------------------------------
# Exercise 6.19: MDN Derivative w.r.t Component Mean Pre-activations
# ---------------------------------------------------------------------------

def exercise_6_19_verify_grad_mu(
    pi: np.ndarray, a_mu: np.ndarray, sigma: np.ndarray, t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Exercise 6.19: Verify dE_n / da_{kl}^mu = gamma_nk * (mu_{kl} - t_{nl}) / sigma_k^2 (Eq 6.46)."""
    mu = np.asarray(a_mu, dtype=np.float64)
    gamma = exercise_6_17_compute_responsibilities(pi, mu, sigma, t)
    t_vec = np.atleast_1d(t)
    K, L = mu.shape

    grad_analytic = np.zeros((K, L))
    for k in range(K):
        grad_analytic[k] = gamma[k] * (mu[k] - t_vec) / (sigma[k] ** 2)

    def nll(mu_mat):
        comp = np.array([
            (2.0 * np.pi * (sigma[k] ** 2)) ** (-0.5 * L) * np.exp(-0.5 * np.sum((t_vec - mu_mat[k]) ** 2) / (sigma[k] ** 2))
            for k in range(K)
        ])
        return - np.log(np.sum(pi * comp))

    eps = 1e-6
    grad_num = np.zeros_like(mu)
    for k in range(K):
        for l in range(L):
            mup = mu.copy()
            mum = mu.copy()
            mup[k, l] += eps
            mum[k, l] -= eps
            grad_num[k, l] = (nll(mup) - nll(mum)) / (2.0 * eps)

    return grad_analytic, grad_num


# ---------------------------------------------------------------------------
# Exercise 6.20: MDN Derivative w.r.t Component Variance Pre-activations
# ---------------------------------------------------------------------------

def exercise_6_20_verify_grad_sigma(
    pi: np.ndarray, mu: np.ndarray, a_sigma: np.ndarray, t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Exercise 6.20: Verify dE_n / da_k^sigma = gamma_nk * (L - ||t_n - mu_k||^2 / sigma_k^2) (Eq 6.47)."""
    sigma = np.exp(a_sigma)
    gamma = exercise_6_17_compute_responsibilities(pi, mu, sigma, t)
    t_vec = np.atleast_1d(t)
    K = len(pi)
    L = len(t_vec)

    grad_analytic = np.zeros(K)
    for k in range(K):
        sq_dist = np.sum((t_vec - mu[k]) ** 2)
        grad_analytic[k] = gamma[k] * (L - sq_dist / (sigma[k] ** 2))

    def nll(act_sig):
        sig = np.exp(act_sig)
        comp = np.array([
            (2.0 * np.pi * (sig[k] ** 2)) ** (-0.5 * L) * np.exp(-0.5 * np.sum((t_vec - mu[k]) ** 2) / (sig[k] ** 2))
            for k in range(K)
        ])
        return - np.log(np.sum(pi * comp))

    eps = 1e-6
    grad_num = np.zeros_like(a_sigma)
    for k in range(K):
        asp = a_sigma.copy()
        asm = a_sigma.copy()
        asp[k] += eps
        asm[k] -= eps
        grad_num[k] = (nll(asp) - nll(asm)) / (2.0 * eps)

    return grad_analytic, grad_num


# ---------------------------------------------------------------------------
# Exercise 6.21: MDN Conditional Mean and Variance
# ---------------------------------------------------------------------------

def exercise_6_21_conditional_mean_variance(
    pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Exercise 6.21: Compute exact MDN conditional mean (Eq 6.48) and variance (Eq 6.50).
    
    mean = sum_k pi_k * mu_k
    s^2 = sum_k pi_k * { sigma_k^2 + ||mu_k - mean||^2 }
    """
    pi = np.asarray(pi, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    K = len(pi)

    # Conditional mean
    cond_mean = np.sum(pi[:, None] * mu, axis=0)

    # Conditional variance
    L = mu.shape[1] if mu.ndim > 1 else 1
    s2 = 0.0
    for k in range(K):
        sq_dist = np.sum((mu[k] - cond_mean) ** 2)
        s2 += pi[k] * (L * (sigma[k] ** 2) + sq_dist)

    return cond_mean, float(s2)


def exercise_6_21_monte_carlo_mean_variance(
    pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray, n_samples: int = 100000, seed: int = 42
) -> Tuple[np.ndarray, float]:
    """Exercise 6.21: Monte Carlo verification of MDN conditional mean and variance."""
    rng = np.random.default_rng(seed)
    K = len(pi)
    L = mu.shape[1]

    # Sample component indices according to mixing coefficients pi
    comp_indices = rng.choice(K, size=n_samples, p=pi)

    # Sample targets
    samples = np.zeros((n_samples, L))
    for k in range(K):
        idx = (comp_indices == k)
        n_k = np.sum(idx)
        if n_k > 0:
            samples[idx] = rng.normal(loc=mu[k], scale=sigma[k], size=(n_k, L))

    mc_mean = np.mean(samples, axis=0)
    # Variance about conditional mean: E[||t - E[t]||^2]
    diff = samples - mc_mean
    mc_var = float(np.mean(np.sum(diff ** 2, axis=1)))
    return mc_mean, mc_var
