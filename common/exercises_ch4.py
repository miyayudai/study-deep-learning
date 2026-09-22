"""
Solutions and numerical verifications for Chapter 4 Exercises 4.1 - 4.12.
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
import scipy.linalg as la
import scipy.special as sp
import scipy.integrate as integrate
import scipy.optimize as opt


# =====================================================================
# Exercise 4.1: Polynomial Normal Equations
# =====================================================================
def solve_polynomial_normal_equations(x: np.ndarray, t: np.ndarray, M: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve A w = T where A_ij = sum_n x_n^(i+j), T_i = sum_n x_n^i t_n (Eq 4.53 - 4.54).
    """
    N = len(x)
    A = np.zeros((M + 1, M + 1), dtype=np.float64)
    T = np.zeros(M + 1, dtype=np.float64)
    for i in range(M + 1):
        T[i] = np.sum((x ** i) * t)
        for j in range(M + 1):
            A[i, j] = np.sum(x ** (i + j))
    w = la.solve(A, T, assume_a='pos')
    return w, A, T


# =====================================================================
# Exercise 4.2: Regularized Polynomial Normal Equations
# =====================================================================
def solve_regularized_polynomial_equations(x: np.ndarray, t: np.ndarray, M: int, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve (A + lambda * I) w = T (Eq 1.4, Exercise 4.2).
    """
    w_unreg, A, T = solve_polynomial_normal_equations(x, t, M)
    A_reg = A + lam * np.eye(M + 1)
    w_reg = la.solve(A_reg, T, assume_a='pos')
    return w_reg, A_reg


# =====================================================================
# Exercise 4.3: Sigmoid to Tanh Parameter Conversion
# =====================================================================
def sigmoid_to_tanh_params(w0: float, w: np.ndarray, mu: np.ndarray, s: float) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    Convert linear combination of sigmoids to tanh functions:
    y(x, w) = w0 + sum_j w_j sigma((x - mu_j)/s)
            = u0 + sum_j u_j tanh((x - mu_j)/(2s))
    where u_j = 0.5 * w_j and u0 = w0 + 0.5 * sum_j w_j.
    """
    w_arr = np.asarray(w, dtype=np.float64)
    u_j = 0.5 * w_arr
    u0 = w0 + 0.5 * np.sum(w_arr)
    s_tanh = 2.0 * s
    return float(u0), u_j, np.asarray(mu, dtype=np.float64), s_tanh


def eval_sigmoid_network(x: np.ndarray, w0: float, w: np.ndarray, mu: np.ndarray, s: float) -> np.ndarray:
    diff = (x[:, None] - mu[None, :]) / s
    sig = 1.0 / (1.0 + np.exp(-diff))
    return w0 + sig @ w


def eval_tanh_network(x: np.ndarray, u0: float, u: np.ndarray, mu: np.ndarray, s_tanh: float) -> np.ndarray:
    diff = (x[:, None] - mu[None, :]) / s_tanh
    tanh_val = np.tanh(diff)
    return u0 + tanh_val @ u


# =====================================================================
# Exercise 4.4: Orthogonal Projection Matrix
# =====================================================================
def compute_projection_matrix(Phi: np.ndarray) -> np.ndarray:
    """Compute orthogonal projection matrix P = Phi (Phi^T Phi)^(-1) Phi^T (Eq 4.59)."""
    return Phi @ la.inv(Phi.T @ Phi) @ Phi.T


# =====================================================================
# Exercise 4.5: Weighted Least Squares (WLS)
# =====================================================================
def solve_weighted_least_squares(Phi: np.ndarray, t: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Solve weighted least squares: min 0.5 * sum_n r_n (t_n - w^T phi_n)^2
    Solution: w* = (Phi^T R Phi)^(-1) Phi^T R t (Eq 4.60).
    """
    R = np.diag(weights)
    A = Phi.T @ R @ Phi
    b = Phi.T @ R @ t
    return la.solve(A, b, assume_a='pos')


# =====================================================================
# Exercise 4.6: Regularized Least Squares
# =====================================================================
def solve_ridge_regression(Phi: np.ndarray, t: np.ndarray, lam: float) -> np.ndarray:
    """
    Solve regularized least squares: w* = (lambda * I + Phi^T Phi)^(-1) Phi^T t (Eq 4.27).
    """
    M = Phi.shape[1]
    A = lam * np.eye(M) + Phi.T @ Phi
    b = Phi.T @ t
    return la.solve(A, b, assume_a='pos')


def regularized_least_squares_loss_and_grad(w: np.ndarray, Phi: np.ndarray, t: np.ndarray, lam: float) -> Tuple[float, np.ndarray]:
    """
    Compute regularized loss E(w) = 0.5 * ||t - Phi w||^2 + 0.5 * lam * ||w||^2 and its gradient.
    """
    res = t - Phi @ w
    loss = 0.5 * np.sum(res ** 2) + 0.5 * lam * np.sum(w ** 2)
    grad = - Phi.T @ res + lam * w
    return float(loss), grad


# =====================================================================
# Exercise 4.7: Multivariate Target with Arbitrary Covariance Sigma
# =====================================================================
def estimate_multivariate_regression_mle(Phi: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate W_ML = (Phi^T Phi)^(-1) Phi^T T and Sigma_ML = (1/N) (T - Phi W)^T (T - Phi W) (Eq 4.61 - 4.63).
    """
    N, K = T.shape
    W_ML = la.inv(Phi.T @ Phi) @ Phi.T @ T
    Residuals = T - Phi @ W_ML
    Sigma_ML = (Residuals.T @ Residuals) / N
    return W_ML, Sigma_ML


# =====================================================================
# Exercise 4.8 & 4.9 & 4.10: Vector Squared Loss Decomposition
# =====================================================================
def decompose_vector_squared_loss(f_vals: np.ndarray, t_samples: np.ndarray) -> Dict[str, float]:
    """
    Given predictions f(x) and conditional samples t ~ p(t|x) for a fixed x:
    E[||f(x) - t||^2] = ||f(x) - E[t|x]||^2 + Tr(Cov[t|x]).
    """
    t_mean = np.mean(t_samples, axis=0)
    expected_loss = float(np.mean(np.sum((f_vals - t_samples) ** 2, axis=-1)))
    fit_loss = float(np.sum((f_vals - t_mean) ** 2))
    # Empirical trace covariance
    t_centered = t_samples - t_mean
    trace_cov = float(np.mean(np.sum(t_centered ** 2, axis=-1)))
    return {
        "expected_loss": expected_loss,
        "fit_loss": fit_loss,
        "trace_cov": trace_cov,
        "sum_components": fit_loss + trace_cov,
    }


# =====================================================================
# Exercise 4.11: Generalized Gaussian Distribution (Lq Noise)
# =====================================================================
def generalized_gaussian_pdf(x: np.ndarray, sigma2: float, q: float) -> np.ndarray:
    """
    p(x | sigma^2, q) = [q / (2 * (2 * sigma^2)^(1/q) * Gamma(1/q))] * exp(- |x|^q / (2 * sigma^2)) (Eq 4.66).
    """
    c = q / (2.0 * ((2.0 * sigma2) ** (1.0 / q)) * sp.gamma(1.0 / q))
    return c * np.exp(- (np.abs(x) ** q) / (2.0 * sigma2))


def generalized_gaussian_log_likelihood(y_pred: np.ndarray, t: np.ndarray, sigma2: float, q: float) -> float:
    """
    ln p(t | X, w, sigma^2) = - (1 / (2 sigma^2)) sum_n |y_n - t_n|^q - (N / q) ln(2 sigma^2) + const (Eq 4.69).
    """
    N = len(t)
    residuals_q = np.sum(np.abs(y_pred - t) ** q)
    c_term = N * np.log(q / (2.0 * sp.gamma(1.0 / q)))
    ll = - (1.0 / (2.0 * sigma2)) * residuals_q - (N / q) * np.log(2.0 * sigma2) + c_term
    return float(ll)


# =====================================================================
# Exercise 4.12: Optimal Lq Predictors
# =====================================================================
def compute_optimal_lq_predictor(
    conditional_samples_or_pdf: Any,
    q: float,
    t_grid: Optional[np.ndarray] = None,
    pdf_values: Optional[np.ndarray] = None
) -> float:
    """
    Find y* that minimizes E[|y - t|^q | x].
    For q = 2: conditional mean.
    For q = 1: conditional median.
    For q -> 0: conditional mode.
    """
    if t_grid is not None and pdf_values is not None:
        def loss_func(y_val):
            integrand = np.abs(y_val - t_grid) ** q * pdf_values
            return integrate.trapezoid(integrand, t_grid)

        init_guess = float(t_grid[np.argmax(pdf_values)])
        res = opt.minimize(loss_func, x0=[init_guess], method='Nelder-Mead')
        return float(res.x[0])
    else:
        samples = np.asarray(conditional_samples_or_pdf, dtype=np.float64)
        def sample_loss(y_val):
            return np.mean(np.abs(y_val - samples) ** q)

        init_guess = float(np.median(samples))
        res = opt.minimize(sample_loss, x0=[init_guess], method='Nelder-Mead')
        return float(res.x[0])
