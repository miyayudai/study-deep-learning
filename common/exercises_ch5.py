"""Chapter 5: Single-layer Networks: Classification
Exercises 5.1 to 5.24 helper functions.

Bishop & Bishop (2024), Deep Learning: Foundations and Concepts, pp. 166-169.
"""

from typing import Callable, List, Optional, Tuple, Union
import numpy as np
from scipy import integrate, special
from scipy.optimize import linprog
from scipy.stats import norm

from common.discriminative_classifiers import (
    GaussianBasisFunctions,
    LogisticRegression,
    ProbitRegression,
    SoftmaxRegression,
    erf_func,
    logit,
    probit,
    probit_deriv,
    sigmoid,
    sigmoid_deriv,
    softmax,
    softmax_jacobian,
)


def exercise_5_1_conditional_expectation(p_Ck: np.ndarray) -> np.ndarray:
    """Exercise 5.1: Compute conditional expectation E[t | x] = sum_k e_k * p(C_k | x)."""
    p_Ck = np.asarray(p_Ck, dtype=np.float64)
    K = len(p_Ck)
    E_t = np.zeros(K)
    for k in range(K):
        e_k = np.zeros(K)
        e_k[k] = 1.0
        E_t += e_k * p_Ck[k]
    return E_t


def exercise_5_2_check_separability(X: np.ndarray, Y: np.ndarray) -> bool:
    """Exercise 5.2: Check if two point sets X and Y are linearly separable via linear programming."""
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    N_x, D = X.shape
    N_y, _ = Y.shape

    c = np.zeros(D + 1)
    A = np.vstack([
        np.hstack([-X, -np.ones((N_x, 1))]),
        np.hstack([Y, np.ones((N_y, 1))])
    ])
    b = -np.ones(N_x + N_y)
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None), method="highs")
    return bool(res.success)


def exercise_5_3_verify_linear_constraint(X: np.ndarray, T: np.ndarray, a: np.ndarray, b: float,
                                         X_test: np.ndarray) -> Tuple[float, float]:
    """Exercise 5.3: Verify that least-squares predictions preserve linear target constraint a^T t + b = 0."""
    N = X.shape[0]
    Phi = np.hstack([np.ones((N, 1)), X])
    W = np.linalg.pinv(Phi) @ T

    N_test = X_test.shape[0]
    Phi_test = np.hstack([np.ones((N_test, 1)), X_test])
    Y_pred = Phi_test @ W

    train_violation = np.max(np.abs(T @ a + b))
    test_violation = np.max(np.abs(Y_pred @ a + b))
    return float(train_violation), float(test_violation)


def exercise_5_4_verify_multiple_constraints(X: np.ndarray, T: np.ndarray, A: np.ndarray,
                                            b_vec: np.ndarray, X_test: np.ndarray) -> Tuple[float, float]:
    """Exercise 5.4: Verify preservation of multiple simultaneous linear constraints A^T t + b = 0."""
    N = X.shape[0]
    Phi = np.hstack([np.ones((N, 1)), X])
    W = np.linalg.pinv(Phi) @ T

    N_test = X_test.shape[0]
    Phi_test = np.hstack([np.ones((N_test, 1)), X_test])
    Y_pred = Phi_test @ W

    train_violation = np.max(np.abs(T @ A + b_vec))
    test_violation = np.max(np.abs(Y_pred @ A + b_vec))
    return float(train_violation), float(test_violation)


def exercise_5_5_compute_f_score(TP: int, FP: int, FN: int, beta: float) -> Tuple[float, float]:
    """Exercise 5.5: Compute F_beta score using P, R definition and formula (5.39)."""
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    f_from_pr = (1.0 + beta ** 2) * P * R / (beta ** 2 * P + R)
    f_formula = (1.0 + beta ** 2) * TP / ((1.0 + beta ** 2) * TP + beta ** 2 * FN + FP)
    return float(f_from_pr), float(f_formula)


def exercise_5_6_compute_bhattacharyya_bound(p_joint1: Callable[[float], float],
                                             p_joint2: Callable[[float], float],
                                             x_min: float = -10.0,
                                             x_max: float = 10.0) -> Tuple[float, float]:
    """Exercise 5.6: Compute true misclassification probability and Bhattacharyya upper bound."""
    def integrand_true(x):
        return min(p_joint1(x), p_joint2(x))

    def integrand_bound(x):
        return np.sqrt(p_joint1(x) * p_joint2(x))

    err_true, _ = integrate.quad(integrand_true, x_min, x_max)
    err_bound, _ = integrate.quad(integrand_bound, x_min, x_max)
    return float(err_true), float(err_bound)


def exercise_5_7_zero_one_loss_decision(p_post: np.ndarray) -> Tuple[int, np.ndarray]:
    """Exercise 5.7: Expected loss for 0-1 loss matrix L_kj = 1 - I_kj."""
    p_post = np.asarray(p_post, dtype=np.float64)
    expected_losses = 1.0 - p_post
    chosen_class = int(np.argmin(expected_losses))
    return chosen_class, expected_losses


def exercise_5_8_expected_loss_decision(L: np.ndarray, p_post: np.ndarray) -> Tuple[int, np.ndarray]:
    """Exercise 5.8: Minimize expected loss for general loss matrix L."""
    L = np.asarray(L, dtype=np.float64)
    p_post = np.asarray(p_post, dtype=np.float64)
    # Expected loss for choosing action j: sum_k L_kj * p(C_k | x)
    expected_losses = L.T @ p_post
    chosen_class = int(np.argmin(expected_losses))
    return chosen_class, expected_losses


def exercise_5_9_average_posterior(p_post_samples: np.ndarray) -> float:
    """Exercise 5.9: Average of posterior probabilities over samples."""
    return float(np.mean(p_post_samples))


def exercise_5_10_reject_threshold(lam: float) -> float:
    """Exercise 5.10: Rejection threshold theta = 1 - lambda for 0-1 loss."""
    return float(1.0 - lam)


def exercise_5_11_verify_sigmoid_properties(a: np.ndarray) -> Tuple[bool, bool]:
    """Exercise 5.11: Verify symmetry sigma(-a) = 1 - sigma(a) and logit inverse."""
    a = np.asarray(a, dtype=np.float64)
    sym = np.allclose(sigmoid(-a), 1.0 - sigmoid(a))
    inv = np.allclose(logit(sigmoid(a)), a)
    return bool(sym), bool(inv)


def exercise_5_12_compute_gaussian_gda_params(mu1: np.ndarray, mu2: np.ndarray,
                                              Sigma: np.ndarray, pi1: float,
                                              pi2: float) -> Tuple[np.ndarray, float]:
    """Exercise 5.12: Compute analytical weight vector w and bias w0 for Gaussian GDA."""
    Sigma_inv = np.linalg.inv(Sigma)
    w = Sigma_inv @ (mu1 - mu2)
    w0 = -0.5 * mu1.T @ Sigma_inv @ mu1 + 0.5 * mu2.T @ Sigma_inv @ mu2 + np.log(pi1 / pi2)
    return w, float(w0)


def exercise_5_13_mle_priors(targets: np.ndarray, K: int) -> np.ndarray:
    """Exercise 5.13: MLE for class priors pi_k = N_k / N."""
    targets = np.asarray(targets, dtype=int).ravel()
    N = len(targets)
    return np.array([np.sum(targets == k) / N for k in range(K)])


def exercise_5_14_mle_gaussian_shared_cov(Phi: np.ndarray, targets: np.ndarray,
                                         K: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """Exercise 5.14: MLE for class means mu_k and shared covariance Sigma."""
    targets = np.asarray(targets, dtype=int).ravel()
    N = len(targets)
    means = [np.mean(Phi[targets == k], axis=0) for k in range(K)]
    Sigma = np.zeros((Phi.shape[1], Phi.shape[1]))
    for k in range(K):
        diff = Phi[targets == k] - means[k]
        Sigma += diff.T @ diff / N
    return means, Sigma


def exercise_5_15_mle_binary_naive_bayes(X: np.ndarray, targets: np.ndarray,
                                        K: int = 2) -> List[np.ndarray]:
    """Exercise 5.15: MLE for Bernoulli Naive Bayes feature probabilities mu_ki."""
    targets = np.asarray(targets, dtype=int).ravel()
    return [np.mean(X[targets == k], axis=0) for k in range(K)]


def exercise_5_16_multistate_naive_bayes_ak(phi_1ofL: np.ndarray, mu_kml: np.ndarray,
                                            pi_k: float) -> float:
    """Exercise 5.16: Compute pre-activation ak for multi-state Naive Bayes."""
    # phi_1ofL shape (M, L), mu_kml shape (M, L)
    return float(np.sum(phi_1ofL * np.log(mu_kml)) + np.log(pi_k))


def exercise_5_17_mle_multistate_naive_bayes(features_1ofL: np.ndarray) -> np.ndarray:
    """Exercise 5.17: MLE for multi-state Naive Bayes probabilities mu_kml = N_kml / N_k."""
    # features_1ofL shape (N_k, M, L)
    return np.mean(features_1ofL, axis=0)


def exercise_5_18_verify_sigmoid_derivative(a: np.ndarray) -> bool:
    """Exercise 5.18: Verify d sigma / da = sigma * (1 - sigma)."""
    return bool(np.allclose(sigmoid_deriv(a), sigmoid(a) * (1.0 - sigmoid(a))))


def exercise_5_19_verify_logistic_gradient(Phi: np.ndarray, t: np.ndarray,
                                           w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exercise 5.19: Compare analytical gradient Phi^T (y - t) with numerical gradient."""
    clf = LogisticRegression(fit_intercept=False)
    grad_analytic = Phi.T @ (sigmoid(Phi @ w) - t)

    eps = 1e-6
    grad_num = np.zeros_like(w)
    for i in range(len(w)):
        e_i = np.zeros_like(w)
        e_i[i] = eps
        loss_plus = clf.compute_loss(Phi, t, w + e_i)
        loss_minus = clf.compute_loss(Phi, t, w - e_i)
        grad_num[i] = (loss_plus - loss_minus) / (2 * eps)

    return grad_analytic, grad_num


def exercise_5_20_separable_growth_check(X: np.ndarray, t: np.ndarray,
                                         iterations: List[int]) -> List[float]:
    """Exercise 5.20: Verify that weight norm grows monotonically on separable data."""
    clf = LogisticRegression(reg=0.0, fit_intercept=False)
    norms = []
    for it in iterations:
        clf.fit(X, t, method="gd", lr=0.5, max_iter=it)
        norms.append(float(np.linalg.norm(clf.w)))
    return norms


def exercise_5_21_verify_softmax_jacobian(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exercise 5.21: Compare analytical Jacobian y_k (I_kj - y_j) with numerical Jacobian."""
    y = softmax(a)
    J_analytic = softmax_jacobian(y)

    eps = 1e-6
    K = len(a)
    J_num = np.zeros((K, K))
    for j in range(K):
        e_j = np.zeros(K)
        e_j[j] = eps
        J_num[:, j] = (softmax(a + e_j) - softmax(a - e_j)) / (2 * eps)

    return J_analytic, J_num


def exercise_5_22_verify_softmax_gradient(Phi: np.ndarray, T: np.ndarray,
                                          W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exercise 5.22: Compare analytical gradient Phi^T (Y - T) with numerical gradient."""
    clf = SoftmaxRegression(fit_intercept=False)
    Y = softmax(Phi @ W, axis=-1)
    grad_analytic = Phi.T @ (Y - T)

    eps = 1e-6
    grad_num = np.zeros_like(W)
    D, K = W.shape
    for d in range(D):
        for k in range(K):
            e_mat = np.zeros_like(W)
            e_mat[d, k] = eps
            lp = clf.compute_loss(Phi, T, W + e_mat)
            lm = clf.compute_loss(Phi, T, W - e_mat)
            grad_num[d, k] = (lp - lm) / (2 * eps)

    return grad_analytic, grad_num


def exercise_5_23_verify_probit_erf(a: np.ndarray) -> bool:
    """Exercise 5.23: Verify Phi(a) = 0.5 * (1 + erf(a / sqrt(2)))."""
    return bool(np.allclose(probit(a), 0.5 * (1.0 + special.erf(a / np.sqrt(2.0)))))


def exercise_5_24_probit_sigmoid_matching_scale() -> Tuple[float, float, float]:
    """Exercise 5.24: Derivative matching of sigma(a) and Phi(lambda * a) at a=0 yields lambda^2 = pi/8."""
    deriv_sig_0 = float(sigmoid_deriv(0.0))  # 1/4 = 0.25
    lambda_val = float(np.sqrt(np.pi / 8.0))
    deriv_probit_0 = float(lambda_val * norm.pdf(0.0))
    return deriv_sig_0, deriv_probit_0, float(lambda_val ** 2)
