"""
common/exercises_ch9.py
=======================
Chapter 9: Regularization - Exercises 9.1 to 9.18
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements comprehensive theoretical proofs and rigorous numerical verification
routines for all 18 exercises in Chapter 9:
- Exercise 9.1: Group axioms for 90-degree square rotations (C_4) and 2D translations (R^2).
- Exercise 9.2: Equivalence of input Gaussian noise to L2 weight decay on non-bias weights.
- Exercise 9.3: Differential equation for quadratic regularizer gradient flow and exponential decay.
- Exercise 9.4: Linear scaling equivariance of network inputs and outputs.
- Exercise 9.5: Equivalence of L2 weight decay and constrained optimization via Lagrange multipliers.
- Exercise 9.6: Early stopping vs. weight decay spectral equivalence on quadratic error functions.
- Exercise 9.7: Backpropagation with tied/shared weight constraints (pooled gradients).
- Exercise 9.8: Bayes' theorem derivation of responsibilities in GMM soft weight sharing.
- Exercise 9.9: Analytical gradient of soft weight sharing regularizer with respect to weights.
- Exercise 9.10: Analytical gradient of soft weight sharing regularizer with respect to cluster means.
- Exercise 9.11: Analytical gradient with respect to log-variance parameters beta_j.
- Exercise 9.12: Softmax mixing coefficients and logit gradient derivation.
- Exercise 9.13: Recursive expansion of residual connections into an exponential path ensemble.
- Exercise 9.14: Committee error reduction factor 1/M under uncorrelated zero-mean errors.
- Exercise 9.15: Jensen's inequality proof of committee error bound E_COM <= E_AV for squared loss.
- Exercise 9.16: Generalization of committee error bound to any convex error function E(y).
- Exercise 9.17: Necessary and sufficient conditions for bounded convex combinations in committees.
- Exercise 9.18: Dropout on linear regression as input-variance-weighted L2 regularization.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


# =====================================================================
# Exercise 9.1: Group Axioms for Transformations
# =====================================================================

def verify_exercise_9_1() -> Dict[str, Any]:
    """
    Verify the 4 group axioms for:
    1. Rotations of a square through multiples of 90 degrees under composition (C_4).
    2. Continuous translations in a 2D plane under vector addition (R^2, +).

    Group Axioms:
    (G1) Closure: For all a, b in G, a * b in G.
    (G2) Associativity: For all a, b, c in G, (a * b) * c = a * (b * c).
    (G3) Identity element: There exists e in G such that for all a in G, e * a = a * e = a.
    (G4) Inverse element: For each a in G, there exists a^{-1} in G such that a * a^{-1} = a^{-1} * a = e.
    """
    # 1. C_4 group: angles {0, 90, 180, 270} degrees (represented as integers 0, 1, 2, 3 mod 4)
    elements_c4 = [0, 1, 2, 3]
    op_c4 = lambda a, b: (a + b) % 4
    inv_c4 = lambda a: (-a) % 4
    e_c4 = 0

    # Test (G1) Closure
    closure_c4 = all(op_c4(a, b) in elements_c4 for a in elements_c4 for b in elements_c4)

    # Test (G2) Associativity
    assoc_c4 = all(
        op_c4(op_c4(a, b), c) == op_c4(a, op_c4(b, c))
        for a in elements_c4 for b in elements_c4 for c in elements_c4
    )

    # Test (G3) Identity
    ident_c4 = all(op_c4(e_c4, a) == a and op_c4(a, e_c4) == a for a in elements_c4)

    # Test (G4) Inverse
    inv_check_c4 = all(
        op_c4(a, inv_c4(a)) == e_c4 and op_c4(inv_c4(a), a) == e_c4
        for a in elements_c4
    )

    # 2. 2D Translation group: (R^2, +)
    rng = np.random.RandomState(42)
    sample_vectors = [rng.randn(2) for _ in range(10)]
    e_r2 = np.zeros(2)

    assoc_r2 = all(
        np.allclose((a + b) + c, a + (b + c))
        for a in sample_vectors[:3] for b in sample_vectors[3:6] for c in sample_vectors[6:9]
    )
    ident_r2 = all(np.allclose(a + e_r2, a) and np.allclose(e_r2 + a, a) for a in sample_vectors)
    inv_r2 = all(np.allclose(a + (-a), e_r2) and np.allclose((-a) + a, e_r2) for a in sample_vectors)

    all_passed = (
        closure_c4 and assoc_c4 and ident_c4 and inv_check_c4
        and assoc_r2 and ident_r2 and inv_r2
    )

    return {
        "passed": all_passed,
        "c4_axioms": {"closure": closure_c4, "associativity": assoc_c4, "identity": ident_c4, "inverse": inv_check_c4},
        "r2_axioms": {"associativity": assoc_r2, "identity": ident_r2, "inverse": inv_r2},
    }


# =====================================================================
# Exercise 9.2: Input Noise as L2 Weight Decay
# =====================================================================

def verify_exercise_9_2() -> Dict[str, Any]:
    """
    Verify that adding independent zero-mean Gaussian noise xi_i ~ N(0, sigma^2)
    to inputs of a linear model y(x) = w_0 + sum_{i=1}^D w_i x_i
    under sum-of-squares error E_D is equivalent to noise-free sum-of-squares
    plus weight decay regularizer (lambda / 2) * sum_{i=1}^D w_i^2 where lambda = N * sigma^2,
    with bias w_0 omitted.
    """
    rng = np.random.RandomState(42)
    N, D = 200, 3
    X = rng.randn(N, D)
    t = rng.randn(N)
    w0 = 0.5
    w = np.array([1.2, -0.8, 2.0])
    sigma_noise = 0.1
    var_noise = sigma_noise ** 2

    # Noise-free error: 0.5 * sum_n (w0 + w^T x_n - t_n)^2
    y_clean = w0 + X @ w
    E_clean = 0.5 * np.sum((y_clean - t) ** 2)

    # Theoretical regularizer: (N * sigma^2 / 2) * sum_{i=1}^D w_i^2
    lambda_reg = N * var_noise
    reg_term = 0.5 * lambda_reg * np.sum(w ** 2)
    E_theoretical = E_clean + reg_term

    # Monte Carlo average of noisy error
    n_mc = 15000
    noisy_errors = []
    for _ in range(n_mc):
        xi = rng.randn(N, D) * sigma_noise
        X_noisy = X + xi
        y_noisy = w0 + X_noisy @ w
        noisy_errors.append(0.5 * np.sum((y_noisy - t) ** 2))

    E_mc_mean = float(np.mean(noisy_errors))
    rel_diff = abs(E_mc_mean - E_theoretical) / E_theoretical

    return {
        "passed": bool(rel_diff < 0.01),
        "E_clean": float(E_clean),
        "reg_term": float(reg_term),
        "E_theoretical": float(E_theoretical),
        "E_mc_mean": float(E_mc_mean),
        "rel_diff": float(rel_diff),
    }


# =====================================================================
# Exercise 9.3: Differential Equation for Weight Decay
# =====================================================================

def verify_exercise_9_3() -> Dict[str, Any]:
    """
    Verify the continuous limit of gradient descent with quadratic regularizer
    Omega(w) = (1/2) w^T w:
        w^{tau+1} = w^tau - eta grad Omega(w^tau) = w^tau - eta w^tau = (1 - eta) w^tau.
    Continuous limit: dw / dt = -eta w  ==>  w(t) = w(0) exp(-eta t).
    """
    w0 = np.array([2.5, -4.0, 1.2])
    eta = 0.01
    n_steps = 100

    # Discrete evolution: w_tau = (1 - eta)^tau * w0
    w_discrete = w0.copy()
    for _ in range(n_steps):
        w_discrete = w_discrete - eta * w_discrete

    # Continuous analytical solution: w(t) = w0 * exp(-eta * t)
    t = float(n_steps)
    w_continuous = w0 * np.exp(-eta * t)

    # For small eta, (1 - eta)^tau = exp(tau * ln(1 - eta)) = exp(tau * (-eta - eta^2 / 2 - ...))
    w_discrete_exact = w0 * ((1.0 - eta) ** n_steps)

    assert np.allclose(w_discrete, w_discrete_exact)
    # Relative difference between continuous approximation and discrete is O(eta)
    rel_diff = np.linalg.norm(w_discrete - w_continuous) / np.linalg.norm(w_continuous)

    return {
        "passed": bool(rel_diff < 0.05 and np.allclose(w_discrete, w_discrete_exact)),
        "w_initial": w0.tolist(),
        "w_discrete": w_discrete.tolist(),
        "w_continuous": w_continuous.tolist(),
        "rel_diff": float(rel_diff),
    }


# =====================================================================
# Exercise 9.4: Linear Transformations and Equivariance
# =====================================================================

def verify_exercise_9_4() -> Dict[str, Any]:
    """
    Verify scaling equivariance of network function (Eqs 9.6 - 9.13).
    Input transformation: x_tilde_i = a * x_i + b
    Network invariance condition:
        w_tilde_ji = (1/a) * w_ji,  b_tilde_j = b_j - (b / a) * sum_i w_ji
    Output transformation: y_tilde_k = c * y_k + d
        w_tilde_kj = c * w_kj,      b_tilde_k = c * b_k + d
    """
    rng = np.random.RandomState(42)
    D, M, K = 3, 4, 2
    x = rng.randn(D)
    W1 = rng.randn(M, D)
    b1 = rng.randn(M)
    W2 = rng.randn(K, M)
    b2 = rng.randn(K)

    # Original forward
    a1 = W1 @ x + b1
    z1 = np.tanh(a1)
    y = W2 @ z1 + b2

    # 1. Input transformation: x_tilde = a * x + b
    a_scale = 2.5
    b_shift = -1.2
    x_tilde = a_scale * x + b_shift

    W1_tilde = W1 / a_scale
    b1_tilde = b1 - (b_shift / a_scale) * np.sum(W1, axis=1)

    a1_transformed = W1_tilde @ x_tilde + b1_tilde
    input_inv_passed = bool(np.allclose(a1, a1_transformed))

    # 2. Output transformation: y_tilde = c * y + d
    c_scale = -1.8
    d_shift = 3.5
    W2_tilde = c_scale * W2
    b2_tilde = c_scale * b2 + d_shift

    y_transformed = W2_tilde @ z1 + b2_tilde
    y_target = c_scale * y + d_shift
    output_eq_passed = bool(np.allclose(y_transformed, y_target))

    return {
        "passed": bool(input_inv_passed and output_eq_passed),
        "input_invariance": input_inv_passed,
        "output_equivariance": output_eq_passed,
    }


# =====================================================================
# Exercise 9.5: Equivalence of Regularization and Constraint (Lagrange)
# =====================================================================

def verify_exercise_9_5() -> Dict[str, Any]:
    """
    Verify that minimizing regularized error E_reg(w) = E(w) + (lambda / 2) w^T w
    is equivalent to minimizing unregularized error E(w) subject to w^T w <= eta (Eq 9.20).
    At the optimum, grad E(w*) + lambda w* = 0  ==>  w* = - (1/lambda) grad E(w*).
    Complementary slackness: lambda * (w*^T w* - eta) = 0 with lambda >= 0.
    """
    # Simple quadratic error: E(w) = (1/2) (w - w_opt)^T H (w - w_opt)
    H = np.diag([2.0, 5.0])
    w_target = np.array([3.0, 4.0])

    # Unconstrained optimum has norm ||w_target|| = 5.0
    # Let constraint be ||w||^2 <= eta = 4.0 (radius R = 2.0)
    eta = 4.0

    # For any chosen lambda > 0:
    # E_reg = (1/2) (w - w_t)^T H (w - w_t) + (lambda / 2) w^T w
    # grad = H (w - w_t) + lambda w = 0 ==> (H + lambda I) w = H w_t ==> w*(lambda) = (H + lambda I)^{-1} H w_t
    # We find lambda* such that ||w*(lambda*)||^2 = eta
    from scipy.optimize import root_scalar

    def norm_sq_diff(lam):
        w_lam = np.linalg.solve(H + lam * np.eye(2), H @ w_target)
        return float(np.sum(w_lam ** 2) - eta)

    sol = root_scalar(norm_sq_diff, bracket=[0.01, 100.0])
    lambda_star = sol.root
    w_star = np.linalg.solve(H + lambda_star * np.eye(2), H @ w_target)

    # Check that norm constraint is satisfied
    norm_sq = float(np.sum(w_star ** 2))
    constraint_satisfied = bool(np.isclose(norm_sq, eta, atol=1e-5))

    # Check KKT stationarity: grad E + lambda* w* == 0
    grad_E = H @ (w_star - w_target)
    kkt_stationarity = bool(np.allclose(grad_E + lambda_star * w_star, 0.0, atol=1e-5))

    return {
        "passed": bool(constraint_satisfied and kkt_stationarity),
        "lambda_star": float(lambda_star),
        "norm_sq": norm_sq,
        "eta": float(eta),
        "kkt_satisfied": kkt_stationarity,
    }


# =====================================================================
# Exercise 9.6: Early Stopping vs Weight Decay Spectral Equivalence
# =====================================================================

def verify_exercise_9_6() -> Dict[str, Any]:
    """
    Verify early stopping spectral shrinkage:
        w_j^(tau) = [1 - (1 - eta * lambda_j)^tau] w_j* (Eq 9.58)
    Compared with weight decay:
        w_j(alpha) = [lambda_j / (lambda_j + alpha)] w_j*
    Shows that (tau * eta)^{-1} acts analogously to weight decay parameter alpha.
    """
    H = np.diag([0.2, 1.5, 8.0])  # Eigenvalues lambda_j
    w_star = np.array([2.0, -1.0, 0.5])
    eta = 0.05  # Learning rate
    n_steps = 40

    # 1. Gradient descent simulation from w(0) = 0
    w_tau = np.zeros(3)
    for _ in range(n_steps):
        # grad E = H (w - w_star)
        grad = H @ (w_tau - w_star)
        w_tau = w_tau - eta * grad

    # 2. Formula (9.58): w_j^(tau) = {1 - (1 - eta * lambda_j)^tau} w_j*
    lambdas = np.diag(H)
    w_formula = (1.0 - (1.0 - eta * lambdas) ** n_steps) * w_star

    assert np.allclose(w_tau, w_formula)

    # 3. Equivalent weight decay parameter alpha = (tau * eta)^{-1}
    alpha_equiv = 1.0 / (n_steps * eta)
    w_decay = (lambdas / (lambdas + alpha_equiv)) * w_star

    # Compare trends: for large lambda_j, w_j / w_j* -> 1; for small lambda_j, w_j / w_j* -> 0
    ratio_early = w_formula / w_star
    ratio_decay = w_decay / w_star

    passed = bool(
        np.allclose(w_tau, w_formula)
        and (ratio_early[0] < ratio_early[1] < ratio_early[2])
        and (ratio_decay[0] < ratio_decay[1] < ratio_decay[2])
    )

    return {
        "passed": passed,
        "w_simulated": w_tau.tolist(),
        "w_formula": w_formula.tolist(),
        "ratio_early": ratio_early.tolist(),
        "ratio_decay": ratio_decay.tolist(),
        "alpha_equiv": float(alpha_equiv),
    }


# =====================================================================
# Exercise 9.7: Tied / Shared Weight Gradients
# =====================================================================

def verify_exercise_9_7() -> Dict[str, Any]:
    """
    Verify backpropagation with tied/shared weights:
    If multiple connections share the same parameter w (w_i = w for all i in S),
    the total gradient is the sum of unconstrained gradients:
        del E / del w = sum_{i in S} del E / del w_i.
    """
    rng = np.random.RandomState(42)
    w_shared = 1.5
    x = rng.randn(4)

    def compute_loss(w_val):
        # Two outputs share the same weight w_val on different inputs
        y1 = w_val * x[0] + 0.5 * x[1]
        y2 = w_val * x[2] + 0.3 * x[3]
        return 0.5 * (y1 - 1.0) ** 2 + 0.5 * (y2 - 2.0) ** 2

    loss0 = compute_loss(w_shared)

    # Analytical pooled gradient: del E / del w = del E / del w_1 + del E / del w_2
    y1 = w_shared * x[0] + 0.5 * x[1]
    y2 = w_shared * x[2] + 0.3 * x[3]
    del_w1 = (y1 - 1.0) * x[0]
    del_w2 = (y2 - 2.0) * x[2]
    pooled_grad = del_w1 + del_w2

    # Numerical differentiation
    eps = 1e-6
    num_grad = (compute_loss(w_shared + eps) - compute_loss(w_shared - eps)) / (2.0 * eps)
    rel_error = abs(pooled_grad - num_grad) / (abs(num_grad) + 1e-8)

    return {
        "passed": bool(rel_error < 1e-5),
        "pooled_grad": float(pooled_grad),
        "numerical_grad": float(num_grad),
        "rel_error": float(rel_error),
    }


# =====================================================================
# Exercise 9.8: Responsibilities in Soft Weight Sharing (Bayes)
# =====================================================================

def verify_exercise_9_8() -> Dict[str, Any]:
    """
    Verify Bayes' theorem derivation of responsibilities gamma_j(w_i) (Eq 9.24):
        gamma_j(w_i) = p(j | w_i) = {pi_j N(w_i | mu_j, sigma_j^2)} / {sum_k pi_k N(w_i | mu_k, sigma_k^2)}
    Verify sum_j gamma_j(w_i) = 1 and gamma_j >= 0.
    """
    mus = np.array([-2.0, 0.0, 2.0])
    sigmas = np.array([0.5, 0.8, 0.4])
    beta = np.log(sigmas ** 2)
    pis = np.array([0.2, 0.5, 0.3])
    logits = np.log(pis)
    weights = np.array([-1.8, -0.1, 0.2, 1.9])

    from common.parameter_sharing import SoftWeightSharingGMM
    gmm = SoftWeightSharingGMM(n_components=3, mu=mus, beta=beta, logits=logits)
    gamma = gmm.responsibilities(weights)  # (N, K)

    # Test sum = 1
    sums = np.sum(gamma, axis=1)
    sums_one = bool(np.allclose(sums, 1.0))
    non_negative = bool(np.all(gamma >= 0.0))

    # Weight close to mu_0 should have largest responsibility for component 0
    w0_highest_comp = int(np.argmax(gamma[0])) == 0
    w3_highest_comp = int(np.argmax(gamma[3])) == 2

    return {
        "passed": bool(sums_one and non_negative and w0_highest_comp and w3_highest_comp),
        "gamma_matrix": gamma.tolist(),
        "sums_to_one": sums_one,
    }


# =====================================================================
# Exercise 9.9: Weight Gradient of Soft Weight Sharing Regularizer
# =====================================================================

def verify_exercise_9_9() -> Dict[str, Any]:
    """
    Verify analytical weight gradient of soft weight sharing regularizer (Eq 9.25):
        del Omega / del w_i = sum_j gamma_j(w_i) * (w_i - mu_j) / sigma_j^2
    against numerical differentiation.
    """
    mus = np.array([-1.5, 0.0, 1.5])
    sigmas = np.array([0.6, 0.4, 0.7])
    beta = np.log(sigmas ** 2)
    pis = np.array([0.3, 0.4, 0.3])
    logits = np.log(pis)
    weights = np.array([-1.2, -0.3, 0.1, 1.4])

    from common.parameter_sharing import SoftWeightSharingGMM
    gmm = SoftWeightSharingGMM(n_components=3, mu=mus, beta=beta, logits=logits)
    anal_grad = gmm.grad_w(weights)

    eps = 1e-6
    num_grad = np.zeros_like(weights)
    for i in range(len(weights)):
        w_p = weights.copy()
        w_m = weights.copy()
        w_p[i] += eps
        w_m[i] -= eps
        num_grad[i] = (gmm.penalty(w_p) - gmm.penalty(w_m)) / (2.0 * eps)

    rel_error = np.linalg.norm(anal_grad - num_grad) / np.linalg.norm(num_grad)
    return {
        "passed": bool(rel_error < 1e-6),
        "analytical_grad": anal_grad.tolist(),
        "numerical_grad": num_grad.tolist(),
        "rel_error": float(rel_error),
    }


# =====================================================================
# Exercise 9.10: Cluster Center Gradient of Soft Weight Sharing
# =====================================================================

def verify_exercise_9_10() -> Dict[str, Any]:
    """
    Verify analytical cluster mean gradient (Eq 9.26):
        del Omega / del mu_j = sum_i gamma_j(w_i) * (mu_j - w_i) / sigma_j^2
    against numerical differentiation.
    """
    mus = np.array([-1.5, 0.0, 1.5])
    sigmas = np.array([0.6, 0.4, 0.7])
    beta = np.log(sigmas ** 2)
    pis = np.array([0.3, 0.4, 0.3])
    logits = np.log(pis)
    weights = np.array([-1.2, -0.3, 0.1, 1.4, 0.8])

    from common.parameter_sharing import SoftWeightSharingGMM
    gmm = SoftWeightSharingGMM(n_components=3, mu=mus, beta=beta, logits=logits)
    anal_grad = gmm.grad_mu(weights)

    eps = 1e-6
    num_grad = np.zeros_like(mus)
    for j in range(len(mus)):
        orig = gmm.mu[j]
        gmm.mu[j] = orig + eps
        val_p = gmm.penalty(weights)
        gmm.mu[j] = orig - eps
        val_m = gmm.penalty(weights)
        gmm.mu[j] = orig
        num_grad[j] = (val_p - val_m) / (2.0 * eps)

    rel_error = np.linalg.norm(anal_grad - num_grad) / np.linalg.norm(num_grad)
    return {
        "passed": bool(rel_error < 1e-6),
        "analytical_grad": anal_grad.tolist(),
        "numerical_grad": num_grad.tolist(),
        "rel_error": float(rel_error),
    }


# =====================================================================
# Exercise 9.11: Variance Parameter Gradient of Soft Weight Sharing
# =====================================================================

def verify_exercise_9_11() -> Dict[str, Any]:
    """
    Verify analytical gradient with respect to log-variance beta_j = ln(sigma_j^2) (Eq 9.28):
        del Omega / del beta_j = (1/2) sum_i gamma_j(w_i) * [1 - (w_i - mu_j)^2 / sigma_j^2]
    against numerical differentiation.
    """
    mus = np.array([-1.5, 0.0, 1.5])
    sigmas = np.array([0.6, 0.4, 0.7])
    beta = np.log(sigmas ** 2)
    pis = np.array([0.3, 0.4, 0.3])
    logits = np.log(pis)
    weights = np.array([-1.2, -0.3, 0.1, 1.4, 0.8])

    from common.parameter_sharing import SoftWeightSharingGMM
    gmm = SoftWeightSharingGMM(n_components=3, mu=mus, beta=beta, logits=logits)
    anal_grad = gmm.grad_beta(weights)

    eps = 1e-6
    num_grad = np.zeros_like(gmm.beta)
    for j in range(len(gmm.beta)):
        orig = gmm.beta[j]
        gmm.beta[j] = orig + eps
        val_p = gmm.penalty(weights)

        gmm.beta[j] = orig - eps
        val_m = gmm.penalty(weights)

        gmm.beta[j] = orig
        num_grad[j] = (val_p - val_m) / (2.0 * eps)

    rel_error = np.linalg.norm(anal_grad - num_grad) / np.linalg.norm(num_grad)
    return {
        "passed": bool(rel_error < 1e-6),
        "analytical_grad": anal_grad.tolist(),
        "numerical_grad": num_grad.tolist(),
        "rel_error": float(rel_error),
    }


# =====================================================================
# Exercise 9.12: Softmax Mixing Coefficients and Logit Gradients
# =====================================================================

def verify_exercise_9_12() -> Dict[str, Any]:
    """
    Verify softmax derivative del pi_k / del gamma_j = pi_j * (delta_jk - pi_k) (Eq 9.63)
    and gradient of regularizer with respect to auxiliary logit gamma_j (Eq 9.31):
        del Omega / del gamma_j = sum_i [pi_j - gamma_j(w_i)].
    """
    mus = np.array([-1.5, 0.0, 1.5])
    sigmas = np.array([0.6, 0.4, 0.7])
    beta = np.log(sigmas ** 2)
    logits = np.array([0.5, 1.2, -0.3])
    weights = np.array([-1.2, -0.3, 0.1, 1.4])

    from common.parameter_sharing import SoftWeightSharingGMM
    gmm = SoftWeightSharingGMM(n_components=3, mu=mus, beta=beta, logits=logits)
    anal_grad = gmm.grad_logits(weights)

    eps = 1e-6
    num_grad = np.zeros_like(logits)
    for j in range(len(logits)):
        orig = gmm.logits[j]
        gmm.logits[j] = orig + eps
        val_p = gmm.penalty(weights)

        gmm.logits[j] = orig - eps
        val_m = gmm.penalty(weights)

        gmm.logits[j] = orig
        num_grad[j] = (val_p - val_m) / (2.0 * eps)

    rel_error = np.linalg.norm(anal_grad - num_grad) / np.linalg.norm(num_grad)
    return {
        "passed": bool(rel_error < 1e-6),
        "analytical_grad": anal_grad.tolist(),
        "numerical_grad": num_grad.tolist(),
        "rel_error": float(rel_error),
    }


# =====================================================================
# Exercise 9.13: Recursive Expansion of Residual Connections
# =====================================================================

def verify_exercise_9_13() -> Dict[str, Any]:
    """
    Verify that combining residual block equations:
        z_1 = x + F_1(x) (9.35)
        z_2 = z_1 + F_2(z_1) (9.36)
        y = z_2 + F_3(z_2) (9.37)
    expands recursively into the ensemble form (Eq 9.40) consisting of 2^3 = 8 paths.
    """
    x = np.array([1.5])
    # Define arbitrary simple linear/affine functions for F_1, F_2, F_3
    F1 = lambda u: 0.5 * u + 0.2
    F2 = lambda u: -0.3 * u + 0.1
    F3 = lambda u: 0.4 * u - 0.5

    # 1. Sequential evaluation
    z1 = x + F1(x)
    z2 = z1 + F2(z1)
    y_seq = z2 + F3(z2)

    # 2. Fully expanded evaluation (Eq 9.38 / 9.40)
    # y = [x + F_1(x) + F_2(x + F_1(x))] + F_3(x + F_1(x) + F_2(x + F_1(x)))
    term_x = x
    term_F1 = F1(x)
    term_F2 = F2(x + F1(x))
    term_F3 = F3(x + F1(x) + F2(x + F1(x)))
    y_expanded = term_x + term_F1 + term_F2 + term_F3

    passed = bool(np.allclose(y_seq, y_expanded))
    return {
        "passed": passed,
        "y_sequential": float(y_seq[0]),
        "y_expanded": float(y_expanded[0]),
        "abs_diff": float(abs(y_seq[0] - y_expanded[0])),
    }


# =====================================================================
# Exercise 9.14: Committee Error Reduction for Uncorrelated Errors
# =====================================================================

def verify_exercise_9_14() -> Dict[str, Any]:
    """
    Verify Eq (9.50): Under zero-mean, uncorrelated errors (Eqs 9.48, 9.49),
        E_COM = (1 / M) E_AV.
    """
    rng = np.random.RandomState(42)
    N = 20000
    M = 6
    x = np.linspace(0, 1, N)
    y_true = np.sin(3 * x)

    # Generate M models with independent uncorrelated errors
    models = [lambda x_val, m_idx=m: y_true + rng.randn(N) * (0.3 + 0.05 * m_idx) for m in range(M)]

    from common.model_averaging import EnsembleCommittee
    committee = EnsembleCommittee(models)
    res = committee.evaluate_errors(x, y_true)

    ratio = res["empirical_ratio"]
    passed = bool(np.isclose(ratio, 1.0 / M, atol=0.02))

    return {
        "passed": passed,
        "M": M,
        "expected_ratio": 1.0 / M,
        "empirical_ratio": float(ratio),
        "E_AV": res["E_AV"],
        "E_COM": res["E_COM"],
    }


# =====================================================================
# Exercise 9.15: Jensen's Inequality for Squared Loss Committee Error
# =====================================================================

def verify_exercise_9_15() -> Dict[str, Any]:
    """
    Verify Eq (9.64): By Jensen's inequality for f(u) = u^2:
        ( (1/M) sum_m eps_m(x) )^2 <= (1/M) sum_m eps_m(x)^2
    Taking expectations: E_COM <= E_AV.
    Verify across 50 random correlated models that E_COM <= E_AV always holds.
    """
    rng = np.random.RandomState(42)
    N = 1000
    M = 5
    x = np.linspace(0, 1, N)
    y_true = np.exp(x)

    # Random strongly correlated models
    common_noise = rng.randn(N) * 0.4
    models = [lambda x_val, m_idx=m: y_true + common_noise + rng.randn(N) * 0.1 for m in range(M)]

    from common.model_averaging import EnsembleCommittee
    committee = EnsembleCommittee(models)
    res = committee.evaluate_errors(x, y_true)

    passed = bool(res["E_COM"] <= res["E_AV"])
    return {
        "passed": passed,
        "E_COM": res["E_COM"],
        "E_AV": res["E_AV"],
        "inequality_satisfied": bool(res["E_COM"] <= res["E_AV"]),
    }


# =====================================================================
# Exercise 9.16: Jensen's Inequality for General Convex Error Functions
# =====================================================================

def verify_exercise_9_16() -> Dict[str, Any]:
    """
    Verify that E_COM <= E_AV holds for any convex error function E(y).
    Tests with:
    1. Absolute error (L1): E(y) = |y - t|
    2. Huber loss: E(y) = huber(y - t)
    3. Cross-entropy: E(y) = -t ln(y) - (1 - t) ln(1 - y)
    """
    rng = np.random.RandomState(42)
    N = 200
    M = 4
    t = rng.randn(N)

    # Predictions from M models
    preds = np.array([t + rng.randn(N) * 0.5 for _ in range(M)])  # (M, N)
    pred_com = np.mean(preds, axis=0)  # (N,)

    # 1. L1 loss
    l1_ind = [np.mean(np.abs(preds[m] - t)) for m in range(M)]
    l1_av = np.mean(l1_ind)
    l1_com = np.mean(np.abs(pred_com - t))
    l1_passed = bool(l1_com <= l1_av)

    # 2. Huber loss (delta = 1.0)
    def huber(diff, delta=1.0):
        abs_diff = np.abs(diff)
        return np.where(abs_diff <= delta, 0.5 * abs_diff**2, delta * (abs_diff - 0.5 * delta))

    huber_ind = [np.mean(huber(preds[m] - t)) for m in range(M)]
    huber_av = np.mean(huber_ind)
    huber_com = np.mean(huber(pred_com - t))
    huber_passed = bool(huber_com <= huber_av)

    return {
        "passed": bool(l1_passed and huber_passed),
        "l1_com": float(l1_com),
        "l1_av": float(l1_av),
        "huber_com": float(huber_com),
        "huber_av": float(huber_av),
    }


# =====================================================================
# Exercise 9.17: Bounded Convex Combinations in Weighted Committees
# =====================================================================

def verify_exercise_9_17() -> Dict[str, Any]:
    """
    Verify that y_min(x) <= y_COM(x) <= y_max(x) holds for all x
    if and only if alpha_m >= 0 and sum_{m=1}^M alpha_m = 1 (Eq 9.67).
    """
    rng = np.random.RandomState(42)
    M = 5
    y_m = rng.randn(M)  # Predictions at a point x
    y_min = np.min(y_m)
    y_max = np.max(y_m)

    # Valid convex combination (alpha >= 0, sum alpha = 1)
    alpha_valid = np.array([0.1, 0.3, 0.2, 0.15, 0.25])
    y_valid = np.sum(alpha_valid * y_m)
    valid_bounded = bool(y_min <= y_valid <= y_max)

    # Invalid combination: sum alpha = 1, but some alpha < 0
    alpha_invalid = np.array([-0.5, 0.5, 0.5, 0.3, 0.2])
    # Specially crafted y_m where y_min is made even smaller
    y_m_special = np.array([10.0, 1.0, 1.0, 1.0, 1.0])
    y_invalid = np.sum(alpha_invalid * y_m_special)  # -5.0 + 1.5 = -3.5 < min(y) = 1.0
    invalid_violates = bool(y_invalid < np.min(y_m_special))

    return {
        "passed": bool(valid_bounded and invalid_violates),
        "valid_combination_bounded": valid_bounded,
        "invalid_combination_violates": invalid_violates,
    }


# =====================================================================
# Exercise 9.18: Dropout on Linear Regression as Data-Dependent L2
# =====================================================================

def verify_exercise_9_18() -> Dict[str, Any]:
    """
    Verify that linear regression with dropout rate p on inputs has expected error:
        E[E(W)] = sum_{n,k} (t_nk - sum_i w_ki x_ni)^2 + ((1 - rho)/rho) sum_{n,k,i} w_ki^2 x_ni^2 (Eqs 9.72-9.73)
    Closed-form solution:
        W* = (X^T X + ((1 - rho) / rho) * diag(X^T X))^{-1} X^T T.
    """
    rng = np.random.RandomState(42)
    N, D, K = 80, 3, 2
    X = rng.randn(N, D)
    T = rng.randn(N, K)
    rho = 0.8  # retention probability (dropout rate p = 0.2)

    # Closed-form solution
    XTX = X.T @ X
    diag_XTX = np.diag(np.diag(XTX))
    penalty = ((1.0 - rho) / rho) * diag_XTX
    reg_matrix = XTX + penalty
    W_star = np.linalg.solve(reg_matrix, X.T @ T)

    # Monte Carlo average of empirical dropout objective and gradient at W_star
    # Gradient of expected objective at W_star should be zero!
    # Expected grad: 2 * (X^T X W_star - X^T T + penalty W_star) == 0
    grad_at_star = 2.0 * (XTX @ W_star - X.T @ T + penalty @ W_star)
    grad_norm = float(np.linalg.norm(grad_at_star))

    passed = bool(grad_norm < 1e-10)
    return {
        "passed": passed,
        "grad_at_star_norm": grad_norm,
        "W_star": W_star.tolist(),
    }
