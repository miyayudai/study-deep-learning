"""Tests for Chapter 7 Exercises (7.1 to 7.14).

Bishop & Bishop (2024), Deep Learning: Foundations and Concepts, pp. 230-232.
"""

import numpy as np
import pytest

from common.exercises_ch7 import (
    exercise_7_1_verify_eigen_quadratic_form,
    exercise_7_2_positive_definite_eigenvalues,
    exercise_7_3_local_minimum_conditions,
    exercise_7_4_linear_regression_hessian,
    exercise_7_5_logistic_regression_hessian,
    exercise_7_6_ellipse_axes_and_lengths,
    exercise_7_7_quadratic_independent_parameters,
    exercise_7_8_sample_mean_variance,
    exercise_7_9_relu_variance_propagation,
    exercise_7_10_eigen_update_derivation,
    exercise_7_11_nesterov_momentum_equivalence,
    exercise_7_12_ema_bias_correction,
    exercise_7_13_line_search_orthogonality,
    exercise_7_14_standardization_moments,
)


def test_exercise_7_1():
    """Exercise 7.1: Verify decoupled eigen-basis quadratic form E(w) (Eq 7.11)."""
    H = np.array([[4.0, 1.0], [1.0, 3.0]])
    w = np.array([2.5, -1.0])
    w_star = np.array([1.0, 0.5])
    res = exercise_7_1_verify_eigen_quadratic_form(H, w, w_star, E_star=2.0)
    assert res["is_equivalent"] is True
    assert np.isclose(res["error_direct"], res["error_eigen"])


def test_exercise_7_2():
    """Exercise 7.2: Verify H is positive definite iff all eigenvalues > 0."""
    H_pd = np.array([[5.0, 2.0], [2.0, 3.0]])
    res_pd = exercise_7_2_positive_definite_eigenvalues(H_pd)
    assert res_pd["all_eigenvalues_positive"] is True
    assert res_pd["empirical_v_quadratics_positive"] is True
    assert res_pd["is_positive_definite"] is True

    H_indef = np.array([[1.0, 3.0], [3.0, 1.0]])
    res_indef = exercise_7_2_positive_definite_eigenvalues(H_indef)
    assert res_indef["all_eigenvalues_positive"] is False
    assert res_indef["is_positive_definite"] is False


def test_exercise_7_3():
    """Exercise 7.3: Check necessary and sufficient conditions for local minimum."""
    grad_zero = np.array([0.0, 0.0])
    H_pd = np.array([[3.0, 1.0], [1.0, 2.0]])
    res_min = exercise_7_3_local_minimum_conditions(grad_zero, H_pd)
    assert res_min["is_stationary"] is True
    assert res_min["is_positive_definite"] is True
    assert res_min["is_local_minimum"] is True

    grad_nonzero = np.array([1.0, -0.5])
    res_not_stat = exercise_7_3_local_minimum_conditions(grad_nonzero, H_pd)
    assert res_not_stat["is_local_minimum"] is False


def test_exercise_7_4():
    """Exercise 7.4: Linear regression Hessian trace and determinant positivity."""
    rng = np.random.default_rng(42)
    x = rng.normal(loc=2.0, scale=3.0, size=50)
    res = exercise_7_4_linear_regression_hessian(x)
    assert res["trace_positive"] is True
    assert res["det_positive"] is True
    assert np.isclose(res["determinant"], res["det_theoretical"])
    assert res["is_minimum"] is True


def test_exercise_7_5():
    """Exercise 7.5: Logistic classification Hessian trace and determinant positivity."""
    rng = np.random.default_rng(123)
    x = rng.normal(size=40)
    res = exercise_7_5_logistic_regression_hessian(x, w=1.2, b=-0.5)
    assert res["trace_positive"] is True
    assert res["det_positive"] is True
    assert res["is_minimum"] is True


def test_exercise_7_6():
    """Exercise 7.6: Elliptical error contours semi-axis lengths r_i propto lambda_i^(-1/2)."""
    H = np.array([[6.0, 2.0], [2.0, 3.0]])
    res = exercise_7_6_ellipse_axes_and_lengths(H, delta_E=2.0)
    assert res["proportional_to_inv_sqrt"] is True
    # Larger eigenvalue has shorter semi-axis length
    evals = res["eigenvalues"]
    semi_axes = res["semi_axis_lengths"]
    if evals[1] > evals[0]:
        assert semi_axes[1] < semi_axes[0]


def test_exercise_7_7():
    """Exercise 7.7: Quadratic independent parameter count W(W + 3) / 2."""
    for W in [1, 2, 5, 20]:
        res = exercise_7_7_quadratic_independent_parameters(W)
        assert res["matches_formula"] == 1
        assert res["total_independent_elements"] == W * (W + 3) // 2


def test_exercise_7_8():
    """Exercise 7.8: Sample mean MSE sigma^2 / N and RMS sigma / sqrt(N)."""
    res = exercise_7_8_sample_mean_variance(N=64, sigma=4.0, num_experiments=3000)
    assert res["relative_error"] < 0.1
    assert np.isclose(res["theoretical_rms"], 4.0 / 8.0)


def test_exercise_7_9():
    """Exercise 7.9: He initialization ReLU variance propagation preservation."""
    res = exercise_7_9_relu_variance_propagation(M=128, input_variance=1.0, num_samples=6000)
    assert abs(res["empirical_a_mean"]) < 0.1
    assert res["variance_preserved"] is True
    assert np.isclose(res["theoretical_z_var"], 1.0)


def test_exercise_7_10():
    """Exercise 7.10: Decoupled eigen-update Delta alpha_i = -eta * lambda_i * alpha_i."""
    H = np.array([[5.0, 1.0], [1.0, 2.0]])
    w = np.array([3.0, -2.0])
    w_star = np.array([0.5, 0.5])
    res = exercise_7_10_eigen_update_derivation(H, w, w_star, lr=0.1)
    assert res["is_exact_match"] is True


def test_exercise_7_11():
    """Exercise 7.11: Nesterov momentum first-order equivalence to standard momentum."""
    # Quadratic function E(w) = 0.5 * w^T H w
    H = np.array([[3.0, 0.5], [0.5, 1.5]])
    w = np.array([1.0, -1.0])
    delta_w_prev = np.array([0.1, -0.05])
    
    def grad_fn(v):
        return H @ v
    def hess_fn(v):
        return H
        
    res = exercise_7_11_nesterov_momentum_equivalence(
        w, delta_w_prev, grad_fn, hess_fn, lr=0.01, mu=0.8
    )
    assert res["is_close_first_order"] is True
    assert res["diff_actual_norm"] < 0.05


def test_exercise_7_12():
    """Exercise 7.12: EMA bias correction factor 1 / (1 - beta^n) (Eq 7.68)."""
    res = exercise_7_12_ema_bias_correction(beta=0.9, num_steps=25, true_mean=10.0)
    assert res["bias_corrected_is_exact"] is True
    # Initial raw EMA is heavily biased towards zero
    assert np.isclose(res["mu_raw"][0], 1.0)  # (1 - 0.9) * 10 = 1.0
    assert np.isclose(res["mu_corrected"][0], 10.0)  # 1.0 / (1 - 0.9) = 10.0


def test_exercise_7_13():
    """Exercise 7.13: Line search minimum orthogonality d^T grad E = 0."""
    A = np.array([[4.0, 1.0], [1.0, 2.0]])
    def error_fn(v):
        return float(0.5 * v.T @ A @ v)
    def grad_fn(v):
        return A @ v
        
    w = np.array([3.0, -2.0])
    d = np.array([-1.0, 0.5])
    res = exercise_7_13_line_search_orthogonality(error_fn, grad_fn, w, d)
    assert res["is_orthogonal"] is True


def test_exercise_7_14():
    """Exercise 7.14: Renormalized variables have zero mean and unit variance."""
    rng = np.random.default_rng(999)
    x = rng.normal(loc=12.5, scale=4.2, size=100)
    res = exercise_7_14_standardization_moments(x)
    assert res["is_zero_mean"] is True
    assert res["is_unit_variance"] is True
