"""Tests for Chapter 7 Section 7.2: Gradient Descent Optimization.

Tests cover:
- Section 7.2.1: Use of Gradient Information (Complexity and parameter counting)
- Section 7.2.2: Batch Gradient Descent (Eq 7.16)
- Section 7.2.3: Stochastic Gradient Descent (Algorithm 7.1, Eq 7.17, 7.18)
- Section 7.2.4: Mini-batches (Algorithm 7.2, diminishing returns sigma / sqrt(B))
- Section 7.2.5: Parameter Initialization (He, Glorot, symmetry breaking, variance propagation)
"""

import os
import numpy as np
import pytest

from common.gradient_descent import (
    OptimizationHistory,
    batch_gradient_descent,
    compute_gradient_noise_vs_batch_size,
    count_independent_quadratic_parameters,
    generate_gd_comparison_figure,
    generate_minibatch_noise_figure,
    generate_variance_propagation_figure,
    glorot_normal_init,
    glorot_uniform_init,
    gradient_vs_function_eval_effort,
    he_normal_init,
    he_uniform_init,
    minibatch_gradient_descent,
    simulate_variance_propagation,
    stochastic_gradient_descent,
    verify_symmetry_breaking,
    zero_init,
)


def test_count_independent_quadratic_parameters():
    """Verify W(W + 3) / 2 parameter count in local quadratic expansion (Exercise 7.7)."""
    assert count_independent_quadratic_parameters(1) == 2
    assert count_independent_quadratic_parameters(2) == 5
    assert count_independent_quadratic_parameters(3) == 9
    assert count_independent_quadratic_parameters(10) == 65
    
    with pytest.raises(ValueError):
        count_independent_quadratic_parameters(0)
    with pytest.raises(ValueError):
        count_independent_quadratic_parameters(-5)


def test_gradient_vs_function_eval_effort():
    """Verify asymptotic computational effort: O(W^3) without gradients vs O(W^2) with gradients."""
    effort = gradient_vs_function_eval_effort(10)
    assert effort["params_count"] == 10
    assert effort["independent_pieces_of_info"] == 65
    assert effort["no_grad_evals"] == 100
    assert effort["no_grad_total_steps"] == 1000
    assert effort["with_grad_evals"] == 10
    assert effort["with_grad_total_steps"] == 100


def test_batch_gradient_descent_quadratic():
    """Verify Batch GD convergence on a 2D positive-definite quadratic surface."""
    # E(w) = 0.5 * w^T A w - b^T w, grad E(w) = A w - b
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([2.0, 4.0])
    w_star = np.linalg.solve(A, b)
    
    def error_fn(w: np.ndarray) -> float:
        return float(0.5 * w.T @ A @ w - b.T @ w)
        
    def grad_fn(w: np.ndarray) -> np.ndarray:
        return A @ w - b
        
    w_init = np.array([-2.0, 5.0])
    hist = batch_gradient_descent(
        w_init=w_init,
        grad_fn=grad_fn,
        error_fn=error_fn,
        lr=0.2,
        max_epochs=150,
        tol=1e-5,
    )
    
    assert isinstance(hist, OptimizationHistory)
    assert hist.converged is True
    assert np.allclose(hist.weights[-1], w_star, atol=1e-4)
    assert hist.errors[-1] <= hist.errors[0]
    assert hist.grad_norms[-1] <= 1e-5


def test_stochastic_gradient_descent_algorithm_7_1():
    """Verify Stochastic Gradient Descent (Algorithm 7.1) on synthetic linear regression."""
    rng = np.random.default_rng(123)
    N = 100
    w_true = np.array([2.0, -1.5])
    X = rng.normal(size=(N, 2))
    y = X @ w_true + rng.normal(scale=0.05, size=N)
    
    def error_fn(w: np.ndarray) -> float:
        err = X @ w - y
        return 0.5 * float(np.mean(err ** 2))
        
    def grad_sample_fn(w: np.ndarray, n: int) -> np.ndarray:
        return (np.dot(X[n], w) - y[n]) * X[n]
        
    w_init = np.array([0.0, 0.0])
    hist = stochastic_gradient_descent(
        w_init=w_init,
        grad_sample_fn=grad_sample_fn,
        n_samples=N,
        error_fn=error_fn,
        lr=0.01,
        max_epochs=20,
        shuffle=True,
        random_state=42,
    )
    
    assert hist.iterations == N * 20
    assert hist.epochs == 20
    assert hist.errors[-1] < hist.errors[0]
    # Check close recovery of parameters
    assert np.allclose(hist.weights[-1], w_true, atol=0.2)


def test_minibatch_gradient_descent_algorithm_7_2():
    """Verify Mini-batch SGD (Algorithm 7.2) on synthetic linear regression."""
    rng = np.random.default_rng(456)
    N = 128
    B = 16
    w_true = np.array([1.0, 3.0])
    X = rng.normal(size=(N, 2))
    y = X @ w_true + rng.normal(scale=0.05, size=N)
    
    def error_fn(w: np.ndarray) -> float:
        err = X @ w - y
        return 0.5 * float(np.mean(err ** 2))
        
    def grad_minibatch_fn(w: np.ndarray, batch_idx: np.ndarray) -> np.ndarray:
        X_b = X[batch_idx]
        y_b = y[batch_idx]
        return (X_b.T @ (X_b @ w - y_b)) / len(batch_idx)
        
    w_init = np.array([0.0, 0.0])
    hist = minibatch_gradient_descent(
        w_init=w_init,
        grad_minibatch_fn=grad_minibatch_fn,
        n_samples=N,
        batch_size=B,
        error_fn=error_fn,
        lr=0.05,
        max_epochs=15,
        shuffle=True,
        random_state=42,
    )
    
    assert hist.epochs == 15
    assert hist.errors[-1] < hist.errors[0]
    assert np.allclose(hist.weights[-1], w_true, atol=0.15)


def test_diminishing_returns_noise_scaling():
    """Verify that empirical gradient estimation noise scales as sigma / sqrt(B) (Section 7.2.4)."""
    rng = np.random.default_rng(789)
    N = 500
    X = rng.normal(size=(N, 3))
    y = X @ np.array([1.0, -1.0, 2.0]) + rng.normal(scale=0.5, size=N)
    w = np.zeros(3)
    
    def grad_sample_fn(w: np.ndarray, n: int) -> np.ndarray:
        return (np.dot(X[n], w) - y[n]) * X[n]
        
    batch_sizes = [4, 16, 64]
    res = compute_gradient_noise_vs_batch_size(
        grad_sample_fn, w, n_samples=N, batch_sizes=batch_sizes, num_trials=120, random_state=42
    )
    
    emp_std = res["empirical_std"]
    # Ratio between B=4 and B=16: theoretical is sqrt(16/4) = 2.0
    ratio_4_16 = emp_std[0] / emp_std[1]
    # Ratio between B=16 and B=64: theoretical is sqrt(64/16) = 2.0
    ratio_16_64 = emp_std[1] / emp_std[2]
    
    assert 1.6 <= ratio_4_16 <= 2.4
    assert 1.6 <= ratio_16_64 <= 2.4


def test_initialization_schemes():
    """Verify statistical properties of He, Glorot, and Zero initializations."""
    fan_in = 200
    fan_out = 100
    rng = np.random.default_rng(101)
    
    # 1. He normal: var ~= 2 / fan_in
    w_he_n = he_normal_init(fan_in, fan_out, rng=rng)
    assert w_he_n.shape == (fan_out, fan_in)
    expected_he_var = 2.0 / fan_in
    assert np.isclose(np.var(w_he_n), expected_he_var, rtol=0.15)
    
    # 2. He uniform: var = (2 * limit)^2 / 12 = limit^2 / 3 = 2 / fan_in
    w_he_u = he_uniform_init(fan_in, fan_out, rng=rng)
    assert np.isclose(np.var(w_he_u), expected_he_var, rtol=0.15)
    
    # 3. Glorot normal: var ~= 2 / (fan_in + fan_out)
    w_glo_n = glorot_normal_init(fan_in, fan_out, rng=rng)
    expected_glo_var = 2.0 / (fan_in + fan_out)
    assert np.isclose(np.var(w_glo_n), expected_glo_var, rtol=0.15)
    
    # 4. Glorot uniform: var ~= 2 / (fan_in + fan_out)
    w_glo_u = glorot_uniform_init(fan_in, fan_out, rng=rng)
    assert np.isclose(np.var(w_glo_u), expected_glo_var, rtol=0.15)
    
    # 5. Zero init
    w_zero = zero_init(fan_in, fan_out)
    assert np.all(w_zero == 0.0)


def test_symmetry_breaking():
    """Verify that zero initialization leads to identical redundant hidden units."""
    res = verify_symmetry_breaking(hidden_units=4, input_dim=3, steps=5)
    assert res["zero_rows_identical"] is True
    assert res["rand_rows_distinct"] is True


def test_variance_propagation_relu():
    """Verify He initialization preserves signal variance across deep layers (Eq 7.21, 7.22)."""
    # He init: var[z^(l)] stays approximately constant
    res_he = simulate_variance_propagation(
        depth=15, width=128, init_type="he", activation="relu", num_samples=3000, seed=42
    )
    final_var_he = res_he["post_act_vars"][-1]
    assert 0.3 < final_var_he < 3.0  # preserved within reasonable order
    
    # Pre-activation mean should stay close to 0 (Eq 7.21)
    assert np.all(np.abs(res_he["pre_act_means"]) < 0.35)
    
    # Standard normal init: var[z^(l)] explodes exponentially by factor of (M/2) per layer
    res_std = simulate_variance_propagation(
        depth=5, width=64, init_type="standard_normal", activation="relu", num_samples=1000, seed=42
    )
    assert res_std["post_act_vars"][-1] > 100.0  # massive explosion
    
    # Xavier init with ReLU: decays by factor of (1/2) per layer
    res_xavier = simulate_variance_propagation(
        depth=10, width=128, init_type="xavier", activation="relu", num_samples=2000, seed=42
    )
    assert res_xavier["post_act_vars"][-1] < 0.1  # decayed heavily


def test_generate_figures():
    """Verify that figure generation functions successfully save images."""
    p1, r1 = generate_minibatch_noise_figure()
    assert os.path.exists(p1)
    
    p2, r2 = generate_variance_propagation_figure()
    assert os.path.exists(p2)
    
    p3, r3 = generate_gd_comparison_figure()
    assert os.path.exists(p3)
