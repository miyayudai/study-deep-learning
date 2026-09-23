"""
Tests for Chapter 9 Section 9.2 Weight Decay (Bishop & Bishop 2024).
"""

from pathlib import Path
import numpy as np
import pytest

from common.weight_decay import (
    QuadraticObjective,
    MLP2Layer,
    sample_mlp_functions,
    lq_penalty,
    lq_contour_points,
    solve_constrained_qp,
    generate_figure_9_3,
    generate_figure_9_4,
    generate_figure_9_5,
    generate_figure_9_6,
)


def test_quadratic_objective_values_and_gradients():
    """Verify quadratic objective and weight-decay regularized gradient (Eq 9.5)."""
    H = np.array([[2.0, 0.5], [0.5, 3.0]])
    w_star = np.array([1.5, -2.0])
    obj = QuadraticObjective(H=H, w_star=w_star, E_0=1.0)

    # At w_star, gradient should be 0 and value should be E_0
    assert np.isclose(obj.value(w_star), 1.0)
    assert np.allclose(obj.gradient(w_star), np.zeros(2))

    w = np.array([0.5, 1.0])
    # Regularized value and gradient
    lam = 0.5
    val_reg = obj.regularized_value(w, lambda_reg=lam)
    expected_val = obj.value(w) + 0.5 * lam * np.sum(w**2)
    assert np.isclose(val_reg, expected_val)

    grad_reg = obj.regularized_gradient(w, lambda_reg=lam)
    expected_grad = obj.gradient(w) + lam * w
    assert np.allclose(grad_reg, expected_grad)


def test_weight_decay_shrinkage_and_effective_parameters():
    """Verify shrinkage formula w_tilde_i = (eta_i / (eta_i + lambda)) * w_star_i and gamma."""
    eta1, eta2 = 0.5, 4.0
    H = np.diag([eta1, eta2])
    w_star = np.array([3.0, 2.0])
    obj = QuadraticObjective(H=H, w_star=w_star)

    lam = 1.0
    w_hat = obj.regularized_optimum(lambda_reg=lam)
    expected_w_hat = np.array([
        (eta1 / (eta1 + lam)) * w_star[0],
        (eta2 / (eta2 + lam)) * w_star[1],
    ])
    assert np.allclose(w_hat, expected_w_hat)

    # Shrinkage along direction 1 (small eigenvalue) must be strictly greater than direction 2
    shrinkage1 = w_hat[0] / w_star[0]  # 0.5 / 1.5 = 1/3
    shrinkage2 = w_hat[1] / w_star[1]  # 4.0 / 5.0 = 0.8
    assert shrinkage1 < shrinkage2

    # Effective number of parameters gamma
    gamma = obj.effective_number_of_parameters(lambda_reg=lam)
    expected_gamma = (eta1 / (eta1 + lam)) + (eta2 / (eta2 + lam))
    assert np.isclose(gamma, expected_gamma)

    # Limits of gamma
    assert np.isclose(obj.effective_number_of_parameters(lambda_reg=0.0), 2.0)
    assert obj.effective_number_of_parameters(lambda_reg=1e6) < 1e-4


def test_mlp_input_transformation_invariance():
    """Verify network mapping invariance under linear transformation of inputs (Eqs 9.8-9.10)."""
    rng = np.random.RandomState(42)
    W1 = rng.randn(6, 1)
    b1 = rng.randn(6, 1)
    W2 = rng.randn(1, 6)
    b2 = rng.randn(1, 1)

    mlp = MLP2Layer(W1=W1, b1=b1, W2=W2, b2=b2)

    a, b = 2.5, -1.2
    mlp_tilde = mlp.transform_inputs(a=a, b=b)

    x = np.linspace(-3, 3, 50)
    x_tilde = a * x + b

    # Under transformed weights and transformed inputs, the outputs must be identical
    y_orig = mlp.forward(x)
    y_trans = mlp_tilde.forward(x_tilde)

    assert np.allclose(y_trans, y_orig, atol=1e-10)


def test_mlp_output_transformation_invariance():
    """Verify network mapping behavior under linear transformation of outputs (Eqs 9.11-9.13)."""
    rng = np.random.RandomState(42)
    W1 = rng.randn(6, 1)
    b1 = rng.randn(6, 1)
    W2 = rng.randn(1, 6)
    b2 = rng.randn(1, 1)

    mlp = MLP2Layer(W1=W1, b1=b1, W2=W2, b2=b2)

    c, d = 3.0, 0.75
    mlp_trans = mlp.transform_outputs(c=c, d=d)

    x = np.linspace(-2, 2, 50)
    y_orig = mlp.forward(x)
    y_trans = mlp_trans.forward(x)

    assert np.allclose(y_trans, c * y_orig + d, atol=1e-10)


def test_consistent_regularizer_scaling():
    """Verify scaling behavior of consistent regularizer (Eq 9.14)."""
    rng = np.random.RandomState(42)
    W1 = rng.randn(4, 1)
    b1 = rng.randn(4, 1)
    W2 = rng.randn(1, 4)
    b2 = rng.randn(1, 1)

    mlp = MLP2Layer(W1=W1, b1=b1, W2=W2, b2=b2)
    a, b = 2.0, 0.5
    mlp_in = mlp.transform_inputs(a=a, b=b)

    # In mlp_in, W1 is scaled by 1/a, so sum(W1^2) is scaled by 1/a^2.
    # Therefore, scaling lambda1 by a^2 preserves the regularizer value:
    lam1, lam2 = 1.5, 2.5
    loss_orig = 0.5 * lam1 * np.sum(mlp.W1**2)
    loss_trans = 0.5 * (lam1 * a**2) * np.sum(mlp_in.W1**2)
    assert np.isclose(loss_orig, loss_trans)


def test_sample_mlp_functions_shapes_and_scales():
    """Verify prior function sampling dimensions, vertical scaling, and horizontal variation."""
    x, y_samples = sample_mlp_functions(
        alpha_1w=1, alpha_1b=1, alpha_2w=1, alpha_2b=1, n_samples=5, n_points=100, seed=42
    )
    assert x.shape == (100,)
    assert y_samples.shape == (5, 100)

    # Vertical scaling with alpha_2w:
    _, y_scaled = sample_mlp_functions(
        alpha_1w=1, alpha_1b=1, alpha_2w=10, alpha_2b=1, n_samples=5, n_points=100, seed=42
    )
    # y_scaled should have ~10x greater amplitude on average
    std_orig = np.std(y_samples)
    std_scaled = np.std(y_scaled)
    assert np.isclose(std_scaled / std_orig, 10.0, rtol=0.2)


def test_generalized_weight_decay_penalty_and_contours():
    """Verify L_q penalty formula and contour radius generation."""
    w = np.array([3.0, -4.0])
    # For q=2: 0.5 * 1.0 * (9 + 16) = 12.5
    assert np.isclose(lq_penalty(w, q=2.0, lambda_reg=1.0), 12.5)
    # For q=1: 0.5 * 1.0 * (3 + 4) = 3.5
    assert np.isclose(lq_penalty(w, q=1.0, lambda_reg=1.0), 3.5)

    # Contour points verification for q=1: |w1| + |w2| = 1
    w1, w2 = lq_contour_points(q=1.0, radius=1.0, n_points=200)
    assert np.allclose(np.abs(w1) + np.abs(w2), 1.0, atol=1e-5)

    # Contour points verification for q=2: w1^2 + w2^2 = 1
    w1_2, w2_2 = lq_contour_points(q=2.0, radius=1.0, n_points=200)
    assert np.allclose(w1_2**2 + w2_2**2, 1.0, atol=1e-5)


def test_sparsity_lasso_vs_ridge():
    """Verify that Lasso (q=1) yields exact sparse solution (w1=0) while Ridge (q=2) yields non-zero."""
    theta = np.deg2rad(45)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    a, b = 1.4, 0.7
    H_inv = R @ np.diag([a**2, b**2]) @ R.T
    H = np.linalg.inv(H_inv)

    g = np.array([0.6, 1.0])
    w_star = np.array([0.0, 1.0]) + H_inv @ g
    eta = 1.0

    # Lasso (q=1)
    w_hat_lasso = solve_constrained_qp(H=H, w_star=w_star, q=1.0, eta=eta, w_init=np.array([0.0, 1.0]))
    assert np.isclose(w_hat_lasso[0], 0.0, atol=1e-5)
    assert np.isclose(w_hat_lasso[1], 1.0, atol=1e-5)

    # Ridge (q=2)
    w_hat_ridge = solve_constrained_qp(H=H, w_star=w_star, q=2.0, eta=eta, w_init=np.array([0.2, 0.95]))
    assert w_hat_ridge[0] > 0.15  # Non-zero first component!
    assert w_hat_ridge[1] > 0.8
    assert np.isclose(np.linalg.norm(w_hat_ridge), 1.0, atol=1e-5)


def test_generate_figures():
    """Verify that Figures 9.3, 9.4, 9.5, and 9.6 generate and save correctly."""
    repo_root = Path(__file__).resolve().parent.parent
    dir_ch9 = repo_root / "9" / "result"
    dir_root = repo_root / "result"

    fig9_3 = generate_figure_9_3()
    fig9_4 = generate_figure_9_4()
    fig9_5 = generate_figure_9_5()
    fig9_6 = generate_figure_9_6()

    assert fig9_3 is not None
    assert fig9_4 is not None
    assert fig9_5 is not None
    assert fig9_6 is not None

    expected_files = [
        "fig_9_3_weight_decay_shrinkage.png",
        "Figure_9_3.png",
        "fig_9_4_prior_network_functions.png",
        "Figure_9_4.png",
        "fig_9_5_lq_regularization_contours.png",
        "Figure_9_5.png",
        "fig_9_6_sparsity_lasso_vs_ridge.png",
        "Figure_9_6.png",
    ]

    for fname in expected_files:
        p_ch9 = dir_ch9 / fname
        p_root = dir_root / fname
        assert p_ch9.exists(), f"Missing {p_ch9}"
        assert p_ch9.stat().st_size > 1000, f"File {p_ch9} is too small"
        assert p_root.exists(), f"Missing {p_root}"
        assert p_root.stat().st_size > 1000, f"File {p_root} is too small"
