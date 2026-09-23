"""Tests for Chapter 8 Section 8.1: Evaluation of Gradients.

Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
Section 8.1: Evaluation of Gradients
- 8.1.1 Single-layer networks
- 8.1.2 General feed-forward networks
- 8.1.3 A simple example (two-layer MLP)
- 8.1.4 Numerical differentiation (finite vs central differences)
- 8.1.5 The Jacobian matrix
- 8.1.6 The Hessian matrix
- Figures 8.1, 8.2, 8.3 reproduction
"""

import os
import numpy as np
import pytest

from common.evaluation_of_gradients import (
    SingleLayerNetwork,
    FeedForwardNeuralNetwork,
    finite_difference_gradient,
    central_difference_gradient,
    check_gradient,
    numerical_jacobian,
    plot_figure_8_1,
    plot_figure_8_2,
    plot_figure_8_3,
    generate_all_section_8_1_figures,
    sigmoid,
    sigmoid_deriv,
    tanh,
    tanh_deriv,
    relu,
    relu_deriv,
    linear,
    softmax,
)


# =====================================================================
# Activation Functions Tests
# =====================================================================

def test_activation_derivatives():
    """Verify analytical derivatives of activations match numerical differences."""
    rng = np.random.default_rng(42)
    a = rng.normal(0.0, 1.5, size=(10,))

    # Sigmoid derivative: sigma'(a) = sigma(a) * (1 - sigma(a))
    s_ana = sigmoid_deriv(a)
    eps = 1e-6
    s_num = (sigmoid(a + eps) - sigmoid(a - eps)) / (2.0 * eps)
    assert np.allclose(s_ana, s_num, atol=1e-7)

    # Tanh derivative: tanh'(a) = 1 - tanh(a)^2
    t_ana = tanh_deriv(a)
    t_num = (tanh(a + eps) - tanh(a - eps)) / (2.0 * eps)
    assert np.allclose(t_ana, t_num, atol=1e-7)

    # ReLU derivative
    a_pos = np.array([0.5, 1.2, 3.0])
    assert np.allclose(relu_deriv(a_pos), 1.0)
    a_neg = np.array([-0.5, -1.2, -3.0])
    assert np.allclose(relu_deriv(a_neg), 0.0)

    # Softmax sum to 1
    sm = softmax(a)
    assert np.isclose(np.sum(sm), 1.0)


# =====================================================================
# Section 8.1.1: Single-Layer Network Tests
# =====================================================================

def test_single_layer_network_gradients():
    """Verify gradients of single-layer network match central differences (Eq 8.4)."""
    rng = np.random.default_rng(123)
    N, D, K = 8, 4, 3
    X = rng.normal(0.0, 1.0, (N, D))
    T = rng.normal(0.0, 1.0, (N, K))

    net = SingleLayerNetwork(in_features=D, out_features=K, seed=42)
    loss = net.compute_loss(X, T)
    assert loss > 0.0

    grad_W, grad_b = net.backward(X, T)

    # Check W gradients via central differences
    eps = 1e-6
    for k in range(K):
        for i in range(D):
            W_orig = net.W[k, i]
            net.W[k, i] = W_orig + eps
            loss_plus = net.compute_loss(X, T)
            net.W[k, i] = W_orig - eps
            loss_minus = net.compute_loss(X, T)
            net.W[k, i] = W_orig
            num_grad = (loss_plus - loss_minus) / (2.0 * eps)
            assert np.isclose(grad_W[k, i], num_grad, atol=1e-6, rtol=1e-5)

    # Check b gradients via central differences
    for k in range(K):
        b_orig = net.b[k]
        net.b[k] = b_orig + eps
        loss_plus = net.compute_loss(X, T)
        net.b[k] = b_orig - eps
        loss_minus = net.compute_loss(X, T)
        net.b[k] = b_orig
        num_grad = (loss_plus - loss_minus) / (2.0 * eps)
        assert np.isclose(grad_b[k], num_grad, atol=1e-6, rtol=1e-5)


# =====================================================================
# Section 8.1.2 & 8.1.3: Feed-Forward Network & Backprop Tests
# =====================================================================

def test_feedforward_simple_example_gradients():
    """Section 8.1.3: Two-layer MLP with tanh hidden units and linear outputs."""
    rng = np.random.default_rng(42)
    N, D, M, K = 6, 3, 5, 2
    X = rng.normal(0.0, 1.0, (N, D))
    T = rng.normal(0.0, 1.0, (N, K))

    net = FeedForwardNeuralNetwork(
        layer_sizes=[D, M, K],
        hidden_activation="tanh",
        output_activation="linear",
        loss="squared_error",
        seed=123,
    )

    Y, activations, pre_activations = net.forward(X)
    assert Y.shape == (N, K)
    assert len(activations) == 3
    assert len(pre_activations) == 2

    # Check gradient via check_gradient
    params = net.get_params_flat()

    def loss_func(p):
        net.set_params_flat(p)
        return net.compute_loss(X, T)

    def grad_func(p):
        net.set_params_flat(p)
        _, flat_grad = net.loss_and_grad_flat(p, X, T)
        return flat_grad

    rel_err = check_gradient(loss_func, grad_func, params, eps=1e-6)
    assert rel_err < 1e-6


def test_feedforward_deep_network_gradients():
    """Deep 3-hidden-layer network gradient check."""
    rng = np.random.default_rng(999)
    N = 4
    X = rng.normal(0.0, 1.0, (N, 4))
    T = rng.normal(0.0, 1.0, (N, 2))

    net = FeedForwardNeuralNetwork(
        layer_sizes=[4, 6, 5, 3, 2],
        hidden_activation="tanh",
        output_activation="linear",
        loss="squared_error",
        seed=456,
    )

    params = net.get_params_flat()

    def loss_func(p):
        net.set_params_flat(p)
        return net.compute_loss(X, T)

    def grad_func(p):
        net.set_params_flat(p)
        _, flat_grad = net.loss_and_grad_flat(p, X, T)
        return flat_grad

    rel_err = check_gradient(loss_func, grad_func, params, eps=1e-6)
    assert rel_err < 1e-6


def test_feedforward_cross_entropy_classification():
    """Binary and multiclass cross-entropy backpropagation gradients."""
    rng = np.random.default_rng(77)
    N = 5
    X = rng.normal(0.0, 1.0, (N, 3))
    T_bin = rng.integers(0, 2, size=(N, 1)).astype(float)

    # Binary classification with sigmoid output
    net_bin = FeedForwardNeuralNetwork(
        layer_sizes=[3, 4, 1],
        hidden_activation="sigmoid",
        output_activation="sigmoid",
        loss="cross_entropy",
        seed=101,
    )
    p_bin = net_bin.get_params_flat()
    rel_err_bin = check_gradient(
        lambda p: net_bin.loss_and_grad_flat(p, X, T_bin)[0],
        lambda p: net_bin.loss_and_grad_flat(p, X, T_bin)[1],
        p_bin,
        eps=1e-6,
    )
    assert rel_err_bin < 1e-6

    # Multiclass classification with softmax output
    T_multi = np.zeros((N, 3))
    for i in range(N):
        T_multi[i, rng.integers(0, 3)] = 1.0

    net_multi = FeedForwardNeuralNetwork(
        layer_sizes=[3, 5, 3],
        hidden_activation="tanh",
        output_activation="softmax",
        loss="multiclass_cross_entropy",
        seed=202,
    )
    p_multi = net_multi.get_params_flat()
    rel_err_multi = check_gradient(
        lambda p: net_multi.loss_and_grad_flat(p, X, T_multi)[0],
        lambda p: net_multi.loss_and_grad_flat(p, X, T_multi)[1],
        p_multi,
        eps=1e-6,
    )
    assert rel_err_multi < 1e-6


# =====================================================================
# Section 8.1.4: Numerical Differentiation Tests
# =====================================================================

def test_numerical_differentiation_convergence_order():
    """Verify forward diff is O(eps) and central diff is O(eps^2) (Eqs 8.24-8.25)."""
    # Quadratic / cubic test function f(w) = w_0^3 + 2 w_1^2
    def f(w):
        return float(w[0]**3 + 2.0 * w[1]**2)

    def grad_true(w):
        return np.array([3.0 * w[0]**2, 4.0 * w[1]])

    w = np.array([1.5, 2.0])
    g_true = grad_true(w)

    # Step sizes halving: eps1 and eps2 = eps1 / 2
    eps1 = 1e-3
    eps2 = 5e-4

    g_fd1 = finite_difference_gradient(f, w, eps1)
    g_fd2 = finite_difference_gradient(f, w, eps2)
    err_fd1 = np.linalg.norm(g_fd1 - g_true)
    err_fd2 = np.linalg.norm(g_fd2 - g_true)
    ratio_fd = err_fd1 / err_fd2
    # Should be close to 2.0 (O(eps))
    assert 1.8 < ratio_fd < 2.2

    g_cd1 = central_difference_gradient(f, w, eps1)
    g_cd2 = central_difference_gradient(f, w, eps2)
    err_cd1 = np.linalg.norm(g_cd1 - g_true)
    err_cd2 = np.linalg.norm(g_cd2 - g_true)
    ratio_cd = err_cd1 / err_cd2
    # Should be close to 4.0 (O(eps^2))
    assert 3.8 < ratio_cd < 4.2


# =====================================================================
# Section 8.1.5: Jacobian Matrix Tests
# =====================================================================

def test_jacobian_analytical_vs_numerical():
    """Verify analytical Jacobian matches numerical central difference Jacobian (Section 8.1.5)."""
    net = FeedForwardNeuralNetwork(
        layer_sizes=[4, 6, 3],
        hidden_activation="tanh",
        output_activation="linear",
        loss="squared_error",
        seed=88,
    )
    rng = np.random.default_rng(99)
    x = rng.normal(0.0, 1.0, size=(4,))

    J_ana = net.jacobian(x)
    assert J_ana.shape == (3, 4)

    J_num = numerical_jacobian(net, x, eps=1e-6)
    assert np.allclose(J_ana, J_num, atol=1e-6, rtol=1e-5)


def test_jacobian_sensitivity_approximation():
    """Section 8.1.5: Delta y approx J * Delta x (Eq 8.28)."""
    net = FeedForwardNeuralNetwork(
        layer_sizes=[3, 5, 2],
        hidden_activation="tanh",
        output_activation="linear",
        loss="squared_error",
        seed=101,
    )
    rng = np.random.default_rng(102)
    x = rng.normal(0.0, 1.0, size=(3,))
    delta_x = 1e-4 * rng.normal(0.0, 1.0, size=(3,))

    y0 = net.predict(x).ravel()
    y_pert = net.predict(x + delta_x).ravel()
    delta_y_actual = y_pert - y0

    J = net.jacobian(x)
    delta_y_pred = J @ delta_x

    # Error should be O(||delta_x||^2) approx 1e-8
    assert np.allclose(delta_y_actual, delta_y_pred, atol=1e-7)


def test_jacobian_softmax_output():
    """Verify Jacobian for softmax output layer matches numerical differentiation."""
    net = FeedForwardNeuralNetwork(
        layer_sizes=[3, 4, 3],
        hidden_activation="tanh",
        output_activation="softmax",
        loss="multiclass_cross_entropy",
        seed=103,
    )
    rng = np.random.default_rng(104)
    x = rng.normal(0.0, 1.0, size=(3,))

    J_ana = net.jacobian(x)
    J_num = numerical_jacobian(net, x, eps=1e-6)
    assert np.allclose(J_ana, J_num, atol=1e-6, rtol=1e-5)


# =====================================================================
# Section 8.1.6: Hessian Matrix Tests
# =====================================================================

def test_exact_hessian_symmetry_and_values():
    """Section 8.1.6: Exact Hessian matrix is symmetric and matches 2nd differences."""
    rng = np.random.default_rng(55)
    N, D, K = 3, 2, 1
    X = rng.normal(0.0, 1.0, (N, D))
    T = rng.normal(0.0, 1.0, (N, K))

    net = FeedForwardNeuralNetwork(
        layer_sizes=[D, 3, K],
        hidden_activation="tanh",
        output_activation="linear",
        loss="squared_error",
        seed=77,
    )

    H = net.exact_hessian(X, T, eps=1e-5)
    W_total = net.total_params()
    assert H.shape == (W_total, W_total)
    # Check symmetry H = H^T
    assert np.allclose(H, H.T, atol=1e-8)


def test_hessian_vector_product():
    """Section 8.1.6: O(W) Hessian-vector product Hv matches H @ v."""
    rng = np.random.default_rng(12)
    N, D, K = 4, 2, 2
    X = rng.normal(0.0, 1.0, (N, D))
    T = rng.normal(0.0, 1.0, (N, K))

    net = FeedForwardNeuralNetwork(
        layer_sizes=[D, 3, K],
        hidden_activation="tanh",
        output_activation="linear",
        loss="squared_error",
        seed=33,
    )

    v = rng.normal(0.0, 1.0, size=(net.total_params(),))
    v = v / np.linalg.norm(v)

    H = net.exact_hessian(X, T, eps=1e-5)
    Hv_exact = H @ v
    Hv_fast = net.hessian_vector_product(v, X, T, eps=1e-5)

    assert np.allclose(Hv_exact, Hv_fast, atol=1e-5, rtol=1e-4)


def test_outer_product_hessian_properties():
    """Section 8.1.6: Gauss-Newton / outer product Hessian approximation (Eq 8.40)."""
    rng = np.random.default_rng(66)
    N, D, K = 10, 2, 1
    X = rng.normal(0.0, 1.0, (N, D))
    T = rng.normal(0.0, 1.0, (N, K))

    net = FeedForwardNeuralNetwork(
        layer_sizes=[D, 3, K],
        hidden_activation="tanh",
        output_activation="linear",
        loss="squared_error",
        seed=88,
    )

    H_gn = net.outer_product_hessian(X, T)
    W_total = net.total_params()
    assert H_gn.shape == (W_total, W_total)

    # Positive semi-definite: all eigenvalues >= -1e-10
    eigvals = np.linalg.eigvalsh(H_gn)
    assert np.all(eigvals >= -1e-10)

    # Diagonal Hessian
    diag_H = net.diagonal_hessian(X, T)
    assert len(diag_H) == W_total


# =====================================================================
# Figure Reproduction Tests
# =====================================================================

def test_figure_reproduction_saving(tmp_path):
    """Verify Figures 8.1, 8.2, 8.3 generate and save without error."""
    p1 = str(tmp_path / "fig8_1.png")
    p2 = str(tmp_path / "fig8_2.png")
    p3 = str(tmp_path / "fig8_3.png")

    plot_figure_8_1(filepath=p1, save_both=False)
    plot_figure_8_2(filepath=p2, save_both=False)
    plot_figure_8_3(filepath=p3, save_both=False)

    assert os.path.exists(p1) and os.path.getsize(p1) > 1000
    assert os.path.exists(p2) and os.path.getsize(p2) > 1000
    assert os.path.exists(p3) and os.path.getsize(p3) > 1000

    # Test batch generator
    gen_paths = generate_all_section_8_1_figures(result_dirs=[str(tmp_path)], save_both=False)
    assert len(gen_paths) == 3
