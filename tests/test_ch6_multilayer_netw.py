"""
Unit tests for Chapter 6 Section 6.2: Multilayer Networks.
Bishop & Bishop (2024), pp. 180-186.

Covers:
- Section 6.2.1: Parameter matrices, parameter counting, augmented matrix form
- Section 6.2.2: Universal approximation capability on 4 target functions
- Section 6.2.3: Hidden unit activation functions, derivatives, and vanishing gradients
- Section 6.2.4: Weight-space symmetries (2^M sign-flip, M! permutations, M! * 2^M total)
- End-to-end training and gradient checking
- Figures 6.9 through 6.12 reproduction verification
"""

import math
import os
import numpy as np
import pytest

from common.multilayer_networks import (
    TwoLayerMLP,
    absolute_act,
    absolute_deriv,
    apply_hidden_unit_permutation,
    apply_hidden_unit_sign_flip,
    compute_weight_space_symmetries,
    generate_all_section_6_2_figures,
    generate_approximation_data,
    generate_figure_6_10,
    generate_figure_6_11,
    generate_figure_6_12,
    generate_figure_6_9,
    hard_tanh,
    hard_tanh_deriv,
    leaky_relu,
    leaky_relu_deriv,
    relu,
    relu_deriv,
    sigmoid,
    sigmoid_deriv,
    softplus,
    softplus_deriv,
    tanh_act,
    tanh_deriv,
    verify_weight_space_symmetry,
)


class TestParameterMatrices:
    """Tests for Section 6.2.1: Parameter matrices & forward propagation."""

    def test_parameter_count(self):
        """Verify formula (D + 1) * M + (M + 1) * K for various architectures."""
        # D=1, M=3, K=1: (1+1)*3 + (3+1)*1 = 6 + 4 = 10
        mlp1 = TwoLayerMLP(n_in=1, n_hidden=3, n_out=1)
        assert mlp1.count_parameters() == 10

        # D=2, M=2, K=1: (2+1)*2 + (2+1)*1 = 6 + 3 = 9
        mlp2 = TwoLayerMLP(n_in=2, n_hidden=2, n_out=1)
        assert mlp2.count_parameters() == 9

        # D=10, M=20, K=5: (10+1)*20 + (20+1)*5 = 220 + 105 = 325
        mlp3 = TwoLayerMLP(n_in=10, n_hidden=20, n_out=5)
        assert mlp3.count_parameters() == 325

        with pytest.raises(ValueError):
            TwoLayerMLP(n_in=0, n_hidden=5, n_out=1)

    def test_augmented_matrix_equivalence(self):
        """Verify Eq (6.10) - (6.12) vector form equals explicit sum with biases."""
        rng = np.random.default_rng(42)
        D, M, K = 3, 4, 2
        mlp = TwoLayerMLP(n_in=D, n_hidden=M, n_out=K, hidden_activation="tanh", output_activation="linear")
        mlp.init_weights(random_state=42)

        X = rng.normal(0, 1, (10, D))
        y_vec, (a1_vec, z1_vec, a2_vec) = mlp.forward(X)

        # Explicit element-wise computation matching Eq (6.7) - (6.9)
        W1_aug, W2_aug = mlp.get_augmented_matrices()
        assert W1_aug.shape == (M, D + 1)
        assert W2_aug.shape == (K, M + 1)

        y_manual = np.zeros((len(X), K))
        for n in range(len(X)):
            x_aug = np.concatenate([[1.0], X[n]])
            # a_j = sum_{i=0}^D w_{ji}^{(1)} x_i
            a1_manual = W1_aug @ x_aug
            z1_manual = np.tanh(a1_manual)
            # z_aug with z0 = 1
            z_aug = np.concatenate([[1.0], z1_manual])
            # a_k = sum_{j=0}^M w_{kj}^{(2)} z_j
            a2_manual = W2_aug @ z_aug
            y_manual[n] = a2_manual

        np.testing.assert_allclose(y_vec, y_manual, atol=1e-12)

    def test_linear_bottleneck_rank_reduction(self):
        """Verify Section 6.2.3: Linear hidden units with bottleneck have rank <= M."""
        rng = np.random.default_rng(101)
        D, M, K = 8, 3, 6  # M < min(D, K)
        mlp = TwoLayerMLP(n_in=D, n_hidden=M, n_out=K, hidden_activation="linear", output_activation="linear")
        mlp.init_weights(random_state=101)

        # Total linear transformation matrix from inputs to outputs: W2 @ W1
        eff_matrix = mlp.W2 @ mlp.W1  # shape (K, D) = (6, 8)
        rank = np.linalg.matrix_rank(eff_matrix)
        assert rank <= M, f"Effective linear matrix rank {rank} exceeds bottleneck width {M}"


class TestActivationFunctions:
    """Tests for Section 6.2.3: Activation functions & properties."""

    def test_tanh_properties(self):
        """Verify tanh is odd and bounded in (-1, 1), with derivative 1 - tanh^2."""
        a = np.linspace(-3.0, 3.0, 100)
        y = tanh_act(a)
        # Bounded
        assert np.all(y > -1.0) and np.all(y < 1.0)
        # Odd function: tanh(-a) = -tanh(a)
        np.testing.assert_allclose(tanh_act(-a), -y, atol=1e-14)
        # Derivative
        dy = tanh_deriv(a)
        dy_num = (tanh_act(a + 1e-6) - tanh_act(a - 1e-6)) / 2e-6
        np.testing.assert_allclose(dy, dy_num, atol=1e-5)

    def test_hard_tanh_properties(self):
        """Verify hard tanh is clipped in [-1, 1] with derivative 1 on (-1, 1)."""
        a = np.array([-2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5])
        expected = np.array([-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0])
        np.testing.assert_allclose(hard_tanh(a), expected)

        dy = hard_tanh_deriv(np.array([-1.5, -0.5, 0.0, 0.5, 1.5]))
        np.testing.assert_allclose(dy, [0.0, 1.0, 1.0, 1.0, 0.0])

    def test_softplus_properties(self):
        """Verify softplus ln(1+e^a) is smooth ReLU and has derivative equal to sigmoid."""
        a = np.linspace(-4.0, 4.0, 100)
        y = softplus(a)
        assert np.all(y > 0.0)
        # For large a, softplus(a) ~ a
        assert np.isclose(softplus(np.array([50.0]))[0], 50.0, atol=1e-5)
        # Derivative is sigmoid
        dy = softplus_deriv(a)
        np.testing.assert_allclose(dy, sigmoid(a), atol=1e-12)

    def test_relu_and_leaky_relu_properties(self):
        """Verify ReLU and Leaky ReLU piecewise linear slopes."""
        a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        # ReLU
        np.testing.assert_allclose(relu(a), [0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(relu_deriv(a), [0.0, 0.0, 0.0, 1.0, 1.0])

        # Leaky ReLU with alpha = 0.2
        np.testing.assert_allclose(leaky_relu(a, alpha=0.2), [-0.4, -0.2, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(leaky_relu_deriv(a, alpha=0.2), [0.2, 0.2, 0.2, 1.0, 1.0])

    def test_absolute_activation_properties(self):
        """Verify absolute activation h(a) = |a| and derivative sign(a)."""
        a = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
        np.testing.assert_allclose(absolute_act(a), [3.0, 1.5, 0.0, 1.5, 3.0])
        np.testing.assert_allclose(absolute_deriv(a), [-1.0, -1.0, 0.0, 1.0, 1.0])

    def test_vanishing_gradients_contrast(self):
        """Verify Section 6.2.3: sigmoid/tanh gradients vanish, while ReLU/softplus do not."""
        large_a = np.array([10.0, 20.0])
        # Sigmoid & tanh gradients vanish exponentially
        assert np.all(sigmoid_deriv(large_a) < 1e-4)
        assert np.all(tanh_deriv(large_a) < 1e-7)

        # ReLU & softplus gradients remain ~ 1.0
        assert np.all(relu_deriv(large_a) == 1.0)
        assert np.all(np.isclose(softplus_deriv(large_a), 1.0, atol=1e-4))


class TestUniversalApproximation:
    """Tests for Section 6.2.2: Universal approximation capability."""

    @pytest.mark.parametrize("func_name", ["x^2", "sin", "abs", "heaviside"])
    def test_target_function_data_generation(self, func_name):
        """Verify 50 data points generated over (-1, 1)."""
        X, y = generate_approximation_data(func_name, n_samples=50, seed=42)
        assert X.shape == (50, 1)
        assert y.shape == (50, 1)
        assert np.all(X >= -1.0) and np.all(X <= 1.0)

    def test_smooth_function_fit(self):
        """Verify 2-layer network with 3 hidden tanh units fits x^2 with small MSE."""
        X, y = generate_approximation_data("x^2", n_samples=50, seed=42)
        mlp = TwoLayerMLP(1, 3, 1, hidden_activation="tanh", output_activation="linear")
        mlp.fit(X, y, max_iter=600, tol=1e-7, n_restarts=10, random_state=42)

        preds = mlp.predict(X)
        mse = float(np.mean((preds - y)**2))
        assert mse < 1e-4, f"MSE {mse} on x^2 is too high"

    def test_step_function_fit(self):
        """Verify 2-layer network with 3 hidden tanh units can model Heaviside step."""
        X, y = generate_approximation_data("heaviside", n_samples=50, seed=42)
        mlp = TwoLayerMLP(1, 3, 1, hidden_activation="tanh", output_activation="linear")
        # Direct sharp step initialization
        mlp.W1 = np.array([[40.0], [0.1], [-0.1]])
        mlp.b1 = np.array([0.0, 5.0, -5.0])
        mlp.W2 = np.array([[0.5, 0.0, 0.0]])
        mlp.b2 = np.array([0.5])
        mlp.fit(X, y, max_iter=100, tol=1e-7)

        preds = mlp.predict(X)
        mse = float(np.mean((preds - y)**2))
        assert mse < 0.01, f"MSE {mse} on Heaviside step is too high"


class TestWeightSpaceSymmetries:
    """Tests for Section 6.2.4: Weight-space symmetries."""

    def test_symmetry_factor_formula(self):
        """Verify M! * 2^M total symmetries for M hidden units."""
        assert compute_weight_space_symmetries(1, is_odd_activation=True) == 1 * 2 == 2
        assert compute_weight_space_symmetries(2, is_odd_activation=True) == 2 * 4 == 8
        assert compute_weight_space_symmetries(3, is_odd_activation=True) == 6 * 8 == 48
        assert compute_weight_space_symmetries(4, is_odd_activation=True) == 24 * 16 == 384

        # Non-odd activation (only permutations)
        assert compute_weight_space_symmetries(3, is_odd_activation=False) == 6

        # Multi-layer network
        assert compute_weight_space_symmetries([3, 4], is_odd_activation=True) == 48 * 384 == 18432

        with pytest.raises(ValueError):
            compute_weight_space_symmetries(0)

    def test_sign_flip_functional_invariance(self):
        """Verify flipping weights into and out of hidden unit leaves mapping identical."""
        rng = np.random.default_rng(77)
        D, M, K = 3, 4, 2
        mlp = TwoLayerMLP(D, M, K, hidden_activation="tanh", output_activation="linear")
        mlp.init_weights(random_state=77)

        X = rng.normal(0, 1, (20, D))
        verification = verify_weight_space_symmetry(mlp, X)
        assert verification["sign_flip_invariant"] is True
        assert verification["sign_flip_max_diff"] < 1e-12

    def test_permutation_functional_invariance(self):
        """Verify permuting hidden units leaves mapping identical."""
        rng = np.random.default_rng(88)
        D, M, K = 2, 3, 1
        mlp = TwoLayerMLP(D, M, K, hidden_activation="tanh", output_activation="sigmoid")
        mlp.init_weights(random_state=88)

        X = rng.normal(0, 1, (20, D))
        verification = verify_weight_space_symmetry(mlp, X)
        assert verification["permutation_invariant"] is True
        assert verification["permutation_diff"] < 1e-12

    def test_relu_sign_flip_is_not_invariant(self):
        """Verify that sign flip does NOT preserve mapping for non-odd activation (ReLU)."""
        rng = np.random.default_rng(99)
        D, M, K = 2, 3, 1
        mlp = TwoLayerMLP(D, M, K, hidden_activation="relu", output_activation="linear")
        mlp.init_weights(random_state=99)

        X = rng.normal(0, 1, (20, D))
        y_orig = mlp.predict(X)
        mlp_flipped = apply_hidden_unit_sign_flip(mlp, 0)
        y_flipped = mlp_flipped.predict(X)

        diff = float(np.max(np.abs(y_orig - y_flipped)))
        assert diff > 1e-3, "ReLU should not be invariant under sign flip"


class TestGradientCheckingAndTraining:
    """Tests for analytical backpropagation gradients."""

    def test_gradient_checking(self):
        """Verify analytical gradients match finite-difference numerical gradients."""
        rng = np.random.default_rng(123)
        D, M, K = 2, 3, 1
        mlp = TwoLayerMLP(D, M, K, hidden_activation="tanh", output_activation="linear")
        mlp.init_weights(random_state=123)

        X = rng.normal(0, 1, (5, D))
        t = rng.normal(0, 1, (5, K))

        loss, grad_analytic = mlp.compute_loss_and_gradients(X, t)
        p0 = mlp.get_params_flat()
        grad_num = np.zeros_like(p0)
        eps = 1e-6

        for i in range(len(p0)):
            p_plus = p0.copy()
            p_plus[i] += eps
            mlp.set_params_flat(p_plus)
            l_plus, _ = mlp.compute_loss_and_gradients(X, t)

            p_minus = p0.copy()
            p_minus[i] -= eps
            mlp.set_params_flat(p_minus)
            l_minus, _ = mlp.compute_loss_and_gradients(X, t)

            grad_num[i] = (l_plus - l_minus) / (2.0 * eps)

        mlp.set_params_flat(p0)
        rel_diff = np.max(np.abs(grad_analytic - grad_num) / (np.abs(grad_num) + 1e-6))
        assert rel_diff < 1e-4, f"Relative gradient error {rel_diff} too large"


class TestFigureGeneration:
    """Tests for Figure 6.9 through 6.12 generation."""

    def test_all_section_6_2_figures(self):
        """Verify Figures 6.9, 6.10, 6.11, 6.12 are generated and saved to both result directories."""
        figs = generate_all_section_6_2_figures()
        assert len(figs) == 4
        assert "fig_6_9" in figs
        assert "fig_6_10" in figs
        assert "fig_6_11" in figs
        assert "fig_6_12" in figs

        expected_files = [
            "fig_6_9_two_layer_network.png",
            "fig_6_10_universal_approximation.png",
            "fig_6_11_classification_hidden_units.png",
            "fig_6_12_activation_functions.png",
        ]

        for fname in expected_files:
            p_ch6 = os.path.join("6", "result", fname)
            p_root = os.path.join("result", fname)
            assert os.path.exists(p_ch6), f"Missing {p_ch6}"
            assert os.path.exists(p_root), f"Missing {p_root}"
            assert os.path.getsize(p_ch6) > 1000, f"File {p_ch6} too small"
            assert os.path.getsize(p_root) > 1000, f"File {p_root} too small"
