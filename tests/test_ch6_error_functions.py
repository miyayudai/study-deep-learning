"""
Tests for Chapter 6 Section 6.4: Error Functions
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts, pp. 194-198.
"""

import os
import pytest
import numpy as np

from common.error_functions import (
    sum_of_squares_loss,
    mean_squared_error,
    gaussian_nll_loss,
    estimate_noise_variance,
    binary_cross_entropy_loss,
    binary_cross_entropy_with_logits,
    multilabel_cross_entropy_loss,
    label_noise_bernoulli,
    softmax,
    multiclass_cross_entropy_loss,
    categorical_cross_entropy_with_logits,
    canonical_preactivation_gradient,
    compare_bce_vs_mse_gradient,
    generate_figure_6_error_functions,
)


class TestRegressionErrors:
    """Test Section 6.4.1 Regression error functions and noise variance estimation."""

    def test_sum_of_squares_scalar(self):
        y = np.array([1.0, 2.0, 3.0])
        t = np.array([1.0, 1.0, 1.0])
        # diff = [0, 1, 2], squared = [0, 1, 4], sum = 5, E = 2.5
        assert sum_of_squares_loss(y, t) == pytest.approx(2.5)

    def test_sum_of_squares_multitarget(self):
        y = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = np.array([[1.0, 1.0], [2.0, 2.0]])
        # diff = [[0, 1], [1, 2]], squared = [[0, 1], [1, 4]], sum = 6, E = 3.0
        assert sum_of_squares_loss(y, t) == pytest.approx(3.0)

    def test_gaussian_nll_loss(self):
        y = np.array([2.0])
        t = np.array([2.0])
        # Perfect fit: sse = 0, nll = 0.5 * ln(sigma^2) + 0.5 * ln(2*pi)
        sigma_sq = 1.0
        expected = 0.5 * np.log(2 * np.pi)
        assert gaussian_nll_loss(y, t, sigma_sq) == pytest.approx(expected)

        with pytest.raises(ValueError):
            gaussian_nll_loss(y, t, sigma_sq=-0.1)

    def test_noise_variance_estimator(self):
        # N=4, K=2, errors are all 2.0 -> squared error = 4.0
        y = np.zeros((4, 2))
        t = np.full((4, 2), 2.0)
        sigma_sq_hat = estimate_noise_variance(y, t)
        assert sigma_sq_hat == pytest.approx(4.0)


class TestBinaryClassificationErrors:
    """Test Section 6.4.2 Binary classification and independent binary error functions."""

    def test_binary_cross_entropy_scalar(self):
        y = np.array([0.8, 0.2])
        t = np.array([1.0, 0.0])
        # -ln(0.8) - ln(0.8) = -2 * ln(0.8)
        expected = -2.0 * np.log(0.8)
        assert binary_cross_entropy_loss(y, t) == pytest.approx(expected)

    def test_bce_with_logits_stability(self):
        # Extreme logits to test numerical stability
        a = np.array([100.0, -100.0])
        t = np.array([1.0, 0.0])
        loss = binary_cross_entropy_with_logits(a, t)
        assert np.isfinite(loss)
        assert loss < 1e-10

    def test_multilabel_cross_entropy(self):
        y = np.array([[0.9, 0.1], [0.8, 0.7]])
        t = np.array([[1.0, 0.0], [1.0, 1.0]])
        loss = multilabel_cross_entropy_loss(y, t)
        assert loss > 0.0

    def test_label_noise_model(self):
        y = np.array([0.8])
        # flip_prob = 0.1 -> p(t=1) = (1 - 0.2) * 0.8 + 0.1 = 0.8 * 0.8 + 0.1 = 0.74
        p_noisy = label_noise_bernoulli(y, flip_prob=0.1)
        assert p_noisy[0] == pytest.approx(0.74)

        with pytest.raises(ValueError):
            label_noise_bernoulli(y, flip_prob=0.6)


class TestMulticlassClassificationErrors:
    """Test Section 6.4.3 Multiclass classification and softmax."""

    def test_softmax_sum_to_one(self):
        a = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
        p = softmax(a)
        assert p.shape == (2, 3)
        np.testing.assert_allclose(np.sum(p, axis=1), [1.0, 1.0])

    def test_softmax_translation_invariance(self):
        """Softmax is invariant to adding a constant to all pre-activations: softmax(a + c) == softmax(a)."""
        a = np.array([1.5, -0.8, 3.2])
        p1 = softmax(a)
        p2 = softmax(a + 100.0)
        np.testing.assert_allclose(p1, p2, atol=1e-12)

    def test_multiclass_cross_entropy(self):
        y = np.array([[0.7, 0.2, 0.1]])
        t = np.array([[1.0, 0.0, 0.0]])
        expected = -np.log(0.7)
        assert multiclass_cross_entropy_loss(y, t) == pytest.approx(expected)

    def test_categorical_cross_entropy_with_logits(self):
        a = np.array([[2.0, 0.0, -1.0]])
        t = np.array([[1.0, 0.0, 0.0]])
        loss_logits = categorical_cross_entropy_with_logits(a, t)
        probs = softmax(a)
        loss_probs = multiclass_cross_entropy_loss(probs, t)
        assert loss_logits == pytest.approx(loss_probs)


class TestCanonicalGradientUnification:
    """Test Eq 6.31: dE / da_k = y_k - t_k across all canonical pairings."""

    def test_regression_canonical_gradient(self):
        # y = a, E = 0.5 (y - t)^2 -> dE/da = y - t
        a = 2.5
        t = 1.0
        y = a
        grad = canonical_preactivation_gradient(y, t)
        assert grad == 1.5

        # Numerical gradient check
        eps = 1e-6
        e_plus = 0.5 * ((a + eps) - t)**2
        e_minus = 0.5 * ((a - eps) - t)**2
        num_grad = (e_plus - e_minus) / (2 * eps)
        assert grad == pytest.approx(num_grad, rel=1e-5)

    def test_binary_canonical_gradient(self):
        # y = sigma(a), E = -[t ln y + (1-t) ln(1-y)] -> dE/da = y - t
        a = 1.2
        t = 1.0
        y = 1.0 / (1.0 + np.exp(-a))
        grad = canonical_preactivation_gradient(y, t)
        assert grad == pytest.approx(y - t)

        eps = 1e-6
        y_plus = 1.0 / (1.0 + np.exp(-(a + eps)))
        y_minus = 1.0 / (1.0 + np.exp(-(a - eps)))
        e_plus = - (t * np.log(y_plus) + (1 - t) * np.log(1 - y_plus))
        e_minus = - (t * np.log(y_minus) + (1 - t) * np.log(1 - y_minus))
        num_grad = (e_plus - e_minus) / (2 * eps)
        assert grad == pytest.approx(num_grad, rel=1e-5)

    def test_multiclass_canonical_gradient(self):
        # y = softmax(a), E = -sum t_k ln y_k -> dE/da_k = y_k - t_k
        a = np.array([1.0, 2.0, -0.5])
        t = np.array([0.0, 1.0, 0.0])
        y = softmax(a)
        grad = canonical_preactivation_gradient(y, t)
        np.testing.assert_allclose(grad, y - t)

        eps = 1e-6
        for k in range(len(a)):
            a_plus = a.copy()
            a_plus[k] += eps
            e_plus = -np.sum(t * np.log(softmax(a_plus)))

            a_minus = a.copy()
            a_minus[k] -= eps
            e_minus = -np.sum(t * np.log(softmax(a_minus)))

            num_grad_k = (e_plus - e_minus) / (2 * eps)
            assert grad[k] == pytest.approx(num_grad_k, rel=1e-5)

    def test_bce_avoids_saturation(self):
        """Cross-entropy gradient remains significant while MSE gradient vanishes for wrong predictions."""
        a_wrong = np.array([-10.0])  # strongly predicting class 0 when true class is 1
        grad_bce, grad_mse = compare_bce_vs_mse_gradient(a_wrong, target=1.0)
        # BCE gradient should be approximately -1.0
        assert abs(grad_bce[0]) > 0.99
        # MSE gradient should vanish to near 0.0
        assert abs(grad_mse[0]) < 1e-4


class TestFigureGeneration:
    """Test Section 6.4 figure generation and persistence."""

    def test_generate_figure(self):
        fig = generate_figure_6_error_functions()
        assert fig is not None
        
        path1 = os.path.join("6", "result", "fig_6_error_functions.png")
        path2 = os.path.join("result", "fig_6_error_functions.png")
        assert os.path.exists(path1)
        assert os.path.exists(path2)
        assert os.path.getsize(path1) > 1000
        assert os.path.getsize(path2) > 1000
