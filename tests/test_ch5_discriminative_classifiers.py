"""
Unit tests for Chapter 5 Section 5.4: Discriminative Classifiers.

Tests cover:
- Activation and link functions (sigmoid, logit, softmax, probit, erf)
- Gaussian basis functions (RBF feature mapping)
- Binary Logistic Regression (IRLS, Gradient Descent, convexity, Hessian, regularization)
- Multi-Class Softmax Regression (cross-entropy, gradients, probability normalization)
- Probit Regression (Gaussian CDF link, gradients, classification)
- Canonical Link Functions & Outlier sensitivity
- Section 5.4 Figure reproductions (Figures 5.15, 5.16, 5.17)
"""

import os
from pathlib import Path
import numpy as np
import pytest

from common.discriminative_classifiers import (
    GaussianBasisFunctions,
    LogisticRegression,
    ProbitRegression,
    SoftmaxRegression,
    compare_logistic_and_probit_outliers,
    erf_func,
    generate_all_section_5_4_figures,
    generate_figure_5_15,
    generate_figure_5_16,
    generate_figure_5_17,
    logit,
    probit,
    probit_deriv,
    sigmoid,
    sigmoid_deriv,
    softmax,
    softmax_jacobian,
    verify_canonical_link_property,
)


class TestActivationAndLinkFunctions:
    """Tests for Section 5.4.1 & 5.4.5 mathematical functions."""

    def test_sigmoid_and_logit_inverses(self):
        """Verify logit is the exact mathematical inverse of sigmoid."""
        a = np.linspace(-6.0, 6.0, 50)
        p = sigmoid(a)
        assert np.all(p > 0.0) and np.all(p < 1.0)
        a_recov = logit(p)
        np.testing.assert_allclose(a_recov, a, atol=1e-10)

    def test_sigmoid_derivative(self):
        """Verify d sigma / da = sigma * (1 - sigma) numerically."""
        a = np.array([-2.5, -1.0, 0.0, 1.0, 2.5])
        eps = 1e-7
        num_deriv = (sigmoid(a + eps) - sigmoid(a - eps)) / (2.0 * eps)
        ana_deriv = sigmoid_deriv(a)
        np.testing.assert_allclose(ana_deriv, num_deriv, atol=1e-6)

    def test_softmax_properties(self):
        """Verify softmax probabilities sum to 1 and match Jacobian derivative."""
        rng = np.random.default_rng(42)
        A = rng.normal(0, 2, (10, 4))
        Y = softmax(A, axis=-1)

        assert Y.shape == (10, 4)
        np.testing.assert_allclose(np.sum(Y, axis=-1), np.ones(10), atol=1e-12)
        assert np.all(Y >= 0.0)

        # Test single sample Jacobian: dy_k / da_j = y_k * (I_kj - y_j)
        a_single = A[0]
        y_single = Y[0]
        J = softmax_jacobian(y_single)
        assert J.shape == (4, 4)

        # Numerical Jacobian
        eps = 1e-6
        J_num = np.zeros((4, 4))
        for j in range(4):
            a_plus = a_single.copy()
            a_minus = a_single.copy()
            a_plus[j] += eps
            a_minus[j] -= eps
            J_num[:, j] = (softmax(a_plus) - softmax(a_minus)) / (2.0 * eps)

        np.testing.assert_allclose(J, J_num, atol=1e-5)

    def test_probit_and_erf_relationship(self):
        """Verify probit(a) == 0.5 * (1 + erf(a / sqrt(2)))."""
        a = np.linspace(-3.0, 3.0, 30)
        prob = probit(a)
        expected = 0.5 * (1.0 + erf_func(a / np.sqrt(2.0)))
        np.testing.assert_allclose(prob, expected, atol=1e-12)

        # Verify derivative
        eps = 1e-7
        num_deriv = (probit(a + eps) - probit(a - eps)) / (2.0 * eps)
        ana_deriv = probit_deriv(a)
        np.testing.assert_allclose(ana_deriv, num_deriv, atol=1e-6)


class TestGaussianBasisFunctions:
    """Tests for Section 5.4.2 fixed nonlinear transformations."""

    def test_basis_transformation_dimensions(self):
        centers = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        scales = 0.8
        rbf = GaussianBasisFunctions(centers=centers, scales=scales, include_bias=True)

        X = np.random.randn(20, 2)
        Phi = rbf.transform(X)

        assert Phi.shape == (20, 4)
        np.testing.assert_allclose(Phi[:, 0], np.ones(20))
        assert np.all(Phi[:, 1:] > 0.0) and np.all(Phi[:, 1:] <= 1.0)


class TestBinaryLogisticRegression:
    """Tests for Section 5.4.3 binary logistic regression and IRLS."""

    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.default_rng(42)
        N = 80
        X0 = rng.normal(loc=[-1.2, -1.0], scale=0.4, size=(N // 2, 2))
        X1 = rng.normal(loc=[1.2, 1.0], scale=0.4, size=(N // 2, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * (N // 2) + [1] * (N // 2))
        return X, y

    def test_irls_convergence(self, synthetic_data):
        X, y = synthetic_data
        model = LogisticRegression(reg=1e-3, fit_intercept=True)
        model.fit(X, y, method="irls", max_iter=25)

        assert len(model.loss_history) < 25
        # Loss must decrease monotonically
        assert model.loss_history[-1] < model.loss_history[0]
        # Accuracy should be 100% on separated clusters
        acc = np.mean(model.predict(X) == y)
        assert acc >= 0.95

    def test_gradient_descent_matches_irls(self, synthetic_data):
        X, y = synthetic_data
        model_irls = LogisticRegression(reg=1e-3, fit_intercept=True).fit(X, y, method="irls")
        model_gd = LogisticRegression(reg=1e-3, fit_intercept=True).fit(X, y, method="gd", lr=0.1, max_iter=2000)

        # Both methods should reach the same minimum
        loss_irls = model_irls.loss_history[-1]
        loss_gd = model_gd.loss_history[-1]
        np.testing.assert_allclose(loss_irls, loss_gd, rtol=0.05)

    def test_hessian_positive_definiteness(self, synthetic_data):
        """Hessian H = Phi^T R Phi must be positive definite (strictly convex loss)."""
        X, y = synthetic_data
        model = LogisticRegression(reg=1e-4, fit_intercept=True).fit(X, y, method="irls")
        Phi = model._prepare_features(X)
        H = model.compute_hessian(Phi, model.w)

        eigenvalues = np.linalg.eigvalsh(H)
        assert np.all(eigenvalues > 0.0), f"Hessian has non-positive eigenvalues: {eigenvalues}"

    def test_l2_regularization_bounds_weights(self):
        """Linearly separable data without regularization causes weight divergence; L2 bounds it."""
        rng = np.random.default_rng(123)
        X = np.array([[-3.0], [-2.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])

        model_unreg = LogisticRegression(reg=0.0).fit(X, y, method="gd", lr=0.2, max_iter=200)
        model_reg = LogisticRegression(reg=1.0).fit(X, y, method="gd", lr=0.2, max_iter=200)

        assert np.linalg.norm(model_reg.w) < np.linalg.norm(model_unreg.w)


class TestSoftmaxRegression:
    """Tests for Section 5.4.4 multi-class softmax regression."""

    @pytest.fixture
    def multiclass_data(self):
        rng = np.random.default_rng(42)
        N_per_class = 40
        X0 = rng.normal(loc=[-2.0, -1.0], scale=0.4, size=(N_per_class, 2))
        X1 = rng.normal(loc=[0.0, 2.0], scale=0.4, size=(N_per_class, 2))
        X2 = rng.normal(loc=[2.0, -1.0], scale=0.4, size=(N_per_class, 2))
        X = np.vstack([X0, X1, X2])
        y = np.array([0] * N_per_class + [1] * N_per_class + [2] * N_per_class)
        return X, y

    def test_multiclass_convergence_and_probabilities(self, multiclass_data):
        X, y = multiclass_data
        model = SoftmaxRegression(reg=1e-3, fit_intercept=True)
        model.fit(X, y, lr=0.08, max_iter=800)

        probs = model.predict_proba(X)
        assert probs.shape == (len(X), 3)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(len(X)), atol=1e-12)

        preds = model.predict(X)
        acc = np.mean(preds == y)
        assert acc >= 0.95

    def test_multiclass_gradient(self, multiclass_data):
        """Verify analytical gradient matches numerical gradient."""
        X, y = multiclass_data
        model = SoftmaxRegression(reg=0.05, fit_intercept=True)
        Phi = model._prepare_features(X)
        T = model._to_one_hot(y, 3)
        D = Phi.shape[1]

        W = np.random.default_rng(42).normal(0, 0.5, (D, 3))
        grad_ana = model.compute_gradient(Phi, T, W)

        eps = 1e-6
        grad_num = np.zeros_like(W)
        for i in range(D):
            for j in range(3):
                W_plus = W.copy()
                W_minus = W.copy()
                W_plus[i, j] += eps
                W_minus[i, j] -= eps
                l_plus = model.compute_loss(Phi, T, W_plus)
                l_minus = model.compute_loss(Phi, T, W_minus)
                grad_num[i, j] = (l_plus - l_minus) / (2.0 * eps)

        np.testing.assert_allclose(grad_ana, grad_num, rtol=1e-4, atol=1e-4)


class TestProbitRegression:
    """Tests for Section 5.4.5 probit regression."""

    def test_probit_training_and_predictions(self):
        rng = np.random.default_rng(42)
        X0 = rng.normal(loc=[-1.5, 0.0], scale=0.5, size=(40, 2))
        X1 = rng.normal(loc=[1.5, 0.0], scale=0.5, size=(40, 2))
        X = np.vstack([X0, X1])
        y = np.array([0] * 40 + [1] * 40)

        model = ProbitRegression(reg=1e-3, fit_intercept=True)
        model.fit(X, y, lr=0.05, max_iter=400)

        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        assert model.score(X, y) >= 0.95

    def test_probit_gradient(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (20, 2))
        y = np.array([0] * 10 + [1] * 10)
        model = ProbitRegression(reg=0.02, fit_intercept=True)
        Phi = model._prepare_features(X)

        w = rng.normal(0, 0.5, Phi.shape[1])
        grad_ana = model.compute_gradient(Phi, y, w)

        eps = 1e-6
        grad_num = np.zeros_like(w)
        for i in range(len(w)):
            w_plus = w.copy()
            w_minus = w.copy()
            w_plus[i] += eps
            w_minus[i] -= eps
            l_plus = model.compute_loss(Phi, y, w_plus)
            l_minus = model.compute_loss(Phi, y, w_minus)
            grad_num[i] = (l_plus - l_minus) / (2.0 * eps)

        np.testing.assert_allclose(grad_ana, grad_num, rtol=1e-4, atol=1e-4)


class TestCanonicalLinksAndOutliers:
    """Tests for Section 5.4.6 canonical links and outlier sensitivity."""

    def test_canonical_link_identities(self):
        for dist in ['bernoulli', 'gaussian', 'poisson']:
            res = verify_canonical_link_property(distribution=dist)
            assert res['identity_holds'] is True
            assert res['f_prime_psi_prime_error'] < 1e-10

    def test_outlier_sensitivity_comparison(self):
        res = compare_logistic_and_probit_outliers()
        assert 'logistic_weight_shift' in res
        assert 'probit_weight_shift' in res
        assert res['logistic_weight_shift'] >= 0.0
        assert res['probit_weight_shift'] >= 0.0


class TestFiguresGeneration:
    """Tests for reproduction of Figures 5.15, 5.16, and 5.17."""

    def test_generate_all_section_5_4_figures(self, tmp_path):
        out_dir = str(tmp_path / "result")
        figs = generate_all_section_5_4_figures(result_dirs=[out_dir])

        assert len(figs) == 3
        for k in ['fig_5_15', 'fig_5_16', 'fig_5_17']:
            assert k in figs

        assert os.path.exists(os.path.join(out_dir, "fig_5_15_nonlinear_basis_classification.png"))
        assert os.path.exists(os.path.join(out_dir, "fig_5_16_single_layer_network.png"))
        assert os.path.exists(os.path.join(out_dir, "fig_5_17_probit_threshold_model.png"))
