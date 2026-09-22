"""
Unit tests for Section 5.4: Discriminative Classifiers.
Bishop & Bishop (2024), Chapter 5, Section 5.4.
"""
import os
import pytest
import numpy as np
from scipy import special
from scipy.stats import norm

from common.discriminative_classifiers import (
    sigmoid,
    sigmoid_deriv,
    logit,
    softmax,
    softmax_jacobian,
    erf_func,
    probit,
    probit_deriv,
    GaussianBasisFunctions,
    LogisticRegression,
    SoftmaxRegression,
    ProbitRegression,
    generate_figure_5_15,
    generate_figure_5_16,
    generate_figure_5_17,
    generate_all_section_5_4_figures,
)


class TestActivationAndLinkFunctions:
    """Tests for activation and link functions in Section 5.4.1, 5.4.3, 5.4.4, 5.4.5."""

    def test_sigmoid_and_derivative(self):
        a = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        sig = sigmoid(a)
        sig_d = sigmoid_deriv(a)
        # Eq (5.72): d sigma / da = sigma(1 - sigma)
        assert np.allclose(sig_d, sig * (1.0 - sig))

        # Numerical gradient verification
        eps = 1e-6
        numerical_grad = (sigmoid(a + eps) - sigmoid(a - eps)) / (2.0 * eps)
        assert np.allclose(sig_d, numerical_grad, atol=1e-5)

    def test_logit_is_inverse_sigmoid(self):
        p = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        a = logit(p)
        recovered_p = sigmoid(a)
        assert np.allclose(p, recovered_p)

    def test_softmax_and_jacobian(self):
        a = np.array([1.2, -0.5, 2.3])
        y = softmax(a)
        assert np.isclose(np.sum(y), 1.0)
        assert np.all(y > 0)

        # Eq (5.78): dy_k / da_j = y_k * (I_kj - y_j)
        J = softmax_jacobian(y)
        K = len(a)
        for k in range(K):
            for j in range(K):
                expected = y[k] * ((1.0 if k == j else 0.0) - y[j])
                assert np.isclose(J[k, j], expected)

        # Numerical Jacobian verification
        eps = 1e-6
        J_num = np.zeros((K, K))
        for j in range(K):
            a_plus = a.copy()
            a_minus = a.copy()
            a_plus[j] += eps
            a_minus[j] -= eps
            J_num[:, j] = (softmax(a_plus) - softmax(a_minus)) / (2.0 * eps)
        assert np.allclose(J, J_num, atol=1e-5)

    def test_probit_and_erf(self):
        a = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        prb = probit(a)
        # Eq (5.88): Phi(a) = 0.5 * (1 + erf(a / sqrt(2)))
        expected = 0.5 * (1.0 + special.erf(a / np.sqrt(2.0)))
        assert np.allclose(prb, expected)

        # Probit derivative equals standard normal PDF: d Phi / da = N(a | 0, 1)
        p_d = probit_deriv(a)
        expected_pdf = norm.pdf(a)
        assert np.allclose(p_d, expected_pdf)


class TestBasisFunctions:
    """Tests for nonlinear fixed basis functions (Section 5.4.2)."""

    def test_gaussian_basis_functions(self):
        centers = np.array([[-1.0, 0.0], [1.0, 0.0]])
        rbf = GaussianBasisFunctions(centers=centers, scales=1.0, include_bias=True)
        assert rbf.num_features == 3  # 1 bias + 2 RBF

        X = np.array([[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
        Phi = rbf.transform(X)
        assert Phi.shape == (3, 3)

        # First column must be bias 1
        assert np.allclose(Phi[:, 0], 1.0)
        # At X[1] = [-1, 0] (center 0), Phi[1, 1] should be exp(0) = 1.0
        assert np.isclose(Phi[1, 1], 1.0)
        # At X[2] = [1, 0] (center 1), Phi[2, 2] should be exp(0) = 1.0
        assert np.isclose(Phi[2, 2], 1.0)


class TestLogisticRegression:
    """Tests for binary logistic regression with IRLS and GD (Section 5.4.3)."""

    def test_irls_and_gd_convergence(self):
        np.random.seed(42)
        N = 100
        # Linearly separable clusters with a little noise
        X1 = np.random.randn(N // 2, 2) * 0.3 + np.array([1.5, 1.5])
        X2 = np.random.randn(N // 2, 2) * 0.3 + np.array([-1.5, -1.5])
        X = np.vstack([X1, X2])
        t = np.array([1] * (N // 2) + [0] * (N // 2))

        # Fit with IRLS
        clf_irls = LogisticRegression(reg=0.1, fit_intercept=True)
        clf_irls.fit(X, t, method="irls", max_iter=20)
        preds_irls = clf_irls.predict(X)
        acc_irls = np.mean(preds_irls == t)
        assert acc_irls >= 0.95

        # Fit with Gradient Descent
        clf_gd = LogisticRegression(reg=0.1, fit_intercept=True)
        clf_gd.fit(X, t, method="gd", lr=0.1, max_iter=200)
        preds_gd = clf_gd.predict(X)
        acc_gd = np.mean(preds_gd == t)
        assert acc_gd >= 0.95

        # Both methods find consistent solutions
        assert np.allclose(clf_irls.w, clf_gd.w, atol=0.2)

    def test_cross_entropy_gradient(self):
        np.random.seed(42)
        X = np.random.randn(10, 3)
        t = np.random.randint(0, 2, size=10)
        clf = LogisticRegression(reg=0.0, fit_intercept=True)
        Phi = clf._prepare_features(X)
        w = np.random.randn(Phi.shape[1])

        # Analytical gradient: Eq (5.75) grad = Phi^T (y - t)
        grad_analytical = clf.compute_gradient(Phi, t, w)

        # Numerical gradient
        eps = 1e-6
        grad_numerical = np.zeros_like(w)
        for i in range(len(w)):
            w_plus = w.copy()
            w_minus = w.copy()
            w_plus[i] += eps
            w_minus[i] -= eps
            l_plus = clf.compute_loss(Phi, t, w_plus)
            l_minus = clf.compute_loss(Phi, t, w_minus)
            grad_numerical[i] = (l_plus - l_minus) / (2.0 * eps)

        assert np.allclose(grad_analytical, grad_numerical, atol=1e-5)

    def test_hessian_positive_semi_definite(self):
        np.random.seed(42)
        X = np.random.randn(15, 3)
        clf = LogisticRegression(reg=0.0, fit_intercept=True)
        Phi = clf._prepare_features(X)
        w = np.random.randn(Phi.shape[1])
        H = clf.compute_hessian(Phi, w)
        eigenvalues = np.linalg.eigvalsh(H)
        assert np.all(eigenvalues >= -1e-10), "Hessian must be positive semi-definite (convexity)"


class TestSoftmaxRegression:
    """Tests for multi-class logistic regression (Section 5.4.4)."""

    def test_softmax_regression_training_and_gradient(self):
        np.random.seed(42)
        N_per_class = 30
        X0 = np.random.randn(N_per_class, 2) * 0.3 + np.array([-2.0, 0.0])
        X1 = np.random.randn(N_per_class, 2) * 0.3 + np.array([2.0, 0.0])
        X2 = np.random.randn(N_per_class, 2) * 0.3 + np.array([0.0, 2.5])
        X = np.vstack([X0, X1, X2])
        t = np.array([0] * N_per_class + [1] * N_per_class + [2] * N_per_class)

        clf = SoftmaxRegression(reg=0.01, fit_intercept=True)
        clf.fit(X, t, lr=0.1, max_iter=300)

        preds = clf.predict(X)
        acc = np.mean(preds == t)
        assert acc >= 0.95

        # Check probability outputs sum to 1
        probs = clf.predict_proba(X)
        assert np.allclose(np.sum(probs, axis=1), 1.0)

        # Test gradient formula Eq (5.81), (5.82): grad_W = Phi^T (Y - T)
        Phi = clf._prepare_features(X)
        T = clf._to_one_hot(t, 3)
        grad_analytical = clf.compute_gradient(Phi, T, clf.W)

        # Numerical gradient w.r.t parameter matrix W
        eps = 1e-6
        grad_num = np.zeros_like(clf.W)
        for i in range(clf.W.shape[0]):
            for j in range(clf.W.shape[1]):
                W_p = clf.W.copy()
                W_m = clf.W.copy()
                W_p[i, j] += eps
                W_m[i, j] -= eps
                l_p = clf.compute_loss(Phi, T, W_p)
                l_m = clf.compute_loss(Phi, T, W_m)
                grad_num[i, j] = (l_p - l_m) / (2.0 * eps)

        assert np.allclose(grad_analytical, grad_num, atol=1e-4)


class TestProbitRegression:
    """Tests for probit regression (Section 5.4.5)."""

    def test_probit_regression_training(self):
        np.random.seed(42)
        N = 80
        X1 = np.random.randn(N // 2, 2) * 0.4 + np.array([1.2, 1.2])
        X2 = np.random.randn(N // 2, 2) * 0.4 + np.array([-1.2, -1.2])
        X = np.vstack([X1, X2])
        t = np.array([1] * (N // 2) + [0] * (N // 2))

        clf = ProbitRegression(reg=0.05, fit_intercept=True)
        clf.fit(X, t, lr=0.08, max_iter=250)

        preds = clf.predict(X)
        acc = np.mean(preds == t)
        assert acc >= 0.90

        probs = clf.predict_proba(X)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


class TestFiguresGeneration:
    """Tests for reproduction of Figures 5.15, 5.16, 5.17."""

    def test_all_section_5_4_figures(self, tmp_path):
        results = generate_all_section_5_4_figures(result_dirs=[str(tmp_path)])
        assert "fig_5_15" in results
        assert "fig_5_16" in results
        assert "fig_5_17" in results
        for k in ["fig_5_15", "fig_5_16", "fig_5_17"]:
            assert hasattr(results[k], "savefig")
