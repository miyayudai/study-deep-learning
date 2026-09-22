"""
Unit tests for Chapter 4 Exercises 4.1 through 4.12.
Validates all proofs, numerical assertions, and mathematical identities.
"""
import pytest
import numpy as np
import scipy.linalg as la
import scipy.special as special
import scipy.integrate as integrate
import scipy.optimize as opt


class TestExercise4_1_PolynomialRegressionNormalEquations:
    """Exercise 4.1: Normal equations for polynomial regression."""

    def test_normal_equations_equivalence(self):
        rng = np.random.default_rng(42)
        N, M = 25, 3
        x = rng.uniform(-1, 1, size=N)
        t = 2.0 * x**2 - 1.5 * x + 0.5 + rng.normal(0, 0.1, size=N)

        # Build A and T directly from Eq 4.54
        A_direct = np.zeros((M + 1, M + 1))
        T_direct = np.zeros(M + 1)
        for i in range(M + 1):
            T_direct[i] = np.sum((x ** i) * t)
            for j in range(M + 1):
                A_direct[i, j] = np.sum(x ** (i + j))

        Phi = np.column_stack([x ** j for j in range(M + 1)])
        A_matrix = Phi.T @ Phi
        T_matrix = Phi.T @ t

        np.testing.assert_allclose(A_direct, A_matrix, atol=1e-12)
        np.testing.assert_allclose(T_direct, T_matrix, atol=1e-12)

        w_direct = la.solve(A_direct, T_direct)
        w_pinv = la.pinv(Phi) @ t
        np.testing.assert_allclose(w_direct, w_pinv, atol=1e-10)


class TestExercise4_2_RegularizedPolynomialRegression:
    """Exercise 4.2: Regularized polynomial normal equations."""

    def test_regularized_system(self):
        rng = np.random.default_rng(42)
        N, M = 20, 4
        lam = 0.7
        x = rng.uniform(-1, 1, size=N)
        t = np.sin(x) + rng.normal(0, 0.1, size=N)

        Phi = np.column_stack([x ** j for j in range(M + 1)])
        A = Phi.T @ Phi
        T = Phi.T @ t

        A_reg = A + lam * np.eye(M + 1)
        w_reg = la.solve(A_reg, T)

        # Optimize regularized error directly
        res = opt.minimize(
            lambda w: 0.5 * np.sum((Phi @ w - t) ** 2) + 0.5 * lam * np.sum(w ** 2),
            np.zeros(M + 1),
            method='BFGS'
        )
        np.testing.assert_allclose(w_reg, res.x, atol=1e-5)


class TestExercise4_3_TanhSigmoidEquivalence:
    """Exercise 4.3: Equivalence between tanh and logistic sigmoid networks."""

    def test_tanh_identity(self):
        a = np.linspace(-5, 5, 200)
        tanh_a = np.tanh(a)
        sig_2a = 1.0 / (1.0 + np.exp(-2.0 * a))
        np.testing.assert_allclose(tanh_a, 2.0 * sig_2a - 1.0, atol=1e-12)

    def test_parameter_mapping(self):
        M = 4
        s = 0.25
        centers = np.linspace(0.1, 0.9, M)
        w = np.array([0.7, 1.5, -0.4, 2.2, -1.1])  # w0 and w1..wM

        # Converted parameters
        u0 = w[0] + 0.5 * np.sum(w[1:])
        uj = 0.5 * w[1:]
        u = np.hstack([[u0], uj])

        x = np.linspace(0, 1, 100)
        sig_mat = 1.0 / (1.0 + np.exp(- (x[:, None] - centers[None, :]) / s))
        y_w = w[0] + sig_mat @ w[1:]

        tanh_mat = np.tanh((x[:, None] - centers[None, :]) / (2.0 * s))
        y_u = u[0] + tanh_mat @ u[1:]

        np.testing.assert_allclose(y_w, y_u, atol=1e-12)


class TestExercise4_4_OrthogonalProjector:
    """Exercise 4.4: Orthogonal projection matrix properties."""

    def test_projector_properties(self):
        rng = np.random.default_rng(42)
        N, M = 15, 3
        Phi = rng.normal(size=(N, M))
        t = rng.normal(size=N)

        P = Phi @ la.inv(Phi.T @ Phi) @ Phi.T

        # Idempotence: P^2 = P
        np.testing.assert_allclose(P @ P, P, atol=1e-12)
        # Symmetry: P^T = P
        np.testing.assert_allclose(P.T, P, atol=1e-12)
        # Residual orthogonality: Phi^T (I - P) t = 0
        residual = (np.eye(N) - P) @ t
        np.testing.assert_allclose(Phi.T @ residual, np.zeros(M), atol=1e-12)


class TestExercise4_5_WeightedLeastSquares:
    """Exercise 4.5: Weighted least squares closed form."""

    def test_wls_solution(self):
        rng = np.random.default_rng(42)
        N, M = 20, 3
        Phi = rng.normal(size=(N, M))
        t = rng.normal(size=N)
        r = rng.uniform(0.2, 3.0, size=N)
        R = np.diag(r)

        w_star = la.solve(Phi.T @ R @ Phi, Phi.T @ R @ t)

        res = opt.minimize(
            lambda w: 0.5 * np.sum(r * ((t - Phi @ w) ** 2)),
            np.zeros(M),
            method='BFGS'
        )
        np.testing.assert_allclose(w_star, res.x, atol=1e-5)


class TestExercise4_6_RidgeRegressionGradient:
    """Exercise 4.6: Regularized least squares gradient condition."""

    def test_gradient_vanishes(self):
        rng = np.random.default_rng(42)
        N, M = 25, 4
        lam = 1.2
        Phi = rng.normal(size=(N, M))
        t = rng.normal(size=N)

        w_sol = la.solve(Phi.T @ Phi + lam * np.eye(M), Phi.T @ t)
        grad = (Phi.T @ Phi + lam * np.eye(M)) @ w_sol - Phi.T @ t
        np.testing.assert_allclose(grad, np.zeros(M), atol=1e-12)


class TestExercise4_7_MultivariateGaussianMLE:
    """Exercise 4.7: Multivariate Gaussian linear regression decoupling."""

    def test_mle_decoupling(self):
        rng = np.random.default_rng(42)
        N, M, K = 30, 4, 3
        Phi = rng.normal(size=(N, M))
        A = rng.normal(size=(K, K))
        Sigma = A @ A.T + np.eye(K)
        W_true = rng.normal(size=(M, K))
        T = Phi @ W_true + rng.multivariate_normal(np.zeros(K), Sigma, size=N)

        # W_ML is independent of Sigma
        W_ml = la.pinv(Phi) @ T
        res = T - Phi @ W_ml
        Sigma_ml = (res.T @ res) / N

        grad_W = Phi.T @ res @ la.inv(Sigma_ml)
        np.testing.assert_allclose(grad_W, np.zeros((M, K)), atol=1e-10)


class TestExercise4_8_4_9_4_10_MultivariateSquaredLoss:
    """Exercises 4.8, 4.9, 4.10: Multivariate expected squared loss and orthogonal decomposition."""

    def test_multivariate_loss_decomposition(self):
        rng = np.random.default_rng(42)
        K = 3
        N_samples = 40000
        mu = np.array([1.0, -1.0, 2.0])
        Sigma = np.array([
            [0.5, 0.2, 0.0],
            [0.2, 0.4, 0.1],
            [0.0, 0.1, 0.6]
        ])
        t_samples = rng.multivariate_normal(mu, Sigma, size=N_samples)

        # Predictor f = mu + delta
        delta = np.array([0.2, -0.3, 0.1])
        f_sub = mu + delta

        # Total empirical loss
        total_loss = np.mean(np.sum((t_samples - f_sub) ** 2, axis=1))

        # Decomposition: ||f - mu||^2 + Tr(Sigma)
        model_error = np.sum(delta ** 2)
        noise_trace = np.trace(Sigma)
        expected_loss = model_error + noise_trace

        np.testing.assert_allclose(total_loss, expected_loss, rtol=0.02)


class TestExercise4_11_GeneralizedGaussian:
    """Exercise 4.11: Generalized Gaussian distribution and L_q loss."""

    def test_normalization_and_gaussian_limit(self):
        def pdf(x, s2, q):
            c = q / (2.0 * (2.0 * s2)**(1.0 / q) * special.gamma(1.0 / q))
            return c * np.exp(- (np.abs(x) ** q) / (2.0 * s2))

        # Check normalization
        for q in [0.7, 1.0, 2.0, 3.0]:
            val, _ = integrate.quad(lambda x: pdf(x, 1.0, q), -np.inf, np.inf)
            assert np.isclose(val, 1.0, atol=1e-6)

        # q=2 matches standard Gaussian
        x = np.linspace(-3, 3, 50)
        p_q2 = pdf(x, 1.0, 2.0)
        p_gauss = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)
        np.testing.assert_allclose(p_q2, p_gauss, atol=1e-12)


class TestExercise4_12_MinkowskiProperties:
    """Exercise 4.12: Minkowski loss limits (mean, median, mode)."""

    def test_continuous_minimizers(self):
        # Gamma distribution Gamma(shape=2, scale=1): p(t) = t * exp(-t)
        # Mean = 2.0, Median ~ 1.6783, Mode = 1.0
        def loss_int(y, q):
            v, _ = integrate.quad(lambda t: (np.abs(y - t) ** q) * t * np.exp(-t), 0.0, 20.0)
            return v

        opt_q2 = opt.minimize_scalar(lambda y: loss_int(y, 2.0), bounds=(0.5, 3.0), method='bounded').x
        opt_q1 = opt.minimize_scalar(lambda y: loss_int(y, 1.0), bounds=(0.5, 3.0), method='bounded').x
        opt_q05 = opt.minimize_scalar(lambda y: loss_int(y, 0.5), bounds=(0.5, 3.0), method='bounded').x
        opt_q01 = opt.minimize_scalar(lambda y: loss_int(y, 0.1), bounds=(0.5, 3.0), method='bounded').x

        # q=2 is mean
        assert np.isclose(opt_q2, 2.0, atol=0.01)
        # q=1 is median
        assert np.isclose(opt_q1, 1.6783, atol=0.01)
        # Strict monotonicity: mode (1.0) < opt_q01 < opt_q05 < opt_q1 (1.678) < opt_q2 (2.0)
        assert 1.0 < opt_q01 < opt_q05 < opt_q1 < opt_q2
