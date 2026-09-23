"""Unit tests for Chapter 6 Exercises: Deep Neural Networks.

Exercises 6.1 to 6.21 (all 21 exercises).
Bishop & Bishop (2024), Chapter 6, pp. 204-207.
"""

import numpy as np
import pytest
from scipy import integrate, special

from common.exercises_ch6 import (
    exercise_6_1_hypersphere_surface,
    exercise_6_1_hypersphere_volume,
    exercise_6_1_verify_polar_integral,
    exercise_6_2_sphere_to_cube_volume_ratio,
    exercise_6_2_stirling_ratio_approximation,
    exercise_6_2_corner_distance_ratio,
    exercise_6_3_gaussian_radial_density,
    exercise_6_3_radial_mode,
    exercise_6_3_density_ratio_origin_to_mode,
    exercise_6_4_convert_sigmoid_to_tanh_weights,
    exercise_6_4_eval_networks,
    exercise_6_5_swish,
    exercise_6_5_swish_deriv,
    exercise_6_5_relu,
    exercise_6_6_tanh_deriv,
    exercise_6_7_softplus,
    exercise_6_7_softplus_inv,
    exercise_6_7_verify_properties,
    exercise_6_8_mle_variance,
    exercise_6_9_mle_variance_multioutput,
    exercise_6_10_mle_covariance,
    exercise_6_10_mahalanobis_loss,
    exercise_6_11_robust_loss,
    exercise_6_11_robust_gradient_preactivation,
    exercise_6_12_tanh_loss,
    exercise_6_12_tanh_loss_from_a,
    exercise_6_12_tanh_gradient_a,
    exercise_6_13_multiclass_cross_entropy,
    exercise_6_14_verify_sigmoid_cross_entropy_grad,
    exercise_6_15_verify_softmax_cross_entropy_grad,
    exercise_6_16_forward_kinematics,
    exercise_6_17_compute_responsibilities,
    exercise_6_18_verify_grad_pi,
    exercise_6_19_verify_grad_mu,
    exercise_6_20_verify_grad_sigma,
    exercise_6_21_conditional_mean_variance,
    exercise_6_21_monte_carlo_mean_variance,
)


class TestChapter6Exercises:
    """Mathematical and numerical verification of Exercises 6.1 - 6.21."""

    def test_exercise_6_1_hypersphere_geometry(self):
        """Exercise 6.1: Hypersphere surface area S_D and volume V_D."""
        # D = 1: S_1 = 2 (two points {-1, 1}), V_1 = 2 (line [-1, 1])
        assert np.isclose(exercise_6_1_hypersphere_surface(1), 2.0)
        assert np.isclose(exercise_6_1_hypersphere_volume(1), 2.0)

        # D = 2: S_2 = 2 * pi, V_2 = pi
        assert np.isclose(exercise_6_1_hypersphere_surface(2), 2.0 * np.pi)
        assert np.isclose(exercise_6_1_hypersphere_volume(2), np.pi)

        # D = 3: S_3 = 4 * pi, V_3 = 4/3 * pi
        assert np.isclose(exercise_6_1_hypersphere_surface(3), 4.0 * np.pi)
        assert np.isclose(exercise_6_1_hypersphere_volume(3), (4.0 / 3.0) * np.pi)

        # D = 4: S_4 = 2 * pi^2, V_4 = 0.5 * pi^2
        assert np.isclose(exercise_6_1_hypersphere_surface(4), 2.0 * (np.pi ** 2))
        assert np.isclose(exercise_6_1_hypersphere_volume(4), 0.5 * (np.pi ** 2))

        # Check V_D = S_D / D relation
        for d in [1, 2, 3, 4, 5, 10]:
            s_d = exercise_6_1_hypersphere_surface(d)
            v_d = exercise_6_1_hypersphere_volume(d)
            assert np.isclose(v_d, s_d / d)

        # Verify polar coordinate integral equality (Eq 6.51)
        for d in [1, 2, 3, 5, 8]:
            lhs, rhs = exercise_6_1_verify_polar_integral(d)
            assert np.isclose(lhs, rhs, rtol=1e-5)

    def test_exercise_6_2_sphere_to_cube_volume_ratio(self):
        """Exercise 6.2: Ratio of hypersphere to hypercube volume and corner distance."""
        # D = 2: pi / 4
        assert np.isclose(exercise_6_2_sphere_to_cube_volume_ratio(2), np.pi / 4.0)

        # D = 3: (4/3 pi) / 8 = pi / 6
        assert np.isclose(exercise_6_2_sphere_to_cube_volume_ratio(3), np.pi / 6.0)

        # Volume ratio must decrease strictly monotonically
        ratios = [exercise_6_2_sphere_to_cube_volume_ratio(d) for d in range(1, 15)]
        for i in range(len(ratios) - 1):
            assert ratios[i] > ratios[i + 1]

        # For high dimensions, ratio approaches 0
        assert exercise_6_2_sphere_to_cube_volume_ratio(20) < 1e-7
        assert exercise_6_2_sphere_to_cube_volume_ratio(50) < 1e-27

        # Stirling approximation accuracy for large D
        for d in [20, 40, 60]:
            exact = exercise_6_2_sphere_to_cube_volume_ratio(d)
            approx = exercise_6_2_stirling_ratio_approximation(d)
            assert np.isclose(exact, approx, rtol=0.05)

        # Distance ratio to corner is sqrt(D)
        for d in [1, 4, 9, 16, 25]:
            assert np.isclose(exercise_6_2_corner_distance_ratio(d), np.sqrt(d))

    def test_exercise_6_3_gaussian_radial_density(self):
        """Exercise 6.3: Radial density of Gaussian in high dimensions and mode."""
        # Integral of p(r) over [0, inf) is 1.0
        for d in [1, 2, 3, 5, 10]:
            for sigma in [0.5, 1.0, 2.0]:
                integral_val, _ = integrate.quad(
                    lambda r: exercise_6_3_gaussian_radial_density(r, D=d, sigma=sigma),
                    0.0, 30.0
                )
                assert np.isclose(integral_val, 1.0, atol=1e-5)

        # Mode r_hat = sigma * sqrt(D - 1)
        for d in [2, 5, 10, 20]:
            sigma = 1.5
            r_mode = exercise_6_3_radial_mode(d, sigma=sigma)
            assert np.isclose(r_mode, sigma * np.sqrt(d - 1))

            # Verify that derivative of ln p(r) is zero at r_mode
            eps = 1e-5
            p_plus = exercise_6_3_gaussian_radial_density(r_mode + eps, D=d, sigma=sigma)
            p_minus = exercise_6_3_gaussian_radial_density(r_mode - eps, D=d, sigma=sigma)
            deriv = (p_plus - p_minus) / (2.0 * eps)
            assert np.abs(deriv) < 1e-4

        # Density ratio p(0) / p(r_hat) = exp((D-1)/2)
        for d in [3, 5, 10]:
            ratio = exercise_6_3_density_ratio_origin_to_mode(d)
            assert np.isclose(ratio, np.exp((d - 1.0) / 2.0))

    def test_exercise_6_4_sigmoid_to_tanh_weights(self):
        """Exercise 6.4: Equivalence between sigmoid and tanh two-layer networks."""
        rng = np.random.default_rng(42)
        D, M, K = 4, 6, 3
        N = 25

        X = rng.normal(size=(N, D))
        W1 = rng.normal(size=(M, D))
        b1 = rng.normal(size=M)
        W2 = rng.normal(size=(K, M))
        b2 = rng.normal(size=K)

        W1_t, b1_t, W2_t, b2_t = exercise_6_4_convert_sigmoid_to_tanh_weights(W1, b1, W2, b2)
        y_sig, y_tanh, max_diff = exercise_6_4_eval_networks(
            X, W1, b1, W2, b2, W1_t, b1_t, W2_t, b2_t
        )

        assert max_diff < 1e-12
        assert np.allclose(y_sig, y_tanh, atol=1e-12)

    def test_exercise_6_5_swish_and_relu(self):
        """Exercise 6.5: Swish activation, derivative, and ReLU limit."""
        x = np.linspace(-3.0, 3.0, 100)

        # Swish at x=0 is 0
        assert np.isclose(exercise_6_5_swish(np.array([0.0]), beta=1.0)[0], 0.0)

        # Swish derivative matches finite differences
        eps = 1e-6
        for beta in [0.1, 1.0, 10.0]:
            h_deriv = exercise_6_5_swish_deriv(x, beta=beta)
            h_num = (exercise_6_5_swish(x + eps, beta=beta) - exercise_6_5_swish(x - eps, beta=beta)) / (2.0 * eps)
            assert np.allclose(h_deriv, h_num, atol=1e-5)

        # Limit as beta -> infty matches ReLU
        x_nonzero = np.array([-3.0, -1.0, -0.1, 0.1, 1.0, 3.0])
        swish_large_beta = exercise_6_5_swish(x_nonzero, beta=1000.0)
        relu_val = exercise_6_5_relu(x_nonzero)
        assert np.allclose(swish_large_beta, relu_val, atol=1e-3)

    def test_exercise_6_6_tanh_deriv(self):
        """Exercise 6.6: Derivative of tanh is 1 - tanh^2."""
        a = np.linspace(-4.0, 4.0, 50)
        deriv_formula = exercise_6_6_tanh_deriv(a)

        eps = 1e-6
        deriv_num = (np.tanh(a + eps) - np.tanh(a - eps)) / (2.0 * eps)
        assert np.allclose(deriv_formula, deriv_num, atol=1e-5)

    def test_exercise_6_7_softplus_properties(self):
        """Exercise 6.7: Softplus properties (Eq 6.62 - 6.65)."""
        a = np.array([-3.5, -1.2, 0.0, 0.8, 2.5, 5.0])
        results = exercise_6_7_verify_properties(a)
        for name, passed in results.items():
            assert passed, f"Property failed: {name}"

    def test_exercise_6_8_mle_variance(self):
        """Exercise 6.8: Variance MLE for single-output regression."""
        rng = np.random.default_rng(10)
        residuals = rng.normal(loc=0.0, scale=1.5, size=200)
        sigma2_opt = exercise_6_8_mle_variance(residuals)

        # Check that negative log likelihood is minimized at sigma2_opt
        N = len(residuals)
        def nll(s2):
            return 0.5 * N * np.log(2.0 * np.pi * s2) + 0.5 * np.sum(residuals ** 2) / s2

        loss_opt = nll(sigma2_opt)
        for delta in [-0.2, -0.05, 0.05, 0.2]:
            assert nll(sigma2_opt + delta) > loss_opt

    def test_exercise_6_9_mle_variance_multioutput(self):
        """Exercise 6.9: Multi-output regression noise variance MLE."""
        rng = np.random.default_rng(20)
        N, K = 150, 4
        residuals = rng.normal(loc=0.0, scale=2.0, size=(N, K))
        sigma2_opt = exercise_6_9_mle_variance_multioutput(residuals)

        def nll(s2):
            return 0.5 * N * K * np.log(2.0 * np.pi * s2) + 0.5 * np.sum(residuals ** 2) / s2

        loss_opt = nll(sigma2_opt)
        for delta in [-0.3, 0.3]:
            assert nll(sigma2_opt + delta) > loss_opt

    def test_exercise_6_10_mle_covariance(self):
        """Exercise 6.10: Full covariance MLE and Mahalanobis loss."""
        rng = np.random.default_rng(30)
        N, K = 200, 3
        true_cov = np.array([[2.0, 0.5, 0.2], [0.5, 1.5, 0.3], [0.2, 0.3, 1.0]])
        residuals = rng.multivariate_normal(mean=np.zeros(K), cov=true_cov, size=N)

        mle_cov = exercise_6_10_mle_covariance(residuals)
        # Check symmetry
        assert np.allclose(mle_cov, mle_cov.T)
        # Check positive-definiteness
        eigenvals = np.linalg.eigvalsh(mle_cov)
        assert np.all(eigenvals > 0)

        # Mahalanobis loss
        loss = exercise_6_10_mahalanobis_loss(residuals, mle_cov)
        assert loss > 0
        # Trace relation: sum_n r_n^T (R^T R / N)^(-1) r_n = Tr( (R^T R / N)^(-1) R^T R ) = N * K
        assert np.isclose(loss, 0.5 * N * K)

    def test_exercise_6_11_robust_loss(self):
        """Exercise 6.11: Binary classification with label noise."""
        rng = np.random.default_rng(42)
        y = rng.uniform(0.01, 0.99, size=50)
        t = rng.integers(0, 2, size=50).astype(float)

        # For eps = 0, matches standard cross entropy
        loss_eps0 = exercise_6_11_robust_loss(y, t, eps=0.0)
        std_ce = - np.sum(t * np.log(y) + (1.0 - t) * np.log(1.0 - y))
        assert np.isclose(loss_eps0, std_ce)

        # Robust gradient bounded when y -> 0 for t = 1
        y_extreme = np.array([1e-5])
        t_extreme = np.array([1.0])
        grad_eps0 = exercise_6_11_robust_gradient_preactivation(y_extreme, t_extreme, eps=0.0)
        grad_eps_robust = exercise_6_11_robust_gradient_preactivation(y_extreme, t_extreme, eps=0.1)

        # When eps=0, grad = y - t = 1e-5 - 1 = -0.99999
        assert np.isclose(grad_eps0[0], -1.0, atol=1e-4)
        # When eps=0.1, the gradient is significantly softened
        assert np.abs(grad_eps_robust[0]) < np.abs(grad_eps0[0])

    def test_exercise_6_12_tanh_loss(self):
        """Exercise 6.12: Binary classification with targets in {-1, +1} and tanh output."""
        rng = np.random.default_rng(50)
        a = rng.normal(size=30)
        y = np.tanh(a)
        t = rng.choice([-1.0, 1.0], size=30)

        loss_y = exercise_6_12_tanh_loss(y, t)
        loss_a = exercise_6_12_tanh_loss_from_a(a, t)
        assert np.isclose(loss_y, loss_a)

        # Gradient dE_n / da_n = y_n - t_n
        grad_analytic = exercise_6_12_tanh_gradient_a(y, t)
        eps = 1e-6
        grad_num = np.zeros_like(a)
        for i in range(len(a)):
            ap = a.copy()
            am = a.copy()
            ap[i] += eps
            am[i] -= eps
            grad_num[i] = (exercise_6_12_tanh_loss_from_a(ap, t) - exercise_6_12_tanh_loss_from_a(am, t)) / (2.0 * eps)
        assert np.allclose(grad_analytic, grad_num, atol=1e-5)

    def test_exercise_6_13_multiclass_cross_entropy(self):
        """Exercise 6.13: Multi-class cross-entropy error function."""
        rng = np.random.default_rng(60)
        N, K = 20, 4
        # Softmax outputs
        logits = rng.normal(size=(N, K))
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # 1-of-K targets
        labels = rng.integers(0, K, size=N)
        t = np.zeros((N, K))
        for i, l in enumerate(labels):
            t[i, l] = 1.0

        loss = exercise_6_13_multiclass_cross_entropy(y, t)
        manual_loss = - np.sum(np.log(y[np.arange(N), labels]))
        assert np.isclose(loss, manual_loss)

    def test_exercise_6_14_sigmoid_cross_entropy_grad(self):
        """Exercise 6.14: Pre-activation gradient for independent sigmoid units."""
        rng = np.random.default_rng(70)
        a = rng.normal(size=5)
        t = rng.integers(0, 2, size=5).astype(float)

        grad_analytic, grad_num = exercise_6_14_verify_sigmoid_cross_entropy_grad(a, t)
        assert np.allclose(grad_analytic, grad_num, atol=1e-5)

    def test_exercise_6_15_softmax_cross_entropy_grad(self):
        """Exercise 6.15: Pre-activation gradient for softmax units."""
        rng = np.random.default_rng(80)
        a = rng.normal(size=5)
        t = np.zeros(5)
        t[rng.integers(0, 5)] = 1.0

        grad_analytic, grad_num = exercise_6_15_verify_softmax_cross_entropy_grad(a, t)
        assert np.allclose(grad_analytic, grad_num, atol=1e-5)

    def test_exercise_6_16_forward_kinematics(self):
        """Exercise 6.16: Planar two-link manipulator forward kinematics."""
        L1, L2 = 0.8, 0.2
        # Case 1: theta1=0, theta2=0 -> horizontal line x1 = L1 + L2 = 1.0, x2 = 0.0
        x1, x2 = exercise_6_16_forward_kinematics(0.0, 0.0, L1=L1, L2=L2)
        assert np.isclose(x1, 1.0)
        assert np.isclose(x2, 0.0)

        # Case 2: theta1=pi/2, theta2=0 -> vertical line x1 = 0.0, x2 = 1.0
        x1, x2 = exercise_6_16_forward_kinematics(np.pi / 2.0, 0.0, L1=L1, L2=L2)
        assert np.isclose(x1, 0.0)
        assert np.isclose(x2, 1.0)

        # Range check: distance from origin is within [|L1-L2|, L1+L2]
        rng = np.random.default_rng(90)
        t1 = rng.uniform(-np.pi, np.pi, size=100)
        t2 = rng.uniform(-np.pi, np.pi, size=100)
        x1_arr, x2_arr = exercise_6_16_forward_kinematics(t1, t2, L1=L1, L2=L2)
        dist = np.sqrt(x1_arr ** 2 + x2_arr ** 2)
        assert np.all(dist <= L1 + L2 + 1e-10)
        assert np.all(dist >= np.abs(L1 - L2) - 1e-10)

    def test_exercise_6_17_responsibilities(self):
        """Exercise 6.17: MDN responsibilities gamma_nk sum to 1 and reflect posterior probabilities."""
        pi = np.array([0.5, 0.3, 0.2])
        mu = np.array([[0.0, 0.0], [2.0, 2.0], [-2.0, -2.0]])
        sigma = np.array([0.5, 1.0, 0.8])
        t = np.array([0.1, 0.05])

        gamma = exercise_6_17_compute_responsibilities(pi, mu, sigma, t)
        assert np.isclose(np.sum(gamma), 1.0)
        # Since t is very close to mu[0] = [0, 0], gamma[0] should dominate
        assert gamma[0] > gamma[1]
        assert gamma[0] > gamma[2]

    def test_exercise_6_18_grad_pi(self):
        """Exercise 6.18: MDN derivative w.r.t mixing coefficient pre-activations."""
        rng = np.random.default_rng(101)
        a_pi = rng.normal(size=3)
        mu = rng.normal(size=(3, 2))
        sigma = rng.uniform(0.3, 1.5, size=3)
        t = rng.normal(size=2)

        grad_an, grad_num = exercise_6_18_verify_grad_pi(a_pi, mu, sigma, t)
        assert np.allclose(grad_an, grad_num, atol=1e-5)

    def test_exercise_6_19_grad_mu(self):
        """Exercise 6.19: MDN derivative w.r.t component mean pre-activations."""
        rng = np.random.default_rng(102)
        pi = np.array([0.4, 0.4, 0.2])
        a_mu = rng.normal(size=(3, 2))
        sigma = rng.uniform(0.3, 1.5, size=3)
        t = rng.normal(size=2)

        grad_an, grad_num = exercise_6_19_verify_grad_mu(pi, a_mu, sigma, t)
        assert np.allclose(grad_an, grad_num, atol=1e-5)

    def test_exercise_6_20_grad_sigma(self):
        """Exercise 6.20: MDN derivative w.r.t component variance pre-activations."""
        rng = np.random.default_rng(103)
        pi = np.array([0.3, 0.5, 0.2])
        mu = rng.normal(size=(3, 2))
        a_sigma = rng.normal(size=3)
        t = rng.normal(size=2)

        grad_an, grad_num = exercise_6_20_verify_grad_sigma(pi, mu, a_sigma, t)
        assert np.allclose(grad_an, grad_num, atol=1e-5)

    def test_exercise_6_21_conditional_mean_variance(self):
        """Exercise 6.21: MDN conditional mean and variance analytical vs Monte Carlo."""
        pi = np.array([0.5, 0.3, 0.2])
        mu = np.array([[1.0, 2.0], [-1.0, 0.0], [0.5, -2.0]])
        sigma = np.array([0.3, 0.6, 0.4])

        an_mean, an_var = exercise_6_21_conditional_mean_variance(pi, mu, sigma)
        mc_mean, mc_var = exercise_6_21_monte_carlo_mean_variance(pi, mu, sigma, n_samples=150000, seed=42)

        assert np.allclose(an_mean, mc_mean, atol=0.03)
        assert np.isclose(an_var, mc_var, rtol=0.03)
