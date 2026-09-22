"""
Unit tests for Chapter 3, Section 3.4: The Exponential Family.
Tests canonical parameter transformations, cumulant generating functions,
first and second derivatives (expectation and covariance of sufficient statistics),
and Maximum Likelihood Estimation via sufficient statistics.
"""
import os
import pytest
import numpy as np
import scipy.linalg as la

from common.probability import (
    ExponentialFamilyBase,
    BernoulliExponential,
    GaussianExponential1D,
    MultinomialExponential,
    VonMisesExponential,
    plot_figure_3_13_exp_family_geometry,
    plot_figure_3_14_exp_family_members,
    plot_figure_3_15_sufficient_statistics_online
)


class TestBernoulliExponential:
    """Tests for Bernoulli distribution in exponential family form (Eq 3.140 - 3.147)."""

    def test_canonical_transformations(self):
        mu_values = [0.1, 0.3, 0.5, 0.8, 0.95]
        for mu in mu_values:
            dist = BernoulliExponential(mu=mu)
            expected_eta = np.log(mu / (1.0 - mu))
            assert np.isclose(dist.eta, expected_eta)
            
            # Test inverse mapping
            dist_eta = BernoulliExponential(eta=expected_eta)
            assert np.isclose(dist_eta.mu, mu)

    def test_cumulant_and_derivatives(self):
        for eta in [-3.0, -1.0, 0.0, 1.5, 4.0]:
            dist = BernoulliExponential(eta=eta)
            A = dist.log_partition()
            expected_A = np.log(1.0 + np.exp(eta))
            assert np.isclose(A, expected_A)
            
            # First derivative E[x] = sigma(eta)
            grad = dist.grad_log_partition()
            assert np.isclose(grad, dist.mu)
            
            # Numerical gradient check
            eps = 1e-6
            num_grad = (dist.log_partition(eta + eps) - dist.log_partition(eta - eps)) / (2 * eps)
            assert np.isclose(grad, num_grad, atol=1e-5)
            
            # Second derivative Var[x] = sigma(eta)(1 - sigma(eta))
            hess = dist.hessian_log_partition()
            expected_hess = dist.mu * (1.0 - dist.mu)
            assert np.isclose(hess, expected_hess)
            assert hess > 0  # Strict convexity

    def test_sufficient_statistics_mle(self):
        np.random.seed(42)
        true_mu = 0.65
        X = (np.random.rand(1000) < true_mu).astype(int)
        
        mle_dist = BernoulliExponential.fit_mle(X)
        assert np.isclose(mle_dist.mu, np.mean(X), atol=1e-10)
        assert np.isclose(mle_dist.mu, true_mu, atol=0.05)


class TestGaussianExponential1D:
    """Tests for Univariate Gaussian in exponential family form (Eq 3.162 - 3.167)."""

    def test_canonical_parameters_and_inverse(self):
        mu = 2.5
        sigma2 = 1.44
        dist = GaussianExponential1D(mu=mu, sigma2=sigma2)
        
        expected_eta1 = mu / sigma2
        expected_eta2 = -1.0 / (2.0 * sigma2)
        assert np.allclose(dist.eta, [expected_eta1, expected_eta2])
        
        # Test inverse from natural parameters
        dist_eta = GaussianExponential1D(eta=dist.eta)
        assert np.isclose(dist_eta.mu, mu)
        assert np.isclose(dist_eta.sigma2, sigma2)

    def test_cumulant_gradient_equals_expectation(self):
        mu = 1.5
        sigma2 = 0.8
        dist = GaussianExponential1D(mu=mu, sigma2=sigma2)
        
        grad = dist.grad_log_partition()
        # E[u(x)] = (E[x], E[x^2])^T = (mu, mu^2 + sigma^2)^T
        expected_grad = np.array([mu, mu**2 + sigma2])
        assert np.allclose(grad, expected_grad)
        
        # Numerical gradient verification
        eps = 1e-6
        num_grad = np.zeros(2)
        for i in range(2):
            eta_plus = dist.eta.copy()
            eta_minus = dist.eta.copy()
            eta_plus[i] += eps
            eta_minus[i] -= eps
            num_grad[i] = (dist.log_partition(eta_plus) - dist.log_partition(eta_minus)) / (2 * eps)
        assert np.allclose(grad, num_grad, atol=1e-5)

    def test_cumulant_hessian_equals_covariance(self):
        mu = -1.2
        sigma2 = 2.0
        dist = GaussianExponential1D(mu=mu, sigma2=sigma2)
        
        hess = dist.hessian_log_partition()
        # Cov[u(x)] where u(x) = (x, x^2)^T
        # Var[x] = sigma^2
        # Cov[x, x^2] = 2 * mu * sigma^2
        # Var[x^2] = 4 * mu^2 * sigma^2 + 2 * sigma^4
        cov_11 = sigma2
        cov_12 = 2.0 * mu * sigma2
        cov_22 = 4.0 * (mu**2) * sigma2 + 2.0 * (sigma2**2)
        expected_hess = np.array([[cov_11, cov_12], [cov_12, cov_22]])
        assert np.allclose(hess, expected_hess)
        
        # Check strict positive-definiteness
        eigvals = la.eigvalsh(hess)
        assert np.all(eigvals > 0)
        
        # Numerical Hessian verification
        eps = 1e-5
        num_hess = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                eta_pp = dist.eta.copy()
                eta_pm = dist.eta.copy()
                eta_mp = dist.eta.copy()
                eta_mm = dist.eta.copy()
                eta_pp[i] += eps; eta_pp[j] += eps
                eta_pm[i] += eps; eta_pm[j] -= eps
                eta_mp[i] -= eps; eta_mp[j] += eps
                eta_mm[i] -= eps; eta_mm[j] -= eps
                num_hess[i, j] = (dist.log_partition(eta_pp) - dist.log_partition(eta_pm)
                                 - dist.log_partition(eta_mp) + dist.log_partition(eta_mm)) / (4 * eps**2)
        assert np.allclose(hess, num_hess, atol=1e-4)

    def test_sufficient_statistics_mle(self):
        np.random.seed(123)
        true_mu = 4.0
        true_sigma2 = 2.25
        X = np.random.normal(true_mu, np.sqrt(true_sigma2), size=5000)
        
        mle_dist = GaussianExponential1D.fit_mle(X)
        assert np.isclose(mle_dist.mu, np.mean(X))
        assert np.isclose(mle_dist.sigma2, np.var(X))
        assert np.isclose(mle_dist.mu, true_mu, atol=0.05)
        assert np.isclose(mle_dist.sigma2, true_sigma2, atol=0.1)


class TestMultinomialExponential:
    """Tests for Multinomial / Categorical canonical representation (Eq 3.155 - 3.161)."""

    def test_softmax_inversion(self):
        mu = np.array([0.2, 0.3, 0.5])
        dist = MultinomialExponential(mu=mu)
        
        expected_eta = np.array([np.log(0.2 / 0.5), np.log(0.3 / 0.5)])
        assert np.allclose(dist.eta, expected_eta)
        
        dist_eta = MultinomialExponential(eta=expected_eta)
        assert np.allclose(dist_eta.mu, mu)

    def test_cumulant_gradient_and_hessian(self):
        dist = MultinomialExponential(mu=np.array([0.15, 0.25, 0.60]))
        grad = dist.grad_log_partition()
        # Should equal mu_{1:M-1}
        assert np.allclose(grad, [0.15, 0.25])
        
        hess = dist.hessian_log_partition()
        # Covariance matrix of categorical variables
        expected_hess = np.diag([0.15, 0.25]) - np.outer([0.15, 0.25], [0.15, 0.25])
        assert np.allclose(hess, expected_hess)
        assert np.all(la.eigvalsh(hess) > 0)

    def test_sufficient_statistics_mle(self):
        np.random.seed(42)
        true_mu = np.array([0.2, 0.5, 0.3])
        N = 3000
        cat_samples = np.random.choice(3, size=N, p=true_mu)
        X = np.eye(3)[cat_samples]
        
        dist = MultinomialExponential.fit_mle(X)
        assert np.allclose(dist.mu, np.mean(X, axis=0))
        assert np.allclose(dist.mu, true_mu, atol=0.03)


class TestVonMisesExponential:
    """Tests for Von Mises canonical exponential representation."""

    def test_parameter_mapping(self):
        theta_0 = np.pi / 3
        m = 3.5
        vm = VonMisesExponential(theta_0=theta_0, m=m)
        
        assert np.isclose(vm.m, m)
        assert np.isclose(vm.theta_0, theta_0)
        
        # Test reconstruct from eta
        vm_eta = VonMisesExponential(eta=vm.eta)
        assert np.isclose(vm_eta.m, m)
        assert np.isclose(vm_eta.theta_0, theta_0)

    def test_gradient_expectation(self):
        vm = VonMisesExponential(theta_0=np.pi / 4, m=2.0)
        grad = vm.grad_log_partition()
        
        from scipy import special
        expected_ratio = special.i1(2.0) / special.i0(2.0)
        expected_grad = expected_ratio * np.array([np.cos(np.pi/4), np.sin(np.pi/4)])
        assert np.allclose(grad, expected_grad)


class TestOnlineSufficientStatistics:
    """Tests online streaming property of sufficient statistics (Subsection 3.4.1)."""

    def test_online_stream_equivalence_to_batch(self):
        np.random.seed(999)
        X = np.random.normal(2.0, 1.5, size=200)
        
        running_sum_u = np.zeros(2)
        for n, x_n in enumerate(X, start=1):
            running_sum_u += np.array([x_n, x_n**2])
            
            # Stream estimate from running sufficient statistics
            sample_mean_u = running_sum_u / n
            mu_online = sample_mean_u[0]
            sigma2_online = sample_mean_u[1] - mu_online**2
            
            if n >= 2:
                # Batch estimate on X[:n]
                batch_dist = GaussianExponential1D.fit_mle(X[:n])
                assert np.isclose(mu_online, batch_dist.mu)
                assert np.isclose(sigma2_online, batch_dist.sigma2)


class TestFigureGeneration:
    """Tests that Section 3.4 figures generate cleanly and return valid objects."""

    def test_figures_run_without_error(self, tmp_path):
        f13, _ = plot_figure_3_13_exp_family_geometry(save_paths=[str(tmp_path / 'fig3_13.png')])
        assert f13 is not None
        assert os.path.exists(tmp_path / 'fig3_13.png')

        f14, _ = plot_figure_3_14_exp_family_members(save_paths=[str(tmp_path / 'fig3_14.png')])
        assert f14 is not None
        assert os.path.exists(tmp_path / 'fig3_14.png')

        f15, _ = plot_figure_3_15_sufficient_statistics_online(save_paths=[str(tmp_path / 'fig3_15.png')])
        assert f15 is not None
        assert os.path.exists(tmp_path / 'fig3_15.png')
