"""
Unit tests for Chapter 3 Section 3.2: The Multivariate Gaussian
Covers Subsections 3.2.1 to 3.2.9 and Figures 3.2 to 3.8.
"""
import os
import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate
import pandas as pd
import pytest

from common.probability import (
    MultivariateGaussian,
    GaussianMixtureModel,
    plot_figure_3_2,
    plot_figure_3_3,
    plot_figure_3_4,
    plot_figure_3_5,
    plot_figure_3_6,
    plot_figure_3_7,
    plot_figure_3_8,
)


class TestMultivariateGaussian:
    """Tests for Subsection 3.2.1 (Geometry), 3.2.2 (Moments), 3.2.3 (Limitations)."""

    def test_initialization_and_properties(self):
        mu = np.array([1.0, 2.0])
        sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
        gauss = MultivariateGaussian(mu=mu, sigma=sigma)

        assert gauss.D == 2
        assert np.allclose(gauss.mean, mu)
        assert np.allclose(gauss.covariance, sigma)
        assert np.allclose(gauss.precision @ sigma, np.eye(2), atol=1e-7)

        # Spectral decomposition Sigma = U Lambda U^T (Eq 3.28, 3.30)
        U = gauss.eigenvectors
        Lambda = np.diag(gauss.eigenvalues)
        assert np.allclose(U @ Lambda @ U.T, sigma, atol=1e-7)
        assert np.allclose(U.T @ U, np.eye(2), atol=1e-7)
        assert np.all(gauss.eigenvalues > 0)

    def test_invalid_covariance_raises(self):
        # Non-symmetric
        with pytest.raises(ValueError, match="symmetric"):
            MultivariateGaussian(mu=[0, 0], sigma=[[1, 2], [3, 1]])

        # Dimension mismatch
        with pytest.raises(ValueError, match="Dimension mismatch"):
            MultivariateGaussian(mu=[0, 0], sigma=[[1]])

        # Non positive-definite
        with pytest.raises(ValueError, match="strictly positive definite"):
            MultivariateGaussian(mu=[0, 0], sigma=[[-1, 0], [0, 1]])

    def test_mahalanobis_and_pdf(self):
        mu = np.array([0.0, 0.0])
        sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
        gauss = MultivariateGaussian(mu=mu, sigma=sigma)

        # At mean, Mahalanobis distance is 0 and density matches analytical formula
        assert gauss.mahalanobis_distance_squared(mu) == pytest.approx(0.0)
        expected_peak = 1.0 / (2 * np.pi)
        assert gauss.pdf(mu) == pytest.approx(expected_peak)

        # Test Mahalanobis distance on sphere
        x = np.array([1.0, 1.0])
        assert gauss.mahalanobis_distance_squared(x) == pytest.approx(2.0)

        # Batch evaluation
        X_batch = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        pdf_batch = gauss.pdf(X_batch)
        assert len(pdf_batch) == 3
        assert pdf_batch[0] == pytest.approx(expected_peak)

    def test_density_normalization_2d(self):
        mu = np.array([0.5, -0.5])
        sigma = np.array([[1.2, 0.4], [0.4, 0.8]])
        gauss = MultivariateGaussian(mu=mu, sigma=sigma)

        # Integrate over [-6, 7] x [-6, 6]
        result, _ = integrate.dblquad(
            lambda y, x: gauss.pdf(np.array([x, y])),
            -6.0, 7.0,
            lambda x: -6.0, lambda x: 6.0
        )
        assert result == pytest.approx(1.0, rel=1e-3)

    def test_eigen_basis_transformation(self):
        mu = np.array([2.0, 3.0])
        sigma = np.array([[2.0, 0.8], [0.8, 1.5]])
        gauss = MultivariateGaussian(mu=mu, sigma=sigma)

        x = np.array([3.5, 4.2])
        y = gauss.transform_to_eigen_basis(x)
        x_recon = gauss.transform_from_eigen_basis(y)
        assert np.allclose(x, x_recon, atol=1e-7)

        # In eigen-basis, Mahalanobis distance squared is sum (y_i^2 / lambda_i) (Eq 3.33)
        dist_sq = gauss.mahalanobis_distance_squared(x)
        dist_sq_eigen = np.sum(y**2 / gauss.eigenvalues)
        assert dist_sq == pytest.approx(dist_sq_eigen, rel=1e-6)

    def test_covariance_type_classification(self):
        spherical = MultivariateGaussian([0, 0], [[2, 0], [0, 2]])
        diagonal = MultivariateGaussian([0, 0], [[2, 0], [0, 1]])
        full = MultivariateGaussian([0, 0], [[2, 0.5], [0.5, 1]])

        assert spherical.covariance_type() == 'spherical'
        assert diagonal.covariance_type() == 'diagonal'
        assert full.covariance_type() == 'full'


class TestConditionalAndMarginal:
    """Tests for Subsection 3.2.4 (Conditional) and 3.2.5 (Marginal)."""

    def test_partitioned_gaussian(self):
        # 2D Gaussian with strong correlation rho = 0.8
        mu = np.array([0.5, 0.5])
        sigma = np.array([[0.04, 0.032], [0.032, 0.04]])
        gauss = MultivariateGaussian(mu=mu, sigma=sigma)

        # Marginal p(x_a) (Section 3.2.5, Eq 3.82)
        marg_a = gauss.marginalize(indices_a=[0])
        assert marg_a.D == 1
        assert marg_a.mean[0] == pytest.approx(mu[0])
        assert marg_a.covariance[0, 0] == pytest.approx(sigma[0, 0])

        # Conditional p(x_a | x_b = 0.7) (Section 3.2.4, Eq 3.79, 3.80)
        xb_val = 0.7
        cond_a = gauss.condition_on(indices_a=[0], indices_b=[1], x_b=[xb_val])
        assert cond_a.D == 1

        expected_cond_mean = mu[0] + (sigma[0, 1] / sigma[1, 1]) * (xb_val - mu[1])
        expected_cond_var = sigma[0, 0] - (sigma[0, 1]**2 / sigma[1, 1])

        assert cond_a.mean[0] == pytest.approx(expected_cond_mean, rel=1e-6)
        assert cond_a.covariance[0, 0] == pytest.approx(expected_cond_var, rel=1e-6)

        # Variance reduction due to conditioning: Sigma_{a|b} < Sigma_{aa}
        assert cond_a.covariance[0, 0] < marg_a.covariance[0, 0]


class TestBayesLinearGaussian:
    """Tests for Subsection 3.2.6 (Bayes' theorem for linear-Gaussian models)."""

    def test_linear_gaussian_system(self):
        # 1D prior x ~ N(0, 1)
        prior = MultivariateGaussian(mu=[0.0], sigma=[[1.0]])
        # Measurement y = 2*x + 1 + eps, eps ~ N(0, 0.5)
        A = np.array([[2.0]])
        b = np.array([1.0])
        L_cov = np.array([[0.5]])

        # Marginal p(y) = N(y | A mu + b, L^-1 + A Sigma A^T)
        # E[y] = 2*(0) + 1 = 1, Var[y] = 0.5 + 4*1 = 4.5
        marginal_y, posterior_x = MultivariateGaussian.bayes_linear_gaussian(
            prior=prior, A=A, b=b, L_cov=L_cov, y_obs=[3.0]
        )

        assert marginal_y.mean[0] == pytest.approx(1.0)
        assert marginal_y.covariance[0, 0] == pytest.approx(4.5)

        # Posterior p(x | y = 3.0):
        # Precision = 1 + 2^2 / 0.5 = 1 + 8 = 9 => Var = 1/9
        # Mean = (1/9) * (2/0.5 * (3 - 1) + 1*0) = (1/9) * (4 * 2) = 8/9
        assert posterior_x is not None
        assert posterior_x.covariance[0, 0] == pytest.approx(1.0 / 9.0, rel=1e-6)
        assert posterior_x.mean[0] == pytest.approx(8.0 / 9.0, rel=1e-6)


class TestMLEAndSequential:
    """Tests for Subsection 3.2.7 (MLE) and 3.2.8 (Sequential estimation)."""

    def test_mle_multivariate(self):
        np.random.seed(42)
        true_mu = np.array([1.5, -2.0])
        true_sigma = np.array([[1.5, 0.7], [0.7, 2.0]])
        true_gauss = MultivariateGaussian(mu=true_mu, sigma=true_sigma)

        X = true_gauss.sample(size=10000, seed=42)
        fitted_gauss = MultivariateGaussian.fit_mle(X, unbiased_cov=False)

        assert np.allclose(fitted_gauss.mean, true_mu, atol=0.08)
        assert np.allclose(fitted_gauss.covariance, true_sigma, atol=0.08)

        # Unbiased covariance has factor N / (N - 1)
        fitted_unbiased = MultivariateGaussian.fit_mle(X, unbiased_cov=True)
        ratio = fitted_unbiased.covariance / fitted_gauss.covariance
        assert np.allclose(ratio, 10000.0 / 9999.0, atol=1e-7)

    def test_sequential_mean_update(self):
        np.random.seed(42)
        data = np.random.randn(50, 2)

        mu_seq = np.zeros(2)
        for i, x in enumerate(data):
            N = i + 1
            mu_seq = MultivariateGaussian.sequential_mean_update(mu_seq, x, N)

        exact_mean = np.mean(data, axis=0)
        assert np.allclose(mu_seq, exact_mean, atol=1e-12)


class TestGaussianMixtureModel:
    """Tests for Subsection 3.2.9 (Mixtures of Gaussians) and Old Faithful."""

    def test_gmm_basic_properties(self):
        comp1 = MultivariateGaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
        comp2 = MultivariateGaussian([3.0, 3.0], [[1.0, 0.0], [0.0, 1.0]])
        gmm = GaussianMixtureModel(pi=[0.4, 0.6], components=[comp1, comp2])

        assert gmm.K == 2
        assert gmm.D == 2
        assert np.isclose(np.sum(gmm.pi), 1.0)

        # Check responsibilities sum to 1
        x = np.array([1.5, 1.5])
        resp = gmm.responsibilities(x)
        assert len(resp) == 2
        assert np.isclose(np.sum(resp), 1.0)

        # Batch responsibilities
        X_batch = np.array([[0.0, 0.0], [3.0, 3.0], [1.5, 1.5]])
        resp_batch = gmm.responsibilities(X_batch)
        assert resp_batch.shape == (3, 2)
        assert np.allclose(np.sum(resp_batch, axis=1), 1.0)

    def test_gmm_em_fitting_synthetic(self):
        np.random.seed(42)
        # Create synthetic 2-cluster data
        c1 = np.random.randn(150, 2) + np.array([-2.0, -2.0])
        c2 = np.random.randn(150, 2) + np.array([2.0, 2.0])
        X = np.vstack([c1, c2])

        gmm = GaussianMixtureModel.fit_em(X, K=2, max_iter=50, seed=42)

        assert gmm.K == 2
        assert np.isclose(np.sum(gmm.pi), 1.0)
        # Check that cluster centers are near (-2, -2) and (2, 2)
        means = np.array([comp.mean for comp in gmm.components])
        dists1 = np.linalg.norm(means - np.array([-2.0, -2.0]), axis=1)
        dists2 = np.linalg.norm(means - np.array([2.0, 2.0]), axis=1)
        assert np.min(dists1) < 0.6
        assert np.min(dists2) < 0.6

    def test_old_faithful_dataset_fit(self):
        csv_path = 'common/data/faithful.csv'
        assert os.path.exists(csv_path)

        df = pd.read_csv(csv_path)
        X = df[['duration', 'waiting']].values
        assert X.shape == (272, 2)

        # Single Gaussian fit
        single_gauss = MultivariateGaussian.fit_mle(X)
        ll_single = np.sum(single_gauss.log_pdf(X))

        # 2-component GMM fit
        gmm = GaussianMixtureModel.fit_em(X, K=2, max_iter=60, seed=42)
        ll_gmm = np.sum(gmm.log_pdf(X))

        # GMM must fit the bi-modal data significantly better than single Gaussian
        assert ll_gmm > ll_single + 50.0


class TestFigureGenerators:
    """Verify that Figures 3.2 to 3.8 are properly generated and saved."""

    def test_figures_exist_and_valid(self):
        expected_figs = [
            'fig3_02_central_limit_theorem.png',
            'fig3_03_gaussian_geometry.png',
            'fig3_04_covariance_geometries.png',
            'fig3_05_conditional_marginal.png',
            'fig3_06_old_faithful_gaussian_and_mixture.png',
            'fig3_07_gaussian_mixture_1d.png',
            'fig3_08_gaussian_mixture_2d.png',
        ]
        for fig_name in expected_figs:
            p1 = os.path.join('3/result', fig_name)
            p2 = os.path.join('result', fig_name)
            assert os.path.exists(p1), f"Missing {p1}"
            assert os.path.getsize(p1) > 5000, f"File {p1} too small"
            assert os.path.exists(p2), f"Missing {p2}"
            assert os.path.getsize(p2) > 5000, f"File {p2} too small"
