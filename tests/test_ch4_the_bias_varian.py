"""
Unit tests for Chapter 4, Section 4.3: The Bias-Variance Trade-off.

Tests:
- SinusoidalDataGenerator synthetic data generation (Bishop Section 4.3, p. 126)
- GaussianFeatureExtractor basis matrix construction (Eq 4.4, with bias term)
- BiasVarianceSimulator ensemble training and regularized least squares (Eq 4.26, 4.27)
- Ensemble prediction average (Eq 4.50)
- Mathematical properties of bias-variance decomposition:
    - (bias)^2 monotonically increases with lambda (Eq 4.51)
    - variance monotonically decreases with lambda (Eq 4.52)
    - (bias)^2 + variance attains an interior minimum around ln(lambda) ~ 0.43
    - test error closely tracks (bias)^2 + variance + noise (Eq 4.46)
- High-fidelity figure generation for Figures 4.7 and 4.8
"""
import os
import pytest
import numpy as np

from common.bias_variance import (
    SinusoidalDataGenerator,
    GaussianFeatureExtractor,
    BiasVarianceSimulator,
    plot_figure_4_7_bias_variance_ensembles,
    plot_figure_4_8_bias_variance_tradeoff
)


class TestSinusoidalDataGenerator:
    """Test data generation for Section 4.3 experiments."""

    def test_sample_shapes_and_noise(self):
        gen = SinusoidalDataGenerator(noise_std=0.3, random_state=42)
        x, t = gen.sample_dataset(n_samples=25)
        assert x.shape == (25,)
        assert t.shape == (25,)
        assert np.all(x >= 0.0) and np.all(x <= 1.0)

        # True function is sin(2 * pi * x)
        h = gen.true_function(x)
        noise = t - h
        assert np.isclose(np.mean(noise), 0.0, atol=0.15)
        assert np.isclose(np.std(noise), 0.3, atol=0.1)

    def test_sample_ensemble(self):
        gen = SinusoidalDataGenerator(noise_std=0.3, random_state=42)
        ensemble = gen.sample_ensemble(n_datasets=20, n_samples=25)
        assert len(ensemble) == 20
        assert ensemble[0][0].shape == (25,)
        assert ensemble[0][1].shape == (25,)


class TestGaussianFeatureExtractor:
    """Test Gaussian basis design matrix construction."""

    def test_feature_dimensions_and_bias(self):
        fe = GaussianFeatureExtractor(m_centers=24, s=0.1)
        assert fe.n_features == 25  # 24 Gaussians + 1 bias
        assert len(fe.centers) == 24

        x = np.linspace(0.0, 1.0, 50)
        phi = fe(x)
        assert phi.shape == (50, 25)
        # Bias column must be all 1.0
        np.testing.assert_allclose(phi[:, 0], 1.0)
        # Gaussian RBF values in (0, 1]
        assert np.all(phi[:, 1:] > 0.0)
        assert np.all(phi[:, 1:] <= 1.0)


class TestBiasVarianceSimulation:
    """Test Bias-Variance decomposition trends and mathematical properties."""

    def test_shrinkage_behavior(self):
        sim = BiasVarianceSimulator(n_datasets=10, n_samples=25, random_state=42)
        # Very large lambda shrinks weights towards zero
        preds_large, _ = sim.fit_ensemble_for_lambda(lam=1e7)
        assert np.all(np.abs(preds_large) < 0.01)

    def test_decomposition_trends_and_interior_minimum(self):
        sim = BiasVarianceSimulator(n_datasets=40, n_samples=25, random_state=42)
        ln_lambdas = np.array([-3.0, -1.0, 0.43, 2.0, 3.0])
        metrics = sim.run_sweep(ln_lambdas)

        bias2 = metrics['bias2']
        var = metrics['variance']
        sum_bv = metrics['bias2_plus_variance']
        test_err = metrics['test_error']

        # Bias squared must increase with lambda
        assert bias2[-1] > bias2[0]

        # Variance must decrease with lambda
        assert var[0] > var[-1]

        # Minimum of (bias)^2 + variance occurs in the interior (near 0.43)
        assert sum_bv[2] < sum_bv[0]
        assert sum_bv[2] < sum_bv[-1]

        # Test error tracks bias^2 + var + sigma^2 (0.09)
        for i in range(len(ln_lambdas)):
            theoretical_err = sum_bv[i] + 0.3 ** 2
            assert np.isclose(test_err[i], theoretical_err, atol=0.04)


class TestFigureGenerationCh4Sec3:
    """Test Figure 4.7 and Figure 4.8 generation and disk saving."""

    def test_figure_4_7(self, tmp_path):
        save_path = str(tmp_path / "fig4_7.png")
        sim = BiasVarianceSimulator(n_datasets=20, n_samples=25, random_state=42)
        fig, axes = plot_figure_4_7_bias_variance_ensembles(simulator=sim, save_paths=[save_path])
        assert fig is not None
        assert axes.shape == (3, 2)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 1000

    def test_figure_4_8(self, tmp_path):
        save_path = str(tmp_path / "fig4_8.png")
        sim = BiasVarianceSimulator(n_datasets=20, n_samples=25, random_state=42)
        fig, ax = plot_figure_4_8_bias_variance_tradeoff(simulator=sim, save_paths=[save_path])
        assert fig is not None
        assert ax is not None
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 1000
