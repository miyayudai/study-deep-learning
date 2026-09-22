"""
Unit tests for Chapter 4, Section 4.2: Decision Theory.
Tests:
- Minkowski loss function calculations and properties (Eq 4.40)
- Analytical decomposition of expected squared loss into model error and irreducible noise (Eq 4.39)
- Minimizer of Minkowski loss: mean (q=2), median (q=1), mode (q->0)
- Multimodal conditional distribution where conditional mean has near-zero density
- Faithful generation of Figures 4.5 and 4.6
"""
import os
import pytest
import numpy as np
import scipy.stats as stats

from common.decision_theory import (
    minkowski_loss,
    expected_squared_loss_decomposition,
    optimal_minkowski_point_prediction,
    plot_figure_4_5_regression_function,
    plot_figure_4_6_minkowski_loss
)


class TestLossFunctions:
    """Tests for Minkowski loss function properties (Section 4.2, Eq 4.40)."""

    def test_minkowski_loss_evaluations(self):
        diff = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        # q = 2: squared loss
        assert np.allclose(minkowski_loss(diff, q=2.0), [4.0, 1.0, 0.0, 1.0, 4.0])
        # q = 1: absolute loss
        assert np.allclose(minkowski_loss(diff, q=1.0), [2.0, 1.0, 0.0, 1.0, 2.0])
        # q = 0.5
        assert np.allclose(minkowski_loss(diff, q=0.5), [np.sqrt(2.0), 1.0, 0.0, 1.0, np.sqrt(2.0)])

    def test_minkowski_loss_symmetry_and_nonnegativity(self):
        d = np.linspace(-3, 3, 50)
        for q in [0.3, 1.0, 2.0, 5.0]:
            loss = minkowski_loss(d, q=q)
            assert np.all(loss >= 0.0)
            assert np.isclose(minkowski_loss(0.0, q=q), 0.0)
            assert np.allclose(loss, minkowski_loss(-d, q=q))

    def test_invalid_q(self):
        with pytest.raises(ValueError):
            minkowski_loss(1.0, q=-1.0)
        with pytest.raises(ValueError):
            minkowski_loss(1.0, q=0.0)


class TestExpectedSquaredLossDecomposition:
    """Tests for Eq 4.39: E[L] = Model Error + Irreducible Noise."""

    def test_optimal_predictor_eliminates_model_error(self):
        # Known model: E[t|x] = 1.5 * x^2 - 0.5, var[t|x] = 0.04
        # p(x) = Uniform(-1, 1), density = 0.5
        cond_mean = lambda x: 1.5 * (x ** 2) - 0.5
        cond_var = lambda x: np.full_like(x, 0.04)
        p_x = lambda x: np.full_like(x, 0.5)

        # Optimal predictor f*(x) = E[t|x]
        f_opt = lambda x: cond_mean(x)
        decomp_opt = expected_squared_loss_decomposition(
            f_opt, cond_mean, cond_var, p_x, x_range=(-1.0, 1.0)
        )

        assert np.isclose(decomp_opt["model_error"], 0.0, atol=1e-8)
        assert np.isclose(decomp_opt["irreducible_noise"], 0.04, atol=1e-6)
        assert np.isclose(decomp_opt["expected_loss"], 0.04, atol=1e-6)

    def test_suboptimal_predictor_adds_exact_model_error(self):
        cond_mean = lambda x: 2.0 * x
        cond_var = lambda x: np.full_like(x, 0.09)
        p_x = lambda x: np.full_like(x, 0.5) # Uniform(-1, 1)

        # Suboptimal predictor f(x) = E[t|x] + 0.3
        bias_offset = 0.3
        f_sub = lambda x: cond_mean(x) + bias_offset
        decomp = expected_squared_loss_decomposition(
            f_sub, cond_mean, cond_var, p_x, x_range=(-1.0, 1.0)
        )

        # Model error = \int_{-1}^1 (0.3)^2 * 0.5 dx = 0.09
        expected_model_err = bias_offset ** 2
        assert np.isclose(decomp["model_error"], expected_model_err, atol=1e-6)
        assert np.isclose(decomp["irreducible_noise"], 0.09, atol=1e-6)
        assert np.isclose(decomp["expected_loss"], expected_model_err + 0.09, atol=1e-6)


class TestOptimalMinkowskiPredictors:
    """Tests that q=2 minimizes mean, q=1 minimizes median, and small q approaches mode."""

    def test_asymmetric_distribution_mean_median_mode(self):
        np.random.seed(42)
        # Log-normal distribution: strongly right-skewed
        # s=0.75, scale=exp(0)=1
        samples = np.random.lognormal(mean=0.0, sigma=0.75, size=20000)

        pred_q2 = optimal_minkowski_point_prediction(samples, q=2.0)
        pred_q1 = optimal_minkowski_point_prediction(samples, q=1.0)
        pred_q_small = optimal_minkowski_point_prediction(samples, q=0.1)

        sample_mean = np.mean(samples)
        sample_median = np.median(samples)

        # 1. q=2 is exact sample mean
        assert np.isclose(pred_q2, sample_mean, atol=1e-8)

        # 2. q=1 is exact sample median
        assert np.isclose(pred_q1, sample_median, atol=1e-8)

        # 3. For right-skewed lognormal distribution: Mode < Median < Mean
        # Mode = exp(mu - sigma^2) = exp(-0.75^2) = exp(-0.5625) ~ 0.57
        # Median = exp(0) = 1.0
        # Mean = exp(mu + sigma^2/2) = exp(0.28125) ~ 1.325
        assert pred_q_small < pred_q1 < pred_q2
        assert np.isclose(sample_median, 1.0, atol=0.05)
        assert pred_q_small < 0.85 # significantly closer to mode ~0.57 than median 1.0


class TestMultimodalDistribution:
    """Tests failure mode of squared loss when conditional distribution is multimodal."""

    def test_conditional_mean_in_low_density_valley(self):
        # Mixture of two distant Gaussians:
        # p(t|x) = 0.5 * N(t | -3.0, 0.3^2) + 0.5 * N(t | 3.0, 0.3^2)
        # Conditional mean is E[t|x] = 0.0
        mu1, mu2 = -3.0, 3.0
        sig = 0.3

        cond_mean = 0.5 * mu1 + 0.5 * mu2 # 0.0
        assert np.isclose(cond_mean, 0.0)

        # Density at the conditional mean t=0
        density_at_mean = 0.5 * stats.norm.pdf(cond_mean, mu1, sig) + 0.5 * stats.norm.pdf(cond_mean, mu2, sig)
        # Peak density at t=3.0
        density_at_mode = 0.5 * stats.norm.pdf(mu2, mu2, sig)

        # The density at the conditional mean is exponentially small (negligible)
        assert density_at_mean < 1e-15
        assert density_at_mode > 0.5
        # This demonstrates why squared loss prediction can be disastrous for multimodal targets.


class TestFigureGenerationCh4Sec2:
    """Tests for Figure 4.5 and Figure 4.6 generation."""

    def test_figures_4_5_and_4_6(self, tmp_path):
        f5, ax5 = plot_figure_4_5_regression_function(
            save_paths=[str(tmp_path / 'fig4_5.png')]
        )
        assert f5 is not None
        assert os.path.exists(tmp_path / 'fig4_5.png')

        f6, axes6 = plot_figure_4_6_minkowski_loss(
            save_paths=[str(tmp_path / 'fig4_6.png')]
        )
        assert f6 is not None
        assert os.path.exists(tmp_path / 'fig4_6.png')
        assert axes6.shape == (2, 2)


class TestAnalyticalDecomposition:
    """Tests for expected_squared_loss_decomposition function."""

    def test_analytical_decomposition(self):
        from common.decision_theory import expected_squared_loss_decomposition

        # Let true mean be h(x) = x^2, true var be 0.1, p(x) uniform on [0, 1]
        f_func = lambda x: x  # suboptimal model f(x) = x
        cond_mean_func = lambda x: x ** 2
        cond_var_func = lambda x: np.full_like(x, 0.1)
        p_x_func = lambda x: np.ones_like(x)

        decomp = expected_squared_loss_decomposition(
            f_func=f_func,
            cond_mean_func=cond_mean_func,
            cond_var_func=cond_var_func,
            p_x_func=p_x_func,
            x_range=(0.0, 1.0)
        )

        # int_0^1 (x - x^2)^2 dx = int_0^1 (x^2 - 2x^3 + x^4) dx = 1/3 - 2/4 + 1/5 = 1/30 approx 0.033333
        expected_model_err = 1.0 / 30.0
        expected_noise = 0.1

        np.testing.assert_allclose(decomp["model_error"], expected_model_err, atol=1e-5)
        np.testing.assert_allclose(decomp["irreducible_noise"], expected_noise, atol=1e-5)
        np.testing.assert_allclose(decomp["expected_loss"], expected_model_err + expected_noise, atol=1e-5)


class TestOptimalMinkowskiPointPrediction:
    """Tests for optimal_minkowski_point_prediction."""

    def test_point_predictions(self):
        from common.decision_theory import optimal_minkowski_point_prediction

        t_samples = np.array([1.0, 2.0, 2.5, 3.0, 10.0])
        pred_mean = optimal_minkowski_point_prediction(t_samples, q=2.0)
        assert np.isclose(pred_mean, np.mean(t_samples))

        pred_median = optimal_minkowski_point_prediction(t_samples, q=1.0)
        assert np.isclose(pred_median, np.median(t_samples))

        pred_small_q = optimal_minkowski_point_prediction(t_samples, q=0.2)
        assert 1.0 <= pred_small_q <= 10.0
