"""
Tests for Chapter 14 Section 14.1: Basic Sampling Algorithms
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts
"""

import numpy as np
import scipy.stats as stats
import pytest
import matplotlib.pyplot as plt

from common.basic_sampling import (
    monte_carlo_expectation,
    convergence_analysis,
    inverse_cdf_sample_exponential,
    inverse_cdf_sample_cauchy,
    box_muller_transform,
    rejection_sample_unit_disk,
    RejectionSampler,
    rejection_sample_gamma,
    rejection_sample_gaussian_cauchy,
    AdaptiveRejectionSampler,
    compute_effective_sample_size,
    importance_sampling,
    sampling_importance_resampling,
    generate_figure_14_1,
    generate_figure_14_2,
    generate_figure_14_3,
    generate_figure_14_4,
    generate_figure_14_5,
    generate_figure_14_6,
    generate_figure_14_7,
    generate_figure_14_8,
    generate_all_section_14_1_figures,
)


class TestMonteCarloExpectations:
    def test_monte_carlo_unbiasedness(self):
        # E[z^2] for z ~ N(0, 1) is 1.0
        rng = np.random.RandomState(42)
        samples = rng.randn(10000)
        res = monte_carlo_expectation(lambda z: z ** 2, samples)
        assert abs(res["estimate"] - 1.0) < 3.0 * res["std_error"]
        assert res["ci_95"][0] <= 1.0 <= res["ci_95"][1]
        assert res["num_samples"] == 10000

    def test_convergence_analysis(self):
        rng = np.random.RandomState(42)
        samples = rng.exponential(scale=2.0, size=5000)
        # E[z] = 2.0
        running_mean, running_std_err, step_indices = convergence_analysis(lambda z: z, samples)
        assert len(running_mean) == 5000
        assert len(running_std_err) == 5000
        # Final running mean should be close to 2.0
        assert abs(running_mean[-1] - 2.0) < 0.1
        # Std error should decrease with 1/sqrt(L)
        assert running_std_err[-1] < running_std_err[50]


class TestStandardDistributions:
    def test_exponential_inversion(self):
        lam = 1.5
        samples = inverse_cdf_sample_exponential(lam, num_samples=10000, seed=42)
        assert len(samples) == 10000
        assert np.all(samples >= 0.0)
        # KS test against scipy.stats.expon(scale=1/lam)
        ks_res = stats.kstest(samples, "expon", args=(0, 1.0 / lam))
        assert ks_res.pvalue > 0.01

    def test_cauchy_inversion(self):
        x0, gamma = 1.0, 2.5
        samples = inverse_cdf_sample_cauchy(x0, gamma, num_samples=10000, seed=42)
        assert len(samples) == 10000
        # Check median and IQR
        median = np.median(samples)
        q25, q75 = np.percentile(samples, [25, 75])
        iqr = q75 - q25
        assert abs(median - x0) < 0.2
        assert abs(iqr - 2.0 * gamma) < 0.35

    def test_box_muller_transform(self):
        z1, z2 = box_muller_transform(num_pairs=10000, seed=42)
        assert len(z1) == 10000
        assert len(z2) == 10000
        # Mean 0, Variance 1
        assert abs(np.mean(z1)) < 0.05
        assert abs(np.var(z1) - 1.0) < 0.05
        assert abs(np.mean(z2)) < 0.05
        assert abs(np.var(z2) - 1.0) < 0.05
        # Independence: correlation close to 0
        corr = np.corrcoef(z1, z2)[0, 1]
        assert abs(corr) < 0.03
        # KS test
        assert stats.kstest(z1, "norm").pvalue > 0.01
        assert stats.kstest(z2, "norm").pvalue > 0.01

    def test_rejection_sample_unit_disk(self):
        accepted, acc_rate, proposals, mask = rejection_sample_unit_disk(num_samples=2000, seed=42)
        assert len(accepted) == 2000
        # Check all points inside unit disk
        radii_sq = np.sum(accepted ** 2, axis=1)
        assert np.all(radii_sq <= 1.0)
        # Check acceptance rate near pi / 4 (~0.7854)
        theoretical_rate = np.pi / 4.0
        assert abs(acc_rate - theoretical_rate) < 0.03


class TestRejectionSampling:
    def test_general_rejection_sampler(self):
        # Target: Beta(2, 3) on [0, 1], Proposal: Uniform(0, 1)
        target_p = lambda z: stats.beta.pdf(z, a=2, b=3)
        proposal_q = lambda z: np.ones_like(z)
        sample_q = lambda n, rng: rng.uniform(0.0, 1.0, size=n)
        # Mode of Beta(2, 3) is (2-1)/(2+3-2) = 1/3, pdf(1/3) = 1.7778
        k = 1.85
        sampler = RejectionSampler(target_p, proposal_q, sample_q, k)
        res = sampler.sample(num_samples=3000, seed=42)
        assert len(res["samples"]) == 3000
        assert np.all((res["samples"] >= 0.0) & (res["samples"] <= 1.0))
        # KS test
        ks_res = stats.kstest(res["samples"], "beta", args=(2, 3))
        assert ks_res.pvalue > 0.01
        # Empirical acceptance rate near 1 / k
        assert abs(res["acceptance_rate"] - (1.0 / k)) < 0.04

    def test_gamma_rejection_sampling(self):
        res = rejection_sample_gamma(a=10.0, b=1.0, num_samples=3000, seed=42)
        assert len(res["samples"]) == 3000
        assert np.all(res["samples"] > 0)
        # KS test against Gamma(10, scale=1)
        ks_res = stats.kstest(res["samples"], "gamma", args=(10.0, 0, 1.0))
        assert ks_res.pvalue > 0.01

    def test_gaussian_cauchy_rejection(self):
        res = rejection_sample_gaussian_cauchy(num_samples=4000, seed=42)
        assert len(res["samples"]) == 4000
        # KS test against standard Gaussian
        ks_res = stats.kstest(res["samples"], "norm")
        assert ks_res.pvalue > 0.01
        # Acceptance rate close to sqrt(2 / pi) =~ 0.7979
        assert abs(res["acceptance_rate"] - res["theoretical_acc_rate"]) < 0.03


class TestAdaptiveRejectionSampling:
    def test_adaptive_rejection_sampler(self):
        # Target: standard normal (log-concave)
        log_p = lambda z: -0.5 * z ** 2
        d_log_p = lambda z: -z
        ars = AdaptiveRejectionSampler(log_p, d_log_p, initial_points=[-1.5, 0.0, 1.5])

        # Test envelope dominates log_p
        z_grid = np.linspace(-3.0, 3.0, 100)
        env_vals = ars.evaluate_envelope(z_grid)
        log_p_vals = log_p(z_grid)
        assert np.all(env_vals >= log_p_vals - 1e-10)

        # Draw samples
        samples = ars.sample(num_samples=1000, seed=42)
        assert len(samples) == 1000
        assert abs(np.mean(samples)) < 0.15


class TestImportanceSampling:
    def test_effective_sample_size(self):
        # Equal weights: ESS = N
        w_equal = np.ones(100)
        assert np.isclose(compute_effective_sample_size(w_equal), 100.0)

        # One dominant weight: ESS close to 1
        w_one = np.zeros(100)
        w_one[0] = 100.0
        assert np.isclose(compute_effective_sample_size(w_one), 1.0)

    def test_importance_sampling_gaussian_mean(self):
        # Target p: N(2.0, 1.0), Proposal q: N(0.0, 2.0^2), Integrand f(z) = z
        target_p = lambda z: stats.norm.pdf(z, loc=2.0, scale=1.0)
        proposal_q = lambda z: stats.norm.pdf(z, loc=0.0, scale=2.0)
        sample_q = lambda n, rng: rng.randn(n) * 2.0

        res = importance_sampling(
            f=lambda z: z,
            target_p=target_p,
            proposal_q=proposal_q,
            sample_q=sample_q,
            num_samples=10000,
            is_normalized=True,
            seed=42,
        )
        assert abs(res["estimate"] - 2.0) < 0.1
        assert res["ess"] > 1000.0

    def test_importance_sampling_ratio_estimator(self):
        # Unnormalized target: p_tilde(z) = 3 * N(1.5, 0.8^2)
        # Proposal q: N(1.0, 1.5^2)
        target_p = lambda z: 3.0 * stats.norm.pdf(z, loc=1.5, scale=0.8)
        proposal_q = lambda z: stats.norm.pdf(z, loc=1.0, scale=1.5)
        sample_q = lambda n, rng: rng.randn(n) * 1.5 + 1.0

        res = importance_sampling(
            f=lambda z: z ** 2,
            target_p=target_p,
            proposal_q=proposal_q,
            sample_q=sample_q,
            num_samples=10000,
            is_normalized=False,
            seed=42,
        )
        # Theoretical E[z^2] = Var + Mean^2 = 0.8^2 + 1.5^2 = 0.64 + 2.25 = 2.89
        assert abs(res["estimate"] - 2.89) < 0.15


class TestSamplingImportanceResampling:
    def test_sir_gaussian_mixture(self):
        # Target: mixture of Gaussians 0.5 N(-2, 0.5^2) + 0.5 N(2, 0.5^2)
        target_p = lambda z: 0.5 * stats.norm.pdf(z, -2.0, 0.5) + 0.5 * stats.norm.pdf(z, 2.0, 0.5)
        proposal_q = lambda z: stats.norm.pdf(z, 0.0, 2.5)
        sample_q = lambda n, rng: rng.randn(n) * 2.5

        res = sampling_importance_resampling(
            target_p=target_p,
            proposal_q=proposal_q,
            sample_q=sample_q,
            num_candidates=10000,
            num_resamples=2000,
            seed=42,
        )
        resamples = res["resamples"]
        assert len(resamples) == 2000
        # Check that resamples form two distinct clusters near -2 and +2
        prop_left = np.mean(resamples < 0)
        assert abs(prop_left - 0.5) < 0.06
        assert res["num_unique_resamples"] > 500


class TestFigureGenerators:
    def test_figures_generation(self):
        figs = generate_all_section_14_1_figures()
        assert len(figs) == 8
        for name, fig in figs.items():
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
