"""Tests for Chapter 20 Section 20.1: Forward Encoder and Diffusion Kernels."""

import os
import pytest
import numpy as np
from common.forward_encoder import (
    NoiseSchedule,
    LinearNoiseSchedule,
    CosineNoiseSchedule,
    ForwardDiffusionEncoder,
    evaluate_true_reverse_distribution,
    generate_figure_20_1,
    generate_figure_20_2,
    generate_figure_20_3,
    generate_figure_20_4,
    generate_all_section_20_1_figures,
)


class TestNoiseSchedules:
    """Test suite for linear and cosine diffusion noise schedules."""

    def test_linear_schedule_properties(self):
        T = 100
        beta_min = 1e-4
        beta_max = 0.02
        schedule = LinearNoiseSchedule(T=T, beta_min=beta_min, beta_max=beta_max)

        assert schedule.T == T
        assert len(schedule.betas) == T + 1
        assert schedule.betas[0] == 0.0
        assert schedule.betas[1] == pytest.approx(beta_min, abs=1e-6)
        assert schedule.betas[T] == pytest.approx(beta_max, abs=1e-6)

        # Monotonicity
        assert np.all(np.diff(schedule.betas[1:]) >= 0.0)
        assert np.all(np.diff(schedule.alphas_cumprod[1:]) <= 0.0)

        # Check square roots identity: sqrt(alpha_bar)^2 + sqrt(1 - alpha_bar)^2 = 1
        for t in range(1, T + 1):
            s1 = schedule.sqrt_alphas_cumprod[t]
            s2 = schedule.sqrt_one_minus_alphas_cumprod[t]
            assert (s1 ** 2 + s2 ** 2) == pytest.approx(1.0, abs=1e-10)

        # SNR decreases monotonically
        assert np.all(np.diff(schedule.snr[1:]) <= 0.0)

    def test_cosine_schedule_properties(self):
        T = 100
        schedule = CosineNoiseSchedule(T=T, s=0.008)

        assert schedule.T == T
        assert schedule.alphas_cumprod[0] == pytest.approx(1.0, abs=1e-6)
        assert schedule.alphas_cumprod[T] < 0.01
        assert np.all(schedule.betas[1:] > 0.0)
        assert np.all(schedule.betas[1:] < 1.0)


class TestForwardDiffusionEncoder:
    """Test suite for ForwardDiffusionEncoder forward Markov chain and direct sampling."""

    def test_single_step_diffusion(self):
        T = 50
        schedule = LinearNoiseSchedule(T=T)
        encoder = ForwardDiffusionEncoder(schedule=schedule)

        z_prev = np.array([2.0, -1.0])
        t = 10
        beta_t = schedule.betas[t]

        # Generate large batch of single steps to verify empirical mean and variance
        N = 30000
        rng = np.random.RandomState(42)
        noise = rng.randn(N, 2)
        steps = encoder.step(z_prev, t=t, noise=noise)

        expected_mean = np.sqrt(1.0 - beta_t) * z_prev
        expected_var = beta_t

        assert np.allclose(np.mean(steps, axis=0), expected_mean, atol=0.03)
        assert np.allclose(np.var(steps, axis=0), expected_var, atol=0.01)

    def test_trajectory_simulation(self):
        T = 20
        schedule = LinearNoiseSchedule(T=T)
        encoder = ForwardDiffusionEncoder(schedule=schedule)

        x = np.array([1.5, -2.0, 0.5])
        traj = encoder.sample_trajectory(x, random_state=42)

        assert traj.shape == (T + 1, 3)
        assert np.allclose(traj[0], x)

    def test_marginal_shortcut_equivalence(self):
        """Verify Eq. 20.5 - 20.6: direct marginal distribution matches simulated Markov chain."""
        T = 50
        schedule = LinearNoiseSchedule(T=T)
        encoder = ForwardDiffusionEncoder(schedule=schedule)

        x = np.array([3.0])
        t = 25
        alpha_bar_t = schedule.alphas_cumprod[t]

        N = 40000
        rng = np.random.RandomState(42)
        noise = rng.randn(N, 1)

        marginal_samples = encoder.sample_marginal(x, t=t, noise=noise)

        expected_mean = np.sqrt(alpha_bar_t) * x
        expected_var = 1.0 - alpha_bar_t

        emp_mean = np.mean(marginal_samples)
        emp_var = np.var(marginal_samples)

        assert abs(emp_mean - expected_mean[0]) < 0.05
        assert abs(emp_var - expected_var) < 0.05

    def test_forward_posterior_parameters_and_bayes_identity(self):
        """Verify Eq. 20.7 - 20.9: tractable posterior parameters q(z_{t-1} | z_t, x)."""
        T = 100
        schedule = LinearNoiseSchedule(T=T)
        encoder = ForwardDiffusionEncoder(schedule=schedule)

        x = np.array([1.0, -1.0])
        z_t = np.array([0.8, -0.5])
        t = 30

        mu_tilde, beta_tilde = encoder.posterior_parameters(z_t, x, t=t)

        # Theoretical coefficients
        alpha_bar_t = schedule.alphas_cumprod[t]
        alpha_bar_prev = schedule.alphas_cumprod[t - 1]
        alpha_t = schedule.alphas[t]
        beta_t = schedule.betas[t]

        c_x = np.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
        c_z = np.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        expected_beta_tilde = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t

        assert np.allclose(mu_tilde, c_x * x + c_z * z_t, atol=1e-12)
        assert beta_tilde == pytest.approx(expected_beta_tilde, abs=1e-12)

        # Boundary condition at t=1: posterior must collapse to deterministic x with variance 0
        mu_1, beta_1 = encoder.posterior_parameters(z_t, x, t=1)
        assert np.allclose(mu_1, x, atol=1e-12)
        assert beta_1 == pytest.approx(0.0, abs=1e-12)


class TestReverseDistributionAnalysis:
    """Test suite for analytical reverse step distribution under large vs small step sizes (Figures 20.3, 20.4)."""

    def test_multimodal_reverse_step(self):
        grid = np.linspace(-5, 5, 500)
        pi = np.array([0.5, 0.5])
        mus = np.array([-2.5, 2.5])
        sigmas = np.array([0.4, 0.4])

        # Large step beta_t = 0.8
        q_rev_large, q_prior = evaluate_true_reverse_distribution(0.0, 0.8, pi, mus, sigmas, grid)
        # Distribution should have two peaks (one near -2.5, one near +2.5)
        peaks = (q_rev_large[1:-1] > q_rev_large[:-2]) & (q_rev_large[1:-1] > q_rev_large[2:])
        num_peaks = np.sum(peaks)
        assert num_peaks >= 2

        # Small step beta_t = 0.01 at z_t = 2.4
        q_rev_small, _ = evaluate_true_reverse_distribution(2.4, 0.01, pi, mus, sigmas, grid)
        peaks_small = (q_rev_small[1:-1] > q_rev_small[:-2]) & (q_rev_small[1:-1] > q_rev_small[2:])
        num_peaks_small = np.sum(peaks_small)
        # Should be unimodal around +2.5
        assert num_peaks_small == 1


class TestSection201Figures:
    """Test suite for Section 20.1 figure generation (Figures 20.1 〜 20.4)."""

    def test_all_section_20_1_figures_generate(self, tmp_path):
        figs = generate_all_section_20_1_figures()
        assert len(figs) == 4
        assert "fig_20_1.png" in figs
        assert "fig_20_2.png" in figs
        assert "fig_20_3.png" in figs
        assert "fig_20_4.png" in figs

        # Verify files created in global result and chapter 20 result
        base_dir = os.path.dirname(os.path.dirname(__file__))
        for fname in ["fig_20_1.png", "fig_20_2.png", "fig_20_3.png", "fig_20_4.png"]:
            p1 = os.path.join(base_dir, "20", "result", fname)
            p2 = os.path.join(base_dir, "result", fname)
            assert os.path.exists(p1), f"Missing {p1}"
            assert os.path.exists(p2), f"Missing {p2}"
            assert os.path.getsize(p1) > 1000
            assert os.path.getsize(p2) > 1000
