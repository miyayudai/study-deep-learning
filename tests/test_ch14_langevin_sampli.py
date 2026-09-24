"""
Unit tests for Chapter 14 Section 14.3: Langevin Sampling.
Covers:
- 14.3.1 Energy-based models (EBMs, score functions, partition function)
- 14.3.2 Maximizing the likelihood (Positive & negative phases of EBM gradient)
- 14.3.3 Langevin dynamics (ULA, MALA, score-based sampling, annealed Langevin)
- Figures 14.13 and 14.14 generation
"""

import os
import numpy as np
import pytest

from common.langevin_sampling import (
    EnergyBasedModel1D,
    unadjusted_langevin_algorithm,
    metropolis_adjusted_langevin_algorithm,
    annealed_langevin_dynamics,
    generate_figure_14_13,
    generate_figure_14_14,
)


def test_ebm_1d_energy_score_and_partition_function():
    """Test 1D EBM on quadratic energy E(x; w) = 0.5 * w * x^2."""
    w = np.array([2.0])  # precision parameter (variance = 1/2)

    def energy_fn(x, param):
        return 0.5 * param[0] * (x ** 2)

    def grad_x_energy(x, param):
        return param[0] * x

    def grad_w_energy(x, param):
        return 0.5 * (x ** 2).reshape(-1, 1)

    ebm = EnergyBasedModel1D(
        energy_fn=energy_fn,
        grad_x_energy_fn=grad_x_energy,
        grad_w_energy_fn=grad_w_energy,
    )

    x_test = np.array([1.5, -2.0, 0.0])
    # Score is -grad_x E = -w * x
    score_val = ebm.score(x_test, w)
    assert np.allclose(score_val, -w[0] * x_test)

    # Numerical partition function: int exp(-0.5 * w * x^2) dx = sqrt(2 * pi / w)
    z_computed = ebm.compute_partition_function(w)
    z_true = np.sqrt(2.0 * np.pi / w[0])
    assert np.isclose(z_computed, z_true, rtol=1e-3)


def test_ebm_log_likelihood_gradient_positive_negative_phases():
    """Test positive and negative phase decomposition of log-likelihood gradient."""
    # E(x; w) = 0.5 * w * x^2
    def energy_fn(x, param):
        return 0.5 * param[0] * (x ** 2)

    def grad_x_energy(x, param):
        return param[0] * x

    def grad_w_energy(x, param):
        return 0.5 * (x ** 2).reshape(-1, 1)

    ebm = EnergyBasedModel1D(
        energy_fn=energy_fn,
        grad_x_energy_fn=grad_x_energy,
        grad_w_energy_fn=grad_w_energy,
    )

    w = np.array([1.0])
    rng = np.random.RandomState(42)
    # True data from Gaussian with var = 1.0 (matches w = 1.0)
    data = rng.randn(10000)
    # Model samples from model distribution (also var = 1.0)
    model = rng.randn(10000)

    grad_dict = ebm.compute_log_likelihood_gradient(data, model, w)
    total_grad = grad_dict["total_grad"]
    pos_grad = grad_dict["positive_phase"]
    neg_grad = grad_dict["negative_phase"]

    # At equilibrium, positive phase and negative phase cancel:
    # E_D[0.5 x^2] = 0.5, E_M[0.5 x^2] = 0.5 -> total_grad \approx 0
    assert np.isclose(pos_grad[0], -0.5, atol=0.03)
    assert np.isclose(neg_grad[0], +0.5, atol=0.03)
    assert np.abs(total_grad[0]) < 0.05


def test_unadjusted_langevin_algorithm_ula():
    """Test ULA on standard Gaussian target."""
    # Target: N(0, 1) -> score s(z) = -z
    score_fn = lambda z: -z

    res = unadjusted_langevin_algorithm(
        score_fn=score_fn,
        init_state=np.array([4.0]),  # Start away from mode
        step_size=0.05,
        num_steps=12000,
        burn_in=2000,
        seed=101,
    )

    samples = res["samples"].flatten()
    assert np.abs(np.mean(samples)) < 0.15
    # Variance should be close to 1.0 (with slight discretization bias O(step_size))
    assert np.abs(np.var(samples) - 1.0) < 0.15
    assert len(res["trajectory"]) == 12001


def test_metropolis_adjusted_langevin_algorithm_mala():
    """Test MALA on standard Gaussian target."""
    target_log_p = lambda z: -0.5 * np.sum(z ** 2)
    score_fn = lambda z: -z

    res = metropolis_adjusted_langevin_algorithm(
        target_log_p=target_log_p,
        score_fn=score_fn,
        init_state=np.array([3.0]),
        step_size=0.25,
        num_steps=8000,
        burn_in=1000,
        seed=777,
    )

    samples = res["samples"].flatten()
    acc_rate = res["acceptance_rate"]

    # MALA acceptance rate with step_size=0.25 on 1D Gaussian is very high (> 0.7)
    assert 0.6 < acc_rate <= 1.0
    assert np.abs(np.mean(samples)) < 0.10
    assert np.abs(np.var(samples) - 1.0) < 0.12
    assert len(res["accepted_steps"]) + len(res["rejected_proposals"]) == 8000


def test_annealed_langevin_dynamics_bridges_modes():
    """Test that annealed Langevin dynamics can sample across isolated modes."""
    # Target: Mixture of two Gaussians at -3 and +3
    # With noise sigma, p_sigma(x) has smoothed scores
    sigmas = [6.0, 3.0, 1.5, 0.8, 0.2]

    def make_score_fn(sigma):
        def score(x):
            # p_sigma(x) \propto 0.5 * N(x | -3, 0.2^2 + sigma^2) + 0.5 * N(x | +3, 0.2^2 + sigma^2)
            var = 0.04 + sigma ** 2
            g1 = np.exp(-0.5 * ((x - 3.0) ** 2) / var)
            g2 = np.exp(-0.5 * ((x + 3.0) ** 2) / var)
            score_val = (g1 * (-(x - 3.0) / var) + g2 * (-(x + 3.0) / var)) / (g1 + g2 + 1e-12)
            return score_val
        return score

    score_fns = [make_score_fn(s) for s in sigmas]

    # Run annealed Langevin starting at x = 0 (the barrier between modes)
    res = annealed_langevin_dynamics(
        score_fns=score_fns,
        sigmas=sigmas,
        init_state=np.array([0.0]),
        num_steps_per_level=200,
        step_size_factor=0.01,
        seed=42,
    )

    traj = res["trajectory"].flatten()
    # At high noise, chain moves freely; at final level, it converges near +3 or -3
    assert len(traj) > 500
    final_x = res["final_sample"][0]
    assert np.abs(final_x - 3.0) < 1.0 or np.abs(final_x + 3.0) < 1.0


def test_figures_14_13_and_14_14_generation(tmp_path):
    """Test generation of Figures 14.13 and 14.14."""
    out_dir = str(tmp_path)

    fig13 = generate_figure_14_13(save_dir=out_dir)
    assert fig13 is not None
    assert os.path.exists(os.path.join(out_dir, "fig_14_13_energy_based_training.png"))

    fig14 = generate_figure_14_14(save_dir=out_dir)
    assert fig14 is not None
    assert os.path.exists(os.path.join(out_dir, "fig_14_14_multimodal_challenge.png"))
