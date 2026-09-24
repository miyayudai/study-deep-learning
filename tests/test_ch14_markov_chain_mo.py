"""
Unit tests for Chapter 14 Section 14.2: Markov Chain Monte Carlo (MCMC).
Covers:
- 14.2.1 The Metropolis Algorithm
- 14.2.2 Markov Chains and Detailed Balance
- 14.2.3 The Metropolis-Hastings Algorithm
- 14.2.4 Gibbs Sampling
- 14.2.5 Ancestral Sampling
- MCMC convergence diagnostics (Autocorrelation, Integrated Autocorrelation Time, ESS)
- Figures 14.9, 14.10, 14.11, 14.12 generation
"""

import os
import numpy as np
import pytest

from common.markov_chain_monte_carlo import (
    metropolis_sampler,
    metropolis_sample_2d_gaussian,
    compute_autocorrelation,
    compute_integrated_autocorrelation_time,
    compute_effective_sample_size_mcmc,
    verify_detailed_balance,
    MetropolisHastingsSampler,
    gibbs_sample_2d_gaussian,
    AncestralSampler,
    generate_figure_14_9,
    generate_figure_14_10,
    generate_figure_14_11,
    generate_figure_14_12,
)


def test_metropolis_sampler_1d_gaussian():
    """Test 1D Metropolis algorithm on standard normal distribution."""
    # Target log prob: standard normal ln p(z) = -0.5 * z^2
    def log_target(z):
        return -0.5 * np.sum(z ** 2)

    res = metropolis_sampler(
        target_log_p=log_target,
        proposal_std=1.0,
        num_samples=10000,
        init_state=np.array([5.0]),  # Start away from mode
        burn_in=2000,
        seed=42,
    )

    samples = res["samples"]
    acc_rate = res["acceptance_rate"]

    assert 0.2 < acc_rate < 0.85
    assert np.abs(np.mean(samples)) < 0.1
    assert np.abs(np.var(samples) - 1.0) < 0.15


def test_metropolis_sample_2d_gaussian():
    """Test 2D Gaussian Metropolis sampling and step logging."""
    mean = np.array([1.0, -1.0])
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])

    res = metropolis_sample_2d_gaussian(
        mean=mean,
        cov=cov,
        num_steps=200,
        proposal_std=0.3,
        init_state=np.array([0.0, 0.0]),
        seed=123,
    )

    assert "trajectory" in res
    assert "accepted_steps" in res
    assert "rejected_proposals" in res
    assert len(res["trajectory"]) == 201
    assert len(res["accepted_steps"]) + len(res["rejected_proposals"]) == 200
    assert 0.0 < res["acceptance_rate"] < 1.0


def test_autocorrelation_and_ess():
    """Test autocorrelation and effective sample size calculations."""
    rng = np.random.RandomState(42)
    # AR(1) process with rho = 0.7: x_t = 0.7 x_{t-1} + e_t
    n = 5000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.randn() * np.sqrt(1 - 0.7 ** 2)

    acf = compute_autocorrelation(x, max_lag=30)
    assert len(acf) == 31
    assert np.isclose(acf[0], 1.0)
    # At lag 1, acf should be close to 0.7
    assert np.abs(acf[1] - 0.7) < 0.06

    tau_int = compute_integrated_autocorrelation_time(x, max_lag=30)
    # For AR(1): tau_int = (1 + rho) / (1 - rho) = 1.7 / 0.3 = 5.67
    assert 4.0 < tau_int < 7.5

    ess = compute_effective_sample_size_mcmc(x, max_lag=30)
    assert 500 < ess < 1500
    assert ess < n


def test_detailed_balance_verification():
    """Test detailed balance equation on symmetric Gaussian transition."""
    # Target p(z) = exp(-0.5 * z^2)
    def target_p(z):
        return float(np.exp(-0.5 * (z ** 2)))

    # Symmetric Metropolis transition kernel between z_a and z_b:
    # T(z_a -> z_b) = q(z_b | z_a) * min(1, p(z_b) / p(z_a))
    def transition_p(z1, z2):
        q = np.exp(-0.5 * (z2 - z1) ** 2)
        alpha = min(1.0, target_p(z2) / target_p(z1))
        return float(q * alpha)

    z1 = 0.5
    z2 = 1.2

    db_res = verify_detailed_balance(
        target_p=target_p,
        transition_p=transition_p,
        z_a=z1,
        z_b=z2,
    )
    assert db_res["is_balanced"]
    assert np.isclose(db_res["flux_ab"], db_res["flux_ba"], atol=1e-10)


def test_metropolis_hastings_asymmetric():
    """Test Metropolis-Hastings with asymmetric proposal distribution."""
    # Target: Gamma distribution Gam(alpha=3, beta=1) on z > 0
    # p(z) \propto z^(alpha-1) exp(-z)
    def target_log_p(z):
        if z[0] <= 0:
            return -np.inf
        return 2.0 * np.log(z[0]) - z[0]

    # Asymmetric proposal: Log-normal proposal ln(z') ~ N(ln(z), sigma^2)
    # q(z' | z) = 1/(z' * sqrt(2*pi)*sigma) * exp(-(ln z' - ln z)^2 / (2 sigma^2))
    sigma = 0.5
    def proposal_sample(z, rng):
        return np.array([np.exp(np.log(z[0]) + rng.randn() * sigma)])

    def proposal_log_q(z_prime, z):
        if z_prime[0] <= 0:
            return -np.inf
        diff = np.log(z_prime[0]) - np.log(z[0])
        return -np.log(z_prime[0]) - 0.5 * (diff / sigma) ** 2

    sampler = MetropolisHastingsSampler(
        target_log_p=target_log_p,
        proposal_sample=proposal_sample,
        proposal_log_q=proposal_log_q,
    )

    res = sampler.sample(
        num_samples=10000,
        init_state=np.array([1.0]),
        burn_in=2000,
        seed=101,
    )

    samples = res["samples"].flatten()
    # True mean of Gam(3, 1) is alpha / beta = 3.0, variance is 3.0
    assert np.abs(np.mean(samples) - 3.0) < 0.25
    assert np.abs(np.var(samples) - 3.0) < 0.55
    assert 0.3 < res["acceptance_rate"] < 0.8


def test_gibbs_sampling_2d_gaussian():
    """Test Gibbs sampling on correlated 2D Gaussian."""
    mean = np.array([2.0, -1.0])
    cov = np.array([[1.5, 0.8], [0.8, 1.2]])

    res = gibbs_sample_2d_gaussian(
        mean=mean,
        cov=cov,
        num_steps=8000,
        init_state=np.array([0.0, 0.0]),
        seed=42,
    )

    samples = res["samples"][1000:]  # Discard burn-in
    sample_mean = np.mean(samples, axis=0)
    sample_cov = np.cov(samples, rowvar=False)

    assert np.allclose(sample_mean, mean, atol=0.1)
    assert np.allclose(sample_cov, cov, atol=0.15)
    assert "trajectory" in res
    assert len(res["trajectory"]) == 2 * 8000 + 1


def test_ancestral_sampler_dag():
    """Test AncestralSampler on a 3-node directed graphical model A -> B -> C."""
    # A ~ Bernoulli(p=0.6)
    # B | A ~ N(2 * A, 0.5^2)
    # C | B ~ N(B - 1, 0.3^2)
    def sample_a(parents, rng):
        return int(rng.rand() < 0.6)

    def sample_b(parents, rng):
        a = parents["A"]
        return float(2.0 * a + rng.randn() * 0.5)

    def sample_c(parents, rng):
        b = parents["B"]
        return float(b - 1.0 + rng.randn() * 0.3)

    sampler = AncestralSampler(
        topo_order=["A", "B", "C"],
        conditionals={
            "A": sample_a,
            "B": sample_b,
            "C": sample_c,
        },
    )

    sample_list = sampler.sample(num_samples=5000, seed=777)
    a_samples = np.array([s["A"] for s in sample_list])
    b_samples = np.array([s["B"] for s in sample_list])
    c_samples = np.array([s["C"] for s in sample_list])

    # E[A] = 0.6
    assert np.abs(np.mean(a_samples) - 0.6) < 0.03
    # E[B] = 2 * E[A] = 1.2
    assert np.abs(np.mean(b_samples) - 1.2) < 0.06
    # E[C] = E[B] - 1 = 0.2
    assert np.abs(np.mean(c_samples) - 0.2) < 0.06


def test_figures_generation(tmp_path):
    """Test generation of Figures 14.9, 14.10, 14.11, 14.12."""
    out_dir = str(tmp_path)

    fig9 = generate_figure_14_9(save_dir=out_dir)
    assert fig9 is not None
    assert os.path.exists(os.path.join(out_dir, "fig_14_9_metropolis_algorithm.png"))

    fig10 = generate_figure_14_10(save_dir=out_dir)
    assert fig10 is not None
    assert os.path.exists(os.path.join(out_dir, "fig_14_10_step_size_scaling.png"))

    fig11 = generate_figure_14_11(save_dir=out_dir)
    assert fig11 is not None
    assert os.path.exists(os.path.join(out_dir, "fig_14_11_diffusive_scaling.png"))

    fig12 = generate_figure_14_12(save_dir=out_dir)
    assert fig12 is not None
    assert os.path.exists(os.path.join(out_dir, "fig_14_12_markov_blanket.png"))
