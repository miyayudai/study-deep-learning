"""
Unit tests for Chapter 3 Section 3.1: Discrete Variables
Covers:
  - 3.1.1 Bernoulli distribution (Eq 3.1 - 3.8)
  - 3.1.2 Binomial distribution (Eq 3.9 - 3.12, Figure 3.1)
  - 3.1.3 Multinomial distribution (Eq 3.13 - 3.24)
"""
import os
import pytest
import numpy as np
from scipy import special

from common.probability import (
    BernoulliDistribution,
    BinomialDistribution,
    MultinomialDistribution,
    plot_figure_3_1
)


# ==============================================================================
# 3.1.1 Bernoulli Distribution Tests
# ==============================================================================

def test_bernoulli_pmf_and_moments():
    """Test Eq (3.1)-(3.4): pmf, normalization, mean, variance."""
    mu = 0.35
    bern = BernoulliDistribution(mu)

    # Eq (3.1), (3.2): p(x=1|mu) = mu, p(x=0|mu) = 1 - mu
    assert bern.pmf(1) == pytest.approx(mu)
    assert bern.pmf(0) == pytest.approx(1.0 - mu)
    assert bern.pmf(1) + bern.pmf(0) == pytest.approx(1.0)

    # Eq (3.3): E[x] = mu
    assert bern.mean == pytest.approx(mu)

    # Eq (3.4): var[x] = mu * (1 - mu)
    assert bern.variance == pytest.approx(mu * (1.0 - mu))

    # Array input
    x_arr = np.array([0, 1, 1, 0])
    expected_probs = np.array([1.0 - mu, mu, mu, 1.0 - mu])
    np.testing.assert_allclose(bern.pmf(x_arr), expected_probs)


def test_bernoulli_invalid_inputs():
    """Test boundary and error conditions for Bernoulli distribution."""
    # Invalid mu
    with pytest.raises(ValueError):
        BernoulliDistribution(-0.1)
    with pytest.raises(ValueError):
        BernoulliDistribution(1.05)

    bern = BernoulliDistribution(0.5)
    # Invalid x (support is strictly {0, 1})
    with pytest.raises(ValueError):
        bern.pmf(2)
    with pytest.raises(ValueError):
        bern.pmf(-1)


def test_bernoulli_log_likelihood_and_mle():
    """Test Eq (3.5)-(3.8): log-likelihood and MLE = m / N."""
    data = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])  # N = 10, m = 6
    N = len(data)
    m = np.sum(data)
    assert m == 6

    # Eq (3.7), (3.8): mu_ML = m / N = 0.6
    mu_ml = BernoulliDistribution.fit_mle(data)
    assert mu_ml == pytest.approx(0.6)

    # Log likelihood at mu = 0.6 should be maximized
    ll_ml = BernoulliDistribution.log_likelihood(data, mu_ml)
    ll_other = BernoulliDistribution.log_likelihood(data, 0.4)
    assert ll_ml > ll_other

    # Mathematical identity: d/d(mu) ln p(D|mu) = m / mu - (N - m) / (1 - mu) = 0 at mu_ml
    deriv = m / mu_ml - (N - m) / (1.0 - mu_ml)
    assert deriv == pytest.approx(0.0, abs=1e-12)


def test_bernoulli_sampling_convergence():
    """Test sample convergence to mean via Law of Large Numbers."""
    mu = 0.7
    bern = BernoulliDistribution(mu)
    samples = bern.sample(size=20000, seed=42)
    assert len(samples) == 20000
    sample_mean = np.mean(samples)
    assert sample_mean == pytest.approx(mu, abs=0.01)


# ==============================================================================
# 3.1.2 Binomial Distribution Tests
# ==============================================================================

def test_binomial_normalization_and_moments():
    """Test Eq (3.9)-(3.12): normalization, mean, and variance."""
    N = 15
    mu = 0.4
    binom = BinomialDistribution(N, mu)

    # Normalization: sum_{m=0}^N Bin(m | N, mu) = 1
    m_vals, probs = binom.pmf_all()
    assert len(m_vals) == N + 1
    assert np.sum(probs) == pytest.approx(1.0, abs=1e-12)

    # Eq (3.11): E[m] = N * mu
    assert binom.mean == pytest.approx(N * mu)
    expected_mean = np.sum(m_vals * probs)
    assert binom.mean == pytest.approx(expected_mean, abs=1e-12)

    # Eq (3.12): var[m] = N * mu * (1 - mu)
    assert binom.variance == pytest.approx(N * mu * (1.0 - mu))
    expected_var = np.sum((m_vals - binom.mean) ** 2 * probs)
    assert binom.variance == pytest.approx(expected_var, abs=1e-12)


def test_binomial_textbook_figure_3_1_values():
    """
    Test exact analytical values for Figure 3.1: N = 10, mu = 0.25.
    Bin(m | 10, 0.25) = (10 choose m) * 0.25^m * 0.75^(10-m).
    """
    N = 10
    mu = 0.25
    binom = BinomialDistribution(N, mu)
    m_vals, probs = binom.pmf_all()

    # Hand-calculated reference values
    # m = 0: 0.75^10 = 0.0563135...
    assert probs[0] == pytest.approx(0.75 ** 10, rel=1e-7)
    # m = 1: 10 * 0.25 * 0.75^9 = 0.1877117...
    assert probs[1] == pytest.approx(10 * 0.25 * (0.75 ** 9), rel=1e-7)
    # m = 2: 45 * 0.25^2 * 0.75^8 = 0.2815675... (Peak mode)
    assert probs[2] == pytest.approx(45 * (0.25 ** 2) * (0.75 ** 8), rel=1e-7)
    # m = 3: 120 * 0.25^3 * 0.75^7 = 0.2502822...
    assert probs[3] == pytest.approx(120 * (0.25 ** 3) * (0.75 ** 7), rel=1e-7)

    # The peak is indeed at m = 2
    assert np.argmax(probs) == 2
    # All probabilities are non-negative and sum to 1
    assert np.all(probs >= 0)
    assert np.sum(probs) == pytest.approx(1.0)


def test_binomial_edge_cases():
    """Test boundary conditions for Binomial: N=1 (Bernoulli equivalence), mu=0, mu=1."""
    # N=1 matches Bernoulli
    b1 = BinomialDistribution(N=1, mu=0.3)
    bern = BernoulliDistribution(mu=0.3)
    assert b1.pmf(0) == pytest.approx(bern.pmf(0))
    assert b1.pmf(1) == pytest.approx(bern.pmf(1))

    # mu=0: all mass on m=0
    b_zero = BinomialDistribution(N=5, mu=0.0)
    assert b_zero.pmf(0) == pytest.approx(1.0)
    assert b_zero.pmf(1) == pytest.approx(0.0)

    # mu=1: all mass on m=N
    b_one = BinomialDistribution(N=5, mu=1.0)
    assert b_one.pmf(5) == pytest.approx(1.0)
    assert b_one.pmf(4) == pytest.approx(0.0)


# ==============================================================================
# 3.1.3 Multinomial Distribution Tests
# ==============================================================================

def test_multinomial_categorical_scheme():
    """Test Eq (3.13)-(3.16): 1-of-K representation and expectations."""
    mu = [0.1, 0.2, 0.4, 0.3]
    cat = MultinomialDistribution(mu=mu, N=1)

    assert cat.K == 4
    np.testing.assert_allclose(cat.mean, np.array(mu))

    # Valid 1-of-K vectors (Eq 3.13)
    x1 = np.array([1, 0, 0, 0])
    x3 = np.array([0, 0, 1, 0])

    # Eq (3.14): p(x | mu) = prod mu_k^{x_k} = mu_k for active k
    assert cat.pmf_categorical(x1) == pytest.approx(0.1)
    assert cat.pmf_categorical(x3) == pytest.approx(0.4)

    # Normalization over all 4 mutually exclusive states (Eq 3.15)
    total_p = sum(cat.pmf_categorical(np.eye(4)[k]) for k in range(4))
    assert total_p == pytest.approx(1.0)

    # Expectation: sum_x p(x|mu) x = mu (Eq 3.16)
    expected_x = sum(cat.pmf_categorical(np.eye(4)[k]) * np.eye(4)[k] for k in range(4))
    np.testing.assert_allclose(expected_x, np.array(mu))


def test_multinomial_sufficient_statistics_and_mle():
    """Test Eq (3.17)-(3.22): sufficient statistics and MLE via Lagrange multipliers."""
    # Observations: N = 6 independent draws of 1-of-3 vectors
    # Say outcome 0 occurs 1 time, outcome 1 occurs 2 times, outcome 2 occurs 3 times
    data = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ])
    N = 6

    # Eq (3.18): sufficient statistics m_k = sum_{n=1}^N x_{nk}
    m = MultinomialDistribution.compute_sufficient_statistics(data)
    np.testing.assert_array_equal(m, np.array([1, 2, 3]))

    # Eq (3.19): sum m_k = N
    assert np.sum(m) == N

    # Eq (3.22): mu_k^ML = m_k / N
    mu_ml = MultinomialDistribution.fit_mle(data)
    expected_mu_ml = np.array([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0])
    np.testing.assert_allclose(mu_ml, expected_mu_ml)

    # fit_mle from counts directly
    mu_ml_from_counts = MultinomialDistribution.fit_mle(m, is_counts=True)
    np.testing.assert_allclose(mu_ml_from_counts, expected_mu_ml)


def test_multinomial_pmf_and_binomial_equivalence():
    """
    Test Eq (3.23), (3.24) and equivalence with Binomial distribution when K=2.
    """
    N = 8
    mu_bin = 0.3
    # Two states: state 0 has prob 0.3, state 1 has prob 0.7
    mult = MultinomialDistribution(mu=[mu_bin, 1.0 - mu_bin], N=N)
    binom = BinomialDistribution(N=N, mu=mu_bin)

    for m0 in range(N + 1):
        m1 = N - m0
        p_mult = mult.pmf(np.array([m0, m1]))
        p_bin = binom.pmf(m0)
        assert p_mult == pytest.approx(p_bin, rel=1e-7)


def test_multinomial_covariance():
    """Test covariance matrix of counts m."""
    N = 10
    mu = [0.2, 0.5, 0.3]
    mult = MultinomialDistribution(mu=mu, N=N)
    cov = mult.covariance
    # Diagonal elements: var[m_k] = N * mu_k * (1 - mu_k)
    assert cov[0, 0] == pytest.approx(10 * 0.2 * 0.8)
    assert cov[1, 1] == pytest.approx(10 * 0.5 * 0.5)
    assert cov[2, 2] == pytest.approx(10 * 0.3 * 0.7)
    # Off-diagonal elements: cov[m_i, m_j] = - N * mu_i * mu_j
    assert cov[0, 1] == pytest.approx(-10 * 0.2 * 0.5)
    assert cov[1, 2] == pytest.approx(-10 * 0.5 * 0.3)


# ==============================================================================
# Figure 3.1 Verification Test
# ==============================================================================

def test_figure_3_1_reproduction_file(tmp_path):
    """Verify Figure 3.1 plotting function and output file creation."""
    test_out = str(tmp_path / "fig3_01_test.png")
    fig, ax = plot_figure_3_1(save_paths=[test_out], show=False)

    assert os.path.exists(test_out)
    assert os.path.getsize(test_out) > 5000  # File should be non-empty valid PNG

    # Also check that the canonical result files exist
    assert os.path.exists("3/result/fig3_01_binomial_distribution.png")
    assert os.path.getsize("3/result/fig3_01_binomial_distribution.png") > 5000
