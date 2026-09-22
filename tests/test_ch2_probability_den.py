"""
Unit tests for Chapter 2 Section 2.2: Probability Densities.
Validates continuous probability distributions, CDF derivative relationships,
theoretical moments, Monte Carlo approximations, variance algebraic expansions,
covariance properties, and generated figures.
"""
import os
import pytest
import numpy as np
from scipy import integrate

from common.probability import (
    UniformDistribution,
    ExponentialDistribution,
    LaplaceDistribution,
    GaussianMixture1D,
    EmpiricalDistribution,
    monte_carlo_expectation,
    compute_variance,
    compute_covariance,
    compute_covariance_matrix
)


def test_continuous_distributions_normalization():
    """
    Test normalization condition (Eq 2.25, 2.28):
      integral_{-infinity}^{infinity} p(x) dx = 1
    """
    # 1. Uniform
    u = UniformDistribution(-1.0, 1.0)
    int_u, _ = integrate.quad(u.pdf, -2.0, 3.0)
    assert int_u == pytest.approx(1.0, rel=1e-6)

    # 2. Exponential (lambda = 1.5)
    e = ExponentialDistribution(lam=1.5)
    int_e, _ = integrate.quad(e.pdf, 0.0, np.inf)
    assert int_e == pytest.approx(1.0, rel=1e-6)

    # 3. Laplace (mu = 1.0, gamma = 1.0)
    l = LaplaceDistribution(mu=1.0, gamma=1.0)
    int_l, _ = integrate.quad(l.pdf, -np.inf, np.inf)
    assert int_l == pytest.approx(1.0, rel=1e-6)

    # 4. Gaussian Mixture
    gm = GaussianMixture1D(weights=[0.38, 0.62], means=[1.4, 3.05], stds=[0.55, 0.38])
    int_gm, _ = integrate.quad(gm.pdf, -np.inf, np.inf)
    assert int_gm == pytest.approx(1.0, rel=1e-6)


def test_theoretical_moments():
    """
    Test that analytical formulas for mean and variance match numerical integrations:
      Uniform(a, b): mean = (a+b)/2, var = (b-a)^2 / 12
      Exponential(lam): mean = 1/lam, var = 1/lam^2
      Laplace(mu, gamma): mean = mu, var = 2*gamma^2
    """
    # Uniform
    a, b = -2.0, 3.0
    u = UniformDistribution(a, b)
    assert u.mean == pytest.approx((a + b) / 2.0)
    assert u.variance == pytest.approx(((b - a) ** 2) / 12.0)

    # Exponential
    lam = 2.5
    e = ExponentialDistribution(lam)
    assert e.mean == pytest.approx(1.0 / lam)
    assert e.variance == pytest.approx(1.0 / (lam ** 2))

    # Laplace
    mu, gamma = 2.0, 1.5
    l = LaplaceDistribution(mu, gamma)
    assert l.mean == pytest.approx(mu)
    assert l.variance == pytest.approx(2.0 * (gamma ** 2))

    # Gaussian Mixture
    gm = GaussianMixture1D(weights=[0.3, 0.7], means=[-1.0, 2.0], stds=[0.5, 0.8])
    expected_mean = 0.3 * (-1.0) + 0.7 * (2.0)
    assert gm.mean == pytest.approx(expected_mean)
    expected_second_moment = 0.3 * (0.5**2 + (-1.0)**2) + 0.7 * (0.8**2 + 2.0**2)
    expected_var = expected_second_moment - (expected_mean ** 2)
    assert gm.variance == pytest.approx(expected_var)


def test_cdf_derivative_equals_pdf():
    """
    Test the fundamental relation (Eq 2.26):
      P'(x) = d/dx [ integral_{-infinity}^x p(t) dt ] = p(x)
    via central finite differences: (P(x+h) - P(x-h)) / (2h) == p(x).
    """
    gm = GaussianMixture1D(weights=[0.38, 0.62], means=[1.4, 3.05], stds=[0.55, 0.38])
    h = 1e-5
    x_test = np.linspace(0.5, 4.0, 20)

    for x in x_test:
        deriv = (gm.cdf(x + h) - gm.cdf(x - h)) / (2 * h)
        assert deriv == pytest.approx(gm.pdf(x), rel=1e-4)

    # Test Laplace for smooth regions (away from mu=1.0)
    l = LaplaceDistribution(mu=1.0, gamma=1.0)
    for x in [0.2, 0.5, 1.8, 2.5]:
        deriv = (l.cdf(x + h) - l.cdf(x - h)) / (2 * h)
        assert deriv == pytest.approx(l.pdf(x), rel=1e-4)


def test_monte_carlo_expectation_convergence():
    """
    Test sample approximation of expectation (Eq 2.40):
      E[f] ~= (1/N) * sum_{n=1}^N f(x_n)
    Convergence to true theoretical mean according to Law of Large Numbers.
    """
    lam = 2.0
    e = ExponentialDistribution(lam=lam)
    true_mean = 1.0 / lam  # 0.5

    # Large sample test
    samples = e.sample(size=100000, seed=42)
    mc_mean = monte_carlo_expectation(samples)
    assert mc_mean == pytest.approx(true_mean, abs=0.01)

    # Test expectation of f(x) = x^2: E[x^2] = var + (E[x])^2 = 1/4 + 1/4 = 0.5
    mc_x2 = monte_carlo_expectation(samples, f=lambda x: x**2)
    assert mc_x2 == pytest.approx(0.5, abs=0.01)


def test_empirical_distribution_properties():
    """
    Test empirical distribution based on Dirac delta functions (Eq 2.37):
      p(x | D) = (1/N) * sum_n delta(x - x_n)
    """
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    emp = EmpiricalDistribution(data)

    assert emp.N == 5
    assert emp.mean == pytest.approx(3.0)
    assert emp.variance == pytest.approx(2.0)

    # Test custom expectation E[x^2] = (1+4+9+16+25)/5 = 55/5 = 11
    assert emp.expectation(lambda x: x**2) == pytest.approx(11.0)

    # Test empirical CDF step function
    assert emp.cdf(0.5) == 0.0
    assert emp.cdf(1.0) == 0.2
    assert emp.cdf(3.5) == 0.6
    assert emp.cdf(5.0) == 1.0
    assert emp.cdf(6.0) == 1.0


def test_variance_algebraic_identity():
    """
    Test variance definition and algebraic expansion (Eq 2.44, 2.45):
      var[f] = E[(f(x) - E[f(x)])^2] = E[f(x)^2] - (E[f(x)])^2
    """
    rng = np.random.default_rng(123)
    samples = rng.normal(loc=2.0, scale=3.0, size=5000)

    # Test for f(x) = x
    res_identity = compute_variance(samples)
    assert res_identity["var_definition"] == pytest.approx(res_identity["var_algebraic"], rel=1e-10)
    assert res_identity["var"] == pytest.approx(9.0, rel=0.05)

    # Test for f(x) = sin(x)
    res_sin = compute_variance(samples, f=np.sin)
    assert res_sin["var_definition"] == pytest.approx(res_sin["var_algebraic"], rel=1e-10)


def test_covariance_and_independence():
    """
    Test covariance definition (Eq 2.47):
      cov[x, y] = E[(x - E[x])(y - E[y])] = E[xy] - E[x]E[y]
    And property: if x and y are independent, cov[x, y] = 0.
    """
    rng = np.random.default_rng(42)
    N = 50000

    # Independent variables
    x_indep = rng.normal(0, 1, size=N)
    y_indep = rng.normal(0, 1, size=N)
    cov_indep = compute_covariance(x_indep, y_indep)

    assert cov_indep["cov_definition"] == pytest.approx(cov_indep["cov_algebraic"], abs=1e-10)
    assert abs(cov_indep["cov"]) < 0.02  # Close to 0

    # Dependent variables: y = 2x + noise
    x_dep = rng.normal(2, 1, size=N)
    y_dep = 2.0 * x_dep + rng.normal(0, 0.5, size=N)
    cov_dep = compute_covariance(x_dep, y_dep)

    # Theoretical cov[x, 2x+eps] = 2 * var[x] = 2.0
    assert cov_dep["cov_definition"] == pytest.approx(cov_dep["cov_algebraic"], rel=1e-10)
    assert cov_dep["cov"] == pytest.approx(2.0, rel=0.05)


def test_covariance_matrix_properties():
    """
    Test covariance matrix for vector variable (Eq 2.48):
      cov[x] = E[(x - E[x])(x - E[x])^T]
    Must be symmetric and positive semi-definite.
    """
    rng = np.random.default_rng(99)
    # Generate 3D data with correlations
    true_cov = np.array([
        [2.0, 0.8, 0.2],
        [0.8, 1.5, -0.4],
        [0.2, -0.4, 1.0]
    ])
    mean = np.array([1.0, -1.0, 0.5])
    X = rng.multivariate_normal(mean, true_cov, size=10000)

    sample_cov = compute_covariance_matrix(X)

    # Shape
    assert sample_cov.shape == (3, 3)

    # Symmetry
    np.testing.assert_allclose(sample_cov, sample_cov.T, atol=1e-12)

    # Positive semi-definiteness: all eigenvalues >= 0
    eigenvalues = np.linalg.eigvalsh(sample_cov)
    assert np.all(eigenvalues >= -1e-12)

    # Proximity to true covariance
    np.testing.assert_allclose(sample_cov, true_cov, atol=0.1)


def test_saved_figures_exist():
    """
    Verify that all textbook figures have been generated and saved to result/ and 2/result/.
    """
    expected_files = [
        "fig2_06_probability_density_and_cdf.png",
        "fig2_07_example_distributions.png"
    ]
    for directory in ["result", os.path.join("2", "result")]:
        for fname in expected_files:
            fpath = os.path.join(directory, fname)
            assert os.path.exists(fpath), f"Missing expected figure: {fpath}"
            assert os.path.getsize(fpath) > 1000, f"File {fpath} is unexpectedly small or empty"
