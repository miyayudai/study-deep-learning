"""
Unit tests for Chapter 2 Section 2.3: The Gaussian Distribution.
Validates:
  - Gaussian PDF normalization and moment integrals (Eq 2.49 - 2.54)
  - Maximum likelihood parameter estimation formulas (Eq 2.55 - 2.58)
  - Maximum likelihood bias and Bessel correction (Eq 2.59 - 2.63)
  - Probabilistic linear regression and predictive distribution (Eq 2.64 - 2.69)
  - Generated textbook figures (Figure 2.8, 2.9, 2.10, 2.11)
"""
import os
import pytest
import numpy as np
from scipy import integrate

from common.probability import (
    Gaussian1D,
    gaussian_maximum_likelihood,
    simulate_gaussian_mle_bias,
    GaussianLinearRegression
)


def test_gaussian_normalization():
    """
    Test normalization condition for 1D Gaussian (Eq 2.51):
      integral_{-infinity}^{infinity} N(x | mu, sigma^2) dx = 1
    """
    test_params = [
        (0.0, 1.0),
        (1.5, 0.5),
        (-2.0, 3.0),
        (0.5, 0.25)
    ]
    for mu, sigma2 in test_params:
        g = Gaussian1D(mu=mu, sigma2=sigma2)
        integral_val, _ = integrate.quad(g.pdf, -np.inf, np.inf)
        assert integral_val == pytest.approx(1.0, rel=1e-6)


def test_gaussian_moments_and_mode():
    """
    Test analytical and numerical moments of Gaussian distribution (Eq 2.52 - 2.54):
      E[x] = mu
      E[x^2] = mu^2 + sigma^2
      var[x] = sigma^2
      mode = mu
      precision beta = 1 / sigma^2
    """
    mu = 1.8
    sigma2 = 2.25
    sigma = 1.5
    g = Gaussian1D(mu=mu, sigma2=sigma2)

    # Analytical properties
    assert g.mean == pytest.approx(mu)
    assert g.variance == pytest.approx(sigma2)
    assert g.precision == pytest.approx(1.0 / sigma2)
    assert g.mode == pytest.approx(mu)
    assert g.sigma == pytest.approx(sigma)

    # Numerical integral for E[x] (Eq 2.52)
    e_x, _ = integrate.quad(lambda x: x * g.pdf(x), -np.inf, np.inf)
    assert e_x == pytest.approx(mu, rel=1e-6)

    # Numerical integral for E[x^2] (Eq 2.53)
    e_x2, _ = integrate.quad(lambda x: (x**2) * g.pdf(x), -np.inf, np.inf)
    assert e_x2 == pytest.approx(mu**2 + sigma2, rel=1e-6)

    # Numerical integral for var[x] = E[(x - mu)^2] (Eq 2.54)
    var_x, _ = integrate.quad(lambda x: ((x - mu)**2) * g.pdf(x), -np.inf, np.inf)
    assert var_x == pytest.approx(sigma2, rel=1e-6)


def test_gaussian_cdf_and_pdf_relation():
    """
    Test that CDF derivative equals PDF and CDF(mu) == 0.5 by symmetry.
    """
    g = Gaussian1D(mu=0.5, sigma2=1.2)
    assert g.cdf(0.5) == pytest.approx(0.5, abs=1e-7)

    h = 1e-5
    for x in [-1.5, 0.0, 0.5, 1.2, 2.5]:
        deriv = (g.cdf(x + h) - g.cdf(x - h)) / (2 * h)
        assert deriv == pytest.approx(g.pdf(x), rel=1e-4)


def test_gaussian_maximum_likelihood():
    """
    Test maximum likelihood estimators for mean and variance (Eq 2.57, 2.58, 2.63, 2.56):
      mu_ML = (1/N) * sum x_n
      sigma2_ML = (1/N) * sum (x_n - mu_ML)^2
      sigma2_unbiased = (N / (N-1)) * sigma2_ML
    """
    data = np.array([1.2, 2.4, 3.1, 4.8, 5.0])
    N = len(data)

    mle = gaussian_maximum_likelihood(data)

    expected_mu = np.mean(data)
    expected_sigma2_ml = np.mean((data - expected_mu) ** 2)
    expected_sigma2_unbiased = np.var(data, ddof=1)

    assert mle["mu_ML"] == pytest.approx(expected_mu)
    assert mle["sigma2_ML"] == pytest.approx(expected_sigma2_ml)
    assert mle["sigma2_unbiased"] == pytest.approx(expected_sigma2_unbiased)
    assert mle["sigma2_unbiased"] == pytest.approx(expected_sigma2_ml * N / (N - 1))

    # Test log likelihood evaluation matches Eq 2.56
    g_ml = Gaussian1D(mu=expected_mu, sigma2=expected_sigma2_ml)
    expected_log_lik = np.sum(g_ml.log_pdf(data))
    assert mle["log_likelihood"] == pytest.approx(expected_log_lik)


def test_maximum_likelihood_bias_and_unbiased_estimators():
    """
    Test bias of maximum likelihood estimators via simulation (Eq 2.59 - 2.63):
      E[mu_ML] = mu
      E[sigma2_ML] = ((N - 1) / N) * sigma2
      E[sigma2_hat] = sigma2 (measured relative to true mean)
      E[sigma2_tilde] = sigma2 (unbiased sample variance)
    """
    mu_true = 2.0
    sigma2_true = 3.0

    # Test N = 2 (severe 50% bias in variance)
    res_n2 = simulate_gaussian_mle_bias(mu=mu_true, sigma2=sigma2_true, N=2, n_trials=40000, seed=123)
    assert res_n2["E_mu_ML"] == pytest.approx(mu_true, abs=0.03)
    assert res_n2["E_sigma2_ML"] == pytest.approx(0.5 * sigma2_true, rel=0.03)
    assert res_n2["E_sigma2_hat"] == pytest.approx(sigma2_true, rel=0.03)
    assert res_n2["E_sigma2_tilde"] == pytest.approx(sigma2_true, rel=0.03)

    # Test N = 5 (20% underestimation of variance)
    res_n5 = simulate_gaussian_mle_bias(mu=mu_true, sigma2=sigma2_true, N=5, n_trials=40000, seed=456)
    assert res_n5["E_sigma2_ML"] == pytest.approx(0.8 * sigma2_true, rel=0.03)
    assert res_n5["E_sigma2_tilde"] == pytest.approx(sigma2_true, rel=0.03)


def test_probabilistic_linear_regression():
    """
    Test probabilistic linear regression with Gaussian noise (Section 2.3.4):
      - MLE for weights w_ML minimizes sum-of-squares error E(w) (Eq 2.67)
      - MLE for variance sigma2_ML equals mean squared residual (Eq 2.68)
      - Log likelihood matches Eq (2.66)
      - Predictive distribution returns correct shape and uncertainty
    """
    rng = np.random.default_rng(789)
    N = 100
    x = rng.uniform(-2.0, 2.0, N)
    true_w = np.array([0.5, -1.2, 0.8])  # degree 2: 0.5 - 1.2*x + 0.8*x^2
    true_sigma = 0.25

    t = true_w[0] + true_w[1] * x + true_w[2] * (x**2) + rng.normal(0, true_sigma, N)

    model = GaussianLinearRegression(degree=2)
    model.fit(x, t)

    # Estimated weights should be close to true weights with N=100
    np.testing.assert_allclose(model.w, true_w, atol=0.1)

    # Estimated variance sigma2_ML (Eq 2.68)
    assert model.sigma2_ml == pytest.approx(true_sigma**2, rel=0.25)
    assert model.sigma_ml == pytest.approx(true_sigma, rel=0.15)

    # Sum of squares error relationship: E(w) = 0.5 * sum (y - t)^2 = 0.5 * N * sigma2_ML
    sse = model.sum_of_squares_error(x, t)
    assert sse == pytest.approx(0.5 * N * model.sigma2_ml)

    # Log likelihood relationship (Eq 2.66):
    # ln p = - 1/(2*sigma2) * sum (y - t)^2 - (N/2)*ln(sigma2) - (N/2)*ln(2*pi)
    #      = - N/2 - (N/2)*ln(sigma2_ML) - (N/2)*ln(2*pi)
    ll = model.log_likelihood(x, t)
    expected_ll = -0.5 * N - 0.5 * N * np.log(model.sigma2_ml) - 0.5 * N * np.log(2.0 * np.pi)
    assert ll == pytest.approx(expected_ll)

    # Predictive distribution (Eq 2.69)
    x_test = np.array([-1.0, 0.0, 1.0])
    y_pred, s_pred = model.predict(x_test)
    assert len(y_pred) == 3
    assert s_pred == model.sigma_ml


def test_saved_figures_exist():
    """
    Verify that all Chapter 2 Section 2.3 figures have been generated and saved
    to both result/ and 2/result/.
    """
    expected_files = [
        "fig2_08_gaussian_distribution.png",
        "fig2_09_gaussian_likelihood.png",
        "fig2_10_mle_bias.png",
        "fig2_11_linear_regression_conditional_gaussian.png"
    ]
    for directory in ["result", os.path.join("2", "result")]:
        for fname in expected_files:
            fpath = os.path.join(directory, fname)
            assert os.path.exists(fpath), f"Missing expected figure: {fpath}"
            assert os.path.getsize(fpath) > 10000, f"File {fpath} is unexpectedly small: {os.path.getsize(fpath)} bytes"
