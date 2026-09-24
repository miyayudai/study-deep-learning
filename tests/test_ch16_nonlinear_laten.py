"""Unit tests for Chapter 16 Section 16.4: Nonlinear Latent Variable Models."""

import os
import tempfile
import numpy as np
import pytest
from scipy.stats import norm

from common.nonlinear_latent_variables import (
    change_of_variables_density,
    NonlinearLatentVariableModel,
    CircleManifoldModel,
    BernoulliObservationModel,
    MultinomialObservationModel,
    uniform_dequantize,
    PixelLikelihoodComparison,
    generate_figure_16_11,
    generate_figure_16_12,
    generate_figure_16_13,
    generate_figure_16_14,
    generate_figure_16_15,
)


def test_change_of_variables_density():
    """Test change of variables formula (Eq. 16.78 - 16.79)."""
    # Let z ~ N(0, 1) and x = 2*z + 3 (linear transform)
    # Then z = (x - 3) / 2, dz/dx = 0.5, |det J| = 0.5
    # True distribution of x is N(3, 4), sigma = 2
    def z_pdf(z):
        return norm.pdf(z, 0, 1)

    def inv_t(x):
        return (x - 3.0) / 2.0

    def jacobian_det(x):
        return 0.5 * np.ones_like(x)

    x_vals = np.linspace(-7, 13, 300)
    density_cov = change_of_variables_density(z_pdf, inv_t, jacobian_det, x_vals)
    density_true = norm.pdf(x_vals, loc=3.0, scale=2.0)

    np.testing.assert_allclose(density_cov, density_true, rtol=1e-5, atol=1e-6)

    # Verify numerical integration over x gives 1.0
    dx = x_vals[1] - x_vals[0]
    total_prob = np.sum(density_cov) * dx
    assert np.isclose(total_prob, 1.0, atol=1e-3)


def test_nonlinear_latent_variable_model():
    """Test generative sampling and Monte Carlo marginal likelihood (Eq. 16.80, 16.83)."""
    rng = np.random.RandomState(42)

    # 2D latent to 3D observed
    def g_func(z):
        # z: (N, 2)
        x1 = z[:, 0]
        x2 = z[:, 1]
        x3 = 0.5 * (x1**2 + x2**2)
        return np.stack([x1, x2, x3], axis=-1)

    model = NonlinearLatentVariableModel(g_func, latent_dim=2, data_dim=3, sigma=0.2)

    # Sampling
    z_s, x_s = model.sample(200, rng=rng)
    assert z_s.shape == (200, 2)
    assert x_s.shape == (200, 3)

    # Conditional log prob
    cond_lp = model.conditional_log_prob(x_s[:5], z_s[:5])
    assert cond_lp.shape == (5,)
    assert np.all(np.isfinite(cond_lp))

    # Monte Carlo marginal likelihood
    mc_lp = model.log_marginal_likelihood_mc(x_s[:3], num_mc_samples=500, rng=rng)
    assert mc_lp.shape == (3,)
    assert np.all(np.isfinite(mc_lp))


def test_circle_manifold_model():
    """Test CircleManifoldModel properties and exact 2D quadrature (Figure 16.13)."""
    model = CircleManifoldModel(sigma=0.3)

    # Test unit circle property of mean
    z = np.linspace(-3, 3, 50)
    g_z = model.g(z)
    radii_sq = g_z[:, 0]**2 + g_z[:, 1]**2
    np.testing.assert_allclose(radii_sq, 1.0, rtol=1e-6)

    # Test 2D quadrature integration on a grid
    x_grid = np.linspace(-2.0, 2.0, 50)
    density = model.marginal_density_grid(x_grid, x_grid, num_quad_points=60)
    assert density.shape == (50, 50)
    assert np.all(density >= 0)

    # Integral over 2D plane should be close to 1
    dx = x_grid[1] - x_grid[0]
    total_mass = np.sum(density) * (dx**2)
    assert 0.95 <= total_mass <= 1.05


def test_bernoulli_observation_model():
    """Test BernoulliObservationModel for binary data (Eq. 16.84)."""
    rng = np.random.RandomState(42)

    def logits(z):
        # Linear projection for test
        w = np.array([[1.5, -0.8], [0.3, 1.2], [-1.0, 0.5]])
        return z @ w.T

    model = BernoulliObservationModel(logits)
    z = np.array([[0.5, -0.2], [-1.0, 1.0]])
    probs = model.probabilities(z)
    assert probs.shape == (2, 3)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Sample binary data
    samples = model.sample(z, rng=rng)
    assert samples.shape == (2, 3)
    assert set(np.unique(samples)).issubset({0.0, 1.0})

    # Log likelihood
    ll = model.log_prob(samples, z)
    assert ll.shape == (2,)
    assert np.all(ll <= 0.0)


def test_multinomial_observation_model():
    """Test MultinomialObservationModel for categorical data (Eq. 16.85 - 16.86)."""
    rng = np.random.RandomState(42)

    def logits(z):
        w = np.array([[1.0, 0.0, -1.0], [0.5, 0.5, 0.0]])
        return z @ w

    model = MultinomialObservationModel(logits)
    z = np.array([[0.2, 0.8], [-0.5, 0.5]])
    probs = model.probabilities(z)
    assert probs.shape == (2, 3)
    # Check softmax sum to 1
    np.testing.assert_allclose(np.sum(probs, axis=-1), np.ones(2), rtol=1e-6)

    # Sample one-hot
    samples = model.sample(z, rng=rng)
    assert samples.shape == (2, 3)
    np.testing.assert_allclose(np.sum(samples, axis=-1), np.ones(2))
    assert set(np.unique(samples)).issubset({0.0, 1.0})

    # Log likelihood
    ll = model.log_prob(samples, z)
    assert ll.shape == (2,)
    assert np.all(ll <= 0.0)


def test_uniform_dequantize():
    """Test dequantization adds uniform noise and avoids discrete point mass (Figure 16.15)."""
    rng = np.random.RandomState(42)
    discrete = np.array([0, 1, 2, 10, 255])
    cont = uniform_dequantize(discrete, scale=1.0, rng=rng)
    assert cont.shape == discrete.shape
    # Values should strictly lie in [k, k + 1)
    for d, c in zip(discrete, cont):
        assert d <= c < d + 1.0


def test_pixel_likelihood_comparison():
    """Test Figure 16.14 pixel distance vs likelihood behavior."""
    # (a) target, (b) corrupted with small MSE, (c) shifted with larger MSE
    target = np.zeros((10, 10))
    cand_b = np.copy(target)
    cand_b[0, 0] = 1.0  # single pixel defect
    cand_c = target + 0.3  # small uniform shift

    mse_b = np.mean((target - cand_b)**2)  # 1/100 = 0.01
    mse_c = np.mean((target - cand_c)**2)  # 0.09

    assert mse_b < mse_c

    # When sigma is small (e.g. 0.1), likelihood of b is much higher than c
    lp_b = PixelLikelihoodComparison.compute_gaussian_log_likelihood(target, cand_b, sigma=0.1)
    lp_c = PixelLikelihoodComparison.compute_gaussian_log_likelihood(target, cand_c, sigma=0.1)
    assert lp_b > lp_c


def test_figure_generators():
    """Verify that all figure generators execute and create valid PNG files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generators = [
            (generate_figure_16_11, "fig_16_11.png"),
            (generate_figure_16_12, "fig_16_12.png"),
            (generate_figure_16_13, "fig_16_13.png"),
            (generate_figure_16_14, "fig_16_14.png"),
            (generate_figure_16_15, "fig_16_15.png"),
        ]
        for gen, fname in generators:
            p = os.path.join(tmpdir, fname)
            fig = gen(save_path=p)
            assert os.path.exists(p)
            assert os.path.getsize(p) > 1000
            import matplotlib.pyplot as plt
            plt.close(fig)
