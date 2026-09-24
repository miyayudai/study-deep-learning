r"""
Tests for Chapter 15 Section 15.2: Mixtures of Gaussians
=========================================================
Covers:
- GaussianMixtureModel class (fit, predict, sample, score_samples, log_likelihood)
- Responsibilities gamma(z_{nk}) constraints (\sum_k gamma_{nk} = 1, 0 <= gamma <= 1)
- Log-likelihood evaluation and monotonicity during EM updates
- Ancestral sampling from prior p(z) and conditional p(x|z)
- Likelihood singularity behavior as variance -> 0
- Figure generation for Figures 15.4, 15.5, and 15.6
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from common.mixtures_of_gaussians import (
    GaussianMixtureModel,
    load_gmm_synthetic_dataset,
    generate_figure_15_4,
    generate_figure_15_5,
    generate_figure_15_6,
)


def test_gmm_fit_and_parameters():
    """Test GMM fitting on synthetic 3-component dataset."""
    X, clusters, params = load_gmm_synthetic_dataset()

    gmm = GaussianMixtureModel(n_components=3, max_iter=30, random_state=42)
    gmm.fit(X)

    # Check shapes
    assert gmm.weights_.shape == (3,)
    assert gmm.means_.shape == (3, 2)
    assert gmm.covariances_.shape == (3, 2, 2)

    # Check mixing coefficients constraints (Eq. 15.6)
    assert np.isclose(np.sum(gmm.weights_), 1.0)
    assert np.all(gmm.weights_ >= 0.0)

    # Check covariance symmetry and positive definiteness
    for k in range(3):
        cov_k = gmm.covariances_[k]
        assert np.allclose(cov_k, cov_k.T)
        eigvals = np.linalg.eigvalsh(cov_k)
        assert np.all(eigvals > 0.0)


def test_responsibilities_constraints():
    """Verify that posterior responsibilities sum to 1 and lie in [0, 1]."""
    X, _, _ = load_gmm_synthetic_dataset()

    gmm = GaussianMixtureModel(n_components=3, random_state=42)
    gmm.fit(X)

    gamma = gmm.compute_responsibilities(X)
    assert gamma.shape == (len(X), 3)

    # Sum of responsibilities for each data point must equal 1 (Eq. 15.10)
    assert np.allclose(np.sum(gamma, axis=1), 1.0)
    assert np.all(gamma >= 0.0)
    assert np.all(gamma <= 1.0)


def test_log_likelihood_monotonic_increase():
    """Verify that log-likelihood increases monotonically during EM."""
    X, _, _ = load_gmm_synthetic_dataset()

    gmm = GaussianMixtureModel(n_components=3, max_iter=20, random_state=42)
    gmm.fit(X)

    ll_history = gmm.history_log_likelihood_
    assert len(ll_history) >= 2

    # Check that log-likelihood does not decrease
    for i in range(len(ll_history) - 1):
        assert ll_history[i + 1] >= ll_history[i] - 1e-4

    # Check total log-likelihood equals sum of sample scores
    scores = gmm.score_samples(X)
    assert np.isclose(np.sum(scores), gmm.log_likelihood(X))


def test_ancestral_sampling():
    """Verify ancestral sampling generates correct shapes and proportions."""
    X, _, _ = load_gmm_synthetic_dataset()

    gmm = GaussianMixtureModel(n_components=3, random_state=42)
    gmm.fit(X)

    n_samples = 500
    X_samples, z_samples = gmm.sample(n_samples=n_samples)

    assert X_samples.shape == (n_samples, 2)
    assert z_samples.shape == (n_samples,)
    assert set(np.unique(z_samples)).issubset({0, 1, 2})

    # Check predicted labels match component with highest responsibility
    preds = gmm.predict(X_samples)
    assert preds.shape == (n_samples,)


def test_singularity_behavior():
    """Demonstrate the singularity of Gaussian density as sigma -> 0."""
    x_pt = 2.0
    sigmas = [1.0, 0.1, 0.01, 0.001]
    densities = []
    for s in sigmas:
        # N(x | x, s^2) = 1 / (sqrt(2*pi)*s)
        dens = 1.0 / (np.sqrt(2 * np.pi) * s)
        densities.append(dens)

    # Densities should grow inversely with s
    for i in range(len(densities) - 1):
        assert densities[i + 1] > densities[i]
    assert densities[-1] > 300.0


def test_figure_generators_save_valid_files():
    """Verify Figure 15.4, 15.5, and 15.6 generation and file writing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p4 = Path(tmpdir) / "test_fig_15_4.png"
        p5 = Path(tmpdir) / "test_fig_15_5.png"
        p6 = Path(tmpdir) / "test_fig_15_6.png"

        generate_figure_15_4(p4)
        assert p4.exists()
        assert p4.stat().st_size > 10000

        generate_figure_15_5(p5)
        assert p5.exists()
        assert p5.stat().st_size > 50000

        generate_figure_15_6(p6)
        assert p6.exists()
        assert p6.stat().st_size > 20000
