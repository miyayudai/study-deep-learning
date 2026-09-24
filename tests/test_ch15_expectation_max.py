r"""
Tests for Chapter 15 Section 15.3: Expectation–Maximization Algorithm
======================================================================
Covers:
- GaussianMixtureEM class (fit_with_custom_init, compute_responsibilities, log_likelihood, history)
- BernoulliMixtureEM class (fit, predict, compute_responsibilities, log-sum-exp trick)
- gmm_to_kmeans_limit function (asymptotic convergence to hard K-means indicators as epsilon -> 0)
- draw_covariance_ellipse helper
- Figure generation for Figures 15.7, 15.8, 15.9, and 15.12
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.expectation_maximization import (
    GaussianMixtureEM,
    BernoulliMixtureEM,
    gmm_to_kmeans_limit,
    draw_covariance_ellipse,
    generate_figure_15_7,
    generate_figure_15_8,
    generate_figure_15_9,
    generate_figure_15_12,
)
from common.kmeans_clustering import load_faithful_dataset


def test_gmm_em_faithful_fitting():
    """Verify GaussianMixtureEM on Old Faithful dataset."""
    X_std, X_raw = load_faithful_dataset()

    init_means = np.array([[-1.5, 1.0], [1.5, -1.0]])
    init_covs = np.array([np.eye(2), np.eye(2)])
    init_weights = np.array([0.5, 0.5])

    model = GaussianMixtureEM(n_components=2, max_iter=10)
    model.fit_with_custom_init(
        X=X_std,
        init_means=init_means,
        init_covs=init_covs,
        init_weights=init_weights,
    )

    # Check history structure
    assert len(model.history_) == 1 + 2 * 10  # init + 10*(E + M)
    assert model.history_[0]["step"] == "init"
    assert model.history_[1]["step"] == "E"
    assert model.history_[2]["step"] == "M"

    # Check final parameters
    assert model.means_.shape == (2, 2)
    assert model.covariances_.shape == (2, 2, 2)
    assert model.weights_.shape == (2,)
    assert np.isclose(np.sum(model.weights_), 1.0)
    assert np.all(model.weights_ > 0.0)

    # Verify covariances are symmetric and positive definite
    for k in range(2):
        cov_k = model.covariances_[k]
        assert np.allclose(cov_k, cov_k.T)
        eigvals = np.linalg.eigvalsh(cov_k)
        assert np.all(eigvals > 0.0)

    # Verify log-likelihood non-decreasing after M-steps
    m_step_lls = [
        entry["log_likelihood"]
        for entry in model.history_
        if entry["step"] == "M"
    ]
    for i in range(len(m_step_lls) - 1):
        assert m_step_lls[i + 1] >= m_step_lls[i] - 1e-4


def test_gmm_responsibilities_normalization():
    """Verify that GMM posterior responsibilities sum to 1 and are bounded in [0, 1]."""
    X_std, _ = load_faithful_dataset()
    init_means = np.array([[-1.5, 1.0], [1.5, -1.0]])
    init_covs = np.array([np.eye(2), np.eye(2)])
    init_weights = np.array([0.5, 0.5])

    model = GaussianMixtureEM(n_components=2, max_iter=2)
    model.fit_with_custom_init(X_std, init_means, init_covs, init_weights)

    gamma = model.compute_responsibilities(
        X_std, model.means_, model.covariances_, model.weights_
    )
    assert gamma.shape == (len(X_std), 2)
    assert np.allclose(np.sum(gamma, axis=1), 1.0)
    assert np.all(gamma >= 0.0)
    assert np.all(gamma <= 1.0)


def test_bernoulli_mixture_em_synthetic():
    """Verify BernoulliMixtureEM on synthetic binary digit-like data."""
    rng = np.random.RandomState(42)
    N, D, K = 150, 16, 3
    # True Bernoulli means
    true_means = np.array([
        [0.9 if i < 8 else 0.1 for i in range(D)],
        [0.1 if i < 8 else 0.9 for i in range(D)],
        [0.8 if i % 2 == 0 else 0.2 for i in range(D)],
    ])
    # Generate synthetic binary observations
    z_true = rng.choice(K, size=N, p=[0.4, 0.3, 0.3])
    X = np.zeros((N, D))
    for n in range(N):
        X[n] = (rng.uniform(size=D) < true_means[z_true[n]]).astype(float)

    bem = BernoulliMixtureEM(n_components=3, max_iter=25, random_state=42)
    bem.fit(X)

    # Check fitted shapes and ranges
    assert bem.weights_.shape == (3,)
    assert np.isclose(np.sum(bem.weights_), 1.0)
    assert np.all(bem.weights_ > 0.0)

    assert bem.means_.shape == (3, D)
    assert np.all(bem.means_ >= 0.0)
    assert np.all(bem.means_ <= 1.0)

    # Verify responsibilities
    gamma = bem.compute_responsibilities(X)
    assert gamma.shape == (N, 3)
    assert np.allclose(np.sum(gamma, axis=1), 1.0)
    assert np.all(gamma >= 0.0)

    # Check predictions
    preds = bem.predict(X)
    assert preds.shape == (N,)
    assert set(np.unique(preds)).issubset({0, 1, 2})

    # Verify log-likelihood history is non-decreasing
    ll_hist = bem.history_log_likelihood_
    assert len(ll_hist) >= 2
    for i in range(len(ll_hist) - 1):
        assert ll_hist[i + 1] >= ll_hist[i] - 1e-4


def test_gmm_to_kmeans_limit():
    """Verify that GMM responsibilities converge to hard K-means indicators as epsilon -> 0."""
    X = np.array([
        [1.0, 1.0],
        [1.2, 0.9],
        [-1.0, -1.0],
        [-0.8, -1.1],
    ])
    centers = np.array([
        [1.0, 1.0],
        [-1.0, -1.0],
    ])

    epsilons = [1.0, 0.1, 0.01, 1e-4]
    gammas = gmm_to_kmeans_limit(X, centers, epsilons)

    assert len(gammas) == 4
    for gamma in gammas:
        assert np.allclose(np.sum(gamma, axis=1), 1.0)
        assert np.all(gamma >= 0.0)

    # For epsilon = 1e-4, responsibilities should be virtually 0 or 1
    gamma_hard = gammas[-1]
    # Points 0, 1 belong to center 0; points 2, 3 belong to center 1
    assert np.allclose(gamma_hard[0], [1.0, 0.0], atol=1e-3)
    assert np.allclose(gamma_hard[1], [1.0, 0.0], atol=1e-3)
    assert np.allclose(gamma_hard[2], [0.0, 1.0], atol=1e-3)
    assert np.allclose(gamma_hard[3], [0.0, 1.0], atol=1e-3)


def test_draw_covariance_ellipse():
    """Verify covariance ellipse drawing function."""
    fig, ax = plt.subplots()
    mean = np.array([0.0, 0.0])
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])

    draw_covariance_ellipse(ax, mean, cov, n_std=1.0, edgecolor="red")
    # Must have added at least 2 patches (white background outline + colored ellipse)
    assert len(ax.patches) >= 2
    plt.close(fig)


def test_figure_generators_save_valid_files():
    """Verify Figure 15.7, 15.8, 15.9, and 15.12 generation and file writing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p7 = Path(tmpdir) / "test_fig_15_7.png"
        p8 = Path(tmpdir) / "test_fig_15_8.png"
        p9 = Path(tmpdir) / "test_fig_15_9.png"
        p12 = Path(tmpdir) / "test_fig_15_12.png"

        generate_figure_15_7(p7)
        assert p7.exists()
        assert p7.stat().st_size > 50000

        generate_figure_15_8(p8)
        assert p8.exists()
        assert p8.stat().st_size > 30000

        generate_figure_15_9(p9)
        assert p9.exists()
        assert p9.stat().st_size > 20000

        generate_figure_15_12(p12)
        assert p12.exists()
        assert p12.stat().st_size > 15000
