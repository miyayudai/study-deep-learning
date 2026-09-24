r"""
Tests for Chapter 15 Section 15.4: Evidence Lower Bound
========================================================
Covers:
- compute_elbo_decomposition (exact variational identity, KL >= 0, KL = 0 at posterior)
- SequentialGaussianMixtureEM (incremental updates of sufficient statistics, convergence)
- MAPGaussianMixtureEM (regularization via parameter priors, singularity prevention)
- Figure generators (Figures 15.10, 15.11, 15.13, 15.14, 15.15, 15.16)
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from common.evidence_lower_bound import (
    compute_elbo_decomposition,
    SequentialGaussianMixtureEM,
    MAPGaussianMixtureEM,
    generate_figure_15_10,
    generate_figure_15_11,
    generate_figure_15_13,
    generate_figure_15_14,
    generate_figure_15_15,
    generate_figure_15_16,
)
from common.kmeans_clustering import load_faithful_dataset


def test_elbo_decomposition_identity():
    """Verify exact decomposition: ln p(X|theta) = L(q, theta) + KL(q||p)."""
    X, _ = load_faithful_dataset()
    K = 2
    means = np.array([[-1.0, 1.0], [1.0, -1.0]])
    covs = np.array([np.eye(2), np.eye(2)])
    weights = np.array([0.5, 0.5])

    # 1. Test with arbitrary non-optimal q distribution
    rng = np.random.RandomState(42)
    q_arbitrary = rng.dirichlet(alpha=[1.0, 1.0], size=len(X))

    decomp = compute_elbo_decomposition(X, means, covs, weights, q=q_arbitrary)
    ll = decomp["log_likelihood"]
    elbo = decomp["elbo"]
    kl = decomp["kl_divergence"]

    # Exact identity: ln p(X) = ELBO + KL
    assert np.isclose(ll, elbo + kl, atol=1e-8)
    # Non-negativity of KL divergence
    assert kl >= 0.0
    # ELBO is a lower bound
    assert elbo <= ll + 1e-8

    # 2. Test with optimal q = p(Z | X, theta) (after E-step)
    decomp_opt = compute_elbo_decomposition(X, means, covs, weights, q=None)
    ll_opt = decomp_opt["log_likelihood"]
    elbo_opt = decomp_opt["elbo"]
    kl_opt = decomp_opt["kl_divergence"]

    assert np.isclose(ll, ll_opt)
    # At posterior, KL divergence vanishes
    assert np.isclose(kl_opt, 0.0, atol=1e-8)
    # ELBO equals log-likelihood exactly
    assert np.isclose(elbo_opt, ll_opt, atol=1e-8)


def test_sequential_gmm_em():
    """Verify SequentialGaussianMixtureEM incremental updates."""
    X, _ = load_faithful_dataset()
    init_means = np.array([[-1.5, 1.0], [1.5, -1.0]])
    init_covs = np.array([np.eye(2), np.eye(2)])
    init_weights = np.array([0.5, 0.5])

    model = SequentialGaussianMixtureEM(n_components=2, n_epochs=5)
    model.fit(X, init_means, init_covs, init_weights)

    # Check fitted shapes
    assert model.means_.shape == (2, 2)
    assert model.covariances_.shape == (2, 2, 2)
    assert model.weights_.shape == (2,)
    assert np.isclose(np.sum(model.weights_), 1.0)
    assert np.all(model.weights_ > 0.0)

    # Check covariances are symmetric and positive definite
    for k in range(2):
        cov_k = model.covariances_[k]
        assert np.allclose(cov_k, cov_k.T)
        eigvals = np.linalg.eigvalsh(cov_k)
        assert np.all(eigvals > 0.0)

    # Check that log-likelihood generally increases over epochs
    ll_hist = model.history_log_likelihood_
    assert len(ll_hist) == 6  # init + 5 epochs
    assert ll_hist[-1] > ll_hist[0]


def test_map_gmm_em_prior_regularization():
    """Verify MAPGaussianMixtureEM with parameter priors."""
    X, _ = load_faithful_dataset()
    init_means = np.array([[-1.5, 1.0], [1.5, -1.0]])
    init_covs = np.array([np.eye(2), np.eye(2)])
    init_weights = np.array([0.5, 0.5])

    map_model = MAPGaussianMixtureEM(
        n_components=2,
        max_iter=15,
        alpha_prior=3.0,
        beta_cov_prior=1.0,
    )
    map_model.fit(X, init_means, init_covs, init_weights)

    assert map_model.means_.shape == (2, 2)
    assert map_model.covariances_.shape == (2, 2, 2)
    assert map_model.weights_.shape == (2,)
    assert np.isclose(np.sum(map_model.weights_), 1.0)
    assert np.all(map_model.weights_ > 0.0)

    # Check positive definiteness
    for k in range(2):
        cov_k = map_model.covariances_[k]
        assert np.allclose(cov_k, cov_k.T)
        eigvals = np.linalg.eigvalsh(cov_k)
        assert np.all(eigvals > 0.0)

    # MAP objective history
    hist = map_model.history_map_objective_
    assert len(hist) == 15
    for i in range(len(hist) - 1):
        assert hist[i + 1] >= hist[i] - 1e-3


def test_figure_generators_save_valid_files():
    """Verify all 6 figure generators (15.10, 15.11, 15.13, 15.14, 15.15, 15.16)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p10 = Path(tmpdir) / "test_fig_15_10.png"
        p11 = Path(tmpdir) / "test_fig_15_11.png"
        p13 = Path(tmpdir) / "test_fig_15_13.png"
        p14 = Path(tmpdir) / "test_fig_15_14.png"
        p15 = Path(tmpdir) / "test_fig_15_15.png"
        p16 = Path(tmpdir) / "test_fig_15_16.png"

        generate_figure_15_10(p10)
        assert p10.exists()
        assert p10.stat().st_size > 15000

        generate_figure_15_11(p11)
        assert p11.exists()
        assert p11.stat().st_size > 15000

        generate_figure_15_13(p13)
        assert p13.exists()
        assert p13.stat().st_size > 15000

        generate_figure_15_14(p14)
        assert p14.exists()
        assert p14.stat().st_size > 15000

        generate_figure_15_15(p15)
        assert p15.exists()
        assert p15.stat().st_size > 15000

        generate_figure_15_16(p16)
        assert p16.exists()
        assert p16.stat().st_size > 15000
