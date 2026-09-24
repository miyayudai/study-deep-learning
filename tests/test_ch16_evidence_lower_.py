"""
Unit tests for Chapter 16 Section 16.3: Evidence Lower Bound (ELBO) and EM for PCA.
Christopher M. Bishop & Hugh Bishop (2024). Deep Learning: Foundations and Concepts.
"""

import os
from pathlib import Path
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pytest

from common.evidence_lower_bound_continuous import (
    ProbabilisticPCA_EM,
    StandardPCA_EM,
    FactorAnalysis_EM,
    generate_figure_16_10,
    FIG16_10_DATA,
)
from common.probabilistic_latent_variables import ProbabilisticPCA


@pytest.fixture
def synthetic_data_5d():
    """Generate 5D synthetic Gaussian data with 2 dominant directions."""
    rng = np.random.default_rng(42)
    N, D, M = 200, 5, 2
    Z = rng.standard_normal((N, M))
    W_true = np.array([
        [2.0, 0.5],
        [1.5, -1.0],
        [0.0, 1.8],
        [-1.2, 0.2],
        [0.8, -0.6]
    ])
    mu_true = np.array([1.0, 2.0, -1.0, 0.5, 3.0])
    sigma_true = 0.3
    eps = rng.normal(0, sigma_true, size=(N, D))
    X = Z @ W_true.T + mu_true + eps
    return X, W_true, mu_true, sigma_true


def test_ppca_em_convergence_and_subspace(synthetic_data_5d):
    """Verify EM for PPCA converges and recovers the principal subspace (Eq. 16.66 - 16.69)."""
    X, _, _, _ = synthetic_data_5d
    M = 2

    # Fit closed-form PPCA as ground truth
    ppca_closed = ProbabilisticPCA(n_components=M).fit(X)

    # Fit EM PPCA
    ppca_em = ProbabilisticPCA_EM(n_components=M, max_iter=150, tol=1e-5, random_state=42).fit(X)

    # 1. Monotonic increase of log-likelihood
    lls = [h["log_likelihood"] for h in ppca_em.history_]
    assert len(lls) > 5
    # Allow tiny float tolerance (1e-6) for numerical noise
    for i in range(1, len(lls)):
        assert lls[i] >= lls[i - 1] - 1e-6

    # 2. Noise variance sigma^2 converges close to closed-form MLE
    assert np.isclose(ppca_em.sigma2_, ppca_closed.sigma2_, rtol=0.05)

    # 3. Principal subspace projection operator P = W (W^T W)^-1 W^T must match
    P_closed = ppca_closed.W_ @ np.linalg.inv(ppca_closed.W_.T @ ppca_closed.W_) @ ppca_closed.W_.T
    P_em = ppca_em.W_ @ np.linalg.inv(ppca_em.W_.T @ ppca_em.W_) @ ppca_em.W_.T
    assert np.allclose(P_closed, P_em, atol=0.05)


def test_standard_pca_em_roweis(synthetic_data_5d):
    """Verify Roweis EM algorithm for deterministic PCA (Eq. 16.70, 16.71)."""
    X, _, _, _ = synthetic_data_5d
    M = 2

    em_pca = StandardPCA_EM(n_components=M, max_iter=50).fit(X)

    # Ground truth top 2 eigenvectors from SVD
    Xc = X - np.mean(X, axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P_svd = Vt[:M].T @ Vt[:M]

    # Subspace projector from EM PCA
    W = em_pca.W_
    P_em = W @ np.linalg.inv(W.T @ W) @ W.T

    # Subspace projectors must be identical
    assert np.allclose(P_svd, P_em, atol=1e-5)


def test_factor_analysis_em():
    """Verify EM for Factor Analysis (Eq. 16.72 - 16.76)."""
    rng = np.random.default_rng(42)
    N, D, M = 150, 4, 2
    Z = rng.standard_normal((N, M))
    W_true = rng.standard_normal((D, M))
    psi_true = np.array([0.1, 0.4, 0.2, 0.9])
    X = Z @ W_true.T + rng.normal(0, np.sqrt(psi_true), size=(N, D))

    fa_em = FactorAnalysis_EM(n_components=M, max_iter=50, tol=1e-4).fit(X)

    # Monotonic log-likelihood
    assert len(fa_em.history_) > 3
    for i in range(1, len(fa_em.history_)):
        assert fa_em.history_[i] >= fa_em.history_[i - 1] - 1e-4

    # Estimated uniquenesses are positive
    assert np.all(fa_em.psi_ > 0)


def test_figure_16_10_generation():
    """Verify that Figure 16.10 generates without error and saves valid files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_file = os.path.join(tmpdir, "fig16_10.png")
        fig = generate_figure_16_10(save_path=save_file)
        assert os.path.exists(save_file)
        plt.close(fig)
        plt.close("all")


def test_figure_16_10_mathematical_consistency():
    """Verify that the data in Figure 16.10 converges exactly to the principal component."""
    X = FIG16_10_DATA
    mu = np.mean(X, axis=0)
    Xc = X - mu
    S = (Xc.T @ Xc) / len(X)
    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(eigvals)[::-1]
    u1 = eigvecs[:, idx[0]]
    expected_slope = u1[1] / u1[0]

    # Run EM PCA on FIG16_10_DATA
    W_init = np.array([[1.0], [-0.5]])
    em = StandardPCA_EM(n_components=1, max_iter=40).fit(X, W_init=W_init)
    converged_slope = em.W_[1, 0] / em.W_[0, 0]

    assert np.isclose(converged_slope, expected_slope, rtol=1e-5)
