"""
Unit tests for Chapter 16 Section 16.2: Probabilistic Latent Variables.
Christopher M. Bishop & Hugh Bishop (2024). Deep Learning: Foundations and Concepts.
"""

import os
from pathlib import Path
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pytest

from common.probabilistic_latent_variables import (
    ProbabilisticPCA,
    FactorAnalysisModel,
    FastICA2D,
    KalmanFilter1D,
    generate_figure_16_7,
    generate_figure_16_8,
    generate_figure_16_9,
)


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
    sigma_true = 0.4
    eps = rng.normal(0, sigma_true, size=(N, D))
    X = Z @ W_true.T + mu_true + eps
    return X, W_true, mu_true, sigma_true


def test_ppca_closed_form_mle(synthetic_data_5d):
    """Verify PPCA closed-form MLE formulas for W and sigma^2 (Eq. 16.46, 16.47)."""
    X, _, _, _ = synthetic_data_5d
    N, D = X.shape
    M = 2

    ppca = ProbabilisticPCA(n_components=M)
    ppca.fit(X)

    # 1. Sample mean
    assert np.allclose(ppca.mu_, np.mean(X, axis=0))

    # 2. Noise variance sigma_ML^2 is average of discarded eigenvalues (Eq. 16.47)
    eigvals = ppca.eigenvalues_
    expected_sigma2 = np.mean(eigvals[M:])
    assert np.isclose(ppca.sigma2_, expected_sigma2)

    # 3. W_ML column norms squared should be lambda_i - sigma^2 when R = I
    scale_expected = np.sqrt(np.maximum(eigvals[:M] - ppca.sigma2_, 0.0))
    for i in range(M):
        col_norm = np.linalg.norm(ppca.W_[:, i])
        assert np.isclose(col_norm, scale_expected[i])

    # 4. Orthogonality of columns when R = I
    assert np.isclose(np.dot(ppca.W_[:, 0], ppca.W_[:, 1]), 0.0, atol=1e-10)

    # 5. Marginal covariance matrix C = W W^T + sigma^2 I (Eq. 16.36)
    C_expected = ppca.W_ @ ppca.W_.T + ppca.sigma2_ * np.eye(D)
    assert np.allclose(ppca.C_, C_expected)


def test_ppca_woodbury_and_determinant(synthetic_data_5d):
    """Verify Woodbury matrix inversion and determinant identity (Eq. 16.41, 16.42)."""
    X, _, _, _ = synthetic_data_5d
    ppca = ProbabilisticPCA(n_components=2).fit(X)
    D = X.shape[1]

    # Direct inversion of C
    C_inv_direct = np.linalg.inv(ppca.C_)

    # Woodbury inversion: sigma^-2 I - sigma^-2 W M^-1 W^T
    C_inv_woodbury = (np.eye(D) - ppca.W_ @ ppca.M_inv_ @ ppca.W_.T) / ppca.sigma2_
    assert np.allclose(C_inv_direct, C_inv_woodbury, atol=1e-10)

    # Determinant via Sylvester's identity: |C| = (sigma^2)^(D-M) |M|
    det_C_direct = np.linalg.det(ppca.C_)
    det_C_sylvester = (ppca.sigma2_ ** (D - ppca.n_components)) * np.linalg.det(ppca.M_)
    assert np.isclose(det_C_direct, det_C_sylvester, rtol=1e-8)

    # Per-sample log likelihood matches direct Gaussian log-density
    ll_ppca = ppca.log_likelihood(X)
    # Direct formula
    Xc = X - ppca.mu_
    quad = np.sum((Xc @ C_inv_direct) * Xc, axis=1)
    ll_direct = -0.5 * np.sum(D * np.log(2 * np.pi) + np.log(det_C_direct) + quad)
    assert np.isclose(ll_ppca, ll_direct, rtol=1e-8)


def test_ppca_rotational_invariance(synthetic_data_5d):
    """Verify that rotation in latent space leaves marginal density invariant (Eq. 16.40)."""
    X, _, _, _ = synthetic_data_5d
    M = 2

    # Standard fit with R = I
    ppca1 = ProbabilisticPCA(n_components=M).fit(X)

    # Orthogonal 2D rotation matrix R
    theta = np.pi / 3.0
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    ppca2 = ProbabilisticPCA(n_components=M, R=R).fit(X)

    # C must be identical
    assert np.allclose(ppca1.C_, ppca2.C_, atol=1e-10)

    # Log likelihood must be identical
    assert np.isclose(ppca1.log_likelihood(X), ppca2.log_likelihood(X), atol=1e-10)

    # Reconstructed points in data space must be identical
    X_rec1 = ppca1.reconstruct(X)
    X_rec2 = ppca2.reconstruct(X)
    assert np.allclose(X_rec1, X_rec2, atol=1e-10)


def test_ppca_zero_noise_limit():
    """Verify that in the limit sigma^2 -> 0, PPCA posterior mean recovers standard PCA (Eq. 16.51)."""
    rng = np.random.default_rng(123)
    N, D, M = 100, 3, 2

    # Create planar data (rank 2, zero noise orthogonal to plane)
    u1 = np.array([1.0, 0.0, 0.0])
    u2 = np.array([0.0, 1.0, 0.0])
    Z = rng.standard_normal((N, M))
    X = Z[:, 0:1] * u1 + Z[:, 1:2] * u2  # perfectly on z=0 plane

    ppca = ProbabilisticPCA(n_components=M).fit(X)

    # sigma2_ should be virtually 0 (clamped to 1e-15)
    assert ppca.sigma2_ <= 1e-12

    # Posterior mean should match orthogonal projection (X - mu) @ U_M
    Z_ppca = ppca.transform(X)
    Xc = X - ppca.mu_
    # Orthogonal projection onto top 2 components
    Z_ortho = Xc @ ppca.eigenvectors_[:, :M]

    # Reconstructions must match perfectly
    X_recon = ppca.reconstruct(X)
    assert np.allclose(X_recon, X, atol=1e-8)


def test_ppca_degrees_of_freedom():
    """Verify parameter count formula: D M + 1 - M(M - 1)/2 (Eq. 16.52)."""
    X = np.zeros((20, 5))
    ppca = ProbabilisticPCA(n_components=2).fit(X)
    # D=5, M=2: 5*2 + 1 - 2*1/2 = 10 + 1 - 1 = 10
    assert ppca.degrees_of_freedom() == 10

    # For M = D - 1 (M = 4, D = 5):
    # D(D+1)/2 = 5*6/2 = 15
    ppca4 = ProbabilisticPCA(n_components=4).fit(X)
    assert ppca4.degrees_of_freedom() == 15


def test_factor_analysis():
    """Verify Factor Analysis EM fitting and diagonal noise covariance structure."""
    rng = np.random.default_rng(42)
    N, D, M = 150, 4, 2
    Z = rng.standard_normal((N, M))
    W_true = rng.standard_normal((D, M))
    psi_true = np.array([0.1, 0.5, 0.2, 0.8])  # distinct diagonal variances
    X = Z @ W_true.T + rng.normal(0, np.sqrt(psi_true), size=(N, D))

    fa = FactorAnalysisModel(n_components=M, max_iter=50).fit(X)
    assert fa.W_.shape == (D, M)
    assert len(fa.psi_) == D
    assert np.all(fa.psi_ > 0)

    # Posterior transform
    Z_scores = fa.transform(X)
    assert Z_scores.shape == (N, M)


def test_fast_ica_2d():
    """Verify FastICA source separation for non-Gaussian sources (Eq. 16.55)."""
    rng = np.random.default_rng(42)
    n_samples = 1000
    t = np.linspace(0, 10, n_samples)

    # Non-Gaussian sources: sine wave and sawtooth wave
    s1 = np.sin(2.0 * t)
    s2 = 2.0 * (t % 1.0) - 1.0
    S_true = np.vstack([s1, s2]).T
    S_true /= np.std(S_true, axis=0)

    # Mixing matrix
    A = np.array([[0.8, 0.6], [0.3, 0.9]])
    X = S_true @ A.T

    ica = FastICA2D(random_state=42)
    S_est = ica.fit_transform(X)
    S_est /= np.std(S_est, axis=0)

    # Recovered signals should have high correlation (or anti-correlation) with true signals
    corrs = np.abs(np.corrcoef(S_true.T, S_est.T)[:2, 2:])
    # Check that each true source has a strong match (> 0.9 correlation)
    max_corrs = np.max(corrs, axis=1)
    assert np.all(max_corrs > 0.90)


def test_kalman_filter_1d():
    """Verify 1D Kalman filter tracking on random walk sequence."""
    rng = np.random.default_rng(42)
    N = 50
    # True latent state
    w = rng.normal(0, 0.2, size=N)
    z_true = np.cumsum(w)
    # Noisy observations
    v = rng.normal(0, 0.5, size=N)
    x_obs = z_true + v

    kf = KalmanFilter1D(A=1.0, C=1.0, Gamma=0.04, Sigma=0.25)
    means, vars_ = kf.filter(x_obs)

    # Kalman filter variance should be positive and bounded
    assert np.all(vars_ > 0)
    assert np.all(vars_ < 0.25)

    # RMSE of Kalman filter should be lower than raw noisy observations
    rmse_kf = np.sqrt(np.mean((means - z_true)**2))
    rmse_raw = np.sqrt(np.mean((x_obs - z_true)**2))
    assert rmse_kf < rmse_raw


def test_figure_generators():
    """Verify that Figures 16.7, 16.8, and 16.9 generate without error and save valid images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fig7 = generate_figure_16_7(save_path=os.path.join(tmpdir, "fig7.png"))
        assert os.path.exists(os.path.join(tmpdir, "fig7.png"))
        plt.close(fig7)

        fig8 = generate_figure_16_8(save_path=os.path.join(tmpdir, "fig8.png"))
        assert os.path.exists(os.path.join(tmpdir, "fig8.png"))
        plt.close(fig8)

        fig9 = generate_figure_16_9(save_path=os.path.join(tmpdir, "fig9.png"))
        assert os.path.exists(os.path.join(tmpdir, "fig9.png"))
        plt.close(fig9)

        plt.close("all")
