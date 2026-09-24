"""
Unit tests for Chapter 16 Section 16.1: Principal Component Analysis (PCA).
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
"""

import os
from pathlib import Path
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pytest

from common.principal_component_analysis import (
    PrincipalComponentAnalysis,
    whiten_data,
    dual_pca_high_dimensional,
    generate_figure_16_1,
    generate_figure_16_2,
    generate_figure_16_3,
    generate_figure_16_4,
    generate_figure_16_5,
    generate_figure_16_6,
)


def test_pca_maximum_variance_and_orthonormality():
    """Verify maximum variance formulation (Eqs. 16.1 - 16.6)."""
    rng = np.random.RandomState(42)
    # Generate 2D Gaussian data with known covariance
    true_cov = np.array([[3.0, 1.5], [1.5, 1.0]])
    X = rng.multivariate_normal(mean=[2.0, -1.0], cov=true_cov, size=200)

    pca = PrincipalComponentAnalysis(n_components=2)
    pca.fit(X)

    # 1. Sample mean matches Eq. 16.1
    assert np.allclose(pca.mean_, np.mean(X, axis=0))

    # 2. Orthonormality u_i^T u_j = delta_ij (Eq. 16.7)
    U = pca.components_
    assert np.allclose(U @ U.T, np.eye(2), atol=1e-12)

    # 3. Eigenvalues in descending order
    assert pca.explained_variance_[0] >= pca.explained_variance_[1]

    # 4. Projected variance equals eigenvalue: u_1^T S u_1 = lambda_1 (Eq. 16.2 & 16.6)
    X_c = X - pca.mean_
    S = (X_c.T @ X_c) / len(X)
    proj_var_1 = U[0] @ S @ U[0]
    assert np.isclose(proj_var_1, pca.explained_variance_[0], atol=1e-12)
    proj_var_2 = U[1] @ S @ U[1]
    assert np.isclose(proj_var_2, pca.explained_variance_[1], atol=1e-12)


def test_pca_minimum_error_reconstruction():
    """Verify minimum projection error formulation (Eqs. 16.11 - 16.18)."""
    rng = np.random.RandomState(42)
    D = 5
    N = 100
    X = rng.randn(N, D) @ rng.randn(D, D)

    # M = 2 components
    M = 2
    pca_m = PrincipalComponentAnalysis(n_components=M)
    pca_m.fit(X)

    # Reconstruction error J
    J = pca_m.reconstruction_error(X)

    # Theoretical minimum error is sum of discarded eigenvalues (Eq. 16.18)
    # Fit full PCA to get all eigenvalues
    pca_full = PrincipalComponentAnalysis(n_components=D)
    pca_full.fit(X)
    discarded_sum = np.sum(pca_full.explained_variance_[M:])

    assert np.isclose(J, discarded_sum, rtol=1e-5)

    # When M = D, reconstruction is exact (J = 0, Eq. 16.9)
    J_full = pca_full.reconstruction_error(X)
    assert np.isclose(J_full, 0.0, atol=1e-12)


def test_pca_whitening():
    """Verify data whitening to zero mean and identity covariance (Eqs. 16.24 - 16.25)."""
    rng = np.random.RandomState(42)
    X = rng.randn(150, 3) @ np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 0.8]]) + [1.0, 2.0, 3.0]

    # Function test
    Y, mean, lambdas, U = whiten_data(X)
    N = len(X)

    # Zero mean
    assert np.allclose(np.mean(Y, axis=0), 0.0, atol=1e-12)

    # Identity covariance: (1/N) Y^T Y = I
    cov_Y = (Y.T @ Y) / N
    assert np.allclose(cov_Y, np.eye(3), atol=1e-12)

    # Class test with whiten=True
    pca_white = PrincipalComponentAnalysis(n_components=3, whiten=True)
    Y_class = pca_white.fit_transform(X)
    cov_class = (Y_class.T @ Y_class) / N
    assert np.allclose(cov_class, np.eye(3), atol=1e-12)


def test_dual_pca_high_dimensional():
    """Verify dual PCA Gram matrix trick for N < D (Section 16.1.5, Eqs. 16.26 - 16.30)."""
    rng = np.random.RandomState(42)
    N, D = 20, 80  # N < D
    X = rng.randn(N, D)

    # Primal PCA
    pca_primal = PrincipalComponentAnalysis(n_components=10, method="primal")
    pca_primal.fit(X)

    # Dual PCA
    pca_dual = PrincipalComponentAnalysis(n_components=10, method="dual")
    pca_dual.fit(X)

    # 1. Eigenvalues must match
    assert np.allclose(pca_dual.explained_variance_, pca_primal.explained_variance_, atol=1e-10)

    # 2. Eigenvectors must span same directions (absolute cosine similarity = 1.0)
    for i in range(10):
        dot = abs(np.dot(pca_dual.components_[i], pca_primal.components_[i]))
        assert np.isclose(dot, 1.0, atol=1e-10)

    # 3. Standalone dual_pca_high_dimensional function
    eigvals_func, eigvecs_func = dual_pca_high_dimensional(X, n_components=10)
    assert np.allclose(eigvals_func, pca_dual.explained_variance_)
    assert np.allclose(eigvecs_func, pca_dual.components_)


def test_figure_generators():
    """Verify that all figure generators run without error and save valid images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fig1 = generate_figure_16_1(save_path=os.path.join(tmpdir, "fig1.png"))
        assert os.path.exists(os.path.join(tmpdir, "fig1.png"))

        fig2 = generate_figure_16_2(save_path=os.path.join(tmpdir, "fig2.png"))
        assert os.path.exists(os.path.join(tmpdir, "fig2.png"))

        fig3 = generate_figure_16_3(save_path=os.path.join(tmpdir, "fig3.png"))
        assert os.path.exists(os.path.join(tmpdir, "fig3.png"))

        fig4 = generate_figure_16_4(save_path=os.path.join(tmpdir, "fig4.png"))
        assert os.path.exists(os.path.join(tmpdir, "fig4.png"))

        fig5 = generate_figure_16_5(save_path=os.path.join(tmpdir, "fig5.png"))
        assert os.path.exists(os.path.join(tmpdir, "fig5.png"))

        fig6 = generate_figure_16_6(save_path=os.path.join(tmpdir, "fig6.png"))
        assert os.path.exists(os.path.join(tmpdir, "fig6.png"))
        plt.close("all")
