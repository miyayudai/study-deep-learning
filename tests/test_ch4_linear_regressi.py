"""
Unit tests for Chapter 4, Section 4.1: Linear Regression.
Tests:
- Basis functions (Polynomial, Gaussian, Sigmoidal)
- Maximum likelihood linear regression (Normal equations, pseudo-inverse, bias formula, residual variance)
- Least squares geometry and orthogonal projection
- Sequential LMS learning
- Regularized least squares / Ridge regression
- Multiple outputs linear regression
- Figure generation for Figures 4.1 to 4.4
"""
import os
import pytest
import numpy as np
import scipy.linalg as la

from common.linear_models import (
    PolynomialBasis,
    GaussianBasis,
    SigmoidalBasis,
    LinearRegression,
    SequentialLinearRegression,
    RidgeRegression,
    MultipleOutputLinearRegression,
    plot_figure_4_1_network_diagram,
    plot_figure_4_2_basis_functions,
    plot_figure_4_3_least_squares_geometry,
    plot_figure_4_4_multiple_outputs_diagram
)


class TestBasisFunctions:
    """Tests for Section 4.1.1 Basis Functions."""

    def test_polynomial_basis(self):
        poly = PolynomialBasis(degree=3)
        x = np.array([0.0, 1.0, 2.0])
        phi = poly(x)
        expected = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 4.0, 8.0]
        ])
        assert np.allclose(phi, expected)

    def test_gaussian_basis(self):
        centers = [-1.0, 0.0, 1.0]
        gauss = GaussianBasis(centers, s=0.5, include_bias=True)
        x = np.array([0.0])
        phi = gauss(x)
        # phi_0 = 1, phi_1 = exp(-0.5 * (0 - (-1))^2 / 0.25) = exp(-2)
        # phi_2 = exp(0) = 1.0, phi_3 = exp(-2)
        assert np.isclose(phi[0, 0], 1.0)
        assert np.isclose(phi[0, 1], np.exp(-2.0))
        assert np.isclose(phi[0, 2], 1.0)
        assert np.isclose(phi[0, 3], np.exp(-2.0))

    def test_sigmoidal_basis(self):
        centers = [0.0]
        sig = SigmoidalBasis(centers, s=1.0, include_bias=False)
        x = np.array([0.0, 2.0, -2.0])
        phi = sig(x)
        assert np.isclose(phi[0, 0], 0.5)
        assert phi[1, 0] > 0.5
        assert phi[2, 0] < 0.5


class TestLinearRegression:
    """Tests for Section 4.1.2 - 4.1.4 Linear Regression & Normal Equations."""

    def test_mle_exact_recovery(self):
        np.random.seed(42)
        X = np.linspace(-1, 1, 50)
        true_w = np.array([2.5, -1.8, 3.2])
        poly = PolynomialBasis(degree=2)
        Phi = poly(X)
        t = Phi @ true_w

        model = LinearRegression(basis_func=poly).fit(X, t)
        assert np.allclose(model.w, true_w, atol=1e-10)
        assert np.isclose(model.sigma2, 0.0, atol=1e-10)

    def test_bias_parameter_formula(self):
        np.random.seed(123)
        X = np.random.uniform(-2, 2, size=60)
        t = 3.0 + 1.5 * X + np.random.normal(0, 0.2, size=60)
        
        poly = PolynomialBasis(degree=1)
        model = LinearRegression(basis_func=poly).fit(X, t)
        
        # Eq 4.18: w_0 = \bar{t} - sum_{j=1}^{M-1} w_j \bar{phi}_j
        t_bar = np.mean(t)
        phi_1_bar = np.mean(X)
        expected_w0 = t_bar - model.w[1] * phi_1_bar
        assert np.isclose(model.w[0], expected_w0)

    def test_least_squares_orthogonality(self):
        # Section 4.1.4: residual vector (t - y) is orthogonal to subspace S spanned by columns of Phi
        np.random.seed(99)
        X = np.random.randn(20)
        t = np.random.randn(20)
        poly = PolynomialBasis(degree=3)
        model = LinearRegression(basis_func=poly).fit(X, t)
        
        y = model.predict(X)
        residuals = t - y
        # Orthogonality: Phi^T @ (t - y) == 0
        assert np.allclose(model.Phi.T @ residuals, np.zeros(4), atol=1e-10)


class TestSequentialLearning:
    """Tests for Section 4.1.5 Sequential Learning / LMS."""

    def test_lms_convergence(self):
        np.random.seed(42)
        true_w = np.array([1.5, -2.0])
        lms = SequentialLinearRegression(n_features=2, learning_rate=0.05)
        
        for _ in range(500):
            x_val = np.random.uniform(-1, 1)
            phi = np.array([1.0, x_val])
            t_val = float(np.dot(phi, true_w))
            lms.update(phi, t_val)
            
        assert np.allclose(lms.w, true_w, atol=0.08)


class TestRegularizedLeastSquares:
    """Tests for Section 4.1.6 Regularized Least Squares (Ridge)."""

    def test_ridge_shrinkage(self):
        np.random.seed(42)
        X = np.linspace(-1, 1, 30)
        t = np.sin(np.pi * X) + np.random.normal(0, 0.1, size=30)
        poly = PolynomialBasis(degree=5)

        mle = LinearRegression(basis_func=poly).fit(X, t)
        ridge_light = RidgeRegression(alpha=0.01, basis_func=poly).fit(X, t)
        ridge_heavy = RidgeRegression(alpha=10.0, basis_func=poly).fit(X, t)

        norm_mle = np.linalg.norm(mle.w)
        norm_light = np.linalg.norm(ridge_light.w)
        norm_heavy = np.linalg.norm(ridge_heavy.w)

        assert norm_heavy < norm_light <= norm_mle


class TestMultipleOutputs:
    """Tests for Section 4.1.7 Multiple Outputs."""

    def test_multiple_outputs_decoupling(self):
        np.random.seed(42)
        X = np.linspace(-1, 1, 40)
        t1 = 2.0 * X + 1.0
        t2 = -1.5 * (X**2) + 0.5
        T = np.column_stack([t1, t2])
        poly = PolynomialBasis(degree=2)

        multi_model = MultipleOutputLinearRegression(basis_func=poly).fit(X, T)
        single_m1 = LinearRegression(basis_func=poly).fit(X, t1)
        single_m2 = LinearRegression(basis_func=poly).fit(X, t2)

        assert np.allclose(multi_model.W[:, 0], single_m1.w)
        assert np.allclose(multi_model.W[:, 1], single_m2.w)


class TestFigureGenerationCh4Sec1:
    """Tests that Figures 4.1, 4.2, 4.3, 4.4 run cleanly and generate output files."""

    def test_figures_4_1_to_4_4(self, tmp_path):
        f1, _ = plot_figure_4_1_network_diagram(save_paths=[str(tmp_path / 'fig4_1.png')])
        assert f1 is not None
        assert os.path.exists(tmp_path / 'fig4_1.png')

        f2, _ = plot_figure_4_2_basis_functions(save_paths=[str(tmp_path / 'fig4_2.png')])
        assert f2 is not None
        assert os.path.exists(tmp_path / 'fig4_2.png')

        f3, _ = plot_figure_4_3_least_squares_geometry(save_paths=[str(tmp_path / 'fig4_3.png')])
        assert f3 is not None
        assert os.path.exists(tmp_path / 'fig4_3.png')

        f4, _ = plot_figure_4_4_multiple_outputs_diagram(save_paths=[str(tmp_path / 'fig4_4.png')])
        assert f4 is not None
        assert os.path.exists(tmp_path / 'fig4_4.png')
