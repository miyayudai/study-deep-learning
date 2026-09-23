"""Unit tests for Chapter 7 Section 7.1: Error Surfaces.

Bishop & Bishop (2024), Chapter 7, pp. 209-213.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pytest

from common.error_surfaces import (
    LocalQuadraticApproximation,
    classify_stationary_point,
    numerical_gradient,
    numerical_hessian,
    generate_figure_7_1,
    generate_figure_7_2,
)


class TestLocalQuadraticApproximation:
    """Mathematical verification of local quadratic approximation (Section 7.1.1)."""

    def test_exact_quadratic_reproduction(self):
        """A true quadratic function must be reproduced exactly by the approximation."""
        # True quadratic: f(w) = 0.5 * w^T H w + b^T w + c
        c = 4.5
        b_true = np.array([1.5, -2.0])
        H_true = np.array([[6.0, 2.0], [2.0, 4.0]])

        def true_f(w):
            return 0.5 * w.T @ H_true @ w + b_true.T @ w + c

        # Around w_hat = [1.0, 1.0]
        w_hat = np.array([1.0, 1.0])
        f_hat = true_f(w_hat)
        b_hat = b_true + H_true @ w_hat
        H_hat = H_true

        quad_approx = LocalQuadraticApproximation(
            center=w_hat,
            value=f_hat,
            gradient=b_hat,
            hessian=H_hat,
        )

        rng = np.random.default_rng(42)
        test_points = rng.normal(size=(50, 2))
        for w in test_points:
            f_val = true_f(w)
            f_approx = quad_approx.evaluate(w)
            assert np.isclose(f_val, f_approx, atol=1e-12)

            grad_val = b_true + H_true @ w
            grad_approx = quad_approx.gradient(w)
            assert np.allclose(grad_val, grad_approx, atol=1e-12)

    def test_eigendecomposition_and_orthonormality(self):
        """H u_i = lambda_i u_i and u_i^T u_j = delta_ij (Eq 7.8, 7.9)."""
        H = np.array([[5.0, 2.0, 1.0], [2.0, 4.0, 0.5], [1.0, 0.5, 3.0]])
        quad = LocalQuadraticApproximation(
            center=np.zeros(3),
            value=0.0,
            gradient=np.zeros(3),
            hessian=H,
        )

        U = quad.eigenvectors
        lambdas = quad.eigenvalues

        # Orthonormality U^T U = I
        assert np.allclose(U.T @ U, np.eye(3), atol=1e-12)
        # Spectral decomposition H = U Lambda U^T
        assert np.allclose(H, U @ np.diag(lambdas) @ U.T, atol=1e-12)

        # Eigenvalue equation H u_i = lambda_i u_i
        for i in range(3):
            u_i = U[:, i]
            assert np.allclose(H @ u_i, lambdas[i] * u_i, atol=1e-12)

    def test_coordinate_transformation(self):
        """Transformation to and from eigen-coordinates (Eq 7.10)."""
        w_star = np.array([2.0, -1.0, 0.5])
        H = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
        quad = LocalQuadraticApproximation(
            center=w_star,
            value=10.0,
            gradient=np.zeros(3),
            hessian=H,
        )

        w = np.array([3.5, 0.5, -1.0])
        xi = quad.to_eigen_coordinates(w)
        w_recon = quad.from_eigen_coordinates(xi)
        assert np.allclose(w, w_recon, atol=1e-12)

        # Decoupled error value in eigen-coordinates matches evaluate(w) (Eq 7.11)
        err_w = quad.evaluate(w)
        err_xi = quad.evaluate_in_eigen_coordinates(xi)
        assert np.isclose(err_w, err_xi, atol=1e-12)

    def test_ellipse_contour_semi_axes(self):
        """Semi-axes lengths are inversely proportional to square root of eigenvalues lambda_i^(-1/2) (Figure 7.2)."""
        w_star = np.array([1.0, 2.0])
        # Eigenvalues 4.0 and 1.0
        H = np.array([[4.0, 0.0], [0.0, 1.0]])
        quad = LocalQuadraticApproximation(
            center=w_star,
            value=0.0,
            gradient=np.zeros(2),
            hessian=H,
        )

        delta_E = 2.0
        # semi-axes: sqrt(2 * delta_E / lambda_i)
        # for lambda=1: sqrt(4/1) = 2.0; for lambda=4: sqrt(4/4) = 1.0
        axes = quad.contour_semi_axes(delta_E)
        assert np.isclose(axes[0], 2.0)  # lambda=1 (ascending order)
        assert np.isclose(axes[1], 1.0)  # lambda=4

        # Check points on contour have error delta_E
        for i in range(2):
            w_contour = w_star + axes[i] * quad.eigenvectors[:, i]
            err = quad.evaluate(w_contour)
            assert np.isclose(err, delta_E, atol=1e-12)

    def test_positive_definiteness(self):
        """v^T H v > 0 for all v != 0 (Eq 7.12 - 7.14)."""
        # Positive definite
        H_pos = np.array([[3.0, 1.0], [1.0, 2.0]])
        q_pos = LocalQuadraticApproximation(np.zeros(2), 0.0, np.zeros(2), H_pos)
        assert q_pos.is_positive_definite()

        # Indefinite (saddle)
        H_saddle = np.array([[2.0, 0.0], [0.0, -1.0]])
        q_saddle = LocalQuadraticApproximation(np.zeros(2), 0.0, np.zeros(2), H_saddle)
        assert not q_saddle.is_positive_definite()


class TestStationaryPointClassification:
    """Verification of stationary point classification."""

    def test_classification_types(self):
        # Minimum: all positive
        assert classify_stationary_point(np.array([2.0, 5.0, 1.0])) == "minimum"
        # Maximum: all negative
        assert classify_stationary_point(np.array([-2.0, -0.5])) == "maximum"
        # Saddle point: mixed
        assert classify_stationary_point(np.array([3.0, -1.5])) == "saddle_point"
        assert classify_stationary_point(np.array([4.0, 1.0, -0.2])) == "saddle_point"
        # Degenerate: zero eigenvalue
        assert classify_stationary_point(np.array([2.0, 0.0])) == "degenerate"


class TestNumericalCalculus:
    """Finite difference gradient and Hessian verification."""

    def test_numerical_gradient_and_hessian(self):
        # Non-linear function: f(w1, w2) = sin(w1) * exp(w2) + w1^2
        def f(w):
            return np.sin(w[0]) * np.exp(w[1]) + w[0] ** 2

        w0 = np.array([0.5, -0.2])

        # Analytical gradient
        grad_an = np.array([
            np.cos(w0[0]) * np.exp(w0[1]) + 2.0 * w0[0],
            np.sin(w0[0]) * np.exp(w0[1]),
        ])
        grad_num = numerical_gradient(f, w0)
        assert np.allclose(grad_an, grad_num, atol=1e-5)

        # Analytical Hessian
        H_an = np.array([
            [-np.sin(w0[0]) * np.exp(w0[1]) + 2.0, np.cos(w0[0]) * np.exp(w0[1])],
            [np.cos(w0[0]) * np.exp(w0[1]), np.sin(w0[0]) * np.exp(w0[1])],
        ])
        H_num = numerical_hessian(f, w0)
        assert np.allclose(H_an, H_num, atol=1e-4)


class TestFigureGenerators:
    """Verification of Figure 7.1 and Figure 7.2 generation and saving."""

    def test_generate_figure_7_1(self, tmp_path):
        out_file = str(tmp_path / "test_fig_7_1.png")
        fig = generate_figure_7_1(filepath=out_file, save_both=True)
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(out_file)
        assert os.path.getsize(out_file) > 1000

        # Verify dual save locations
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ch7_path = os.path.join(root, "7", "result", "fig_7_1_error_surface.png")
        root_path = os.path.join(root, "result", "fig_7_1_error_surface.png")
        assert os.path.exists(ch7_path)
        assert os.path.exists(root_path)
        plt.close(fig)

    def test_generate_figure_7_2(self, tmp_path):
        out_file = str(tmp_path / "test_fig_7_2.png")
        fig = generate_figure_7_2(filepath=out_file, save_both=True)
        assert isinstance(fig, plt.Figure)
        assert os.path.exists(out_file)
        assert os.path.getsize(out_file) > 1000

        # Verify dual save locations
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ch7_path = os.path.join(root, "7", "result", "fig_7_2_error_contours.png")
        root_path = os.path.join(root, "result", "fig_7_2_error_contours.png")
        assert os.path.exists(ch7_path)
        assert os.path.exists(root_path)
        plt.close(fig)
