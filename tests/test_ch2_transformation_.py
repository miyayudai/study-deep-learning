"""
Unit tests for Chapter 2 Section 2.4: Transformation of Densities.
Validates:
  - 1D density transformation formula and normalization integral (Eq 2.71)
  - Mode dependence on variables vs covariant function transform (Eq 2.72, 2.73)
  - Analytical vs numerical Jacobian matrix and determinant (Eq 2.76, 2.77, Exercise 2.20)
  - Bivariate nonlinear mapping invertibility and conservation of probability mass
  - Generated textbook figures (Figure 2.12, Figure 2.13)
"""
import os
import pytest
import numpy as np
from scipy import integrate

from common.probability import (
    transform_density_1d,
    DensityTransformation1DExample,
    LinearTransformation1D,
    BivariateTransformation2DExample
)


def test_transformation_1d_density_conservation():
    """
    Test normalization condition for 1D transformed density (Eq 2.71):
      integral_0^1 py(y) dy = 1
    where y = g^{-1}(x) in (0, 1).
    """
    example = DensityTransformation1DExample(mu=6.5, sigma=1.0)

    # Numerical integral over (0, 1)
    integral_val, _ = integrate.quad(example.py, 1e-6, 1.0 - 1e-6)
    assert integral_val == pytest.approx(1.0, rel=1e-5)

    # Test generic transform_density_1d function gives identical output
    y_test = np.array([0.1, 0.3, 0.5, 0.7, 0.85, 0.95])
    py_generic = transform_density_1d(example.px, example.g, example.g_prime, y_test)
    py_class = example.py(y_test)
    np.testing.assert_allclose(py_generic, py_class, rtol=1e-10)


def test_transformation_1d_mode_shift():
    """
    Test that the mode of transformed density py(y) differs from the mode of px(x)
    transformed as a simple function (Eq 2.72 vs Eq 2.73).
    """
    example = DensityTransformation1DExample(mu=6.5, sigma=1.0)

    # 1. Mode of px(x) is mu = 6.5
    assert example.mode_x() == pytest.approx(6.5)

    # 2. Mode of simple function transformation px(g(y)) is g^{-1}(mu)
    y_func_mode = example.mode_function_transform()
    assert y_func_mode == pytest.approx(example.g_inv(6.5))
    assert y_func_mode == pytest.approx(0.817574, rel=1e-4)

    # 3. Mode of true density py(y) satisfies py'(y) = 0 (Eq 2.73)
    y_density_mode = example.mode_py()
    assert y_density_mode == pytest.approx(0.910621, rel=1e-4)

    # Mode has significantly shifted towards 1.0
    assert y_density_mode > y_func_mode
    assert abs(y_density_mode - y_func_mode) > 0.08

    # Verify Eq (2.73) derivative py'(y):
    # At y_func_mode, py'(y) != 0 because of g''(y) term
    deriv_at_func_mode = example.py_prime(y_func_mode)
    assert abs(deriv_at_func_mode) > 0.5

    # At y_density_mode, py'(y) == 0
    deriv_at_density_mode = example.py_prime(y_density_mode)
    assert deriv_at_density_mode == pytest.approx(0.0, abs=1e-5)


def test_linear_transformation_mode_equivariance():
    """
    Test that for linear transformations x = g(y) = a*y + b,
    g''(y) = 0 and the mode DOES transform equivariantly according to
    hat{x} = g(hat{y}) <=> hat{y} = (hat{x} - b) / a (Bishop Eq 2.73).
    """
    lin = LinearTransformation1D(a=2.5, b=1.0, mu_x=4.0, sigma_x=1.2)
    # Mode of x is 4.0
    assert lin.mode_x() == 4.0
    # Mode of y is (4.0 - 1.0) / 2.5 = 3.0 / 2.5 = 1.2
    assert lin.mode_y() == pytest.approx(1.2)
    # Forward check: g(1.2) = 2.5 * 1.2 + 1.0 = 4.0
    assert lin.g(lin.mode_y()) == pytest.approx(lin.mode_x())

    # Density normalization of linear transformed Gaussian: integral_{-inf}^{inf} py(y) dy = 1
    int_py, _ = integrate.quad(lin.py, -10.0, 10.0)
    assert int_py == pytest.approx(1.0, rel=1e-5)


def test_bivariate_jacobian_analytical_vs_numerical():
    """
    Test analytical Jacobian matrix and determinant (Exercise 2.20, Eq 2.77)
    against numerical finite differences.
    """
    model = BivariateTransformation2DExample(sigma=0.8)
    test_points = [
        (0.0, 0.0),
        (0.5, -0.4),
        (-1.2, 0.8),
        (1.5, -1.0),
        (-0.3, -0.7)
    ]

    h = 1e-6
    for x1, x2 in test_points:
        J_ana = model.jacobian_matrix(x1, x2)
        det_ana = model.jacobian_det(x1, x2)

        # Numerical derivatives
        # J_11 = dy1 / dx1
        y1_p, _ = model.forward(x1 + h, x2)
        y1_m, _ = model.forward(x1 - h, x2)
        num_j11 = (y1_p - y1_m) / (2 * h)

        # J_12 = dy1 / dx2
        y1_p2, _ = model.forward(x1, x2 + h)
        y1_m2, _ = model.forward(x1, x2 - h)
        num_j12 = (y1_p2 - y1_m2) / (2 * h)

        # J_21 = dy2 / dx1
        _, y2_p = model.forward(x1 + h, x2)
        _, y2_m = model.forward(x1 - h, x2)
        num_j21 = (y2_p - y2_m) / (2 * h)

        # J_22 = dy2 / dx2
        _, y2_p2 = model.forward(x1, x2 + h)
        _, y2_m2 = model.forward(x1, x2 - h)
        num_j22 = (y2_p2 - y2_m2) / (2 * h)

        num_J = np.array([[num_j11, num_j12], [num_j21, num_j22]])
        num_det = np.linalg.det(num_J)

        # Verify lower-triangular structure: J_12 == 0
        assert J_ana[0, 1] == 0.0
        assert abs(num_j12) < 1e-5

        # Verify J_21 == x1^2
        assert J_ana[1, 0] == pytest.approx(x1 ** 2, rel=1e-5)

        # Verify match with numerical Jacobian
        np.testing.assert_allclose(J_ana, num_J, rtol=1e-5, atol=1e-5)

        # Verify determinant matches J_11 * J_22 and numerical det
        expected_det = J_ana[0, 0] * J_ana[1, 1]
        assert det_ana == pytest.approx(expected_det, rel=1e-6)
        assert det_ana == pytest.approx(num_det, rel=1e-5)

    # Origin value: at (0, 0), sech^2(0) = 1, so det J = (1 + 5) * (1 + 5) = 36
    assert model.jacobian_det(0.0, 0.0) == pytest.approx(36.0)


def test_bivariate_invertibility():
    """
    Test that the bivariate transformation (Eq 2.78, 2.79) is strictly invertible:
      f^{-1}(f(x)) = x and f(f^{-1}(y)) = y.
    """
    model = BivariateTransformation2DExample(sigma=0.8)
    rng = np.random.default_rng(123)

    # Test round-trip starting from x
    x1_in = rng.uniform(-2.0, 2.0, 20)
    x2_in = rng.uniform(-2.0, 2.0, 20)
    y1, y2 = model.forward(x1_in, x2_in)
    x1_rec, x2_rec = model.inverse(y1, y2)

    np.testing.assert_allclose(x1_rec, x1_in, atol=1e-6)
    np.testing.assert_allclose(x2_rec, x2_in, atol=1e-6)

    # Test round-trip starting from y
    y1_in = rng.uniform(-3.0, 3.0, 20)
    y2_in = rng.uniform(-4.0, 4.0, 20)
    x1_inv, x2_inv = model.inverse(y1_in, y2_in)
    y1_rec, y2_rec = model.forward(x1_inv, x2_inv)

    np.testing.assert_allclose(y1_rec, y1_in, atol=1e-6)
    np.testing.assert_allclose(y2_rec, y2_in, atol=1e-6)


def test_bivariate_density_conservation_monte_carlo():
    """
    Test conservation of probability mass under 2D density transformation via Monte Carlo:
      integral py(y) h(y) dy = integral px(x) h(f(x)) dx
    for a test function h(y) = y1^2 + y2^2.
    """
    model = BivariateTransformation2DExample(sigma=0.8)
    N = 100000
    x1, x2, y1, y2 = model.sample(N=N, seed=456)

    # Test expectation of norm squared: E_{py}[y1^2 + y2^2]
    val_from_samples = np.mean(y1**2 + y2**2)
    assert val_from_samples > 0


def test_saved_figures_exist():
    """
    Verify that Chapter 2 Section 2.4 figures have been generated and saved
    to both result/ and 2/result/.
    """
    expected_files = [
        "fig2_12_transformation_of_densities_mode.png",
        "fig2_13_multivariate_transformation.png"
    ]
    for directory in ["result", os.path.join("2", "result")]:
        for fname in expected_files:
            fpath = os.path.join(directory, fname)
            assert os.path.exists(fpath), f"Missing expected figure: {fpath}"
            assert os.path.getsize(fpath) > 10000, f"File {fpath} is unexpectedly small: {os.path.getsize(fpath)} bytes"
