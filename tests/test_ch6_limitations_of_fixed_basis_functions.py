"""
Unit tests for Chapter 6 Section 6.1: Limitations of Fixed Basis Functions.
Bishop & Bishop (2024), pp. 171-180.

Covers:
- Section 6.1.1: Polynomial feature count combinatorics and grid cell explosion
- Section 6.1.2: Hypersphere volume shell fraction & Gaussian radial density concentration
- Section 6.1.3: Data manifolds and dimensionality advantages
- Section 6.1.4: Data-dependent basis functions (Radial Basis Functions)
- Figures 6.1 through 6.8 reproduction verification
"""

import os
import numpy as np
import pytest
import scipy.integrate as integrate
from sklearn.datasets import load_iris

from common.basis_limitations import (
    GridClassifier,
    RadialBasisFunctions,
    gaussian_radial_density,
    gaussian_radial_mode,
    generate_all_section_6_1_figures,
    generate_figure_6_1,
    generate_figure_6_2,
    generate_figure_6_3,
    generate_figure_6_4,
    generate_figure_6_5,
    generate_figure_6_6,
    generate_figure_6_7,
    generate_figure_6_8,
    grid_cell_count,
    hypersphere_volume_shell_fraction,
    polynomial_feature_count,
)


class TestCurseOfDimensionality:
    """Tests for Section 6.1.1: The curse of dimensionality."""

    def test_polynomial_feature_count(self):
        """Verify binom(D + M, M) formula for independent polynomial coefficients."""
        # 1D polynomial of order 3: 1, x, x^2, x^3 -> 4 terms
        assert polynomial_feature_count(D=1, M=3) == 4

        # 2D polynomial of order 2: 1, x1, x2, x1^2, x1*x2, x2^2 -> 6 terms
        assert polynomial_feature_count(D=2, M=2) == 6

        # 3D polynomial of order 3: binom(6, 3) = 20 terms
        assert polynomial_feature_count(D=3, M=3) == 20

        # D=10, M=3: binom(13, 3) = 286 terms
        assert polynomial_feature_count(D=10, M=3) == 286

        # D=100, M=3: binom(103, 3) = 176,851 terms (O(D^3))
        assert polynomial_feature_count(D=100, M=3) == 176851

        with pytest.raises(ValueError):
            polynomial_feature_count(D=0, M=2)
        with pytest.raises(ValueError):
            polynomial_feature_count(D=2, M=-1)

    def test_grid_cell_count_explosion(self):
        """Verify K^D exponential scaling of regular grid partitioning."""
        assert grid_cell_count(num_intervals_per_dim=3, D=1) == 3
        assert grid_cell_count(num_intervals_per_dim=3, D=2) == 9
        assert grid_cell_count(num_intervals_per_dim=3, D=3) == 27
        assert grid_cell_count(num_intervals_per_dim=3, D=10) == 59049

        with pytest.raises(ValueError):
            grid_cell_count(0, 2)

    def test_grid_classifier(self):
        """Verify GridClassifier fits Iris data, predicts majority class, and handles empty cells."""
        iris = load_iris()
        X = iris.data[:, :2]
        y = iris.target

        clf = GridClassifier(x_bins=4, y_bins=4)
        clf.fit(X, y, num_classes=3)

        preds = clf.predict(X)
        assert len(preds) == len(y)

        # Most training points should be classified correctly
        valid_mask = (preds >= 0)
        acc = np.mean(preds[valid_mask] == y[valid_mask])
        assert acc >= 0.70

        # Test query in unpopulated region (e.g. far corner [4.2, 4.4])
        empty_pred = clf.predict(np.array([[4.25, 4.45]]))
        assert empty_pred[0] == -1


class TestHighDimensionalSpaces:
    """Tests for Section 6.1.2: High-dimensional spaces geometry."""

    def test_hypersphere_volume_shell_fraction_bounds(self):
        """Verify 1 - (1 - eps)^D volume fraction properties."""
        eps = np.linspace(0.0, 1.0, 50)
        for D in [1, 2, 5, 20]:
            frac = hypersphere_volume_shell_fraction(eps, D)
            assert np.isclose(frac[0], 0.0)
            assert np.isclose(frac[-1], 1.0)
            assert np.all(np.diff(frac) >= 0.0), "Fraction must be monotonically increasing"

        # At eps = 0.1 for D = 20: 1 - 0.9^20 = 1 - 0.121576 = 0.8784
        frac_20_01 = hypersphere_volume_shell_fraction(0.1, 20)
        assert np.isclose(frac_20_01, 1.0 - 0.9 ** 20, atol=1e-10)
        assert frac_20_01 > 0.85, "Over 85% of volume in outer 10% shell in 20D"

    def test_gaussian_radial_density_integral_normalization(self):
        """Verify int_0^inf p(r) dr = 1 for D in {1, 2, 5, 20}."""
        sigma = 0.5
        for D in [1, 2, 5, 20]:
            integral, _ = integrate.quad(
                lambda r: gaussian_radial_density(r, D=D, sigma=sigma),
                0.0, 15.0, limit=200
            )
            assert np.isclose(integral, 1.0, atol=1e-3), f"p(r) must integrate to 1 for D={D}"

    def test_gaussian_radial_density_mode(self):
        """Verify mode location r_hat = sqrt(D - 1) * sigma."""
        sigma = 0.5
        for D in [1, 2, 5, 20]:
            expected_mode = gaussian_radial_mode(D, sigma=sigma)
            if D == 1:
                assert expected_mode == 0.0
                p_0 = gaussian_radial_density(0.0, D=1, sigma=sigma)
                p_eps = gaussian_radial_density(0.05, D=1, sigma=sigma)
                assert p_0 > p_eps
            else:
                # Dense grid around theoretical mode
                r_fine = np.linspace(expected_mode - 0.1, expected_mode + 0.1, 1000)
                densities = gaussian_radial_density(r_fine, D=D, sigma=sigma)
                numerical_mode = r_fine[np.argmax(densities)]
                assert np.isclose(numerical_mode, expected_mode, atol=1e-3)

    def test_gaussian_density_specific_values(self):
        """Verify exact values matching Bishop Figure 6.5."""
        sigma = 0.5
        # D = 1 at r = 0: p(0) = 2 / (0.5 * sqrt(2*pi)) = 4 / sqrt(2*pi) ~= 1.595769
        p_d1_0 = gaussian_radial_density(0.0, D=1, sigma=sigma)
        expected_d1_0 = 4.0 / np.sqrt(2.0 * np.pi)
        assert np.isclose(p_d1_0, expected_d1_0)

        # D = 2 at r = 0.5: p(0.5) = 4 * 0.5 * exp(-2 * 0.25) = 2 * exp(-0.5) ~= 1.21306
        p_d2_mode = gaussian_radial_density(0.5, D=2, sigma=sigma)
        expected_d2_mode = 2.0 * np.exp(-0.5)
        assert np.isclose(p_d2_mode, expected_d2_mode)


class TestDataDependentBasisFunctions:
    """Tests for Section 6.1.4: Radial Basis Functions (Eq 6.6)."""

    def test_rbf_transformation_properties(self):
        """Verify phi_n(x) = exp(-||x - x_n||^2 / s^2)."""
        centers = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        rbf = RadialBasisFunctions(centers=centers, scale=1.0, include_bias=True)
        assert rbf.num_features == 4

        X = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0]
        ])
        Phi = rbf.transform(X)

        assert Phi.shape == (3, 4)
        # Bias column must be 1.0
        assert np.allclose(Phi[:, 0], 1.0)
        # Point at center 0: distance 0 -> exp(0) = 1.0
        assert np.isclose(Phi[0, 1], 1.0)
        # Point at center 1: distance 0 -> exp(0) = 1.0
        assert np.isclose(Phi[1, 2], 1.0)
        # Point [2, 0] from center [0, 0]: distance squared is 4 -> exp(-4)
        assert np.isclose(Phi[2, 1], np.exp(-4.0))


class TestFigureGeneration:
    """Tests for Figures 6.1 to 6.8 generation functions."""

    def test_all_section_6_1_figures(self, tmp_path):
        figs = generate_all_section_6_1_figures(result_dirs=[str(tmp_path)])
        assert len(figs) == 8
        for fig_name in ["fig_6_1", "fig_6_2", "fig_6_3", "fig_6_4",
                         "fig_6_5", "fig_6_6", "fig_6_7", "fig_6_8"]:
            assert fig_name in figs
            assert hasattr(figs[fig_name], "savefig")

        # Verify all files were saved
        expected_files = [
            "fig_6_1_iris_data.png",
            "fig_6_2_grid_partitioning.png",
            "fig_6_3_curse_of_dimensionality_grid.png",
            "fig_6_4_hypersphere_volume_fraction.png",
            "fig_6_5_gaussian_radial_density.png",
            "fig_6_6_dimension_projection_separability.png",
            "fig_6_7_digit_manifold.png",
            "fig_6_8_natural_vs_random_images.png"
        ]
        for fname in expected_files:
            fpath = tmp_path / fname
            assert fpath.exists(), f"Expected figure {fname} to exist"
            assert fpath.stat().st_size > 1000, f"Expected {fname} to have non-zero size"
