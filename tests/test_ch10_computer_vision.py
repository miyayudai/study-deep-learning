"""
tests/test_ch10_computer_vision.py
===================================
Unit tests for Section 10.1 Computer Vision & Section 10.1.1 Image Data
(Bishop & Bishop 2024, Deep Learning: Foundations and Concepts).
"""

import os
from pathlib import Path
import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.computer_vision import (
    ImageData,
    PixelPermutation,
    generate_figure_image_representation,
    generate_figure_spatial_correlation,
    generate_figure_cv_taxonomy,
)


class TestImageData:
    def test_synthetic_natural_scene_shape_and_bounds(self):
        H, W = 32, 48
        img = ImageData.create_synthetic_natural_scene(height=H, width=W, seed=42)
        assert img.shape == (H, W, 3)
        assert np.all(img >= 0.0)
        assert np.all(img <= 1.0)

    def test_synthetic_natural_scene_determinism(self):
        img1 = ImageData.create_synthetic_natural_scene(height=16, width=16, seed=123)
        img2 = ImageData.create_synthetic_natural_scene(height=16, width=16, seed=123)
        np.testing.assert_array_equal(img1, img2)

    def test_grayscale_conversion_bt601(self):
        # Pure Red: 0.299
        pure_red = np.zeros((2, 2, 3))
        pure_red[:, :, 0] = 1.0
        gray_r = ImageData.to_grayscale(pure_red)
        np.testing.assert_allclose(gray_r, 0.299, atol=1e-5)

        # Pure Green: 0.587
        pure_green = np.zeros((2, 2, 3))
        pure_green[:, :, 1] = 1.0
        gray_g = ImageData.to_grayscale(pure_green)
        np.testing.assert_allclose(gray_g, 0.587, atol=1e-5)

        # Pure Blue: 0.114
        pure_blue = np.zeros((2, 2, 3))
        pure_blue[:, :, 2] = 1.0
        gray_b = ImageData.to_grayscale(pure_blue)
        np.testing.assert_allclose(gray_b, 0.114, atol=1e-5)

        # Pure White: 0.299 + 0.587 + 0.114 = 1.0
        pure_white = np.ones((4, 4, 3))
        gray_w = ImageData.to_grayscale(pure_white)
        np.testing.assert_allclose(gray_w, 1.0, atol=1e-5)

    def test_spatial_autocorrelation_properties(self):
        H, W = 64, 64
        img = ImageData.create_synthetic_natural_scene(height=H, width=W, seed=42)
        gray = ImageData.to_grayscale(img)
        lags, autocorr = ImageData.compute_spatial_autocorrelation(gray, max_lag=10)

        # Lag 0 must be exactly 1.0
        assert autocorr[0] == pytest.approx(1.0, abs=1e-5)
        assert len(lags) == 11
        # Natural scenes have high spatial correlation for neighboring pixels
        assert autocorr[1] > 0.8
        assert autocorr[2] > 0.6
        # Autocorrelation should decay with distance
        assert autocorr[0] > autocorr[5]

    def test_constant_image_autocorrelation(self):
        const_img = np.ones((10, 10)) * 0.5
        lags, autocorr = ImageData.compute_spatial_autocorrelation(const_img, max_lag=5)
        np.testing.assert_array_equal(autocorr, np.ones(6))


class TestPixelPermutation:
    def test_permute_preserves_histogram_2d(self):
        img = np.random.RandomState(42).rand(20, 20)
        perm_img, perm = PixelPermutation.permute_image(img, seed=1)
        assert perm_img.shape == img.shape
        # Marginal histogram (sorted values) must be identical
        np.testing.assert_array_almost_equal(np.sort(img.ravel()), np.sort(perm_img.ravel()))

    def test_permute_preserves_histogram_3d(self):
        img = np.random.RandomState(42).rand(16, 16, 3)
        perm_img, perm = PixelPermutation.permute_image(img, seed=1)
        assert perm_img.shape == img.shape
        for c in range(3):
            np.testing.assert_array_almost_equal(
                np.sort(img[:, :, c].ravel()),
                np.sort(perm_img[:, :, c].ravel())
            )

    def test_permute_destroys_spatial_correlation(self):
        H, W = 64, 64
        img = ImageData.create_synthetic_natural_scene(height=H, width=W, seed=42)
        gray = ImageData.to_grayscale(img)
        perm_gray, _ = PixelPermutation.permute_image(gray, seed=42)

        _, ac_natural = ImageData.compute_spatial_autocorrelation(gray, max_lag=5)
        _, ac_perm = ImageData.compute_spatial_autocorrelation(perm_gray, max_lag=5)

        # Natural scene lag-1 correlation is high, permuted image drops near 0
        assert ac_natural[1] > 0.8
        assert abs(ac_perm[1]) < 0.15

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            PixelPermutation.permute_image(np.ones((2, 2, 2, 2)))


class TestFigureGeneration:
    def test_generate_figures(self, tmp_path):
        fig1 = generate_figure_image_representation(save_dir=str(tmp_path))
        assert isinstance(fig1, plt.Figure)
        assert (tmp_path / "fig_10_image_data_representation.png").exists()
        plt.close(fig1)

        fig2 = generate_figure_spatial_correlation(save_dir=str(tmp_path))
        assert isinstance(fig2, plt.Figure)
        assert (tmp_path / "fig_10_spatial_correlation.png").exists()
        plt.close(fig2)

        fig3 = generate_figure_cv_taxonomy(save_dir=str(tmp_path))
        assert isinstance(fig3, plt.Figure)
        assert (tmp_path / "fig_10_computer_vision_taxonomy.png").exists()
        plt.close(fig3)
