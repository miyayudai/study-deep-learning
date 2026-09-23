"""
tests/test_ch10_visualizing_tra.py
===================================
Unit tests for Section 10.3 Visualizing Trained CNNs
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
"""

import os
from pathlib import Path
import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.visualizing_cnn import (
    GaborFilter,
    GradCAM,
    FGSM,
    DeepDream,
    generate_figure_10_11,
    generate_figure_10_12,
    generate_figure_10_13,
    generate_figure_10_14,
    generate_figure_10_15,
    generate_figure_10_16,
    generate_figure_10_17,
    generate_figure_10_18,
)


class TestGaborFilter:
    def test_gabor_rotation_equations(self):
        # Equations (10.7) & (10.8):
        # theta = 0 -> x_tilde = x, y_tilde = y
        # G(x, 0) = exp(-alpha * x^2) * sin(omega * x)
        x = np.array([0.5])
        y = np.array([0.0])
        val_th0 = GaborFilter.evaluate(x, y, theta=0.0, omega=2.0, alpha=1.0, beta=1.0)
        expected_th0 = np.exp(-1.0 * 0.25) * np.sin(2.0 * 0.5)
        np.testing.assert_allclose(val_th0, expected_th0, atol=1e-6)

        # theta = pi/2 -> x_tilde = y, y_tilde = -x
        # For (0, 0.5), x_tilde = 0.5, y_tilde = 0 -> should match val_th0!
        x_rot = np.array([0.0])
        y_rot = np.array([0.5])
        val_th_pi2 = GaborFilter.evaluate(x_rot, y_rot, theta=np.pi / 2, omega=2.0, alpha=1.0, beta=1.0)
        np.testing.assert_allclose(val_th_pi2, expected_th0, atol=1e-6)

    def test_gabor_bounds_and_amplitude(self):
        coords = np.linspace(-2, 2, 25)
        xx, yy = np.meshgrid(coords, coords)
        A = 2.5
        g = GaborFilter.evaluate(xx, yy, A=A)
        assert np.max(g) <= A
        assert np.min(g) >= -A

    def test_gabor_kernel_shape(self):
        k = GaborFilter.generate_kernel(size=31)
        assert k.shape == (31, 31)


class TestGradCAM:
    def test_gradcam_weights_and_heatmap(self):
        # Equation (10.9): alpha_k = (1/Mk) sum grad
        H, W, K = 7, 7, 3
        # Dummy gradients: channel 0 has mean 2.0, channel 1 has mean -1.0, channel 2 has mean 0.5
        grad_a = np.zeros((H, W, K))
        grad_a[:, :, 0] = 2.0
        grad_a[:, :, 1] = -1.0
        grad_a[:, :, 2] = 0.5

        alphas = GradCAM.compute_weights(grad_a)
        np.testing.assert_allclose(alphas, [2.0, -1.0, 0.5])

        # Feature activations:
        feature_maps = np.zeros((H, W, K))
        feature_maps[2:5, 2:5, 0] = 1.0  # Hot patch in channel 0
        feature_maps[0:2, 0:2, 1] = 3.0  # Hot patch in negative channel 1

        # Equation (10.10): L = ReLU( sum alpha_k * A^(k) )
        heatmap = GradCAM.compute_heatmap(feature_maps, alphas)
        assert heatmap.shape == (H, W)
        assert np.all(heatmap >= 0.0)
        # Hot patch from channel 0 should have maximum activation 1.0
        assert np.isclose(heatmap[3, 3], 1.0)
        # Negative channel 1 should be suppressed by ReLU to 0.0
        assert np.isclose(heatmap[0, 0], 0.0)


class TestFGSM:
    def test_fgsm_adversarial_perturbation(self):
        # Equation (10.11): x_adv = x + eps * sign(grad)
        img = np.ones((5, 5, 3)) * 0.5
        grad = np.array([
            [1.0, -2.0, 0.5],
            [-0.1, 3.0, -1.2],
        ])  # Shape (2, 3) broadcastable
        grad_full = np.ones((5, 5, 3))
        grad_full[0, 0, 0] = -1.0

        eps = 0.01
        adv_img, pert = FGSM.generate_adversarial_sample(img, grad_full, epsilon=eps)
        assert adv_img.shape == img.shape
        # Check perturbation is exactly +/- eps
        assert np.isclose(pert[0, 0, 0], -eps)
        assert np.isclose(pert[1, 1, 1], eps)
        # Check pixel clipping
        assert np.all(adv_img >= 0.0) and np.all(adv_img <= 1.0)


class TestDeepDream:
    def test_deepdream_objective(self):
        # Equation (10.12): F(I) = sum a_{ijk}^2
        act = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30
        obj = DeepDream.compute_objective(act)
        assert obj == 30.0

    def test_deepdream_gradient_step(self):
        img = np.ones((10, 10, 3)) * 0.5
        grad = np.ones((10, 10, 3)) * 0.2
        new_img = DeepDream.gradient_step(img, grad, step_size=0.1, blur_sigma=0.0)
        # Expected without blur: 0.5 + 0.1 * 0.2 = 0.52
        np.testing.assert_allclose(new_img, 0.52)


class TestFigureGenerators:
    def test_generate_all_section_10_3_figures(self, tmp_path):
        figs = [
            generate_figure_10_11(save_dir=str(tmp_path)),
            generate_figure_10_12(save_dir=str(tmp_path)),
            generate_figure_10_13(save_dir=str(tmp_path)),
            generate_figure_10_14(save_dir=str(tmp_path)),
            generate_figure_10_15(save_dir=str(tmp_path)),
            generate_figure_10_16(save_dir=str(tmp_path)),
            generate_figure_10_17(save_dir=str(tmp_path)),
            generate_figure_10_18(save_dir=str(tmp_path)),
        ]
        for idx, fig in enumerate(figs, start=11):
            assert isinstance(fig, plt.Figure)
            assert (tmp_path / f"Figure_10_{idx}.png").exists()
            plt.close(fig)
