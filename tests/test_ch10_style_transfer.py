"""
tests/test_ch10_style_transfer.py
=================================
Unit tests for Section 10.6 Style Transfer
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
"""

import os
from pathlib import Path
import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.style_transfer import (
    compute_content_loss,
    compute_content_gradient,
    compute_gram_matrix,
    compute_style_loss_layer,
    compute_style_gradient_layer,
    compute_total_loss,
    create_synthetic_content_image,
    create_synthetic_style_image,
    create_stylized_image,
    generate_figure_10_32,
)


class TestContentLoss:
    def test_identical_activations(self):
        act = np.random.randn(8, 8, 4)
        loss = compute_content_loss(act, act)
        assert loss == pytest.approx(0.0)

    def test_analytical_loss_and_gradient(self):
        act_G = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # shape (1, 2, 2)
        act_C = np.array([[[1.0, 0.0], [0.0, 4.0]]])  # shape (1, 2, 2)
        # diff: [[[0, 2], [3, 0]]] -> diff^2 sum = 4 + 9 = 13
        # loss = 0.5 * 13 = 6.5
        loss = compute_content_loss(act_G, act_C)
        assert loss == pytest.approx(6.5)

        grad = compute_content_gradient(act_G, act_C)
        np.testing.assert_allclose(grad, act_G - act_C)

    def test_gradient_finite_difference(self):
        rng = np.random.default_rng(42)
        act_G = rng.normal(0, 1, (4, 4, 3))
        act_C = rng.normal(0, 1, (4, 4, 3))

        analytical_grad = compute_content_gradient(act_G, act_C)

        eps = 1e-6
        numerical_grad = np.zeros_like(act_G)
        for i in range(act_G.shape[0]):
            for j in range(act_G.shape[1]):
                for k in range(act_G.shape[2]):
                    act_plus = act_G.copy()
                    act_plus[i, j, k] += eps
                    loss_plus = compute_content_loss(act_plus, act_C)

                    act_minus = act_G.copy()
                    act_minus[i, j, k] -= eps
                    loss_minus = compute_content_loss(act_minus, act_C)

                    numerical_grad[i, j, k] = (loss_plus - loss_minus) / (2.0 * eps)

        np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-5, atol=1e-5)


class TestGramMatrixAndStyleLoss:
    def test_gram_matrix_properties(self):
        rng = np.random.default_rng(123)
        act = rng.normal(0, 1, (6, 6, 5))
        F = compute_gram_matrix(act)

        # 1. Shape check
        assert F.shape == (5, 5)

        # 2. Symmetry check: F^T == F
        np.testing.assert_allclose(F, F.T)

        # 3. Positive semi-definiteness: eigenvalues >= -1e-10
        eigvals = np.linalg.eigvalsh(F)
        assert np.all(eigvals >= -1e-10)

    def test_gram_matrix_exact_computation(self):
        # 1x2 spatial, 2 channels
        act = np.array([[[1.0, 2.0],
                         [3.0, 4.0]]])  # shape (1, 2, 2)
        # Reshaped A is [[1, 2], [3, 4]]
        # A^T @ A = [[1, 3], [2, 4]] @ [[1, 2], [3, 4]]
        # = [[1+9, 2+12], [2+12, 4+16]] = [[10, 14], [14, 20]]
        F = compute_gram_matrix(act)
        np.testing.assert_allclose(F, [[10.0, 14.0], [14.0, 20.0]])

    def test_style_loss_zero_on_identical(self):
        act = np.random.randn(4, 4, 3)
        loss = compute_style_loss_layer(act, act)
        assert loss == pytest.approx(0.0)

    def test_style_loss_gradient_finite_difference(self):
        rng = np.random.default_rng(999)
        act_G = rng.normal(0, 0.5, (3, 3, 2))
        act_S = rng.normal(0, 0.5, (3, 3, 2))

        analytical_grad = compute_style_gradient_layer(act_G, act_S)

        eps = 1e-6
        numerical_grad = np.zeros_like(act_G)
        for i in range(act_G.shape[0]):
            for j in range(act_G.shape[1]):
                for k in range(act_G.shape[2]):
                    act_plus = act_G.copy()
                    act_plus[i, j, k] += eps
                    loss_plus = compute_style_loss_layer(act_plus, act_S)

                    act_minus = act_G.copy()
                    act_minus[i, j, k] -= eps
                    loss_minus = compute_style_loss_layer(act_minus, act_S)

                    numerical_grad[i, j, k] = (loss_plus - loss_minus) / (2.0 * eps)

        np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-4)


class TestTotalLossAndSynthesizer:
    def test_total_loss_weighting(self):
        act_G = np.ones((4, 4, 2)) * 2.0
        act_C = np.ones((4, 4, 2)) * 1.0
        act_S = np.ones((4, 4, 2)) * 0.5

        total, c_loss, s_loss = compute_total_loss(
            act_G_content=act_G,
            act_C_content=act_C,
            layers_G_style=[act_G],
            layers_S_style=[act_S],
            alpha=2.0,
            beta=100.0,
        )

        assert total == pytest.approx(2.0 * c_loss + 100.0 * s_loss)
        assert c_loss > 0.0
        assert s_loss > 0.0

    def test_synthetic_images_generation(self):
        content = create_synthetic_content_image(40, 50)
        style = create_synthetic_style_image(40, 50)
        stylized = create_stylized_image(content, style)

        assert content.shape == (40, 50, 3)
        assert style.shape == (40, 50, 3)
        assert stylized.shape == (40, 50, 3)
        assert 0.0 <= np.min(stylized) <= np.max(stylized) <= 1.0


class TestFigure1032:
    def test_figure_10_32_generation(self, tmp_path):
        tmp_dir = str(tmp_path)
        fig = generate_figure_10_32(save_dir=tmp_dir)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        out_path = os.path.join(tmp_dir, "Figure_10_32.png")
        assert os.path.isfile(out_path)
        assert os.path.getsize(out_path) > 10000
