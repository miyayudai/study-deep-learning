"""
tests/test_ch9_residual_connec.py
=================================
Unit tests for Section 9.5 Residual Connections
(Bishop & Bishop 2024, He et al. 2015a, Balduzzi et al. 2017, Li et al. 2017).
"""

import os
import numpy as np
import pytest
from common.residual_connections import (
    DeepFeedforwardNetwork,
    DeepResidualNetwork,
    ResidualBlock,
    LossLandscapeSimulator,
    generate_figure_9_12,
    generate_figure_9_13,
    generate_figure_9_14,
    generate_figure_9_15,
    generate_figure_9_16,
)


class TestDeepFeedforwardNetwork:
    def test_2_layer_network(self):
        net = DeepFeedforwardNetwork(d_in=1, d_hidden=16, d_out=1, n_layers=2, seed=1)
        x = np.linspace(-1, 1, 20)
        jacs = net.jacobian(x)
        assert len(jacs) == 20
        # 2 layers should have smooth Jacobian with minimal sign changes
        sign_changes = np.sum(np.diff(np.sign(jacs)) != 0)
        assert sign_changes <= 5

    def test_shattered_gradients_25_layers(self):
        net = DeepFeedforwardNetwork(d_in=1, d_hidden=30, d_out=1, n_layers=25, seed=10)
        x = np.linspace(-2, 2, 200)
        jacs = net.jacobian(x)
        assert len(jacs) == 200
        # Deep plain ReLU network experiences shattered gradients (many sign changes)
        sign_changes = np.sum(np.diff(np.sign(jacs)) != 0)
        assert sign_changes >= 10


class TestDeepResidualNetwork:
    def test_resnet_jacobian_stability(self):
        resnet = DeepResidualNetwork(d_in=1, d_hidden=20, d_out=1, n_blocks=25, seed=5)
        x = np.linspace(-2, 2, 200)
        jacs = resnet.jacobian(x)
        assert len(jacs) == 200
        # ResNet Jacobian maintains continuity and does not shatter into white noise
        assert np.all(np.isfinite(jacs))


class TestResidualBlock:
    def test_post_activation(self):
        block = ResidualBlock(d_in=8, d_out=8, variant="post_activation", seed=42)
        x = np.random.randn(5, 8)
        out = block.forward(x)
        assert out.shape == (5, 8)

    def test_pre_activation(self):
        block = ResidualBlock(d_in=8, d_out=8, variant="pre_activation", seed=42)
        x = np.random.randn(5, 8)
        out = block.forward(x)
        assert out.shape == (5, 8)

    def test_dimension_matching_projection(self):
        # When d_in != d_out, projection shortcut is used (Eq 9.41)
        block = ResidualBlock(d_in=8, d_out=16, variant="post_activation", seed=42)
        assert block.W_proj is not None
        assert block.W_proj.shape == (8, 16)
        x = np.random.randn(5, 8)
        out = block.forward(x)
        assert out.shape == (5, 16)


class TestLossLandscapeSimulator:
    def test_surface_shapes_and_smoothness(self):
        X, Y, Z_without, Z_with = LossLandscapeSimulator.compute_surfaces(grid_size=30)
        assert X.shape == (30, 30)
        assert Y.shape == (30, 30)
        assert Z_without.shape == (30, 30)
        assert Z_with.shape == (30, 30)

        # Rugged surface should have higher laplacian / roughness than smooth bowl
        lap_without = np.abs(np.diff(Z_without, n=2, axis=0)).mean()
        lap_with = np.abs(np.diff(Z_with, n=2, axis=0)).mean()
        assert lap_without > lap_with


class TestFigureGenerators:
    def test_figure_9_12(self, tmp_path):
        save_dir = str(tmp_path)
        fig = generate_figure_9_12(save_dir=save_dir)
        assert os.path.exists(os.path.join(save_dir, "Figure_9_12.png"))
        assert os.path.exists(os.path.join(save_dir, "fig_9_12_shattered_gradients.png"))

    def test_figure_9_13(self, tmp_path):
        save_dir = str(tmp_path)
        fig = generate_figure_9_13(save_dir=save_dir)
        assert os.path.exists(os.path.join(save_dir, "Figure_9_13.png"))
        assert os.path.exists(os.path.join(save_dir, "fig_9_13_residual_network_architecture.png"))

    def test_figure_9_14(self, tmp_path):
        save_dir = str(tmp_path)
        fig = generate_figure_9_14(save_dir=save_dir)
        assert os.path.exists(os.path.join(save_dir, "Figure_9_14.png"))
        assert os.path.exists(os.path.join(save_dir, "fig_9_14_loss_landscapes_3d.png"))

    def test_figure_9_15(self, tmp_path):
        save_dir = str(tmp_path)
        fig = generate_figure_9_15(save_dir=save_dir)
        assert os.path.exists(os.path.join(save_dir, "Figure_9_15.png"))
        assert os.path.exists(os.path.join(save_dir, "fig_9_15_residual_expanded_ensemble.png"))

    def test_figure_9_16(self, tmp_path):
        save_dir = str(tmp_path)
        fig = generate_figure_9_16(save_dir=save_dir)
        assert os.path.exists(os.path.join(save_dir, "Figure_9_16.png"))
        assert os.path.exists(os.path.join(save_dir, "fig_9_16_residual_block_variants.png"))
