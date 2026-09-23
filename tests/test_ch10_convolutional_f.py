"""
tests/test_ch10_convolutional_f.py
===================================
Unit tests for Section 10.2 Convolutional Filters
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
"""

import os
from pathlib import Path
import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.convolutional_filters import (
    compute_conv_output_dim,
    pad2d,
    conv1d,
    conv2d,
    max_pool2d,
    avg_pool2d,
    compute_effective_receptive_field,
    analyze_vgg16_parameters,
    VERTICAL_EDGE_FILTER,
    HORIZONTAL_EDGE_FILTER,
    generate_figure_10_1,
    generate_figure_10_2,
    generate_figure_10_3,
    generate_figure_10_4,
    generate_figure_10_5,
    generate_figure_10_6,
    generate_figure_10_7,
    generate_figure_10_8,
    generate_figure_10_9,
    generate_figure_10_10,
)


class TestConvolutionDimensions:
    def test_output_dim_formula(self):
        # Equation (10.5): floor((in + 2P - M) / S) + 1
        # J=3, M=2, P=0, S=1 -> (3 - 2)/1 + 1 = 2
        assert compute_conv_output_dim(3, 2, stride=1, padding=0) == 2
        # J=4, M=3, P=1, S=1 -> (4 + 2 - 3)/1 + 1 = 4 (Same convolution)
        assert compute_conv_output_dim(4, 3, stride=1, padding=1) == 4
        # J=224, M=3, P=1, S=1 -> 224
        assert compute_conv_output_dim(224, 3, stride=1, padding=1) == 224
        # J=224, M=2, P=0, S=2 -> (224 - 2)/2 + 1 = 112 (MaxPool downsampling)
        assert compute_conv_output_dim(224, 2, stride=2, padding=0) == 112
        # J=7, M=3, P=0, S=2 -> floor(4/2) + 1 = 3
        assert compute_conv_output_dim(7, 3, stride=2, padding=0) == 3

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            compute_conv_output_dim(5, 3, stride=0)
        with pytest.raises(ValueError):
            compute_conv_output_dim(3, 5, stride=1, padding=0)


class TestPadding:
    def test_pad2d_shapes(self):
        img_2d = np.ones((4, 4))
        padded_2d = pad2d(img_2d, padding=1, constant_value=0.0)
        assert padded_2d.shape == (6, 6)
        assert padded_2d[0, 0] == 0.0
        assert padded_2d[1, 1] == 1.0

        img_3d = np.ones((4, 4, 3))
        padded_3d = pad2d(img_3d, padding=2, constant_value=5.0)
        assert padded_3d.shape == (8, 8, 3)
        assert padded_3d[0, 0, 0] == 5.0
        assert padded_3d[2, 2, 0] == 1.0


class TestConv1D:
    def test_conv1d_parameter_sharing(self):
        # Figure 10.2: 5 inputs, kernel size 2 -> 4 outputs
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        kernel = np.array([0.5, -0.5])
        out = conv1d(x, kernel, stride=1, padding=0)
        assert len(out) == 4
        # Expected:
        # z1 = 1*0.5 + 2*(-0.5) = -0.5
        # z2 = 2*0.5 + 3*(-0.5) = -0.5
        # z3 = 3*0.5 + 4*(-0.5) = -0.5
        # z4 = 4*0.5 + 5*(-0.5) = -0.5
        np.testing.assert_allclose(out, [-0.5, -0.5, -0.5, -0.5])


class TestConv2D:
    def test_figure_10_3_algebraic_conformance(self):
        # 3x3 image I with distinct test values
        # a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9
        I = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        # 2x2 kernel K
        # j=10, k=20, l=30, m=40
        K = np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ])

        C = conv2d(I, K, stride=1, padding=0)
        assert C.shape == (2, 2)

        # Expected from Figure 10.3 equations:
        # C11 = aj + bk + dl + em = 1*10 + 2*20 + 4*30 + 5*40 = 10 + 40 + 120 + 200 = 370
        # C12 = bj + ck + el + fm = 2*10 + 3*20 + 5*30 + 6*40 = 20 + 60 + 150 + 240 = 470
        # C21 = dj + ek + gl + hm = 4*10 + 5*20 + 7*30 + 8*40 = 40 + 100 + 210 + 320 = 670
        # C22 = ej + fk + hl + im = 5*10 + 6*20 + 8*30 + 9*40 = 50 + 120 + 240 + 360 = 770
        expected = np.array([
            [370.0, 470.0],
            [670.0, 770.0],
        ])
        np.testing.assert_allclose(C, expected)

    def test_edge_detection_filters(self):
        # Flat region: both edge filters should give 0.0
        flat = np.ones((5, 5)) * 0.7
        res_v = conv2d(flat, VERTICAL_EDGE_FILTER, padding=0)
        res_h = conv2d(flat, HORIZONTAL_EDGE_FILTER, padding=0)
        np.testing.assert_allclose(res_v, 0.0, atol=1e-10)
        np.testing.assert_allclose(res_h, 0.0, atol=1e-10)

        # Vertical step edge: left=0.0, right=1.0
        step_v = np.zeros((5, 5))
        step_v[:, 2:] = 1.0
        res_step_v = conv2d(step_v, VERTICAL_EDGE_FILTER, padding=0)
        # Center column across the edge has positive response:
        # [-1*0, 0*0, 1*1] * 3 = 3.0
        assert np.all(res_step_v[:, 1] > 2.0)

    def test_multi_channel_and_1x1_conv(self):
        H, W, C_in, C_out = 8, 8, 3, 4
        img = np.random.RandomState(42).randn(H, W, C_in)
        # 1x1 convolution across channels
        kernel_1x1 = np.random.RandomState(42).randn(1, 1, C_in, C_out)
        out = conv2d(img, kernel_1x1, stride=1, padding=0)
        assert out.shape == (H, W, C_out)

        # Mathematical check of 1x1 pointwise projection:
        # out[i, j, k] = sum_c img[i, j, c] * kernel_1x1[0, 0, c, k]
        for c_out_idx in range(C_out):
            expected_ij = np.sum(img[2, 3, :] * kernel_1x1[0, 0, :, c_out_idx])
            assert np.isclose(out[2, 3, c_out_idx], expected_ij)


class TestPooling:
    def test_max_pool2d_figure_10_8(self):
        # Test values from Figure 10.8
        input_matrix = np.array([
            [1.2, 3.4, 0.8, 2.1],
            [4.5, 2.0, 1.9, 3.8],
            [0.3, 1.7, 5.2, 4.1],
            [2.8, 3.1, 2.4, 0.9],
        ])
        pooled = max_pool2d(input_matrix, pool_size=2, stride=2)
        expected = np.array([
            [4.5, 3.8],
            [3.1, 5.2],
        ])
        assert pooled.shape == (2, 2)
        np.testing.assert_allclose(pooled, expected)

    def test_avg_pool2d(self):
        input_matrix = np.array([
            [1.0, 3.0],
            [2.0, 4.0],
        ])
        pooled = avg_pool2d(input_matrix, pool_size=2, stride=2)
        assert pooled.shape == (1, 1)
        assert np.isclose(pooled[0, 0], 2.5)


class TestReceptiveFieldAndVGG16:
    def test_receptive_field_growth_figure_10_9(self):
        # Two 3x3 layers with stride 1
        layers = [
            {"kernel_size": 3, "stride": 1},
            {"kernel_size": 3, "stride": 1},
        ]
        rf_records = compute_effective_receptive_field(layers)
        assert rf_records[0]["receptive_field"] == 3
        assert rf_records[1]["receptive_field"] == 5  # Exactly matches Figure 10.9!

    def test_vgg16_parameter_accounting(self):
        vgg = analyze_vgg16_parameters()
        # Bishop textbook page 301:
        # "In total there are roughly 138 million independently learnable parameters in VGG-16,
        # the majority of which (nearly 103 million) are in the first fully connected layer"
        assert 138_000_000 < vgg["total_params"] < 139_000_000
        # FC1 is at layer index 18 (fc6)
        fc1_params = vgg["layers"][18]["total_params"]
        assert 102_000_000 < fc1_params < 103_000_000
        assert fc1_params == 25088 * 4096 + 4096  # Exact formula


class TestFigureGenerators:
    def test_generate_all_section_10_2_figures(self, tmp_path):
        figs = [
            generate_figure_10_1(save_dir=str(tmp_path)),
            generate_figure_10_2(save_dir=str(tmp_path)),
            generate_figure_10_3(save_dir=str(tmp_path)),
            generate_figure_10_4(save_dir=str(tmp_path)),
            generate_figure_10_5(save_dir=str(tmp_path)),
            generate_figure_10_6(save_dir=str(tmp_path)),
            generate_figure_10_7(save_dir=str(tmp_path)),
            generate_figure_10_8(save_dir=str(tmp_path)),
            generate_figure_10_9(save_dir=str(tmp_path)),
            generate_figure_10_10(save_dir=str(tmp_path)),
        ]
        for idx, fig in enumerate(figs, start=1):
            assert isinstance(fig, plt.Figure)
            assert (tmp_path / f"Figure_10_{idx}.png").exists()
            plt.close(fig)
