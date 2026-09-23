"""
tests/test_ch10_image_segmentat.py
==================================
Unit tests for Section 10.5 Image Segmentation
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
"""

import os
from pathlib import Path
import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.image_segmentation import (
    Unpooling,
    conv_transpose2d,
    generate_figure_10_26,
    generate_figure_10_27,
    generate_figure_10_28,
    generate_figure_10_29,
    generate_figure_10_30,
    generate_figure_10_31,
)


class TestUnpooling:
    def test_average_unpool2d(self):
        x = np.array([[1.0, 2.0],
                      [3.0, 4.0]])
        out = Unpooling.average_unpool2d(x, scale=2)
        assert out.shape == (4, 4)
        expected = np.array([
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 4.0],
        ])
        np.testing.assert_allclose(out, expected)

    def test_max_unpool2d_fixed(self):
        x = np.array([[1.0, 2.0],
                      [3.0, 4.0]])
        out = Unpooling.max_unpool2d_fixed(x, scale=2)
        assert out.shape == (4, 4)
        expected = np.array([
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        np.testing.assert_allclose(out, expected)

    def test_max_pool_with_indices_fig_10_29(self):
        # Bishop Figure 10.29 exact 4x4 numerical example
        x = np.array([
            [5, 2, 4, 2],
            [7, 1, 0, 3],
            [7, 4, 3, 8],
            [9, 6, 8, 9]
        ], dtype=np.int64)

        pooled, indices = Unpooling.max_pool_with_indices(x, pool_size=2)
        assert pooled.shape == (2, 2)
        assert indices.shape == (2, 2)

        # Expected pooled values:
        # TL block: [5, 2 / 7, 1] -> max 7 at (1, 0) -> flat index 2
        # TR block: [4, 2 / 0, 3] -> max 4 at (0, 0) -> flat index 0
        # BL block: [7, 4 / 9, 6] -> max 9 at (1, 0) -> flat index 2
        # BR block: [3, 8 / 8, 9] -> max 9 at (1, 1) -> flat index 3
        np.testing.assert_array_equal(pooled, [[7, 4], [9, 9]])
        np.testing.assert_array_equal(indices, [[2, 0], [2, 3]])

        # Test unpooling with indices
        unpooled = Unpooling.max_unpool_with_indices(pooled, indices, pool_size=2)
        assert unpooled.shape == (4, 4)
        expected_unpooled = np.array([
            [0, 0, 4, 0],
            [7, 0, 0, 0],
            [0, 0, 0, 0],
            [9, 0, 0, 9]
        ], dtype=np.int64)
        np.testing.assert_array_equal(unpooled, expected_unpooled)


class TestTransposedConvolution:
    def test_output_shape(self):
        # Hin = 4, M = 3, stride = 2, padding = 0 -> (4-1)*2 + 3 = 9
        x = np.ones((4, 4))
        w = np.ones((3, 3))
        out = conv_transpose2d(x, w, stride=2, padding=0)
        assert out.shape == (9, 9)

        # With padding = 1 -> 9 - 2*1 = 7
        out_pad = conv_transpose2d(x, w, stride=2, padding=1)
        assert out_pad.shape == (7, 7)

    def test_fig_10_30_overlap(self):
        # Input 2x2: [[z1, z2], [0, 0]]
        # Kernel 3x3 of all ones
        # Top-left patch: cols 0, 1, 2
        # Top-right patch: cols 2, 3, 4
        # Col 2 is the overlap column and should equal z1 + z2
        z1, z2 = 2.0, 5.0
        x = np.array([[z1, z2],
                      [0.0, 0.0]])
        k = np.ones((3, 3))
        out = conv_transpose2d(x, k, stride=2, padding=0)
        assert out.shape == (5, 5)

        # Overlap in row 0, col 2 should be z1 + z2
        assert out[0, 2] == pytest.approx(z1 + z2)
        assert out[0, 0] == pytest.approx(z1)
        assert out[0, 4] == pytest.approx(z2)

    def test_matrix_transpose_duality_1d(self):
        # Exercise 10.13: Transposed convolution matches matrix transpose A^T x
        # Let downsampling strided convolution be y = A x
        # x: length 5, kernel: length 3 [w0, w1, w2], stride: 2
        # Output y length = (5 - 3) // 2 + 1 = 2
        # y[0] = w0*x[0] + w1*x[1] + w2*x[2]
        # y[1] = w0*x[2] + w1*x[3] + w2*x[4]
        # A matrix is 2x5:
        w = np.array([1.5, -2.0, 0.5])
        A = np.array([
            [w[0], w[1], w[2], 0.0, 0.0],
            [0.0, 0.0, w[0], w[1], w[2]]
        ])
        y = np.array([3.0, 4.0])

        # Dual operation: A^T y
        expected = A.T @ y

        # Compute via 1D transposed convolution expansion:
        # Hin = 2, kernel = 3, stride = 2 -> Hout = (2-1)*2 + 3 = 5
        out_trans = np.zeros(5)
        for i in range(2):
            out_trans[i * 2: i * 2 + 3] += y[i] * w

        np.testing.assert_allclose(out_trans, expected)


class TestFigureGenerations:
    def test_figures_exist_and_generate(self, tmp_path):
        tmp_dir = str(tmp_path)
        fig26 = generate_figure_10_26(save_dir=tmp_dir)
        fig27 = generate_figure_10_27(save_dir=tmp_dir)
        fig28 = generate_figure_10_28(save_dir=tmp_dir)
        fig29 = generate_figure_10_29(save_dir=tmp_dir)
        fig30 = generate_figure_10_30(save_dir=tmp_dir)
        fig31 = generate_figure_10_31(save_dir=tmp_dir)

        assert isinstance(fig26, plt.Figure)
        assert isinstance(fig27, plt.Figure)
        assert isinstance(fig28, plt.Figure)
        assert isinstance(fig29, plt.Figure)
        assert isinstance(fig30, plt.Figure)
        assert isinstance(fig31, plt.Figure)

        plt.close('all')

        for fnum in range(26, 32):
            fpath = os.path.join(tmp_dir, f"Figure_10_{fnum}.png")
            assert os.path.isfile(fpath)
            assert os.path.getsize(fpath) > 1000
