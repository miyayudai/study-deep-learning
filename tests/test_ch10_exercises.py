"""
tests/test_ch10_exercises.py
============================
Unit tests for Chapter 10 Exercises (10.1 - 10.13)
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
"""

import numpy as np
import pytest

from common.exercises_ch10 import (
    solve_exercise_10_1,
    solve_exercise_10_2,
    solve_exercise_10_3,
    solve_exercise_10_4,
    solve_exercise_10_5,
    solve_exercise_10_6,
    solve_exercise_10_7,
    solve_exercise_10_8,
    solve_exercise_10_9,
    solve_exercise_10_10,
    solve_exercise_10_11,
    solve_exercise_10_12,
    solve_exercise_10_13,
)


class TestChapter10Exercises:
    def test_exercise_10_1_linear_projection_maximization(self):
        w = np.array([3.0, 4.0])
        res = solve_exercise_10_1(w)
        assert res["norm_w"] == pytest.approx(5.0)
        assert res["max_y"] == pytest.approx(5.0)
        np.testing.assert_allclose(res["optimal_x"], [0.6, 0.8])
        assert np.linalg.norm(res["optimal_x"]) == pytest.approx(1.0)

    def test_exercise_10_2_toeplitz_matrix_convolution(self):
        w = np.array([1.0, 2.0, 3.0])
        D = 5
        res = solve_exercise_10_2(D, w)
        assert res["shape"] == (3, 5)
        # Row 0: [1, 2, 3, 0, 0]
        # Row 1: [0, 1, 2, 3, 0]
        # Row 2: [0, 0, 1, 2, 3]
        expected_A = np.array([
            [1.0, 2.0, 3.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 3.0, 0.0],
            [0.0, 0.0, 1.0, 2.0, 3.0],
        ])
        np.testing.assert_allclose(res["matrix_A"], expected_A)

    def test_exercise_10_3_cross_correlation_vs_convolution(self):
        rng = np.random.default_rng(103)
        X = rng.normal(0, 1, (6, 6))
        W = rng.normal(0, 1, (3, 3))
        res = solve_exercise_10_3(X, W)
        assert res["is_equivalent"] is True
        np.testing.assert_allclose(res["cross_correlation"], res["convolution_with_flipped"])

    def test_exercise_10_4_batch_norm_translation_equivariance(self):
        res = solve_exercise_10_4()
        assert "equivariance" in res["explanation"].lower()
        assert res["independent_dimensions"] == ("channels",)
        assert res["shared_dimensions"] == ("batch_size", "height", "width")

    def test_exercise_10_5_continuous_convolution_commutativity(self):
        t = np.linspace(0, 2, 40)
        f = np.exp(-t)
        g = np.sin(np.pi * t)
        res = solve_exercise_10_5(f, g, dt=2.0 / 40)
        assert res["is_commutative"] is True
        assert res["max_diff"] < 1e-12

    def test_exercise_10_6_same_padding_formula(self):
        for M in [1, 3, 5, 7, 9]:
            res = solve_exercise_10_6(M)
            assert res["same_padding_P"] == (M - 1) // 2

    def test_exercise_10_7_strided_conv_dimension(self):
        # W=8, M=3, S=2, P=0 -> (8 - 3) // 2 + 1 = 2 + 1 = 3
        res = solve_exercise_10_7(W=8, M=3, S=2, P=0)
        assert res["W_out"] == 3
        assert res["has_remainder"] is True
        assert res["discarded_pixels"] == 1

        # W=7, M=3, S=2, P=0 -> (7 - 3) // 2 + 1 = 2 + 1 = 3 (no remainder)
        res2 = solve_exercise_10_7(W=7, M=3, S=2, P=0)
        assert res2["W_out"] == 3
        assert res2["has_remainder"] is False

    def test_exercise_10_8_vgg16_parameter_reduction(self):
        res = solve_exercise_10_8(C=64)
        assert res["ratio_5x5"] == pytest.approx(18.0 / 25.0)
        assert res["ratio_7x7"] == pytest.approx(27.0 / 49.0)

    def test_exercise_10_9_separable_2d_convolution(self):
        u = np.array([1.0, 2.0, 1.0])
        v = np.array([2.0, -1.0, 3.0])
        rng = np.random.default_rng(109)
        X = rng.normal(0, 1, (7, 7))
        res = solve_exercise_10_9(u, v, X)
        assert res["is_equal"] is True
        assert res["ops_standard"] == 9
        assert res["ops_separable"] == 6
        assert res["theoretical_speedup"] == pytest.approx(1.5)

    def test_exercise_10_10_deepdream_gradient(self):
        rng = np.random.default_rng(110)
        a = rng.normal(0, 1, (4, 4, 3))
        res = solve_exercise_10_10(a)
        assert res["is_identical_to_activation"] is True
        np.testing.assert_allclose(res["analytical_gradient"], a)

    def test_exercise_10_11_object_detection_probabilities(self):
        logits_single = np.array([0.5, 1.2, -0.3])
        logit_obj = 1.0
        logits_cond = np.array([1.2, -0.3])
        res = solve_exercise_10_11(logits_single, logit_obj, logits_cond)
        assert res["p_single"].shape == (3,)
        assert np.sum(res["p_single"]) == pytest.approx(1.0)
        assert 0.0 <= res["p_obj"] <= 1.0
        assert np.sum(res["p_cond"]) == pytest.approx(1.0)

    def test_exercise_10_12_sliding_window_speedup(self):
        res = solve_exercise_10_12()
        assert res["naive_multiplications"] == 1656
        assert res["conv_multiplications"] == 504
        assert res["speedup_factor"] == pytest.approx(1656.0 / 504.0)

    def test_exercise_10_13_transposed_convolution_duality(self):
        # x length 5, y length 2, kernel length 3, stride 2
        # (5 - 3) // 2 + 1 = 2
        x = np.array([1.0, -2.0, 3.0, 0.5, -1.0])
        y = np.array([2.5, -1.5])
        w = np.array([0.8, -1.2, 0.4])
        res = solve_exercise_10_13(x, y, w, S=2)
        assert res["is_dual"] is True
        assert res["matches_direct_transposed_conv"] is True
        assert res["inner_product_Ax_y"] == pytest.approx(res["inner_product_x_ATy"])
