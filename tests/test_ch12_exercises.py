"""
Tests for Chapter 12 Exercises (12.1 to 12.16)
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.exercises_ch12 import (
    exercise_12_1_lagrange_multiplier,
    exercise_12_2_verify_softmax,
    exercise_12_3_orthogonal_attention,
    exercise_12_4_expected_inner_product_squared,
    exercise_12_5_multihead_low_rank,
    exercise_12_6_sparse_parameter_sharing,
    exercise_12_7_permutation_equivariance,
    exercise_12_8_high_dim_orthogonality,
    exercise_12_9_concatenation_vs_addition,
    exercise_12_10_sinusoidal_shift,
    exercise_12_11_bow_mle,
    exercise_12_12_table_growth,
    exercise_12_13_ngram_conditional,
    exercise_12_14_rnn_inference,
    exercise_12_15_greedy_vs_global,
    exercise_12_16_bert_large_parameter_count,
    generate_figure_12_ex6,
)


class TestChapter12Exercises:
    def test_exercise_12_1(self):
        res = exercise_12_1_lagrange_multiplier(N=5)
        assert res["verified"] is True
        assert np.isclose(res["max_value"], 1.0)
        assert res["theoretical_bound"] == 1.0

    def test_exercise_12_2(self):
        res = exercise_12_2_verify_softmax(N=6, D=8)
        assert res["verified"] is True
        assert res["all_non_negative"] is True
        assert res["sums_to_one"] is True

    def test_exercise_12_3(self):
        res = exercise_12_3_orthogonal_attention(N=4, scale=10.0)
        assert res["verified"] is True
        assert res["reconstruction_error"] < 1e-3

    def test_exercise_12_4(self):
        res = exercise_12_4_expected_inner_product_squared(D=16, num_samples=30000)
        assert res["verified"] is True
        assert abs(res["empirical_mean"] - 16.0) < 3.0 * res["std_error"]

    def test_exercise_12_5(self):
        res = exercise_12_5_multihead_low_rank(N=6, D=16, H=4)
        assert res["verified"] is True
        assert res["max_diff"] < 1e-7
        assert all(r <= res["Dv"] for r in res["ranks"])

    def test_exercise_12_6(self):
        res = exercise_12_6_sparse_parameter_sharing(N=3, D=2)
        assert res["verified"] is True
        assert res["max_diff"] < 1e-7
        assert res["block_matrix_shape"] == (6, 6)

    def test_exercise_12_7(self):
        res = exercise_12_7_permutation_equivariance(N=5, D=8, num_heads=2)
        assert res["verified"] is True
        assert res["max_error"] < 1e-7

    def test_exercise_12_8(self):
        res = exercise_12_8_high_dim_orthogonality(dimensions=[2, 8, 32, 128], num_trials=2000)
        assert res["verified"] is True
        # Check variance monotonically decreases with D
        vars_cos = [res["results"][D]["var_cos"] for D in [2, 8, 32, 128]]
        assert vars_cos[0] > vars_cos[1] > vars_cos[2] > vars_cos[3]

    def test_exercise_12_9(self):
        res = exercise_12_9_concatenation_vs_addition(D_x=6, D_e=4, D_out=8)
        assert res["verified"] is True
        assert res["max_diff"] < 1e-9

    def test_exercise_12_10(self):
        res = exercise_12_10_sinusoidal_shift(n=5, k=3, D=8)
        assert res["verified"] is True
        assert res["max_diff"] < 1e-7

    def test_exercise_12_11(self):
        res = exercise_12_11_bow_mle()
        assert res["verified"] is True

    def test_exercise_12_12(self):
        res = exercise_12_12_table_growth(V=10, max_n=5)
        assert res["is_exponential"] is True
        assert len(res["table_entries_per_step"]) == 5

    def test_exercise_12_13(self):
        res = exercise_12_13_ngram_conditional()
        assert res["verified"] is True
        assert np.isclose(res["sum_of_probs"], 1.0)

    def test_exercise_12_14(self):
        res = exercise_12_14_rnn_inference(seq_len=6)
        assert res["verified"] is True
        assert len(res["generated_tokens"]) == 6

    def test_exercise_12_15(self):
        res = exercise_12_15_greedy_vs_global()
        assert res["verified"] is True
        assert "greedy_sequence" in res
        assert "global_sequence" in res

    def test_exercise_12_16(self):
        res = exercise_12_16_bert_large_parameter_count()
        assert res["verified"] is True
        # Exactly ~334.6M to 340M
        assert 330.0 < res["approx_millions"] < 350.0

    def test_generate_figure_12_ex6(self, tmp_path):
        save_dir = str(tmp_path)
        fig = generate_figure_12_ex6(save_dir)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
