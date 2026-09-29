"""Tests for Chapter 20 Exercises (Exercises 20.1 〜 20.20)."""

import pytest
import numpy as np
from common.exercises_ch20 import (
    verify_exercise_20_1,
    verify_exercise_20_2,
    verify_exercise_20_3,
    verify_exercise_20_4,
    verify_exercise_20_5,
    verify_exercise_20_6,
    verify_exercise_20_7,
    verify_exercise_20_8,
    verify_exercise_20_9,
    verify_exercise_20_10,
    verify_exercise_20_11,
    verify_exercise_20_12,
    verify_exercise_20_13,
    verify_exercise_20_14,
    verify_exercise_20_15,
    verify_exercise_20_16,
    verify_exercise_20_17,
    verify_exercise_20_18,
    verify_exercise_20_19,
    verify_exercise_20_20,
)


class TestChapter20Exercises:
    """Comprehensive test suite covering all 20 exercises of Chapter 20."""

    def test_exercise_20_1_gaussian_marginalization(self):
        res = verify_exercise_20_1(random_state=42)
        assert res["mean_diff"] < 0.05
        assert res["var_diff"] < 0.05

    def test_exercise_20_2_snr_monotonic_decrease(self):
        res = verify_exercise_20_2(T=100)
        assert res["is_strictly_decreasing"] is True
        assert res["snr_start"] > res["snr_end"]

    def test_exercise_20_3_cosine_schedule_derivative(self):
        res = verify_exercise_20_3(T=1000)
        assert res["rel_err"] < 0.05

    def test_exercise_20_4_posterior_parameters(self):
        res = verify_exercise_20_4(random_state=42)
        assert res["diff_mu"] < 1e-12
        assert res["diff_beta"] < 1e-12

    def test_exercise_20_5_boundary_at_t1(self):
        res = verify_exercise_20_5(random_state=42)
        assert res["diff_x"] < 1e-12
        assert res["beta_1"] == pytest.approx(0.0, abs=1e-12)

    def test_exercise_20_6_asymptotic_kl_to_prior(self):
        res = verify_exercise_20_6(T=1000)
        assert res["alpha_bar_T"] < 1e-4
        assert res["kl_to_prior"] < 0.01

    def test_exercise_20_7_telescoping_product(self):
        res = verify_exercise_20_7(T=10)
        assert res["diff"] < 1e-12

    def test_exercise_20_8_gaussian_kl_equal_isotropic_cov(self):
        res = verify_exercise_20_8(D=3, random_state=42)
        assert res["diff"] < 0.05

    def test_exercise_20_9_noise_parameterization(self):
        res = verify_exercise_20_9(random_state=42)
        assert res["diff"] < 1e-12

    def test_exercise_20_10_mean_vs_noise_loss(self):
        res = verify_exercise_20_10(random_state=42)
        assert res["max_diff"] < 1e-12

    def test_exercise_20_11_standard_normal_score(self):
        res = verify_exercise_20_11(random_state=42)
        assert res["diff"] < 1e-4

    def test_exercise_20_12_implicit_score_matching_ibp(self):
        res = verify_exercise_20_12(random_state=42)
        assert res["diff"] < 0.05

    def test_exercise_20_13_vincent_theorem(self):
        res = verify_exercise_20_13(sigma=0.5, random_state=42)
        assert res["diff"] < 1e-12

    def test_exercise_20_14_conditional_score_to_noise(self):
        res = verify_exercise_20_14(random_state=42)
        assert res["diff"] < 1e-12

    def test_exercise_20_15_continuous_limit_vp_sde(self):
        res = verify_exercise_20_15(T=1000)
        assert res["diff"] < 1e-4

    def test_exercise_20_16_probability_flow_ode(self):
        res = verify_exercise_20_16(random_state=42)
        assert not np.isnan(res["ode_drift"])

    def test_exercise_20_17_classifier_guidance_bayes(self):
        res = verify_exercise_20_17()
        assert res["grad_ln_p_y"] == 0.0

    def test_exercise_20_18_classifier_guidance_mean_shift(self):
        res = verify_exercise_20_18(gamma=2.5, random_state=42)
        assert res["diff"] < 1e-12

    def test_exercise_20_19_cfg_score_difference(self):
        res = verify_exercise_20_19(gamma=3.0, random_state=42)
        assert res["diff"] < 1e-12

    def test_exercise_20_20_cfg_interpolation_extrapolation(self):
        res = verify_exercise_20_20()
        assert res["is_g0_uncond"] is True
        assert res["is_g1_cond"] is True
        assert res["is_g2_extrap"] is True
