"""Tests for Chapter 8 Exercises (8.1 to 8.18).

Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
Chapter 8: Backpropagation
"""

import pytest
import numpy as np

from common.exercises_ch8 import (
    verify_exercise_8_1,
    verify_exercise_8_2,
    verify_exercise_8_3,
    verify_exercise_8_4,
    verify_exercise_8_5,
    verify_exercise_8_6,
    verify_exercise_8_7,
    verify_exercise_8_8,
    verify_exercise_8_9,
    verify_exercise_8_10,
    verify_exercise_8_11,
    verify_exercise_8_12,
    verify_exercise_8_13,
    verify_exercise_8_14,
    verify_exercise_8_15,
    verify_exercise_8_16,
    verify_exercise_8_17,
    verify_exercise_8_18,
    verify_all_ch8_exercises,
)


def test_exercise_8_1_backpropagation():
    res = verify_exercise_8_1()
    assert res["verified"] is True
    assert res["max_err_W1"] < 1e-7


def test_exercise_8_2_matrix_backpropagation():
    res = verify_exercise_8_2()
    assert res["verified"] is True
    assert res["max_err_W0"] < 1e-7


def test_exercise_8_3_taylor_central_difference():
    res = verify_exercise_8_3()
    assert res["verified"] is True
    assert abs(res["slope_central"] - 2.0) < 0.1
    assert abs(res["slope_forward"] - 1.0) < 0.1


def test_exercise_8_4_skip_connections_gradients():
    res = verify_exercise_8_4()
    assert res["verified"] is True
    assert res["max_err_Ws"] < 1e-7


def test_exercise_8_5_forward_jacobian():
    res = verify_exercise_8_5()
    assert res["verified"] is True
    assert res["max_err"] < 1e-7


def test_exercise_8_6_exact_two_layer_hessian():
    res = verify_exercise_8_6()
    assert res["verified"] is True
    assert res["err_H22"] < 1e-6
    assert res["err_H21"] < 1e-6


def test_exercise_8_7_skip_connections_hessian():
    res = verify_exercise_8_7()
    assert res["verified"] is True
    assert res["err_H_skip_skip"] < 1e-6


def test_exercise_8_8_multi_output_outer_product_hessian():
    res = verify_exercise_8_8()
    assert res["verified"] is True
    assert res["max_err"] < 1e-6


def test_exercise_8_9_expected_squared_loss_hessian():
    res = verify_exercise_8_9()
    assert res["verified"] is True
    assert abs(res["residual_curvature_term"]) < 0.02


def test_exercise_8_10_logistic_sigmoid_hessian():
    res = verify_exercise_8_10()
    assert res["verified"] is True
    assert res["max_err"] < 1e-6


def test_exercise_8_11_softmax_hessian():
    res = verify_exercise_8_11()
    assert res["verified"] is True
    assert res["max_err"] < 1e-6


def test_exercise_8_12_sherman_morrison_inverse_hessian():
    res = verify_exercise_8_12()
    assert res["verified"] is True
    assert res["max_rel_err"] < 1e-10


def test_exercise_8_13_symbolic_derivative_swelling():
    res = verify_exercise_8_13()
    assert res["verified"] is True
    assert res["err_autodiff"] < 1e-12


def test_exercise_8_14_logistic_map_trace():
    res = verify_exercise_8_14()
    assert res["verified"] is True
    assert len(res["trace"]) == 4


def test_exercise_8_15_forward_tangent_derivation():
    res = verify_exercise_8_15()
    assert res["verified"] is True
    assert res["err"] < 1e-12


def test_exercise_8_16_reverse_adjoint_derivation():
    res = verify_exercise_8_16()
    assert res["verified"] is True


def test_exercise_8_17_exact_evaluation_at_1_2():
    res = verify_exercise_8_17()
    assert res["verified"] is True
    # 2 + 2 * e^2 = 16.7781121978613
    assert abs(res["val_analytic"] - (2.0 + 2.0 * np.exp(2.0))) < 1e-12
    assert abs(res["val_forward"] - res["val_analytic"]) < 1e-12
    assert abs(res["val_reverse"] - res["val_analytic"]) < 1e-12


def test_exercise_8_18_single_pass_jvp():
    res = verify_exercise_8_18()
    assert res["verified"] is True
    assert res["max_err"] < 1e-12


def test_all_ch8_exercises_master():
    results = verify_all_ch8_exercises()
    assert len(results) == 18
    for key, res in results.items():
        assert res["verified"] is True, f"Exercise {key} failed verification: {res}"
