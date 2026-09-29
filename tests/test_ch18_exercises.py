r"""Tests for Chapter 18 Exercises: Normalizing Flows (Bishop & Bishop, 2024)."""

import numpy as np
import pytest

from common.exercises_ch18 import (
    verify_exercise_18_1,
    verify_exercise_18_2,
    verify_exercise_18_3,
    verify_exercise_18_4,
    verify_exercise_18_5,
    verify_exercise_18_6,
    verify_exercise_18_7,
    verify_exercise_18_8,
    verify_exercise_18_9,
    verify_exercise_18_10,
    verify_exercise_18_11,
)


def test_exercise_18_1():
    r"""Test Exercise 18.1: J K = I and det(J) = 1 / det(K)."""
    res = verify_exercise_18_1(dim=3, random_state=42)
    assert res["diff_eye"] < 1e-12
    assert res["diff_det"] < 1e-12


def test_exercise_18_2():
    r"""Test Exercise 18.2: Composition and inverse reversal."""
    res = verify_exercise_18_2(n_layers=4, dim=3, random_state=42)
    assert res["reconstruction_error"] < 1e-12


def test_exercise_18_3():
    r"""Test Exercise 18.3: Linear change of variables and volume preservation."""
    res = verify_exercise_18_3(dim=4, random_state=42)
    assert res["diff_identity"] < 1e-5
    assert np.isclose(res["det_J"], 1.0, atol=1e-5)


def test_exercise_18_4():
    r"""Test Exercise 18.4: Autoregressive lower triangular Jacobian and determinant."""
    res = verify_exercise_18_4(dim=4, random_state=42)
    assert res["max_upper_tri_error"] < 1e-5
    assert res["diag_diff"] < 1e-5
    assert np.isclose(res["det_numerical"], res["det_analytic"], rtol=1e-4)


def test_exercise_18_5():
    r"""Test Exercise 18.5: Limit \epsilon -> 0 of ResNet forward equation."""
    errors = verify_exercise_18_5()
    # As epsilon decreases, error should decrease linearly
    assert errors[-1] < 1e-4
    assert errors[-1] < errors[0]
    # Check monotonic decrease
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i]


def test_exercise_18_6():
    r"""Test Exercise 18.6: Limit \epsilon -> 0 of ResNet backward equation."""
    errors = verify_exercise_18_6()
    assert errors[-1] < 1e-4
    assert errors[-1] < errors[0]
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i]


def test_exercise_18_7():
    r"""Test Exercise 18.7: Derivative of loss function via continuous integration."""
    res = verify_exercise_18_7(T=1.0, n_steps=200)
    assert res["diff"] < 1e-2


def test_exercise_18_8():
    r"""Test Exercise 18.8: 1D probability density transformation."""
    res = verify_exercise_18_8(z_val=0.5, delta_t=1e-4)
    assert res["diff"] < 1e-3


def test_exercise_18_9():
    r"""Test Exercise 18.9: Quantile trajectories match flow lines."""
    res = verify_exercise_18_9()
    assert res["max_diff"] < 1e-4


def test_exercise_18_10():
    r"""Test Exercise 18.10: Computational symmetry of CNF forward and inverse."""
    res = verify_exercise_18_10()
    assert res["ratio"] == 1.0


def test_exercise_18_11():
    r"""Test Exercise 18.11: Unbiasedness of Hutchinson's trace estimator."""
    res = verify_exercise_18_11(dim=4, M_samples=5000, random_state=42)
    assert res["diff_gauss"] < 0.2
    assert res["diff_rade"] < 0.2
