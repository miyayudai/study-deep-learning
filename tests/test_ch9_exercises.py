"""
tests/test_ch9_exercises.py
===========================
Comprehensive unit test suite for Chapter 9 Exercises (Exercises 9.1 - 9.18).
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
"""

import pytest
from common.exercises_ch9 import (
    verify_exercise_9_1,
    verify_exercise_9_2,
    verify_exercise_9_3,
    verify_exercise_9_4,
    verify_exercise_9_5,
    verify_exercise_9_6,
    verify_exercise_9_7,
    verify_exercise_9_8,
    verify_exercise_9_9,
    verify_exercise_9_10,
    verify_exercise_9_11,
    verify_exercise_9_12,
    verify_exercise_9_13,
    verify_exercise_9_14,
    verify_exercise_9_15,
    verify_exercise_9_16,
    verify_exercise_9_17,
    verify_exercise_9_18,
)


def test_exercise_9_1():
    """Exercise 9.1: Group axioms for C_4 square rotations and 2D translations."""
    res = verify_exercise_9_1()
    assert res["passed"] is True
    assert res["c4_axioms"]["closure"] is True
    assert res["c4_axioms"]["associativity"] is True
    assert res["c4_axioms"]["identity"] is True
    assert res["c4_axioms"]["inverse"] is True


def test_exercise_9_2():
    """Exercise 9.2: Input noise equivalence to L2 weight decay."""
    res = verify_exercise_9_2()
    assert res["passed"] is True
    assert res["rel_diff"] < 0.02


def test_exercise_9_3():
    """Exercise 9.3: Quadratic regularizer gradient flow and exponential decay."""
    res = verify_exercise_9_3()
    assert res["passed"] is True
    assert res["rel_diff"] < 0.05


def test_exercise_9_4():
    """Exercise 9.4: Linear scaling equivariance of network inputs and outputs."""
    res = verify_exercise_9_4()
    assert res["passed"] is True
    assert res["input_invariance"] is True
    assert res["output_equivariance"] is True


def test_exercise_9_5():
    """Exercise 9.5: Lagrange multipliers for weight decay and constraint equivalence."""
    res = verify_exercise_9_5()
    assert res["passed"] is True
    assert res["kkt_satisfied"] is True


def test_exercise_9_6():
    """Exercise 9.6: Early stopping vs weight decay spectral shrinkage."""
    res = verify_exercise_9_6()
    assert res["passed"] is True
    assert res["alpha_equiv"] > 0


def test_exercise_9_7():
    """Exercise 9.7: Backpropagation with tied/shared weights."""
    res = verify_exercise_9_7()
    assert res["passed"] is True
    assert res["rel_error"] < 1e-5


def test_exercise_9_8():
    """Exercise 9.8: Responsibilities in soft weight sharing (Bayes theorem)."""
    res = verify_exercise_9_8()
    assert res["passed"] is True
    assert res["sums_to_one"] is True


def test_exercise_9_9():
    """Exercise 9.9: Weight gradient of soft weight sharing regularizer."""
    res = verify_exercise_9_9()
    assert res["passed"] is True
    assert res["rel_error"] < 1e-6


def test_exercise_9_10():
    """Exercise 9.10: Cluster center gradient of soft weight sharing."""
    res = verify_exercise_9_10()
    assert res["passed"] is True
    assert res["rel_error"] < 1e-6


def test_exercise_9_11():
    """Exercise 9.11: Variance parameter gradient of soft weight sharing."""
    res = verify_exercise_9_11()
    assert res["passed"] is True
    assert res["rel_error"] < 1e-6


def test_exercise_9_12():
    """Exercise 9.12: Softmax mixing coefficients and logit gradients."""
    res = verify_exercise_9_12()
    assert res["passed"] is True
    assert res["rel_error"] < 1e-6


def test_exercise_9_13():
    """Exercise 9.13: Recursive expansion of residual connections into path ensemble."""
    res = verify_exercise_9_13()
    assert res["passed"] is True
    assert res["abs_diff"] < 1e-12


def test_exercise_9_14():
    """Exercise 9.14: Committee error reduction factor 1/M under uncorrelated errors."""
    res = verify_exercise_9_14()
    assert res["passed"] is True
    assert abs(res["empirical_ratio"] - res["expected_ratio"]) < 0.03


def test_exercise_9_15():
    """Exercise 9.15: Jensen's inequality proof of committee error bound E_COM <= E_AV."""
    res = verify_exercise_9_15()
    assert res["passed"] is True
    assert res["inequality_satisfied"] is True


def test_exercise_9_16():
    """Exercise 9.16: Generalization of committee error bound to any convex error function."""
    res = verify_exercise_9_16()
    assert res["passed"] is True
    assert res["l1_com"] <= res["l1_av"]
    assert res["huber_com"] <= res["huber_av"]


def test_exercise_9_17():
    """Exercise 9.17: Bounded convex combinations in weighted committees."""
    res = verify_exercise_9_17()
    assert res["passed"] is True
    assert res["valid_combination_bounded"] is True
    assert res["invalid_combination_violates"] is True


def test_exercise_9_18():
    """Exercise 9.18: Dropout on linear regression as input-variance-weighted L2."""
    res = verify_exercise_9_18()
    assert res["passed"] is True
    assert res["grad_at_star_norm"] < 1e-9
