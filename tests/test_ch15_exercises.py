"""
Unit tests for Chapter 15 Exercises (Exercises 15.1 through 15.24).
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
"""

import pytest
from common.exercises_ch15 import (
    solve_exercise_15_1, verify_exercise_15_1,
    solve_exercise_15_2, verify_exercise_15_2,
    solve_exercise_15_3, verify_exercise_15_3,
    solve_exercise_15_4, verify_exercise_15_4,
    solve_exercise_15_5, verify_exercise_15_5,
    solve_exercise_15_6, verify_exercise_15_6,
    solve_exercise_15_7, verify_exercise_15_7,
    solve_exercise_15_8, verify_exercise_15_8,
    solve_exercise_15_9, verify_exercise_15_9,
    solve_exercise_15_10, verify_exercise_15_10,
    solve_exercise_15_11, verify_exercise_15_11,
    solve_exercise_15_12, verify_exercise_15_12,
    solve_exercise_15_13, verify_exercise_15_13,
    solve_exercise_15_14, verify_exercise_15_14,
    solve_exercise_15_15, verify_exercise_15_15,
    solve_exercise_15_16, verify_exercise_15_16,
    solve_exercise_15_17, verify_exercise_15_17,
    solve_exercise_15_18, verify_exercise_15_18,
    solve_exercise_15_19, verify_exercise_15_19,
    solve_exercise_15_20, verify_exercise_15_20,
    solve_exercise_15_21, verify_exercise_15_21,
    solve_exercise_15_22, verify_exercise_15_22,
    solve_exercise_15_23, verify_exercise_15_23,
    solve_exercise_15_24, verify_exercise_15_24,
    solve_all_exercises,
)


def test_exercise_15_1():
    assert verify_exercise_15_1()


def test_exercise_15_2():
    assert verify_exercise_15_2()


def test_exercise_15_3():
    assert verify_exercise_15_3()


def test_exercise_15_4():
    assert verify_exercise_15_4()


def test_exercise_15_5():
    assert verify_exercise_15_5()


def test_exercise_15_6():
    assert verify_exercise_15_6()


def test_exercise_15_7():
    assert verify_exercise_15_7()


def test_exercise_15_8():
    assert verify_exercise_15_8()


def test_exercise_15_9():
    assert verify_exercise_15_9()


def test_exercise_15_10():
    assert verify_exercise_15_10()


def test_exercise_15_11():
    assert verify_exercise_15_11()


def test_exercise_15_12():
    assert verify_exercise_15_12()


def test_exercise_15_13():
    assert verify_exercise_15_13()


def test_exercise_15_14():
    assert verify_exercise_15_14()


def test_exercise_15_15():
    assert verify_exercise_15_15()


def test_exercise_15_16():
    assert verify_exercise_15_16()


def test_exercise_15_17():
    assert verify_exercise_15_17()


def test_exercise_15_18():
    assert verify_exercise_15_18()


def test_exercise_15_19():
    assert verify_exercise_15_19()


def test_exercise_15_20():
    assert verify_exercise_15_20()


def test_exercise_15_21():
    assert verify_exercise_15_21()


def test_exercise_15_22():
    assert verify_exercise_15_22()


def test_exercise_15_23():
    assert verify_exercise_15_23()


def test_exercise_15_24():
    assert verify_exercise_15_24()


def test_solve_all_exercises():
    all_sols = solve_all_exercises()
    assert len(all_sols) == 24
    for k in range(1, 25):
        key = f"15.{k}"
        assert key in all_sols
        assert "derivation" in all_sols[key]
