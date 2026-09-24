"""
Unit tests for Chapter 14 Exercises (Exercises 14.1 through 14.18).
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
"""

import pytest
from common.exercises_ch14 import (
    solve_exercise_14_1, verify_exercise_14_1,
    solve_exercise_14_2, verify_exercise_14_2,
    solve_exercise_14_3, verify_exercise_14_3,
    solve_exercise_14_4, verify_exercise_14_4,
    solve_exercise_14_5, verify_exercise_14_5,
    solve_exercise_14_6, verify_exercise_14_6,
    solve_exercise_14_7, verify_exercise_14_7,
    solve_exercise_14_8, verify_exercise_14_8,
    solve_exercise_14_9, verify_exercise_14_9,
    solve_exercise_14_10, verify_exercise_14_10,
    solve_exercise_14_11, verify_exercise_14_11,
    solve_exercise_14_12, verify_exercise_14_12,
    solve_exercise_14_13, verify_exercise_14_13,
    solve_exercise_14_14, verify_exercise_14_14,
    solve_exercise_14_15, verify_exercise_14_15,
    solve_exercise_14_16, verify_exercise_14_16,
    solve_exercise_14_17, verify_exercise_14_17,
    solve_exercise_14_18, verify_exercise_14_18,
    solve_all_exercises,
)


def test_exercise_14_1():
    assert verify_exercise_14_1()


def test_exercise_14_2():
    assert verify_exercise_14_2()


def test_exercise_14_3():
    assert verify_exercise_14_3()


def test_exercise_14_4():
    assert verify_exercise_14_4()


def test_exercise_14_5():
    assert verify_exercise_14_5()


def test_exercise_14_6():
    assert verify_exercise_14_6()


def test_exercise_14_7():
    assert verify_exercise_14_7()


def test_exercise_14_8():
    assert verify_exercise_14_8()


def test_exercise_14_9():
    assert verify_exercise_14_9()


def test_exercise_14_10():
    assert verify_exercise_14_10()


def test_exercise_14_11():
    assert verify_exercise_14_11()


def test_exercise_14_12():
    assert verify_exercise_14_12()


def test_exercise_14_13():
    assert verify_exercise_14_13()


def test_exercise_14_14():
    assert verify_exercise_14_14()


def test_exercise_14_15():
    assert verify_exercise_14_15()


def test_exercise_14_16():
    assert verify_exercise_14_16()


def test_exercise_14_17():
    assert verify_exercise_14_17()


def test_exercise_14_18():
    assert verify_exercise_14_18()


def test_all_exercises_solution_dict():
    res = solve_all_exercises()
    assert len(res) == 18
    for i in range(1, 19):
        key = f"ex_14_{i}"
        assert key in res
        assert "explanation" in res[key]
