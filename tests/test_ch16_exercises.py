"""
Unit tests for Chapter 16 Exercises (Exercises 16.1 through 16.26).
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
"""

import pytest
from common.exercises_ch16 import (
    solve_exercise_16_1, verify_exercise_16_1,
    solve_exercise_16_2, verify_exercise_16_2,
    solve_exercise_16_3, verify_exercise_16_3,
    solve_exercise_16_4, verify_exercise_16_4,
    solve_exercise_16_5, verify_exercise_16_5,
    solve_exercise_16_6, verify_exercise_16_6,
    solve_exercise_16_7, verify_exercise_16_7,
    solve_exercise_16_8, verify_exercise_16_8,
    solve_exercise_16_9, verify_exercise_16_9,
    solve_exercise_16_10, verify_exercise_16_10,
    solve_exercise_16_11, verify_exercise_16_11,
    solve_exercise_16_12, verify_exercise_16_12,
    solve_exercise_16_13, verify_exercise_16_13,
    solve_exercise_16_14, verify_exercise_16_14,
    solve_exercise_16_15, verify_exercise_16_15,
    solve_exercise_16_16, verify_exercise_16_16,
    solve_exercise_16_17, verify_exercise_16_17,
    solve_exercise_16_18, verify_exercise_16_18,
    solve_exercise_16_19, verify_exercise_16_19,
    solve_exercise_16_20, verify_exercise_16_20,
    solve_exercise_16_21, verify_exercise_16_21,
    solve_exercise_16_22, verify_exercise_16_22,
    solve_exercise_16_23, verify_exercise_16_23,
    solve_exercise_16_24, verify_exercise_16_24,
    solve_exercise_16_25, verify_exercise_16_25,
    solve_exercise_16_26, verify_exercise_16_26,
    solve_all_exercises,
    verify_all_exercises,
)


def test_exercise_16_1():
    assert verify_exercise_16_1()


def test_exercise_16_2():
    assert verify_exercise_16_2()


def test_exercise_16_3():
    assert verify_exercise_16_3()


def test_exercise_16_4():
    assert verify_exercise_16_4()


def test_exercise_16_5():
    assert verify_exercise_16_5()


def test_exercise_16_6():
    assert verify_exercise_16_6()


def test_exercise_16_7():
    assert verify_exercise_16_7()


def test_exercise_16_8():
    assert verify_exercise_16_8()


def test_exercise_16_9():
    assert verify_exercise_16_9()


def test_exercise_16_10():
    assert verify_exercise_16_10()


def test_exercise_16_11():
    assert verify_exercise_16_11()


def test_exercise_16_12():
    assert verify_exercise_16_12()


def test_exercise_16_13():
    assert verify_exercise_16_13()


def test_exercise_16_14():
    assert verify_exercise_16_14()


def test_exercise_16_15():
    assert verify_exercise_16_15()


def test_exercise_16_16():
    assert verify_exercise_16_16()


def test_exercise_16_17():
    assert verify_exercise_16_17()


def test_exercise_16_18():
    assert verify_exercise_16_18()


def test_exercise_16_19():
    assert verify_exercise_16_19()


def test_exercise_16_20():
    assert verify_exercise_16_20()


def test_exercise_16_21():
    assert verify_exercise_16_21()


def test_exercise_16_22():
    assert verify_exercise_16_22()


def test_exercise_16_23():
    assert verify_exercise_16_23()


def test_exercise_16_24():
    assert verify_exercise_16_24()


def test_exercise_16_25():
    assert verify_exercise_16_25()


def test_exercise_16_26():
    assert verify_exercise_16_26()


def test_all_exercises_solved_and_verified():
    solutions = solve_all_exercises()
    assert len(solutions) == 26
    verifications = verify_all_exercises()
    assert len(verifications) == 26
    assert all(verifications.values())
