"""
Tests for Chapter 13 Exercises (13.1 to 13.10)
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts
"""

import numpy as np
import pytest

from common.exercises_ch13 import (
    solve_exercise_13_1,
    solve_exercise_13_2,
    solve_exercise_13_3,
    solve_exercise_13_4,
    solve_exercise_13_5,
    solve_exercise_13_6,
    solve_exercise_13_7,
    solve_exercise_13_8,
    solve_exercise_13_9,
    solve_exercise_13_10,
)


class TestChapter13Exercises:
    def test_exercise_13_1(self):
        res = solve_exercise_13_1()
        assert res["is_correct"] is True
        assert res["P"].shape == (5, 5)
        # Permutation matrix should be orthogonal: P @ P.T = I
        assert np.allclose(res["P"] @ res["P"].T, np.eye(5))
        assert np.allclose(res["A_perm"], res["expected_A_perm"])

    def test_exercise_13_2(self):
        res = solve_exercise_13_2()
        assert res["is_equal"] is True
        assert len(res["diag_A2"]) == 5
        assert np.array_equal(res["diag_A2"], res["degrees"])

        # Also test on a custom random adjacency matrix
        rng = np.random.RandomState(123)
        A_rand = rng.randint(0, 2, (6, 6)).astype(float)
        A_rand = np.triu(A_rand, 1) + np.triu(A_rand, 1).T
        res_rand = solve_exercise_13_2(A_rand)
        assert res_rand["is_equal"] is True

    def test_exercise_13_3(self):
        res = solve_exercise_13_3()
        assert res["is_correct"] is True
        assert len(res["edges"]) == 7
        assert res["A"].shape == (5, 5)
        # Check degrees: [3, 4, 3, 2, 2]
        expected_degrees = np.array([3, 4, 3, 2, 2])
        assert np.array_equal(res["degrees"], expected_degrees)

    def test_exercise_13_4(self):
        res = solve_exercise_13_4()
        assert res["is_equal"] is True
        assert res["X_tilde"].shape == res["X"].shape

    def test_exercise_13_5(self):
        res = solve_exercise_13_5()
        assert res["is_equal"] is True
        assert res["A_tilde"].shape == res["A"].shape

    def test_exercise_13_6(self):
        res = solve_exercise_13_6()
        assert res["is_Z_equal"] is True
        assert res["is_Arg_equal"] is True

    def test_exercise_13_7(self):
        res = solve_exercise_13_7()
        assert res["is_equivariant"] is True

    def test_exercise_13_8(self):
        res = solve_exercise_13_8()
        assert "equivariance" in res["proof_summary"].lower()

    def test_exercise_13_9(self):
        res = solve_exercise_13_9()
        assert "transformer" in res["proof_summary"].lower()

    def test_exercise_13_10(self):
        res = solve_exercise_13_10()
        assert res["is_correct"] is True
        assert res["report"]["translation_invariance_features"] is True
        assert res["report"]["rotation_invariance_features"] is True
        assert res["report"]["reflection_invariance_features"] is True
        assert res["report"]["translation_equivariance"] is True
        assert res["report"]["rotation_equivariance"] is True
        assert res["report"]["reflection_equivariance"] is True
