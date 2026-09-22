"""
Unit tests for Chapter 2 Section 2.1: The Rules of Probability.
Tests the fundamental rules of probability (sum rule, product rule, Bayes' theorem,
conditional normalization, independence) and textbook numerical results (medical screening example).
"""
import os
import pytest
import numpy as np

from common.probability import (
    compute_joint_marginal_conditional,
    bayes_rule,
    medical_screening_model,
    check_independence,
    generate_2d_sine_data
)


def test_medical_screening_numbers():
    """
    Test the exact textbook values from Section 2.1.1 & 2.1.4:
      - p(C = 1) = 0.01 (prevalence)
      - p(C = 0) = 0.99
      - p(T = 1 | C = 1) = 0.90 (sensitivity / true positive rate)
      - p(T = 0 | C = 1) = 0.10 (false negative rate)
      - p(T = 1 | C = 0) = 0.03 (false positive rate)
      - p(T = 0 | C = 0) = 0.97 (specificity / true negative rate)
      - Eq (2.20): p(T = 1) = 0.03 * 0.99 + 0.90 * 0.01 = 0.0387
      - Eq (2.21, 2.22): p(C = 1 | T = 1) = (0.90 * 0.01) / 0.0387 = 90 / 387 ~= 0.232558
    """
    res = medical_screening_model(
        p_cancer=0.01,
        p_pos_given_cancer=0.90,
        p_pos_given_no_cancer=0.03
    )

    assert res["p_cancer"] == pytest.approx(0.01)
    assert res["p_no_cancer"] == pytest.approx(0.99)
    assert res["p_pos_given_cancer"] == pytest.approx(0.90)
    assert res["p_neg_given_cancer"] == pytest.approx(0.10)
    assert res["p_pos_given_no_cancer"] == pytest.approx(0.03)
    assert res["p_neg_given_no_cancer"] == pytest.approx(0.97)

    # Eq (2.20)
    expected_p_pos = 0.03 * 0.99 + 0.90 * 0.01
    assert expected_p_pos == pytest.approx(0.0387, rel=1e-7)
    assert res["p_pos"] == pytest.approx(0.0387, rel=1e-7)

    # Eq (2.22)
    expected_p_cancer_given_pos = (0.90 * 0.01) / 0.0387
    assert expected_p_cancer_given_pos == pytest.approx(90.0 / 387.0, rel=1e-7)
    assert res["p_cancer_given_pos"] == pytest.approx(90.0 / 387.0, rel=1e-7)

    # Negative test result
    expected_p_neg = 1.0 - 0.0387
    assert res["p_neg"] == pytest.approx(expected_p_neg, rel=1e-7)
    expected_p_cancer_given_neg = (0.10 * 0.01) / expected_p_neg
    assert res["p_cancer_given_neg"] == pytest.approx(expected_p_cancer_given_neg, rel=1e-7)


def test_sum_rule_marginalization():
    """
    Test the sum rule (marginalization):
      p(X = x_i) = sum_j p(X = x_i, Y = y_j)  (Eq 2.4 / 2.8)
      p(Y = y_j) = sum_i p(X = x_i, Y = y_j)
      sum_i p(X = x_i) = 1                    (Eq 2.3)
      sum_j p(Y = y_j) = 1
    """
    # 5x3 table as in Figure 2.4
    nij = np.array([
        [2, 5, 3],
        [4, 1, 6],
        [7, 3, 2],
        [5, 8, 1],
        [3, 4, 9],
    ], dtype=np.float64)

    res = compute_joint_marginal_conditional(nij)
    p_XY = res["p_XY"]
    p_X = res["p_X"]
    p_Y = res["p_Y"]

    # Sum rule for X
    np.testing.assert_allclose(p_X, np.sum(p_XY, axis=1), atol=1e-12)
    # Sum rule for Y
    np.testing.assert_allclose(p_Y, np.sum(p_XY, axis=0), atol=1e-12)

    # Normalization
    assert np.sum(p_X) == pytest.approx(1.0, rel=1e-12)
    assert np.sum(p_Y) == pytest.approx(1.0, rel=1e-12)
    assert np.sum(p_XY) == pytest.approx(1.0, rel=1e-12)


def test_product_rule_and_conditional():
    """
    Test the product rule:
      p(X, Y) = p(Y | X) * p(X)  (Eq 2.7 / 2.9)
      p(X, Y) = p(X | Y) * p(Y)
    And conditional normalization:
      sum_j p(Y = y_j | X = x_i) = 1  (Eq 2.6)
      sum_i p(X = x_i | Y = y_j) = 1
    """
    nij = np.array([
        [10, 20],
        [30, 40],
        [15, 25]
    ], dtype=np.float64)

    res = compute_joint_marginal_conditional(nij)
    p_XY = res["p_XY"]
    p_X = res["p_X"]
    p_Y = res["p_Y"]
    p_Y_given_X = res["p_Y_given_X"]
    p_X_given_Y = res["p_X_given_Y"]

    # Product rule 1: p(X, Y) = p(Y | X) * p(X)
    reconstructed_1 = p_Y_given_X * p_X[:, np.newaxis]
    np.testing.assert_allclose(p_XY, reconstructed_1, atol=1e-12)

    # Product rule 2: p(X, Y) = p(X | Y) * p(Y)
    reconstructed_2 = p_X_given_Y * p_Y[np.newaxis, :]
    np.testing.assert_allclose(p_XY, reconstructed_2, atol=1e-12)

    # Conditional distributions must sum to 1 along conditioning dimension
    np.testing.assert_allclose(np.sum(p_Y_given_X, axis=1), np.ones(3), atol=1e-12)
    np.testing.assert_allclose(np.sum(p_X_given_Y, axis=0), np.ones(2), atol=1e-12)


def test_bayes_theorem():
    """
    Test Bayes' theorem:
      p(Y | X) = p(X | Y) * p(Y) / p(X)  (Eq 2.10)
    Verify via medical screening parameters.
    """
    prior_y = np.array([0.99, 0.01])  # Y = 0 (no cancer), Y = 1 (cancer)
    # likelihood p(X | Y): shape (2, 2) where X=0 (test neg), X=1 (test pos)
    # col 0: Y=0 -> [p(T=0|C=0), p(T=1|C=0)] = [0.97, 0.03]
    # col 1: Y=1 -> [p(T=0|C=1), p(T=1|C=1)] = [0.10, 0.90]
    likelihood_x_given_y = np.array([
        [0.97, 0.10],  # X = 0 (test neg)
        [0.03, 0.90],  # X = 1 (test pos)
    ])

    marginal_x, posterior_y_given_x = bayes_rule(prior_y, likelihood_x_given_y)

    # Marginal p(X=1) = p(T=1) = 0.0387
    assert marginal_x[1] == pytest.approx(0.0387, rel=1e-7)
    assert marginal_x[0] == pytest.approx(1.0 - 0.0387, rel=1e-7)

    # Posterior p(Y=1 | X=1) = p(Cancer | Pos) = 90 / 387
    assert posterior_y_given_x[1, 1] == pytest.approx(90.0 / 387.0, rel=1e-7)
    # Posterior p(Y=0 | X=1) = p(No Cancer | Pos) = 297 / 387
    assert posterior_y_given_x[1, 0] == pytest.approx(297.0 / 387.0, rel=1e-7)


def test_independence_properties():
    """
    Test independence:
      p(X, Y) = p(X) * p(Y) <=> X and Y are independent.
      Then p(Y | X) = p(Y) and p(X | Y) = p(X).
    """
    # Independent case
    px = np.array([0.2, 0.5, 0.3])
    py = np.array([0.4, 0.6])
    p_indep = np.outer(px, py)

    is_ind, diff = check_independence(p_indep)
    assert is_ind is True
    assert diff < 1e-10

    # Dependent case
    p_dep = np.array([
        [0.15, 0.05],
        [0.10, 0.40],
        [0.15, 0.15],
    ])
    is_ind_dep, diff_dep = check_independence(p_dep)
    assert is_ind_dep is False
    assert diff_dep > 0.01


def test_generate_2d_sine_data():
    """
    Test 2D sine synthetic data generation for Figure 2.1.
    """
    # Unobserved x2
    data_unobs = generate_2d_sine_data(n_samples=50, noise_std=0.1, fixed_x2=None, seed=123)
    assert len(data_unobs["x1"]) == 50
    assert len(data_unobs["x2"]) == 50
    assert len(data_unobs["t"]) == 50
    assert np.all(data_unobs["x1"] >= 0.0) and np.all(data_unobs["x1"] <= 1.0)
    assert np.all(data_unobs["x2"] >= 0.0) and np.all(data_unobs["x2"] <= 1.0)

    # Fixed x2
    data_fixed = generate_2d_sine_data(n_samples=50, noise_std=0.0, fixed_x2=0.25, seed=123)
    # sin(2*pi*0.25) = 1.0, so y_true = sin(2*pi*x1)
    expected_y = np.sin(2.0 * np.pi * data_fixed["x1"])
    np.testing.assert_allclose(data_fixed["y_true"], expected_y, atol=1e-12)
    np.testing.assert_allclose(data_fixed["t"], expected_y, atol=1e-12)


def test_saved_figures_exist():
    """
    Verify that all textbook figures have been generated and saved to result/ and 2/result/.
    """
    expected_files = [
        "fig2_01_two_dimensional_regression.png",
        "fig2_02_bent_coin.png",
        "fig2_03_medical_screening.png",
        "fig2_04_sum_product_rules.png",
        "fig2_05_joint_marginal_conditional.png"
    ]
    for directory in ["result", os.path.join("2", "result")]:
        for fname in expected_files:
            fpath = os.path.join(directory, fname)
            assert os.path.exists(fpath), f"Missing expected figure: {fpath}"
            assert os.path.getsize(fpath) > 1000, f"File {fpath} is unexpectedly small or empty"
