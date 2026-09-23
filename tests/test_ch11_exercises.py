"""
Tests for Chapter 11 Exercises 11.1 - 11.20
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.exercises_ch11 import (
    exercise_11_1_normalization_proof,
    exercise_11_2_verify_acyclicity,
    exercise_11_3_table_11_1,
    exercise_11_4_evaluate_distributions,
    exercise_11_5_noisy_or,
    exercise_11_6_linear_gaussian_mean,
    exercise_11_7_linear_gaussian_covariance,
    exercise_11_8_param_count,
    exercise_11_9_fig_11_7_moments,
    exercise_11_10_verify_joint_gaussian,
    exercise_11_11_decomposition_property,
    exercise_11_12_markov_blanket_d_separation,
    exercise_11_13_head_to_head_descendant,
    generate_figure_11_32,
    exercise_11_14_car_fuel_driver_report,
    exercise_11_15_naive_bayes_mle,
    exercise_11_16_markov_properties,
    exercise_11_17_markov_d_separation,
    exercise_11_18_state_space_expansion,
    exercise_11_19_state_space_d_separation,
    exercise_11_20_forward_backward_smoothing,
)


def test_exercise_11_1():
    # 3-node DAG: x1 -> x2 -> x3
    cpts = {
        "x1": np.array([0.4, 0.6]),
        "x2": np.array([[0.7, 0.3], [0.2, 0.8]]),
        "x3": np.array([[0.9, 0.1], [0.5, 0.5]]),
    }
    parents = {"x1": [], "x2": ["x1"], "x3": ["x2"]}
    order = ["x1", "x2", "x3"]
    total_prob = exercise_11_1_normalization_proof(cpts, parents, order)
    assert np.isclose(total_prob, 1.0)


def test_exercise_11_2():
    nodes = ["1", "2", "3", "4"]
    acyclic_edges = [("1", "2"), ("2", "3"), ("3", "4"), ("1", "3")]
    is_acyclic, order = exercise_11_2_verify_acyclicity(nodes, acyclic_edges)
    assert is_acyclic is True
    assert len(order) == 4

    cyclic_edges = [("1", "2"), ("2", "3"), ("3", "1")]
    is_acyclic_cycle, _ = exercise_11_2_verify_acyclicity(nodes[:3], cyclic_edges)
    assert is_acyclic_cycle is False


def test_exercise_11_3():
    res = exercise_11_3_table_11_1()
    assert res["is_marginally_dependent"] is True
    assert res["is_conditionally_independent"] is True
    assert np.isclose(res["p_a"][0], 0.6)
    assert np.isclose(res["p_a"][1], 0.4)


def test_exercise_11_4():
    res = exercise_11_4_evaluate_distributions()
    assert res["factorization_valid"] is True
    assert res["max_factorization_error"] < 1e-10


def test_exercise_11_5():
    # Noisy-OR test
    # Leak only
    prob_leak = exercise_11_5_noisy_or(mu0=0.2, mu_vec=np.array([0.5, 0.7]), x_vec=np.array([0, 0]))
    assert np.isclose(prob_leak, 0.2)

    # Boolean OR limit (mu0=0, mu_i -> 1)
    prob_or_0 = exercise_11_5_noisy_or(mu0=0.0, mu_vec=np.array([0.9999, 0.9999]), x_vec=np.array([0, 0]))
    prob_or_1 = exercise_11_5_noisy_or(mu0=0.0, mu_vec=np.array([0.9999, 0.9999]), x_vec=np.array([1, 0]))
    assert np.isclose(prob_or_0, 0.0, atol=1e-3)
    assert np.isclose(prob_or_1, 1.0, atol=1e-3)


def test_exercise_11_6_and_7():
    # Linear-Gaussian DAG
    W = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.3, 0.4, 0.0]
    ])
    b = np.array([1.0, 2.0, 0.5])
    v = np.array([0.5, 0.8, 1.2])

    mu = exercise_11_6_linear_gaussian_mean(W, b)
    Sigma = exercise_11_7_linear_gaussian_covariance(W, v)

    assert mu[0] == 1.0
    assert mu[1] == 0.5 * 1.0 + 2.0  # 2.5
    assert np.isclose(mu[2], 0.3 * 1.0 + 0.4 * 2.5 + 0.5)  # 1.8

    assert Sigma[0, 0] == 0.5
    assert Sigma[1, 0] == 0.5 * 0.5  # 0.25
    assert np.isclose(Sigma[1, 1], 0.5**2 * 0.5 + 0.8)


def test_exercise_11_8():
    assert exercise_11_8_param_count(1) == 1
    assert exercise_11_8_param_count(3) == 6
    assert exercise_11_8_param_count(5) == 15


def test_exercise_11_9():
    b = np.array([1.0, -0.5, 2.0])
    v = np.array([1.2, 0.7, 0.4])
    w21, w31, w32 = 0.6, -0.4, 0.8

    mu, Sigma = exercise_11_9_fig_11_7_moments(b, v, w21, w31, w32)

    # Check with recursive functions
    W = np.array([
        [0.0, 0.0, 0.0],
        [w21, 0.0, 0.0],
        [w31, w32, 0.0]
    ])
    mu_rec = exercise_11_6_linear_gaussian_mean(W, b)
    Sigma_rec = exercise_11_7_linear_gaussian_covariance(W, v)

    np.testing.assert_allclose(mu, mu_rec, atol=1e-10)
    np.testing.assert_allclose(Sigma, Sigma_rec, atol=1e-10)


def test_exercise_11_10():
    # 2 vector nodes each of dim 2
    W_blocks = [
        [None, None],
        [np.array([[0.5, 0.0], [0.0, 0.5]]), None]
    ]
    b_vectors = [np.array([1.0, 2.0]), np.array([0.0, -1.0])]
    Sigma_blocks = [np.eye(2) * 0.5, np.eye(2) * 0.8]

    mu, Sigma = exercise_11_10_verify_joint_gaussian(W_blocks, b_vectors, Sigma_blocks)
    assert mu.shape == (4,)
    assert Sigma.shape == (4, 4)
    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(Sigma)
    assert all(eigvals > 0)


def test_exercise_11_11():
    # Construct a valid joint distribution satisfying a _|_ b, c | d
    # p(a, b, c, d) = p(d) * p(a | d) * p(b, c | d)
    p_d = np.array([0.4, 0.6])
    p_a_given_d = np.array([[0.3, 0.7], [0.8, 0.2]])  # (d, a)
    p_bc_given_d = np.zeros((2, 2, 2))  # (d, b, c)
    p_bc_given_d[0] = np.array([[0.1, 0.2], [0.3, 0.4]])
    p_bc_given_d[1] = np.array([[0.4, 0.3], [0.2, 0.1]])

    p_joint = np.zeros((2, 2, 2, 2))  # (a, b, c, d)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    p_joint[a, b, c, d] = p_d[d] * p_a_given_d[d, a] * p_bc_given_d[d, b, c]

    assert exercise_11_11_decomposition_property(p_joint) is True


def test_exercise_11_12():
    assert exercise_11_12_markov_blanket_d_separation() is True


def test_exercise_11_13():
    unobs, obs_d = exercise_11_13_head_to_head_descendant()
    assert unobs is True
    assert obs_d is False


def test_figure_11_32(tmp_path):
    fig = generate_figure_11_32(save_dir=str(tmp_path))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_exercise_11_14():
    res = exercise_11_14_car_fuel_driver_report()
    assert np.isclose(res["p_D0"], 0.352, atol=1e-4)
    assert np.isclose(res["p_F0_given_D0"], 0.2125, atol=1e-4)
    # Explaining away: observing battery flat reduces probability of empty tank!
    assert res["p_F0_given_D0_B0"] < res["p_F0_given_D0"]
    assert np.isclose(res["p_F0_given_D0_B0"], 0.1096, atol=1e-3)


def test_exercise_11_15():
    X = np.array([[1.0, 2.0], [1.2, 1.8], [5.0, 6.0]])
    y = np.array([0, 0, 1])
    mle = exercise_11_15_naive_bayes_mle(X, y)
    np.testing.assert_allclose(mle["priors"], [2/3, 1/3])
    assert mle["means"].shape == (2, 2)


def test_exercise_11_16():
    res = exercise_11_16_markov_properties()
    assert res["first_order_verified"] is True
    assert res["second_order_verified"] is True


def test_exercise_11_17():
    is_1st, is_2nd = exercise_11_17_markov_d_separation(N=5)
    assert is_1st is True
    assert is_2nd is True


def test_exercise_11_18():
    K = 2
    # 2nd-order transition tensor
    T2 = np.array([
        [[0.8, 0.2], [0.4, 0.6]],
        [[0.3, 0.7], [0.1, 0.9]]
    ])
    T_pair = exercise_11_18_state_space_expansion(T2)
    assert T_pair.shape == (4, 4)
    np.testing.assert_allclose(np.sum(T_pair, axis=1), np.ones(4), atol=1e-10)


def test_exercise_11_19():
    assert exercise_11_19_state_space_d_separation(N=4) is True


def test_exercise_11_20():
    A = np.array([[0.8, 0.2], [0.3, 0.7]])
    B = np.array([[0.9, 0.1], [0.2, 0.8]])
    pi = np.array([0.5, 0.5])
    obs = np.array([0, 0, 1, 1])

    gamma, log_lik = exercise_11_20_forward_backward_smoothing(A, B, pi, obs)
    assert gamma.shape == (4, 2)
    np.testing.assert_allclose(np.sum(gamma, axis=1), np.ones(4), atol=1e-10)
    assert log_lik > 0.0
