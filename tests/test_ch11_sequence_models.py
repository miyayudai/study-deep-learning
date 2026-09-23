"""
Tests for Chapter 11 Section 11.3: Sequence Models
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.sequence_models import (
    MarkovChainSimulator,
    HiddenMarkovModel,
    generate_figure_11_27,
    generate_figure_11_28,
    generate_figure_11_29,
    generate_figure_11_30,
    generate_figure_11_31,
)
from common.graphical_models import DirectedGraph
from common.conditional_independence import check_d_separation


class TestMarkovChainSimulator:
    def test_sampling(self):
        # 2-state Markov chain
        T = np.array([[0.8, 0.2],
                      [0.3, 0.7]])
        pi0 = np.array([0.5, 0.5])
        mc = MarkovChainSimulator(T, pi0)
        seq = mc.sample_sequence(length=100, seed=42)
        assert len(seq) == 100
        assert set(np.unique(seq)).issubset({0, 1})

    def test_stationary_distribution(self):
        # 2-state chain: stationary distribution is pi = [0.3 / (0.2 + 0.3), 0.2 / (0.2 + 0.3)] = [0.6, 0.4]
        T = np.array([[0.8, 0.2],
                      [0.3, 0.7]])
        pi0 = np.array([1.0, 0.0])
        mc = MarkovChainSimulator(T, pi0)
        pi_stat = mc.compute_stationary_distribution()
        np.testing.assert_allclose(pi_stat, [0.6, 0.4], atol=1e-5)
        np.testing.assert_allclose(pi_stat @ T, pi_stat, atol=1e-5)
        assert np.isclose(np.sum(pi_stat), 1.0)


class TestHiddenMarkovModel:
    def test_forward_and_viterbi(self):
        # Classic Rainy/Sunny HMM
        # States: 0=Rainy, 1=Sunny
        # Observations: 0=Walk, 1=Shop, 2=Clean
        A = np.array([[0.7, 0.3],
                      [0.4, 0.6]])
        B = np.array([[0.1, 0.4, 0.5],
                      [0.6, 0.3, 0.1]])
        pi = np.array([0.6, 0.4])

        hmm = HiddenMarkovModel(A, B, pi)
        obs = np.array([0, 1, 2])  # Walk, Shop, Clean

        likelihood, alpha = hmm.forward_algorithm(obs)
        assert likelihood > 0.0
        assert alpha.shape == (3, 2)

        # Hand calculate step 0:
        # alpha_0 = pi * B[:, obs[0]] = [0.6 * 0.1, 0.4 * 0.6] = [0.06, 0.24]
        np.testing.assert_allclose(alpha[0], [0.06, 0.24], atol=1e-5)

        # Step 1:
        # prev @ A = [0.06*0.7 + 0.24*0.4, 0.06*0.3 + 0.24*0.6] = [0.042 + 0.096, 0.018 + 0.144] = [0.138, 0.162]
        # alpha_1 = [0.138 * 0.4, 0.162 * 0.3] = [0.0552, 0.0486]
        np.testing.assert_allclose(alpha[1], [0.0552, 0.0486], atol=1e-5)

        # Viterbi decoding
        path = hmm.viterbi_algorithm(obs)
        assert len(path) == len(obs)
        assert all(state in [0, 1] for state in path)


class TestSequenceGraphicalProperties:
    def test_markov_chain_d_separation(self):
        # 1st-order Markov chain: x1 -> x2 -> x3 -> x4
        dag = DirectedGraph(["x1", "x2", "x3", "x4"])
        dag.add_edge("x1", "x2")
        dag.add_edge("x2", "x3")
        dag.add_edge("x3", "x4")

        # x3 is independent of x1 given x2
        assert check_d_separation(dag, {"x1"}, {"x3"}, {"x2"}) is True
        # x3 is not independent of x1 given nothing
        assert check_d_separation(dag, {"x1"}, {"x3"}, set()) is False

    def test_second_order_markov_d_separation(self):
        # 2nd-order Markov chain
        dag = DirectedGraph(["x1", "x2", "x3", "x4"])
        dag.add_edge("x1", "x2")
        dag.add_edge("x2", "x3")
        dag.add_edge("x3", "x4")
        dag.add_edge("x1", "x3")
        dag.add_edge("x2", "x4")

        # x4 independent of x1 given {x2, x3}
        assert check_d_separation(dag, {"x1"}, {"x4"}, {"x2", "x3"}) is True
        # x4 NOT independent of x1 given only x3 (because x1 -> x2 -> x4 is active)
        assert check_d_separation(dag, {"x1"}, {"x4"}, {"x3"}) is False

    def test_state_space_model_d_separation(self):
        # State-space model: z1 -> z2 -> z3, z1 -> x1, z2 -> x2, z3 -> x3
        dag = DirectedGraph(["z1", "z2", "z3", "x1", "x2", "x3"])
        dag.add_edge("z1", "z2")
        dag.add_edge("z2", "z3")
        dag.add_edge("z1", "x1")
        dag.add_edge("z2", "x2")
        dag.add_edge("z3", "x3")

        # x1 independent of x2 given z1
        assert check_d_separation(dag, {"x1"}, {"x2"}, {"z1"}) is True

        # x1 NOT independent of x3 when no z is observed
        assert check_d_separation(dag, {"x1"}, {"x3"}, set()) is False


class TestFigureGenerators:
    @pytest.mark.parametrize("fig_func", [
        generate_figure_11_27,
        generate_figure_11_28,
        generate_figure_11_29,
        generate_figure_11_30,
        generate_figure_11_31,
    ])
    def test_figure_generation(self, fig_func, tmp_path):
        fig = fig_func(save_dir=str(tmp_path))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
