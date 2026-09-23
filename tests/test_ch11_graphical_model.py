"""
tests/test_ch11_graphical_model.py
==================================
Unit tests for Section 11.1 Graphical Models
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
"""

import os
from pathlib import Path
import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.graphical_models import (
    DirectedGraph,
    LinearGaussianDAG,
    count_discrete_parameters,
    compute_bayes_discrete,
    generate_figure_11_1,
    generate_figure_11_2,
    generate_figure_11_3,
    generate_figure_11_4,
    generate_figure_11_5,
    generate_figure_11_6,
    generate_figure_11_7,
    generate_figure_11_8,
    generate_figure_11_9,
    generate_figure_11_10,
    generate_figure_11_11,
    generate_figure_11_12,
    generate_figure_11_13,
)


class TestDAGRepresentation:
    def test_fig_11_1_factorization(self):
        dag = DirectedGraph(["a", "b", "c"])
        dag.add_edge("a", "b")
        dag.add_edge("a", "c")
        dag.add_edge("b", "c")

        order = dag.topological_sort()
        assert order == ["a", "b", "c"]
        factorization = dag.factorization_formula()
        assert "p(a)" in factorization
        assert "p(b | a)" in factorization
        assert "p(c | a, b)" in factorization

    def test_fig_11_2_dag(self):
        dag = DirectedGraph([f"x{i}" for i in range(1, 8)])
        dag.add_edge("x1", "x4")
        dag.add_edge("x2", "x4")
        dag.add_edge("x3", "x4")
        dag.add_edge("x1", "x5")
        dag.add_edge("x3", "x5")
        dag.add_edge("x4", "x6")
        dag.add_edge("x4", "x7")
        dag.add_edge("x5", "x7")

        order = dag.topological_sort()
        assert len(order) == 7
        assert order.index("x1") < order.index("x4")
        assert order.index("x4") < order.index("x6")
        assert order.index("x4") < order.index("x7")
        assert order.index("x5") < order.index("x7")

    def test_cycle_detection(self):
        dag = DirectedGraph(["x", "y"])
        dag.add_edge("x", "y")
        dag.add_edge("y", "x")
        with pytest.raises(ValueError, match="directed cycle"):
            dag.topological_sort()


class TestDiscreteParameters:
    def test_discrete_parameter_counts(self):
        # K = 3 states
        K = 3
        assert count_discrete_parameters('fully_connected_two', num_states=K) == K ** 2 - 1  # 8
        assert count_discrete_parameters('independent', num_nodes=2, num_states=K) == 2 * (K - 1)  # 4

        # Chain of M = 5 nodes, K = 4 states
        M, K = 5, 4
        # (K - 1) + (M - 1) * K * (K - 1) = 3 + 4 * 4 * 3 = 3 + 48 = 51
        assert count_discrete_parameters('chain_general', num_nodes=M, num_states=K) == 51

        # Homogeneous chain: K^2 - 1 = 15
        assert count_discrete_parameters('chain_shared', num_nodes=M, num_states=K) == K ** 2 - 1

        # Binary parent-child: M = 4 parents
        assert count_discrete_parameters('parent_child_full', num_parents=4) == 4 + 16  # 20
        assert count_discrete_parameters('parent_child_logistic', num_parents=4) == 4 + 5  # 9


class TestLinearGaussianNetwork:
    def test_fig_11_7_exact_moments(self):
        # Figure 11.7: x1 -> x2 -> x3
        dag = DirectedGraph(["x1", "x2", "x3"])
        dag.add_edge("x1", "x2")
        dag.add_edge("x2", "x3")

        b1, b2, b3 = 1.0, -0.5, 2.0
        w21, w32 = 1.5, -2.0
        v1, v2, v3 = 0.8, 1.2, 0.5

        biases = {"x1": b1, "x2": b2, "x3": b3}
        weights = {("x2", "x1"): w21, ("x3", "x2"): w32}
        variances = {"x1": v1, "x2": v2, "x3": v3}

        lg_dag = LinearGaussianDAG(dag, weights, biases, variances)
        means, cov = lg_dag.compute_joint_mean_and_cov()

        # Analytical expectation from Eq 11.14:
        # mu1 = b1
        # mu2 = b2 + w21 * b1
        # mu3 = b3 + w32 * b2 + w32 * w21 * b1
        exp_mu1 = b1
        exp_mu2 = b2 + w21 * b1
        exp_mu3 = b3 + w32 * b2 + w32 * w21 * b1
        np.testing.assert_allclose(means, [exp_mu1, exp_mu2, exp_mu3])

        # Analytical covariance from recursion:
        # cov(x1, x1) = v1
        # cov(x1, x2) = w21 * cov(x1, x1) = w21 * v1
        # cov(x2, x2) = w21 * cov(x2, x1) + v2 = w21^2 * v1 + v2
        # cov(x1, x3) = w32 * cov(x1, x2) = w32 * w21 * v1
        # cov(x2, x3) = w32 * cov(x2, x2)
        # cov(x3, x3) = w32 * cov(x3, x2) + v3
        assert cov[0, 0] == pytest.approx(v1)
        assert cov[0, 1] == pytest.approx(w21 * v1)
        assert cov[1, 1] == pytest.approx((w21 ** 2) * v1 + v2)
        assert cov[0, 2] == pytest.approx(w32 * w21 * v1)
        assert cov[1, 2] == pytest.approx(w32 * cov[1, 1])
        assert cov[2, 2] == pytest.approx(w32 * cov[2, 1] + v3)

        # Covariance must be symmetric positive-definite
        np.testing.assert_allclose(cov, cov.T)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0)


class TestBayesInversion:
    def test_bayes_discrete_inversion(self):
        # Disease testing example: x in {0, 1} (healthy, diseased)
        # y in {0, 1} (negative test, positive test)
        p_x = np.array([0.99, 0.01])  # 1% prior prevalence
        # Likelihood p(y | x): rows are y=0, y=1; cols are x=0, x=1
        # Sensitivity 95% (p(y=1 | x=1) = 0.95), False positive rate 5% (p(y=1 | x=0) = 0.05)
        p_y_given_x = np.array([
            [0.95, 0.05],  # y=0: neg test
            [0.05, 0.95],  # y=1: pos test
        ])

        p_y, p_x_given_y = compute_bayes_discrete(p_x, p_y_given_x)

        # Total probability for y
        assert np.sum(p_y) == pytest.approx(1.0)
        # Marginal p(y=1) = 0.05 * 0.99 + 0.95 * 0.01 = 0.0495 + 0.0095 = 0.0590
        assert p_y[1] == pytest.approx(0.0590)

        # Posterior p(x=1 | y=1) = (0.95 * 0.01) / 0.0590 = 0.0095 / 0.0590 approx 0.1610
        assert p_x_given_y[1, 1] == pytest.approx(0.0095 / 0.0590)
        # Column sums of posterior must equal 1
        np.testing.assert_allclose(np.sum(p_x_given_y, axis=0), [1.0, 1.0])


class TestFigureGenerations:
    def test_all_13_figures(self, tmp_path):
        tmp_dir = str(tmp_path)
        generators = [
            generate_figure_11_1,
            generate_figure_11_2,
            generate_figure_11_3,
            generate_figure_11_4,
            generate_figure_11_5,
            generate_figure_11_6,
            generate_figure_11_7,
            generate_figure_11_8,
            generate_figure_11_9,
            generate_figure_11_10,
            generate_figure_11_11,
            generate_figure_11_12,
            generate_figure_11_13,
        ]

        for i, gen in enumerate(generators, 1):
            fig = gen(save_dir=tmp_dir)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

            fpath = os.path.join(tmp_dir, f"Figure_11_{i}.png")
            assert os.path.isfile(fpath), f"Figure_11_{i}.png not found in {tmp_dir}"
            assert os.path.getsize(fpath) > 1000
