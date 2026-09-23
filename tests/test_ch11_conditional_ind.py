"""
tests/test_ch11_conditional_ind.py
==================================
Unit tests for Section 11.2 Conditional Independence
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
"""

import os
from pathlib import Path
import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.graphical_models import DirectedGraph
from common.conditional_independence import (
    CarFuelSystem,
    check_d_separation,
    get_markov_blanket,
    generate_figure_11_14,
    generate_figure_11_15,
    generate_figure_11_16,
    generate_figure_11_17,
    generate_figure_11_18,
    generate_figure_11_19,
    generate_figure_11_20,
    generate_figure_11_21,
    generate_figure_11_22,
    generate_figure_11_23,
    generate_figure_11_24,
    generate_figure_11_25,
    generate_figure_11_26,
)


class TestThreeCanonicalGraphs:
    def test_tail_to_tail(self):
        # c -> a and c -> b
        dag = DirectedGraph(["a", "b", "c"])
        dag.add_edge("c", "a")
        dag.add_edge("c", "b")

        # Unconditioned: dependent
        assert not check_d_separation(dag, {"a"}, {"b"}, set())
        # Conditioned on c: independent
        assert check_d_separation(dag, {"a"}, {"b"}, {"c"})

    def test_head_to_tail(self):
        # a -> c -> b
        dag = DirectedGraph(["a", "b", "c"])
        dag.add_edge("a", "c")
        dag.add_edge("c", "b")

        # Unconditioned: dependent
        assert not check_d_separation(dag, {"a"}, {"b"}, set())
        # Conditioned on c: independent
        assert check_d_separation(dag, {"a"}, {"b"}, {"c"})

    def test_head_to_head(self):
        # a -> c <- b
        dag = DirectedGraph(["a", "b", "c"])
        dag.add_edge("a", "c")
        dag.add_edge("b", "c")

        # Unconditioned: independent
        assert check_d_separation(dag, {"a"}, {"b"}, set())
        # Conditioned on c: dependent (collider activated)
        assert not check_d_separation(dag, {"a"}, {"b"}, {"c"})


class TestCarFuelSystemExplainingAway:
    def test_car_fuel_system_exact_probabilities(self):
        car = CarFuelSystem()

        # Eq 11.32
        p_G0 = car.compute_marginal_p_G0()
        assert p_G0 == pytest.approx(0.315)

        # Eq 11.33
        p_G0_given_F0 = car.compute_conditional_p_G0_given_F0()
        assert p_G0_given_F0 == pytest.approx(0.81)

        # Eq 11.34
        p_F0_given_G0 = car.compute_posterior_p_F0_given_G0()
        assert p_F0_given_G0 == pytest.approx(0.081 / 0.315, rel=1e-3)
        assert p_F0_given_G0 == pytest.approx(0.257, abs=1e-3)

        # Eq 11.35: 0.09 / 0.81 = 1 / 9 approx 0.111
        p_F0_given_G0_B0 = car.compute_posterior_p_F0_given_G0_B0()
        assert p_F0_given_G0_B0 == pytest.approx(0.09 / 0.81, rel=1e-3)
        assert p_F0_given_G0_B0 == pytest.approx(0.111, abs=1e-3)

        # Explaining away: observing B=0 lowers the probability that F=0
        assert p_F0_given_G0_B0 < p_F0_given_G0


class TestDSeparationAndMarkovBlanket:
    def test_d_separation_fig_11_21(self):
        # Graph (a): a -> e -> c -> b
        dag_a = DirectedGraph(["a", "e", "c", "b"])
        dag_a.add_edge("a", "e")
        dag_a.add_edge("e", "c")
        dag_a.add_edge("c", "b")
        assert check_d_separation(dag_a, {"a"}, {"b"}, {"e"})

        # Graph (b): a -> c <- d, c -> b
        dag_b = DirectedGraph(["a", "d", "c", "b"])
        dag_b.add_edge("a", "c")
        dag_b.add_edge("d", "c")
        dag_b.add_edge("c", "b")
        # Head-to-head at c: observing c opens path from a to d
        assert not check_d_separation(dag_b, {"a"}, {"d"}, {"c"})
        # Without observing c, a and d are independent
        assert check_d_separation(dag_b, {"a"}, {"d"}, set())

    def test_markov_blanket_property(self):
        # Node xi with parents p1, p2, children c1, c2, and co-parents k1, k2
        dag = DirectedGraph(["xi", "p1", "p2", "c1", "c2", "k1", "k2", "other"])
        dag.add_edge("p1", "xi")
        dag.add_edge("p2", "xi")
        dag.add_edge("xi", "c1")
        dag.add_edge("xi", "c2")
        dag.add_edge("k1", "c1")
        dag.add_edge("k2", "c2")
        dag.add_edge("other", "p1")

        mb = get_markov_blanket(dag, "xi")
        assert mb == {"p1", "p2", "c1", "c2", "k1", "k2"}
        assert "other" not in mb

        # Given Markov blanket, xi is d-separated from the rest of the graph ("other")
        assert check_d_separation(dag, {"xi"}, {"other"}, mb)


class TestFigureGenerations:
    def test_all_13_figures(self, tmp_path):
        tmp_dir = str(tmp_path)
        generators = [
            generate_figure_11_14,
            generate_figure_11_15,
            generate_figure_11_16,
            generate_figure_11_17,
            generate_figure_11_18,
            generate_figure_11_19,
            generate_figure_11_20,
            generate_figure_11_21,
            generate_figure_11_22,
            generate_figure_11_23,
            generate_figure_11_24,
            generate_figure_11_25,
            generate_figure_11_26,
        ]

        for i, gen in enumerate(generators, 14):
            fig = gen(save_dir=tmp_dir)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

            fpath = os.path.join(tmp_dir, f"Figure_11_{i}.png")
            assert os.path.isfile(fpath), f"Figure_11_{i}.png not found in {tmp_dir}"
            assert os.path.getsize(fpath) > 1000
