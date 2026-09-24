"""
Tests for Chapter 13 Section 13.1: Machine Learning on Graphs
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.machine_learning_on_graphs import (
    Graph,
    build_permutation_matrix,
    check_node_equivariance,
    check_graph_invariance,
    simple_gcn_layer,
    simple_graph_pooling,
    generate_figure_13_1,
    generate_figure_13_2,
)


class TestGraphProperties:
    def test_graph_initialization_and_matrices(self):
        # 5-node graph from Figure 13.2: (A, C), (B, C), (C, D), (C, E), (D, E)
        edges = [(0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]
        g = Graph(num_nodes=5, edges=edges, node_labels=["A", "B", "C", "D", "E"])

        # Check Adjacency Matrix A (Figure 13.2b)
        expected_A = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0],
        ], dtype=float)
        np.testing.assert_array_equal(g.A, expected_A)

        # Check Degrees: A:1, B:1, C:4, D:2, E:2
        expected_degrees = np.array([1, 1, 4, 2, 2], dtype=float)
        np.testing.assert_array_equal(g.degrees, expected_degrees)
        np.testing.assert_array_equal(g.degree_matrix, np.diag(expected_degrees))

        # Check Laplacian L = D - A
        L = g.laplacian
        # Row sums of Laplacian must be zero: L @ 1 = 0
        np.testing.assert_allclose(np.sum(L, axis=1), np.zeros(5), atol=1e-7)
        # Laplacian must be positive semi-definite (eigenvalues >= 0)
        eigvals = np.linalg.eigvalsh(L)
        assert np.all(eigvals >= -1e-7)
        # Smallest eigenvalue must be 0 for connected graph
        assert np.isclose(eigvals[0], 0.0, atol=1e-7)

    def test_normalized_laplacians(self):
        edges = [(0, 1), (1, 2), (2, 0)]  # Triangle graph
        g = Graph(num_nodes=3, edges=edges)

        L_sym = g.normalized_laplacian
        # Eigenvalues of L_sym must lie in [0, 2]
        eigvals = np.linalg.eigvalsh(L_sym)
        assert np.all(eigvals >= -1e-7)
        assert np.all(eigvals <= 2.0 + 1e-7)

        # Renormalized adjacency with self loops
        A_hat = g.renormalized_adjacency
        assert A_hat.shape == (3, 3)
        assert np.all(A_hat > 0)  # Triangle with self-loops has full connectivity

    def test_directed_graph(self):
        edges = [(0, 1), (1, 2)]
        g = Graph(num_nodes=3, edges=edges, is_directed=True)
        assert g.A[0, 1] == 1.0
        assert g.A[1, 0] == 0.0  # Directed: asymmetric


class TestPermutations:
    def test_permutation_matrix(self):
        # Permutation (A, B, C, D, E) -> (C, E, A, D, B) = [2, 4, 0, 3, 1]
        perm = [2, 4, 0, 3, 1]
        P = build_permutation_matrix(perm)

        # Check P is orthogonal: P @ P.T = I
        np.testing.assert_allclose(np.matmul(P, P.T), np.eye(5), atol=1e-7)
        np.testing.assert_allclose(np.matmul(P.T, P), np.eye(5), atol=1e-7)

        # Check Eq 13.1 form exactly matching textbook
        expected_P = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
        ], dtype=float)
        np.testing.assert_array_equal(P, expected_P)

    def test_graph_permutation_method(self):
        edges = [(0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]
        X = np.arange(10).reshape(5, 2).astype(float)
        g = Graph(num_nodes=5, edges=edges, node_features=X, node_labels=["A", "B", "C", "D", "E"])

        perm = [2, 4, 0, 3, 1]
        P = build_permutation_matrix(perm)
        g_perm = g.permute(perm)

        # Check Eq 13.4: X_tilde = P X
        expected_X = np.matmul(P, X)
        np.testing.assert_allclose(g_perm.node_features, expected_X, atol=1e-7)

        # Check Eq 13.5: A_tilde = P A P^T (Figure 13.2c)
        expected_A = np.matmul(np.matmul(P, g.A), P.T)
        np.testing.assert_allclose(g_perm.A, expected_A, atol=1e-7)
        assert g_perm.node_labels == ["C", "E", "A", "D", "B"]


class TestEquivarianceAndInvariance:
    def test_gcn_node_equivariance(self):
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        g = Graph(num_nodes=4, edges=edges)
        rng = np.random.default_rng(42)
        X = rng.normal(size=(4, 6))

        # Check random permutations
        for seed in [1, 2, 3]:
            perm = rng.permutation(4)
            is_equiv, diff = check_node_equivariance(simple_gcn_layer, g.A, X, perm)
            assert is_equiv is True
            assert diff < 1e-6

    def test_graph_pooling_invariance(self):
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        g = Graph(num_nodes=4, edges=edges)
        rng = np.random.default_rng(42)
        X = rng.normal(size=(4, 6))

        perm = [3, 1, 0, 2]
        for pool in ["mean", "sum", "max"]:
            fn = lambda A, X_mat: simple_graph_pooling(A, X_mat, pool_type=pool)
            is_inv, diff = check_graph_invariance(fn, g.A, X, perm)
            assert is_inv is True
            assert diff < 1e-6


class TestFigureGenerators:
    def test_figures_generation(self, tmp_path):
        save_dir = str(tmp_path)
        f1 = generate_figure_13_1(save_dir)
        f2 = generate_figure_13_2(save_dir)

        assert isinstance(f1, plt.Figure)
        assert isinstance(f2, plt.Figure)
        plt.close("all")
