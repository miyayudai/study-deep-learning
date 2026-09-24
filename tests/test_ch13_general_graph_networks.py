"""
tests/test_ch13_general_graph_networks.py
=========================================
Unit tests for Chapter 13 Section 13.3: General Graph Networks.
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
"""

import os
from pathlib import Path
import pytest
import numpy as np

from common.machine_learning_on_graphs import Graph, build_permutation_matrix
from common.general_graph_networks import (
    compute_attention_bilinear,
    compute_attention_mlp,
    GATLayer,
    MultiHeadGATLayer,
    EdgeNodeMPNNLayer,
    GeneralMPNNLayer,
    compute_dirichlet_energy,
    compute_mean_pairwise_distance,
    jumping_knowledge_pool,
    apply_drop_edge,
    apply_node_dropout,
    EGNNLayer,
    check_egnn_equivariance,
    generate_figure_13_5,
)


# =============================================================================
# 1. Graph Attention Networks (GAT, Section 13.3.1, Eqs 13.24-13.28)
# =============================================================================

def test_gat_attention_coefficients():
    """Test bilinear (Eq 13.27) and MLP (Eq 13.28) attention normalization."""
    h_n = np.array([1.0, 0.5])
    neigh_embs = np.array([
        [0.8, -0.2],
        [0.1,  1.2],
        [-0.5, 0.4],
    ])
    W_attn = np.eye(2)

    # Bilinear attention
    attn_bi = compute_attention_bilinear(h_n, neigh_embs, W_attn)
    assert len(attn_bi) == 3
    # Axioms: A_{nm} >= 0 and sum A_{nm} == 1 (Eqs 13.25, 13.26)
    assert np.all(attn_bi >= 0.0)
    assert np.isclose(np.sum(attn_bi), 1.0)


def test_gat_layer_forward_and_equivariance():
    """Test single-head GAT layer forward and permutation equivariance."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    rng = np.random.RandomState(42)
    X = rng.randn(4, 5)
    graph = Graph(num_nodes=4, edges=edges, node_features=X, is_directed=False)

    gat = GATLayer(in_dim=5, out_dim=3, attn_type="bilinear", seed=42)
    H_out = gat.forward(graph)
    assert H_out.shape == (4, 3)

    # Permutation equivariance
    perm = [2, 0, 3, 1]
    P = build_permutation_matrix(perm)
    A_perm = P @ graph.A @ P.T
    X_perm = P @ X
    edges_perm = []
    for i in range(4):
        for j in range(4):
            if A_perm[i, j] > 0 and i < j:
                edges_perm.append((i, j))
    graph_perm = Graph(num_nodes=4, edges=edges_perm, node_features=X_perm, is_directed=False)

    H_perm = gat.forward(graph_perm)
    assert np.allclose(H_perm, P @ H_out, atol=1e-5)


def test_multi_head_gat():
    """Test Multi-Head GAT layer (num_heads=3, projection)."""
    edges = [(0, 1), (1, 2), (2, 0)]
    X = np.eye(3)
    graph = Graph(num_nodes=3, edges=edges, node_features=X, is_directed=False)

    mhat = MultiHeadGATLayer(in_dim=3, out_dim_per_head=4, num_heads=3, seed=12)
    H_out = mhat.forward(graph)
    assert H_out.shape == (3, 4)
    assert np.all(np.isfinite(H_out))


# =============================================================================
# 2. Edge Embeddings (Section 13.3.2, Eqs 13.29-13.31)
# =============================================================================

def test_edge_node_mpnn():
    """Test EdgeNodeMPNNLayer edge and node updating (Eqs 13.29-13.31)."""
    edges = [(0, 1), (1, 2), (2, 0)]
    graph = Graph(num_nodes=3, edges=edges, is_directed=False)

    H = np.ones((3, 4))
    E_dict = {
        (0, 1): np.array([0.5, -0.5]),
        (1, 2): np.array([0.2,  0.8]),
        (2, 0): np.array([-0.1, 0.4]),
    }

    layer = EdgeNodeMPNNLayer(node_dim=4, edge_dim=2, seed=99)
    H_next, E_next = layer.forward(graph, H, E_dict)

    assert H_next.shape == (3, 4)
    assert len(E_next) >= 3
    for k, v in E_next.items():
        assert v.shape == (2,)


# =============================================================================
# 3. Graph Embeddings & General MPNN (Section 13.3.3, Algorithm 13.2)
# =============================================================================

def test_general_mpnn_layer():
    """Test GeneralMPNNLayer with simultaneous node, edge, and graph states (Eqs 13.32-13.35)."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    graph = Graph(num_nodes=4, edges=edges, is_directed=False)

    H = np.random.RandomState(42).randn(4, 3)
    E_dict = {(u, v): np.zeros(2) for u, v in edges}
    g = np.array([1.0, 0.0])  # graph embedding (dim 2)

    layer = GeneralMPNNLayer(node_dim=3, edge_dim=2, graph_dim=2, seed=7)
    H_next, E_next, g_next = layer.forward(graph, H, E_dict, g)

    assert H_next.shape == (4, 3)
    assert g_next.shape == (2,)
    assert len(E_next) >= 4


# =============================================================================
# 4. Over-smoothing Analysis & Mitigations (Section 13.3.4, Eqs 13.36, 13.37)
# =============================================================================

def test_dirichlet_energy_and_pairwise_distance():
    """Test Dirichlet energy and mean pairwise distance metrics."""
    # Graph with 3 nodes
    A = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    L_sym = np.array([
        [1.0, -0.5, -0.5],
        [-0.5, 1.0, -0.5],
        [-0.5, -0.5, 1.0],
    ])

    # Distinct node embeddings
    H_distinct = np.array([
        [10.0, 0.0],
        [0.0, 10.0],
        [-5.0, -5.0],
    ])
    e_distinct = compute_dirichlet_energy(H_distinct, L_sym)
    dist_distinct = compute_mean_pairwise_distance(H_distinct)
    assert e_distinct > 0.0
    assert dist_distinct > 5.0

    # Over-smoothed (identical) embeddings -> Dirichlet energy = 0
    H_smoothed = np.ones((3, 2)) * 4.0
    e_smoothed = compute_dirichlet_energy(H_smoothed, L_sym)
    dist_smoothed = compute_mean_pairwise_distance(H_smoothed)
    assert np.isclose(e_smoothed, 0.0)
    assert np.isclose(dist_smoothed, 0.0)


def test_jumping_knowledge_pool():
    """Test Jumping Knowledge Network representations (Eq 13.37)."""
    h1 = np.ones((4, 2)) * 1.0
    h2 = np.ones((4, 2)) * 2.0
    h3 = np.ones((4, 2)) * 3.0

    jk_concat = jumping_knowledge_pool([h1, h2, h3], mode="concat")
    assert jk_concat.shape == (4, 6)

    jk_max = jumping_knowledge_pool([h1, h2, h3], mode="max")
    assert jk_max.shape == (4, 2)
    assert np.allclose(jk_max, 3.0)


# =============================================================================
# 5. Regularization (Section 13.3.5)
# =============================================================================

def test_drop_edge_and_node_dropout():
    """Test DropEdge and node dropout regularization."""
    A = np.ones((6, 6)) - np.eye(6)
    A_drop = apply_drop_edge(A, p_drop=0.5, seed=42)
    # Number of edges should decrease
    assert np.sum(A_drop) < np.sum(A)
    # Symmetry preserved
    assert np.allclose(A_drop, A_drop.T)

    X = np.ones((10, 4))
    X_drop = apply_node_dropout(X, p_drop=0.3, seed=42)
    assert np.sum(X_drop == 0.0) > 0


# =============================================================================
# 6. Geometric Deep Learning (EGNN / E(n)-Equivariance, Section 13.3.6)
# =============================================================================

def test_egnn_e3_equivariance():
    """Verify E(3) translation, rotation, and reflection equivariance in EGNN (Eqs 13.38-13.41)."""
    edges = [(0, 1), (1, 2), (2, 0)]
    graph = Graph(num_nodes=3, edges=edges, is_directed=False)

    rng = np.random.RandomState(42)
    H = rng.randn(3, 4)
    R = rng.randn(3, 3)  # 3D coordinates

    egnn = EGNNLayer(node_dim=4, edge_dim=2, coord_scale=0.05, seed=10)
    equiv_report = check_egnn_equivariance(egnn, graph, H, R)

    assert equiv_report["translation_equivariance"], "EGNN must be translation equivariant"
    assert equiv_report["translation_invariance_features"], "Node features must be invariant to translation"
    assert equiv_report["rotation_equivariance"], "EGNN must be rotation equivariant"
    assert equiv_report["rotation_invariance_features"], "Node features must be invariant to rotation"
    assert equiv_report["reflection_equivariance"], "EGNN must be reflection equivariant"
    assert equiv_report["reflection_invariance_features"], "Node features must be invariant to reflection"


# =============================================================================
# 7. Publication Figure Generation
# =============================================================================

def test_figure_13_5_generation(tmp_path):
    """Test Figure 13.5 generation and saving."""
    fig = generate_figure_13_5(save_dir=str(tmp_path))
    assert (tmp_path / "fig_13_5_general_graph_updates.png").exists()
    assert (tmp_path / "Figure_13_5.png").exists()
