"""
tests/test_ch13_neural_message_passing.py
=========================================
Unit tests for Chapter 13 Section 13.2: Neural Message-Passing.
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
"""

import os
from pathlib import Path
import pytest
import numpy as np

from common.machine_learning_on_graphs import Graph, build_permutation_matrix
from common.neural_message_passing import (
    relu,
    d_relu,
    sigmoid,
    softmax,
    conv2d_filter_standard,
    conv2d_filter_equivariant,
    check_conv_permutation_sensitivity,
    aggregate_sum,
    aggregate_mean,
    aggregate_norm,
    aggregate_max,
    aggregate_min,
    aggregate_deep_sets,
    MLP,
    update_linear,
    update_shared,
    update_concat,
    MessagePassingLayer,
    GCNLayer,
    GraphConvolutionalNetwork,
    NodeClassifier,
    EdgeClassifier,
    GraphClassifier,
    check_mpnn_permutation_equivariance,
    check_graph_readout_invariance,
    compute_receptive_field,
    create_figure_13_4_graph,
    create_karate_club_graph,
    create_synthetic_graph_dataset,
    generate_figure_13_3,
    generate_figure_13_4,
)


# =============================================================================
# 1. Convolutional Filters (Section 13.2.1)
# =============================================================================

def test_conv2d_filter_standard_and_equivariant():
    """Test standard 2D conv filter (Eq 13.8) and equivariant filter (Eq 13.9)."""
    patch = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    # Filter with distinct weights
    weights = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ])
    b = 0.5

    std_out = conv2d_filter_standard(patch, weights, bias=b)
    expected_linear = np.sum(patch * weights) + b
    assert np.isclose(std_out, expected_linear)

    # Equivariant filter
    center_val = patch[1, 1]  # 5.0
    neighbors = np.delete(patch.flatten(), 4)  # 1,2,3,4,6,7,8,9
    w_self = 0.5
    w_neigh = 0.2

    equiv_out = conv2d_filter_equivariant(center_val, neighbors, w_self, w_neigh, bias=b)
    expected_equiv = w_neigh * np.sum(neighbors) + w_self * center_val + b
    assert np.isclose(equiv_out, expected_equiv)


def test_conv_permutation_sensitivity():
    """Verify standard conv is sensitive to neighbor permutation, but graph conv is invariant."""
    patch = np.array([
        [1.2, 2.4, 0.8],
        [3.1, 5.0, 1.7],
        [4.2, 0.5, 2.9],
    ])
    weights = np.array([
        [0.1, -0.2, 0.5],
        [0.3, 1.0, -0.4],
        [0.8, 0.2, -0.1],
    ])
    res = check_conv_permutation_sensitivity(patch, weights, w_self=1.0, w_neigh=0.25)
    # Standard conv changes under permutation
    assert res["std_diff"] > 1e-4
    # Graph equivariant filter remains identical
    assert res["is_graph_equivariant"] is True
    assert res["graph_diff"] < 1e-12


# =============================================================================
# 2. Aggregation Operators (Section 13.2.3)
# =============================================================================

def test_aggregation_operators():
    """Test sum (Eq 13.12), mean (Eq 13.13), norm (Eq 13.14), max, and min aggregations."""
    embs = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])

    # Sum
    s = aggregate_sum(embs)
    assert np.allclose(s, [12.0, 15.0, 18.0])

    # Mean
    m = aggregate_mean(embs)
    assert np.allclose(m, [4.0, 5.0, 6.0])

    # Max & Min
    mx = aggregate_max(embs)
    mn = aggregate_min(embs)
    assert np.allclose(mx, [7.0, 8.0, 9.0])
    assert np.allclose(mn, [1.0, 2.0, 3.0])

    # Empty neighbors
    empty_embs = np.zeros((0, 3))
    assert np.allclose(aggregate_sum(empty_embs), np.zeros(3))
    assert np.allclose(aggregate_mean(empty_embs), np.zeros(3))

    # Normalized Kipf & Welling aggregation (Eq 13.14)
    all_H = np.array([
        [1.0, 0.0],
        [2.0, 4.0],
        [0.0, 2.0],
    ])
    degrees = np.array([2.0, 1.0, 1.0])
    # Node 0 has neighbors [1, 2]
    # norm = 1/sqrt(deg(0)*deg(1)) * H[1] + 1/sqrt(deg(0)*deg(2)) * H[2]
    #      = 1/sqrt(2) * [2, 4] + 1/sqrt(2) * [0, 2] = [2/sqrt(2), 6/sqrt(2)]
    norm_res = aggregate_norm(0, [1, 2], all_H, degrees)
    expected_norm = (all_H[1] + all_H[2]) / np.sqrt(2.0)
    assert np.allclose(norm_res, expected_norm)


def test_deep_sets_aggregation():
    """Test Deep Sets / MLP-parameterized aggregation (Eq 13.15)."""
    phi = MLP([3, 4, 3], seed=42)
    theta = MLP([3, 4, 2], seed=43)

    embs = np.array([
        [1.0, -1.0, 0.5],
        [0.2, 2.1, -0.4],
        [3.0, 0.0, 1.5],
    ])

    out1 = aggregate_deep_sets(embs, phi, theta, pool_op="sum")
    assert out1.shape == (2,)

    # Permute neighbor ordering -> output must be identical (permutation invariance)
    perm = [2, 0, 1]
    out2 = aggregate_deep_sets(embs[perm], phi, theta, pool_op="sum")
    assert np.allclose(out1, out2, atol=1e-12)


# =============================================================================
# 3. Update Operators (Section 13.2.4)
# =============================================================================

def test_update_operators():
    """Test linear (Eq 13.16), shared (Eq 13.17), and concat update operators."""
    h_self = np.array([1.0, 2.0])
    z_neigh = np.array([3.0, 4.0])

    W_self = np.array([[1.0, 0.0], [0.0, 1.0]])
    W_neigh = np.array([[2.0, 0.0], [0.0, 2.0]])
    b = np.array([0.1, -0.1])

    # Linear update
    u_lin = update_linear(h_self, z_neigh, W_self, W_neigh, b, activation=relu)
    expected_lin = np.maximum(0.0, h_self @ W_self + z_neigh @ W_neigh + b)
    assert np.allclose(u_lin, expected_lin)

    # Shared weight update
    u_sh = update_shared(h_self, z_neigh, W_self, b, activation=relu)
    expected_sh = np.maximum(0.0, (h_self + z_neigh) @ W_self + b)
    assert np.allclose(u_sh, expected_sh)

    # Concat update
    W_cat = np.ones((4, 2))
    u_cat = update_concat(h_self, z_neigh, W_cat, b, activation=relu)
    assert u_cat.shape == (2,)


# =============================================================================
# 4. Message-Passing Layer & Permutation Equivariance (Algorithm 13.1, Eq 13.19)
# =============================================================================

def test_message_passing_layer_forward():
    """Test MessagePassingLayer forward with different aggregation types."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    g = Graph(num_nodes=4, edges=edges, is_directed=False)

    for agg in ["sum", "mean", "norm", "max", "deep_sets"]:
        layer = MessagePassingLayer(in_dim=4, out_dim=3, aggregate_type=agg, seed=123)
        H_out = layer.forward(g)
        assert H_out.shape == (4, 3)
        assert np.all(np.isfinite(H_out))


def test_mpnn_permutation_equivariance():
    """Verify layer-level permutation equivariance (Eq 13.19)."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3)]
    rng = np.random.RandomState(99)
    X = rng.randn(5, 4)
    g = Graph(num_nodes=5, edges=edges, node_features=X, is_directed=False)

    for agg in ["sum", "mean", "norm", "max"]:
        layer = MessagePassingLayer(in_dim=4, out_dim=6, aggregate_type=agg, seed=42)
        assert check_mpnn_permutation_equivariance(layer, g)


# =============================================================================
# 5. GCN Layer & Multi-Layer GCN Network (Eqs 13.18, 13.19)
# =============================================================================

def test_gcn_layer_and_network():
    """Test GCNLayer and multi-layer GraphConvolutionalNetwork."""
    edges = [(0, 1), (1, 2), (2, 0)]
    A = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    X = np.eye(3)

    gcn = GraphConvolutionalNetwork(layer_dims=[3, 8, 2], seed=42)
    H_final = gcn.forward(A, X)
    assert H_final.shape == (3, 2)

    # Check forward all layers
    all_H = gcn.forward_all_layers(A, X)
    assert len(all_H) == 3
    assert all_H[0].shape == (3, 3)
    assert all_H[1].shape == (3, 8)
    assert all_H[2].shape == (3, 2)


# =============================================================================
# 6. Node Classification Head (Section 13.2.5, Eqs 13.20, 13.21)
# =============================================================================

def test_node_classifier_zacharys_karate_club():
    """Test node classification training on Zachary's Karate Club graph."""
    graph, labels = create_karate_club_graph()
    N = graph.num_nodes

    # One-hot targets
    targets = np.zeros((N, 2))
    for i, c in enumerate(labels):
        targets[i, c] = 1.0

    # Labeled train nodes: only 4 nodes (2 from each class)
    train_mask = np.zeros(N, dtype=bool)
    train_mask[[0, 1, 32, 33]] = True  # Semi-supervised setup

    classifier = NodeClassifier(in_dim=N, hidden_dims=[16], num_classes=2, seed=42)
    losses = classifier.fit(graph.A, graph.node_features, targets, train_mask, epochs=80, lr=0.1)

    # Loss should decrease
    assert losses[-1] < losses[0]

    # Predict
    preds = classifier.predict(graph.A, graph.node_features)
    train_acc = np.mean(preds[train_mask] == labels[train_mask])
    assert train_acc >= 0.99  # Fits training nodes well

    # Transductive evaluation on unlabelled nodes
    unlabeled_mask = ~train_mask
    unlabeled_acc = np.mean(preds[unlabeled_mask] == labels[unlabeled_mask])
    assert unlabeled_acc > 0.65  # Better than chance through graph propagation


# =============================================================================
# 7. Edge Classification / Link Prediction (Section 13.2.6, Eq 13.22)
# =============================================================================

def test_edge_classifier_link_prediction():
    """Test pairwise edge probability p(n, m) = sigma(h_n^T h_m) (Eq 13.22)."""
    classifier = EdgeClassifier(embed_dim=4, use_bilinear=False)

    h1 = np.array([1.0, 0.0, 1.0, 0.0])
    h2 = np.array([1.0, 0.0, 1.0, 0.0])  # Identical -> high prob
    h3 = np.array([-1.0, 0.0, -1.0, 0.0])  # Opposite -> low prob

    p12 = classifier.predict_pair(h1, h2)
    p13 = classifier.predict_pair(h1, h3)

    assert p12 > 0.8
    assert p13 < 0.2

    # Matrix prediction
    H = np.vstack([h1, h2, h3])
    P_mat = classifier.predict_matrix(H)
    assert P_mat.shape == (3, 3)
    assert np.all(P_mat >= 0.0) and np.all(P_mat <= 1.0)
    # Symmetry of dot-product
    assert np.allclose(P_mat, P_mat.T)


# =============================================================================
# 8. Graph Classification & Readout (Section 13.2.7, Eq 13.23)
# =============================================================================

def test_graph_classifier_invariance():
    """Verify graph-level classification is invariant to node permutations (Eq 13.23)."""
    classifier = GraphClassifier(node_embed_dim=4, readout_dims=[8, 2], pool_type="sum", seed=42)

    rng = np.random.RandomState(123)
    H = rng.randn(6, 4)

    assert check_graph_readout_invariance(classifier, H)


def test_synthetic_graph_dataset_generation():
    """Test generating synthetic graph dataset for graph classification."""
    graphs, labels = create_synthetic_graph_dataset(num_graphs=10, seed=42)
    assert len(graphs) == 10
    assert len(labels) == 10
    assert set(labels) == {0, 1}


# =============================================================================
# 9. Receptive Field Analysis (Figure 13.4)
# =============================================================================

def test_receptive_field_expansion():
    """Verify receptive field growth on Figure 13.4 graph: 1 -> 3 -> 8 nodes."""
    _, _, A = create_figure_13_4_graph()
    target_node = 10

    fields = compute_receptive_field(A, target_node, max_hops=2)
    assert len(fields[0]) == 1
    assert fields[0] == {10}

    assert len(fields[1]) == 3
    assert fields[1] == {10, 6, 9}

    assert len(fields[2]) == 8
    assert fields[2] == {1, 3, 4, 6, 7, 8, 9, 10}


# =============================================================================
# 10. Publication Figure Generation
# =============================================================================

def test_figure_generation(tmp_path):
    """Test Figure 13.3 and Figure 13.4 generation and file saving."""
    fig13_3 = generate_figure_13_3(save_dir=str(tmp_path))
    assert (tmp_path / "fig_13_3_convolutional_filters.png").exists()
    assert (tmp_path / "Figure_13_3.png").exists()

    fig13_4 = generate_figure_13_4(save_dir=str(tmp_path))
    assert (tmp_path / "fig_13_4_receptive_field_expansion.png").exists()
    assert (tmp_path / "Figure_13_4.png").exists()
