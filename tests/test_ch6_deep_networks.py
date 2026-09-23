"""
Tests for Chapter 6 Section 6.3: Deep Networks
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts, pp. 186-194.
"""

import math
import os
import pytest
import numpy as np

from common.deep_networks import (
    montufar_linear_regions_bound,
    count_1d_linear_regions,
    DistributedFeatureEncoder,
    DeepMLP,
    TransferLearningClassifier,
    normalize_embeddings,
    info_nce_loss,
    batch_info_nce_loss,
    clip_loss,
    supervised_contrastive_loss,
    FeedForwardDAG,
    build_fig_6_15_network,
    create_image_tensor_dataset,
    tensor_contraction_example,
    generate_figure_6_13,
    generate_figure_6_14,
    generate_figure_6_15,
)


class TestLinearRegions:
    """Test Montufar et al. (2014) linear regions bound and empirical counting."""

    def test_shallow_bound(self):
        """For depth=1, bound is sum_{j=0}^D binom(M, j)."""
        width = 10
        dim = 2
        # binom(10,0) + binom(10,1) + binom(10,2) = 1 + 10 + 45 = 56
        expected = 1 + 10 + 45
        assert montufar_linear_regions_bound(depth=1, width=width, input_dim=dim) == expected

    def test_exponential_depth_scaling(self):
        """Depth increases number of regions exponentially."""
        width = 6
        dim = 2
        b1 = montufar_linear_regions_bound(depth=1, width=width, input_dim=dim)
        b2 = montufar_linear_regions_bound(depth=2, width=width, input_dim=dim)
        b3 = montufar_linear_regions_bound(depth=3, width=width, input_dim=dim)
        
        factor = width // dim  # 3
        # b2 / b1 should equal factor^dim = 3^2 = 9
        assert b2 == 9 * b1
        assert b3 == 81 * b1

    def test_invalid_parameters(self):
        """Depth, width, and dim must be >= 1."""
        with pytest.raises(ValueError):
            montufar_linear_regions_bound(depth=0, width=5, input_dim=2)
        with pytest.raises(ValueError):
            montufar_linear_regions_bound(depth=2, width=0, input_dim=2)
        with pytest.raises(ValueError):
            montufar_linear_regions_bound(depth=2, width=5, input_dim=0)

    def test_count_1d_linear_regions_relu(self):
        """Piecewise linear function f(x) = max(0, x) + max(0, x - 1) has 3 regions."""
        def f(x):
            return np.maximum(0.0, x) + np.maximum(0.0, x - 1.0)
            
        num_regions, _, _ = count_1d_linear_regions(f, x_range=(-2.0, 3.0), num_samples=3000)
        assert num_regions == 3


class TestDistributedRepresentations:
    """Test distributed vs localist representations (Section 6.3.2)."""

    def test_exponential_capacity(self):
        attrs = ["glasses", "hat", "beard"]
        encoder = DistributedFeatureEncoder(attrs)
        assert encoder.M == 3
        assert encoder.total_combinations == 8

    def test_distributed_encoding_decoding(self):
        attrs = ["glasses", "hat", "beard"]
        encoder = DistributedFeatureEncoder(attrs)
        
        vec = encoder.encode_distributed(["glasses", "beard"])
        np.testing.assert_allclose(vec, [1.0, 0.0, 1.0])
        
        decoded = encoder.decode_distributed(vec)
        assert decoded == ["glasses", "beard"]

    def test_localist_one_hot(self):
        attrs = ["glasses", "hat", "beard"]
        encoder = DistributedFeatureEncoder(attrs)
        
        # 'glasses' + 'beard' has binary 101 = 5
        loc_vec = encoder.encode_localist(["glasses", "beard"])
        assert loc_vec.shape == (8,)
        assert np.sum(loc_vec) == 1.0
        assert loc_vec[5] == 1.0


class TestDeepMLP:
    """Test Multi-Layer Perceptron forward propagation and feature extraction."""

    def test_mlp_shapes(self):
        mlp = DeepMLP(layer_dims=[3, 8, 16, 2], hidden_activation="relu", output_activation="linear", seed=42)
        assert mlp.num_layers == 3
        assert len(mlp.weights) == 3
        assert mlp.weights[0].shape == (8, 3)
        assert mlp.weights[1].shape == (16, 8)
        assert mlp.weights[2].shape == (2, 16)
        
        X = np.random.randn(10, 3)
        pre_acts, acts = mlp.forward(X)
        assert len(acts) == 4
        assert acts[0].shape == (10, 3)
        assert acts[1].shape == (10, 8)
        assert acts[2].shape == (10, 16)
        assert acts[3].shape == (10, 2)

    def test_feature_extraction(self):
        mlp = DeepMLP(layer_dims=[4, 12, 6, 2], hidden_activation="tanh", seed=42)
        X = np.random.randn(5, 4)
        features = mlp.extract_features(X, layer_idx=-2)
        assert features.shape == (5, 6)

    def test_predict_matches_forward(self):
        mlp = DeepMLP(layer_dims=[2, 4, 1], hidden_activation="relu", seed=42)
        x = np.array([[1.0, -0.5]])
        pred = mlp.predict(x)
        _, acts = mlp.forward(x)
        np.testing.assert_allclose(pred, acts[-1])


class TestTransferLearning:
    """Test transfer learning classification and backbone freezing."""

    def test_transfer_classifier_freeze(self):
        backbone = DeepMLP(layer_dims=[4, 8, 6, 2], hidden_activation="relu", seed=42)
        initial_weights = [w.copy() for w in backbone.weights]
        
        transfer_model = TransferLearningClassifier(backbone, num_target_classes=3, freeze_backbone=True)
        assert transfer_model.W_head.shape == (3, 6)
        
        X = np.random.randn(20, 4)
        y = np.random.randint(0, 3, size=20)
        losses = transfer_model.fit_head(X, y, lr=0.1, epochs=30)
        
        assert len(losses) == 30
        assert losses[-1] < losses[0]
        
        # Backbone weights must remain unchanged
        for w_init, w_curr in zip(initial_weights, backbone.weights):
            np.testing.assert_allclose(w_init, w_curr)

    def test_transfer_predictions_sum_to_one(self):
        backbone = DeepMLP(layer_dims=[3, 5, 4, 1], hidden_activation="tanh", seed=42)
        transfer_model = TransferLearningClassifier(backbone, num_target_classes=4)
        
        X = np.random.randn(7, 3)
        _, probs = transfer_model.forward(X)
        assert probs.shape == (7, 4)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(7), atol=1e-6)


class TestContrastiveLearning:
    """Test contrastive loss formulations: InfoNCE, CLIP, SupCon."""

    def test_normalize_embeddings(self):
        reps = np.array([[3.0, 4.0], [1.0, -1.0]])
        normed = normalize_embeddings(reps)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0])

    def test_info_nce_loss_perfect_vs_imperfect(self):
        anchor = np.array([1.0, 0.0])
        pos_perfect = np.array([1.0, 0.0])
        negs = np.array([[0.0, 1.0], [0.0, -1.0]])
        
        loss_perfect = info_nce_loss(anchor, pos_perfect, negs, temperature=1.0)
        
        # With imperfect positive
        pos_imperfect = np.array([0.5, 0.5])
        loss_imperfect = info_nce_loss(anchor, pos_imperfect, negs, temperature=1.0)
        
        assert loss_perfect < loss_imperfect

    def test_batch_info_nce_loss(self):
        np.random.seed(42)
        reps_a = np.random.randn(8, 16)
        reps_b = reps_a + 0.05 * np.random.randn(8, 16)
        
        loss = batch_info_nce_loss(reps_a, reps_b, temperature=0.1)
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_clip_loss_symmetric(self):
        np.random.seed(42)
        img_reps = np.random.randn(6, 8)
        txt_reps = img_reps + 0.01 * np.random.randn(6, 8)
        
        loss1 = clip_loss(img_reps, txt_reps, temperature=1.0)
        loss2 = clip_loss(txt_reps, img_reps, temperature=1.0)
        np.testing.assert_allclose(loss1, loss2, atol=1e-7)

    def test_supervised_contrastive_loss(self):
        reps = np.array([
            [1.0, 0.0],
            [0.98, 0.05],
            [0.0, 1.0],
            [0.05, 0.99]
        ])
        labels = np.array([0, 0, 1, 1])
        loss = supervised_contrastive_loss(reps, labels, temperature=0.2)
        assert loss >= 0.0


class TestFeedForwardDAG:
    """Test FeedForwardDAG network, topological sort, and analytical backprop."""

    def test_cycle_detection(self):
        dag = FeedForwardDAG(node_names=["a", "b", "c"], input_nodes=["a"], output_nodes=["c"])
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        dag.add_edge("c", "b")  # Directed cycle b -> c -> b
        with pytest.raises(ValueError, match="Cycle detected"):
            dag.topological_sort()

    def test_fig_6_15_network_structure(self):
        dag = build_fig_6_15_network()
        order = dag.topological_sort()
        assert order.index("x1") < order.index("z1")
        assert order.index("x2") < order.index("z1")
        assert order.index("z1") < order.index("z2")
        assert order.index("z1") < order.index("z3")
        assert order.index("z2") < order.index("y1")
        assert order.index("z2") < order.index("y2")
        assert order.index("z3") < order.index("y2")
        assert order.index("x2") < order.index("y2")

    def test_fig_6_15_forward_values(self):
        dag = build_fig_6_15_network()
        out = dag.predict({"x1": 0.5, "x2": -0.2})
        assert "y1" in out and "y2" in out
        assert np.isfinite(out["y1"]) and np.isfinite(out["y2"])

    def test_dag_gradient_checking(self):
        """Verify analytical DAG backpropagation gradients against numerical finite differences."""
        dag = build_fig_6_15_network()
        inp = {"x1": 0.3, "x2": -0.4}
        target = {"y1": 1.2, "y2": -0.5}
        
        # Analytical gradients
        grad_w, grad_b = dag.backward(inp, target)
        
        # Numerical gradient for each weight
        eps = 1e-6
        for edge in dag.weights:
            w_orig = dag.weights[edge]
            
            dag.weights[edge] = w_orig + eps
            out_plus = dag.predict(inp)
            e_plus = 0.5 * ((out_plus["y1"] - target["y1"])**2 + (out_plus["y2"] - target["y2"])**2)
            
            dag.weights[edge] = w_orig - eps
            out_minus = dag.predict(inp)
            e_minus = 0.5 * ((out_minus["y1"] - target["y1"])**2 + (out_minus["y2"] - target["y2"])**2)
            
            dag.weights[edge] = w_orig
            num_grad = (e_plus - e_minus) / (2 * eps)
            
            diff = abs(grad_w[edge] - num_grad)
            denom = max(abs(grad_w[edge]), abs(num_grad), 1e-4)
            assert diff / denom < 1e-4, f"Weight grad check failed for {edge}: {grad_w[edge]} vs {num_grad}"
            
        # Numerical gradient for each bias
        for node in dag.biases:
            b_orig = dag.biases[node]
            
            dag.biases[node] = b_orig + eps
            out_plus = dag.predict(inp)
            e_plus = 0.5 * ((out_plus["y1"] - target["y1"])**2 + (out_plus["y2"] - target["y2"])**2)
            
            dag.biases[node] = b_orig - eps
            out_minus = dag.predict(inp)
            e_minus = 0.5 * ((out_minus["y1"] - target["y1"])**2 + (out_minus["y2"] - target["y2"])**2)
            
            dag.biases[node] = b_orig
            num_grad = (e_plus - e_minus) / (2 * eps)
            
            diff = abs(grad_b[node] - num_grad)
            denom = max(abs(grad_b[node]), abs(num_grad), 1e-4)
            assert diff / denom < 1e-4, f"Bias grad check failed for {node}: {grad_b[node]} vs {num_grad}"


class TestTensors:
    """Test Section 6.3.7 tensor representations and contractions."""

    def test_tensor_dataset_shape(self):
        X = create_image_tensor_dataset(num_images=5, height=16, width=16, channels=3)
        assert X.shape == (5, 16, 16, 3)

    def test_tensor_contraction(self):
        X = np.random.randn(4, 8, 8, 3)
        W = np.random.randn(5, 3)
        Y = tensor_contraction_example(X, W)
        assert Y.shape == (4, 8, 8, 5)
        # Verify first element against explicit dot product
        expected_0_0_0 = X[0, 0, 0, :] @ W.T
        np.testing.assert_allclose(Y[0, 0, 0, :], expected_0_0_0)


class TestFigureArtifacts:
    """Test that all figures 6.13, 6.14, 6.15 are generated correctly."""

    def test_generate_figures(self):
        fig13 = generate_figure_6_13()
        assert fig13 is not None
        
        fig14 = generate_figure_6_14()
        assert fig14 is not None
        
        fig15 = generate_figure_6_15()
        assert fig15 is not None
        
        # Verify file existence and sizes in both locations
        for f in ["fig_6_13_transfer_learning.png", "fig_6_14_contrastive_learning.png", "fig_6_15_general_dag_network.png"]:
            path1 = os.path.join("6", "result", f)
            path2 = os.path.join("result", f)
            assert os.path.exists(path1), f"Missing {path1}"
            assert os.path.exists(path2), f"Missing {path2}"
            assert os.path.getsize(path1) > 1000, f"Empty file {path1}"
            assert os.path.getsize(path2) > 1000, f"Empty file {path2}"
