"""
tests/test_ch9_parameter_shari.py
=================================
Unit tests for Section 9.4 Parameter Sharing & Soft Weight Sharing
(Bishop & Bishop 2024, Nowlan & Hinton 1992).
"""

import os
import numpy as np
import pytest
from common.parameter_sharing import (
    SoftWeightSharingGMM,
    SoftWeightSharingMLP,
    HardWeightSharingMLP,
    generate_figure_soft_weight_sharing_prior,
    generate_figure_weight_clustering,
    generate_figure_hard_vs_soft_comparison,
)


class TestSoftWeightSharingGMM:
    def test_initialization_and_properties(self):
        gmm = SoftWeightSharingGMM(n_components=3, mu=[-1.0, 0.0, 1.0], beta=[0.0, -0.5, 0.5])
        assert gmm.K == 3
        np.testing.assert_allclose(gmm.mu, [-1.0, 0.0, 1.0])
        assert np.all(gmm.sigma2 > 0)
        assert np.all(gmm.sigma > 0)
        np.testing.assert_allclose(np.sum(gmm.pi), 1.0)
        assert np.all(gmm.pi > 0)

    def test_responsibilities_normalization(self):
        gmm = SoftWeightSharingGMM(n_components=4)
        w = np.array([-2.0, -0.5, 0.0, 0.8, 2.5])
        gamma = gmm.responsibilities(w)

        assert gamma.shape == (5, 4)
        assert np.all(gamma >= 0.0)
        assert np.all(gamma <= 1.0)
        # Each row must sum to 1 (Eq 9.24)
        row_sums = np.sum(gamma, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(5), rtol=1e-6)

    def test_finite_difference_gradients(self):
        rng = np.random.RandomState(42)
        gmm = SoftWeightSharingGMM(
            n_components=3,
            mu=[-1.5, 0.2, 1.8],
            beta=[0.2, -0.3, 0.1],
            logits=[0.4, -0.1, 0.7],
        )
        w = rng.randn(15)
        errs = gmm.check_gradients(w, lambda_reg=1.5, eps=1e-6)

        assert errs["error_w"] < 1e-5, f"Weight grad error too high: {errs['error_w']}"
        assert errs["error_mu"] < 1e-5, f"Center grad error too high: {errs['error_mu']}"
        assert errs["error_beta"] < 1e-5, f"Variance grad error too high: {errs['error_beta']}"
        assert errs["error_logits"] < 1e-5, f"Logits grad error too high: {errs['error_logits']}"

    def test_center_gradient_stationary_point(self):
        # Setting mu_j to the responsibility-weighted average of weights
        # should make dOmega / dmu_j equal zero (Eq 9.26).
        gmm = SoftWeightSharingGMM(n_components=3, mu=[0.0, 0.0, 0.0])
        w = np.array([-1.0, -0.9, 0.0, 0.1, 1.0, 1.1])
        gamma = gmm.responsibilities(w)
        optimal_mu = np.sum(gamma * w.reshape(-1, 1), axis=0) / np.sum(gamma, axis=0)
        gmm.mu = optimal_mu

        grad_mu = gmm.grad_mu(w, lambda_reg=1.0)
        np.testing.assert_allclose(grad_mu, np.zeros(3), atol=1e-5)

    def test_restoring_force_direction(self):
        gmm = SoftWeightSharingGMM(n_components=1, mu=[0.0], beta=[0.0], logits=[0.0])
        # For w < 0, restoring force should be positive (pushing towards 0)
        # For w > 0, restoring force should be negative (pushing towards 0)
        force_left = gmm.effective_force(np.array([-1.0]))
        force_right = gmm.effective_force(np.array([1.0]))
        assert force_left[0] > 0
        assert force_right[0] < 0


class TestSoftWeightSharingMLP:
    def test_mlp_forward_and_predict(self):
        mlp = SoftWeightSharingMLP(d_in=2, d_hidden=16, d_out=1, seed=0)
        X = np.random.randn(10, 2)
        h, y = mlp.forward(X)
        assert h.shape == (10, 16)
        assert y.shape == (10, 1)
        y_pred = mlp.predict(X)
        np.testing.assert_allclose(y, y_pred)

    def test_mlp_fit_reduces_loss(self):
        mlp = SoftWeightSharingMLP(d_in=1, d_hidden=32, d_out=1, n_components=3, seed=42)
        X = np.linspace(-1, 1, 40).reshape(-1, 1)
        y = np.sin(np.pi * X)

        res = mlp.fit(X, y, epochs=150, lr_w=0.03, lr_gmm=0.005, lambda_reg=0.002)
        history = res["history"]
        assert history["mse_loss"][-1] < history["mse_loss"][0]


class TestHardWeightSharingMLP:
    def test_hard_sharing_constraints(self):
        mlp = HardWeightSharingMLP(d_in=1, d_hidden=32, d_out=1, n_groups=6, seed=42)
        X = np.linspace(-1, 1, 30).reshape(-1, 1)
        y = np.cos(np.pi * X)

        losses = mlp.fit(X, y, epochs=50, lr=0.02)
        assert len(losses) == 50
        assert len(mlp.unique_weights) == 6

        # Check that actual weight matrices contain only values from unique_weights
        w1_unique = np.unique(mlp.W1)
        for val in w1_unique:
            assert np.any(np.isclose(val, mlp.unique_weights))


class TestFigureGenerators:
    def test_soft_weight_sharing_prior_figure(self, tmp_path):
        save_dir = str(tmp_path)
        fig, axes = generate_figure_soft_weight_sharing_prior(save_dir=save_dir)
        assert len(axes) == 3
        out_file = os.path.join(save_dir, "fig_9_soft_weight_sharing_prior.png")
        assert os.path.exists(out_file)
        assert os.path.getsize(out_file) > 1000

    def test_weight_clustering_figure(self, tmp_path):
        save_dir = str(tmp_path)
        fig, axes = generate_figure_weight_clustering(save_dir=save_dir)
        assert len(axes) == 3
        out_file = os.path.join(save_dir, "fig_9_soft_weight_sharing_clustering.png")
        assert os.path.exists(out_file)
        assert os.path.getsize(out_file) > 1000

    def test_hard_vs_soft_comparison_figure(self, tmp_path):
        save_dir = str(tmp_path)
        fig, axes = generate_figure_hard_vs_soft_comparison(save_dir=save_dir)
        assert len(axes) == 2
        out_file = os.path.join(save_dir, "fig_9_hard_vs_soft_comparison.png")
        assert os.path.exists(out_file)
        assert os.path.getsize(out_file) > 1000
