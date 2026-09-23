"""
tests/test_ch9_model_averaging.py
================================
Unit tests for Section 9.6: Model Averaging & Subsection 9.6.1: Dropout.
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
"""

import os
from pathlib import Path
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common.model_averaging import (
    EnsembleCommittee,
    DropoutMLP,
    LinearRegressionDropout,
    generate_figure_9_17,
    generate_figure_committee_theory,
    generate_figure_mc_dropout,
    generate_figure_dropout_coadaptation,
)


class TestEnsembleCommittee:
    """Tests for EnsembleCommittee mathematical properties (Eqs 9.42 - 9.50, 9.64 - 9.67)."""

    def test_uncorrelated_error_reduction(self):
        """Verify Eq (9.50) / Exercise 9.14: E_COM = (1/M) E_AV when errors are uncorrelated."""
        rng = np.random.RandomState(42)
        N = 10000
        M = 5
        x = np.linspace(-1, 1, N)
        y_true = np.sin(x)

        # Create M models with independent zero-mean errors
        models = []
        for m in range(M):
            err = rng.randn(N) * 0.5
            models.append(lambda x_val, e=err: y_true + e)

        committee = EnsembleCommittee(models)
        metrics = committee.evaluate_errors(x, y_true)

        E_AV = metrics["E_AV"]
        E_COM = metrics["E_COM"]
        empirical_ratio = metrics["empirical_ratio"]

        # Should be very close to 1/M = 0.2
        assert np.isclose(empirical_ratio, 1.0 / M, atol=0.03)
        assert np.isclose(metrics["mean_corr"], 0.0, atol=0.03)

    def test_jensens_inequality_bound(self):
        """Verify Eq (9.64) / Exercise 9.15: E_COM <= E_AV for any models."""
        rng = np.random.RandomState(42)
        N = 1000
        M = 4
        x = np.linspace(0, 1, N)
        y_true = np.cos(2 * np.pi * x)

        # Correlated models
        common_noise = rng.randn(N) * 0.3
        models = []
        for m in range(M):
            ind_noise = rng.randn(N) * 0.2
            models.append(lambda x_val, e=common_noise + ind_noise: y_true + e)

        committee = EnsembleCommittee(models)
        metrics = committee.evaluate_errors(x, y_true)

        assert metrics["E_COM"] <= metrics["E_AV"]
        assert 0.0 < metrics["empirical_ratio"] <= 1.0

    def test_identical_models_equality(self):
        """Verify equality holds in Eq (9.64) if and only if models are identical."""
        N = 500
        x = np.linspace(0, 1, N)
        y_true = 2.0 * x

        identical_model = lambda x_val: y_true + 0.4
        committee = EnsembleCommittee([identical_model, identical_model, identical_model])
        metrics = committee.evaluate_errors(x, y_true)

        assert np.isclose(metrics["E_COM"], metrics["E_AV"])
        assert np.isclose(metrics["empirical_ratio"], 1.0)

    def test_optimal_weights_simplex(self):
        """Verify optimal weights lie on simplex and improve committee performance (Exercise 9.17)."""
        rng = np.random.RandomState(42)
        N = 200
        M = 3
        # Model 1 has high error, model 2 has low error
        err1 = rng.randn(N) * 1.5
        err2 = rng.randn(N) * 0.2
        err3 = rng.randn(N) * 0.8
        error_matrix = np.array([err1, err2, err3])

        alpha_opt = EnsembleCommittee.compute_optimal_weights(error_matrix)

        # Simplex constraints
        assert np.isclose(np.sum(alpha_opt), 1.0)
        assert np.all(alpha_opt >= -1e-6)

        # Model 2 should receive the highest weight
        assert alpha_opt[1] > alpha_opt[0]
        assert alpha_opt[1] > alpha_opt[2]

        # Verify weighted error is lower than uniform
        err_opt = np.mean((alpha_opt @ error_matrix) ** 2)
        err_uniform = np.mean((np.mean(error_matrix, axis=0)) ** 2)
        assert err_opt < err_uniform


class TestDropoutMLP:
    """Tests for DropoutMLP forward, backward, expectation, and MC Dropout."""

    def test_inverted_dropout_expectation(self):
        """Verify that across random masks, Inverted Dropout preserves activation expectation."""
        rng = np.random.RandomState(42)
        mlp = DropoutMLP([10, 30, 5], activation="relu", seed=42)
        X = rng.randn(20, 10)

        # Sample 2000 forward passes with p_hidden = 0.5
        p_hidden = 0.5
        acts_samples = []
        for _ in range(2000):
            _, acts, _, _ = mlp.forward(X, training=True, p_hidden=p_hidden, mode="inverted")
            acts_samples.append(acts[1])  # Hidden layer activations

        mean_acts = np.mean(acts_samples, axis=0)

        # Deterministic activation without dropout
        _, acts_det, _, _ = mlp.forward(X, training=False, p_hidden=p_hidden, mode="inverted")
        det_acts = acts_det[1]

        # Expectations should match closely
        diff = np.abs(mean_acts - det_acts)
        assert np.mean(diff) < 0.03, f"Mean absolute difference too high: {np.mean(diff)}"
        assert np.max(diff) < 0.25, f"Expected activations should match expectation, max diff={np.max(diff)}"

    def test_gradient_finite_difference_fixed_mask(self):
        """Verify backpropagation gradient against numerical differentiation with fixed mask."""
        mlp = DropoutMLP([4, 6, 2], activation="tanh", seed=42)
        X = np.random.randn(3, 4)
        y = np.random.randn(3, 2)

        # Get forward pass with fixed mask
        mlp.rng = np.random.RandomState(123)
        _, acts, pre_acts, masks = mlp.forward(X, training=True, p_hidden=0.4, mode="inverted")
        grad_W, grad_b = mlp.backward(y, acts, pre_acts, masks, mode="inverted", p_hidden=0.4)

        eps = 1e-6
        # Check gradient for W[0]
        num_grad_W0 = np.zeros_like(mlp.W[0])
        for r in range(mlp.W[0].shape[0]):
            for c in range(mlp.W[0].shape[1]):
                orig = mlp.W[0][r, c]
                mlp.W[0][r, c] = orig + eps
                # Manual forward with SAME fixed mask
                # Layer 0
                a0 = X.copy()
                z1_p = a0 @ mlp.W[0] + mlp.b[0]
                a1_p = np.tanh(z1_p)
                a1_p = (a1_p * masks[1]) / (1.0 - 0.4)
                z2_p = a1_p @ mlp.W[1] + mlp.b[1]
                loss_p = 0.5 * np.mean((z2_p - y) ** 2)

                mlp.W[0][r, c] = orig - eps
                z1_m = a0 @ mlp.W[0] + mlp.b[0]
                a1_m = np.tanh(z1_m)
                a1_m = (a1_m * masks[1]) / (1.0 - 0.4)
                z2_m = a1_m @ mlp.W[1] + mlp.b[1]
                loss_m = 0.5 * np.mean((z2_m - y) ** 2)

                mlp.W[0][r, c] = orig
                num_grad_W0[r, c] = (loss_p - loss_m) / (2.0 * eps)

        rel_error = np.linalg.norm(grad_W[0] - num_grad_W0) / (np.linalg.norm(grad_W[0]) + 1e-8)
        assert rel_error < 1e-4, f"Analytical gradient does not match finite differences! Rel error: {rel_error}"

    def test_mc_dropout_uncertainty(self):
        """Verify Monte Carlo Dropout produces non-negative variance and correct output shapes."""
        mlp = DropoutMLP([2, 16, 1], activation="relu", seed=42)
        X = np.random.randn(15, 2)

        mean, var, samples = mlp.predict_mc_dropout(X, n_samples=30, p_hidden=0.4)

        assert mean.shape == (15, 1)
        assert var.shape == (15, 1)
        assert samples.shape == (30, 15, 1)
        assert np.all(var >= 0.0)

    def test_fit_decreases_training_loss(self):
        """Verify that training with dropout reduces loss over epochs."""
        mlp = DropoutMLP([2, 20, 1], activation="tanh", seed=42)
        X = np.random.randn(50, 2)
        y = np.sum(X, axis=1, keepdims=True)

        loss_history = mlp.fit(X, y, epochs=150, lr=0.05, batch_size=25, p_hidden=0.2)
        assert loss_history[-1] < loss_history[0]


class TestLinearRegressionDropout:
    """Tests for Exercise 9.18: Dropout on Linear Regression as diagonal L2 penalty."""

    def test_analytical_vs_empirical_weights(self):
        """Verify that analytical expected weights match empirical MC dropout weights (Exercise 9.18)."""
        rng = np.random.RandomState(42)
        N, D = 100, 3
        X = rng.randn(N, D)
        true_w = np.array([1.5, -2.0, 0.8])
        y = X @ true_w + rng.randn(N) * 0.1

        rho = 0.7  # 30% dropout rate
        w_anal = LinearRegressionDropout.regularized_analytical_weights(X, y, rho=rho)
        w_emp = LinearRegressionDropout.empirical_mc_weights(X, y, rho=rho, n_trials=600, seed=42)

        # Expected weights should match within sampling error
        assert np.allclose(w_anal, w_emp, atol=0.15)


class TestFigureGenerators:
    """Tests for Figure 9.17 and pedagogical figure generators."""

    def test_figure_9_17_generation(self, tmp_path):
        """Verify Figure 9.17 generation and saving."""
        fig = generate_figure_9_17(save_dir=str(tmp_path))
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3
        assert (tmp_path / "Figure_9_17.png").exists()
        assert (tmp_path / "fig_9_17_dropout_architecture.png").exists()
        plt.close(fig)

    def test_pedagogical_figures_generation(self, tmp_path):
        """Verify pedagogical figures generation."""
        fig_comm = generate_figure_committee_theory(save_dir=str(tmp_path))
        assert isinstance(fig_comm, plt.Figure)
        assert (tmp_path / "fig_9_ensemble_bias_variance_reduction.png").exists()
        plt.close(fig_comm)

        fig_mc = generate_figure_mc_dropout(save_dir=str(tmp_path))
        assert isinstance(fig_mc, plt.Figure)
        assert (tmp_path / "fig_9_mc_dropout_uncertainty.png").exists()
        plt.close(fig_mc)

        fig_co = generate_figure_dropout_coadaptation(save_dir=str(tmp_path))
        assert isinstance(fig_co, plt.Figure)
        assert (tmp_path / "fig_9_dropout_coadaptation.png").exists()
        plt.close(fig_co)
