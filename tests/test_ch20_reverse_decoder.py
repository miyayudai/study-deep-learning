"""Tests for Chapter 20 Section 20.2: Reverse Decoder and Diffusion Generation."""

import os
import pytest
import numpy as np
from common.reverse_decoder import (
    sinusoidal_time_embedding,
    TimeConditionedMLP,
    DiffusionModel,
    generate_figure_20_5,
    generate_figure_20_6,
    generate_figure_20_7,
    generate_all_section_20_2_figures,
)


class TestSinusoidalTimeEmbedding:
    """Test suite for positional time embeddings."""

    def test_embedding_shapes_and_values(self):
        timesteps = np.array([1, 10, 50, 100])
        embed_dim = 16
        emb = sinusoidal_time_embedding(timesteps, embed_dim)

        assert emb.shape == (4, 16)
        assert np.all(emb >= -1.0) and np.all(emb <= 1.0)
        # Distinct timesteps produce distinct embeddings
        assert not np.allclose(emb[0], emb[1])


class TestTimeConditionedMLP:
    """Test suite for TimeConditionedMLP and analytical backpropagation."""

    def test_forward_pass_shape(self):
        mlp = TimeConditionedMLP(input_dim=2, hidden_dim=32, time_embed_dim=16, random_state=42)
        z = np.random.randn(5, 2)
        t = np.array([5, 12, 20, 50, 99])

        out, cache = mlp.forward(z, t)
        assert out.shape == (5, 2)
        assert "h1" in cache and "h2" in cache

    def test_analytical_gradient_check(self):
        mlp = TimeConditionedMLP(input_dim=2, hidden_dim=16, time_embed_dim=8, random_state=42)
        z = np.random.randn(4, 2)
        t = np.array([10, 20, 30, 40])

        out, cache = mlp.forward(z, t)
        grad_out = np.random.randn(4, 2)
        ana_grads = mlp.backward(grad_out, cache)

        # Finite difference check on W3
        eps = 1e-5
        num_dW3 = np.zeros_like(mlp.W3)
        for i in range(mlp.W3.shape[0]):
            for j in range(mlp.W3.shape[1]):
                mlp.W3[i, j] += eps
                out_pos, _ = mlp.forward(z, t)
                loss_pos = np.sum(out_pos * grad_out) / 4.0

                mlp.W3[i, j] -= 2.0 * eps
                out_neg, _ = mlp.forward(z, t)
                loss_neg = np.sum(out_neg * grad_out) / 4.0

                mlp.W3[i, j] += eps
                num_dW3[i, j] = (loss_pos - loss_neg) / (2.0 * eps)

        assert np.allclose(ana_grads[4], num_dW3, atol=1e-4)


class TestDiffusionModelTrainingAndSampling:
    """Test suite for DiffusionModel training loss, Adam optimization, and generation."""

    def test_training_loss_and_fit(self):
        diff = DiffusionModel(data_dim=2, T=30, random_state=42)
        rng = np.random.RandomState(42)
        X = rng.randn(40, 2) + np.array([2.0, -1.0])

        loss_0, _ = diff.compute_training_loss(X, random_state=42)
        assert loss_0 > 0.0

        # Fit model for a few steps
        losses = diff.fit(X, epochs=15, batch_size=20, lr=0.01, random_state=42)
        assert len(losses) == 15
        assert losses[-1] < losses[0]

    def test_sampling_and_trajectory(self):
        diff = DiffusionModel(data_dim=2, T=20, random_state=42)
        samples, traj = diff.sample(num_samples=5, return_trajectory=True, random_state=42)

        assert samples.shape == (5, 2)
        assert not np.any(np.isnan(samples))
        # Trajectory shape: (T + 1, num_samples, data_dim)
        assert traj.shape == (21, 5, 2)
        assert np.allclose(traj[-1], samples)


class TestSection202Figures:
    """Test suite for Section 20.2 figure generation (Figures 20.5 〜 20.7)."""

    def test_all_section_20_2_figures_generate(self):
        figs = generate_all_section_20_2_figures()
        assert len(figs) == 3
        assert "fig_20_5.png" in figs
        assert "fig_20_6.png" in figs
        assert "fig_20_7.png" in figs

        base_dir = os.path.dirname(os.path.dirname(__file__))
        for fname in ["fig_20_5.png", "fig_20_6.png", "fig_20_7.png"]:
            p1 = os.path.join(base_dir, "20", "result", fname)
            p2 = os.path.join(base_dir, "result", fname)
            assert os.path.exists(p1), f"Missing {p1}"
            assert os.path.exists(p2), f"Missing {p2}"
            assert os.path.getsize(p1) > 1000
            assert os.path.getsize(p2) > 1000
