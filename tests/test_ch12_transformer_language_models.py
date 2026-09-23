"""
Tests for Chapter 12 Section 12.3: Transformer Language Models
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.transformer_language_models import (
    CrossAttention,
    DecoderTransformerBlock,
    TextGenerationSampler,
    LoRALinear,
    generate_figure_12_15,
    generate_figure_12_16,
    generate_figure_12_17,
    generate_figure_12_18,
    generate_figure_12_19,
    generate_figure_12_20,
    generate_figure_12_21,
)


class TestDecoderAndCrossAttention:
    def test_decoder_causal_mask(self):
        T, d_model, num_heads = 4, 16, 2
        rng = np.random.default_rng(42)
        X = rng.normal(size=(T, d_model))

        block = DecoderTransformerBlock(d_model=d_model, num_heads=num_heads, seed=42)
        out, A = block.forward(X)

        assert out.shape == (T, d_model)
        assert A.shape == (num_heads, T, T)

        # Check causal property: upper triangular of A must be zero
        for h in range(num_heads):
            for i in range(T):
                for j in range(i + 1, T):
                    assert np.isclose(A[h, i, j], 0.0, atol=1e-5)
                # Lower triangular elements sum to 1
                np.testing.assert_allclose(np.sum(A[h, i, :i+1]), 1.0, atol=1e-5)

    def test_causal_lookahead_invariance(self):
        """Modifying future token X[3] must NOT change output at step 0 or 1."""
        T, d_model, num_heads = 4, 16, 2
        rng = np.random.default_rng(42)
        X1 = rng.normal(size=(T, d_model))
        X2 = X1.copy()
        X2[3] += 100.0  # Large perturbation at future step 3

        block = DecoderTransformerBlock(d_model=d_model, num_heads=num_heads, seed=42)
        out1, _ = block.forward(X1)
        out2, _ = block.forward(X2)

        # Outputs at steps 0, 1, 2 must be identical
        np.testing.assert_allclose(out1[:3], out2[:3], atol=1e-5)

    def test_cross_attention(self):
        T_dec, T_enc, d_model, num_heads = 3, 5, 16, 2
        rng = np.random.default_rng(42)
        Y_dec = rng.normal(size=(T_dec, d_model))
        Z_enc = rng.normal(size=(T_enc, d_model))

        cross = CrossAttention(d_model=d_model, num_heads=num_heads, seed=42)
        out, A = cross.forward(Y_dec, Z_enc)

        assert out.shape == (T_dec, d_model)
        assert A.shape == (num_heads, T_dec, T_enc)
        for h in range(num_heads):
            # Sum over encoder keys must equal 1 for every decoder query
            np.testing.assert_allclose(np.sum(A[h], axis=-1), np.ones(T_dec), atol=1e-5)


class TestTextGenerationSampling:
    def test_greedy_and_temperature(self):
        logits = np.array([1.0, 5.0, 2.0, 0.5])
        best_token = TextGenerationSampler.greedy_search(logits)
        assert best_token == 1

        # Near-zero temperature should match greedy
        t_token = TextGenerationSampler.sample_temperature(logits, temperature=1e-3, seed=42)
        assert t_token == 1

    def test_top_k_and_top_p(self):
        logits = np.array([10.0, 9.0, 1.0, 0.5, 0.1])
        # Top-2 tokens are 0 and 1
        for _ in range(10):
            k_token = TextGenerationSampler.sample_top_k(logits, k=2, seed=None)
            assert k_token in [0, 1]

        # Top-p with p=0.8 should only pick from {0, 1}
        for _ in range(10):
            p_token = TextGenerationSampler.sample_top_p(logits, p=0.8, seed=None)
            assert p_token in [0, 1]

    def test_beam_search(self):
        # Deterministic transition model
        def mock_step_fn(seq):
            # Biased towards alternating between 0 and 1
            last = seq[-1]
            if last == 0:
                return np.array([0.1, 5.0, 0.2])  # Prefers 1
            else:
                return np.array([5.0, 0.1, 0.2])  # Prefers 0

        beams = TextGenerationSampler.beam_search(
            mock_step_fn, start_tokens=[0], beam_width=2, max_steps=4
        )
        assert len(beams) == 2
        # Best sequence should be [0, 1, 0, 1, 0]
        best_seq, best_score = beams[0]
        assert best_seq == [0, 1, 0, 1, 0]
        # Descending scores
        assert beams[0][1] >= beams[1][1]


class TestLoRA:
    def test_lora_initialization_and_forward(self):
        N, D, R = 4, 32, 4
        rng = np.random.default_rng(42)
        X = rng.normal(size=(N, D))

        lora = LoRALinear(in_features=D, out_features=D, rank=R, alpha=16.0, seed=42)

        # Initially B = 0, so output equals base output X @ W_0
        out_init = lora.forward(X)
        expected_base = X @ lora.W_0
        np.testing.assert_allclose(out_init, expected_base, atol=1e-7)

        # Perturb B to simulate adaptation
        lora.B = rng.normal(size=(R, D))
        out_adapted = lora.forward(X)
        # Should now differ from base
        assert not np.allclose(out_adapted, expected_base, atol=1e-3)

        # Merging weights gives identical inference
        W_hat = lora.merge_weights()
        merged_out = X @ W_hat
        np.testing.assert_allclose(out_adapted, merged_out, atol=1e-7)


class TestFigureGenerators:
    @pytest.mark.parametrize("fig_func", [
        generate_figure_12_15,
        generate_figure_12_16,
        generate_figure_12_17,
        generate_figure_12_18,
        generate_figure_12_19,
        generate_figure_12_20,
        generate_figure_12_21,
    ])
    def test_figure_generation(self, fig_func, tmp_path):
        fig = fig_func(save_dir=str(tmp_path))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
