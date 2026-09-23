"""
Tests for Chapter 12 Section 12.1: Attention
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.attention import (
    softmax,
    ScaledDotProductAttention,
    MultiHeadAttention,
    LayerNorm,
    TransformerMLP,
    TransformerLayer,
    sinusoidal_positional_encoding,
    generate_figure_12_1,
    generate_figure_12_2,
    generate_figure_12_3,
    generate_figure_12_4,
    generate_figure_12_5,
    generate_figure_12_6,
    generate_figure_12_7,
    generate_figure_12_8,
    generate_figure_12_9,
    generate_figure_12_10,
)


class TestAttentionMath:
    def test_softmax_properties(self):
        x = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
        s = softmax(x, axis=-1)
        # Sum to 1 along last axis
        np.testing.assert_allclose(np.sum(s, axis=-1), [1.0, 1.0], atol=1e-7)
        # All non-negative
        assert np.all(s >= 0.0)
        # Translation invariance: shifting by constant does not change softmax
        s_shifted = softmax(x - 500.0, axis=-1)
        np.testing.assert_allclose(s, s_shifted, atol=1e-7)

    def test_scaled_dot_product_attention_basic(self):
        N, M, d_k, d_v = 4, 6, 8, 12
        rng = np.random.default_rng(42)
        Q = rng.normal(size=(N, d_k))
        K = rng.normal(size=(M, d_k))
        V = rng.normal(size=(M, d_v))

        attn = ScaledDotProductAttention(d_k)
        Y, A = attn.forward(Q, K, V)

        assert Y.shape == (N, d_v)
        assert A.shape == (N, M)
        # Attention weights sum to 1 across keys
        np.testing.assert_allclose(np.sum(A, axis=-1), np.ones(N), atol=1e-7)
        assert np.all(A >= 0.0)

    def test_scaled_dot_product_attention_mask(self):
        N, d_k, d_v = 3, 4, 4
        rng = np.random.default_rng(42)
        Q = rng.normal(size=(N, d_k))
        K = rng.normal(size=(N, d_k))
        V = rng.normal(size=(N, d_v))

        # Causal mask: cannot attend to future tokens (j > i)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)

        attn = ScaledDotProductAttention(d_k)
        Y, A = attn.forward(Q, K, V, mask=mask)

        # Upper triangular elements of A (future tokens) must be 0
        assert np.isclose(A[0, 1], 0.0, atol=1e-5)
        assert np.isclose(A[0, 2], 0.0, atol=1e-5)
        assert np.isclose(A[1, 2], 0.0, atol=1e-5)
        # First row attends 100% to position 0
        assert np.isclose(A[0, 0], 1.0, atol=1e-5)


class TestMultiHeadAttention:
    def test_multi_head_attention_shapes_and_weights(self):
        N, d_model, num_heads = 5, 16, 4
        rng = np.random.default_rng(42)
        X = rng.normal(size=(N, d_model))

        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, seed=42)
        Y, all_A = mha.forward(X)

        assert Y.shape == (N, d_model)
        assert all_A.shape == (num_heads, N, N)
        for h in range(num_heads):
            np.testing.assert_allclose(np.sum(all_A[h], axis=-1), np.ones(N), atol=1e-7)
            assert np.all(all_A[h] >= 0.0)


class TestTransformerComponents:
    def test_layer_norm(self):
        N, D = 4, 8
        rng = np.random.default_rng(42)
        X = rng.normal(loc=5.0, scale=3.0, size=(N, D))

        ln = LayerNorm(d_model=D)
        Z = ln.forward(X)

        assert Z.shape == (N, D)
        # Mean across feature dim for each token is ~0
        np.testing.assert_allclose(np.mean(Z, axis=-1), np.zeros(N), atol=1e-5)
        # Variance across feature dim for each token is ~1
        np.testing.assert_allclose(np.var(Z, axis=-1), np.ones(N), atol=1e-3)

    def test_transformer_mlp(self):
        N, D, D_ff = 4, 8, 32
        rng = np.random.default_rng(42)
        Z = rng.normal(size=(N, D))

        mlp = TransformerMLP(d_model=D, d_ff=D_ff, seed=42)
        out = mlp.forward(Z)
        assert out.shape == (N, D)


class TestTransformerLayer:
    def test_post_ln_and_pre_ln(self):
        N, D, H = 4, 16, 4
        rng = np.random.default_rng(42)
        X = rng.normal(size=(N, D))

        # Post-LN
        layer_post = TransformerLayer(d_model=D, num_heads=H, pre_norm=False, seed=42)
        out_post, A_post = layer_post.forward(X)
        assert out_post.shape == (N, D)
        assert A_post.shape == (H, N, N)

        # Pre-LN
        layer_pre = TransformerLayer(d_model=D, num_heads=H, pre_norm=True, seed=42)
        out_pre, A_pre = layer_pre.forward(X)
        assert out_pre.shape == (N, D)
        assert A_pre.shape == (H, N, N)

    def test_permutation_equivariance(self):
        """
        Exercise 12.7: A transformer layer without positional encoding
        is strictly permutation equivariant.
        """
        N, D, H = 5, 8, 2
        rng = np.random.default_rng(42)
        X = rng.normal(size=(N, D))

        layer = TransformerLayer(d_model=D, num_heads=H, seed=42)
        Y_orig, _ = layer.forward(X)

        # Permute input rows
        perm = np.array([2, 0, 4, 1, 3])
        X_perm = X[perm]
        Y_perm, _ = layer.forward(X_perm)

        # Output must be permuted by the exact same permutation
        np.testing.assert_allclose(Y_perm, Y_orig[perm], atol=1e-6)


class TestPositionalEncoding:
    def test_sinusoidal_pe_properties(self):
        N, D = 50, 16
        pe = sinusoidal_positional_encoding(N, D, wavelength_base=10000.0)

        assert pe.shape == (N, D)
        # Bounded between -1 and 1
        assert np.all(pe >= -1.0) and np.all(pe <= 1.0)

        # Near positions have higher dot product than distant positions
        # Compare sim(r_10, r_11) with sim(r_10, r_40)
        sim_near = np.dot(pe[10], pe[11])
        sim_far = np.dot(pe[10], pe[40])
        assert sim_near > sim_far

    def test_pe_breaks_permutation_equivariance(self):
        N, D, H = 5, 8, 2
        rng = np.random.default_rng(42)
        X = rng.normal(size=(N, D))
        R = sinusoidal_positional_encoding(N, D)

        layer = TransformerLayer(d_model=D, num_heads=H, seed=42)
        Y_with_pe, _ = layer.forward(X + R)

        perm = np.array([2, 0, 4, 1, 3])
        # When we permute tokens but assign fresh positional encodings for their new slots:
        Y_perm_with_pe, _ = layer.forward(X[perm] + R)

        # Outputs at slot 0 now represent token 2 at slot 0, which differs from token 2 at slot 2!
        assert not np.allclose(Y_perm_with_pe, Y_with_pe[perm], atol=1e-3)


class TestFigureGenerators:
    @pytest.mark.parametrize("fig_func", [
        generate_figure_12_1,
        generate_figure_12_2,
        generate_figure_12_3,
        generate_figure_12_4,
        generate_figure_12_5,
        generate_figure_12_6,
        generate_figure_12_7,
        generate_figure_12_8,
        generate_figure_12_9,
        generate_figure_12_10,
    ])
    def test_figure_generation(self, fig_func, tmp_path):
        fig = fig_func(save_dir=str(tmp_path))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
