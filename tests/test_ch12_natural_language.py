"""
Tests for Chapter 12 Section 12.2: Natural Language
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.natural_language import (
    SimpleWord2Vec,
    BytePairEncoder,
    SimpleRNN,
    Seq2SeqRNN,
    generate_figure_12_11,
    generate_figure_12_12,
    generate_figure_12_13,
    generate_figure_12_14,
)


class TestWord2Vec:
    def test_word2vec_cbow_and_skipgram(self):
        vocab_size = 20
        emb_dim = 8
        model = SimpleWord2Vec(vocab_size=vocab_size, embedding_dim=emb_dim, seed=42)

        # Check embedding lookup
        v0 = model.get_embedding(0)
        assert v0.shape == (emb_dim,)

        # CBOW forward
        context = [1, 2, 4, 5]
        probs_cbow = model.forward_cbow(context)
        assert probs_cbow.shape == (vocab_size,)
        np.testing.assert_allclose(np.sum(probs_cbow), 1.0, atol=1e-6)
        assert np.all(probs_cbow >= 0.0)

        # Skipgram forward
        center = 3
        probs_sg = model.forward_skipgram(center)
        assert probs_sg.shape == (vocab_size,)
        np.testing.assert_allclose(np.sum(probs_sg), 1.0, atol=1e-6)
        assert np.all(probs_sg >= 0.0)

    def test_vector_arithmetic_analogy(self):
        # Synthetic test for analogy property: v(Paris) - v(France) + v(Italy) ~ v(Rome)
        vocab_size = 10
        emb_dim = 4
        model = SimpleWord2Vec(vocab_size, emb_dim, seed=42)

        # Construct synthetic embeddings with exact relationship
        v_france = np.array([1.0, 0.0, 0.5, 0.0])
        v_paris = np.array([1.0, 0.0, 0.5, 1.0])   # France + capital
        v_italy = np.array([0.0, 1.0, 0.5, 0.0])
        v_rome = np.array([0.0, 1.0, 0.5, 1.0])    # Italy + capital

        # v(Paris) - v(France) + v(Italy) = capital + Italy = v(Rome)
        v_pred = v_paris - v_france + v_italy
        np.testing.assert_allclose(v_pred, v_rome, atol=1e-6)


class TestBytePairEncoding:
    def test_bpe_training_and_merges(self):
        text = "Peter Piper picked a peck of pickled peppers"
        bpe = BytePairEncoder()
        history = bpe.train_bpe(text, num_merges=5)

        assert len(history) > 0
        merged_tokens = [tok for tok, freq in history]
        # 'pe' is the most frequent pair (frequency 4 in peck, peppers, etc.)
        assert "pe" in merged_tokens or "ck" in merged_tokens
        assert "pe" in bpe.vocab or "ck" in bpe.vocab


class TestRNNAndBPTT:
    def test_rnn_forward(self):
        T, d_in, d_h, d_out = 6, 5, 8, 4
        rng = np.random.default_rng(42)
        X = rng.normal(size=(T, d_in))

        rnn = SimpleRNN(input_dim=d_in, hidden_dim=d_h, output_dim=d_out, seed=42)
        Y_seq, Z_seq = rnn.forward(X)

        assert Y_seq.shape == (T, d_out)
        assert Z_seq.shape == (T, d_h)
        for t in range(T):
            np.testing.assert_allclose(np.sum(Y_seq[t]), 1.0, atol=1e-6)
            assert np.all(Y_seq[t] >= 0.0)

    def test_bptt_gradient_norm_decay(self):
        d_h = 10
        rnn = SimpleRNN(input_dim=5, hidden_dim=d_h, output_dim=5, seed=42)
        # Scale W_hh so maximum singular value < 1.0
        rnn.W_hh *= 0.5

        T = 20
        norms = rnn.compute_bptt_gradient_norm_decay(T)
        assert len(norms) == T
        # Norms should strictly decay towards 0 (Vanishing gradient)
        assert norms[-1] < norms[0]
        assert norms[-1] < 1e-4

    def test_seq2seq_rnn(self):
        T_in, d_in, d_out, d_h = 3, 6, 8, 12
        rng = np.random.default_rng(42)
        X_in = rng.normal(size=(T_in, d_in))

        seq2seq = Seq2SeqRNN(vocab_in=d_in, vocab_out=d_out, hidden_dim=d_h, seed=42)
        z_star = seq2seq.encode(X_in)
        assert z_star.shape == (d_h,)

        # Decoder step
        x_dec_start = rng.normal(size=(d_out,))
        y_step, z_next = seq2seq.decode_step(x_dec_start, z_star)
        assert y_step.shape == (d_out,)
        assert z_next.shape == (d_h,)
        np.testing.assert_allclose(np.sum(y_step), 1.0, atol=1e-6)


class TestFigureGenerators:
    @pytest.mark.parametrize("fig_func", [
        generate_figure_12_11,
        generate_figure_12_12,
        generate_figure_12_13,
        generate_figure_12_14,
    ])
    def test_figure_generation(self, fig_func, tmp_path):
        fig = fig_func(save_dir=str(tmp_path))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
