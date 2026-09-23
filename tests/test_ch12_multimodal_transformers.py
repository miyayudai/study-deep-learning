"""
Tests for Chapter 12 Section 12.4: Multimodal Transformers
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.multimodal_transformers import (
    VisionTransformer,
    VectorQuantizer,
    AutoregressiveImageGenerator,
    AudioMelSpectrogramProcessor,
    MultimodalTokenManager,
    generate_figure_12_22,
    generate_figure_12_23,
    generate_figure_12_24,
    generate_figure_12_25,
    generate_figure_12_26,
    generate_figure_12_27,
)


class TestVisionTransformer:
    def test_patchify_and_unpatchify(self):
        H, W, C = 32, 32, 3
        P = 8
        model = VisionTransformer(img_size=(H, W), patch_size=P, in_channels=C)

        rng = np.random.default_rng(42)
        images = rng.uniform(0, 1, size=(2, H, W, C))

        patches = model.patchify(images)
        expected_patches = (H // P) * (W // P)
        expected_dim = P * P * C
        assert patches.shape == (2, expected_patches, expected_dim)

        reconstructed = model.unpatchify(patches)
        assert reconstructed.shape == (2, H, W, C)
        np.testing.assert_allclose(images, reconstructed, atol=1e-6)

    def test_single_image_patchify(self):
        H, W, C = 16, 16, 1
        P = 4
        model = VisionTransformer(img_size=(H, W), patch_size=P, in_channels=C)

        rng = np.random.default_rng(42)
        single_img = rng.uniform(0, 1, size=(H, W, C))
        patches = model.patchify(single_img)
        assert patches.shape == (1, 16, 16)

        rec = model.unpatchify(patches[0])
        assert rec.shape == (H, W, C)
        np.testing.assert_allclose(single_img, rec, atol=1e-6)

    def test_forward_and_predict(self):
        H, W, C = 16, 16, 3
        P = 4
        num_classes = 5
        d_model = 32
        model = VisionTransformer(
            img_size=(H, W),
            patch_size=P,
            in_channels=C,
            num_classes=num_classes,
            d_model=d_model,
            num_heads=2,
            num_layers=2,
            mlp_dim=64,
            seed=42,
        )

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, size=(3, H, W, C))

        logits, probs = model.forward(x)
        assert logits.shape == (3, num_classes)
        assert probs.shape == (3, num_classes)

        # Probabilities sum to 1
        np.testing.assert_allclose(np.sum(probs, axis=-1), np.ones(3), atol=1e-5)
        # All probabilities non-negative
        assert np.all(probs >= 0.0)

        preds = model.predict(x)
        assert preds.shape == (3,)
        assert np.all(preds >= 0) and np.all(preds < num_classes)

    def test_return_attention_maps(self):
        model = VisionTransformer(
            img_size=(16, 16),
            patch_size=8,
            in_channels=1,
            num_classes=2,
            d_model=16,
            num_heads=2,
            num_layers=2,
            seed=42,
        )
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, size=(2, 16, 16, 1))

        _, _, attn_maps = model.forward(x, return_attention_maps=True)
        assert len(attn_maps) == 2  # 2 layers
        # Each layer attn map has shape (B, num_heads, seq_len, seq_len)
        # seq_len = num_patches + 1 = 4 + 1 = 5
        assert attn_maps[0].shape == (2, 2, 5, 5)


class TestVectorQuantizer:
    def test_nearest_codebook_lookup(self):
        K, D = 4, 2
        vq = VectorQuantizer(num_embeddings=K, embedding_dim=D, seed=42)
        # Set manual distinct codebook vectors
        vq.codebook = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ])

        test_points = np.array([
            [0.9, 0.1],   # Closest to codebook 0
            [0.1, 0.95],  # Closest to codebook 1
            [-0.8, -0.1], # Closest to codebook 2
            [0.05, -1.2], # Closest to codebook 3
        ])

        indices, z_q = vq.quantize(test_points)
        expected_indices = np.array([0, 1, 2, 3])
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_allclose(z_q, vq.codebook[expected_indices])

    def test_dequantize(self):
        vq = VectorQuantizer(num_embeddings=8, embedding_dim=4, seed=42)
        idx = np.array([0, 3, 7, 2])
        z_q = vq.dequantize(idx)
        assert z_q.shape == (4, 4)
        np.testing.assert_allclose(z_q, vq.codebook[idx])

    def test_losses_and_ste(self):
        vq = VectorQuantizer(num_embeddings=4, embedding_dim=3, commitment_cost=0.5, seed=42)
        x = np.ones((2, 3))
        _, z_q = vq.quantize(x)

        losses = vq.compute_loss(x, z_q)
        assert "vq_loss" in losses
        assert "commitment_loss" in losses
        assert "total_loss" in losses
        assert losses["total_loss"] >= 0.0

        # Straight-through gradient copying
        grad = np.array([[1.0, -2.0, 3.0], [0.5, 0.0, -1.5]])
        ste_grad = vq.straight_through_backward(grad)
        np.testing.assert_array_equal(grad, ste_grad)

    def test_fit_kmeans(self):
        vq = VectorQuantizer(num_embeddings=2, embedding_dim=2, seed=42)
        # Two distinct clusters
        rng = np.random.default_rng(42)
        cluster1 = rng.normal(loc=[-5.0, -5.0], scale=0.2, size=(30, 2))
        cluster2 = rng.normal(loc=[5.0, 5.0], scale=0.2, size=(30, 2))
        data = np.concatenate([cluster1, cluster2], axis=0)

        vq.fit_kmeans(data, max_iters=20)
        # One codebook should be near (-5, -5) and the other near (5, 5)
        dist_c1 = [np.linalg.norm(c - [-5.0, -5.0]) for c in vq.codebook]
        dist_c2 = [np.linalg.norm(c - [5.0, 5.0]) for c in vq.codebook]
        assert min(dist_c1) < 1.0
        assert min(dist_c2) < 1.0


class TestAutoregressiveImageGenerator:
    def test_raster_scan_coordinates(self):
        gen = AutoregressiveImageGenerator(grid_size=(3, 4))
        coords = gen.raster_scan_coordinates()
        assert len(coords) == 12
        assert coords[0] == (0, 0)
        assert coords[1] == (0, 1)
        assert coords[3] == (0, 3)
        assert coords[4] == (1, 0)
        assert coords[-1] == (2, 3)

    def test_sample_step_by_step(self):
        gen = AutoregressiveImageGenerator(grid_size=(4, 4), num_tokens=6, seed=42)
        frames = gen.sample_step_by_step()
        assert len(frames) == 16
        # In frame 0, only 1 pixel is filled
        assert np.sum(frames[0] != -1) == 1
        # In final frame, all 16 pixels are filled
        assert np.sum(frames[-1] != -1) == 16
        assert np.all(frames[-1] >= 0) and np.all(frames[-1] < 6)


class TestAudioMelSpectrogramProcessor:
    def test_mel_scale_inversion(self):
        processor = AudioMelSpectrogramProcessor()
        freqs = np.array([100.0, 440.0, 1000.0, 4000.0, 8000.0])
        mels = processor.hz_to_mel(freqs)
        rec_freqs = processor.mel_to_hz(mels)
        np.testing.assert_allclose(freqs, rec_freqs, rtol=1e-5)

    def test_filterbank_properties(self):
        processor = AudioMelSpectrogramProcessor(n_mels=40, n_fft=512)
        fb = processor.mel_filterbank
        assert fb.shape == (40, 257)
        # All non-negative
        assert np.all(fb >= 0.0)
        # Triangular peaks are <= 1.0
        assert np.all(fb <= 1.0)

    def test_spectrogram_and_patchify(self):
        processor = AudioMelSpectrogramProcessor(sample_rate=16000, n_fft=512, hop_length=160, n_mels=64)
        t, audio = processor.synthesize_whale_song(duration_sec=1.0)
        spec = processor.compute_spectrogram(audio)

        assert spec.shape[0] == 64
        assert spec.shape[1] > 0

        # Patchify for AST
        patches = processor.spectrogram_to_patches(spec, patch_size=(16, 16))
        assert patches.ndim == 2
        assert patches.shape[1] == 16 * 16


class TestMultimodalTokenManager:
    def test_vocabulary_and_formatting(self):
        mgr = MultimodalTokenManager(text_vocab_size=100, image_codebook_size=50)
        assert mgr.total_vocab_size == 5 + 100 + 50

        seq_t2i = mgr.format_sequence(text_ids=[10, 20], image_ids=[5, 15], task="text_to_image")
        assert seq_t2i[0] == mgr.BOS
        assert seq_t2i[-1] == mgr.EOS
        assert mgr.BOI in seq_t2i
        assert mgr.EOI in seq_t2i


class TestFigureGenerators:
    def test_all_figures_generation(self, tmp_path):
        save_dir = str(tmp_path)
        f22 = generate_figure_12_22(save_dir)
        f23 = generate_figure_12_23(save_dir)
        f24 = generate_figure_12_24(save_dir)
        f25 = generate_figure_12_25(save_dir)
        f26 = generate_figure_12_26(save_dir)
        f27 = generate_figure_12_27(save_dir)

        assert isinstance(f22, plt.Figure)
        assert isinstance(f23, plt.Figure)
        assert isinstance(f24, plt.Figure)
        assert isinstance(f25, plt.Figure)
        assert isinstance(f26, plt.Figure)
        assert isinstance(f27, plt.Figure)

        plt.close("all")
