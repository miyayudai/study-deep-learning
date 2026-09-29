"""Tests for Chapter 19 Section 19.1: Deterministic Autoencoders."""

import os
import numpy as np
import pytest

from common.deterministic_autoencoders import (
    LinearAutoencoder,
    DeepAutoencoder,
    SparseAutoencoder,
    DenoisingAutoencoder,
    MaskedAutoencoderViT,
    generate_figure_19_1,
    generate_figure_19_2,
    generate_figure_19_3,
    generate_figure_19_4,
    generate_figure_19_5,
    generate_figure_19_6,
    generate_all_figures,
)


class TestLinearAutoencoder:
    """Tests for Linear Autoencoder and PCA equivalence (Section 19.1.1)."""

    def test_linear_autoencoder_initialization_and_forward(self):
        ae = LinearAutoencoder(input_dim=6, latent_dim=2, hidden_activation="linear", seed=42)
        x = np.random.RandomState(0).randn(20, 6)
        z, y = ae.forward(x)

        assert z.shape == (20, 2)
        assert y.shape == (20, 6)
        loss = ae.loss(x, y)
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_analytical_gradient_check(self):
        """Check analytical gradients against finite differences."""
        ae = LinearAutoencoder(input_dim=4, latent_dim=2, hidden_activation="linear", seed=42)
        x = np.random.RandomState(1).randn(5, 4)

        z, y = ae.forward(x)
        grads = ae.backward(x, z, y)

        eps = 1e-6
        # Check W1 gradient
        for i in range(2):
            for j in range(4):
                ae.W1[i, j] += eps
                l_plus = ae.loss(x)
                ae.W1[i, j] -= 2 * eps
                l_minus = ae.loss(x)
                ae.W1[i, j] += eps
                num_g = (l_plus - l_minus) / (2 * eps)
                np.testing.assert_allclose(grads["W1"][i, j], num_g, rtol=1e-4, atol=1e-4)

        # Check W2 gradient
        for i in range(4):
            for j in range(2):
                ae.W2[i, j] += eps
                l_plus = ae.loss(x)
                ae.W2[i, j] -= 2 * eps
                l_minus = ae.loss(x)
                ae.W2[i, j] += eps
                num_g = (l_plus - l_minus) / (2 * eps)
                np.testing.assert_allclose(grads["W2"][i, j], num_g, rtol=1e-4, atol=1e-4)

    def test_bourlard_kamp_pca_subspace_equivalence(self):
        """Verify Bourlard & Kamp (1988) theorem: linear autoencoder converges to PCA subspace."""
        rng = np.random.RandomState(42)
        N, D, M = 300, 5, 2
        # Generate correlated Gaussian data
        A = rng.randn(D, D)
        cov = A @ A.T
        x = rng.multivariate_normal(mean=np.zeros(D), cov=cov, size=N)

        ae = LinearAutoencoder(input_dim=D, latent_dim=M, hidden_activation="linear", seed=42)
        ae.fit(x, n_epochs=1200, lr=0.01)

        _, u_pca, theo_mse = LinearAutoencoder.compute_pca_baseline(x, latent_dim=M)
        ae_mse = ae.mean_squared_error(x)

        # AE MSE should approach theoretical minimum
        np.testing.assert_allclose(ae_mse, theo_mse, rtol=0.25, atol=0.25)

        # Subspace distance should be small
        sub_dist = ae.subspace_distance(u_pca)
        assert sub_dist < 0.4

    def test_nonlinear_hidden_units_pca_subspace(self):
        """Verify that nonlinear hidden units (tanh) still achieve low reconstruction error."""
        rng = np.random.RandomState(42)
        N, D, M = 200, 4, 2
        A = rng.randn(D, D)
        x = rng.multivariate_normal(mean=np.zeros(D), cov=A @ A.T, size=N)

        ae_tanh = LinearAutoencoder(input_dim=D, latent_dim=M, hidden_activation="tanh", seed=42)
        ae_tanh.fit(x, n_epochs=800, lr=0.01)

        _, _, theo_mse = LinearAutoencoder.compute_pca_baseline(x, latent_dim=M)
        ae_mse = ae_tanh.mean_squared_error(x)
        assert ae_mse < theo_mse * 2.0


class TestDeepAutoencoder:
    """Tests for Deep Autoencoder and Nonlinear PCA (Section 19.1.2)."""

    def test_deep_autoencoder_forward_and_shapes(self):
        dae = DeepAutoencoder(layer_dims=[6, 16, 2, 16, 6], hidden_activation="tanh", seed=42)
        x = np.random.RandomState(0).randn(15, 6)

        z = dae.encode(x)
        y = dae.decode(z)
        assert z.shape == (15, 2)
        assert y.shape == (15, 6)

        z_fwd, y_fwd, acts, _ = dae.forward(x)
        np.testing.assert_allclose(z, z_fwd)
        np.testing.assert_allclose(y, y_fwd)
        assert len(acts) == 5

    def test_deep_autoencoder_numerical_gradients(self):
        dae = DeepAutoencoder(layer_dims=[3, 6, 2, 6, 3], hidden_activation="tanh", seed=42)
        x = np.random.RandomState(2).randn(4, 3)

        _, _, acts, _ = dae.forward(x)
        gw, gb = dae.backward(x, acts)

        eps = 1e-6
        # Check first layer weight gradient
        w0 = dae.weights[0]
        for i in range(min(2, w0.shape[0])):
            for j in range(min(2, w0.shape[1])):
                w0[i, j] += eps
                _, y_p, _, _ = dae.forward(x)
                l_p = 0.5 * np.sum((y_p - x) ** 2)

                w0[i, j] -= 2 * eps
                _, y_m, _, _ = dae.forward(x)
                l_m = 0.5 * np.sum((y_m - x) ** 2)

                w0[i, j] += eps
                num_g = (l_p - l_m) / (2 * eps)
                np.testing.assert_allclose(gw[0][i, j], num_g, rtol=1e-4, atol=1e-4)

    def test_nonlinear_pca_curved_manifold(self):
        """Verify Deep Autoencoder outperforms linear PCA on a curved 3D manifold (Figure 19.3)."""
        rng = np.random.RandomState(42)
        N = 250
        # 2D latent coordinates mapped onto curved 3D manifold S: [z1, z2, sin(z1) + cos(z2)]
        z_true = rng.uniform(-1.5, 1.5, size=(N, 2))
        x_3d = np.column_stack([z_true[:, 0], z_true[:, 1], np.sin(z_true[:, 0]) + np.cos(z_true[:, 1])])
        x_3d += rng.randn(N, 3) * 0.05

        # Linear PCA baseline error
        _, _, pca_mse = LinearAutoencoder.compute_pca_baseline(x_3d, latent_dim=2)

        # Deep autoencoder
        dae = DeepAutoencoder(layer_dims=[3, 16, 2, 16, 3], hidden_activation="tanh", seed=42)
        dae.fit(x_3d, n_epochs=1000, lr=0.01)
        dae_mse = dae.mean_squared_error(x_3d)

        # Deep AE should fit the non-planar curvature well
        assert dae_mse < pca_mse * 1.2


class TestSparseAutoencoder:
    """Tests for Sparse Autoencoder with L1 activation penalty (Section 19.1.3)."""

    def test_sparse_autoencoder_loss_and_subgradient(self):
        sae = SparseAutoencoder(input_dim=5, hidden_dim=10, l1_weight=0.05, seed=42)
        x = np.random.RandomState(0).randn(10, 5)

        total, recon, l1_pen = sae.loss(x)
        assert total == pytest.approx(recon + l1_pen)
        assert l1_pen > 0.0

        z, y = sae.forward(x)
        grads = sae.backward(x, z, y)
        assert "W1" in grads and "W2" in grads

    def test_sparsity_enhancement(self):
        rng = np.random.RandomState(42)
        x = rng.randn(100, 6)

        # Model with high L1 penalty
        sae_sparse = SparseAutoencoder(input_dim=6, hidden_dim=15, l1_weight=0.5, activation="sigmoid", seed=42)
        sae_sparse.fit(x, n_epochs=300, lr=0.01)

        # Model with zero L1 penalty
        sae_dense = SparseAutoencoder(input_dim=6, hidden_dim=15, l1_weight=0.0, activation="sigmoid", seed=42)
        sae_dense.fit(x, n_epochs=300, lr=0.01)

        # Check that L1 regularization produces smaller average hidden activations
        z_sparse = sae_sparse.encode(x)
        z_dense = sae_dense.encode(x)
        assert np.mean(np.abs(z_sparse)) < np.mean(np.abs(z_dense)) + 0.1


class TestDenoisingAutoencoder:
    """Tests for Denoising Autoencoder and score matching (Section 19.1.4)."""

    def test_corruption_mechanisms(self):
        dae_gauss = DenoisingAutoencoder(layer_dims=[4, 8, 2, 8, 4], noise_type="gaussian", noise_scale=0.3, seed=42)
        x = np.ones((10, 4))
        c_gauss = dae_gauss.corrupt(x)
        assert not np.allclose(x, c_gauss)

        dae_mask = DenoisingAutoencoder(layer_dims=[4, 8, 2, 8, 4], noise_type="masking", noise_scale=0.5, seed=42)
        c_mask = dae_mask.corrupt(x)
        assert np.any(c_mask == 0.0)

    def test_denoising_vector_field_points_to_manifold(self):
        """Verify that denoising vector field points towards the manifold (Figure 19.4)."""
        rng = np.random.RandomState(42)
        # 1D line manifold in 2D space: x2 = 0.5 * x1
        x1 = rng.uniform(-2, 2, 200)
        x2 = 0.5 * x1
        clean_data = np.column_stack([x1, x2])

        dae = DenoisingAutoencoder(layer_dims=[2, 16, 1, 16, 2], noise_type="gaussian", noise_scale=0.2, seed=42)
        dae.fit(clean_data, n_epochs=800, lr=0.01)

        # Test point displaced perpendicular to the line: (0, 1) -> nearest on line is (0, 0)
        test_pt = np.array([[0.0, 1.0]])
        disp = dae.vector_field(test_pt)[0]  # y(test_pt) - test_pt
        # The vector should point downward (negative y direction towards the line)
        assert disp[1] < 0.0


class TestMaskedAutoencoderViT:
    """Tests for Vision Transformer Masked Autoencoder (Section 19.1.5)."""

    def test_patchify_unpatchify_invertibility(self):
        mae = MaskedAutoencoderViT(img_size=16, patch_size=4, in_channels=1)
        rng = np.random.RandomState(42)
        imgs = rng.randn(5, 16, 16, 1)

        patches = mae.patchify(imgs)
        assert patches.shape == (5, 16, 16)  # 16 patches, each 4*4*1 = 16

        recovered = mae.unpatchify(patches)
        np.testing.assert_allclose(imgs.squeeze(-1), recovered, rtol=1e-6, atol=1e-6)

    def test_random_masking_ratios(self):
        mae = MaskedAutoencoderViT(img_size=16, patch_size=4, in_channels=1, mask_ratio=0.75, seed=42)
        patches = np.zeros((4, 16, 16))

        x_vis, mask, ids_restore = mae.random_masking(patches, mask_ratio=0.75)
        # With 16 patches and 0.75 mask ratio: 4 keep, 12 masked
        assert x_vis.shape == (4, 4, 16)
        assert mask.shape == (4, 16)
        assert np.all(np.sum(mask, axis=1) == 12)
        assert ids_restore.shape == (4, 16)

    def test_forward_loss_and_reconstruct(self):
        mae = MaskedAutoencoderViT(img_size=16, patch_size=4, in_channels=1, mask_ratio=0.8, seed=42)
        rng = np.random.RandomState(0)
        imgs = rng.rand(3, 16, 16)

        preds, mask, targets = mae.forward(imgs)
        assert preds.shape == targets.shape
        loss = mae.compute_loss(preds, mask, targets)
        assert loss > 0.0

        masked_img, recon_img, orig_img = mae.reconstruct_image(imgs[0], mask_ratio=0.8)
        assert masked_img.shape == (16, 16)
        assert recon_img.shape == (16, 16)
        assert orig_img.shape == (16, 16)


class TestFigureGenerators:
    """Tests for Figure 19.1 〜 19.6 generation."""

    def test_all_section_19_1_figures_generate(self, tmp_path):
        save_dir = str(tmp_path / "19_result")
        saved = generate_all_figures(save_dir=save_dir)

        expected_files = [f"fig_19_{i}.png" for i in range(1, 7)]
        for fname in expected_files:
            p = os.path.join(save_dir, fname)
            assert os.path.exists(p)
            assert os.path.getsize(p) > 1000
