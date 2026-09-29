"""Tests for Chapter 19 Section 19.2: Variational Autoencoders."""

import os
import numpy as np
import pytest

from common.variational_autoencoders import (
    GaussianEncoder,
    GaussianDecoder,
    VariationalAutoencoder,
    generate_figure_19_7,
    generate_figure_19_8,
    generate_figure_19_9,
    generate_figure_19_10,
    generate_figure_19_11,
    generate_all_figures,
)


class TestEncoderAndDecoder:
    """Tests for Gaussian Encoder and Decoder networks."""

    def test_encoder_forward_and_reparameterization(self):
        enc = GaussianEncoder(input_dim=4, hidden_dim=8, latent_dim=2, seed=42)
        x = np.random.RandomState(0).randn(10, 4)

        mu, logvar, h1 = enc.forward(x)
        assert mu.shape == (10, 2)
        assert logvar.shape == (10, 2)
        assert h1.shape == (10, 8)

        # Fixed eps = 0 -> z must equal mu
        eps_zero = np.zeros_like(mu)
        z_zero, _ = enc.sample(mu, logvar, eps=eps_zero)
        np.testing.assert_allclose(z_zero, mu)

        # Random sample
        z_rand, eps = enc.sample(mu, logvar)
        assert z_rand.shape == (10, 2)
        assert eps.shape == (10, 2)

    def test_decoder_forward_and_log_prob(self):
        dec = GaussianDecoder(latent_dim=2, hidden_dim=8, output_dim=4, obs_noise_std=0.5, seed=42)
        z = np.random.RandomState(0).randn(10, 2)
        x = np.random.RandomState(1).randn(10, 4)

        y, h1 = dec.forward(z)
        assert y.shape == (10, 4)
        assert h1.shape == (10, 8)

        log_prob = dec.log_prob(x, y)
        assert log_prob.shape == (10,)
        # Log prob should be negative real numbers
        assert np.all(log_prob < 0.0)


class TestVariationalAutoencoder:
    """Tests for Complete VAE Model (Section 19.2)."""

    def test_analytical_kl_divergence_properties(self):
        # 1. Standard Gaussian: mu=0, logvar=0 -> KL = 0
        mu_zero = np.zeros((5, 3))
        logvar_zero = np.zeros((5, 3))
        kl_zero = VariationalAutoencoder.kl_divergence(mu_zero, logvar_zero)
        np.testing.assert_allclose(kl_zero, 0.0, atol=1e-12)

        # 2. Non-zero mu and logvar -> KL > 0
        mu_shift = np.ones((5, 3)) * 1.5
        logvar_shift = np.ones((5, 3)) * 0.5
        kl_pos = VariationalAutoencoder.kl_divergence(mu_shift, logvar_shift)
        assert np.all(kl_pos > 0.0)

        # 3. Monte Carlo verification of KL: E_q[ln q - ln p]
        rng = np.random.RandomState(42)
        M = 2
        mu = np.array([[1.0, -0.5]])
        logvar = np.array([[0.4, -0.2]])
        std = np.exp(0.5 * logvar)

        analytical_kl = VariationalAutoencoder.kl_divergence(mu, logvar)[0]

        # MC sample
        S = 200000
        eps = rng.randn(S, M)
        z = mu + std * eps
        # ln q(z)
        ln_q = -0.5 * M * np.log(2 * np.pi) - 0.5 * np.sum(logvar) - 0.5 * np.sum(((z - mu) / std) ** 2, axis=1)
        # ln p(z)
        ln_p = -0.5 * M * np.log(2 * np.pi) - 0.5 * np.sum(z**2, axis=1)
        mc_kl = float(np.mean(ln_q - ln_p))

        np.testing.assert_allclose(analytical_kl, mc_kl, rtol=0.03, atol=0.03)

    def test_reparameterization_gradient_correctness(self):
        """Verify analytical gradients of ELBO with fixed eps match numerical differences."""
        vae = VariationalAutoencoder(input_dim=3, hidden_dim=6, latent_dim=2, obs_noise_std=0.5, seed=42)
        x = np.random.RandomState(42).randn(4, 3)
        eps = np.random.RandomState(7).randn(4, 2)

        y, mu, logvar, z = vae.forward(x, eps=eps)
        _, _, h_enc = vae.encoder.forward(x)
        _, h_dec = vae.decoder.forward(z)

        grads = vae.backward(x, y, mu, logvar, z, eps, h_enc, h_dec, beta=1.0)

        delta = 1e-6
        # Check decoder W2 gradient
        w2 = vae.decoder.W2
        for i in range(w2.shape[0]):
            for j in range(min(2, w2.shape[1])):
                w2[i, j] += delta
                elbo_p, _, _ = vae.compute_elbo(x, beta=1.0, eps=eps)
                w2[i, j] -= 2 * delta
                elbo_m, _, _ = vae.compute_elbo(x, beta=1.0, eps=eps)
                w2[i, j] += delta
                # loss is negative ELBO, scaled by N
                num_g = -(elbo_p - elbo_m) / (2 * delta) * len(x)
                np.testing.assert_allclose(grads["dec_W2"][i, j], num_g, rtol=1e-4, atol=1e-4)

        # Check encoder W_mu gradient
        w_mu = vae.encoder.W_mu
        for i in range(w_mu.shape[0]):
            for j in range(min(2, w_mu.shape[1])):
                w_mu[i, j] += delta
                elbo_p, _, _ = vae.compute_elbo(x, beta=1.0, eps=eps)
                w_mu[i, j] -= 2 * delta
                elbo_m, _, _ = vae.compute_elbo(x, beta=1.0, eps=eps)
                w_mu[i, j] += delta
                num_g = -(elbo_p - elbo_m) / (2 * delta) * len(x)
                np.testing.assert_allclose(grads["enc_W_mu"][i, j], num_g, rtol=1e-4, atol=1e-4)

    def test_vae_training_and_generation(self):
        rng = np.random.RandomState(42)
        N = 300
        # 2D data distributed along an ellipse
        theta = rng.uniform(0, 2 * np.pi, N)
        x = np.column_stack([2.0 * np.cos(theta), 0.8 * np.sin(theta)]) + rng.randn(N, 2) * 0.05

        vae = VariationalAutoencoder(input_dim=2, hidden_dim=16, latent_dim=2, obs_noise_std=0.3, seed=42)
        initial_elbo, _, _ = vae.compute_elbo(x)
        loss_hist = vae.fit(x, n_epochs=500, lr=0.01)
        final_elbo, final_recon, final_kl = vae.compute_elbo(x)

        # ELBO should improve significantly
        assert final_elbo > initial_elbo + 1.0

        # Prior sampling should produce points in plausible range
        samples = vae.sample_prior(n_samples=50)
        assert samples.shape == (50, 2)
        assert np.all(np.abs(samples) < 10.0)

    def test_beta_vae_kl_control(self):
        rng = np.random.RandomState(42)
        x = rng.randn(100, 4)

        vae_low_beta = VariationalAutoencoder(input_dim=4, hidden_dim=8, latent_dim=2, beta=0.1, seed=42)
        vae_low_beta.fit(x, n_epochs=300, lr=0.01)
        _, _, kl_low = vae_low_beta.compute_elbo(x)

        vae_high_beta = VariationalAutoencoder(input_dim=4, hidden_dim=8, latent_dim=2, beta=5.0, seed=42)
        vae_high_beta.fit(x, n_epochs=300, lr=0.01)
        _, _, kl_high = vae_high_beta.compute_elbo(x)

        # High beta should penalize KL more heavily, resulting in smaller KL divergence
        assert kl_high < kl_low


class TestSection192Figures:
    """Tests for Figure 19.7 〜 19.11 generation."""

    def test_all_section_19_2_figures_generate(self, tmp_path):
        save_dir = str(tmp_path / "19_result")
        saved = generate_all_figures(save_dir=save_dir)

        expected_files = [f"fig_19_{i}.png" for i in range(7, 12)]
        for fname in expected_files:
            p = os.path.join(save_dir, fname)
            assert os.path.exists(p)
            assert os.path.getsize(p) > 1000
