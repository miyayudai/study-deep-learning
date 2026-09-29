"""Unit tests for Chapter 17 Section 17.1: Adversarial Training."""

import os
import tempfile
import numpy as np
import pytest

from common.adversarial_training import (
    GANLoss,
    OptimalDiscriminator,
    Toy1DGAN,
    generate_figure_17_1,
    generate_figure_17_2,
    generate_figure_17_3,
)


def test_gan_losses():
    """Test standard discriminator, minimax, non-saturating, and LSGAN losses."""
    d_real = np.array([0.9, 0.8, 0.95])
    d_synth = np.array([0.1, 0.2, 0.05])

    # 1. Discriminator cross entropy (Eq. 17.6)
    loss_d = GANLoss.discriminator_loss(d_real, d_synth)
    expected_d = -np.mean(np.log(d_real)) - np.mean(np.log(1.0 - d_synth))
    assert np.isclose(loss_d, expected_d)
    assert loss_d > 0.0

    # 2. Minimax generator loss (Eq. 17.9)
    loss_g_minimax = GANLoss.generator_minimax_loss(d_synth)
    expected_minimax = -np.mean(np.log(1.0 - d_synth))
    assert np.isclose(loss_g_minimax, expected_minimax)

    # 3. Non-saturating generator loss (Eq. 17.10)
    loss_g_nonsat = GANLoss.generator_non_saturating_loss(d_synth)
    expected_nonsat = -np.mean(np.log(d_synth))
    assert np.isclose(loss_g_nonsat, expected_nonsat)

    # 4. LSGAN losses
    ls_d, ls_g = GANLoss.lsgan_losses(d_real, d_synth)
    assert ls_d >= 0.0
    assert ls_g >= 0.0

    # 5. WGAN-GP gradient penalty
    grad_norm = np.array([1.2, 0.8, 1.0])
    gp = GANLoss.wgan_gp_penalty(grad_norm, target_norm=1.0)
    expected_gp = np.mean((grad_norm - 1.0)**2)
    assert np.isclose(gp, expected_gp)


def test_optimal_discriminator_and_jsd():
    """Test Goodfellow optimal discriminator d*(x) and Jensen-Shannon divergence."""
    p_data = np.array([0.2, 0.5, 0.3])
    p_g_identical = np.copy(p_data)

    # When p_g == p_data, d*(x) = 0.5 everywhere
    d_star_ident = OptimalDiscriminator.d_star(p_data, p_g_identical)
    np.testing.assert_allclose(d_star_ident, 0.5)

    # JS divergence should be zero for identical distributions
    jsd_ident = OptimalDiscriminator.jensen_shannon_divergence(p_data, p_g_identical)
    assert np.isclose(jsd_ident, 0.0, atol=1e-6)

    # For disjoint distributions
    p_data_disjoint = np.array([1.0, 0.0])
    p_g_disjoint = np.array([0.0, 1.0])
    jsd_disjoint = OptimalDiscriminator.jensen_shannon_divergence(p_data_disjoint, p_g_disjoint)
    assert np.isclose(jsd_disjoint, np.log(2.0), atol=1e-5)


def test_toy_1d_gan_forward_and_sample():
    """Test Toy1DGAN generator and discriminator forward propagation."""
    model = Toy1DGAN(latent_dim=1, hidden_dim=8, seed=42)

    # Sample generator
    rng = np.random.RandomState(42)
    synth = model.sample_generator(50, rng=rng)
    assert synth.shape == (50, 1)

    # Discriminator forward
    probs, h1 = model.forward_discriminator(synth)
    assert probs.shape == (50, 1)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert h1.shape == (50, 8)


def test_toy_1d_gan_train_step():
    """Test Toy1DGAN parameter update step (Eq. 17.7 - 17.8)."""
    model = Toy1DGAN(latent_dim=1, hidden_dim=8, seed=42)
    rng = np.random.RandomState(42)

    real_data = rng.normal(loc=2.0, scale=0.5, size=(40, 1))
    W_g1_before = np.copy(model.W_g1)
    W_d1_before = np.copy(model.W_d1)

    # Non-saturating step
    metrics = model.train_step(real_data, lr=0.01, non_saturating=True, rng=rng)
    assert "loss_d" in metrics and np.isfinite(metrics["loss_d"])
    assert "loss_g" in metrics and np.isfinite(metrics["loss_g"])

    # Parameters should update
    assert not np.allclose(model.W_g1, W_g1_before)
    assert not np.allclose(model.W_d1, W_d1_before)

    # Minimax step
    metrics_mm = model.train_step(real_data, lr=0.01, non_saturating=False, rng=rng)
    assert np.isfinite(metrics_mm["loss_g"])


def test_gradient_behavior_near_zero():
    """Test gradient comparison for -ln(d) vs ln(1 - d) near d = 0 (Figure 17.3)."""
    d_small = 0.01

    # Derivative of -ln(d) is -1/d
    grad_nonsat = -1.0 / d_small  # -100

    # Derivative of -ln(1 - d) is 1/(1 - d)
    grad_minimax = 1.0 / (1.0 - d_small)  # ~1.01

    # Magnitude of non-saturating gradient is much larger near d = 0
    assert abs(grad_nonsat) > 50.0 * abs(grad_minimax)


def test_figure_generators():
    """Verify Figure 17.1, 17.2, 17.3 generators produce valid PNG files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generators = [
            (generate_figure_17_1, "fig_17_1.png"),
            (generate_figure_17_2, "fig_17_2.png"),
            (generate_figure_17_3, "fig_17_3.png"),
        ]
        for gen, fname in generators:
            p = os.path.join(tmpdir, fname)
            fig = gen(save_path=p)
            assert os.path.exists(p)
            assert os.path.getsize(p) > 1000
            import matplotlib.pyplot as plt
            plt.close(fig)
