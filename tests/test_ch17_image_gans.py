"""Unit tests for Chapter 17 Section 17.2: Image GANs (DCGAN, BigGAN, CycleGAN)."""

import os
import shutil
import tempfile
import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.image_gans import (
    conv2d_transpose_spatial_shape,
    conv2d_spatial_shape,
    conv2d_transpose_forward,
    conv2d_forward,
    batchnorm2d_forward,
    leaky_relu,
    relu,
    sigmoid,
    tanh,
    DCGANGenerator,
    DCGANDiscriminator,
    CycleGANLoss,
    LatentSpaceExplorer,
    generate_figure_17_4,
    generate_figure_17_5,
    generate_figure_17_6,
    generate_figure_17_7,
    generate_figure_17_8,
    generate_figure_17_9,
    generate_figure_17_10,
)


def test_conv2d_transpose_spatial_shape():
    """Verify transposed convolution output dimensions for DCGAN progression."""
    # DCGAN generator layers with stride=2, padding=1, kernel=4
    layers = [
        (4, 4, 8, 8),
        (8, 8, 16, 16),
        (16, 16, 32, 32),
        (32, 32, 64, 64),
    ]
    for in_h, in_w, exp_h, exp_w in layers:
        out_h, out_w = conv2d_transpose_spatial_shape(in_h, in_w, k_h=4, k_w=4, stride=2, padding=1)
        assert out_h == exp_h
        assert out_w == exp_w


def test_conv2d_spatial_shape():
    """Verify standard convolution output dimensions for DCGAN discriminator."""
    layers = [
        (64, 64, 32, 32),
        (32, 32, 16, 16),
        (16, 16, 8, 8),
        (8, 8, 4, 4),
    ]
    for in_h, in_w, exp_h, exp_w in layers:
        out_h, out_w = conv2d_spatial_shape(in_h, in_w, k_h=4, k_w=4, stride=2, padding=1)
        assert out_h == exp_h
        assert out_w == exp_w

    # Final discriminator layer: 4x4 with kernel 4x4, stride 1, pad 0 -> 1x1
    out_h, out_w = conv2d_spatial_shape(4, 4, k_h=4, k_w=4, stride=1, padding=0)
    assert out_h == 1
    assert out_w == 1


def test_conv2d_transpose_forward():
    """Test 2D transposed convolution forward pass tensor operations."""
    x = np.random.randn(2, 4, 4, 4)
    w = np.random.randn(4, 8, 4, 4)
    b = np.random.randn(8)
    out = conv2d_transpose_forward(x, w, b, stride=2, padding=1)
    assert out.shape == (2, 8, 8, 8)
    assert not np.isnan(out).any()


def test_conv2d_forward():
    """Test 2D convolution forward pass tensor operations."""
    x = np.random.randn(2, 4, 8, 8)
    w = np.random.randn(8, 4, 4, 4)
    b = np.random.randn(8)
    out = conv2d_forward(x, w, b, stride=2, padding=1)
    assert out.shape == (2, 8, 4, 4)
    assert not np.isnan(out).any()


def test_batchnorm2d_and_activations():
    """Verify batch normalization statistics and non-linearities."""
    x = np.random.normal(5.0, 2.0, (10, 4, 8, 8))
    bn_out = batchnorm2d_forward(x)
    assert np.allclose(np.mean(bn_out, axis=(0, 2, 3)), 0.0, atol=1e-4)
    assert np.allclose(np.var(bn_out, axis=(0, 2, 3)), 1.0, atol=1e-4)

    vals = np.array([-2.0, 0.0, 2.0])
    assert np.allclose(leaky_relu(vals, 0.2), [-0.4, 0.0, 2.0])
    assert np.allclose(relu(vals), [0.0, 0.0, 2.0])
    assert np.allclose(sigmoid(np.array([0.0])), [0.5])
    assert np.allclose(tanh(np.array([0.0])), [0.0])


def test_dcgan_generator_and_discriminator():
    """Test full forward pass of DCGAN generator and discriminator models."""
    generator = DCGANGenerator(latent_dim=20, base_channels=4, out_channels=3, seed=42)
    discriminator = DCGANDiscriminator(in_channels=3, base_channels=4, seed=42)

    z = np.random.randn(2, 20)
    fake_imgs = generator.forward(z)
    assert fake_imgs.shape == (2, 3, 64, 64)
    assert np.all(fake_imgs >= -1.0) and np.all(fake_imgs <= 1.0)

    # Sample method
    sampled = generator.sample(num_samples=3, seed=123)
    assert sampled.shape == (3, 3, 64, 64)

    # Discriminator forward
    probs = discriminator.forward(fake_imgs)
    assert probs.shape == (2, 1)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_cyclegan_losses():
    """Verify CycleGAN cycle consistency loss (Eq 17.12) and total error (Eq 17.13)."""
    x = np.random.randn(5, 3, 32, 32)
    y = np.random.randn(5, 3, 32, 32)

    # Perfect reconstruction -> zero cycle error
    err_zero = CycleGANLoss.cycle_consistency_error(x, x, y, y)
    assert np.isclose(err_zero, 0.0)

    # Non-perfect reconstruction
    x_rec = x + 0.1
    y_rec = y - 0.2
    err = CycleGANLoss.cycle_consistency_error(x, x_rec, y, y_rec)
    assert err > 0.0

    # Total error (Eq 17.13)
    e_gan_x = 0.5
    e_gan_y = 0.6
    e_cyc = 1.2
    eta = 10.0
    total = CycleGANLoss.total_cyclegan_error(e_gan_x, e_gan_y, e_cyc, eta=eta)
    expected = 0.5 + 0.6 + 10.0 * 1.2
    assert np.isclose(total, expected)

    # Identity error
    id_err = CycleGANLoss.identity_error(x, x, y, y)
    assert np.isclose(id_err, 0.0)


def test_latent_space_exploration():
    """Verify linear and spherical latent interpolation and vector arithmetic."""
    z0 = np.array([1.0, 0.0, 0.0])
    z1 = np.array([0.0, 1.0, 0.0])

    # Linear interpolation
    lin_interp = LatentSpaceExplorer.linear_interpolate(z0, z1, num_steps=5)
    assert lin_interp.shape == (5, 3)
    assert np.allclose(lin_interp[0], z0)
    assert np.allclose(lin_interp[-1], z1)

    # Spherical interpolation (slerp)
    slerp_interp = LatentSpaceExplorer.spherical_interpolate(z0, z1, num_steps=5)
    assert slerp_interp.shape == (5, 3)
    assert np.allclose(slerp_interp[0], z0)
    assert np.allclose(slerp_interp[-1], z1)
    # Unit norms preserved
    norms = np.linalg.norm(slerp_interp, axis=1)
    assert np.allclose(norms, 1.0)

    # Vector arithmetic: (smiling woman) - (neutral woman) + (neutral man)
    z_pos1 = np.array([1.0, 2.0, 3.0])
    z_neg = np.array([1.0, 0.0, 0.0])
    z_pos2 = np.array([0.0, 0.0, 1.0])
    z_res = LatentSpaceExplorer.latent_vector_arithmetic(z_pos1, z_neg, z_pos2)
    assert np.allclose(z_res, [0.0, 2.0, 4.0])

    # Pixel arithmetic
    x_res = LatentSpaceExplorer.pixel_vector_arithmetic(z_pos1, z_neg, z_pos2, clip_range=(-1.0, 3.0))
    assert np.allclose(x_res, [0.0, 2.0, 3.0])


def test_figure_generators():
    """Verify all textbook figures 17.4 through 17.10 are generated without errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generators = [
            (generate_figure_17_4, "fig_17_4.png"),
            (generate_figure_17_5, "fig_17_5.png"),
            (generate_figure_17_6, "fig_17_6.png"),
            (generate_figure_17_7, "fig_17_7.png"),
            (generate_figure_17_8, "fig_17_8.png"),
            (generate_figure_17_9, "fig_17_9.png"),
            (generate_figure_17_10, "fig_17_10.png"),
        ]
        for gen_fn, filename in generators:
            path = os.path.join(tmpdir, filename)
            fig = gen_fn(save_path=path)
            assert isinstance(fig, plt.Figure)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            plt.close(fig)
