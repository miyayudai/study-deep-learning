"""Adversarial Training for Generative Adversarial Networks (Chapter 17, Section 17.1).

This module implements:
- Standard minimax GAN loss and non-saturating generator loss (Eq. 17.5, 17.6, 17.9, 17.10)
- Optimal discriminator and Jensen-Shannon divergence analysis
- Least-Squares GAN (LSGAN) and Wasserstein GAN with Gradient Penalty (WGAN-GP, Eq. 17.11)
- 1D Toy GAN model and training dynamics for density modeling
- Conditional GAN (cGAN) conditioning framework
- Faithful reproduction of textbook figures:
  - Figure 17.1: Schematic diagram of GAN architecture with kittens
  - Figure 17.2: Conceptual illustration of GAN training difficulties and discriminator smoothing
  - Figure 17.3: Gradients of -ln(d) vs ln(1 - d)
"""

import os
from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import norm
from PIL import Image

from common.plot_utils import save_fig


class GANLoss:
    """Error functions and objectives for Generative Adversarial Networks (Section 17.1.1 & 17.1.2)."""

    @staticmethod
    def discriminator_loss(
        d_real: np.ndarray,
        d_synth: np.ndarray,
        eps: float = 1e-12,
    ) -> float:
        r"""Compute standard cross-entropy error for discriminator (Eq. 17.6).

        .. math::
            E_{\text{disc}} = -\frac{1}{N_{\text{real}}} \sum_{n \in \text{real}} \ln d(\mathbf{x}_n)
                             - \frac{1}{N_{\text{synth}}} \sum_{n \in \text{synth}} \ln(1 - d(\mathbf{g}(\mathbf{z}_n)))
        """
        d_real_c = np.clip(d_real, eps, 1.0 - eps)
        d_synth_c = np.clip(d_synth, eps, 1.0 - eps)
        loss_real = -np.mean(np.log(d_real_c))
        loss_synth = -np.mean(np.log(1.0 - d_synth_c))
        return float(loss_real + loss_synth)

    @staticmethod
    def generator_minimax_loss(d_synth: np.ndarray, eps: float = 1e-12) -> float:
        r"""Compute original minimax loss for generator (Eq. 17.9).

        .. math::
            E_{\text{gen}}^{\text{minimax}} = -\frac{1}{N_{\text{synth}}} \sum_{n \in \text{synth}} \ln(1 - d(\mathbf{g}(\mathbf{z}_n)))
        """
        d_synth_c = np.clip(d_synth, eps, 1.0 - eps)
        return float(-np.mean(np.log(1.0 - d_synth_c)))

    @staticmethod
    def generator_non_saturating_loss(d_synth: np.ndarray, eps: float = 1e-12) -> float:
        r"""Compute modified non-saturating loss for generator (Eq. 17.10).

        .. math::
            E_{\text{gen}}^{\text{non-sat}} = -\frac{1}{N_{\text{synth}}} \sum_{n \in \text{synth}} \ln d(\mathbf{g}(\mathbf{z}_n))
        """
        d_synth_c = np.clip(d_synth, eps, 1.0 - eps)
        return float(-np.mean(np.log(d_synth_c)))

    @staticmethod
    def lsgan_losses(d_real: np.ndarray, d_synth: np.ndarray) -> Tuple[float, float]:
        r"""Least-Squares GAN (LSGAN, Mao et al. 2016) losses."""
        loss_d = 0.5 * np.mean((d_real - 1.0) ** 2) + 0.5 * np.mean(d_synth ** 2)
        loss_g = 0.5 * np.mean((d_synth - 1.0) ** 2)
        return float(loss_d), float(loss_g)

    @staticmethod
    def wgan_gp_penalty(
        grad_norm: np.ndarray,
        target_norm: float = 1.0,
    ) -> float:
        r"""Compute WGAN-GP gradient penalty (Eq. 17.11, Gulrajani et al. 2017).

        .. math::
            \mathcal{L}_{\text{GP}} = \mathbb{E}\left[ (\|\nabla_{\mathbf{x}} d(\mathbf{x})\| - 1)^2 \right]
        """
        return float(np.mean((grad_norm - target_norm) ** 2))


class OptimalDiscriminator:
    """Theoretical analysis of optimal discriminator and Jensen-Shannon divergence (Goodfellow et al. 2014)."""

    @staticmethod
    def d_star(p_data: np.ndarray, p_g: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        r"""Optimal discriminator value given real and generative densities:

        .. math::
            d^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}
        """
        denom = p_data + p_g + eps
        return p_data / denom

    @staticmethod
    def jensen_shannon_divergence(
        p_data: np.ndarray,
        p_g: np.ndarray,
        dx: float = 1.0,
        eps: float = 1e-12,
    ) -> float:
        r"""Compute Jensen-Shannon Divergence D_JS(p_data || p_g).

        .. math::
            D_{\text{JS}}(p \parallel q) = \frac{1}{2} D_{\text{KL}}(p \parallel m) + \frac{1}{2} D_{\text{KL}}(q \parallel m)
            \quad \text{where} \quad m = \frac{1}{2}(p + q)
        """
        p = np.clip(p_data, eps, None)
        q = np.clip(p_g, eps, None)
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log(p / m)) * dx
        kl_qm = np.sum(q * np.log(q / m)) * dx
        return float(0.5 * kl_pm + 0.5 * kl_qm)


class Toy1DGAN:
    r"""1D Toy GAN for analyzing training dynamics and convergence (Section 17.1.2).

    Parameters
    ----------
    latent_dim : int
        Dimensionality of latent variable z (default 1).
    hidden_dim : int
        Number of hidden units in MLP generator and discriminator.
    """

    def __init__(self, latent_dim: int = 1, hidden_dim: int = 16, seed: int = 42) -> None:
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        rng = np.random.RandomState(seed)

        # Generator: z -> hidden (tanh) -> x (linear)
        self.W_g1 = rng.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / latent_dim)
        self.b_g1 = np.zeros(hidden_dim)
        self.W_g2 = rng.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b_g2 = np.zeros(1)

        # Discriminator: x -> hidden (LeakyReLU 0.2) -> logit (linear) -> sigmoid
        self.W_d1 = rng.randn(1, hidden_dim) * np.sqrt(2.0 / 1)
        self.b_d1 = np.zeros(hidden_dim)
        self.W_d2 = rng.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b_d2 = np.zeros(1)

    def forward_generator(self, z: np.ndarray) -> np.ndarray:
        """Map latent z to data space x."""
        h1 = np.tanh(z @ self.W_g1 + self.b_g1)
        x = h1 @ self.W_g2 + self.b_g2
        return x

    def forward_discriminator(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute discriminator probability d(x) and pre-activation logits."""
        # LeakyReLU
        a1 = x @ self.W_d1 + self.b_d1
        h1 = np.where(a1 > 0, a1, 0.2 * a1)
        logits = h1 @ self.W_d2 + self.b_d2
        # Sigmoid
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        return probs, h1

    def sample_generator(self, n_samples: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Sample synthetic points by passing Gaussian noise through generator."""
        if rng is None:
            rng = np.random.RandomState()
        z = rng.randn(n_samples, self.latent_dim)
        return self.forward_generator(z)

    def train_step(
        self,
        real_data: np.ndarray,
        lr: float = 0.01,
        non_saturating: bool = True,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, float]:
        """Execute one alternating update step (Eq. 17.7 - 17.8)."""
        if rng is None:
            rng = np.random.RandomState()

        n_real = len(real_data)
        z = rng.randn(n_real, self.latent_dim)

        # 1. Update Discriminator
        synth_data = self.forward_generator(z)
        d_real, h1_real = self.forward_discriminator(real_data)
        d_synth, h1_synth = self.forward_discriminator(synth_data)

        # Gradients w.r.t discriminator weights
        # Real: d_loss/d_logit = d_real - 1
        grad_logit_real = (d_real - 1.0) / n_real
        grad_W_d2_real = h1_real.T @ grad_logit_real
        grad_b_d2_real = np.sum(grad_logit_real, axis=0)

        dh1_real = grad_logit_real @ self.W_d2.T
        da1_real = dh1_real * np.where(h1_real > 0, 1.0, 0.2)
        grad_W_d1_real = real_data.T @ da1_real
        grad_b_d1_real = np.sum(da1_real, axis=0)

        # Synth: d_loss/d_logit = d_synth
        grad_logit_synth = d_synth / n_real
        grad_W_d2_synth = h1_synth.T @ grad_logit_synth
        grad_b_d2_synth = np.sum(grad_logit_synth, axis=0)

        dh1_synth = grad_logit_synth @ self.W_d2.T
        da1_synth = dh1_synth * np.where(h1_synth > 0, 1.0, 0.2)
        grad_W_d1_synth = synth_data.T @ da1_synth
        grad_b_d1_synth = np.sum(da1_synth, axis=0)

        # Apply discriminator updates (Eq. 17.7: gradient descent)
        self.W_d2 -= lr * (grad_W_d2_real + grad_W_d2_synth)
        self.b_d2 -= lr * (grad_b_d2_real + grad_b_d2_synth)
        self.W_d1 -= lr * (grad_W_d1_real + grad_W_d1_synth)
        self.b_d1 -= lr * (grad_b_d1_real + grad_b_d1_synth)

        # 2. Update Generator
        z2 = rng.randn(n_real, self.latent_dim)
        h1_g = np.tanh(z2 @ self.W_g1 + self.b_g1)
        x_g = h1_g @ self.W_g2 + self.b_g2

        d_g, h1_dg = self.forward_discriminator(x_g)

        if non_saturating:
            # Generator minimizes -ln(d_g) -> grad w.r.t logit is (d_g - 1)
            grad_logit_g = (d_g - 1.0) / n_real
            loss_g = -float(np.mean(np.log(np.clip(d_g, 1e-12, 1.0))))
        else:
            # Minimax: generator maximizes ln(1 - d_g) or minimizes -ln(1 - d_g)
            grad_logit_g = d_g / n_real
            loss_g = -float(np.mean(np.log(np.clip(1.0 - d_g, 1e-12, 1.0))))

        dh1_dg = grad_logit_g @ self.W_d2.T
        da1_dg = dh1_dg * np.where(h1_dg > 0, 1.0, 0.2)
        dx_g = da1_dg @ self.W_d1.T

        # Backprop into generator
        grad_W_g2 = h1_g.T @ dx_g
        grad_b_g2 = np.sum(dx_g, axis=0)

        dh1_g = dx_g @ self.W_g2.T
        da1_g = dh1_g * (1.0 - h1_g ** 2)
        grad_W_g1 = z2.T @ da1_g
        grad_b_g1 = np.sum(da1_g, axis=0)

        self.W_g2 -= lr * grad_W_g2
        self.b_g2 -= lr * grad_b_g2
        self.W_g1 -= lr * grad_W_g1
        self.b_g1 -= lr * grad_b_g1

        loss_d = GANLoss.discriminator_loss(d_real, d_synth)
        return {"loss_d": loss_d, "loss_g": loss_g}


# =====================================================================
# Figure Generators for Figures 17.1, 17.2, 17.3
# =====================================================================

def generate_figure_17_1(
    save_path: Optional[str] = "result/fig_17_1_gan_architecture.png",
    kitten_dir: Optional[str] = "common/assets",
) -> plt.Figure:
    """Generate Figure 17.1: Schematic diagram of a GAN (kittens, Generator, Discriminator)."""
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.set_facecolor('white')

    # Generator Box
    gen_box = patches.FancyBboxPatch((0.26, 0.40), 0.16, 0.12,
                                    boxstyle="round,pad=0.02,rounding_size=0.03",
                                    facecolor='#ff8a8a', edgecolor='black', lw=1.2)
    ax.add_patch(gen_box)
    ax.text(0.34, 0.46, 'Generator', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(0.34, 0.35, r'$\mathbf{g}(\mathbf{z}, \mathbf{w})$', fontsize=12, ha='center', va='center')

    # Input z arrow
    ax.text(0.18, 0.46, r'$\mathbf{z}$', fontsize=13, ha='center', va='center', fontweight='bold')
    ax.annotate('', xy=(0.26, 0.46), xytext=(0.20, 0.46),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5, mutation_scale=12))

    # Kitten images
    kitten_paths = []
    if kitten_dir and os.path.exists(kitten_dir):
        kitten_paths = [os.path.join(kitten_dir, f'img-00{i}.png') for i in range(6)]
    has_kittens = all(os.path.exists(p) for p in kitten_paths) if kitten_paths else False

    def draw_image_stack(x_center, y_center, img_indices, label_title, label_y):
        offsets = [(-0.03, 0.03), (0.0, 0.0), (0.03, -0.03)]
        for (dx, dy), idx in zip(offsets, img_indices):
            px = x_center + dx
            py = y_center + dy
            rect = patches.Rectangle((px - 0.05, py - 0.06), 0.10, 0.12,
                                     facecolor='white', edgecolor='black', lw=0.8, zorder=2)
            ax.add_patch(rect)
            if has_kittens:
                img = Image.open(kitten_paths[idx])
                ax.imshow(img, extent=(px - 0.045, px + 0.045, py - 0.055, py + 0.055),
                          zorder=3)
        ax.text(x_center, label_y, label_title, fontsize=12, ha='center', va='center')

    # Real images stack (top)
    draw_image_stack(0.54, 0.72, [0, 1, 2], 'real images', 0.88)

    # Synthetic images stack (bottom)
    draw_image_stack(0.54, 0.38, [3, 4, 5], 'synthetic images', 0.22)

    # Arrow from Generator to Synthetic images
    ax.annotate('', xy=(0.47, 0.46), xytext=(0.42, 0.46),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5, mutation_scale=12))

    # Discriminator Box
    disc_box = patches.FancyBboxPatch((0.70, 0.52), 0.18, 0.12,
                                     boxstyle="round,pad=0.02,rounding_size=0.03",
                                     facecolor='#ff8a8a', edgecolor='black', lw=1.2)
    ax.add_patch(disc_box)
    ax.text(0.79, 0.58, 'Discriminator', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(0.79, 0.47, r'$d(\mathbf{x}, \boldsymbol{\phi})$', fontsize=12, ha='center', va='center')

    # Arrows from image stacks into Discriminator
    ax.plot([0.62, 0.66, 0.66, 0.70], [0.72, 0.72, 0.61, 0.61], color='black', lw=1.2)
    ax.annotate('', xy=(0.70, 0.61), xytext=(0.66, 0.61),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5, mutation_scale=12))

    ax.plot([0.62, 0.66, 0.66, 0.70], [0.38, 0.38, 0.55, 0.55], color='black', lw=1.2)
    ax.annotate('', xy=(0.70, 0.55), xytext=(0.66, 0.55),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5, mutation_scale=12))

    # Output arrow t
    ax.annotate('', xy=(0.95, 0.58), xytext=(0.88, 0.58),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5, mutation_scale=12))
    ax.text(0.97, 0.58, r'$t$', fontsize=13, ha='left', va='center', style='italic', fontweight='bold')

    ax.set_xlim(0.12, 1.02)
    ax.set_ylim(0.15, 0.95)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path, dpi=300)
    return fig


def generate_figure_17_2(
    save_path: Optional[str] = "result/fig_17_2_gan_training_difficulty.png",
) -> plt.Figure:
    """Generate Figure 17.2: Conceptual illustration of GAN training difficulties and discriminator smoothing."""
    fig, ax = plt.subplots(figsize=(8.0, 3.2))

    x = np.linspace(-6, 6, 500)
    mu_data, sigma_data = -2.0, 0.8
    mu_g, sigma_g = 2.0, 0.8

    p_data = norm.pdf(x, mu_data, sigma_data)
    p_g = norm.pdf(x, mu_g, sigma_g)

    p_data_norm = p_data / np.max(p_data) * 0.75
    p_g_norm = p_g / np.max(p_g) * 0.75

    # Optimal steep discriminator
    d_opt = 1.0 / (1.0 + np.exp(6.0 * x))
    # Smoothed discriminator
    d_smooth = 1.0 / (1.0 + np.exp(0.7 * x))

    ax.plot(x, p_data_norm, color='red', lw=1.8, label=r'$p_{\mathrm{Data}}(x)$')
    ax.plot(x, p_g_norm, color='blue', lw=1.8, label=r'$p_G(x)$')
    ax.plot(x, d_opt * 0.95, color='green', lw=1.8, label=r'$d(x)$')
    ax.plot(x, d_smooth * 0.95, color='green', lw=1.8, linestyle='--', label=r'$\tilde{d}(x)$')

    rng = np.random.RandomState(42)
    samples_data = rng.normal(mu_data, sigma_data, 7)
    samples_g = rng.normal(mu_g, sigma_g, 7)

    ax.scatter(samples_data, np.zeros_like(samples_data), color='red', s=35, zorder=5)
    ax.scatter(samples_g, np.zeros_like(samples_g), color='blue', s=35, zorder=5)

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r'$x$', fontsize=12, labelpad=5)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1.2)

    ax.legend(loc='center left', bbox_to_anchor=(1.03, 0.65), frameon=True, fontsize=11)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path, dpi=300)
    return fig


def generate_figure_17_3(
    save_path: Optional[str] = "result/fig_17_3_loss_gradients.png",
) -> plt.Figure:
    """Generate Figure 17.3: Gradients of -ln(d) vs ln(1 - d)."""
    fig, ax = plt.subplots(figsize=(6.0, 3.2))

    d = np.linspace(0.001, 0.999, 500)
    loss_neg_log_d = -np.log(d)
    loss_log_1_minus_d = np.log(1.0 - d)

    ax.plot(d, loss_neg_log_d, color='red', lw=1.8)
    ax.plot(d, loss_log_1_minus_d, color='blue', lw=1.8)
    ax.axhline(0, color='black', lw=0.8)

    ax.text(0.72, 0.85, r'$-\ln(d)$', fontsize=12, color='black')
    ax.text(0.55, -0.9, r'$\ln(1 - d)$', fontsize=12, color='black')

    ax.set_xlim(0, 1)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel(r'$x$', fontsize=12, labelpad=5)
    ax.set_xticks([0, 1])
    ax.set_yticks([-2, 0, 2])

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(1.2)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path, dpi=300)
    return fig
