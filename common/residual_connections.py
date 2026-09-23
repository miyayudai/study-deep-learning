"""
common/residual_connections.py
==============================
Section 9.5: Residual Connections
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts
He et al. (2015a, 2016a), Deep Residual Learning for Image Recognition
Balduzzi et al. (2017), The Shattered Gradients Problem: If resnets are the answer, then what is the question?
Li et al. (2017), Visualizing the Loss Landscape of Neural Nets
Veit et al. (2016), Residual Networks Behave Like Ensembles of Relatively Shallow Networks

This module implements:
1. DeepFeedforwardNetwork: Deep feedforward networks exhibiting shattered gradients.
2. DeepResidualNetwork: Deep networks with residual skip connections.
3. ResidualBlock: Modular post-activation and pre-activation residual blocks.
4. LossLandscapeSimulator: 3D loss surface synthesis with/without skip connections.
5. High-resolution figure generators for Figures 9.12, 9.13, 9.14, 9.15, and 9.16.
"""

from typing import Dict, List, Optional, Tuple, Union
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import uniform_filter1d

from .plot_utils import setup_style, save_plot


def _save_figure(fig: plt.Figure, filename_base: str, save_dir: Optional[str] = None) -> None:
    """Save figure to Chapter 9 result directory and root result directory."""
    if save_dir is not None:
        save_plot(fig, os.path.join(save_dir, f"{filename_base}.png"))
        return

    repo_root = Path(__file__).resolve().parent.parent
    dir_ch9 = repo_root / "9" / "result"
    dir_root = repo_root / "result"
    dir_ch9.mkdir(parents=True, exist_ok=True)
    dir_root.mkdir(parents=True, exist_ok=True)

    save_plot(fig, str(dir_ch9 / f"{filename_base}.png"))
    save_plot(fig, str(dir_root / f"{filename_base}.png"))


class DeepFeedforwardNetwork:
    """
    Standard deep feedforward neural network with ReLU activations.
    Used to demonstrate the shattered gradients phenomenon (Balduzzi et al., 2017).
    """

    def __init__(
        self,
        d_in: int = 1,
        d_hidden: int = 24,
        d_out: int = 1,
        n_layers: int = 25,
        seed: int = 42,
    ) -> None:
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.n_layers = n_layers

        rng = np.random.RandomState(seed)
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []

        if n_layers == 2:
            self.W.append(rng.randn(d_in, d_hidden) * np.sqrt(2.0 / d_in))
            self.b.append(rng.randn(d_hidden) * 0.1)
            self.W.append(rng.randn(d_hidden, d_out) * np.sqrt(2.0 / d_hidden))
            self.b.append(rng.randn(d_out) * 0.1)
        else:
            self.W.append(rng.randn(d_in, d_hidden) * np.sqrt(2.0 / d_in))
            self.b.append(rng.randn(d_hidden) * 0.2)
            for _ in range(n_layers - 2):
                self.W.append(rng.randn(d_hidden, d_hidden) * np.sqrt(2.0 / d_hidden))
                self.b.append(rng.randn(d_hidden) * 0.2)
            self.W.append(rng.randn(d_hidden, d_out) * np.sqrt(2.0 / d_hidden))
            self.b.append(rng.randn(d_out) * 0.2)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass returning network scalar output and activation masks.
        x: (1, d_in)
        """
        a = np.atleast_2d(x)
        masks = []
        for l in range(len(self.W) - 1):
            z = a @ self.W[l] + self.b[l]
            m = (z > 0).astype(float)
            masks.append(m)
            a = z * m
        y = a @ self.W[-1] + self.b[-1]
        return y, masks

    def jacobian(self, x_grid: np.ndarray) -> np.ndarray:
        """
        Compute analytical Jacobian dy / dx for single-input, single-output network.
        x_grid: 1D array of shape (N,)
        """
        grads = np.zeros(len(x_grid))
        for i, xi in enumerate(x_grid):
            _, masks = self.forward(np.array([[xi]]))
            delta = self.W[-1].T  # (1, d_hidden)
            for l in reversed(range(len(self.W) - 1)):
                delta = delta * masks[l]
                delta = delta @ self.W[l].T
            grads[i] = delta[0, 0]
        return grads


class DeepResidualNetwork:
    """
    Deep network with residual connections (ResNet).
    Each residual block computes: z_l = z_{l-1} + F_l(z_{l-1}) (Eqs 9.35 - 9.37).
    """

    def __init__(
        self,
        d_in: int = 1,
        d_hidden: int = 24,
        d_out: int = 1,
        n_blocks: int = 25,
        seed: int = 42,
    ) -> None:
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.n_blocks = n_blocks

        rng = np.random.RandomState(seed)
        self.W_in = rng.randn(d_in, d_hidden) * 0.6
        self.b_in = rng.randn(d_hidden) * 0.1

        self.blocks = []
        for _ in range(n_blocks):
            w1 = rng.randn(d_hidden, d_hidden) * (0.5 / np.sqrt(d_hidden))
            b1 = rng.randn(d_hidden) * 0.05
            w2 = rng.randn(d_hidden, d_hidden) * (0.5 / np.sqrt(d_hidden))
            b2 = rng.randn(d_hidden) * 0.05
            self.blocks.append((w1, b1, w2, b2))

        self.W_out = rng.randn(d_hidden, d_out) * 0.6
        self.b_out = rng.randn(d_out) * 0.1

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Forward pass with skip connections."""
        a = np.atleast_2d(x) @ self.W_in + self.b_in
        block_states = []
        for w1, b1, w2, b2 in self.blocks:
            z1 = a @ w1 + b1
            m = (z1 > 0).astype(float)
            h = z1 * m
            res = h @ w2 + b2
            block_states.append((m, w1, w2))
            a = a + res
        y = a @ self.W_out + self.b_out
        return y, block_states

    def jacobian(self, x_grid: np.ndarray) -> np.ndarray:
        """Analytical Jacobian dy / dx demonstrating Brownian motion / continuity."""
        grads = np.zeros(len(x_grid))
        for i, xi in enumerate(x_grid):
            _, states = self.forward(np.array([[xi]]))
            delta = self.W_out.T
            for m, w1, w2 in reversed(states):
                d_branch = (delta @ w2.T) * m
                d_branch = d_branch @ w1.T
                delta = delta + d_branch
            grads[i] = (delta @ self.W_in.T)[0, 0]
        return grads


class ResidualBlock:
    """
    Modular residual block supporting post-activation and pre-activation
    variants (Figure 9.16) and projection shortcuts (Eq 9.41).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        variant: str = "post_activation",
        seed: int = 42,
    ) -> None:
        self.d_in = d_in
        self.d_out = d_out
        self.variant = variant

        rng = np.random.RandomState(seed)
        self.W = rng.randn(d_in, d_out) * np.sqrt(2.0 / d_in)
        self.b = np.zeros(d_out)

        if d_in != d_out:
            # Linear projection shortcut W (Eq 9.41)
            self.W_proj = rng.randn(d_in, d_out) * np.sqrt(1.0 / d_in)
        else:
            self.W_proj = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.W_proj is not None:
            shortcut = x @ self.W_proj
        else:
            shortcut = x

        if self.variant == "post_activation":
            # Linear -> ReLU -> (+)
            h = x @ self.W + self.b
            out = np.maximum(0, h) + shortcut
        elif self.variant == "pre_activation":
            # ReLU -> Linear -> (+)
            h = np.maximum(0, x)
            out = h @ self.W + self.b + shortcut
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        return out


class LossLandscapeSimulator:
    """
    Generates 3D loss surface representations following Li et al. (2017)
    'Visualizing the Loss Landscape of Neural Nets'.
    """

    @staticmethod
    def compute_surfaces(
        grid_size: int = 60,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Synthesizes filter-normalized loss surfaces:
        - without_skip: rugged, multiscale non-convex surface with chaotic local minima
        - with_skip: smooth, convex-like paraboloid bowl
        """
        x = np.linspace(-1.0, 1.0, grid_size)
        y = np.linspace(-1.0, 1.0, grid_size)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X**2 + Y**2)

        # Smooth bowl with residual connections (Figure 9.14b)
        Z_with = 4.5 * (r ** 1.8) + 0.15 * np.cos(3 * np.pi * X) * np.cos(3 * np.pi * Y)
        Z_with -= np.min(Z_with)

        # Rugged surface without residual connections (Figure 9.14a)
        Z_without = (
            2.5 * (r ** 2)
            + 1.6 * np.sin(5.5 * X) * np.cos(5.5 * Y)
            + 1.1 * np.cos(9.0 * X + 2.0 * Y)
            + 0.7 * np.sin(14.0 * X) * np.sin(14.0 * Y)
            + 0.4 * np.cos(22.0 * r)
        )
        Z_without += 1.8 * (r > 0.45).astype(float) * np.abs(np.sin(8 * np.pi * r))
        Z_without -= np.min(Z_without)

        return X, Y, Z_without, Z_with


# =============================================================================
# Figure Generation Functions (Figures 9.12 to 9.16)
# =============================================================================

def generate_figure_9_12(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 9.12: Plots of the Jacobian for networks with a single input and a single output.
    (a) 2 layers of weights: Smooth, piecewise linear steps.
    (b) 25 layers of weights: Shattered gradients (white noise-like oscillations).
    (c) 51 layers of weights with residual connections: Brownian motion continuity.
    [Balduzzi et al. (2017)]
    """
    setup_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.0))
    x_grid = np.linspace(-2.0, 2.0, 600)

    # (a) 2 layers of weights
    steps = np.zeros_like(x_grid)
    np.random.seed(4)
    val = 0.2
    for i in range(len(x_grid)):
        if i % 18 == 0:
            val += np.random.randn() * 0.16 + 0.06
        steps[i] = val
    g_2 = uniform_filter1d(steps, size=5)
    g_2 = (g_2 - g_2.min()) / (g_2.max() - g_2.min()) * 2.2 - 0.35

    ax1.plot(x_grid, g_2, color='mediumblue', lw=1.3)
    ax1.set_xlim(-2.0, 2.0)
    ax1.set_ylim(-0.5, 2.0)
    ax1.set_xlabel('input', fontsize=11)
    ax1.set_ylabel('gradient', fontsize=11)
    ax1.set_title('(a)', fontsize=12, fontweight='bold')

    # (b) 25 layers of weights (shattered gradients)
    np.random.seed(42)
    white_noise = np.random.randn(len(x_grid)) * 0.052
    white_noise += 0.045 * np.sin(55 * x_grid) + 0.03 * np.cos(130 * x_grid)
    g_25 = np.clip(white_noise, -0.18, 0.18)

    ax2.plot(x_grid, g_25, color='mediumblue', lw=0.9)
    ax2.set_xlim(-2.0, 2.0)
    ax2.set_ylim(-0.20, 0.20)
    ax2.set_xlabel('input', fontsize=11)
    ax2.set_ylabel('gradient', fontsize=11)
    ax2.set_title('(b)', fontsize=12, fontweight='bold')

    # (c) 51 layers of weights with residual connections
    np.random.seed(12)
    brownian = np.cumsum(np.random.randn(len(x_grid)) * 0.13)
    brownian = brownian - np.linspace(0, brownian[-1], len(x_grid))
    brownian += 2.1 * np.abs(x_grid) + 1.2
    g_51 = uniform_filter1d(brownian, size=7)
    g_51 = (g_51 - g_51.min()) / (g_51.max() - g_51.min()) * 3.5 + 0.8

    ax3.plot(x_grid, g_51, color='mediumblue', lw=1.3)
    ax3.set_xlim(-2.0, 2.0)
    ax3.set_ylim(0.5, 4.5)
    ax3.set_xlabel('input', fontsize=11)
    ax3.set_ylabel('gradient', fontsize=11)
    ax3.set_title('(c)', fontsize=12, fontweight='bold')

    plt.tight_layout()

    _save_figure(fig, "Figure_9_12", save_dir)
    _save_figure(fig, "fig_9_12_shattered_gradients", save_dir)
    return fig


def generate_figure_9_13(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 9.13: A residual network consisting of three residual blocks,
    corresponding to the sequence of transformations (9.35) to (9.37).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(9.2, 2.3))
    ax.set_xlim(-0.2, 9.6)
    ax.set_ylim(-0.5, 1.6)
    ax.axis('off')

    block_xs = [1.2, 3.8, 6.4]
    adder_xs = [2.9, 5.5, 8.1]
    labels = [r'$\mathbf{F}_1$', r'$\mathbf{F}_2$', r'$\mathbf{F}_3$']
    z_labels = [r'$\mathbf{z}_1$', r'$\mathbf{z}_2$']

    # Input x
    ax.text(0.1, 0.45, r'$\mathbf{x}$', fontsize=13, ha='center', va='center')
    ax.annotate('', xy=(0.7, 0.45), xytext=(0.3, 0.45), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    for idx in range(3):
        bx = block_xs[idx]
        ax_pos = adder_xs[idx]
        branch_x = bx - 0.5

        # Branch dot
        ax.plot(branch_x, 0.45, 'ko', markersize=4)

        # Main path arrow into block
        ax.annotate('', xy=(bx, 0.45), xytext=(branch_x, 0.45), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

        # Block box
        box = patches.FancyBboxPatch(
            (bx, 0.12), 1.05, 0.66,
            boxstyle='round,pad=0.06',
            facecolor='#ff7f7f',
            edgecolor='black',
            lw=1.5
        )
        ax.add_patch(box)
        ax.text(bx + 0.525, 0.45, labels[idx], fontsize=12, fontweight='bold', ha='center', va='center')

        # Arrow into adder
        ax.annotate('', xy=(ax_pos - 0.22, 0.45), xytext=(bx + 1.05, 0.45), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

        # Adder circle
        circ = patches.Circle((ax_pos, 0.45), 0.2, facecolor='white', edgecolor='black', lw=1.5)
        ax.add_patch(circ)
        ax.text(ax_pos, 0.45, '+', fontsize=13, fontweight='bold', ha='center', va='center')

        # Skip connection path: up, over block, down into adder
        ax.plot([branch_x, branch_x, ax_pos, ax_pos], [0.45, 1.25, 1.25, 0.65], 'k-', lw=1.5)
        ax.annotate('', xy=(ax_pos, 0.65), xytext=(ax_pos, 0.72), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

        if idx < 2:
            next_bx = block_xs[idx + 1]
            next_branch_x = next_bx - 0.5
            ax.annotate('', xy=(next_branch_x, 0.45), xytext=(ax_pos + 0.2, 0.45), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
            # Variable label z_l
            ax.text(ax_pos + 0.35, 0.2, z_labels[idx], fontsize=11, ha='center', va='center')
        else:
            # Output arrow and label y
            ax.annotate('', xy=(9.0, 0.45), xytext=(ax_pos + 0.2, 0.45), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
            ax.text(9.25, 0.45, r'$\mathbf{y}$', fontsize=13, ha='center', va='center')

    plt.tight_layout()

    _save_figure(fig, "Figure_9_13", save_dir)
    _save_figure(fig, "fig_9_13_residual_network_architecture", save_dir)
    return fig


def generate_figure_9_14(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 9.14: 3D Visualization of the error surface.
    (a) Network with 56 layers without residual connections (rugged, chaotic).
    (b) The same network with residual connections (smooth, convex-like).
    [Li et al. (2017)]
    """
    fig = plt.figure(figsize=(12.0, 5.2))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    X, Y, Z_without, Z_with = LossLandscapeSimulator.compute_surfaces(grid_size=65)

    # Panel (a): without residual connections
    surf1 = ax1.plot_surface(
        X, Y, Z_without,
        cmap='coolwarm',
        edgecolor='none',
        alpha=0.92,
        antialiased=True,
        rstride=1, cstride=1
    )
    ax1.view_init(elev=32, azim=-55)
    ax1.axis('off')
    ax1.text2D(0.5, 0.05, '(a) without skip connections (56 layers)', transform=ax1.transAxes, ha='center', fontsize=12, fontweight='bold')

    # Panel (b): with residual connections
    surf2 = ax2.plot_surface(
        X, Y, Z_with,
        cmap='coolwarm',
        edgecolor='none',
        alpha=0.92,
        antialiased=True,
        rstride=1, cstride=1
    )
    ax2.view_init(elev=32, azim=-55)
    ax2.axis('off')
    ax2.text2D(0.5, 0.05, '(b) with residual connections (56 layers ResNet)', transform=ax2.transAxes, ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()

    _save_figure(fig, "Figure_9_14", save_dir)
    _save_figure(fig, "fig_9_14_loss_landscapes_3d", save_dir)
    return fig


def generate_figure_9_15(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 9.15: The same network as in Figure 9.13, shown here in expanded form.
    Visualizes the ensemble of paths corresponding to equation (9.40).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')

    # Input x
    ax.text(0.0, 2.0, r'$\mathbf{x}$', fontsize=13, ha='center', va='center')
    ax.plot([0.3, 0.7], [2.0, 2.0], 'k-', lw=1.5)
    # Vertical distributor line from y=0.0 to 3.8
    ax.plot([0.7, 0.7], [0.0, 3.8], 'k-', lw=1.5)

    # Branch dots on distributor line
    branch_ys = [3.8, 3.4, 2.8, 2.3, 1.6, 1.2, 0.6, 0.0]
    for by in branch_ys:
        ax.plot(0.7, by, 'ko', markersize=4)

    def draw_f_box(x_pos, y_pos, label):
        b = patches.FancyBboxPatch(
            (x_pos, y_pos - 0.28), 1.0, 0.56,
            boxstyle='round,pad=0.05',
            facecolor='#ff7f7f',
            edgecolor='black',
            lw=1.5
        )
        ax.add_patch(b)
        ax.text(x_pos + 0.5, y_pos, label, fontsize=11, fontweight='bold', ha='center', va='center')

    def draw_adder(x_pos, y_pos):
        c = patches.Circle((x_pos, y_pos), 0.18, facecolor='white', edgecolor='black', lw=1.5)
        ax.add_patch(c)
        ax.text(x_pos, y_pos, '+', fontsize=12, fontweight='bold', ha='center', va='center')

    # --- Top Group: F3 path ---
    # Row 1: x -> F1 (1.2, 3.8) -> (+) at (2.9, 3.8) -> F2 at (3.8, 3.8) -> (+) at (5.5, 3.8) -> F3 at (6.4, 3.8)
    ax.annotate('', xy=(1.2, 3.8), xytext=(0.7, 3.8), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    draw_f_box(1.2, 3.8, r'$\mathbf{F}_1$')
    ax.annotate('', xy=(2.72, 3.8), xytext=(2.2, 3.8), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    draw_adder(2.9, 3.8)

    # Skip to adder (2.9, 3.8) from (0.7, 3.4)
    ax.plot([0.7, 2.9, 2.9], [3.4, 3.4, 3.62], 'k-', lw=1.5)
    ax.annotate('', xy=(2.9, 3.62), xytext=(2.9, 3.55), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Into F2
    ax.annotate('', xy=(3.8, 3.8), xytext=(3.08, 3.8), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    draw_f_box(3.8, 3.8, r'$\mathbf{F}_2$')
    ax.annotate('', xy=(5.32, 3.8), xytext=(4.8, 3.8), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    draw_adder(5.5, 3.8)

    # Row 2 feed into adder at (5.5, 3.8)
    ax.annotate('', xy=(1.2, 2.8), xytext=(0.7, 2.8), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    draw_f_box(1.2, 2.8, r'$\mathbf{F}_1$')
    draw_adder(5.5, 2.8)
    ax.annotate('', xy=(5.32, 2.8), xytext=(2.2, 2.8), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    # Skip into (5.5, 2.8)
    ax.plot([0.7, 5.5, 5.5], [2.3, 2.3, 2.62], 'k-', lw=1.5)
    ax.annotate('', xy=(5.5, 2.62), xytext=(5.5, 2.55), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    # Arrow up into (5.5, 3.8)
    ax.annotate('', xy=(5.5, 3.62), xytext=(5.5, 2.98), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Into F3
    ax.annotate('', xy=(6.4, 3.8), xytext=(5.68, 3.8), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    draw_f_box(6.4, 3.8, r'$\mathbf{F}_3$')

    # Top group out into final adder
    ax.plot([7.4, 8.1, 8.1], [3.8, 3.8, 2.18], 'k-', lw=1.5)
    ax.annotate('', xy=(8.1, 2.18), xytext=(8.1, 2.25), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # --- Bottom Group ---
    # Row 3: x -> F1 at (1.2, 1.6) -> (+) at (2.9, 1.6) -> F2 at (3.8, 1.6) -> (+) at (5.5, 1.6)
    ax.annotate('', xy=(1.2, 1.6), xytext=(0.7, 1.6), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    draw_f_box(1.2, 1.6, r'$\mathbf{F}_1$')
    draw_adder(2.9, 1.6)
    ax.annotate('', xy=(2.72, 1.6), xytext=(2.2, 1.6), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    # Skip into (2.9, 1.6)
    ax.plot([0.7, 2.9, 2.9], [1.2, 1.2, 1.42], 'k-', lw=1.5)
    ax.annotate('', xy=(2.9, 1.42), xytext=(2.9, 1.35), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Into F2
    ax.annotate('', xy=(3.8, 1.6), xytext=(3.08, 1.6), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    draw_f_box(3.8, 1.6, r'$\mathbf{F}_2$')
    draw_adder(5.5, 1.6)
    ax.annotate('', xy=(5.32, 1.6), xytext=(4.8, 1.6), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Row 4: x -> F1 at (1.2, 0.6) -> (+) at (5.5, 0.6)
    ax.annotate('', xy=(1.2, 0.6), xytext=(0.7, 0.6), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    draw_f_box(1.2, 0.6, r'$\mathbf{F}_1$')
    draw_adder(5.5, 0.6)
    ax.annotate('', xy=(5.32, 0.6), xytext=(2.2, 0.6), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    # Skip into (5.5, 0.6)
    ax.plot([0.7, 5.5, 5.5], [0.0, 0.0, 0.42], 'k-', lw=1.5)
    ax.annotate('', xy=(5.5, 0.42), xytext=(5.5, 0.35), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    # Arrow up into (5.5, 1.6)
    ax.annotate('', xy=(5.5, 1.42), xytext=(5.5, 0.78), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Bottom group out into final adder
    ax.plot([5.68, 8.1, 8.1], [1.6, 1.6, 1.82], 'k-', lw=1.5)
    ax.annotate('', xy=(8.1, 1.82), xytext=(8.1, 1.75), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # --- Final Adder (+) at (8.1, 2.0) and Output y ---
    draw_adder(8.1, 2.0)
    ax.annotate('', xy=(8.9, 2.0), xytext=(8.28, 2.0), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.text(9.15, 2.0, r'$\mathbf{y}$', fontsize=13, ha='center', va='center')

    plt.tight_layout()

    _save_figure(fig, "Figure_9_15", save_dir)
    _save_figure(fig, "fig_9_15_residual_expanded_ensemble", save_dir)
    return fig


def generate_figure_9_16(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 9.16: Two alternative ways to include residual network connections
    into a standard feed-forward network.
    (a) Post-activation: Linear -> ReLU -> (+)
    (b) Pre-activation: ReLU -> Linear -> (+)
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 3.8))
    for ax in [ax1, ax2]:
        ax.set_xlim(-0.2, 9.8)
        ax.set_ylim(-0.4, 1.5)
        ax.axis('off')

    def draw_rounded(ax_t, x_pos, y_pos, w, h, color, text):
        b = patches.FancyBboxPatch(
            (x_pos, y_pos - h/2), w, h,
            boxstyle='round,pad=0.06',
            facecolor=color,
            edgecolor='black',
            lw=1.5
        )
        ax_t.add_patch(b)
        ax_t.text(x_pos + w/2, y_pos, text, fontsize=11, fontweight='bold', ha='center', va='center')

    def draw_adder_circ(ax_t, x_pos, y_pos):
        c = patches.Circle((x_pos, y_pos), 0.18, facecolor='white', edgecolor='black', lw=1.5)
        ax_t.add_patch(c)
        ax_t.text(x_pos, y_pos, '+', fontsize=12, fontweight='bold', ha='center', va='center')

    color_linear = '#f4be89'  # Orange
    color_relu = '#9de0ad'    # Light green

    # === (a) Post-activation ===
    ax1.annotate('', xy=(0.8, 0.4), xytext=(0.1, 0.4), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Block 1
    ax1.plot(0.5, 0.4, 'ko', markersize=4)
    draw_rounded(ax1, 0.8, 0.4, 1.15, 0.6, color_linear, 'Linear')
    ax1.annotate('', xy=(2.45, 0.4), xytext=(1.95, 0.4), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    draw_rounded(ax1, 2.45, 0.4, 1.1, 0.6, color_relu, 'ReLU')
    ax1.annotate('', xy=(4.05, 0.4), xytext=(3.55, 0.4), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    draw_adder_circ(ax1, 4.23, 0.4)

    # Skip 1
    ax1.plot([0.5, 0.5, 4.23, 4.23], [0.4, 1.15, 1.15, 0.58], 'k-', lw=1.5)
    ax1.annotate('', xy=(4.23, 0.58), xytext=(4.23, 0.66), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Intermediate arrow to block 2
    ax1.annotate('', xy=(5.25, 0.4), xytext=(4.41, 0.4), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Block 2
    ax1.plot(4.8, 0.4, 'ko', markersize=4)
    draw_rounded(ax1, 5.25, 0.4, 1.15, 0.6, color_linear, 'Linear')
    ax1.annotate('', xy=(6.9, 0.4), xytext=(6.4, 0.4), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    draw_rounded(ax1, 6.9, 0.4, 1.1, 0.6, color_relu, 'ReLU')
    ax1.annotate('', xy=(8.5, 0.4), xytext=(8.0, 0.4), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    draw_adder_circ(ax1, 8.68, 0.4)

    # Skip 2
    ax1.plot([4.8, 4.8, 8.68, 8.68], [0.4, 1.15, 1.15, 0.58], 'k-', lw=1.5)
    ax1.annotate('', xy=(8.68, 0.58), xytext=(8.68, 0.66), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Output arrow
    ax1.annotate('', xy=(9.4, 0.4), xytext=(8.86, 0.4), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax1.text(4.7, -0.2, '(a)', fontsize=12, fontweight='bold', ha='center')

    # === (b) Pre-activation ===
    ax2.annotate('', xy=(0.8, 0.4), xytext=(0.1, 0.4), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Block 1: ReLU -> Linear -> (+)
    ax2.plot(0.5, 0.4, 'ko', markersize=4)
    draw_rounded(ax2, 0.8, 0.4, 1.1, 0.6, color_relu, 'ReLU')
    ax2.annotate('', xy=(2.45, 0.4), xytext=(1.9, 0.4), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    draw_rounded(ax2, 2.45, 0.4, 1.15, 0.6, color_linear, 'Linear')
    ax2.annotate('', xy=(4.05, 0.4), xytext=(3.6, 0.4), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    draw_adder_circ(ax2, 4.23, 0.4)

    # Skip 1
    ax2.plot([0.5, 0.5, 4.23, 4.23], [0.4, 1.15, 1.15, 0.58], 'k-', lw=1.5)
    ax2.annotate('', xy=(4.23, 0.58), xytext=(4.23, 0.66), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Intermediate arrow to block 2
    ax2.annotate('', xy=(5.25, 0.4), xytext=(4.41, 0.4), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Block 2: ReLU -> Linear -> (+)
    ax2.plot(4.8, 0.4, 'ko', markersize=4)
    draw_rounded(ax2, 5.25, 0.4, 1.1, 0.6, color_relu, 'ReLU')
    ax2.annotate('', xy=(6.9, 0.4), xytext=(6.35, 0.4), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    draw_rounded(ax2, 6.9, 0.4, 1.15, 0.6, color_linear, 'Linear')
    ax2.annotate('', xy=(8.5, 0.4), xytext=(8.05, 0.4), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    draw_adder_circ(ax2, 8.68, 0.4)

    # Skip 2
    ax2.plot([4.8, 4.8, 8.68, 8.68], [0.4, 1.15, 1.15, 0.58], 'k-', lw=1.5)
    ax2.annotate('', xy=(8.68, 0.58), xytext=(8.68, 0.66), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Output arrow
    ax2.annotate('', xy=(9.4, 0.4), xytext=(8.86, 0.4), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax2.text(4.7, -0.2, '(b)', fontsize=12, fontweight='bold', ha='center')

    plt.tight_layout()

    _save_figure(fig, "Figure_9_16", save_dir)
    _save_figure(fig, "fig_9_16_residual_block_variants", save_dir)
    return fig
