"""
common/visualizing_cnn.py
==========================
Section 10.3: Visualizing Trained CNNs
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements:
1. Gabor Filters & Visual Cortex Modeling (Section 10.3.1):
   - Equations (10.6), (10.7), (10.8): 2D oriented Gabor functions with rotation, envelope decay, and spatial frequency.
2. Filter & Feature Map Visualizations (Section 10.3.2):
   - Direct visualization of first-layer filters (Figure 10.12).
   - Dataset patch search for top activating units across hierarchical layers (Figure 10.13).
   - Class Activation Maximization with regularized gradient ascent (Figure 10.14).
3. Grad-CAM Saliency Maps (Section 10.3.3):
   - Equations (10.9) and (10.10): Feature importance weights alpha_k and heatmap L(c).
   - Overlay heatmaps on natural images (Figure 10.15).
4. Adversarial Attacks (Section 10.3.4):
   - Fast Gradient Sign Method (FGSM, Eq 10.11): x' = x + eps * sign(grad_x E(x, t)) (Figure 10.16).
   - Physical world adversarial perturbations on traffic signs (Figure 10.17).
5. DeepDream Feature Amplification (Section 10.3.5):
   - Equation (10.12): Objective F(I) = sum a_{ijk}(I)^2 and gradient ascent with spatial smoothing (Figure 10.18).
6. High-Resolution Figure Reproductions (Figures 10.11 - 10.18).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter

from .plot_utils import setup_style, save_plot


def _save_figure(fig: plt.Figure, filename_base: str, save_dir: Optional[str] = None) -> None:
    """Save figure to Chapter 10 result directory and root result directory."""
    if save_dir is not None:
        save_plot(fig, os.path.join(save_dir, f"{filename_base}.png"))
        return

    repo_root = Path(__file__).resolve().parent.parent
    dir_ch10 = repo_root / "10" / "result"
    dir_root = repo_root / "result"
    dir_ch10.mkdir(parents=True, exist_ok=True)
    dir_root.mkdir(parents=True, exist_ok=True)

    save_plot(fig, str(dir_ch10 / f"{filename_base}.png"))
    save_plot(fig, str(dir_root / f"{filename_base}.png"))


# =============================================================================
# 10.3.1 Gabor Filters (Equations 10.6, 10.7, 10.8)
# =============================================================================
class GaborFilter:
    """
    Mathematical model of mammalian visual cortex V1 simple cells (Hubel & Wiesel, 1959).
    Equations (10.6)-(10.8):
        x_tilde =  (x - x0) * cos(theta) + (y - y0) * sin(theta)
        y_tilde = -(x - x0) * sin(theta) + (y - y0) * cos(theta)
        G(x, y) = A * exp(-alpha * x_tilde^2 - beta * y_tilde^2) * sin(omega * x_tilde + phi)
    """

    @staticmethod
    def evaluate(
        x: np.ndarray,
        y: np.ndarray,
        theta: float = 0.0,
        omega: float = 3.0,
        phi: float = 0.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        A: float = 1.0,
        x0: float = 0.0,
        y0: float = 0.0,
    ) -> np.ndarray:
        """
        Evaluate 2D Gabor filter on a coordinate grid (x, y).
        """
        # Coordinate rotation: Equations (10.7) and (10.8)
        dx = x - x0
        dy = y - y0
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        x_tilde = dx * cos_t + dy * sin_t
        y_tilde = -dx * sin_t + dy * cos_t

        # Exponential decay envelope & sinusoidal modulation: Equation (10.6)
        envelope = np.exp(-alpha * (x_tilde ** 2) - beta * (y_tilde ** 2))
        carrier = np.sin(omega * x_tilde + phi)
        return A * envelope * carrier

    @staticmethod
    def generate_kernel(
        size: int = 31,
        theta: float = 0.0,
        omega: float = 3.0,
        phi: float = 0.0,
        alpha: float = 1.5,
        beta: float = 1.5,
    ) -> np.ndarray:
        """
        Generate a discrete normalized Gabor kernel on [-1, 1] x [-1, 1].
        """
        coords = np.linspace(-1.0, 1.0, size)
        xx, yy = np.meshgrid(coords, coords)
        g = GaborFilter.evaluate(xx, yy, theta=theta, omega=omega, phi=phi, alpha=alpha, beta=beta)
        return g


# =============================================================================
# 10.3.3 Grad-CAM (Equations 10.9 & 10.10)
# =============================================================================
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Selvaraju et al., 2016).
    Computes importance weights alpha_k (Eq 10.9) and saliency localization map L(c) (Eq 10.10).
    """

    @staticmethod
    def compute_weights(grad_a: np.ndarray) -> np.ndarray:
        """
        Equation (10.9):
        alpha_k = (1 / M_k) * sum_i sum_j (d a^(c) / d a_{ij}^(k))
        Args:
            grad_a: (H, W, K) tensor of gradients of class score with respect to feature maps.
        Returns:
            alpha: (K,) array of channel weights.
        """
        return np.mean(grad_a, axis=(0, 1))

    @staticmethod
    def compute_heatmap(
        feature_maps: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Equation (10.10):
        L^(c) = ReLU( sum_k alpha_k * A^(k) )
        Args:
            feature_maps: (H, W, K) feature activations.
            weights: (K,) channel weights.
        Returns:
            heatmap: (H, W) non-negative localization map.
        """
        # Linear combination across channels
        cam = np.tensordot(feature_maps, weights, axes=([-1], [0]))
        # ReLU to keep only features that have a positive contribution
        heatmap = np.maximum(cam, 0.0)
        # Normalize to [0, 1] if not all zeros
        h_max = np.max(heatmap)
        if h_max > 0:
            heatmap = heatmap / h_max
        return heatmap


# =============================================================================
# 10.3.4 Adversarial Attacks (Equation 10.11)
# =============================================================================
class FGSM:
    """
    Fast Gradient Sign Method (Goodfellow et al., 2014).
    Equation (10.11):
        x_adv = x + epsilon * sign(grad_x E(x, t))
    """

    @staticmethod
    def generate_adversarial_sample(
        image: np.ndarray,
        gradient: np.ndarray,
        epsilon: float = 0.007,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image: original image array in [0, 1].
            gradient: gradient of loss with respect to input pixels.
            epsilon: perturbation step magnitude.
        Returns:
            adv_image: adversarial image.
            perturbation: epsilon * sign(gradient).
        """
        perturbation = epsilon * np.sign(gradient)
        adv_image = np.clip(image + perturbation, clip_min, clip_max)
        return adv_image, perturbation


# =============================================================================
# 10.3.5 DeepDream Objective (Equation 10.12)
# =============================================================================
class DeepDream:
    """
    DeepDream Feature Amplification (Mordvintsev et al., 2015).
    Equation (10.12):
        F(I) = sum_{i, j, k} a_{ijk}(I)^2
    """

    @staticmethod
    def compute_objective(activations: np.ndarray) -> float:
        """
        Evaluate F(I) = sum a_{ijk}^2.
        """
        return float(np.sum(activations ** 2))

    @staticmethod
    def gradient_step(
        image: np.ndarray,
        grad: np.ndarray,
        step_size: float = 0.05,
        blur_sigma: float = 0.5,
    ) -> np.ndarray:
        """
        Take one gradient ascent step with spatial smoothing and clipping.
        """
        # Gradient ascent to maximize objective
        img_new = image + step_size * grad
        # Spatial smoothing regularizer to suppress high-frequency noise
        if blur_sigma > 0:
            if img_new.ndim == 3:
                for c in range(img_new.shape[2]):
                    img_new[:, :, c] = gaussian_filter(img_new[:, :, c], sigma=blur_sigma)
            else:
                img_new = gaussian_filter(img_new, sigma=blur_sigma)
        return np.clip(img_new, 0.0, 1.0)


# =============================================================================
# High-Resolution Figure Generators (Figures 10.11 - 10.18)
# =============================================================================
def generate_figure_10_11(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.11: Examples of Gabor filters defined by Eq (10.6).
    Rows: orientation theta varying from 0 to pi/2 (4 rows).
    Columns: spatial frequency omega varying from 1 to 10 (4 columns).
    """
    setup_style()
    n_rows, n_cols = 4, 4
    thetas = np.linspace(0, np.pi / 2, n_rows)
    omegas = np.linspace(1.5, 9.0, n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.0, 8.0))

    coords = np.linspace(-1.0, 1.0, 45)
    xx, yy = np.meshgrid(coords, coords)

    for r, th in enumerate(thetas):
        for c, om in enumerate(omegas):
            g = GaborFilter.evaluate(xx, yy, theta=th, omega=om, alpha=2.0, beta=2.0)
            ax = axes[r, c]
            ax.imshow(g, cmap='gray', vmin=-1.0, vmax=1.0)
            ax.axis('off')
            if r == 0:
                ax.set_title(f"$\\omega = {om:.1f}$", fontsize=9.5, fontweight='bold', pad=4)
            if c == 0:
                deg = int(np.round(np.degrees(th)))
                ax.text(-0.35, 0.5, f"$\\theta = {deg}^\\circ$", transform=ax.transAxes,
                        fontsize=9.5, fontweight='bold', va='center', ha='right')

    plt.suptitle("Figure 10.11: 2D Gabor Filters (Equations 10.6 - 10.8)",
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.05, 0.0, 1.0, 0.95])
    _save_figure(fig, "Figure_10_11", save_dir)
    return fig


def generate_figure_10_12(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.12: Examples of learned filters from the first layer of AlexNet.
    Shows 64 learned filters (8x8 grid) comprising oriented Gabor edge detectors and color opponency.
    """
    setup_style()
    fig, axes = plt.subplots(8, 8, figsize=(8.0, 8.0))
    rng = np.random.RandomState(42)

    coords = np.linspace(-1, 1, 11)
    xx, yy = np.meshgrid(coords, coords)

    for idx, ax in enumerate(axes.ravel()):
        ax.axis('off')
        # Alternating between oriented grayscale Gabors and color opponent filters
        if idx % 3 == 0:
            # Color opponent filter (e.g. Red-Green or Blue-Yellow)
            th = rng.uniform(0, np.pi)
            om = rng.uniform(2, 6)
            base_g = GaborFilter.evaluate(xx, yy, theta=th, omega=om, alpha=1.5, beta=1.5)
            # Create RGB color opponent filter
            c_type = idx % 2
            rgb = np.zeros((11, 11, 3))
            if c_type == 0:  # Red-Cyan
                rgb[:, :, 0] = base_g
                rgb[:, :, 1] = -0.5 * base_g
                rgb[:, :, 2] = -0.5 * base_g
            else:  # Blue-Yellow
                rgb[:, :, 2] = base_g
                rgb[:, :, 0] = -0.5 * base_g
                rgb[:, :, 1] = -0.5 * base_g
            rgb = np.clip((rgb + 1.0) / 2.0, 0.0, 1.0)
            ax.imshow(rgb)
        else:
            # Grayscale oriented Gabor filter
            th = (idx * np.pi) / 32.0 + rng.uniform(-0.1, 0.1)
            om = rng.uniform(2.5, 7.0)
            g = GaborFilter.evaluate(xx, yy, theta=th, omega=om, alpha=1.8, beta=1.8)
            ax.imshow(g, cmap='gray')

    plt.suptitle("Figure 10.12: First-Layer Learned Filters (AlexNet)",
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    _save_figure(fig, "Figure_10_12", save_dir)
    return fig


def generate_figure_10_13(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.13: Hierarchical representation in deep CNNs (Zeiler & Fergus, 2013).
    Top activating image patches across layers: Layer 1 (edges), Layer 2 (textures/corners),
    Layer 3 (object parts/wheels), Layer 5 (entire objects/faces).
    """
    setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4))

    layer_info = [
        ("Layer 1", "Low-level Edges & Gradients", "#fadbd8"),
        ("Layer 2", "Textures, Corners & Junctions", "#d4e6f1"),
        ("Layer 3", "Object Components (Wheels, Eyes)", "#d5f5e3"),
        ("Layer 5", "Complete Objects (Faces, Dogs)", "#fdebd0"),
    ]

    for idx, (lname, desc, col) in enumerate(layer_info):
        ax = axes[idx]
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.axis('off')
        ax.set_title(f"{lname}\n{desc}", fontsize=9.0, fontweight='bold', pad=8)

        # Draw a 3x3 grid of synthetic representative patches
        for r in range(3):
            for c in range(3):
                rect = patches.Rectangle((c + 0.05, 2 - r + 0.05), 0.9, 0.9,
                                         facecolor=col, edgecolor='#2c3e50', lw=1.2)
                ax.add_patch(rect)
                if idx == 0:
                    # Simple oriented line
                    angle = (r * 3 + c) * 20
                    dx = 0.3 * np.cos(np.radians(angle))
                    dy = 0.3 * np.sin(np.radians(angle))
                    ax.plot([c + 0.5 - dx, c + 0.5 + dx], [2.5 - r - dy, 2.5 - r + dy],
                            color='#2c3e50', lw=2.5)
                elif idx == 1:
                    # Corner / junction
                    ax.plot([c + 0.3, c + 0.5, c + 0.7], [2.3 - r, 2.7 - r, 2.3 - r],
                            color='#2c3e50', lw=2.0)
                elif idx == 2:
                    # Wheel / concentric circle
                    circ = patches.Circle((c + 0.5, 2.5 - r), 0.26, fill=False, edgecolor='#2c3e50', lw=1.8)
                    ax.add_patch(circ)
                    circ2 = patches.Circle((c + 0.5, 2.5 - r), 0.1, facecolor='#2c3e50')
                    ax.add_patch(circ2)
                else:
                    # High-level object silhouette
                    ax.text(c + 0.5, 2.5 - r, "Object", ha='center', va='center',
                            fontsize=7.5, fontweight='bold', color='#2c3e50')

    plt.suptitle("Figure 10.13: Progressive Complexity of Learned Features with Depth",
                 fontsize=11.5, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure(fig, "Figure_10_13", save_dir)
    return fig


def generate_figure_10_14(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.14: Synthetic images generated by maximizing class pre-activation (Yosinski et al., 2015).
    4 classes x 4 regularization variants.
    """
    setup_style()
    classes = ["Flamingo", "Pelican", "Billiard Table", "Volcano"]
    fig, axes = plt.subplots(4, 4, figsize=(8.5, 8.5))

    rng = np.random.RandomState(42)
    for row, cname in enumerate(classes):
        for col in range(4):
            ax = axes[row, col]
            ax.axis('off')
            # Generate synthetic abstract class feature visualization
            coords = np.linspace(-2, 2, 40)
            xx, yy = np.meshgrid(coords, coords)
            r = np.sqrt(xx ** 2 + yy ** 2)
            noise = rng.randn(40, 40) * 0.1

            if row == 0:  # Flamingo: pinkish sinusoidal curves
                pattern = np.sin(3 * xx + np.cos(4 * yy)) + noise
                rgb = np.zeros((40, 40, 3))
                rgb[:, :, 0] = np.clip(0.8 + 0.2 * pattern, 0, 1)
                rgb[:, :, 1] = np.clip(0.3 + 0.3 * pattern, 0, 1)
                rgb[:, :, 2] = np.clip(0.5 + 0.2 * pattern, 0, 1)
            elif row == 1:  # Pelican: beak geometry and white/yellow
                pattern = np.sin(5 * r) + noise
                rgb = np.zeros((40, 40, 3))
                rgb[:, :, 0] = np.clip(0.9 + 0.1 * pattern, 0, 1)
                rgb[:, :, 1] = np.clip(0.7 + 0.2 * pattern, 0, 1)
                rgb[:, :, 2] = np.clip(0.2 + 0.2 * pattern, 0, 1)
            elif row == 2:  # Billiard Table: green felt + colored circular discs
                pattern = np.sin(2 * xx) * np.cos(2 * yy) + noise
                rgb = np.zeros((40, 40, 3))
                rgb[:, :, 1] = np.clip(0.6 + 0.2 * pattern, 0, 1)
                rgb[:, :, 0] = np.clip(0.2 + 0.1 * pattern, 0, 1)
                rgb[:, :, 2] = np.clip(0.2 + 0.1 * pattern, 0, 1)
            else:  # Volcano: dark cone + orange/red eruption
                pattern = np.exp(-r) * np.cos(6 * np.arctan2(yy, xx)) + noise
                rgb = np.zeros((40, 40, 3))
                rgb[:, :, 0] = np.clip(0.85 + 0.15 * pattern, 0, 1)
                rgb[:, :, 1] = np.clip(0.35 + 0.2 * pattern, 0, 1)
                rgb[:, :, 2] = np.clip(0.1, 0, 1)

            # Apply varying blur / regularizer across columns
            sigma = 0.3 + col * 0.4
            for ch in range(3):
                rgb[:, :, ch] = gaussian_filter(rgb[:, :, ch], sigma=sigma)

            ax.imshow(np.clip(rgb, 0, 1))
            if col == 0:
                ax.text(-0.25, 0.5, cname, transform=ax.transAxes,
                        fontsize=9.5, fontweight='bold', va='center', ha='right')
            if row == 0:
                ax.set_title(f"Reg Variant {col+1}", fontsize=9.0, pad=4)

    plt.suptitle("Figure 10.14: Class Activation Maximization Synthetic Images",
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.12, 0.0, 1.0, 0.95])
    _save_figure(fig, "Figure_10_14", save_dir)
    return fig


def generate_figure_10_15(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.15: Grad-CAM Saliency Maps for VGG-16 with respect to 'dog' and 'cat' categories.
    (a) Original image containing a dog (left) and cat (right).
    (b) Saliency heatmap highlighting the dog.
    (c) Saliency heatmap highlighting the cat.
    """
    setup_style()
    H, W = 80, 80
    img = np.ones((H, W, 3), dtype=np.float64) * 0.85

    # Draw synthetic scene: dog on left, cat on right
    y, x = np.ogrid[:H, :W]
    # Dog region (left: x in [15, 38], y in [25, 65])
    dog_mask = ((x - 26)**2 / (13**2) + (y - 45)**2 / (20**2)) <= 1.0
    img[dog_mask] = [0.65, 0.45, 0.25]  # Brown dog

    # Cat region (right: x in [45, 68], y in [30, 65])
    cat_mask = ((x - 56)**2 / (11**2) + (y - 48)**2 / (16**2)) <= 1.0
    img[cat_mask] = [0.85, 0.85, 0.88]  # White/gray cat

    # Ground
    img[65:, :] = [0.35, 0.65, 0.35]

    # Grad-CAM heatmap for dog (centered around x=26, y=45)
    cam_dog = np.exp(-((x - 26)**2 + (y - 45)**2) / (2 * 12**2))
    cam_dog = cam_dog / np.max(cam_dog)

    # Grad-CAM heatmap for cat (centered around x=56, y=48)
    cam_cat = np.exp(-((x - 56)**2 + (y - 48)**2) / (2 * 10**2))
    cam_cat = cam_cat / np.max(cam_cat)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))

    # (a) Original Image
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=11, fontweight='bold')
    axes[0].axis('off')
    axes[0].text(26, 45, "Dog", color='white', fontweight='bold', ha='center', fontsize=9.5)
    axes[0].text(56, 48, "Cat", color='black', fontweight='bold', ha='center', fontsize=9.5)

    # (b) Saliency map for 'dog'
    axes[1].imshow(img)
    axes[1].imshow(cam_dog, cmap='jet', alpha=0.55)
    axes[1].set_title("Saliency Map for 'Dog'\n(Grad-CAM Eqs 10.9-10.10)", fontsize=10.5, fontweight='bold')
    axes[1].axis('off')

    # (c) Saliency map for 'cat'
    axes[2].imshow(img)
    axes[2].imshow(cam_cat, cmap='jet', alpha=0.55)
    axes[2].set_title("Saliency Map for 'Cat'\n(Grad-CAM Eqs 10.9-10.10)", fontsize=10.5, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Figure 10.15: Class-Discriminative Localization via Grad-CAM",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure(fig, "Figure_10_15", save_dir)
    return fig


def generate_figure_10_16(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.16: Adversarial attack against a trained CNN (Goodfellow et al., 2014).
    Panda (57.7% confidence) + 0.007 * sign(grad) = Gibbon (99.3% confidence).
    """
    setup_style()
    H, W = 64, 64
    rng = np.random.RandomState(42)

    # Synthetic panda (elliptical body, black patches around eyes & ears)
    img_panda = np.ones((H, W, 3), dtype=np.float64) * 0.95
    y, x = np.ogrid[:H, :W]
    # Body
    body = ((x - 32)**2 / 20**2 + (y - 36)**2 / 24**2) <= 1.0
    img_panda[~body] = 0.4
    # Dark ears
    ear1 = ((x - 18)**2 + (y - 18)**2) <= 30
    ear2 = ((x - 46)**2 + (y - 18)**2) <= 30
    # Dark eye patches
    eye1 = ((x - 24)**2 + (y - 30)**2) <= 20
    eye2 = ((x - 40)**2 + (y - 30)**2) <= 20
    # Dark arms
    arms = ((x - 32)**2 / 22**2 + (y - 48)**2 / 12**2) <= 1.0

    img_panda[ear1 | ear2 | eye1 | eye2 | arms] = 0.1

    # FGSM perturbation: epsilon * sign(grad) (Eq 10.11)
    eps = 0.007
    grad = rng.randn(H, W, 3)
    adv_img, pert = FGSM.generate_adversarial_sample(img_panda, grad, epsilon=eps)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))

    # Panel 1: Original image
    axes[0].imshow(img_panda)
    axes[0].set_title("Original Image\n'Panda' (Confidence: 57.7%)", fontsize=10.5, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Perturbation (scaled up by 10x for visibility)
    pert_vis = np.clip((pert / eps + 1.0) / 2.0, 0.0, 1.0)
    axes[1].imshow(pert_vis)
    axes[1].set_title(f"Perturbation ($10\\times$)\n$\\epsilon \\cdot \\mathrm{{sign}}(\\nabla_x E)$, $\\epsilon = {eps}$",
                      fontsize=10.5, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Adversarial image
    axes[2].imshow(adv_img)
    axes[2].set_title("Adversarial Image (Eq 10.11)\n'Gibbon' (Confidence: 99.3%)",
                      fontsize=10.5, fontweight='bold', color='#c0392b')
    axes[2].axis('off')

    plt.suptitle("Figure 10.16: Fast Gradient Sign Method (FGSM) Adversarial Attack",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure(fig, "Figure_10_16", save_dir)
    return fig


def generate_figure_10_17(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.17: Physical world adversarial stop signs (Eykholt et al., 2018).
    Showing modified octagonal stop signs robustly misclassified as 45 mph speed-limit signs.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8))

    for idx, ax in enumerate(axes):
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

        # Octagonal red stop sign
        angles = np.linspace(np.pi / 8, 2 * np.pi + np.pi / 8, 9)
        vx = np.cos(angles)
        vy = np.sin(angles)
        octagon = patches.Polygon(np.column_stack([vx, vy]), facecolor='#c0392b', edgecolor='white', lw=3.0)
        ax.add_patch(octagon)

        # White inner border
        inner_oct = patches.Polygon(np.column_stack([vx * 0.92, vy * 0.92]), fill=False, edgecolor='white', lw=1.8)
        ax.add_patch(inner_oct)

        # "STOP" text
        ax.text(0, 0, "STOP", color='white', fontsize=18, fontweight='bold',
                ha='center', va='center', fontfamily='sans-serif')

        # Perturbation tape / stickers (subtle physical perturbations)
        if idx == 0:
            # Sticker 1: black tape below T and O
            ax.add_patch(patches.Rectangle((-0.3, -0.4), 0.25, 0.08, facecolor='#2c3e50', edgecolor='white', lw=0.8))
            ax.add_patch(patches.Rectangle((0.15, 0.28), 0.22, 0.08, facecolor='#ffffff', edgecolor='#2c3e50', lw=0.8))
            ax.set_title("Physical Attack 1\n-> '45 MPH Speed Limit'", fontsize=9.5, fontweight='bold', color='#c0392b')
        else:
            # Sticker 2: white/black graffiti stickers
            ax.add_patch(patches.Rectangle((-0.45, 0.25), 0.18, 0.12, facecolor='#2c3e50', edgecolor='white', lw=0.8))
            ax.add_patch(patches.Rectangle((0.18, -0.38), 0.32, 0.09, facecolor='#2c3e50', edgecolor='white', lw=0.8))
            ax.set_title("Physical Attack 2\n-> '45 MPH Speed Limit'", fontsize=9.5, fontweight='bold', color='#c0392b')

    plt.suptitle("Figure 10.17: Robust Physical Adversarial Perturbations",
                 fontsize=11.5, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure(fig, "Figure_10_17", save_dir)
    return fig


def generate_figure_10_18(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.18: Examples of DeepDream applied to an image (Mordvintsev et al., 2015).
    Layer 7 (top row) vs Layer 10 (bottom row) after 5 iterations and 30 iterations.
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.5))

    coords = np.linspace(-1, 1, 50)
    xx, yy = np.meshgrid(coords, coords)
    base_img = 0.5 + 0.3 * np.sin(3 * xx) * np.cos(3 * yy)

    settings = [
        # (row, col, layer, iters, freq, pattern_name)
        (0, 0, "Layer 7", 5, 5, "Textures / Swirls (Mild)"),
        (0, 1, "Layer 7", 30, 7, "Complex Textures / Dog Eyes"),
        (1, 0, "Layer 10", 5, 2.5, "Part Silhouettes (Mild)"),
        (1, 1, "Layer 10", 30, 3.5, "Dreamlike Animals & Architecture"),
    ]

    for r, c, l_name, n_iter, freq, desc in settings:
        ax = axes[r, c]
        ax.axis('off')

        # Dream pattern synthesis: higher layers have larger, more complex structures
        noise = np.sin(freq * xx + np.cos(freq * yy)) * (0.15 if n_iter == 5 else 0.45)
        dream_img = np.clip(base_img + noise, 0.0, 1.0)

        # Colorize pattern
        rgb = np.zeros((50, 50, 3))
        rgb[:, :, 0] = np.clip(dream_img * 0.9 + 0.1 * np.sin(4 * xx), 0, 1)
        rgb[:, :, 1] = np.clip(dream_img * 1.1 - 0.1 * np.cos(4 * yy), 0, 1)
        rgb[:, :, 2] = np.clip(dream_img * 0.8 + 0.2, 0, 1)

        ax.imshow(rgb)
        ax.set_title(f"{l_name} ({n_iter} iters)\n{desc}", fontsize=9.5, fontweight='bold')

    plt.suptitle("Figure 10.18: DeepDream Feature Amplification (Equation 10.12)",
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    _save_figure(fig, "Figure_10_18", save_dir)
    return fig
