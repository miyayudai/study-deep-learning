"""
common/computer_vision.py
=========================
Section 10.1: Computer Vision & Section 10.1.1: Image Data
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements:
1. ImageData: Multi-channel image representations, quantization, color channel decompositions,
   and 2D spatial autocorrelation analysis demonstrating the statistical structure of natural images.
2. PixelPermutation: Demonstration of permutation destruction of spatial structure and why
   permutation-invariant feedforward MLPs fail to capture visual priors.
3. ComputerVisionTaxonomy: Specification and characteristics of the 10 canonical vision tasks
   outlined in Section 10.1.
4. High-resolution pedagogical figure generators saved to 10/result/ and result/.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


class ImageData:
    """
    Representation and statistical analysis of 2D / 3D image data (Section 10.1.1).
    """

    @staticmethod
    def create_synthetic_natural_scene(
        height: int = 64,
        width: int = 64,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generate a synthetic natural-like image with smooth spatial gradients and structural edges.
        Returns:
            rgb_image: (height, width, 3) float array in [0.0, 1.0].
        """
        rng = np.random.RandomState(seed)
        y = np.linspace(-1, 1, height)[:, np.newaxis]
        x = np.linspace(-1, 1, width)[np.newaxis, :]

        # Sky gradient (blue dominance)
        sky_r = np.clip(0.3 + 0.2 * y, 0, 1)
        sky_g = np.clip(0.5 + 0.3 * y, 0, 1)
        sky_b = np.clip(0.85 + 0.15 * y, 0, 1)

        # Ground / hills (green-brown dominance)
        hill1 = y > (0.15 * np.sin(3 * x) - 0.1)
        hill2 = y > (0.25 * np.cos(2 * x) + 0.2)

        r = np.where(hill1, 0.35 + 0.1 * np.sin(5 * x), sky_r)
        g = np.where(hill1, 0.65 + 0.1 * np.cos(5 * y), sky_g)
        b = np.where(hill1, 0.25 + 0.05 * y, sky_b)

        r = np.where(hill2, 0.25 + 0.05 * x, r)
        g = np.where(hill2, 0.45 + 0.1 * y, g)
        b = np.where(hill2, 0.15, b)

        # Add smooth local textures
        noise = rng.randn(height, width, 3) * 0.02
        img = np.stack([r, g, b], axis=-1) + noise
        return np.clip(img, 0.0, 1.0)

    @staticmethod
    def to_grayscale(rgb: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to luminance grayscale via ITU-R BT.601 standard:
        Y = 0.299 R + 0.587 G + 0.114 B.
        """
        weights = np.array([0.299, 0.587, 0.114])
        return np.tensordot(rgb, weights, axes=([-1], [0]))

    @staticmethod
    def compute_spatial_autocorrelation(
        gray_image: np.ndarray,
        max_lag: int = 15,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the 1D horizontal spatial autocorrelation function of an image:
        C(Delta x) = E[(I(x, y) - mu)(I(x + Delta x, y) - mu)] / sigma^2.
        """
        H, W = gray_image.shape
        mu = np.mean(gray_image)
        var = np.var(gray_image)
        if var == 0:
            return np.arange(max_lag + 1), np.ones(max_lag + 1)

        centered = gray_image - mu
        lags = np.arange(max_lag + 1)
        autocorr = np.zeros(max_lag + 1)

        for lag in lags:
            if lag == 0:
                autocorr[lag] = 1.0
            else:
                prod = centered[:, :-lag] * centered[:, lag:]
                autocorr[lag] = np.mean(prod) / var

        return lags, autocorr


class PixelPermutation:
    """
    Demonstrates that randomly permuting pixels preserves the 1D marginal histogram
    while completely destroying 2D spatial correlations and visual semantics (Section 10.1.1).
    """

    @staticmethod
    def permute_image(
        image: np.ndarray,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a fixed random permutation to all pixel coordinates.
        Returns:
            permuted_image: array of same shape with permuted pixel locations.
            perm_indices: 1D permutation array.
        """
        rng = np.random.RandomState(seed)
        shape = image.shape
        H, W = shape[0], shape[1]
        n_pixels = H * W
        perm = rng.permutation(n_pixels)

        if image.ndim == 2:
            flat = image.ravel()
            perm_flat = flat[perm]
            return perm_flat.reshape(H, W), perm
        elif image.ndim == 3:
            C = shape[2]
            flat = image.reshape(n_pixels, C)
            perm_flat = flat[perm]
            return perm_flat.reshape(H, W, C), perm
        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}")


def generate_figure_image_representation(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Pedagogical figure: Multi-channel image tensor representation (H x W x C),
    RGB color channel decomposition, and pixel intensity grid.
    """
    setup_style()
    img_rgb = ImageData.create_synthetic_natural_scene(height=32, width=32, seed=42)
    img_gray = ImageData.to_grayscale(img_rgb)

    fig = plt.figure(figsize=(11.5, 4.2))

    # 1. RGB Image
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(img_rgb)
    ax1.set_title("RGB Image $(H \\times W \\times 3)$", fontsize=11, fontweight='bold')
    ax1.axis('off')

    # 2. Red Channel
    ax2 = fig.add_subplot(1, 4, 2)
    im_r = ax2.imshow(img_rgb[:, :, 0], cmap='Reds', vmin=0, vmax=1)
    ax2.set_title("Red Channel $(C=0)$", fontsize=11, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im_r, ax=ax2, fraction=0.046, pad=0.04)

    # 3. Green Channel
    ax3 = fig.add_subplot(1, 4, 3)
    im_g = ax3.imshow(img_rgb[:, :, 1], cmap='Greens', vmin=0, vmax=1)
    ax3.set_title("Green Channel $(C=1)$", fontsize=11, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im_g, ax=ax3, fraction=0.046, pad=0.04)

    # 4. Blue Channel
    ax4 = fig.add_subplot(1, 4, 4)
    im_b = ax4.imshow(img_rgb[:, :, 2], cmap='Blues', vmin=0, vmax=1)
    ax4.set_title("Blue Channel $(C=2)$", fontsize=11, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im_b, ax=ax4, fraction=0.046, pad=0.04)

    plt.tight_layout()
    _save_figure(fig, "fig_10_image_data_representation", save_dir)
    return fig


def generate_figure_spatial_correlation(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Pedagogical figure: Natural scene vs Permuted pixels vs White noise,
    and their corresponding spatial autocorrelation decay curves.
    """
    setup_style()
    H, W = 48, 48
    img_rgb = ImageData.create_synthetic_natural_scene(height=H, width=W, seed=42)
    img_gray = ImageData.to_grayscale(img_rgb)

    # 1. Permuted image
    img_perm_gray, _ = PixelPermutation.permute_image(img_gray, seed=42)

    # 2. Independent white noise
    rng = np.random.RandomState(42)
    img_noise = rng.uniform(0.0, 1.0, (H, W))

    # Compute autocorrelations
    max_lag = 16
    lags, ac_natural = ImageData.compute_spatial_autocorrelation(img_gray, max_lag=max_lag)
    _, ac_perm = ImageData.compute_spatial_autocorrelation(img_perm_gray, max_lag=max_lag)
    _, ac_noise = ImageData.compute_spatial_autocorrelation(img_noise, max_lag=max_lag)

    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.6))

    # Panel 1: Natural image
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title("(a) Natural Scene", fontsize=10, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Permuted image
    axes[1].imshow(img_perm_gray, cmap='gray')
    axes[1].set_title("(b) Permuted Pixels", fontsize=10, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: White noise
    axes[2].imshow(img_noise, cmap='gray')
    axes[2].set_title("(c) White Noise", fontsize=10, fontweight='bold')
    axes[2].axis('off')

    # Panel 4: Autocorrelation decay
    axes[3].plot(lags, ac_natural, 'o-', color='#1f77b4', lw=2.0, label='Natural Scene', markersize=4)
    axes[3].plot(lags, ac_perm, 's--', color='#d62728', lw=1.8, label='Permuted Image', markersize=4)
    axes[3].plot(lags, ac_noise, '^:', color='#7f7f7f', lw=1.5, label='White Noise', markersize=4)
    axes[3].set_xlim(0, max_lag)
    axes[3].set_ylim(-0.25, 1.05)
    axes[3].set_xlabel('Spatial Lag $\\Delta x$ (pixels)', fontsize=10)
    axes[3].set_ylabel('Autocorrelation $C(\\Delta x)$', fontsize=10)
    axes[3].set_title('(d) Spatial Autocorrelation', fontsize=10, fontweight='bold')
    axes[3].legend(frameon=True, fontsize=8.5, loc='upper right')
    axes[3].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    _save_figure(fig, "fig_10_spatial_correlation", save_dir)
    return fig


def generate_figure_cv_taxonomy(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Pedagogical figure: Taxonomy of the 10 Computer Vision Tasks outlined in Section 10.1.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis('off')

    tasks = [
        ("1. Classification", "Assign label to entire image (e.g. benign vs malignant)", "#d0e1fd"),
        ("2. Object Detection", "Bounding boxes + category labels for instances", "#d0e1fd"),
        ("3. Segmentation", "Pixel-wise classification (semantic / instance / panoptic)", "#d0e1fd"),
        ("4. Captioning", "Generate natural language description from image", "#d5f5e3"),
        ("5. Image Synthesis", "Generate photorealistic images from noise or text", "#fef9e7"),
        ("6. Inpainting", "Replace masked missing regions with coherent pixels", "#fef9e7"),
        ("7. Style Transfer", "Render content in artistic style of reference", "#fef9e7"),
        ("8. Super-Resolution", "Upscale low-resolution image with fine details", "#fef9e7"),
        ("9. Depth Prediction", "Estimate metric distance / depth map from 2D views", "#fadbd8"),
        ("10. Scene Reconstruction", "Build 3D geometry / NeRF / Gaussian Splats from 2D", "#fadbd8"),
    ]

    ax.text(5.5, 5.6, "The 10 Canonical Tasks of Computer Vision (Bishop & Bishop 2024, Section 10.1)",
            fontsize=13, fontweight='bold', ha='center', va='center')

    # Draw 2 columns of 5 boxes
    col_xs = [0.4, 5.7]
    box_w = 4.9
    box_h = 0.82
    y_starts = [4.3, 3.3, 2.3, 1.3, 0.3]

    for idx, (title, desc, color) in enumerate(tasks):
        col = idx // 5
        row = idx % 5
        x = col_xs[col]
        y = y_starts[row]

        box = patches.FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle='round,pad=0.06',
            facecolor=color,
            edgecolor='#2c3e50',
            lw=1.3,
        )
        ax.add_patch(box)
        ax.text(x + 0.2, y + 0.52, title, fontsize=10.5, fontweight='bold', va='center')
        ax.text(x + 0.2, y + 0.22, desc, fontsize=9.0, va='center', color='#34495e')

    plt.tight_layout()
    _save_figure(fig, "fig_10_computer_vision_taxonomy", save_dir)
    return fig
