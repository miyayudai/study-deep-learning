"""
common/image_segmentation.py
============================
Section 10.5: Image Segmentation
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements:
1. Semantic Segmentation Representation & Evaluation (Section 10.5.1):
   - Pixel-wise classification tensor (H x W x C) and mean Intersection-over-Union (mIoU).
2. Unpooling Operations (Section 10.5.2):
   - Nearest-neighbor / average-unpooling (Figure 10.28(a)).
   - Fixed max-unpooling (Figure 10.28(b)).
   - Max-unpooling with switch variables (SegNet, Figure 10.29).
3. Transposed Convolutions / Fractionally Strided Convolutions (Section 10.5.3, Exercise 10.13):
   - 2D transposed convolution kernel expansion (Figure 10.30).
   - Matrix transpose duality between strided down-sampling and up-sampling.
4. U-net Architecture Simulation (Section 10.5.4, Figure 10.31):
   - Symmetrical encoder-decoder structure and skip connections.
5. High-Resolution Figure Reproductions (Figures 10.26 - 10.31).
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


# =============================================================================
# 10.5.2 Unpooling Functions (Figures 10.28 & 10.29)
# =============================================================================
class Unpooling:
    @staticmethod
    def average_unpool2d(x: np.ndarray, scale: int = 2) -> np.ndarray:
        """
        Nearest-neighbor / average-unpooling analogue (Figure 10.28(a)).
        Each input element is replicated into a scale x scale block.
        """
        return np.repeat(np.repeat(x, scale, axis=0), scale, axis=1)

    @staticmethod
    def max_unpool2d_fixed(x: np.ndarray, scale: int = 2) -> np.ndarray:
        """
        Fixed top-left max-unpooling analogue (Figure 10.28(b)).
        Each input element is placed at top-left of scale x scale block; others are 0.
        """
        H, W = x.shape[:2]
        out_shape = (H * scale, W * scale) + x.shape[2:]
        out = np.zeros(out_shape, dtype=x.dtype)
        out[::scale, ::scale] = x
        return out

    @staticmethod
    def max_pool_with_indices(x: np.ndarray, pool_size: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        2D Max-pooling with switch indices recording (Figure 10.29).
        Returns:
            pooled: (H//pool_size, W//pool_size)
            indices: array of flat index (0..pool_size^2-1) within each block.
        """
        H, W = x.shape
        out_h = H // pool_size
        out_w = W // pool_size
        pooled = np.zeros((out_h, out_w), dtype=x.dtype)
        indices = np.zeros((out_h, out_w), dtype=np.int64)

        for i in range(out_h):
            for j in range(out_w):
                patch = x[i * pool_size:(i + 1) * pool_size, j * pool_size:(j + 1) * pool_size]
                flat_idx = np.argmax(patch)
                pooled[i, j] = patch.ravel()[flat_idx]
                indices[i, j] = flat_idx
        return pooled, indices

    @staticmethod
    def max_unpool_with_indices(
        pooled: np.ndarray,
        indices: np.ndarray,
        pool_size: int = 2,
    ) -> np.ndarray:
        """
        Restore values into the exact recorded max locations (Figure 10.29).
        """
        out_h, out_w = pooled.shape[0] * pool_size, pooled.shape[1] * pool_size
        out = np.zeros((out_h, out_w), dtype=pooled.dtype)

        for i in range(pooled.shape[0]):
            for j in range(pooled.shape[1]):
                flat_idx = indices[i, j]
                r_offset = flat_idx // pool_size
                c_offset = flat_idx % pool_size
                out[i * pool_size + r_offset, j * pool_size + c_offset] = pooled[i, j]
        return out


# =============================================================================
# 10.5.3 Transposed Convolution (Figure 10.30 & Exercise 10.13)
# =============================================================================
def conv_transpose2d(
    input_map: np.ndarray,
    kernel: np.ndarray,
    stride: int = 2,
    padding: int = 0,
) -> np.ndarray:
    """
    2D Transposed Convolution / Fractionally Strided Convolution (Figure 10.30).
    Args:
        input_map: (Hin, Win) array.
        kernel: (M, M) array.
        stride: output stride.
        padding: border crop width.
    Returns:
        output_map: (Hout, Wout) array where Hout = (Hin - 1) * stride + M - 2 * padding.
    """
    Hin, Win = input_map.shape
    M = kernel.shape[0]
    Hout_unpadded = (Hin - 1) * stride + M
    Wout_unpadded = (Win - 1) * stride + M
    out = np.zeros((Hout_unpadded, Wout_unpadded), dtype=np.float64)

    for i in range(Hin):
        for j in range(Win):
            val = input_map[i, j]
            out[i * stride:i * stride + M, j * stride:j * stride + M] += val * kernel

    if padding > 0:
        return out[padding:Hout_unpadded - padding, padding:Wout_unpadded - padding]
    return out


# =============================================================================
# High-Resolution Figure Reproductions (Figures 10.26 - 10.31)
# =============================================================================
def generate_figure_10_26(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.26: Example of semantic image segmentation.
    Left: RGB input image (street scene).
    Right: Pixel-wise semantic segmentation map with distinct class colors
    (blue: car, red: pedestrian, orange: traffic light, green: vegetation, gray: road).
    """
    setup_style()
    H, W = 80, 100
    rgb = np.ones((H, W, 3), dtype=np.float64) * 0.9
    seg = np.zeros((H, W, 3), dtype=np.float64)

    # Class palette
    c_sky = [0.85, 0.92, 0.98]
    c_road = [0.5, 0.5, 0.5]
    c_veg = [0.18, 0.8, 0.25]
    c_car = [0.16, 0.5, 0.9]      # Blue
    c_ped = [0.9, 0.2, 0.2]      # Red
    c_light = [0.95, 0.6, 0.1]   # Orange

    # 1. Sky & Road
    rgb[:35, :] = [0.75, 0.88, 0.95]
    seg[:35, :] = c_sky
    rgb[35:, :] = [0.4, 0.42, 0.45]
    seg[35:, :] = c_road

    # 2. Vegetation / trees on side
    y, x = np.ogrid[:H, :W]
    tree = (x < 25) & (y < 45)
    rgb[tree] = [0.2, 0.6, 0.25]
    seg[tree] = c_veg

    # 3. Cars
    car1 = ((x - 55)**2 / 16**2 + (y - 52)**2 / 10**2) <= 1.0
    car2 = ((x - 82)**2 / 12**2 + (y - 48)**2 / 8**2) <= 1.0
    rgb[car1] = [0.2, 0.3, 0.7]
    seg[car1] = c_car
    rgb[car2] = [0.6, 0.2, 0.2]
    seg[car2] = c_car

    # 4. Pedestrian
    ped = (x >= 32) & (x <= 38) & (y >= 40) & (y <= 62)
    rgb[ped] = [0.85, 0.3, 0.2]
    seg[ped] = c_ped

    # 5. Traffic light
    tl = (x >= 42) & (x <= 46) & (y >= 15) & (y <= 28)
    rgb[tl] = [0.9, 0.55, 0.1]
    seg[tl] = c_light

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax1.imshow(rgb)
    ax1.set_title("Input RGB Image", fontsize=11, fontweight='bold')
    ax1.axis('off')

    ax2.imshow(seg)
    ax2.set_title("Semantic Segmentation (Pixel-wise)", fontsize=11, fontweight='bold')
    ax2.axis('off')

    # Legend
    legend_patches = [
        patches.Patch(facecolor=c_car, label='Car (Blue)'),
        patches.Patch(facecolor=c_ped, label='Pedestrian (Red)'),
        patches.Patch(facecolor=c_light, label='Traffic Light (Orange)'),
        patches.Patch(facecolor=c_veg, label='Vegetation (Green)'),
        patches.Patch(facecolor=c_road, label='Road (Gray)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=5, frameon=True, fontsize=8.5)

    plt.suptitle("Figure 10.26: Semantic Image Segmentation", fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.0, 0.08, 1.0, 0.95])
    _save_figure(fig, "Figure_10_26", save_dir)
    return fig


def generate_figure_10_27(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.27: Encoder-Decoder architecture for semantic segmentation.
    High resolution -> Mid resolution -> Low resolution (bottleneck) -> Mid resolution -> High resolution.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.0)
    ax.axis('off')
    ax.set_title("Figure 10.27: Encoder-Decoder Convolutional Architecture for Segmentation",
                 fontsize=11.5, fontweight='bold', pad=12)

    # Blocks: (x_center, width, height, color, label, res_desc)
    blocks = [
        (1.2, 0.8, 3.4, '#fadbd8', "Input\nImage", "High Res"),
        (2.8, 0.9, 2.5, '#f9e79f', "Down 1\n(Conv/Pool)", "Mid Res"),
        (4.4, 1.0, 1.6, '#d4e6f1', "Down 2\n(Conv/Pool)", "Low Res\n(Bottleneck)"),
        (6.0, 1.0, 1.6, '#d4e6f1', "Bottleneck\nLatent", "Low Res"),
        (7.6, 0.9, 2.5, '#d5f5e3', "Up 1\n(TransConv)", "Mid Res"),
        (9.4, 0.8, 3.4, '#e8daef', "Output\nMask", "High Res"),
    ]

    for xc, w, h, col, lbl, res_txt in blocks:
        yb = 2.4 - h / 2.0
        rect = patches.Rectangle((xc - w / 2.0, yb), w, h, facecolor=col, edgecolor='#2c3e50', lw=1.5)
        ax.add_patch(rect)
        ax.text(xc, 2.4, lbl, ha='center', va='center', fontsize=8.5, fontweight='bold')
        ax.text(xc, yb - 0.25, res_txt, ha='center', va='top', fontsize=8.0, color='#34495e')

    # Flow arrows
    for xa in [1.8, 3.4, 5.1, 6.7, 8.2]:
        ax.annotate("", xy=(xa + 0.6, 2.4), xytext=(xa, 2.4),
                    arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=2.0))

    # Encoder / Decoder brackets
    ax.text(3.0, 4.4, "Encoder (Down-sampling)", ha='center', fontsize=10, fontweight='bold', color='#c0392b')
    ax.text(7.6, 4.4, "Decoder (Up-sampling)", ha='center', fontsize=10, fontweight='bold', color='#27ae60')

    plt.tight_layout()
    _save_figure(fig, "Figure_10_27", save_dir)
    return fig


def generate_figure_10_28(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.28: Unpooling operations.
    (a) Analogue of average pooling (replication into 2x2 blocks).
    (b) Analogue of max-pooling (placement at top-left, zeros elsewhere).
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.4))

    for ax in [ax1, ax2]:
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(-0.5, 5.0)
        ax.axis('off')

    # Input 2x2: [[1, 2], [3, 4]]
    input_vals = [[1, 2], [3, 4]]

    # --- (a) Average Unpooling ---
    ax1.set_title("(a) Average Unpooling (Value Replication)", fontsize=10.5, fontweight='bold', pad=10)
    # Draw input 2x2 on left
    for r in range(2):
        for c in range(2):
            rect = patches.Rectangle((0.5 + c, 3.0 - r), 0.9, 0.9, facecolor='#fcf3cf', edgecolor='#b7950b', lw=1.5)
            ax1.add_patch(rect)
            ax1.text(0.95 + c, 3.45 - r, str(input_vals[r][c]), ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.text(1.45, 1.8, "Input ($2 \\times 2$)", ha='center', fontsize=9.5, fontweight='bold')

    ax1.annotate("", xy=(3.6, 3.0), xytext=(2.7, 3.0),
                 arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=2.0))

    # Output 4x4 replicated: each quadrant filled with its input value
    out_a = [[1, 1, 2, 2],
             [1, 1, 2, 2],
             [3, 3, 4, 4],
             [3, 3, 4, 4]]
    colors_quad = ['#fadbd8', '#d4e6f1', '#d5f5e3', '#fdebd0']

    for r in range(4):
        for c in range(4):
            q = (0 if (r < 2 and c < 2) else 1 if (r < 2 and c >= 2) else 2 if (r >= 2 and c < 2) else 3)
            rect = patches.Rectangle((4.2 + c * 0.75, 3.6 - r * 0.75), 0.7, 0.7,
                                     facecolor=colors_quad[q], edgecolor='#2c3e50', lw=1.2)
            ax1.add_patch(rect)
            ax1.text(4.2 + c * 0.75 + 0.35, 3.6 - r * 0.75 + 0.35, str(out_a[r][c]),
                     ha='center', va='center', fontsize=11, fontweight='bold')
    ax1.text(5.7, 0.9, "Replicated Output ($4 \\times 4$)", ha='center', fontsize=9.5, fontweight='bold')

    # --- (b) Max Unpooling (Fixed top-left) ---
    ax2.set_title("(b) Max Unpooling (Fixed Top-Left)", fontsize=10.5, fontweight='bold', pad=10)
    for r in range(2):
        for c in range(2):
            rect = patches.Rectangle((0.5 + c, 3.0 - r), 0.9, 0.9, facecolor='#fcf3cf', edgecolor='#b7950b', lw=1.5)
            ax2.add_patch(rect)
            ax2.text(0.95 + c, 3.45 - r, str(input_vals[r][c]), ha='center', va='center', fontsize=14, fontweight='bold')
    ax2.text(1.45, 1.8, "Input ($2 \\times 2$)", ha='center', fontsize=9.5, fontweight='bold')

    ax2.annotate("", xy=(3.6, 3.0), xytext=(2.7, 3.0),
                 arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=2.0))

    out_b = [[1, 0, 2, 0],
             [0, 0, 0, 0],
             [3, 0, 4, 0],
             [0, 0, 0, 0]]

    for r in range(4):
        for c in range(4):
            q = (0 if (r < 2 and c < 2) else 1 if (r < 2 and c >= 2) else 2 if (r >= 2 and c < 2) else 3)
            is_nonzero = (out_b[r][c] != 0)
            fc = colors_quad[q] if is_nonzero else '#f8f9f9'
            tc = '#2c3e50' if is_nonzero else '#7f8c8d'
            rect = patches.Rectangle((4.2 + c * 0.75, 3.6 - r * 0.75), 0.7, 0.7,
                                     facecolor=fc, edgecolor='#2c3e50', lw=1.2)
            ax2.add_patch(rect)
            ax2.text(4.2 + c * 0.75 + 0.35, 3.6 - r * 0.75 + 0.35, str(out_b[r][c]),
                     ha='center', va='center', fontsize=11, fontweight='bold', color=tc)
    ax2.text(5.7, 0.9, "Fixed Placement Output ($4 \\times 4$)", ha='center', fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    _save_figure(fig, "Figure_10_28", save_dir)
    return fig


def generate_figure_10_29(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.29: Max-unpooling with switch variables (SegNet).
    Left: 4x4 input to max-pooling.
    Middle: 2x2 pooled map and recorded max locations.
    Right: 4x4 reconstructed map placing values into original locations.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.0)
    ax.axis('off')
    ax.set_title("Figure 10.29: Max-Unpooling with Switch Variables (Preserving Max Indices)",
                 fontsize=11.5, fontweight='bold', pad=12)

    # Values from textbook Figure 10.29
    # 4x4 input on left
    vals_in = [
        [5, 2, 4, 2],
        [7, 1, 0, 3],
        [7, 4, 3, 8],
        [9, 6, 8, 9]
    ]
    # Max in each 2x2:
    # TL: max(5,2,7,1) = 7 at (1, 0)
    # TR: max(4,2,0,3) = 4 at (0, 0)
    # BL: max(7,4,9,6) = 9 at (1, 0)
    # BR: max(3,8,8,9) = 9 at (1, 1)

    # 1. Input 4x4
    x0 = 0.5
    for r in range(4):
        for c in range(4):
            is_max = ((r, c) in [(1, 0), (0, 2), (3, 0), (3, 3)])
            fc = '#fadbd8' if is_max else '#f4f6f7'
            ec = '#c0392b' if is_max else '#bdc3c7'
            lw = 1.8 if is_max else 0.8
            rect = patches.Rectangle((x0 + c * 0.65, 3.6 - r * 0.65), 0.62, 0.62,
                                     facecolor=fc, edgecolor=ec, lw=lw)
            ax.add_patch(rect)
            ax.text(x0 + c * 0.65 + 0.31, 3.6 - r * 0.65 + 0.31, str(vals_in[r][c]),
                    ha='center', va='center', fontsize=10, fontweight='bold' if is_max else 'normal',
                    color='#900c3f' if is_max else '#2c3e50')

    ax.text(x0 + 1.3, 0.8, "Encoder Max-Pool\nInput ($4 \\times 4$)", ha='center', fontsize=9.0, fontweight='bold')

    # Arrow to pooled
    ax.annotate("", xy=(3.8, 2.6), xytext=(3.3, 2.6),
                arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=2.0))
    ax.text(3.55, 2.85, "Pool", ha='center', fontsize=8.5, fontweight='bold')

    # 2. 2x2 Pooled Map
    vals_pooled = [[7, 4], [9, 9]]
    x_mid = 4.2
    for r in range(2):
        for c in range(2):
            rect = patches.Rectangle((x_mid + c * 0.75, 3.0 - r * 0.75), 0.7, 0.7,
                                     facecolor='#d4e6f1', edgecolor='#2980b9', lw=1.8)
            ax.add_patch(rect)
            ax.text(x_mid + c * 0.75 + 0.35, 3.0 - r * 0.75 + 0.35, str(vals_pooled[r][c]),
                    ha='center', va='center', fontsize=12, fontweight='bold', color='#1b4f72')

    ax.text(x_mid + 0.75, 0.8, "Pooled Map ($2 \\times 2$)\n+ Saved Indices", ha='center', fontsize=9.0, fontweight='bold')

    # Arrow to unpooled
    ax.annotate("", xy=(6.8, 2.6), xytext=(6.1, 2.6),
                arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=2.0))
    ax.text(6.45, 2.85, "Unpool", ha='center', fontsize=8.5, fontweight='bold')

    # 3. 4x4 Restored Map
    vals_out = [
        [0, 0, 4, 0],
        [7, 0, 0, 0],
        [0, 0, 0, 0],
        [9, 0, 0, 9]
    ]
    x_out = 7.3
    for r in range(4):
        for c in range(4):
            is_val = (vals_out[r][c] != 0)
            fc = '#d5f5e3' if is_val else '#f4f6f7'
            ec = '#27ae60' if is_val else '#bdc3c7'
            tc = '#145a32' if is_val else '#7f8c8d'
            lw = 1.8 if is_val else 0.8
            rect = patches.Rectangle((x_out + c * 0.65, 3.6 - r * 0.65), 0.62, 0.62,
                                     facecolor=fc, edgecolor=ec, lw=lw)
            ax.add_patch(rect)
            ax.text(x_out + c * 0.65 + 0.31, 3.6 - r * 0.65 + 0.31, str(vals_out[r][c]),
                    ha='center', va='center', fontsize=10, fontweight='bold' if is_val else 'normal', color=tc)

    ax.text(x_out + 1.3, 0.8, "Decoder Unpool\nOutput ($4 \\times 4$)", ha='center', fontsize=9.0, fontweight='bold')

    plt.tight_layout()
    _save_figure(fig, "Figure_10_29", save_dir)
    return fig


def generate_figure_10_30(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.30: Transpose convolution for a 3x3 filter with an output stride of 2.
    Input 2x2 -> Output 5x5.
    Red patch from top-left input, Blue patch from top-right input; overlapping cells summed.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 5.2)
    ax.axis('off')
    ax.set_title("Figure 10.30: Transpose Convolution ($3 \\times 3$ Filter, Output Stride $S=2$)",
                 fontsize=11.5, fontweight='bold', pad=12)

    # 1. Input 2x2
    # Top-left is Red, Top-right is Blue
    x0 = 0.8
    y0 = 2.4
    sz_in = 0.8
    rect_r = patches.Rectangle((x0, y0 + sz_in), sz_in, sz_in, facecolor='#e74c3c', edgecolor='#922b21', lw=2.0)
    rect_b = patches.Rectangle((x0 + sz_in, y0 + sz_in), sz_in, sz_in, facecolor='#3498db', edgecolor='#1d6fa5', lw=2.0)
    rect_3 = patches.Rectangle((x0, y0), sz_in, sz_in, facecolor='#eaeded', edgecolor='#7f8c8d', lw=1.2)
    rect_4 = patches.Rectangle((x0 + sz_in, y0), sz_in, sz_in, facecolor='#eaeded', edgecolor='#7f8c8d', lw=1.2)
    ax.add_patch(rect_r)
    ax.add_patch(rect_b)
    ax.add_patch(rect_3)
    ax.add_patch(rect_4)

    ax.text(x0 + sz_in / 2.0, y0 + 1.5 * sz_in, "$z_1$", color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(x0 + 1.5 * sz_in, y0 + 1.5 * sz_in, "$z_2$", color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(x0 + sz_in, y0 - 0.45, "Input Map ($2 \\times 2$)", ha='center', fontsize=9.5, fontweight='bold')

    # Arrow to Output
    ax.annotate("", xy=(4.2, y0 + 0.8), xytext=(3.0, y0 + 0.8),
                arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=2.2))
    ax.text(3.6, y0 + 1.1, "Transpose\nConv $3 \\times 3$\nStride 2", ha='center', fontsize=8.5, fontweight='bold')

    # 2. Output 5x5
    x_out = 4.8
    y_out = 0.8
    sz_out = 0.72

    # Draw full 5x5 grid
    for r in range(5):
        for c in range(5):
            rect = patches.Rectangle((x_out + c * sz_out, y_out + (4 - r) * sz_out), sz_out, sz_out,
                                     facecolor='#f8f9f9', edgecolor='#bdc3c7', lw=1.0)
            ax.add_patch(rect)

    # Red patch: rows 0..2, cols 0..2 (top-left 3x3)
    for r in range(3):
        for c in range(3):
            rect = patches.Rectangle((x_out + c * sz_out, y_out + (4 - r) * sz_out), sz_out, sz_out,
                                     facecolor='#fadbd8', edgecolor='#e74c3c', alpha=0.5, lw=1.5)
            ax.add_patch(rect)

    # Blue patch: rows 0..2, cols 2..4 (top-right 3x3, shifted right by stride 2)
    for r in range(3):
        for c in range(2, 5):
            # Overlap at column 2!
            is_overlap = (c == 2)
            fc = '#d4e6f1' if not is_overlap else '#d7bde2'  # Purple overlap
            ec = '#2980b9' if not is_overlap else '#8e44ad'
            rect = patches.Rectangle((x_out + c * sz_out, y_out + (4 - r) * sz_out), sz_out, sz_out,
                                     facecolor=fc, edgecolor=ec, alpha=0.6, lw=1.5)
            ax.add_patch(rect)

    # Label overlap column
    ax.text(x_out + 2.5 * sz_out, y_out + 4.5 * sz_out + 0.25, "Overlap Column (Summed)",
            ha='center', fontsize=8.0, fontweight='bold', color='#8e44ad')

    ax.text(x_out + 2.5 * sz_out, y_out - 0.45, "Output Feature Map ($5 \\times 5$)",
            ha='center', fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    _save_figure(fig, "Figure_10_30", save_dir)
    return fig


def generate_figure_10_31(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.31: The U-net architecture (Ronneberger et al., 2015).
    Symmetrical arrangement of down-sampling and up-sampling layers with skip connections.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 6.0)
    ax.axis('off')
    ax.set_title("Figure 10.31: The U-net Architecture for Semantic Segmentation",
                 fontsize=12, fontweight='bold', pad=12)

    # Left branch (Encoder / Down-sampling)
    # Level 1 (Top left): 572x572 -> 64
    # Level 2: 280x280 -> 128
    # Level 3: 136x136 -> 256
    # Level 4: 64x64 -> 512
    # Bottom (Bottleneck): 28x28 -> 1024

    levels = [
        # (level_idx, y_center, x_enc, x_dec, height, channels_enc, channels_dec, dim_txt)
        (0, 4.6, 1.0, 9.5, 1.4, 64, 64, "572×572"),
        (1, 3.4, 2.2, 8.3, 1.1, 128, 128, "280×280"),
        (2, 2.3, 3.4, 7.1, 0.85, 256, 256, "136×136"),
        (3, 1.3, 4.5, 6.0, 0.65, 512, 512, "64×64"),
    ]

    # Draw encoder blocks & decoder blocks
    w_block = 0.6
    for idx, yc, xe, xd, h, ce, cd, dtxt in levels:
        # Encoder block (Blue)
        re = patches.Rectangle((xe - w_block / 2.0, yc - h / 2.0), w_block, h,
                               facecolor='#d4e6f1', edgecolor='#2980b9', lw=1.5)
        ax.add_patch(re)
        ax.text(xe, yc, str(ce), ha='center', va='center', fontsize=8.0, fontweight='bold')
        ax.text(xe - 0.45, yc, dtxt, ha='right', va='center', fontsize=7.0, color='#34495e')

        # Decoder block (Green)
        rd = patches.Rectangle((xd - w_block / 2.0, yc - h / 2.0), w_block, h,
                               facecolor='#d5f5e3', edgecolor='#27ae60', lw=1.5)
        ax.add_patch(rd)
        ax.text(xd, yc, str(cd), ha='center', va='center', fontsize=8.0, fontweight='bold')
        ax.text(xd + 0.45, yc, dtxt, ha='left', va='center', fontsize=7.0, color='#34495e')

        # Skip connection arrow (Horizontal dashed line from encoder to decoder)
        ax.annotate("", xy=(xd - w_block / 2.0, yc), xytext=(xe + w_block / 2.0, yc),
                    arrowprops=dict(arrowstyle="->", color='#8e44ad', lw=1.8, linestyle='--'))
        ax.text((xe + xd) / 2.0, yc + 0.15, "skip concat", color='#8e44ad',
                ha='center', fontsize=7.0, fontweight='bold')

    # Bottom bottleneck block
    xb, yb, hb = 5.25, 0.35, 0.5
    rb = patches.Rectangle((xb - 0.4, yb - hb / 2.0), 0.8, hb, facecolor='#fadbd8', edgecolor='#c0392b', lw=1.5)
    ax.add_patch(rb)
    ax.text(xb, yb, "1024", ha='center', va='center', fontsize=8.0, fontweight='bold')
    ax.text(xb, yb - 0.4, "28×28", ha='center', fontsize=7.5, color='#34495e')

    # Downsampling red arrows
    for i in range(len(levels) - 1):
        y1 = levels[i][1] - levels[i][4] / 2.0
        x1 = levels[i][2]
        y2 = levels[i + 1][1] + levels[i + 1][4] / 2.0
        x2 = levels[i + 1][2]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color='#c0392b', lw=1.6))

    # Encoder to bottleneck arrow
    ax.annotate("", xy=(xb - 0.2, yb + hb / 2.0), xytext=(levels[3][2], levels[3][1] - levels[3][4] / 2.0),
                arrowprops=dict(arrowstyle="->", color='#c0392b', lw=1.6))

    # Bottleneck to decoder arrow
    ax.annotate("", xy=(levels[3][3], levels[3][1] - levels[3][4] / 2.0), xytext=(xb + 0.2, yb + hb / 2.0),
                arrowprops=dict(arrowstyle="->", color='#27ae60', lw=1.6))

    # Upsampling green arrows
    for i in range(len(levels) - 1, 0, -1):
        y1 = levels[i][1] + levels[i][4] / 2.0
        x1 = levels[i][3]
        y2 = levels[i - 1][1] - levels[i - 1][4] / 2.0
        x2 = levels[i - 1][3]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color='#27ae60', lw=1.6))

    # Legend
    leg = [
        patches.Patch(facecolor='#d4e6f1', edgecolor='#2980b9', label='Encoder Conv Block'),
        patches.Patch(facecolor='#fadbd8', edgecolor='#c0392b', label='Bottleneck Latent'),
        patches.Patch(facecolor='#d5f5e3', edgecolor='#27ae60', label='Decoder Up-Conv Block'),
        patches.Patch(edgecolor='#8e44ad', facecolor='none', linestyle='--', label='Copy & Concatenate (Skip)'),
    ]
    ax.legend(handles=leg, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=True, fontsize=8.5)

    plt.tight_layout()
    _save_figure(fig, "Figure_10_31", save_dir)
    return fig
