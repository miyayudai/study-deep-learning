"""
common/convolutional_filters.py
================================
Section 10.2: Convolutional Filters
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements:
1. Discrete Convolution Operations:
   - 1D cross-correlation with parameter sharing (Figure 10.2).
   - 2D cross-correlation (Eq 10.2) supporting arbitrary kernels, strides (Eq 10.5), and padding (Figure 10.5).
   - Multi-channel 2D convolutions (M x M x C_in x C_out) and 1x1 convolutions (Figures 10.6, 10.7).
2. Spatial Pooling:
   - Max-pooling and Average-pooling with arbitrary window and stride (Figure 10.8).
3. Receptive Field Analysis:
   - Cumulative stride and effective receptive field growth calculator (Figure 10.9).
4. VGG-16 Architecture Analysis:
   - Complete layer-by-layer parameter and connection accounting (Figure 10.10, Exercise 10.8).
5. Canonical Hand-crafted Filters:
   - Vertical edge filter (Eq 10.3) and horizontal edge filter (Eq 10.4).
6. High-Resolution Figure Reproductions:
   - Figure 10.1: Receptive field and kernel visualization.
   - Figure 10.2: 1D convolution parameter sharing.
   - Figure 10.3: 2D convolution algebraic representation.
   - Figure 10.4: Edge detection with vertical and horizontal filters.
   - Figure 10.5: Zero-padding illustration.
   - Figure 10.6: Multi-channel filter across RGB.
   - Figure 10.7: Multiple output feature channels.
   - Figure 10.8: Max-pooling operation.
   - Figure 10.9: Effective receptive field growth with depth.
   - Figure 10.10: Complete VGG-16 architecture.
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
# Canonical Filters (Equations 10.3 & 10.4)
# =============================================================================
VERTICAL_EDGE_FILTER = np.array([
    [-1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
], dtype=np.float64)

HORIZONTAL_EDGE_FILTER = VERTICAL_EDGE_FILTER.T  # Transpose: Equation (10.4)


# =============================================================================
# Mathematical Convolution & Pooling Functions
# =============================================================================
def compute_conv_output_dim(in_dim: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
    """
    Compute 1D output dimension according to Equation (10.5):
    dim_out = floor((dim_in + 2*P - M) / S) + 1.
    """
    if stride <= 0:
        raise ValueError(f"Stride must be positive, got {stride}")
    if in_dim + 2 * padding < kernel_size:
        raise ValueError(f"Input dimension ({in_dim}) + 2*padding ({2*padding}) must be >= kernel size ({kernel_size})")
    return int(np.floor((in_dim + 2 * padding - kernel_size) / stride)) + 1


def pad2d(image: np.ndarray, padding: int, constant_value: float = 0.0) -> np.ndarray:
    """
    Pad a 2D image (H, W) or 3D tensor (H, W, C) with symmetric border of constant_value.
    """
    if padding < 0:
        raise ValueError("Padding cannot be negative")
    if padding == 0:
        return image.copy()

    if image.ndim == 2:
        return np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=constant_value)
    elif image.ndim == 3:
        return np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=constant_value)
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")


def conv1d(x: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    1D discrete cross-correlation with parameter sharing (Figure 10.2).
    Args:
        x: 1D array of input activations of length N.
        kernel: 1D array of kernel weights of length M.
        stride: integer stride step size.
        padding: number of zero-padding elements at each boundary.
    Returns:
        1D output feature map.
    """
    x = np.asarray(x, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if padding > 0:
        x = np.pad(x, (padding, padding), mode='constant', constant_values=0.0)

    N = len(x)
    M = len(kernel)
    out_len = compute_conv_output_dim(N - 2 * padding, M, stride, padding)
    out = np.zeros(out_len, dtype=np.float64)

    for i in range(out_len):
        start = i * stride
        out[i] = np.sum(x[start:start + M] * kernel)
    return out


def conv2d(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
    padding: int = 0,
    bias: float = 0.0,
) -> np.ndarray:
    """
    2D discrete cross-correlation (Equation 10.2) supporting single-channel or multi-channel tensors.
    Args:
        image: (H, W) or (H, W, C_in) float array.
        kernel: (M, M) or (M, M, C_in) or (M, M, C_in, C_out) float array.
        stride: spatial stride.
        padding: zero-padding border width.
        bias: scalar bias or array of length C_out.
    Returns:
        feature map of shape (H_out, W_out) or (H_out, W_out, C_out).
    """
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    # Standardize to 3D image (H, W, C_in) and 4D kernel (M, M, C_in, C_out)
    is_2d_input = (image.ndim == 2)
    if is_2d_input:
        img_3d = image[:, :, np.newaxis]
    else:
        img_3d = image

    H, W, C_in = img_3d.shape

    if kernel.ndim == 2:
        kernel_4d = kernel[:, :, np.newaxis, np.newaxis]
    elif kernel.ndim == 3:
        # (M, M, C_in) -> 1 output channel
        kernel_4d = kernel[:, :, :, np.newaxis]
    elif kernel.ndim == 4:
        kernel_4d = kernel
    else:
        raise ValueError(f"Invalid kernel dimensions: {kernel.ndim}")

    M_h, M_w, k_cin, C_out = kernel_4d.shape
    if k_cin != C_in:
        raise ValueError(f"Kernel channel count {k_cin} does not match image channel count {C_in}")

    # Compute output spatial dimensions (Equation 10.5)
    H_out = compute_conv_output_dim(H, M_h, stride, padding)
    W_out = compute_conv_output_dim(W, M_w, stride, padding)

    # Apply padding
    padded_img = pad2d(img_3d, padding, constant_value=0.0)

    # Handle bias
    if np.isscalar(bias):
        b = np.full(C_out, float(bias), dtype=np.float64)
    else:
        b = np.asarray(bias, dtype=np.float64)
        if len(b) != C_out:
            raise ValueError(f"Bias length {len(b)} does not match C_out {C_out}")

    output = np.zeros((H_out, W_out, C_out), dtype=np.float64)

    for i in range(H_out):
        r_start = i * stride
        r_end = r_start + M_h
        for j in range(W_out):
            c_start = j * stride
            c_end = c_start + M_w
            patch = padded_img[r_start:r_end, c_start:c_end, :]  # (M_h, M_w, C_in)
            for k in range(C_out):
                output[i, j, k] = np.sum(patch * kernel_4d[:, :, :, k]) + b[k]

    if is_2d_input and C_out == 1 and kernel.ndim == 2:
        return output[:, :, 0]
    return output


def max_pool2d(
    feature_map: np.ndarray,
    pool_size: int = 2,
    stride: int = 2,
) -> np.ndarray:
    """
    2D Max-pooling operation (Figure 10.8).
    Args:
        feature_map: (H, W) or (H, W, C) array.
        pool_size: spatial pooling window dimension.
        stride: spatial stride.
    Returns:
        pooled feature map.
    """
    fm = np.asarray(feature_map, dtype=np.float64)
    is_2d = (fm.ndim == 2)
    if is_2d:
        fm_3d = fm[:, :, np.newaxis]
    else:
        fm_3d = fm

    H, W, C = fm_3d.shape
    H_out = compute_conv_output_dim(H, pool_size, stride, padding=0)
    W_out = compute_conv_output_dim(W, pool_size, stride, padding=0)

    output = np.zeros((H_out, W_out, C), dtype=np.float64)

    for i in range(H_out):
        r_start = i * stride
        r_end = r_start + pool_size
        for j in range(W_out):
            c_start = j * stride
            c_end = c_start + pool_size
            patch = fm_3d[r_start:r_end, c_start:c_end, :]
            output[i, j, :] = np.max(patch, axis=(0, 1))

    if is_2d:
        return output[:, :, 0]
    return output


def avg_pool2d(
    feature_map: np.ndarray,
    pool_size: int = 2,
    stride: int = 2,
) -> np.ndarray:
    """
    2D Average-pooling operation.
    """
    fm = np.asarray(feature_map, dtype=np.float64)
    is_2d = (fm.ndim == 2)
    if is_2d:
        fm_3d = fm[:, :, np.newaxis]
    else:
        fm_3d = fm

    H, W, C = fm_3d.shape
    H_out = compute_conv_output_dim(H, pool_size, stride, padding=0)
    W_out = compute_conv_output_dim(W, pool_size, stride, padding=0)

    output = np.zeros((H_out, W_out, C), dtype=np.float64)

    for i in range(H_out):
        r_start = i * stride
        r_end = r_start + pool_size
        for j in range(W_out):
            c_start = j * stride
            c_end = c_start + pool_size
            patch = fm_3d[r_start:r_end, c_start:c_end, :]
            output[i, j, :] = np.mean(patch, axis=(0, 1))

    if is_2d:
        return output[:, :, 0]
    return output


# =============================================================================
# Receptive Field & Architecture Accounting
# =============================================================================
def compute_effective_receptive_field(layers: List[Dict[str, int]]) -> List[Dict[str, int]]:
    """
    Compute cumulative stride and effective receptive field (ERF) across multiple layers (Figure 10.9).
    Equation:
        R_l = R_{l-1} + (M_l - 1) * J_{l-1}
        J_l = J_{l-1} * S_l
    where R_0 = 1, J_0 = 1.
    """
    rf = 1
    jump = 1
    records = []

    for idx, layer in enumerate(layers):
        k = layer.get("kernel_size", 1)
        s = layer.get("stride", 1)
        rf = rf + (k - 1) * jump
        jump = jump * s
        records.append({
            "layer_index": idx + 1,
            "kernel_size": k,
            "stride": s,
            "receptive_field": rf,
            "cumulative_jump": jump,
        })
    return records


def analyze_vgg16_parameters() -> Dict[str, Any]:
    """
    Detailed layer-by-layer parameter and MACs analysis of the VGG-16 network (Figure 10.10, Exercise 10.8).
    VGG-16 specification:
    Input: 224 x 224 x 3
    Conv Block 1: 2 x (3x3 conv, 64) -> MaxPool
    Conv Block 2: 2 x (3x3 conv, 128) -> MaxPool
    Conv Block 3: 3 x (3x3 conv, 256) -> MaxPool
    Conv Block 4: 3 x (3x3 conv, 512) -> MaxPool
    Conv Block 5: 3 x (3x3 conv, 512) -> MaxPool
    Classifier: FC 4096 -> FC 4096 -> FC 1000
    """
    stages = [
        # (name, type, Cin, Cout, k, s, pad, Hin, Win)
        ("conv1_1", "conv", 3, 64, 3, 1, 1, 224, 224),
        ("conv1_2", "conv", 64, 64, 3, 1, 1, 224, 224),
        ("pool1",   "pool", 64, 64, 2, 2, 0, 224, 224),

        ("conv2_1", "conv", 64, 128, 3, 1, 1, 112, 112),
        ("conv2_2", "conv", 128, 128, 3, 1, 1, 112, 112),
        ("pool2",   "pool", 128, 128, 2, 2, 0, 112, 112),

        ("conv3_1", "conv", 128, 256, 3, 1, 1, 56, 56),
        ("conv3_2", "conv", 256, 256, 3, 1, 1, 56, 56),
        ("conv3_3", "conv", 256, 256, 3, 1, 1, 56, 56),
        ("pool3",   "pool", 256, 256, 2, 2, 0, 56, 56),

        ("conv4_1", "conv", 256, 512, 3, 1, 1, 28, 28),
        ("conv4_2", "conv", 512, 512, 3, 1, 1, 28, 28),
        ("conv4_3", "conv", 512, 512, 3, 1, 1, 28, 28),
        ("pool4",   "pool", 512, 512, 2, 2, 0, 28, 28),

        ("conv5_1", "conv", 512, 512, 3, 1, 1, 14, 14),
        ("conv5_2", "conv", 512, 512, 3, 1, 1, 14, 14),
        ("conv5_3", "conv", 512, 512, 3, 1, 1, 14, 14),
        ("pool5",   "pool", 512, 512, 2, 2, 0, 14, 14),

        ("fc6",     "fc",   7 * 7 * 512, 4096, 1, 1, 0, 1, 1),
        ("fc7",     "fc",   4096, 4096, 1, 1, 0, 1, 1),
        ("fc8",     "fc",   4096, 1000, 1, 1, 0, 1, 1),
    ]

    layer_details = []
    total_weights = 0
    total_biases = 0
    total_connections = 0

    for name, ltype, cin, cout, k, s, p, hin, win in stages:
        if ltype == "conv":
            hout = compute_conv_output_dim(hin, k, s, p)
            wout = compute_conv_output_dim(win, k, s, p)
            weights = (k * k * cin) * cout
            biases = cout
            connections = hout * wout * weights
        elif ltype == "pool":
            hout = compute_conv_output_dim(hin, k, s, p)
            wout = compute_conv_output_dim(win, k, s, p)
            weights = 0
            biases = 0
            connections = hout * wout * cout * (k * k)
        elif ltype == "fc":
            hout, wout = 1, 1
            weights = cin * cout
            biases = cout
            connections = weights
        else:
            continue

        total_weights += weights
        total_biases += biases
        total_connections += connections

        layer_details.append({
            "name": name,
            "type": ltype,
            "input_shape": (hin, win, cin) if ltype != "fc" else (cin,),
            "output_shape": (hout, wout, cout) if ltype != "fc" else (cout,),
            "weights": weights,
            "biases": biases,
            "total_params": weights + biases,
            "connections": connections,
        })

    return {
        "layers": layer_details,
        "total_weights": total_weights,
        "total_biases": total_biases,
        "total_params": total_weights + total_biases,
        "total_connections": total_connections,
    }


# =============================================================================
# Publication-Quality Textbook Figure Generators (Figure 10.1 to Figure 10.10)
# =============================================================================
def generate_figure_10_1(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.1: (a) Illustration of a receptive field (3x3 patch on image connected to hidden unit).
                 (b) Weight values visualized as a 3x3 kernel matrix.
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.4), gridspec_kw={'width_ratios': [1.3, 1.0]})

    # --- (a) Receptive Field ---
    ax1.set_xlim(-0.5, 7.5)
    ax1.set_ylim(-0.5, 6.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title("(a) Receptive Field & Hidden Unit", fontsize=11, fontweight='bold', pad=10)

    # Draw image grid (5x5) at bottom
    grid_x0, grid_y0 = 1.0, 0.5
    for r in range(5):
        for c in range(5):
            is_rf = (1 <= r <= 3) and (1 <= c <= 3)
            fc = '#ffcccc' if is_rf else '#e8e8e8'
            ec = '#c0392b' if is_rf else '#7f8c8d'
            rect = patches.Rectangle((grid_x0 + c, grid_y0 + (4 - r)), 0.9, 0.9, facecolor=fc, edgecolor=ec, lw=1.2)
            ax1.add_patch(rect)

    ax1.text(grid_x0 + 2.45, grid_y0 - 0.35, "Input Image Grid ($5 \\times 5$)", ha='center', fontsize=9.5, fontweight='bold')

    # Draw receptive field boundary
    rf_box = patches.Rectangle((grid_x0 + 0.95, grid_y0 + 0.95), 2.9, 2.9, fill=False, edgecolor='#d62728', lw=2.2, linestyle='--')
    ax1.add_patch(rf_box)
    ax1.text(grid_x0 - 0.2, grid_y0 + 2.4, "3×3 Receptive\nField Patch", color='#d62728', fontsize=8.5, fontweight='bold', va='center', ha='right')

    # Hidden unit above
    hx, hy = grid_x0 + 2.45, 5.8
    circle = patches.Circle((hx, hy), 0.38, facecolor='#3498db', edgecolor='#1d6fa5', lw=2.0, zorder=5)
    ax1.add_patch(circle)
    ax1.text(hx, hy, "$z$", color='white', fontsize=12, fontweight='bold', ha='center', va='center', zorder=6)
    ax1.text(hx + 0.55, hy, "Hidden Unit\n$z = \\mathrm{ReLU}(\\mathbf{w}^\\top\\mathbf{x} + w_0)$", fontsize=9.0, va='center')

    # Feedforward connections from RF corners to hidden unit
    rf_corners = [
        (grid_x0 + 1.0, grid_y0 + 3.9),
        (grid_x0 + 3.9, grid_y0 + 3.9),
        (grid_x0 + 1.0, grid_y0 + 1.0),
        (grid_x0 + 3.9, grid_y0 + 1.0),
    ]
    for cx, cy in rf_corners:
        ax1.annotate("", xy=(hx, hy - 0.38), xytext=(cx, cy),
                     arrowprops=dict(arrowstyle="->", color='#e74c3c', lw=1.2, alpha=0.8))

    # --- (b) Kernel Matrix Visualization ---
    ax2.set_xlim(-0.5, 4.5)
    ax2.set_ylim(-0.5, 4.5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title("(b) Kernel Weights $\\mathbf{K}$ ($3 \\times 3$)", fontsize=11, fontweight='bold', pad=10)

    # Weights from textbook Figure 10.1(b)
    weights = [
        [0.4, 1.7, 0.9],
        [2.3, -2.1, 4.0],
        [-1.4, 0.7, 2.1],
    ]

    kx0, ky0 = 0.5, 0.5
    for r in range(3):
        for c in range(3):
            val = weights[r][c]
            # Color map based on value
            norm_val = (val + 2.5) / 7.0
            color = plt.cm.coolwarm(norm_val)
            rect = patches.Rectangle((kx0 + c, ky0 + (2 - r)), 0.92, 0.92, facecolor=color, edgecolor='#2c3e50', lw=1.5)
            ax2.add_patch(rect)
            text_color = 'white' if abs(val) > 2.0 else 'black'
            ax2.text(kx0 + c + 0.46, ky0 + (2 - r) + 0.46, f"{val:+.1f}",
                     ha='center', va='center', fontsize=10.5, fontweight='bold', color=text_color)

    ax2.text(kx0 + 1.4, ky0 - 0.35, "Learned Kernel Filter $\\mathbf{w}$", ha='center', fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    _save_figure(fig, "Figure_10_1", save_dir)
    return fig


def generate_figure_10_2(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.2: 1D convolution with kernel width 2, sparse shared weights.
    Showing 6 connections but only 2 independent learnable parameters (red and blue links).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')
    ax.set_title("Figure 10.2: 1D Convolution with Parameter Sharing ($M=2$)", fontsize=11, fontweight='bold', pad=12)

    # Input nodes (x_1 to x_5)
    n_in = 5
    x_in_pos = np.arange(1, n_in + 1)
    y_in = 0.3

    for idx, x in enumerate(x_in_pos):
        c = patches.Circle((x, y_in), 0.22, facecolor='#ecf0f1', edgecolor='#2c3e50', lw=1.6, zorder=5)
        ax.add_patch(c)
        ax.text(x, y_in, f"$x_{idx+1}$", ha='center', va='center', fontsize=10, fontweight='bold', zorder=6)

    ax.text(0.1, y_in, "Input Layer\n$\\mathbf{x}$", ha='right', va='center', fontsize=9.5, fontweight='bold')

    # Hidden nodes (z_1 to z_4)
    n_hidden = 4
    x_hid_pos = np.arange(1.5, 1.5 + n_hidden)
    y_hid = 1.9

    for idx, x in enumerate(x_hid_pos):
        c = patches.Circle((x, y_hid), 0.22, facecolor='#d5dbdb', edgecolor='#2c3e50', lw=1.6, zorder=5)
        ax.add_patch(c)
        ax.text(x, y_hid, f"$z_{idx+1}$", ha='center', va='center', fontsize=10, fontweight='bold', zorder=6)

    ax.text(0.1, y_hid, "Feature Map\n$\\mathbf{z}$", ha='right', va='center', fontsize=9.5, fontweight='bold')

    # Connections: kernel size 2
    # Left link (w_1: red), Right link (w_2: blue)
    color_w1 = '#e74c3c'  # Red
    color_w2 = '#2980b9'  # Blue

    for i in range(n_hidden):
        x_h = x_hid_pos[i]
        # Link from x_{i+1} (w1: red)
        x_in1 = x_in_pos[i]
        ax.annotate("", xy=(x_h - 0.05, y_hid - 0.22), xytext=(x_in1 + 0.05, y_in + 0.22),
                    arrowprops=dict(arrowstyle="->", color=color_w1, lw=2.2))
        # Link from x_{i+2} (w2: blue)
        x_in2 = x_in_pos[i + 1]
        ax.annotate("", xy=(x_h + 0.05, y_hid - 0.22), xytext=(x_in2 - 0.05, y_in + 0.22),
                    arrowprops=dict(arrowstyle="->", color=color_w2, lw=2.2))

    # Legend / parameter explanation
    patch_r = patches.Patch(color=color_w1, label=r"Shared Weight $w_1$ (Red links)")
    patch_b = patches.Patch(color=color_w2, label=r"Shared Weight $w_2$ (Blue links)")
    ax.legend(handles=[patch_r, patch_b], loc='upper right', frameon=True, fontsize=9.0)

    ax.text(3.0, -0.2, "6 sparse connections, but only 2 independent learnable parameters ($w_1, w_2$)",
            ha='center', fontsize=9.5, style='italic', color='#34495e')

    plt.tight_layout()
    _save_figure(fig, "Figure_10_2", save_dir)
    return fig


def generate_figure_10_3(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.3: Example of a 3x3 image convolved with a 2x2 filter to give a 2x2 feature map.
    Showing explicit algebraic terms (C_11 = aj + bk + dl + em, etc.).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title("Figure 10.3: 2D Convolution of a $3 \\times 3$ Image with a $2 \\times 2$ Filter",
                 fontsize=12, fontweight='bold', pad=12)

    # 1. Image I (3x3)
    x0, y0 = 0.5, 1.2
    box_sz = 1.0
    symbols_I = [['a', 'b', 'c'],
                 ['d', 'e', 'f'],
                 ['g', 'h', 'i']]

    for r in range(3):
        for c in range(3):
            # Highlight top-left 2x2 receptive field
            in_rf = (r <= 1 and c <= 1)
            fc = '#fff3cd' if in_rf else '#ffffff'
            rect = patches.Rectangle((x0 + c * box_sz, y0 + (2 - r) * box_sz), box_sz, box_sz,
                                     facecolor=fc, edgecolor='#34495e', lw=1.5)
            ax.add_patch(rect)
            ax.text(x0 + (c + 0.5) * box_sz, y0 + (2.5 - r) * box_sz, symbols_I[r][c],
                    ha='center', va='center', fontsize=14, fontweight='bold')

    ax.text(x0 + 1.5 * box_sz, y0 - 0.5, "Image $\\mathbf{I}$ ($3 \\times 3$)",
            ha='center', fontsize=11, fontweight='bold')

    # Convolution operator
    ax.text(4.0, y0 + 1.5 * box_sz, "$\\ast$", ha='center', va='center', fontsize=26, fontweight='bold')

    # 2. Kernel K (2x2)
    kx0 = 4.8
    symbols_K = [['j', 'k'],
                 ['l', 'm']]

    for r in range(2):
        for c in range(2):
            rect = patches.Rectangle((kx0 + c * box_sz, y0 + (1.5 - r) * box_sz), box_sz, box_sz,
                                     facecolor='#d0e1fd', edgecolor='#2c3e50', lw=1.5)
            ax.add_patch(rect)
            ax.text(kx0 + (c + 0.5) * box_sz, y0 + (2.0 - r) * box_sz, symbols_K[r][c],
                    ha='center', va='center', fontsize=14, fontweight='bold')

    ax.text(kx0 + box_sz, y0 - 0.5, "Kernel $\\mathbf{K}$ ($2 \\times 2$)",
            ha='center', fontsize=11, fontweight='bold')

    # Equals sign
    ax.text(7.4, y0 + 1.5 * box_sz, "$=$", ha='center', va='center', fontsize=24, fontweight='bold')

    # 3. Feature map C (2x2) with algebraic formulas
    cx0 = 8.1
    c_w = 2.7
    c_h = 1.4
    terms = [
        ["$aj + bk +$\n$dl + em$", "$bj + ck +$\n$el + fm$"],
        ["$dj + ek +$\n$gl + hm$", "$ej + fk +$\n$hl + im$"]
    ]

    for r in range(2):
        for c in range(2):
            fc = '#fff3cd' if (r == 0 and c == 0) else '#e8f8f5'
            rect = patches.Rectangle((cx0 + c * c_w, y0 + (1 - r) * c_h + 0.3), c_w, c_h,
                                     facecolor=fc, edgecolor='#16a085', lw=1.8)
            ax.add_patch(rect)
            ax.text(cx0 + (c + 0.5) * c_w, y0 + (1.5 - r) * c_h + 0.3, terms[r][c],
                    ha='center', va='center', fontsize=10.5, fontweight='bold')

    ax.text(cx0 + c_w, y0 - 0.5, "Feature Map $\\mathbf{C} = \\mathbf{I} \\ast \\mathbf{K}$ ($2 \\times 2$)",
            ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    _save_figure(fig, "Figure_10_3", save_dir)
    return fig


def generate_figure_10_4(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.4: Edge detection using convolutional filters.
    (a) Original image with synthetic geometric structures.
    (b) Convolving with vertical edge filter (Eq 10.3).
    (c) Convolving with horizontal edge filter (Eq 10.4).
    """
    setup_style()
    # Create sample image with vertical, horizontal, and diagonal edges + circular disc
    H, W = 64, 64
    img = np.zeros((H, W), dtype=np.float64)

    # Background gradient
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    xx, yy = np.meshgrid(x, y)

    # Vertical stripe
    img[:, 16:32] += 0.7
    # Horizontal stripe
    img[24:40, :] += 0.5
    # Central circle
    disc = (xx**2 + (yy - 0.1)**2) < 0.18
    img[disc] = 0.95
    img = np.clip(img, 0.0, 1.0)

    # Convolve with vertical edge filter (Eq 10.3)
    out_v = conv2d(img, VERTICAL_EDGE_FILTER, padding=1)

    # Convolve with horizontal edge filter (Eq 10.4)
    out_h = conv2d(img, HORIZONTAL_EDGE_FILTER, padding=1)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))

    # (a) Original
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("(a) Original Image", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # (b) Vertical edges
    v_abs_max = max(abs(out_v.min()), abs(out_v.max()), 1e-4)
    im_v = axes[1].imshow(out_v, cmap='gray', vmin=-v_abs_max, vmax=v_abs_max)
    axes[1].set_title("(b) Vertical Edge Filter (Eq 10.3)", fontsize=11, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im_v, ax=axes[1], fraction=0.046, pad=0.04)

    # (c) Horizontal edges
    h_abs_max = max(abs(out_h.min()), abs(out_h.max()), 1e-4)
    im_h = axes[2].imshow(out_h, cmap='gray', vmin=-h_abs_max, vmax=h_abs_max)
    axes[2].set_title("(c) Horizontal Edge Filter (Eq 10.4)", fontsize=11, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im_h, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    _save_figure(fig, "Figure_10_4", save_dir)
    return fig


def generate_figure_10_5(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.5: Illustration of a 4x4 image padded with P=1 additional pixels to create a 6x6 image.
    Faithful reproduction of page 311.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Figure 10.5: Padded $4 \\times 4$ Image with $P=1$ ($6 \\times 6$ Grid)",
                 fontsize=11, fontweight='bold', pad=12)

    # Outer 6x6 grid
    for r in range(6):
        for c in range(6):
            is_interior = (1 <= r <= 4) and (1 <= c <= 4)
            if is_interior:
                fc = '#d4efdf'  # Soft green for interior original pixels
                ec = '#27ae60'
                txt = f"$x_{{{r}{c}}}$"
                tc = '#145a32'
                lw = 1.8
            else:
                fc = '#f2f4f4'  # Grayish border for zero padding
                ec = '#bdc3c7'
                txt = "0"
                tc = '#7f8c8d'
                lw = 1.0

            rect = patches.Rectangle((c, 5 - r), 0.94, 0.94, facecolor=fc, edgecolor=ec, lw=lw)
            ax.add_patch(rect)
            ax.text(c + 0.47, 5 - r + 0.47, txt, ha='center', va='center',
                    fontsize=11 if is_interior else 12, fontweight='bold', color=tc)

    # Highlight boundary between padding and image
    inner_box = patches.Rectangle((1.0, 1.0), 3.94, 3.94, fill=False, edgecolor='#27ae60', lw=2.5)
    ax.add_patch(inner_box)

    ax.text(3.0, -0.3, "Original $4 \\times 4$ array padded with border $P=1$ (Zero-padding)",
            ha='center', fontsize=9.5, style='italic', color='#2c3e50')

    plt.tight_layout()
    _save_figure(fig, "Figure_10_5", save_dir)
    return fig


def generate_figure_10_6(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.6: (a) Illustration of a multi-dimensional filter across R, G, B channels.
                 (b) 3x3x3 kernel tensor with 27 weights.
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={'width_ratios': [1.2, 1.0]})

    # --- (a) Multi-dimensional filter across channels ---
    ax1.set_xlim(-0.5, 7.0)
    ax1.set_ylim(-0.5, 6.0)
    ax1.axis('off')
    ax1.set_title("(a) Convolution Across R, G, B Channels", fontsize=11, fontweight='bold', pad=10)

    # 3 Staggered channel planes (Blue, Green, Red)
    channels_info = [
        ("Blue (B)", '#3498db', 0.0, 0.0),
        ("Green (G)", '#2ecc71', 0.6, 0.6),
        ("Red (R)", '#e74c3c', 1.2, 1.2),
    ]

    for name, col, dx, dy in channels_info:
        # 4x4 grid plane
        for r in range(4):
            for c in range(4):
                is_patch = (1 <= r <= 3) and (1 <= c <= 3)
                fc = col if is_patch else '#f8f9f9'
                alpha = 0.7 if is_patch else 0.35
                rect = patches.Rectangle((dx + c * 0.7, dy + (3 - r) * 0.7), 0.62, 0.62,
                                         facecolor=fc, edgecolor=col, alpha=alpha, lw=1.2)
                ax1.add_patch(rect)
        ax1.text(dx + 2.8, dy + 0.1, name, fontsize=8.5, fontweight='bold', color=col)

    # Hidden unit receiving from all 3 channels
    hx, hy = 5.8, 4.2
    c_out = patches.Circle((hx, hy), 0.36, facecolor='#9b59b6', edgecolor='#6c3483', lw=2.0, zorder=5)
    ax1.add_patch(c_out)
    ax1.text(hx, hy, "$z$", color='white', fontsize=11, fontweight='bold', ha='center', va='center', zorder=6)
    ax1.text(hx, hy - 0.7, "Feature Map\nUnit", ha='center', fontsize=9.0, fontweight='bold')

    # Connecting arrows from patches to hidden unit
    ax1.annotate("", xy=(hx - 0.36, hy), xytext=(1.2 + 2.1, 1.2 + 1.4),
                 arrowprops=dict(arrowstyle="->", color='#e74c3c', lw=1.6))
    ax1.annotate("", xy=(hx - 0.36, hy - 0.1), xytext=(0.6 + 2.1, 0.6 + 1.4),
                 arrowprops=dict(arrowstyle="->", color='#2ecc71', lw=1.6))
    ax1.annotate("", xy=(hx - 0.36, hy - 0.2), xytext=(0.0 + 2.1, 0.0 + 1.4),
                 arrowprops=dict(arrowstyle="->", color='#3498db', lw=1.6))

    # --- (b) 3x3x3 Tensor Visualization ---
    ax2.set_xlim(-0.5, 5.0)
    ax2.set_ylim(-0.5, 5.5)
    ax2.axis('off')
    ax2.set_title("(b) Kernel Tensor ($3 \\times 3 \\times 3$, 27 weights)", fontsize=11, fontweight='bold', pad=10)

    # 3 slices of 3x3 kernels
    slices = [
        ("Channel 2 (B)", '#d4e6f1', 0.2, 0.4),
        ("Channel 1 (G)", '#d5f5e3', 0.7, 1.1),
        ("Channel 0 (R)", '#fadbd8', 1.2, 1.8),
    ]

    for sname, col, sx, sy in slices:
        for r in range(3):
            for c in range(3):
                rect = patches.Rectangle((sx + c * 0.8, sy + (2 - r) * 0.8), 0.72, 0.72,
                                         facecolor=col, edgecolor='#2c3e50', lw=1.2)
                ax2.add_patch(rect)
                ax2.text(sx + c * 0.8 + 0.36, sy + (2 - r) * 0.8 + 0.36, "$w$",
                         ha='center', va='center', fontsize=8.5, color='#34495e')

    ax2.text(2.5, -0.2, "Total: $3 \\times 3 \\times 3 = 27$ weights (+ 1 bias)",
            ha='center', fontsize=9.5, fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    _save_figure(fig, "Figure_10_6", save_dir)
    return fig


def generate_figure_10_7(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.7: Multi-channel convolutional layer with multiple output feature channels.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis('off')
    ax.set_title("Figure 10.7: Multi-channel Convolution with Multiple Output Channels ($C_{\\mathrm{OUT}}$)",
                 fontsize=11, fontweight='bold', pad=12)

    # Input channels C_IN = 3
    x_in = 1.0
    for i, col in enumerate(['#3498db', '#2ecc71', '#e74c3c']):
        rect = patches.Rectangle((x_in + i * 0.35, 1.2 + i * 0.35), 2.2, 2.2,
                                 facecolor='#f4f6f7', edgecolor=col, lw=2.0)
        ax.add_patch(rect)
    ax.text(x_in + 1.45, 0.7, "Input Channels\n($C_{\\mathrm{IN}} = 3$)", ha='center', fontsize=9.5, fontweight='bold')

    # Filter bank symbol
    ax.text(4.7, 2.7, "$\\bigotimes$\n$\\mathbf{W} \\in \\mathbb{R}^{M \\times M \\times C_{\\mathrm{IN}} \\times C_{\\mathrm{OUT}}}$",
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Output channels C_OUT = 4
    x_out = 7.0
    out_colors = ['#9b59b6', '#e67e22', '#1abc9c', '#f1c40f']
    for j, col in enumerate(out_colors):
        rect = patches.Rectangle((x_out + j * 0.4, 1.0 + j * 0.4), 1.9, 1.9,
                                 facecolor='#fcf3cf' if j == 3 else '#f4f6f7', edgecolor=col, lw=2.0)
        ax.add_patch(rect)
    ax.text(x_out + 1.55, 0.5, "Output Feature Maps\n($C_{\\mathrm{OUT}} = 4$)", ha='center', fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    _save_figure(fig, "Figure_10_7", save_dir)
    return fig


def generate_figure_10_8(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.8: Illustration of max-pooling.
    Showing 4x4 input mapped to 2x2 output via 2x2 max filter with stride 2.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis('off')
    ax.set_title("Figure 10.8: Illustration of Max-Pooling ($2 \\times 2$ Window, Stride 2)",
                 fontsize=11, fontweight='bold', pad=12)

    # 4x4 input feature map values
    input_vals = [
        [1.2, 3.4, 0.8, 2.1],
        [4.5, 2.0, 1.9, 3.8],
        [0.3, 1.7, 5.2, 4.1],
        [2.8, 3.1, 2.4, 0.9],
    ]
    # Max in each 2x2 quadrant:
    # Top-left: max(1.2, 3.4, 4.5, 2.0) = 4.5
    # Top-right: max(0.8, 2.1, 1.9, 3.8) = 3.8
    # Bottom-left: max(0.3, 1.7, 2.8, 3.1) = 3.1
    # Bottom-right: max(5.2, 4.1, 2.4, 0.9) = 5.2

    quad_colors = [
        ('#fadbd8', '#e74c3c'),  # Red quad (top-left)
        ('#d4e6f1', '#2980b9'),  # Blue quad (top-right)
        ('#d5f5e3', '#27ae60'),  # Green quad (bottom-left)
        ('#fdebd0', '#d35400'),  # Orange quad (bottom-right)
    ]

    x0, y0 = 0.6, 0.8
    box_sz = 0.9

    for r in range(4):
        for c in range(4):
            q_idx = (0 if (r < 2 and c < 2) else
                     1 if (r < 2 and c >= 2) else
                     2 if (r >= 2 and c < 2) else 3)
            bg, border = quad_colors[q_idx]
            val = input_vals[r][c]
            # Is this the max element in the quadrant?
            is_max = ((r, c) in [(1, 0), (1, 3), (3, 1), (2, 2)])
            lw = 2.2 if is_max else 1.0
            rect = patches.Rectangle((x0 + c * box_sz, y0 + (3 - r) * box_sz), box_sz, box_sz,
                                     facecolor=bg, edgecolor=border, lw=lw)
            ax.add_patch(rect)
            ax.text(x0 + (c + 0.5) * box_sz, y0 + (3.5 - r) * box_sz, f"{val:.1f}",
                    ha='center', va='center', fontsize=10.5,
                    fontweight='bold' if is_max else 'normal',
                    color='#900c3f' if is_max else '#2c3e50')

    ax.text(x0 + 2 * box_sz, y0 - 0.45, "Input Feature Map ($4 \\times 4$)",
            ha='center', fontsize=10.5, fontweight='bold')

    # Arrow with "Max-Pool 2x2"
    ax.annotate("", xy=(6.5, y0 + 1.8), xytext=(4.7, y0 + 1.8),
                arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=2.5))
    ax.text(5.6, y0 + 2.15, "Max-Pool\n$2 \\times 2$, Stride 2", ha='center', fontsize=9.5, fontweight='bold')

    # 2x2 pooled output map
    ox0, oy0 = 7.2, 1.4
    out_sz = 1.3
    out_vals = [[4.5, 3.8],
                [3.1, 5.2]]

    for r in range(2):
        for c in range(2):
            q_idx = r * 2 + c
            bg, border = quad_colors[q_idx]
            val = out_vals[r][c]
            rect = patches.Rectangle((ox0 + c * out_sz, oy0 + (1 - r) * out_sz), out_sz, out_sz,
                                     facecolor=bg, edgecolor=border, lw=2.5)
            ax.add_patch(rect)
            ax.text(ox0 + (c + 0.5) * out_sz, oy0 + (1.5 - r) * out_sz, f"{val:.1f}",
                    ha='center', va='center', fontsize=14, fontweight='bold', color=border)

    ax.text(ox0 + out_sz, oy0 - 0.5, "Pooled Output ($2 \\times 2$)",
            ha='center', fontsize=10.5, fontweight='bold')

    plt.tight_layout()
    _save_figure(fig, "Figure_10_8", save_dir)
    return fig


def generate_figure_10_9(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.9: Effective receptive field growth with depth in a multilayer CNN.
    Faithful reproduction of page 315 showing 1 output unit -> 3 middle units -> 5 input units.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 4.0)
    ax.axis('off')
    ax.set_title("Figure 10.9: Growth of Effective Receptive Field with Depth",
                 fontsize=11, fontweight='bold', pad=12)

    # 3 Layers:
    # Input Layer (y = 0.5): 7 units (indices 0..6)
    # Middle Layer (y = 1.8): 5 units (indices 1..5)
    # Output Layer (y = 3.1): 3 units (indices 2..4), with center unit (idx 3) highlighted red

    y_in = 0.5
    y_mid = 1.8
    y_out = 3.1

    # Draw Input Units
    x_in = np.arange(1, 8)
    for idx, x in enumerate(x_in):
        is_rf = (2 <= idx <= 6)  # 5 units in receptive field
        fc = '#fadbd8' if is_rf else '#eaeded'
        ec = '#e74c3c' if is_rf else '#7f8c8d'
        c = patches.Circle((x, y_in), 0.22, facecolor=fc, edgecolor=ec, lw=1.5, zorder=5)
        ax.add_patch(c)
        ax.text(x, y_in, f"{idx}", ha='center', va='center', fontsize=8.5, zorder=6)

    ax.text(0.3, y_in, "Input Layer\n(5 units in RF)", ha='right', va='center', fontsize=9.0, fontweight='bold')

    # Draw Middle Units
    x_mid = np.arange(2, 7)
    for idx, x in enumerate(x_mid):
        is_rf = (1 <= idx <= 3)  # 3 units in RF
        fc = '#fadbd8' if is_rf else '#eaeded'
        ec = '#e74c3c' if is_rf else '#7f8c8d'
        c = patches.Circle((x, y_mid), 0.22, facecolor=fc, edgecolor=ec, lw=1.5, zorder=5)
        ax.add_patch(c)

    ax.text(0.3, y_mid, "Middle Layer\n(3 units in RF)", ha='right', va='center', fontsize=9.0, fontweight='bold')

    # Draw Output Units
    x_out = np.arange(3, 6)
    for idx, x in enumerate(x_out):
        is_center = (idx == 1)  # Center unit
        fc = '#e74c3c' if is_center else '#eaeded'
        ec = '#922b21' if is_center else '#7f8c8d'
        c = patches.Circle((x, y_out), 0.24, facecolor=fc, edgecolor=ec, lw=2.0, zorder=5)
        ax.add_patch(c)
        if is_center:
            ax.text(x, y_out, "RF", color='white', ha='center', va='center', fontsize=9.0, fontweight='bold', zorder=6)

    ax.text(0.3, y_out, "Output Layer\n(1 target unit)", ha='right', va='center', fontsize=9.0, fontweight='bold')

    # Draw receptive field connections
    # Center output unit (x=4) connects to middle units at x=3, 4, 5
    for xm in [3, 4, 5]:
        ax.plot([4, xm], [y_out - 0.24, y_mid + 0.22], color='#e74c3c', lw=1.8, alpha=0.85)

    # Middle units at x=3, 4, 5 connect to their respective 3 inputs
    for xm in [3, 4, 5]:
        for xi in [xm - 1, xm, xm + 1]:
            ax.plot([xm, xi], [y_mid - 0.22, y_in + 0.22], color='#e74c3c', lw=1.2, alpha=0.6)

    # Bracket showing 5-unit receptive field at input
    bracket_x = [2.0, 6.0]
    ax.annotate("", xy=(bracket_x[0], y_in - 0.35), xytext=(bracket_x[1], y_in - 0.35),
                arrowprops=dict(arrowstyle="<->", color='#c0392b', lw=1.8))
    ax.text(4.0, y_in - 0.58, "Effective Receptive Field = 5 Units ($R_2 = 1 + 2 + 2 = 5$)",
            ha='center', fontsize=9.5, fontweight='bold', color='#c0392b')

    plt.tight_layout()
    _save_figure(fig, "Figure_10_9", save_dir)
    return fig


def generate_figure_10_10(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.10: The architecture of a typical convolutional network: VGG-16.
    Faithful reproduction of page 317 diagram.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.set_xlim(-0.5, 12.0)
    ax.set_ylim(-0.5, 5.0)
    ax.axis('off')
    ax.set_title("Figure 10.10: VGG-16 Deep Convolutional Network Architecture",
                 fontsize=12, fontweight='bold', pad=12)

    # Pipeline blocks: (name, label, width, height, color, desc)
    stages = [
        ("Input", "224×224×3", 0.5, 3.8, '#d6eaf8', "Input"),
        ("Block 1", "224×224×64\n(2× conv)", 0.7, 3.8, '#fadbd8', "Conv 64"),
        ("Pool 1", "112×112×64", 0.4, 2.7, '#f9e79f', "Pool /2"),
        ("Block 2", "112×112×128\n(2× conv)", 0.7, 2.7, '#fadbd8', "Conv 128"),
        ("Pool 2", "56×56×128", 0.4, 1.9, '#f9e79f', "Pool /2"),
        ("Block 3", "56×56×256\n(3× conv)", 0.8, 1.9, '#fadbd8', "Conv 256"),
        ("Pool 3", "28×28×256", 0.4, 1.3, '#f9e79f', "Pool /2"),
        ("Block 4", "28×28×512\n(3× conv)", 0.9, 1.3, '#fadbd8', "Conv 512"),
        ("Pool 4", "14×14×512", 0.4, 0.9, '#f9e79f', "Pool /2"),
        ("Block 5", "14×14×512\n(3× conv)", 0.9, 0.9, '#fadbd8', "Conv 512"),
        ("Pool 5", "7×7×512", 0.4, 0.6, '#f9e79f', "Pool /2"),
        ("FC 1", "4096", 0.5, 2.5, '#d5f5e3', "FC 4096"),
        ("FC 2", "4096", 0.5, 2.5, '#d5f5e3', "FC 4096"),
        ("Output", "1000", 0.4, 1.5, '#e8daef', "Softmax\n1000"),
    ]

    curr_x = 0.2
    gap = 0.22

    for name, dim_lbl, w, h, col, desc in stages:
        y_bottom = 2.2 - h / 2.0
        rect = patches.Rectangle((curr_x, y_bottom), w, h, facecolor=col, edgecolor='#2c3e50', lw=1.2)
        ax.add_patch(rect)
        # Label above or below
        ax.text(curr_x + w / 2.0, y_bottom + h + 0.15, desc, ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        ax.text(curr_x + w / 2.0, y_bottom - 0.2, dim_lbl, ha='center', va='top', fontsize=7.0, color='#34495e')
        curr_x += w + gap

    # Legend at bottom
    leg_patches = [
        patches.Patch(facecolor='#fadbd8', edgecolor='#2c3e50', label='Convolution (3×3, Stride 1, Same Pad) + ReLU'),
        patches.Patch(facecolor='#f9e79f', edgecolor='#2c3e50', label='Max Pooling (2×2, Stride 2)'),
        patches.Patch(facecolor='#d5f5e3', edgecolor='#2c3e50', label='Fully Connected + ReLU'),
        patches.Patch(facecolor='#e8daef', edgecolor='#2c3e50', label='Output Softmax (1000 Classes)'),
    ]
    ax.legend(handles=leg_patches, loc='lower center', bbox_to_anchor=(0.5, -0.22),
              ncol=4, frameon=True, fontsize=8.5)

    plt.tight_layout()
    _save_figure(fig, "Figure_10_10", save_dir)
    return fig
