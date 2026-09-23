"""
common/style_transfer.py
========================
Section 10.6: Style Transfer
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements:
1. Content Loss (Eq 10.14):
   - Sum of squared differences between feature activations of generated image G
     and content image C:
     E_content(G, C) = 0.5 * sum_{i,j,k} (a_{ijk}(G) - a_{ijk}(C))^2
2. Gram Matrix & Feature Correlation (Eq 10.15):
   - Spatial inner product of activation feature maps:
     F_{kk'}(G) = sum_{i,j} a_{ijk}(G) a_{ijk'}(G)
3. Style Loss (Eq 10.16 & Eq 10.17):
   - Normalized squared Frobenius norm difference between Gram matrices:
     E_style^{(l)}(G, S) = (1 / (4 * H_l^2 * W_l^2 * K_l^2)) * sum_{k, k'} (F_{kk'}(G) - F_{kk'}(S))^2
     E_style(G, S) = sum_l lambda_l * E_style^{(l)}(G, S)
4. Total Loss & Optimization (Eq 10.13):
   - E(G) = alpha * E_content(G, C) + beta * E_style(G, S)
   - Gradient computation with respect to generated image pixels x_G
5. High-Resolution Figure Reproduction:
   - Figure 10.32: Neural Style Transfer (Content image C, Style image S, and Stylized output G).
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
# Mathematical Formulations (Eqs 10.13 - 10.17)
# =============================================================================

def compute_content_loss(act_G: np.ndarray, act_C: np.ndarray) -> float:
    """
    Equation 10.14: Content loss between generated feature map G and content C.
    E_content(G, C) = 0.5 * sum_{i,j,k} (a_{ijk}(G) - a_{ijk}(C))^2
    Args:
        act_G: (H, W, K) feature activations of generated image.
        act_C: (H, W, K) feature activations of content image.
    Returns:
        Scalar content loss.
    """
    diff = act_G - act_C
    return float(0.5 * np.sum(diff ** 2))


def compute_content_gradient(act_G: np.ndarray, act_C: np.ndarray) -> np.ndarray:
    """
    Derivative of content loss with respect to pre-activations a_{ijk}(G):
    d(E_content) / d(a_{ijk}(G)) = a_{ijk}(G) - a_{ijk}(C)
    """
    return act_G - act_C


def compute_gram_matrix(act: np.ndarray) -> np.ndarray:
    """
    Equation 10.15: Gram matrix of unnormalized cross-correlations between feature maps.
    F_{kk'}(G) = sum_{i=1}^H sum_{j=1}^W a_{ijk}(G) a_{ijk'}(G)
    Args:
        act: (H, W, K) feature activation tensor.
    Returns:
        F: (K, K) symmetric Gram matrix.
    """
    H, W, K = act.shape
    # Reshape (H, W, K) -> (H * W, K)
    A = act.reshape(-1, K)
    # F = A^T @ A
    return A.T @ A


def compute_style_loss_layer(act_G: np.ndarray, act_S: np.ndarray) -> float:
    """
    Equation 10.16: Style loss for a single layer l.
    E_style^{(l)}(G, S) = 1 / (4 * H^2 * W^2 * K^2) * sum_{k,k'} (F_{kk'}(G) - F_{kk'}(S))^2
    Args:
        act_G: (H, W, K) activations of generated image at layer l.
        act_S: (H, W, K) activations of style image at layer l.
    Returns:
        Scalar style loss for layer l.
    """
    H, W, K = act_G.shape
    F_G = compute_gram_matrix(act_G)
    F_S = compute_gram_matrix(act_S)
    diff = F_G - F_S
    norm_factor = 4.0 * (H ** 2) * (W ** 2) * (K ** 2)
    return float(np.sum(diff ** 2) / norm_factor)


def compute_style_gradient_layer(act_G: np.ndarray, act_S: np.ndarray) -> np.ndarray:
    """
    Derivative of style loss for layer l with respect to pre-activations a_{ijk}(G):
    d(E_style^{(l)}) / d(a_{ijk}(G)) = 1 / (H^2 * W^2 * K^2) * sum_{k'} (F_{kk'}(G) - F_{kk'}(S)) * a_{ijk'}(G)
    """
    H, W, K = act_G.shape
    F_G = compute_gram_matrix(act_G)
    F_S = compute_gram_matrix(act_S)
    diff = F_G - F_S  # (K, K)
    norm_factor = (H ** 2) * (W ** 2) * (K ** 2)

    A_G = act_G.reshape(-1, K)  # (HW, K)
    # grad_A = (A_G @ diff.T) / norm_factor
    grad_A = (A_G @ diff) / norm_factor
    return grad_A.reshape(H, W, K)


def compute_total_loss(
    act_G_content: np.ndarray,
    act_C_content: np.ndarray,
    layers_G_style: List[np.ndarray],
    layers_S_style: List[np.ndarray],
    alpha: float = 1.0,
    beta: float = 1000.0,
    style_weights: Optional[List[float]] = None,
) -> Tuple[float, float, float]:
    """
    Equation 10.13 & Eq 10.17: Total Style Transfer Objective.
    E(G) = alpha * E_content(G, C) + beta * sum_l lambda_l E_style^{(l)}(G, S)
    Returns:
        (total_loss, content_loss, weighted_style_loss)
    """
    e_content = compute_content_loss(act_G_content, act_C_content)

    num_style_layers = len(layers_G_style)
    if style_weights is None:
        style_weights = [1.0 / num_style_layers] * num_style_layers

    e_style = 0.0
    for l_idx in range(num_style_layers):
        l_loss = compute_style_loss_layer(layers_G_style[l_idx], layers_S_style[l_idx])
        e_style += style_weights[l_idx] * l_loss

    total = alpha * e_content + beta * e_style
    return total, e_content, e_style


# =============================================================================
# Synthetic Artistic Image Synthesizer & Figure 10.32 Reproduction
# =============================================================================

def create_synthetic_content_image(H: int = 120, W: int = 160) -> np.ndarray:
    """
    Create a clean photographic architectural scene (bridge / building + sky).
    High structural content, crisp straight lines and geometric silhouettes.
    """
    img = np.zeros((H, W, 3), dtype=np.float64)
    h_mid = H // 2

    # Sky gradient (top half)
    if h_mid > 0:
        for i in range(h_mid):
            img[i, :, :] = [0.65 - 0.15 * (i / h_mid), 0.75 - 0.1 * (i / h_mid), 0.9]

    # Water reflection (bottom half)
    for i in range(h_mid, H):
        denom = max(1, H - h_mid)
        factor = (i - h_mid) / denom
        img[i, :, :] = [0.2 + 0.1 * factor, 0.35 + 0.1 * factor, 0.5 + 0.15 * factor]

    # Normalized relative dimensions
    # Left tower
    t1_x1, t1_x2 = int(0.16 * W), int(0.28 * W)
    t1_y1, t1_y2 = int(0.20 * H), min(H, h_mid + int(0.08 * H))
    img[t1_y1:t1_y2, t1_x1:t1_x2, :] = [0.82, 0.78, 0.72]

    # Left spire
    c1_mid = (t1_x1 + t1_x2) // 2
    for r in range(int(0.10 * H), t1_y1):
        w_spire = max(1, int((t1_x2 - t1_x1) * 0.5 * (r - int(0.10 * H)) / max(1, (t1_y1 - int(0.10 * H)))))
        c_left = max(0, c1_mid - w_spire)
        c_right = min(W, c1_mid + w_spire + 1)
        img[r, c_left:c_right, :] = [0.3, 0.3, 0.35]

    # Right tower
    t2_x1, t2_x2 = int(0.72 * W), int(0.84 * W)
    t2_y1, t2_y2 = int(0.20 * H), min(H, h_mid + int(0.08 * H))
    img[t2_y1:t2_y2, t2_x1:t2_x2, :] = [0.82, 0.78, 0.72]

    # Right spire
    c2_mid = (t2_x1 + t2_x2) // 2
    for r in range(int(0.10 * H), t2_y1):
        w_spire = max(1, int((t2_x2 - t2_x1) * 0.5 * (r - int(0.10 * H)) / max(1, (t2_y1 - int(0.10 * H)))))
        c_left = max(0, c2_mid - w_spire)
        c_right = min(W, c2_mid + w_spire + 1)
        img[r, c_left:c_right, :] = [0.3, 0.3, 0.35]

    # Suspension bridge deck
    d_y1 = max(0, h_mid - max(1, int(0.02 * H)))
    d_y2 = min(H, h_mid + max(2, int(0.04 * H)))
    d_x1 = int(0.10 * W)
    d_x2 = int(0.90 * W)
    img[d_y1:d_y2, d_x1:d_x2, :] = [0.25, 0.25, 0.28]

    # Parabolic suspension cable between towers
    c_cable_start = c1_mid
    c_cable_end = c2_mid
    c_cable_mid = (c1_mid + c2_mid) / 2.0
    half_span = max(1.0, (c2_mid - c1_mid) / 2.0)
    for c in range(c_cable_start, min(W, c_cable_end + 1)):
        norm_c = (c - c_cable_mid) / half_span
        r_cable = int(0.20 * H + 0.24 * H * (norm_c ** 2))
        if 0 <= r_cable < H:
            r_top = min(H, r_cable + max(1, int(0.015 * H)))
            img[r_cable:r_top, c, :] = [0.2, 0.2, 0.25]
            if c % max(2, int(0.04 * W)) == 0 and r_cable < h_mid:
                img[r_cable:h_mid, c, :] = [0.3, 0.3, 0.32]

    # Add gentle realistic noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.015, img.shape)
    return np.clip(img + noise, 0.0, 1.0)



def create_synthetic_style_image(H: int = 120, W: int = 160) -> np.ndarray:
    """
    Create a vibrant Van Gogh 'Starry Night' style swirling artistic image.
    Rich golden-yellow spirals, swirling deep indigo and cobalt blues, expressive brushstrokes.
    """
    img = np.zeros((H, W, 3), dtype=np.float64)
    y, x = np.ogrid[:H, :W]

    # Swirling vortex pattern
    center_y, center_x = H * 0.45, W * 0.45
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    theta = np.arctan2(y - center_y, x - center_x)

    # Coils / spirals
    spiral1 = np.sin(0.35 * r - 4.0 * theta)
    spiral2 = np.cos(0.2 * r + 2.0 * theta)

    # Swirling blues
    img[:, :, 0] = 0.1 + 0.25 * np.sin(0.15 * r) + 0.1 * spiral1
    img[:, :, 1] = 0.25 + 0.35 * np.cos(0.12 * r) + 0.15 * spiral2
    img[:, :, 2] = 0.65 + 0.3 * np.sin(0.08 * r + theta)

    # Golden stars & crescent moon
    star1 = np.exp(-((x - W * 0.78) ** 2 + (y - H * 0.25) ** 2) / 120.0)
    star2 = np.exp(-((x - W * 0.22) ** 2 + (y - H * 0.22) ** 2) / 90.0)
    stars = star1 + star2
    img[:, :, 0] += 1.2 * stars
    img[:, :, 1] += 0.95 * stars
    img[:, :, 2] += 0.2 * stars

    # Impasto brushstroke texture modulation
    brush = (np.sin(x * 1.2 + y * 0.8) * np.cos(x * 0.7 - y * 1.1)) * 0.12
    for c in range(3):
        img[:, :, c] += brush

    return np.clip(img, 0.0, 1.0)


def create_stylized_image(content: np.ndarray, style: np.ndarray) -> np.ndarray:
    """
    Simulate the stylized output image G combining the sharp edges and structural content
    of C with the color palette, brushstrokes, and swirling texture of S.
    """
    H, W, _ = content.shape
    # Extract luminance / structural edges from content
    lum_c = 0.299 * content[:, :, 0] + 0.587 * content[:, :, 1] + 0.114 * content[:, :, 2]
    # Edge gradient
    gy, gx = np.gradient(lum_c)
    edges = np.sqrt(gx ** 2 + gy ** 2)
    edges = np.clip(edges * 5.0, 0.0, 1.0)

    # Stylized synthesis:
    # 1. Base color palette from style image
    base = style.copy()

    # 2. Modulate by content structure (contrast & lighting)
    # Brightness modulation
    lum_mod = (lum_c - np.mean(lum_c)) * 0.65
    for c in range(3):
        base[:, :, c] += lum_mod

    # 3. Preserve strong structural contours (spires, bridge deck, horizon)
    edge_mask = edges[:, :, np.newaxis]
    stylized = (1.0 - edge_mask * 0.8) * base + (edge_mask * 0.8) * content

    # 4. Swirl enhancement in smooth sky regions
    sky_mask = (lum_c > 0.5) & (edges < 0.15)
    for c in range(3):
        stylized[sky_mask, c] = 0.3 * content[sky_mask, c] + 0.7 * style[sky_mask, c]

    return np.clip(stylized, 0.0, 1.0)


def generate_figure_10_32(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.32: Neural Style Transfer (Gatys et al., 2015).
    Left: Content image C (photograph of bridge / architecture).
    Middle: Style image S (Van Gogh artistic painting with vibrant swirling strokes).
    Right: Generated image G (synthesized image matching content of C and style of S).
    Bottom panel: Gram matrix cross-correlation representation (Eq 10.15).
    """
    setup_style()
    H, W = 140, 180
    content = create_synthetic_content_image(H, W)
    style = create_synthetic_style_image(H, W)
    stylized = create_stylized_image(content, style)

    # Gram matrix simulation for bottom panel
    K = 16
    rng = np.random.default_rng(106)
    # Content feature Gram matrix: diagonal-heavy, sparse correlations
    A_c = rng.normal(0, 0.5, (100, K))
    gram_c = (A_c.T @ A_c) / 100.0
    # Style feature Gram matrix: rich off-diagonal cross-feature correlations
    A_s = rng.normal(0, 0.5, (100, K))
    v = rng.uniform(0.3, 0.9, K)
    gram_s = (A_s.T @ A_s) / 100.0 + np.outer(v, v)
    gram_s = gram_s / np.max(np.abs(gram_s))

    fig = plt.figure(figsize=(11.5, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 0.85], hspace=0.32, wspace=0.25)

    # Top Row: Images C, S, G
    ax_c = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_g = fig.add_subplot(gs[0, 2])

    ax_c.imshow(content)
    ax_c.set_title("Content Image $C$\n(Photographic Scene)", fontsize=10.5, fontweight='bold')
    ax_c.axis('off')

    ax_s.imshow(style)
    ax_s.set_title("Style Image $S$\n(Artistic Painting)", fontsize=10.5, fontweight='bold')
    ax_s.axis('off')

    ax_g.imshow(stylized)
    ax_g.set_title("Generated Image $G$\n(Neural Style Transfer)", fontsize=10.5, fontweight='bold', color='#922b21')
    ax_g.axis('off')

    # Bottom Row: Mathematical and Gram matrix representation
    ax_gram_c = fig.add_subplot(gs[1, 0])
    ax_diag = fig.add_subplot(gs[1, 1])
    ax_gram_s = fig.add_subplot(gs[1, 2])

    # Left bottom: Content Gram matrix
    im_gc = ax_gram_c.imshow(gram_c, cmap='coolwarm', vmin=-0.8, vmax=0.8)
    ax_gram_c.set_title("Content Gram Matrix $F(C)$\n(Localized Co-occurrence)", fontsize=9.0, fontweight='bold')
    ax_gram_c.set_xticks([])
    ax_gram_c.set_yticks([])
    plt.colorbar(im_gc, ax=ax_gram_c, fraction=0.046, pad=0.04)

    # Middle bottom: Optimization Flow Diagram
    ax_diag.axis('off')
    ax_diag.text(0.5, 0.85, "Total Loss: $E(G) = \\alpha E_{\\mathrm{content}} + \\beta E_{\\mathrm{style}}$",
                 ha='center', va='center', fontsize=9.5, fontweight='bold', color='#2c3e50')
    ax_diag.text(0.5, 0.55, "$E_{\\mathrm{content}}(G, C) = \\frac{1}{2} \\sum_{ijk} (a_{ijk}(G) - a_{ijk}(C))^2$",
                 ha='center', va='center', fontsize=8.5, color='#1b4f72')
    ax_diag.text(0.5, 0.25, "$E_{\\mathrm{style}}(G, S) = \\sum_l \\frac{\\lambda_l}{4 H_l^2 W_l^2 K_l^2} \\|F(G) - F(S)\\|^2_F$",
                 ha='center', va='center', fontsize=8.5, color='#922b21')
    ax_diag.text(0.5, 0.02, "Optimize pixel values $\\mathbf{x}_G$ via gradient descent (weights $\\mathbf{w}$ fixed)",
                 ha='center', va='center', fontsize=8.0, style='italic', color='#7f8c8d')

    # Right bottom: Style Gram matrix
    im_gs = ax_gram_s.imshow(gram_s, cmap='magma', vmin=0.0, vmax=1.0)
    ax_gram_s.set_title("Style Gram Matrix $F(S)$\n(Global Texture & Color Signature)", fontsize=9.0, fontweight='bold')
    ax_gram_s.set_xticks([])
    ax_gram_s.set_yticks([])
    plt.colorbar(im_gs, ax=ax_gram_s, fraction=0.046, pad=0.04)

    plt.suptitle("Figure 10.32: Neural Style Transfer and Gram Matrix Correlation Matching",
                 fontsize=12.5, fontweight='bold', y=0.98)
    _save_figure(fig, "Figure_10_32", save_dir)
    return fig


if __name__ == "__main__":
    fig = generate_figure_10_32()
    print("Figure 10.32 generated successfully!")
