"""
common/object_detection.py
==========================
Section 10.4: Object Detection
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements:
1. Bounding Box Representations & Metrics (Section 10.4.1 & 10.4.2):
   - Continuous and pixel coordinate systems (x_c, y_c, w, h) and (x1, y1, x2, y2).
   - Exact Intersection-over-Union (IoU) calculation (Figure 10.20).
2. Sliding Window Analysis & Convolutional Acceleration (Section 10.4.3, Exercise 10.12):
   - Computational complexity (MACs) for naive repeated evaluations vs fully convolutional sliding windows.
   - Network architectures of Figures 10.22 and 10.23.
3. Multi-Scale Detection & Coordinate Transformations (Section 10.4.4, Figure 10.24).
4. Non-Maximum Suppression (NMS, Section 10.4.5, Figure 10.25):
   - Confidence thresholding and greedy IoU suppression.
5. Fast Region Proposal Principles (Section 10.4.6, R-CNN, Fast R-CNN, Faster R-CNN).
6. High-Resolution Figure Reproductions (Figures 10.19 - 10.25).
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
# 10.4.1 & 10.4.2 Bounding Box & IoU
# =============================================================================
class BoundingBox:
    """
    Representation of 2D bounding boxes and geometric operations.
    Coordinates can be:
    - [x1, y1, x2, y2]: top-left (x1, y1) and bottom-right (x2, y2)
    - [xc, yc, w, h]: center (xc, yc), width w, height h
    """

    @staticmethod
    def center_to_corners(box: np.ndarray) -> np.ndarray:
        """Convert [xc, yc, w, h] to [x1, y1, x2, y2]."""
        xc, yc, w, h = box[0], box[1], box[2], box[3]
        return np.array([xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0])

    @staticmethod
    def corners_to_center(box: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [xc, yc, w, h]."""
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1])

    @staticmethod
    def compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
        """
        Compute Intersection over Union (IoU) between two bounding boxes in [x1, y1, x2, y2] format.
        IoU = Area(A n B) / Area(A u B).
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0.0, xB - xA)
        inter_h = max(0.0, yB - yA)
        inter_area = inter_w * inter_h

        areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

        union_area = areaA + areaB - inter_area
        if union_area <= 0:
            return 0.0
        return float(inter_area / union_area)


# =============================================================================
# 10.4.3 Sliding Window Computational Accounting (Exercise 10.12)
# =============================================================================
def analyze_sliding_window_efficiency() -> Dict[str, Any]:
    """
    Calculate computational complexity for Figure 10.22 vs Figure 10.23 (Exercise 10.12).
    Model 1 (Figure 10.22):
    Input: 6x6 image
    Conv: 3x3 filter (valid, stride 1) -> 4x4 feature map (16 units, 9 mults each = 144 mults)
    Pool: 2x2 pool (stride 2) -> 2x2 feature map (4 units, 0 mults)
    FC / Conv: 2x2 filter -> 1 output unit (4 mults)
    Total mults for 1 pass on 6x6: 144 + 4 = 148 mults.

    Naive Sliding Window on 8x8 image:
    Number of 6x6 window positions on 8x8 image (stride 1): (8 - 6 + 1)^2 = 3^2 = 9 positions.
    Total naive mults: 9 * 148 = 1332 mults.

    Model 2 (Figure 10.23, Expanded Convolutional Implementation on 8x8 image):
    Conv layer: 8x8 input with 3x3 filter valid stride 1 -> 6x6 feature map (36 units * 9 = 324 mults)
    Pool layer: 6x6 map with 2x2 pool stride 2 -> 3x3 feature map (9 units, 0 mults)
    FC / Conv layer: 3x3 map with 2x2 filter valid stride 1 -> 2x2 output (4 units * 4 = 16 mults)
    Total convolutional mults: 324 + 16 = 340 mults.

    Speedup ratio: 1332 / 340 = 3.9176x.
    """
    single_pass_conv = 16 * 9
    single_pass_fc = 4
    single_pass_total = single_pass_conv + single_pass_fc

    num_windows = 9
    naive_total = num_windows * single_pass_total

    conv_expanded_conv = 36 * 9
    conv_expanded_fc = 4 * 4
    conv_expanded_total = conv_expanded_conv + conv_expanded_fc

    speedup = naive_total / conv_expanded_total

    return {
        "single_pass_total": single_pass_total,
        "num_windows": num_windows,
        "naive_total_multiplications": naive_total,
        "conv_expanded_total_multiplications": conv_expanded_total,
        "speedup_ratio": speedup,
    }


# =============================================================================
# 10.4.5 Non-Maximum Suppression (NMS)
# =============================================================================
class NonMaxSuppression:
    """
    Greedy Non-Maximum Suppression algorithm (Section 10.4.5).
    """

    @staticmethod
    def apply(
        boxes: np.ndarray,
        scores: np.ndarray,
        score_threshold: float = 0.7,
        iou_threshold: float = 0.5,
    ) -> List[int]:
        """
        Args:
            boxes: (N, 4) array in [x1, y1, x2, y2] format.
            scores: (N,) array of detection probabilities.
            score_threshold: minimum confidence score.
            iou_threshold: IoU overlap threshold for suppression.
        Returns:
            keep_indices: list of selected bounding box indices.
        """
        # Step 1: Filter out low-confidence detections
        valid_indices = np.where(scores >= score_threshold)[0]
        if len(valid_indices) == 0:
            return []

        # Sort remaining boxes by score in descending order
        sorted_order = valid_indices[np.argsort(-scores[valid_indices])]
        keep = []

        while len(sorted_order) > 0:
            best_idx = sorted_order[0]
            keep.append(int(best_idx))
            if len(sorted_order) == 1:
                break

            remaining_indices = sorted_order[1:]
            ious = np.array([BoundingBox.compute_iou(boxes[best_idx], boxes[i]) for i in remaining_indices])

            # Keep only boxes whose IoU with best_idx is <= iou_threshold
            survivors = remaining_indices[ious <= iou_threshold]
            sorted_order = survivors

        return keep


# =============================================================================
# High-Resolution Figure Reproductions (Figures 10.19 - 10.25)
# =============================================================================
def generate_figure_10_19(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.19: Bounding boxes around objects from different classes.
    Blue: Car, Red: Pedestrian, Orange: Traffic Light.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Draw synthetic road scene background
    # Sky
    sky = patches.Rectangle((0, 3.5), 10, 2.5, facecolor='#d6eaf8', edgecolor='none')
    ax.add_patch(sky)
    # Buildings
    b1 = patches.Rectangle((0.5, 2.2), 2.5, 2.8, facecolor='#bdc3c7', edgecolor='#7f8c8d')
    b2 = patches.Rectangle((6.8, 2.2), 2.8, 2.8, facecolor='#a6acaf', edgecolor='#7f8c8d')
    ax.add_patch(b1)
    ax.add_patch(b2)
    # Road
    road = patches.Rectangle((0, 0), 10, 2.5, facecolor='#566573', edgecolor='none')
    ax.add_patch(road)
    # Road markings
    for mx in [1.5, 4.5, 7.5]:
        ax.plot([mx, mx + 1.2], [1.2, 1.2], color='white', lw=3.0, linestyle='--')

    # Bounding Boxes:
    # 1. Cars (Blue)
    car_boxes = [
        (3.2, 0.6, 2.8, 1.4, "car 0.94"),
        (0.8, 0.4, 2.0, 1.1, "car 0.88"),
    ]
    for x, y, w, h, lbl in car_boxes:
        rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor='#2980b9', lw=2.4)
        ax.add_patch(rect)
        ax.text(x, y + h + 0.1, lbl, color='white', fontsize=8.5, fontweight='bold',
                bbox=dict(boxstyle='square,pad=0.15', facecolor='#2980b9', edgecolor='none'))

    # 2. Pedestrians (Red)
    ped_boxes = [
        (6.5, 1.4, 0.8, 2.0, "pedestrian 0.91"),
        (7.8, 1.5, 0.7, 1.9, "pedestrian 0.85"),
    ]
    for x, y, w, h, lbl in ped_boxes:
        rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor='#c0392b', lw=2.4)
        ax.add_patch(rect)
        ax.text(x, y + h + 0.1, lbl, color='white', fontsize=8.0, fontweight='bold',
                bbox=dict(boxstyle='square,pad=0.15', facecolor='#c0392b', edgecolor='none'))

    # 3. Traffic Light (Orange)
    tl_box = (2.8, 3.6, 0.6, 1.6, "traffic light 0.96")
    rect = patches.Rectangle((tl_box[0], tl_box[1]), tl_box[2], tl_box[3], fill=False, edgecolor='#e67e22', lw=2.4)
    ax.add_patch(rect)
    ax.text(tl_box[0] - 0.4, tl_box[1] + tl_box[3] + 0.1, tl_box[4], color='white', fontsize=8.0, fontweight='bold',
            bbox=dict(boxstyle='square,pad=0.15', facecolor='#e67e22', edgecolor='none'))

    ax.set_title("Figure 10.19: Multi-Class Object Detection with Bounding Boxes",
                 fontsize=12, fontweight='bold', pad=12)

    plt.tight_layout()
    _save_figure(fig, "Figure_10_19", save_dir)
    return fig


def generate_figure_10_20(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.20: Illustration of Intersection-over-Union (IoU).
    Left: Area of Intersection (green)
    Right: Area of Union (green)
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.0))

    for ax in [ax1, ax2]:
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, 4.5)
        ax.set_aspect('equal')
        ax.axis('off')

    # Box A (Blue: Predicted)
    bA = [0.8, 0.8, 3.8, 3.2]  # [x1, y1, x2, y2]
    # Box B (Red: Ground Truth)
    bB = [2.0, 1.5, 4.8, 3.8]

    # --- Left: Area of Intersection ---
    ax1.set_title("Area of Intersection $(B \\cap G)$", fontsize=11, fontweight='bold', pad=10)
    # Blue box outline
    ax1.add_patch(patches.Rectangle((bA[0], bA[1]), bA[2] - bA[0], bA[3] - bA[1],
                                    fill=False, edgecolor='#2980b9', lw=2.2, label='Predicted (B)'))
    # Red box outline
    ax1.add_patch(patches.Rectangle((bB[0], bB[1]), bB[2] - bB[0], bB[3] - bB[1],
                                    fill=False, edgecolor='#c0392b', lw=2.2, label='Ground Truth (G)'))
    # Intersection fill (Green)
    ix1, iy1 = max(bA[0], bB[0]), max(bA[1], bB[1])
    ix2, iy2 = min(bA[2], bB[2]), min(bA[3], bB[3])
    ax1.add_patch(patches.Rectangle((ix1, iy1), ix2 - ix1, iy2 - iy1,
                                    facecolor='#2ecc71', edgecolor='#27ae60', lw=1.5, alpha=0.85))
    ax1.text((ix1 + ix2) / 2.0, (iy1 + iy2) / 2.0, "Intersection", color='white',
             fontsize=9.5, fontweight='bold', ha='center', va='center')
    ax1.legend(loc='lower right', frameon=True, fontsize=8.5)

    # --- Right: Area of Union ---
    ax2.set_title("Area of Union $(B \\cup G)$", fontsize=11, fontweight='bold', pad=10)
    # Blue box fill (Green)
    ax2.add_patch(patches.Rectangle((bA[0], bA[1]), bA[2] - bA[0], bA[3] - bA[1],
                                    facecolor='#2ecc71', edgecolor='#2980b9', lw=2.2, alpha=0.85))
    # Red box fill (Green)
    ax2.add_patch(patches.Rectangle((bB[0], bB[1]), bB[2] - bB[0], bB[3] - bB[1],
                                    facecolor='#2ecc71', edgecolor='#c0392b', lw=2.2, alpha=0.85))
    ax2.text(2.8, 2.3, "Union", color='white', fontsize=12, fontweight='bold', ha='center', va='center')

    # Formula text at bottom
    iou_val = BoundingBox.compute_iou(np.array(bA), np.array(bB))
    fig.text(0.5, 0.04, f"$\\mathrm{{IoU}} = \\frac{{\\mathrm{{Area}}(B \\cap G)}}{{\\mathrm{{Area}}(B \\cup G)}} = {iou_val:.2f}$",
             ha='center', fontsize=12, fontweight='bold', color='#2c3e50')

    plt.tight_layout(rect=[0.0, 0.08, 1.0, 0.96])
    _save_figure(fig, "Figure_10_20", save_dir)
    return fig


def generate_figure_10_21(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.21: Illustration of replicated calculations in sliding windows.
    Two overlapping input windows (red and blue) sharing receptive fields (green).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Figure 10.21: Computational Redundancy of Sliding Windows",
                 fontsize=11, fontweight='bold', pad=12)

    # 7x6 image grid
    for r in range(6):
        for c in range(7):
            rect = patches.Rectangle((c, 5 - r), 0.94, 0.94, facecolor='#f8f9f9', edgecolor='#bdc3c7', lw=0.8)
            ax.add_patch(rect)

    # Window 1 (Red: top-left 4x4)
    w1 = patches.Rectangle((0.02, 2.02), 3.9, 3.9, fill=False, edgecolor='#c0392b', lw=2.6, linestyle='-')
    ax.add_patch(w1)
    ax.text(0.2, 5.7, "Window 1 (Red)", color='#c0392b', fontsize=9.5, fontweight='bold')

    # Window 2 (Blue: shifted right by 1 pixel, 4x4)
    w2 = patches.Rectangle((1.02, 2.02), 3.9, 3.9, fill=False, edgecolor='#2980b9', lw=2.6, linestyle='--')
    ax.add_patch(w2)
    ax.text(4.8, 5.7, "Window 2 (Blue)", color='#2980b9', fontsize=9.5, fontweight='bold')

    # Shared receptive field patch (Green: 3x3 overlap region)
    rf_shared = patches.Rectangle((1.05, 2.05), 2.84, 2.84,
                                  facecolor='#2ecc71', edgecolor='#27ae60', lw=2.0, alpha=0.45)
    ax.add_patch(rf_shared)
    ax.text(2.47, 3.47, "Shared Receptive Field\n(Evaluated twice in naive sliding window)",
            color='#145a32', fontsize=8.5, fontweight='bold', ha='center', va='center')

    plt.tight_layout()
    _save_figure(fig, "Figure_10_21", save_dir)
    return fig


def generate_figure_10_22(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.22: Example of a simple convolutional network.
    6x6 input image -> 3x3 conv -> 2x2 pooling -> fully connected (1 unit).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 5.0)
    ax.axis('off')
    ax.set_title("Figure 10.22: Simple CNN for Object Detection ($6 \\times 6$ Input)",
                 fontsize=11.5, fontweight='bold', pad=12)

    stages = [
        ("Input Image", "6×6", 6, 0.5, '#eaeded'),
        ("3×3 Conv", "4×4", 4, 3.2, '#fadbd8'),
        ("2×2 Pooling", "2×2", 2, 5.8, '#f9e79f'),
        ("Fully Connected", "1×1", 1, 8.2, '#d5f5e3'),
    ]

    for title, dim_lbl, sz, x_center, col in stages:
        box_w = sz * 0.45
        box_h = sz * 0.45
        y_bottom = 2.2 - box_h / 2.0
        x_left = x_center - box_w / 2.0
        rect = patches.Rectangle((x_left, y_bottom), box_w, box_h,
                                 facecolor=col, edgecolor='#2c3e50', lw=1.5)
        ax.add_patch(rect)
        # Sub-grid cells
        for r in range(sz):
            for c in range(sz):
                c_w = box_w / sz
                c_h = box_h / sz
                ax.add_patch(patches.Rectangle((x_left + c * c_w, y_bottom + r * c_h),
                                               c_w, c_h, fill=False, edgecolor='#7f8c8d', lw=0.7))

        ax.text(x_center, y_bottom + box_h + 0.25, title, ha='center', fontsize=9.5, fontweight='bold')
        ax.text(x_center, y_bottom - 0.35, dim_lbl, ha='center', fontsize=9.0, color='#34495e')

    # Arrows between stages
    for x_arr in [1.95, 4.5, 6.75]:
        ax.annotate("", xy=(x_arr + 0.5, 2.2), xytext=(x_arr, 2.2),
                    arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=2.0))

    plt.tight_layout()
    _save_figure(fig, "Figure_10_22", save_dir)
    return fig


def generate_figure_10_23(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.23: Convolutional sliding window expansion on an 8x8 image.
    Blue regions indicate additional computation, while yellow/white regions are shared.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.set_xlim(-0.5, 11.0)
    ax.set_ylim(-0.5, 5.2)
    ax.axis('off')
    ax.set_title("Figure 10.23: Convolutional Sliding Window Acceleration ($8 \\times 8$ Input)",
                 fontsize=11.5, fontweight='bold', pad=12)

    stages = [
        ("Expanded Input", "8×8", 8, 6, 0.8),
        ("Expanded Conv", "6×6", 6, 4, 3.8),
        ("Expanded Pool", "3×3", 3, 2, 6.7),
        ("Output Map", "2×2", 2, 1, 9.2),
    ]

    for title, dim_lbl, total_sz, orig_sz, x_center in stages:
        scale = 0.35
        box_w = total_sz * scale
        box_h = total_sz * scale
        y_bottom = 2.4 - box_h / 2.0
        x_left = x_center - box_w / 2.0

        for r in range(total_sz):
            for c in range(total_sz):
                # Is this cell in the original 6x6 computation or additional?
                is_original = (r < orig_sz) and (c < orig_sz)
                fc = '#fcf3cf' if is_original else '#d4e6f1'  # Yellow for shared, Blue for additional
                ec = '#7f8c8d' if is_original else '#2980b9'
                rect = patches.Rectangle((x_left + c * scale, y_bottom + (total_sz - 1 - r) * scale),
                                         scale, scale, facecolor=fc, edgecolor=ec, lw=1.0)
                ax.add_patch(rect)

        ax.text(x_center, y_bottom + box_h + 0.25, title, ha='center', fontsize=9.5, fontweight='bold')
        ax.text(x_center, y_bottom - 0.35, dim_lbl, ha='center', fontsize=9.0, color='#34495e')

    # Legend
    p_orig = patches.Patch(facecolor='#fcf3cf', edgecolor='#7f8c8d', label='Shared Computation (Top-left window)')
    p_add = patches.Patch(facecolor='#d4e6f1', edgecolor='#2980b9', label='Additional Computation for all 9 windows')
    ax.legend(handles=[p_orig, p_add], loc='lower center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, frameon=True, fontsize=8.5)

    plt.tight_layout()
    _save_figure(fig, "Figure_10_23", save_dir)
    return fig


def generate_figure_10_24(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.24: Detection across multiple scales and aspect ratios using fixed input window.
    (a) Original image.
    (b) Horizontally scaled image with fixed detection window (red box).
    (c) Projection back into original image space.
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))

    # (a) Original image with a wide object (e.g. car)
    axes[0].set_xlim(0, 10)
    axes[0].set_ylim(0, 8)
    axes[0].axis('off')
    axes[0].set_title("(a) Original Image", fontsize=10.5, fontweight='bold')
    # Draw original wide car silhouette
    car_orig = patches.Rectangle((3.0, 2.5), 4.5, 2.2, facecolor='#d5dbdb', edgecolor='#2c3e50', lw=2.0)
    axes[0].add_patch(car_orig)
    axes[0].text(5.25, 3.6, "Wide Object", ha='center', va='center', fontsize=10, fontweight='bold')

    # (b) Scaled image (horizontal compression by 0.5)
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 8)
    axes[1].axis('off')
    axes[1].set_title("(b) Horizontally Scaled Image", fontsize=10.5, fontweight='bold')
    # Compressed car
    car_scaled = patches.Rectangle((3.8, 2.5), 2.25, 2.2, facecolor='#d5dbdb', edgecolor='#2c3e50', lw=2.0)
    axes[1].add_patch(car_scaled)
    # Fixed-size square detection window (Red box)
    det_win = patches.Rectangle((3.6, 2.3), 2.6, 2.6, fill=False, edgecolor='#c0392b', lw=2.5, linestyle='--')
    axes[1].add_patch(det_win)
    axes[1].text(4.9, 1.6, "Fixed Detection Window\n(Fits scaled object)", color='#c0392b',
                 ha='center', fontsize=8.5, fontweight='bold')

    # (c) Projected back to original space
    axes[2].set_xlim(0, 10)
    axes[2].set_ylim(0, 8)
    axes[2].axis('off')
    axes[2].set_title("(c) Projected Bounding Box", fontsize=10.5, fontweight='bold')
    axes[2].add_patch(patches.Rectangle((3.0, 2.5), 4.5, 2.2, facecolor='#d5dbdb', edgecolor='#2c3e50', lw=2.0))
    # Projected wide bounding box
    proj_box = patches.Rectangle((2.7, 2.3), 5.1, 2.6, fill=False, edgecolor='#27ae60', lw=2.5)
    axes[2].add_patch(proj_box)
    axes[2].text(5.25, 3.6, "Wide Object", ha='center', va='center', fontsize=10, fontweight='bold')
    axes[2].text(5.25, 1.6, "True Aspect Ratio Recovered", color='#27ae60',
                 ha='center', fontsize=8.5, fontweight='bold')

    plt.suptitle("Figure 10.24: Multi-Scale and Aspect-Ratio Object Detection",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure(fig, "Figure_10_24", save_dir)
    return fig


def generate_figure_10_25(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 10.25: Non-Maximum Suppression (NMS).
    Red box: Highest probability detection (0.95).
    Blue boxes: Eliminated overlapping proposals (0.81, 0.75).
    Green box: Preserved distinct object instance (0.91).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title("Figure 10.25: Non-Maximum Suppression (NMS)", fontsize=12, fontweight='bold', pad=12)

    # Object 1 area (Left)
    # Highest probability: Red box (0.95)
    box_red = patches.Rectangle((1.5, 2.0), 3.2, 3.8, fill=False, edgecolor='#c0392b', lw=3.0)
    ax.add_patch(box_red)
    ax.text(1.5, 6.0, "0.95 (Winner)", color='white', fontsize=9.5, fontweight='bold',
            bbox=dict(boxstyle='square,pad=0.2', facecolor='#c0392b', edgecolor='none'))

    # Overlapping candidates: Blue boxes (0.81, 0.75)
    box_blue1 = patches.Rectangle((1.1, 1.8), 3.4, 3.6, fill=False, edgecolor='#2980b9', lw=1.8, linestyle='--')
    ax.add_patch(box_blue1)
    ax.text(0.9, 1.4, "0.81 (Suppressed)", color='#2980b9', fontsize=8.5, fontweight='bold')

    box_blue2 = patches.Rectangle((1.9, 2.4), 3.0, 3.5, fill=False, edgecolor='#2980b9', lw=1.8, linestyle='--')
    ax.add_patch(box_blue2)
    ax.text(3.3, 1.8, "0.75 (Suppressed)", color='#2980b9', fontsize=8.5, fontweight='bold')

    # Object 2 area (Right: distinct instance)
    # Preserved detection: Green box (0.91)
    box_green = patches.Rectangle((6.2, 2.2), 3.0, 3.6, fill=False, edgecolor='#27ae60', lw=2.8)
    ax.add_patch(box_green)
    ax.text(6.2, 6.0, "0.91 (Winner)", color='white', fontsize=9.5, fontweight='bold',
            bbox=dict(boxstyle='square,pad=0.2', facecolor='#27ae60', edgecolor='none'))

    # Explanation legend
    patch_w = patches.Patch(edgecolor='#c0392b', facecolor='none', lw=2.5, label='Highest Probability Detection (Kept)')
    patch_s = patches.Patch(edgecolor='#2980b9', facecolor='none', lw=1.8, linestyle='--', label='Overlapping IoU > 0.5 (Suppressed)')
    patch_d = patches.Patch(edgecolor='#27ae60', facecolor='none', lw=2.5, label='Independent Instance (Kept)')
    ax.legend(handles=[patch_w, patch_s, patch_d], loc='lower center', bbox_to_anchor=(0.5, -0.16),
              ncol=3, frameon=True, fontsize=8.5)

    plt.tight_layout()
    _save_figure(fig, "Figure_10_25", save_dir)
    return fig
