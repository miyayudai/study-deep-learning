"""
Generate and verify publication-quality figures for Chapter 2 Section 2.1:
  - Figure 2.1: Two-dimensional sine regression and uncertainty types
  - Figure 2.2: Bent coin (concave vs convex, frequentist vs Bayesian)
  - Figure 2.3: Cancer screening accuracy pictograms (100 people each)
  - Figure 2.4: Sum and product rules 5x3 array grid
  - Figure 2.5: Joint, marginal, and conditional histogram estimates (N=60)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.plot_utils import setup_style, save_plot
from common.probability import generate_2d_sine_data, compute_joint_marginal_conditional


def generate_figure_2_1(save_dir="result"):
    """
    Figure 2.1: 2D sine regression:
      (a) 3D surface plot of y = sin(2*pi*x1)*sin(2*pi*x2)
      (b) 100 data points with x2 unobserved (apparent high noise)
      (c) 100 data points with x2 fixed (low noise)
    """
    fig = plt.figure(figsize=(13, 4), dpi=300)

    # (a) 3D Surface plot
    ax_a = fig.add_subplot(1, 3, 1, projection='3d')
    x1_grid = np.linspace(0, 1, 60)
    x2_grid = np.linspace(0, 1, 60)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    Y = np.sin(2 * np.pi * X1) * np.sin(2 * np.pi * X2)

    surf = ax_a.plot_surface(
        X1, X2, Y,
        cmap='Reds',
        edgecolor='darkred',
        linewidth=0.2,
        alpha=0.85,
        antialiased=True
    )
    ax_a.set_xlabel(r'$x_1$', fontsize=11, labelpad=5)
    ax_a.set_ylabel(r'$x_2$', fontsize=11, labelpad=5)
    ax_a.set_zlabel(r'$y$', fontsize=11, labelpad=5)
    ax_a.set_title('(a)', y=-0.15, fontsize=12)
    ax_a.view_init(elev=28, azim=-125)
    ax_a.set_zlim(-1.2, 1.2)
    ax_a.grid(True, alpha=0.3)

    # (b) Unobserved x2
    ax_b = fig.add_subplot(1, 3, 2)
    data_unobs = generate_2d_sine_data(n_samples=100, noise_std=0.15, fixed_x2=None, seed=42)
    ax_b.scatter(
        data_unobs["x1"], data_unobs["t"],
        color='red', edgecolors='darkred', s=28, alpha=0.9, zorder=3
    )
    ax_b.set_xlabel(r'$x_1$', fontsize=11)
    ax_b.set_ylabel(r'$y$', fontsize=11)
    ax_b.set_xlim(-0.05, 1.05)
    ax_b.set_ylim(-1.5, 1.5)
    ax_b.set_title('(b)', y=-0.22, fontsize=12)
    ax_b.grid(True, linestyle='--', alpha=0.4)

    # (c) Fixed x2 = 0.25 (sin(2*pi*0.25) = 1.0)
    ax_c = fig.add_subplot(1, 3, 3)
    data_fixed = generate_2d_sine_data(n_samples=100, noise_std=0.15, fixed_x2=0.25, seed=42)
    ax_c.scatter(
        data_fixed["x1"], data_fixed["t"],
        color='red', edgecolors='darkred', s=28, alpha=0.9, zorder=3
    )
    x1_curve = np.linspace(0, 1, 200)
    y_curve = np.sin(2 * np.pi * x1_curve)
    ax_c.plot(x1_curve, y_curve, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    ax_c.set_xlabel(r'$x_1$', fontsize=11)
    ax_c.set_xlim(-0.05, 1.05)
    ax_c.set_ylim(-1.5, 1.5)
    ax_c.set_title('(c)', y=-0.22, fontsize=12)
    ax_c.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    p1 = os.path.join(save_dir, "fig2_01_two_dimensional_regression.png")
    save_plot(fig, p1)
    p2 = os.path.join("2", "result", "fig2_01_two_dimensional_regression.png")
    save_plot(fig, p2)
    plt.close(fig)


def generate_figure_2_2(save_dir="result"):
    """
    Figure 2.2: Bent coin illustration (concave 60% vs convex 40%).
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), dpi=300)

    # Parametric bent coin visualization
    theta = np.linspace(0, 2 * np.pi, 200)
    r = np.linspace(0, 1, 100)
    R, THETA = np.meshgrid(r, theta)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)

    # Left: Concave side up (valley shape, parabolic bend z = 0.35*(X^2 - Y^2) or cup)
    Z_concave = 0.4 * (X**2 + 0.3 * Y**2)
    im0 = axes[0].contourf(X, Y, Z_concave, levels=30, cmap='bone')
    axes[0].contour(X, Y, Z_concave, levels=8, colors='silver', linewidths=0.6, alpha=0.7)
    axes[0].plot(np.cos(theta), np.sin(theta), color='gray', linewidth=3)
    axes[0].set_aspect('equal')
    axes[0].axis('off')
    axes[0].set_title("Concave side up", fontsize=12, pad=10)
    axes[0].text(0.5, -0.15, "60%", transform=axes[0].transAxes,
                 ha='center', va='center', fontsize=14, fontweight='bold', color='#1f2937')

    # Right: Convex side up (inverted mound)
    Z_convex = -Z_concave
    im1 = axes[1].contourf(X, Y, Z_convex, levels=30, cmap='bone_r')
    axes[1].contour(X, Y, Z_convex, levels=8, colors='silver', linewidths=0.6, alpha=0.7)
    axes[1].plot(np.cos(theta), np.sin(theta), color='gray', linewidth=3)
    axes[1].set_aspect('equal')
    axes[1].axis('off')
    axes[1].set_title("Convex side up", fontsize=12, pad=10)
    axes[1].text(0.5, -0.15, "40%", transform=axes[1].transAxes,
                 ha='center', va='center', fontsize=14, fontweight='bold', color='#1f2937')

    plt.tight_layout()
    p1 = os.path.join(save_dir, "fig2_02_bent_coin.png")
    save_plot(fig, p1)
    p2 = os.path.join("2", "result", "fig2_02_bent_coin.png")
    save_plot(fig, p2)
    plt.close(fig)


def draw_person_icon(ax, x, y, color, scale=0.35):
    """Draw a stylized human pictogram at (x, y)."""
    # Head: circle
    head = patches.Circle((x, y + scale * 0.75), scale * 0.22, color=color, zorder=3)
    ax.add_patch(head)

    # Body and legs: rounded torso + legs
    # Torso: rectangle with width scale*0.38, height scale*0.5
    body = patches.FancyBboxPatch(
        (x - scale * 0.22, y - scale * 0.1),
        scale * 0.44, scale * 0.65,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        color=color, zorder=3
    )
    ax.add_patch(body)

    # Legs: two vertical rounded lines or narrow rectangles
    leg1 = patches.Rectangle(
        (x - scale * 0.18, y - scale * 0.6),
        scale * 0.14, scale * 0.55,
        color=color, zorder=3
    )
    leg2 = patches.Rectangle(
        (x + scale * 0.04, y - scale * 0.6),
        scale * 0.14, scale * 0.55,
        color=color, zorder=3
    )
    ax.add_patch(leg1)
    ax.add_patch(leg2)


def generate_figure_2_3(save_dir="result"):
    """
    Figure 2.3: Medical screening accuracy pictogram:
      - Left: No Cancer (100 people: 97 blue = test negative, 3 red = false positive)
      - Right: Cancer (100 people: 10 blue = false negative, 90 red = test positive)
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.5), dpi=300)

    # Colors matching textbook: bright blue (#1e88e5) and bright red (#e53935)
    c_blue = '#1e88e5'
    c_red = '#e53935'

    # Left: No Cancer (10x10 = 100 people)
    # Bottom 3 people in row 0 are red (false positives), remaining 97 are blue
    ax_left = axes[0]
    ax_left.set_xlim(-0.8, 9.8)
    ax_left.set_ylim(-1.5, 10.5)
    ax_left.axis('off')
    ax_left.set_title("No Cancer", fontsize=13, fontweight='bold', pad=15)

    count = 0
    for row in range(10):
        for col in range(10):
            # In textbook: 3 red icons are at the bottom right
            if row == 0 and col >= 7:
                color = c_red
            else:
                color = c_blue
            draw_person_icon(ax_left, col, row, color, scale=0.42)
            count += 1

    # Right: Cancer (10x10 = 100 people)
    # Top row has 10 blue (false negatives), remaining 90 are red
    ax_right = axes[1]
    ax_right.set_xlim(-0.8, 9.8)
    ax_right.set_ylim(-1.5, 10.5)
    ax_right.axis('off')
    ax_right.set_title("Cancer", fontsize=13, fontweight='bold', pad=15)

    for row in range(10):
        for col in range(10):
            # Top row: row == 9 -> blue (10 false negatives)
            if row == 9:
                color = c_blue
            else:
                color = c_red
            draw_person_icon(ax_right, col, row, color, scale=0.42)

    plt.tight_layout()
    p1 = os.path.join(save_dir, "fig2_03_medical_screening.png")
    save_plot(fig, p1)
    p2 = os.path.join("2", "result", "fig2_03_medical_screening.png")
    save_plot(fig, p2)
    plt.close(fig)


def generate_figure_2_4(save_dir="result"):
    """
    Figure 2.4: Sum and product rules 5x3 array grid illustration.
    """
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
    ax.set_xlim(-1.2, 6.5)
    ax.set_ylim(-0.8, 4.5)
    ax.axis('off')

    # Draw 5 columns x 3 rows grid in red
    cols, rows = 5, 3
    for c in range(cols):
        for r in range(rows):
            rect = patches.Rectangle(
                (c, r), 1, 1,
                fill=False, edgecolor='red', linewidth=1.8
            )
            ax.add_patch(rect)

    # Highlighted cell: column 3 (i=4), row 1 (j=2) (0-indexed: col=3, row=1)
    target_c, target_r = 3, 1
    ax.text(target_c + 0.5, target_r + 0.5, r'$n_{ij}$',
            ha='center', va='center', fontsize=14, fontstyle='italic')

    # Column label x_i below column target_c
    ax.text(target_c + 0.5, -0.3, r'$x_i$',
            ha='center', va='center', fontsize=14, fontstyle='italic')

    # Row label y_j to the left of row target_r
    ax.text(-0.4, target_r + 0.5, r'$y_j$',
            ha='center', va='center', fontsize=14, fontstyle='italic')

    # Curly bracket above column target_c for c_i
    # Draw horizontal bracket above cell target_c
    bx0, bx1 = target_c, target_c + 1.0
    by = rows + 0.15
    bh = 0.25
    bm = (bx0 + bx1) / 2.0
    
    # Path for horizontal curly bracket:
    verts_top = [
        (bx0, by),
        (bx0 + 0.1, by + bh),
        (bm - 0.05, by + bh),
        (bm, by + bh + 0.1),
        (bm + 0.05, by + bh),
        (bx1 - 0.1, by + bh),
        (bx1, by)
    ]
    codes_top = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3
    ]
    path_top = Path(verts_top, codes_top)
    patch_top = patches.PathPatch(path_top, facecolor='none', edgecolor='black', lw=1.5)
    ax.add_patch(patch_top)
    ax.text(bm, by + bh + 0.25, r'$c_i$', ha='center', va='bottom', fontsize=14, fontstyle='italic')

    # Curly bracket to the right of row target_r for r_j
    ry0, ry1 = target_r, target_r + 1.0
    rx = cols + 0.15
    rw = 0.25
    rm = (ry0 + ry1) / 2.0
    verts_right = [
        (rx, ry0),
        (rx + rw, ry0 + 0.1),
        (rx + rw, rm - 0.05),
        (rx + rw + 0.1, rm),
        (rx + rw, rm + 0.05),
        (rx + rw, ry1 - 0.1),
        (rx, ry1)
    ]
    codes_right = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3
    ]
    path_right = Path(verts_right, codes_right)
    patch_right = patches.PathPatch(path_right, facecolor='none', edgecolor='black', lw=1.5)
    ax.add_patch(patch_right)
    ax.text(rx + rw + 0.25, rm, r'$r_j$', ha='left', va='center', fontsize=14, fontstyle='italic')

    plt.tight_layout()
    p1 = os.path.join(save_dir, "fig2_04_sum_product_rules.png")
    save_plot(fig, p1)
    p2 = os.path.join("2", "result", "fig2_04_sum_product_rules.png")
    save_plot(fig, p2)
    plt.close(fig)


def generate_figure_2_5(save_dir="result"):
    """
    Figure 2.5: Joint, marginal, and conditional distributions:
      - Top-left: N=60 data points in 9x2 grid (red boundaries, blue dots)
      - Top-right: Histogram of p(Y) (2 horizontal bars)
      - Bottom-left: Histogram of p(X) (9 vertical bars)
      - Bottom-right: Histogram of p(X | Y = 1) (9 vertical bars for bottom row)
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5), dpi=300)

    # Generate synthetic discrete points matching textbook distribution
    # 9 columns (X=1..9) and 2 rows (Y=1, Y=2)
    # Bottom row Y=1 has more points, top row Y=2 has points clustered towards right
    rng = np.random.default_rng(2024)

    # Specify empirical counts nij matching Figure 2.5 visual appearance
    # Total points N = 60
    # Y = 1 (bottom row): 37 points across 9 columns
    # Y = 2 (top row): 23 points across 9 columns
    counts_y1 = np.array([3, 5, 8, 8, 6, 4, 2, 1, 0]) # sum = 37
    counts_y2 = np.array([0, 1, 2, 3, 4, 5, 4, 3, 1]) # sum = 23
    # nij shape (9, 2): rows = X, cols = Y
    nij = np.column_stack([counts_y1, counts_y2])
    N = int(np.sum(nij))
    assert N == 60, f"Total points {N} != 60"

    res = compute_joint_marginal_conditional(nij)
    p_X = res["p_X"]
    p_Y = res["p_Y"]
    p_X_given_Y1 = res["p_XY"][:, 0] / np.sum(res["p_XY"][:, 0])

    # Subplot 1: Top-Left: Joint distribution grid with points
    ax_tl = axes[0, 0]
    ax_tl.set_xlim(0, 9)
    ax_tl.set_ylim(0, 2)
    for c in range(9):
        for r in range(2):
            rect = patches.Rectangle((c, r), 1, 1, fill=False, edgecolor='red', linewidth=1.6)
            ax_tl.add_patch(rect)

    # Scatter points with jitter inside cells
    for c in range(9):
        # Y = 1 (r = 0)
        n1 = counts_y1[c]
        if n1 > 0:
            jx = rng.uniform(c + 0.15, c + 0.85, n1)
            jy = rng.uniform(0.15, 0.85, n1)
            ax_tl.scatter(jx, jy, color='blue', s=26, zorder=3)
        # Y = 2 (r = 1)
        n2 = counts_y2[c]
        if n2 > 0:
            jx = rng.uniform(c + 0.15, c + 0.85, n2)
            jy = rng.uniform(1.15, 1.85, n2)
            ax_tl.scatter(jx, jy, color='blue', s=26, zorder=3)

    ax_tl.set_title(r'$p(X, Y)$', fontsize=12, pad=10)
    ax_tl.set_xlabel(r'$X$', fontsize=11)
    ax_tl.set_yticks([0.5, 1.5])
    ax_tl.set_yticklabels([r'$Y = 1$', r'$Y = 2$'], fontsize=11)
    ax_tl.set_xticks([])
    ax_tl.tick_params(left=False)
    for spine in ax_tl.spines.values():
        spine.set_visible(False)

    # Subplot 2: Top-Right: Horizontal histogram of p(Y)
    ax_tr = axes[0, 1]
    y_pos = [0.5, 1.5]
    bar_height = 0.65
    ax_tr.barh(y_pos, [p_Y[0], p_Y[1]], height=bar_height,
               color='#b0b7f7', edgecolor='black', linewidth=1.0)
    ax_tr.set_title(r'$p(Y)$', fontsize=12, pad=10)
    ax_tr.set_ylim(0, 2)
    ax_tr.set_xlim(0, 0.8)
    ax_tr.set_xticks([])
    ax_tr.set_yticks([])
    for spine in ax_tr.spines.values():
        spine.set_visible(True)

    # Subplot 3: Bottom-Left: Vertical histogram of p(X)
    ax_bl = axes[1, 0]
    x_pos = np.arange(9) + 0.5
    bar_width = 0.75
    ax_bl.bar(x_pos, p_X, width=bar_width,
              color='#b0b7f7', edgecolor='black', linewidth=1.0)
    ax_bl.set_title(r'$p(X)$', fontsize=12, pad=10)
    ax_bl.set_xlabel(r'$X$', fontsize=11)
    ax_bl.set_xlim(0, 9)
    ax_bl.set_ylim(0, 0.25)
    ax_bl.set_xticks([])
    ax_bl.set_yticks([])
    for spine in ax_bl.spines.values():
        spine.set_visible(True)

    # Subplot 4: Bottom-Right: Vertical histogram of p(X | Y = 1)
    ax_br = axes[1, 1]
    ax_br.bar(x_pos, p_X_given_Y1, width=bar_width,
              color='#b0b7f7', edgecolor='black', linewidth=1.0)
    ax_br.set_title(r'$p(X | Y = 1)$', fontsize=12, pad=10)
    ax_br.set_xlabel(r'$X$', fontsize=11)
    ax_br.set_xlim(0, 9)
    ax_br.set_ylim(0, 0.30)
    ax_br.set_xticks([])
    ax_br.set_yticks([])
    for spine in ax_br.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    p1 = os.path.join(save_dir, "fig2_05_joint_marginal_conditional.png")
    save_plot(fig, p1)
    p2 = os.path.join("2", "result", "fig2_05_joint_marginal_conditional.png")
    save_plot(fig, p2)
    plt.close(fig)


if __name__ == "__main__":
    setup_style()
    print("Generating Figure 2.1...")
    generate_figure_2_1()
    print("Generating Figure 2.2...")
    generate_figure_2_2()
    print("Generating Figure 2.3...")
    generate_figure_2_3()
    print("Generating Figure 2.4...")
    generate_figure_2_4()
    print("Generating Figure 2.5...")
    generate_figure_2_5()
    print("All figures successfully generated!")
