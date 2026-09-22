"""
Generate Section 3.5 figures:
- fig3_13_histogram_density.png
- fig3_14_kernel_density.png
- fig3_15_knn_density.png
- fig3_16_knn_classifier.png
"""
import os
import sys
import numpy as np
import scipy.stats as stats
from scipy import special
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from common.plot_utils import setup_style

setup_style()

# Exact dataset of 50 points matching Bishop Figure 3.13 bin counts:
# Bin counts for Delta=0.04 across 25 bins:
# [0, 0, 1, 0, 0, 2, 1, 0, 3, 1, 2, 1, 0, 0, 0, 1, 1, 5, 5, 4, 5, 4, 5, 7, 2]
TARGET_COUNTS_004 = [0, 0, 1, 0, 0, 2, 1, 0, 3, 1, 2, 1, 0, 0, 0, 1, 1, 5, 5, 4, 5, 4, 5, 7, 2]

def get_bishop_50_points():
    bin_edges = np.linspace(0, 1, 26)
    rng = np.random.default_rng(2024)
    pts = []
    for i, count in enumerate(TARGET_COUNTS_004):
        if count > 0:
            low, high = bin_edges[i], bin_edges[i+1]
            sub = rng.uniform(low + 0.003, high - 0.003, size=count)
            pts.extend(sub)
    pts = np.sort(np.array(pts))
    return pts

def true_mixture_pdf(x):
    return 0.3 * stats.norm.pdf(x, 0.3, 0.1) + 0.7 * stats.norm.pdf(x, 0.8, 0.1)

# Exact vector points for Figure 3.16 extracted directly from Bishop vector PDF:
RED_POINTS_FIG16 = np.array([
    [8.23022, 103.2732],
    [49.50769, 101.94794],
    [40.10266, 96.97552],
    [18.15771, 84.70715],
    [38.0127, 73.43536],
    [54.47144, 82.71924],
    [30.6977, 55.53198],
    [18.41895, 41.94006],
    [41.1477, 40.6131],
    [50.29144, 62.49402],
])

BLUE_POINTS_FIG16 = np.array([
    [85.82312, 90.67609],
    [73.54437, 73.10321],
    [62.04938, 45.255],
    [45.58893, 26.68896],
    [30.17517, 31.33093],
    [32.78772, 5.13849],
    [60.48187, 26.02637],
    [78.76935, 45.58722],
    [86.8681, 67.79858],
    [105.15558, 61.16876],
    [92.35437, 52.21704],
    [86.08435, 27.35162],
    [76.67938, 8.45343],
    [68.58063, 4.47583],
])

QUERY_POINT_FIG16 = np.array([63.35565, 78.07733])

BOUNDARY_VERTICES_FIG16 = np.array([
    [2.8382, 14.65088],
    [5.33662, 15.62637],
    [29.874, 42.8154],
    [38.84541, 32.20819],
    [52.97905, 36.71645],
    [50.08994, 49.72699],
    [65.39897, 60.1689],
    [60.48793, 70.93283],
    [69.57568, 88.95464],
    [67.6878, 96.38852],
    [76.48343, 124.72174],
])


def plot_figure_3_13(save_paths=None, show=False):
    """
    Figure 3.13: Histogram density estimation for Delta = 0.04, 0.08, 0.25
    over 50 data points drawn from mixture 0.3*N(0.3, 0.1^2) + 0.7*N(0.8, 0.1^2).
    """
    pts = get_bishop_50_points()
    N = len(pts)
    deltas = [0.04, 0.08, 0.25]
    
    fig, axes = plt.subplots(3, 1, figsize=(6, 5.5), dpi=300)
    x_eval = np.linspace(0, 1, 400)
    p_true = true_mixture_pdf(x_eval)
    
    for ax, delta in zip(axes, deltas):
        n_bins = int(np.round(1.0 / delta))
        bin_edges = np.linspace(0, 1, n_bins + 1)
        counts, _ = np.histogram(pts, bins=bin_edges)
        densities = counts / (N * delta)
        
        # Plot histogram bars
        for i in range(n_bins):
            ax.bar(bin_edges[i], densities[i], width=delta, align='edge',
                   facecolor='#7B88FF', edgecolor='black', linewidth=0.8)
        
        # True distribution in bright green
        ax.plot(x_eval, p_true, color='#00DD00', linewidth=2.0)
        
        # Labels and limits
        ax.text(0.04, 3.8, rf"$\Delta = {delta}$", fontsize=12, verticalalignment='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 5)
        ax.set_yticks([0, 5])
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(direction='in', top=True, right=True)
        for s in ax.spines.values():
            s.set_linewidth(0.8)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Saved: {p}")
    if show:
        plt.show()
    return fig, axes


def plot_figure_3_14(save_paths=None, show=False):
    """
    Figure 3.14: Kernel density estimation with Gaussian kernel for h = 0.005, 0.07, 0.2
    over 50 data points drawn from mixture 0.3*N(0.3, 0.1^2) + 0.7*N(0.8, 0.1^2).
    """
    pts = get_bishop_50_points()
    N = len(pts)
    hs = [0.005, 0.07, 0.2]
    
    fig, axes = plt.subplots(3, 1, figsize=(6, 5.5), dpi=300)
    x_eval = np.linspace(0, 1, 600)
    p_true = true_mixture_pdf(x_eval)
    
    for ax, h in zip(axes, hs):
        diff = (x_eval[:, np.newaxis] - pts[np.newaxis, :]) / h
        p_kde = np.sum(np.exp(-0.5 * diff**2) / np.sqrt(2.0 * np.pi), axis=1) / (N * h)
        
        # True distribution in bright green
        ax.plot(x_eval, p_true, color='#00DD00', linewidth=2.0)
        # KDE in bright blue
        ax.plot(x_eval, p_kde, color='#0000FF', linewidth=1.8)
        
        # Labels and limits
        ax.text(0.04, 3.8, rf"$h = {h}$", fontsize=12, verticalalignment='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 5)
        ax.set_yticks([0, 5])
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(direction='in', top=True, right=True)
        for s in ax.spines.values():
            s.set_linewidth(0.8)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Saved: {p}")
    if show:
        plt.show()
    return fig, axes


def plot_figure_3_15(save_paths=None, show=False):
    """
    Figure 3.15: K-Nearest-Neighbour density estimation for K = 1, 5, 30
    over 50 data points drawn from mixture 0.3*N(0.3, 0.1^2) + 0.7*N(0.8, 0.1^2).
    """
    pts = get_bishop_50_points()
    N = len(pts)
    Ks = [1, 5, 30]
    
    fig, axes = plt.subplots(3, 1, figsize=(6, 5.5), dpi=300)
    x_eval = np.linspace(0, 1, 1000)
    p_true = true_mixture_pdf(x_eval)
    
    for ax, K in zip(axes, Ks):
        dists = np.abs(x_eval[:, np.newaxis] - pts[np.newaxis, :])
        sorted_dists = np.sort(dists, axis=1)
        r_k = sorted_dists[:, K - 1]
        r_k = np.maximum(r_k, 1e-12)
        v_k = 2.0 * r_k
        p_knn = K / (N * v_k)
        
        # True distribution in bright green
        ax.plot(x_eval, p_true, color='#00DD00', linewidth=2.0)
        # KNN in bright blue, clipped to 5 for clean display
        p_knn_clipped = np.minimum(p_knn, 5.0)
        ax.plot(x_eval, p_knn_clipped, color='#0000FF', linewidth=1.8)
        
        # Labels and limits
        ax.text(0.04, 3.8, rf"$K = {K}$", fontsize=12, verticalalignment='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 5)
        ax.set_yticks([0, 5])
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(direction='in', top=True, right=True)
        for s in ax.spines.values():
            s.set_linewidth(0.8)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Saved: {p}")
    if show:
        plt.show()
    return fig, axes


def plot_figure_3_16(save_paths=None, show=False):
    """
    Figure 3.16:
    (a) K-nearest-neighbour classifier (K=3) query point diamond connected to 3 nearest points.
    (b) Nearest-neighbour (K=1) piecewise linear decision boundary.
    Using exact vector coordinates extracted from Bishop (2024).
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), dpi=300)
    
    # ------------------ (a) K = 3 Classifier ------------------
    ax1 = axes[0]
    nn_pts = [
        [54.47144, 82.71924],
        [50.29144, 62.49402],
        [73.54437, 73.10321]
    ]
    for pt in nn_pts:
        ax1.plot([QUERY_POINT_FIG16[0], pt[0]], [QUERY_POINT_FIG16[1], pt[1]],
                 color='#00DD00', linewidth=2.0, zorder=2)
    
    ax1.scatter(RED_POINTS_FIG16[:, 0], RED_POINTS_FIG16[:, 1],
                color='#FF0000', s=45, zorder=4, edgecolors='none')
    ax1.scatter(BLUE_POINTS_FIG16[:, 0], BLUE_POINTS_FIG16[:, 1],
                color='#0000FF', s=45, zorder=4, edgecolors='none')
    ax1.scatter(QUERY_POINT_FIG16[0], QUERY_POINT_FIG16[1],
                marker='D', s=70, facecolors='white', edgecolors='black', linewidth=1.5, zorder=5)
    
    ax1.annotate('', xy=(125, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15))
    ax1.annotate('', xy=(0, 125), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15))
    ax1.text(122, -10, r"$x_1$", fontsize=13)
    ax1.text(-10, 122, r"$x_2$", fontsize=13)
    ax1.text(60, -18, "(a)", fontsize=13, horizontalalignment='center')
    
    ax1.set_xlim(-5, 130)
    ax1.set_ylim(-5, 130)
    ax1.axis('off')
    ax1.set_aspect('equal')

    # ------------------ (b) K = 1 Decision Boundary ------------------
    ax2 = axes[1]
    ax2.plot(BOUNDARY_VERTICES_FIG16[:, 0], BOUNDARY_VERTICES_FIG16[:, 1],
             color='#00DD00', linewidth=2.0, zorder=3)
    
    ax2.scatter(RED_POINTS_FIG16[:, 0], RED_POINTS_FIG16[:, 1],
                color='#FF0000', s=45, zorder=4, edgecolors='none')
    ax2.scatter(BLUE_POINTS_FIG16[:, 0], BLUE_POINTS_FIG16[:, 1],
                color='#0000FF', s=45, zorder=4, edgecolors='none')

    ax2.annotate('', xy=(125, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15))
    ax2.annotate('', xy=(0, 125), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15))
    ax2.text(122, -10, r"$x_1$", fontsize=13)
    ax2.text(-10, 122, r"$x_2$", fontsize=13)
    ax2.text(60, -18, "(b)", fontsize=13, horizontalalignment='center')
    
    ax2.set_xlim(-5, 130)
    ax2.set_ylim(-5, 130)
    ax2.axis('off')
    ax2.set_aspect('equal')

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Saved: {p}")
    if show:
        plt.show()
    return fig, axes


if __name__ == '__main__':
    os.makedirs('3/result', exist_ok=True)
    os.makedirs('result', exist_ok=True)
    
    plot_figure_3_13(save_paths=['3/result/fig3_13_histogram_density.png', 'result/fig3_13_histogram_density.png'])
    plot_figure_3_14(save_paths=['3/result/fig3_14_kernel_density.png', 'result/fig3_14_kernel_density.png'])
    plot_figure_3_15(save_paths=['3/result/fig3_15_knn_density.png', 'result/fig3_15_knn_density.png'])
    plot_figure_3_16(save_paths=['3/result/fig3_16_knn_classifier.png', 'result/fig3_16_knn_classifier.png'])
    print("All Section 3.5 figures generated successfully!")
