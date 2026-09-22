"""
Generate and verify Figures 3.9, 3.10, 3.11, 3.12 for Section 3.3 Periodic Variables.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special
from scipy.optimize import root_scalar

def plot_figure_3_9(save_paths=None, show=False):
    """
    Faithfully reproduce Figure 3.9 from Bishop & Bishop (2024), page 90:
    Representation of periodic variables as 2D unit vectors x_n on the unit circle,
    and their sample average vector x_bar inside the circle.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', linewidth=1.2, zorder=1)
    
    # Axes lines
    ax.axhline(0, color='black', linewidth=0.8, zorder=1)
    ax.axvline(0, color='black', linewidth=0.8, zorder=1)
    
    # Observations on the circle
    thetas = np.array([0.35, 1.1, 2.3, 3.2])  # 4 representative points
    colors = ['#1E56A0', '#1E56A0', '#1E56A0', '#1E56A0']
    
    for i, th in enumerate(thetas):
        x_i = np.cos(th)
        y_i = np.sin(th)
        ax.annotate('', xy=(x_i, y_i), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='#1E56A0', lw=1.5, mutation_scale=12))
        ax.scatter([x_i], [y_i], color='#1E56A0', s=35, zorder=5)
        # Label each vector x_n
        offset = 0.12
        ax.text(x_i + offset*np.cos(th), y_i + offset*np.sin(th), rf'$\mathbf{{x}}_{i+1}$',
                fontsize=12, ha='center', va='center', color='#1E56A0', fontweight='bold')

    # Mean vector x_bar
    x_vecs = np.column_stack([np.cos(thetas), np.sin(thetas)])
    x_bar = np.mean(x_vecs, axis=0)
    r_bar = np.linalg.norm(x_bar)
    theta_bar = np.arctan2(x_bar[1], x_bar[0])
    
    ax.annotate('', xy=(x_bar[0], x_bar[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#E02020', lw=2.2, mutation_scale=14))
    ax.scatter([x_bar[0]], [x_bar[1]], color='#E02020', s=45, zorder=6)
    
    # Text annotations for x_bar, r, theta
    ax.text(x_bar[0] - 0.08, x_bar[1] + 0.08, r'$\bar{\mathbf{x}}$', fontsize=14, color='#E02020', fontweight='bold')
    
    # Arc for theta_bar
    arc_theta = np.linspace(0, theta_bar, 50)
    arc_r = 0.25
    ax.plot(arc_r * np.cos(arc_theta), arc_r * np.sin(arc_theta), color='black', linewidth=1.0)
    ax.text(0.30 * np.cos(theta_bar / 2), 0.30 * np.sin(theta_bar / 2), r'$\bar{\theta}$', fontsize=12)
    
    # Label r
    mid_r = 0.5 * x_bar
    ax.text(mid_r[0] + 0.05, mid_r[1] - 0.05, r'$r$', fontsize=12, color='#E02020')
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.set_title('Figure 3.9: Periodic variables on unit circle\nand sample mean vector', fontsize=11, pad=12)
    
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)
        
    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Saved: {p}")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax


def plot_figure_3_10(save_paths=None, show=False):
    """
    Faithfully reproduce Figure 3.10 from Bishop & Bishop (2024), page 91:
    2D Gaussian conditioned on the unit circle yielding the von Mises distribution.
    """
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=300)
    
    # Gaussian parameters
    r0 = 1.6
    theta0 = np.radians(45)
    mu1 = r0 * np.cos(theta0)
    mu2 = r0 * np.sin(theta0)
    sigma = 0.8
    
    # Grid for 2D Gaussian density
    x = np.linspace(-2.2, 2.6, 200)
    y = np.linspace(-2.2, 2.6, 200)
    X, Y = np.meshgrid(x, y)
    
    dist_sq = (X - mu1)**2 + (Y - mu2)**2
    density = np.exp(-0.5 * dist_sq / (sigma**2)) / (2 * np.pi * sigma**2)
    
    # Contours of p(x) in blue
    levels = np.linspace(0.04, density.max() * 0.95, 6)
    cs = ax.contour(X, Y, density, levels=levels, colors='#1E56A0', linewidths=1.2)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    
    # Unit circle r=1 in red
    circle_theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(circle_theta), np.sin(circle_theta), color='#E02020', linewidth=2.2, label='Unit circle ($r=1$)')
    
    # Axes lines
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.7)
    
    # Mark Gaussian center mu
    ax.scatter([mu1], [mu2], color='#1E56A0', s=40, zorder=5)
    ax.text(mu1 + 0.1, mu2 + 0.1, r'$\boldsymbol{\mu} = (r_0 \cos\theta_0, r_0 \sin\theta_0)$',
            fontsize=10, color='#1E56A0', fontweight='bold')
    
    # Labels
    ax.text(0.65, -0.9, r'$r = 1$', fontsize=12, color='#E02020', fontweight='bold')
    ax.text(-1.8, 1.8, r'$p(\mathbf{x})$', fontsize=13, color='#1E56A0')
    
    ax.set_xlim(-2.2, 2.6)
    ax.set_ylim(-2.2, 2.6)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.set_title('Figure 3.10: 2D Gaussian conditioned on unit circle\n(von Mises derivation)', fontsize=11, pad=12)
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)
        
    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Saved: {p}")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax


def plot_figure_3_11(save_paths=None, show=False):
    """
    Faithfully reproduce Figure 3.11 from Bishop & Bishop (2024), page 92:
    The von Mises distribution plotted for two different parameter values:
    Left: Cartesian plot. Right: Polar plot.
    m = 5, theta0 = pi/4 (red)
    m = 1, theta0 = 3*pi/4 (blue)
    """
    fig = plt.figure(figsize=(11, 4.8), dpi=300)
    
    # Parameters
    m1, th1 = 5.0, np.pi / 4.0
    m2, th2 = 1.0, 3.0 * np.pi / 4.0
    
    theta = np.linspace(0, 2*np.pi, 500)
    
    # von Mises pdf: 1 / (2*pi*I0(m)) * exp(m * cos(theta - theta0))
    # using numerically stable formulation: exp(m*(cos - 1)) / (2*pi*i0e(m))
    pdf1 = np.exp(m1 * (np.cos(theta - th1) - 1.0)) / (2 * np.pi * special.i0e(m1))
    pdf2 = np.exp(m2 * (np.cos(theta - th2) - 1.0)) / (2 * np.pi * special.i0e(m2))
    
    # 1. Cartesian plot (left)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(theta, pdf1, color='#E02020', linewidth=2.0, label=r'$m=5, \theta_0 = \pi/4$')
    ax1.plot(theta, pdf2, color='#1E56A0', linewidth=2.0, label=r'$m=1, \theta_0 = 3\pi/4$')
    
    ax1.set_xlim(0, 2*np.pi)
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax1.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax1.set_xlabel(r'$\theta$', fontsize=12)
    ax1.set_ylabel(r'$p(\theta)$', fontsize=12)
    ax1.set_title('(a) Cartesian plot', fontsize=12)
    ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax1.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for spine in ax1.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)
        
    # 2. Polar plot (right)
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    ax2.plot(theta, pdf1, color='#E02020', linewidth=2.0)
    ax2.plot(theta, pdf2, color='#1E56A0', linewidth=2.0)
    
    # Add radial lines / labels for mean angles
    ax2.plot([th1, th1], [0, np.max(pdf1)], color='#E02020', linestyle='--', linewidth=1.2)
    ax2.plot([th2, th2], [0, np.max(pdf2)], color='#1E56A0', linestyle='--', linewidth=1.2)
    
    ax2.set_theta_zero_location('E')
    ax2.set_theta_direction(1)
    ax2.set_title('(b) Polar plot', fontsize=12, pad=15)
    ax2.tick_params(labelsize=9)
    
    fig.suptitle('Figure 3.11: The von Mises distribution (Cartesian and Polar plots)', fontsize=13, y=1.02)
    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Saved: {p}")
    if show:
        plt.show()
    plt.close(fig)
    return fig, (ax1, ax2)


def plot_figure_3_12(save_paths=None, show=False):
    """
    Faithfully reproduce Figure 3.12 from Bishop & Bishop (2024), page 93:
    Left: Modified Bessel function I0(m) defined by (3.130).
    Right: Function A(m) = I1(m) / I0(m) defined by (3.136).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
    
    m = np.linspace(0, 10, 300)
    I0 = special.i0(m)
    I1 = special.i1(m)
    A = np.zeros_like(m)
    A[m > 0] = I1[m > 0] / I0[m > 0]
    A[0] = 0.0  # limit as m -> 0 is 0
    
    # (a) I0(m)
    ax1.plot(m, I0, color='#1E56A0', linewidth=2.0)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 3000)
    ax1.set_xlabel(r'$m$', fontsize=12)
    ax1.set_ylabel(r'$I_0(m)$', fontsize=12)
    ax1.set_title(r'(a) Zeroth-order Bessel function $I_0(m)$', fontsize=11)
    ax1.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for spine in ax1.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)
        
    # (b) A(m) = I1(m) / I0(m)
    ax2.plot(m, A, color='#E02020', linewidth=2.0)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 1.0)
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set_xlabel(r'$m$', fontsize=12)
    ax2.set_ylabel(r'$A(m)$', fontsize=12)
    ax2.set_title(r'(b) Ratio function $A(m) = I_1(m) / I_0(m)$', fontsize=11)
    ax2.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for spine in ax2.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)
        
    fig.suptitle('Figure 3.12: Bessel function $I_0(m)$ and ratio function $A(m)$', fontsize=13, y=1.02)
    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Saved: {p}")
    if show:
        plt.show()
    plt.close(fig)
    return fig, (ax1, ax2)

if __name__ == '__main__':
    paths_3_9 = ['result/fig3_09_periodic_vectors.png', '3/result/fig3_09_periodic_vectors.png']
    paths_3_10 = ['result/fig3_10_von_mises_conditioning.png', '3/result/fig3_10_von_mises_conditioning.png']
    paths_3_11 = ['result/fig3_11_von_mises_distribution.png', '3/result/fig3_11_von_mises_distribution.png']
    paths_3_12 = ['result/fig3_12_bessel_and_A_functions.png', '3/result/fig3_12_bessel_and_A_functions.png']
    
    plot_figure_3_9(save_paths=paths_3_9)
    plot_figure_3_10(save_paths=paths_3_10)
    plot_figure_3_11(save_paths=paths_3_11)
    plot_figure_3_12(save_paths=paths_3_12)
    print("All Figures 3.9 - 3.12 generated successfully!")
