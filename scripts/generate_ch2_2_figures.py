"""
Generate publication-quality figures for Chapter 2 Section 2.2:
  - Figure 2.6: Probability density p(x), CDF P(x), and delta x strip
  - Figure 2.7: Example distributions (Uniform in red, Exponential in blue, Laplace in green)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.plot_utils import setup_style, save_plot
from common.probability import (
    GaussianMixture1D,
    UniformDistribution,
    ExponentialDistribution,
    LaplaceDistribution
)


def generate_figure_2_6(save_dirs=("result", "2/result")):
    """
    Figure 2.6: Probability density p(x) and cumulative distribution function P(x).
    Shows:
      - Continuous bimodal density p(x) in red
      - Cumulative distribution function P(x) in blue
      - Shaded strip of width delta x under p(x)
      - Axis arrows and textbook mathematical annotations
    """
    # Define a clean bimodal Gaussian mixture matching the textbook Figure 2.6
    # Components: peak 1 at 1.4, peak 2 at 3.05
    gm = GaussianMixture1D(
        weights=[0.38, 0.62],
        means=[1.40, 3.05],
        stds=[0.55, 0.38]
    )

    x = np.linspace(0.0, 4.8, 600)
    p_x = gm.pdf(x)
    P_x = gm.cdf(x)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Shaded strip at x_slice with width dx
    x_slice = 1.22
    dx = 0.18
    mask_strip = (x >= x_slice) & (x <= x_slice + dx)
    x_strip = x[mask_strip]
    p_strip = p_x[mask_strip]

    # Fill green strip
    ax.fill_between(x_strip, 0, p_strip, color='#66e066', alpha=0.9, zorder=2)

    # Plot p(x) in red
    ax.plot(x, p_x, color='red', linewidth=2.0, zorder=3)

    # Plot P(x) in blue
    ax.plot(x, P_x, color='blue', linewidth=2.0, zorder=3)

    # Annotations matching Figure 2.6
    # 'p(x)' label near peak 2
    ax.text(2.65, 0.72, r'$p(x)$', color='black', fontsize=14, ha='right', va='center')
    # 'P(x)' label near top right of blue curve
    ax.text(3.95, 0.98, r'$P(x)$', color='black', fontsize=14, ha='left', va='top')
    # '\delta x' under the green strip
    ax.text(x_slice + dx / 2, -0.045, r'$\delta x$', color='black', fontsize=13, ha='center', va='top')
    # 'x' label at right end of x-axis
    ax.text(4.75, -0.045, r'$x$', color='black', fontsize=13, ha='center', va='top')

    # Configure axes with arrows at end (textbook style)
    ax.set_xlim(-0.1, 4.9)
    ax.set_ylim(-0.06, 1.12)

    # Remove all ticks and ticklabels
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove default spines
    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_visible(False)

    # Draw arrow axes
    ax.annotate(
        '', xy=(4.9, 0), xytext=(-0.05, 0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15)
    )
    ax.annotate(
        '', xy=(-0.05, 1.10), xytext=(-0.05, 0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15)
    )

    ax.grid(False)
    plt.tight_layout()

    for sdir in save_dirs:
        out_path = os.path.join(sdir, "fig2_06_probability_density_and_cdf.png")
        save_plot(fig, out_path)

    plt.close(fig)


def generate_figure_2_7(save_dirs=("result", "2/result")):
    """
    Figure 2.7: Plots of:
      - Uniform distribution over (-1, 1) in red
      - Exponential distribution with lambda = 1 in blue
      - Laplace distribution with mu = 1, gamma = 1 in green
      Range x in [-2, 4].
    """
    u_dist = UniformDistribution(-1.0, 1.0)
    e_dist = ExponentialDistribution(1.0)
    l_dist = LaplaceDistribution(1.0, 1.0)

    # Prepare piecewise curves with sharp step boundaries
    # 1. Uniform
    x_u_plot = [-2.0, -1.0, -1.0, 1.0, 1.0, 4.0]
    y_u_plot = [0.0, 0.0, 0.5, 0.5, 0.0, 0.0]

    # 2. Exponential (0 for x < 0, vertical jump to 1.0 at x=0, then exp(-x))
    x_e_neg = np.linspace(-2.0, 0.0, 200)
    y_e_neg = np.zeros_like(x_e_neg)
    x_e_pos = np.linspace(0.0, 4.0, 400)
    y_e_pos = np.exp(-x_e_pos)
    x_e_plot = np.concatenate([x_e_neg, [0.0], x_e_pos])
    y_e_plot = np.concatenate([y_e_neg, [1.0], y_e_pos])

    # 3. Laplace: mu=1, gamma=1
    x_l_plot = np.linspace(-2.0, 4.0, 600)
    y_l_plot = l_dist.pdf(x_l_plot)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)

    # Grey vertical line at x = 0
    ax.axvline(0, color='#cccccc', linewidth=1.2, zorder=1)

    # Plot curves
    ax.plot(x_u_plot, y_u_plot, color='red', linewidth=1.8, label='Uniform', zorder=2)
    ax.plot(x_e_plot, y_e_plot, color='blue', linewidth=1.8, label='Exponential', zorder=3)
    ax.plot(x_l_plot, y_l_plot, color='green', linewidth=1.8, label='Laplace', zorder=4)

    # Limits and ticks
    ax.set_xlim(-2.0, 4.0)
    ax.set_ylim(0.0, 1.25)
    ax.set_xticks([-2, -1, 0, 1, 2, 3, 4])
    ax.set_yticks([0.0, 0.5, 1.0])

    ax.set_xlabel(r'$x$', fontsize=12, labelpad=8)
    ax.set_ylabel(r'$p(x)$', fontsize=12, labelpad=8, rotation=0, y=0.55)

    # Box frame with dark borders (as in Bishop textbook)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.3)

    ax.tick_params(direction='out', length=5, width=1.2, colors='black')
    ax.grid(False)

    plt.tight_layout()

    for sdir in save_dirs:
        out_path = os.path.join(sdir, "fig2_07_example_distributions.png")
        save_plot(fig, out_path)

    plt.close(fig)


if __name__ == "__main__":
    generate_figure_2_6()
    generate_figure_2_7()
    print("Figures 2.6 and 2.7 successfully generated and saved!")
