"""
Generate publication-quality figures for Chapter 2 Section 2.3:
  - Figure 2.8: 1D Gaussian distribution showing mean mu and 2sigma width
  - Figure 2.9: Gaussian likelihood function with data points and density heights
  - Figure 2.10: Illustration of bias in maximum likelihood variance with N=2 data points (a, b, c)
  - Figure 2.11: Probabilistic linear regression with Gaussian noise and vertical predictive distribution
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.plot_utils import setup_style, save_plot
from common.probability import (
    Gaussian1D,
    gaussian_maximum_likelihood,
    simulate_gaussian_mle_bias,
    GaussianLinearRegression
)


def generate_figure_2_8(save_dirs=("result", "2/result")):
    """
    Figure 2.8: Univariate Gaussian distribution N(x | mu, sigma^2).
    Shows:
      - Red bell curve
      - Axis arrows: horizontal x, vertical N(x | mu, sigma^2)
      - Mean mu tick and label
      - Double-ended arrow between inflection points mu - sigma and mu + sigma with label 2sigma
    """
    mu = 0.0
    sigma = 1.0
    gauss = Gaussian1D(mu=mu, sigma2=sigma**2)

    x = np.linspace(-3.6, 3.6, 500)
    p_x = gauss.pdf(x)

    fig, ax = plt.subplots(figsize=(6, 4.8), dpi=300)

    # Plot Gaussian curve in red
    ax.plot(x, p_x, color='red', linewidth=2.0, zorder=3)

    # Inflection points: x = mu +- sigma, y = N(mu +- sigma | mu, sigma^2) = 1/(sqrt(2*pi)*sigma) * e^(-0.5)
    y_inflec = gauss.pdf(mu + sigma)
    # Draw double-ended arrow for 2*sigma
    ax.annotate(
        '', xy=(mu + sigma, y_inflec), xytext=(mu - sigma, y_inflec),
        arrowprops=dict(arrowstyle="<|-|>", color="black", lw=1.3, mutation_scale=12)
    )
    # Label "2\sigma" centered above arrow
    ax.text(mu, y_inflec + 0.015, r'$2\sigma$', color='black', fontsize=13, ha='center', va='bottom')

    # Tick at mu on x-axis
    ax.plot([mu, mu], [-0.015, 0.015], color='black', linewidth=1.5, zorder=4)
    ax.text(mu, -0.04, r'$\mu$', color='black', fontsize=14, ha='center', va='top')

    # Axis arrows
    ax.annotate(
        '', xy=(3.65, 0), xytext=(-3.7, 0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15)
    )
    ax.annotate(
        '', xy=(-3.65, 0.45), xytext=(-3.65, 0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15)
    )

    # Axis labels
    ax.text(3.55, -0.04, r'$x$', color='black', fontsize=14, ha='center', va='top')
    ax.text(-3.6, 0.41, r'$\mathcal{N}(x \mid \mu, \sigma^2)$', color='black', fontsize=13, ha='left', va='center')

    # Setup limits and spines
    ax.set_xlim(-3.75, 3.8)
    ax.set_ylim(-0.06, 0.46)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    for sdir in save_dirs:
        os.makedirs(sdir, exist_ok=True)
        save_plot(fig, os.path.join(sdir, "fig2_08_gaussian_distribution.png"))
    plt.close(fig)


def generate_figure_2_9(save_dirs=("result", "2/result")):
    """
    Figure 2.9: Likelihood function for a Gaussian distribution.
    Shows:
      - Red bell curve
      - Data points x_n on horizontal axis (grey dots)
      - Vertical green lines from axis to curve
      - Blue dots on curve at (x_n, p(x_n))
      - Axis arrows: x and p(x)
      - Annotations: x_n under one point, N(x_n | mu, sigma^2) near the blue point
    """
    mu = 0.0
    sigma = 1.0
    gauss = Gaussian1D(mu=mu, sigma2=sigma**2)

    x = np.linspace(-3.5, 3.5, 500)
    p_x = gauss.pdf(x)

    fig, ax = plt.subplots(figsize=(6, 4.8), dpi=300)

    # Plot Gaussian curve in red
    ax.plot(x, p_x, color='red', linewidth=2.0, zorder=3)

    # Data points x_n matching textbook Figure 2.9 distribution
    # Points from left to right: ~ -2.0, -1.0, -0.3, -0.1, 0.7, 1.0, 2.0
    x_points = np.array([-2.1, -1.0, -0.28, -0.05, 0.75, 1.05, 2.0])
    y_points = gauss.pdf(x_points)

    # Vertical green lines and dots
    for xn, yn in zip(x_points, y_points):
        # Vertical green line
        ax.plot([xn, xn], [0, yn], color='#008000', linewidth=1.5, zorder=2)
        # Grey dot on x-axis
        ax.plot(xn, 0, marker='o', color='#777777', markersize=6.5, zorder=4)
        # Blue dot on curve
        ax.plot(xn, yn, marker='o', color='#0000ff', markersize=6.5, zorder=4)

    # Axis arrows
    ax.annotate(
        '', xy=(3.55, 0), xytext=(-3.6, 0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15)
    )
    ax.annotate(
        '', xy=(-3.55, 0.45), xytext=(-3.55, 0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15)
    )

    # Labels
    ax.text(3.45, -0.04, r'$x$', color='black', fontsize=14, ha='center', va='top')
    ax.text(-3.55, 0.25, r'$p(x)$', color='black', fontsize=13, ha='right', va='center')

    # Annotation for x_n and N(x_n | mu, sigma^2) at rightmost point x_points[-1]
    xn_labeled = x_points[-1]
    yn_labeled = y_points[-1]
    ax.text(xn_labeled, -0.04, r'$x_n$', color='black', fontsize=13, ha='center', va='top')
    ax.text(xn_labeled + 0.1, yn_labeled + 0.04, r'$\mathcal{N}(x_n \mid \mu, \sigma^2)$',
            color='black', fontsize=12, ha='left', va='bottom')

    # Limits and spines
    ax.set_xlim(-3.7, 3.7)
    ax.set_ylim(-0.06, 0.46)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    for sdir in save_dirs:
        os.makedirs(sdir, exist_ok=True)
        save_plot(fig, os.path.join(sdir, "fig2_09_gaussian_likelihood.png"))
    plt.close(fig)


def generate_figure_2_10(save_dirs=("result", "2/result")):
    """
    Figure 2.10: Illustration of how bias arises in maximum likelihood variance.
    3 subplots (a, b, c) showing:
      - True Gaussian distribution (red curve)
      - True mean mu (dashed grey vertical line)
      - Green dots: N=2 sample data points
      - Blue curve: fitted Gaussian distribution with sample mean mu_ML and variance sigma^2_ML
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=300)

    # True distribution parameters
    mu_true = 0.0
    sigma_true = 1.0
    gauss_true = Gaussian1D(mu=mu_true, sigma2=sigma_true**2)

    x_grid = np.linspace(-3.2, 3.2, 400)
    pdf_true = gauss_true.pdf(x_grid)

    # 3 representative datasets of N=2 points
    # (a) both points to the left of mu
    # (b) points symmetric / straddling mu
    # (c) both points to the right of mu
    datasets = [
        np.array([-1.5, -0.6]),
        np.array([-0.45, 0.45]),
        np.array([0.7, 1.6])
    ]

    for idx, (ax, data) in enumerate(zip(axes, datasets)):
        mu_ml = np.mean(data)
        sigma2_ml = np.mean((data - mu_ml)**2)
        sigma_ml = np.sqrt(sigma2_ml)
        gauss_ml = Gaussian1D(mu=mu_ml, sigma2=sigma2_ml)
        pdf_ml = gauss_ml.pdf(x_grid)

        # Baseline horizontal axis
        ax.plot([-3.2, 3.2], [0, 0], color='black', linewidth=1.5, zorder=1)

        # Vertical dashed line at true mu
        ax.axvline(mu_true, color='#888888', linestyle='--', linewidth=1.4, zorder=2)

        # Plot true distribution in red
        ax.plot(x_grid, pdf_true, color='red', linewidth=2.0, zorder=3)

        # Plot fitted ML distribution in blue
        ax.plot(x_grid, pdf_ml, color='blue', linewidth=2.0, zorder=4)

        # Plot green data points
        ax.plot(data, [0, 0], marker='o', color='#008000', markersize=8, linestyle='None', zorder=5)

        # Label mu under dashed line
        ax.text(mu_true, -0.12, r'$\mu$', color='black', fontsize=14, ha='center', va='top')

        # Subplot letter label: (a), (b), (c)
        label_letter = ['(a)', '(b)', '(c)'][idx]
        ax.set_title(label_letter, fontsize=12, pad=8)

        # Settings
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-0.15, 1.05)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ['top', 'right', 'left', 'bottom']:
            ax.spines[s].set_visible(False)
        ax.grid(False)

    plt.tight_layout()
    for sdir in save_dirs:
        os.makedirs(sdir, exist_ok=True)
        save_plot(fig, os.path.join(sdir, "fig2_10_mle_bias.png"))
    plt.close(fig)


def generate_figure_2_11(save_dirs=("result", "2/result")):
    """
    Figure 2.11: Probabilistic Linear Regression with Gaussian Noise.
    Shows:
      - Synthetic data points (blue dots) scattered with Gaussian noise
      - Regression curve y(x, w) in red
      - Vertical line at x_0
      - Vertically oriented Gaussian density p(t | x_0, w, sigma^2) in blue
      - Axes labeled x and t
      - Annotations: y(x, w), x_0, p(t | x_0, w, sigma^2)
    """
    rng = np.random.default_rng(42)
    N = 50
    x_data = rng.uniform(-1.0, 1.0, size=N)

    # Underlying nonlinear curve: gentle inflection in middle, curves up on right, down on left
    noise_sigma = 0.18
    y_true_func = lambda x: 0.7 * (x**3) + 0.1 * np.sin(np.pi * x)
    t_data = y_true_func(x_data) + rng.normal(0, noise_sigma, size=N)

    # Fit a degree-3 regression model
    model = GaussianLinearRegression(degree=3)
    model.fit(x_data, t_data)

    x_line = np.linspace(-1.05, 1.05, 300)
    y_line, _ = model.predict(x_line)

    fig, ax = plt.subplots(figsize=(6, 5.2), dpi=300)

    # Scatter synthetic data points in blue
    ax.scatter(x_data, t_data, color='#0000ff', s=24, zorder=3)

    # Regression curve in red
    ax.plot(x_line, y_line, color='red', linewidth=2.0, zorder=4)

    # Evaluation point x_0
    x0 = -0.05
    y0, sigma_ml = model.predict(np.array([x0]))
    y0 = y0[0]

    # Vertical grey line at x_0
    ax.plot([x0, x0], [-0.85, 0.85], color='#888888', linewidth=1.5, zorder=2)

    # Vertical Gaussian density at x_0
    # Bell curve points outwards (to the right) along x
    t_range = np.linspace(y0 - 3.5 * sigma_ml, y0 + 3.5 * sigma_ml, 300)
    # Density values
    p_t = (1.0 / (np.sqrt(2 * np.pi) * sigma_ml)) * np.exp(-0.5 * ((t_range - y0) / sigma_ml)**2)
    p_t = p_t - p_t[0]  # ensure ends touch x0 exactly
    # Scale density for visual proportion
    scale_factor = 0.12 / np.max(p_t)
    x_curve = x0 + p_t * scale_factor

    # Plot vertical Gaussian bell curve in blue
    ax.plot(x_curve, t_range, color='#0000ff', linewidth=2.0, zorder=5)

    # Axis arrows
    ax.annotate(
        '', xy=(1.15, -0.85), xytext=(-1.12, -0.85),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15)
    )
    ax.annotate(
        '', xy=(-1.1, 0.95), xytext=(-1.1, -0.85),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15)
    )

    # Axis labels
    ax.text(1.15, -0.92, r'$x$', color='black', fontsize=14, ha='center', va='top')
    ax.text(-1.15, 0.95, r'$t$', color='black', fontsize=14, ha='right', va='center')

    # Annotations matching Figure 2.11
    ax.text(x0, -0.92, r'$x_0$', color='black', fontsize=13, ha='center', va='top')
    ax.text(0.9, 0.72, r'$y(x, \mathbf{w})$', color='black', fontsize=13, ha='right', va='bottom')
    ax.text(x0 + 0.05, y0 - 0.35, r'$p(t \mid x_0, \mathbf{w}, \sigma^2)$', color='black', fontsize=12, ha='left', va='center')

    # Limits and spines
    ax.set_xlim(-1.18, 1.25)
    ax.set_ylim(-1.0, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    for sdir in save_dirs:
        os.makedirs(sdir, exist_ok=True)
        save_plot(fig, os.path.join(sdir, "fig2_11_linear_regression.png"))
        save_plot(fig, os.path.join(sdir, "fig2_11_linear_regression_conditional_gaussian.png"))
    plt.close(fig)


if __name__ == "__main__":
    generate_figure_2_8()
    generate_figure_2_9()
    generate_figure_2_10()
    generate_figure_2_11()
    print("Figures 2.8, 2.9, 2.10, and 2.11 successfully generated and saved!")
