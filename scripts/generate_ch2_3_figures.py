"""
Generate publication-quality figures for Chapter 2 Section 2.3:
  - Figure 2.8: Plot of a Gaussian distribution showing mean mu and standard deviation 2*sigma
  - Figure 2.9: Illustration of likelihood function for the Gaussian distribution
  - Figure 2.10: Illustration of bias in maximum likelihood estimation of mean and variance
  - Figure 2.11: Schematic illustration of Gaussian conditional distribution in linear regression
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.plot_utils import setup_style, save_plot
from common.probability import (
    Gaussian1D,
    gaussian_maximum_likelihood,
    GaussianLinearRegression
)


def generate_figure_2_8(save_dirs=("result", "2/result")):
    """
    Figure 2.8: Plot of a Gaussian distribution for a single continuous variable x
    showing the mean mu and standard deviation sigma (2*sigma width arrow).
    """
    mu = 0.0
    sigma = 1.0
    g = Gaussian1D(mu=mu, sigma2=sigma**2)

    x = np.linspace(-3.5, 3.5, 600)
    y = g.pdf(x)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)

    # Plot Gaussian curve in red
    ax.plot(x, y, color='red', linewidth=2.0, zorder=3)

    # Inflection points at x = mu +- sigma, height is y_inflec = N(mu) * exp(-0.5)
    y_inflec = g.pdf(mu + sigma)

    # Draw horizontal double-headed arrow at inflection height across 2*sigma
    ax.annotate(
        '', xy=(mu + sigma, y_inflec), xytext=(mu - sigma, y_inflec),
        arrowprops=dict(arrowstyle="<|-|>", color="black", lw=1.5, mutation_scale=12)
    )
    # Label '2\sigma' above arrow
    ax.text(mu, y_inflec + 0.02, r'$2\sigma$', fontsize=13, ha='center', va='bottom')

    # Tick and label at mu on x-axis
    ax.plot([mu, mu], [-0.015, 0.015], color='black', lw=1.5)
    ax.text(mu, -0.035, r'$\mu$', fontsize=13, ha='center', va='top')

    # Arrow axes (Bishop style)
    ax.set_xlim(-3.6, 3.8)
    ax.set_ylim(-0.05, 0.46)
    ax.set_xticks([])
    ax.set_yticks([])

    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_visible(False)

    # Horizontal axis with arrow
    ax.annotate(
        '', xy=(3.7, 0), xytext=(-3.6, 0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=14)
    )
    ax.text(3.6, -0.035, r'$x$', fontsize=13, ha='center', va='top')

    # Vertical axis with arrow
    ax.annotate(
        '', xy=(-3.5, 0.44), xytext=(-3.5, 0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=14)
    )
    ax.text(-3.4, 0.42, r'$\mathcal{N}(x|\mu, \sigma^2)$', fontsize=13, ha='left', va='center')

    ax.grid(False)
    plt.tight_layout()

    for sdir in save_dirs:
        out_path = os.path.join(sdir, "fig2_08_gaussian_distribution.png")
        save_plot(fig, out_path)

    plt.close(fig)


def generate_figure_2_9(save_dirs=("result", "2/result")):
    """
    Figure 2.9: Illustration of the likelihood function for the Gaussian distribution.
    Data points {x_n} in grey on axis, vertical green lines to blue points on red curve.
    """
    mu = 0.0
    sigma = 1.0
    g = Gaussian1D(mu=mu, sigma2=sigma**2)

    x = np.linspace(-3.2, 3.5, 600)
    y = g.pdf(x)

    # 6 representative data points matching Figure 2.9 in textbook
    x_data = np.array([-2.1, -1.3, -0.4, 0.2, 1.1, 1.9])
    y_data = g.pdf(x_data)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)

    # Plot Gaussian curve in red
    ax.plot(x, y, color='red', linewidth=2.0, zorder=3)

    # Vertical green lines and points
    for xn, yn in zip(x_data, y_data):
        ax.plot([xn, xn], [0, yn], color='#1b9e77', linewidth=1.6, zorder=2)
        # Grey point on axis
        ax.scatter([xn], [0], color='#7f7f7f', s=35, zorder=4)
        # Blue point on curve
        ax.scatter([xn], [yn], color='#1f77b4', s=45, zorder=4)

    # Annotations matching Figure 2.9
    # Label the rightmost point xn and N(xn|mu, sigma2)
    xn_target = x_data[-1]
    yn_target = y_data[-1]
    ax.text(xn_target, -0.035, r'$x_n$', fontsize=13, ha='center', va='top')
    ax.text(xn_target + 0.15, yn_target + 0.035, r'$\mathcal{N}(x_n|\mu, \sigma^2)$', fontsize=12, ha='left', va='bottom')

    # Arrow axes
    ax.set_xlim(-3.4, 3.7)
    ax.set_ylim(-0.05, 0.46)
    ax.set_xticks([])
    ax.set_yticks([])

    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_visible(False)

    # Horizontal axis
    ax.annotate(
        '', xy=(3.6, 0), xytext=(-3.3, 0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=14)
    )
    ax.text(3.5, -0.035, r'$x$', fontsize=13, ha='center', va='top')

    # Vertical axis
    ax.annotate(
        '', xy=(-3.2, 0.44), xytext=(-3.2, 0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=14)
    )
    ax.text(-3.45, 0.25, r'$p(x)$', fontsize=13, ha='right', va='center')

    ax.grid(False)
    plt.tight_layout()

    for sdir in save_dirs:
        out_path = os.path.join(sdir, "fig2_09_gaussian_likelihood.png")
        save_plot(fig, out_path)

    plt.close(fig)


def generate_figure_2_10(save_dirs=("result", "2/result")):
    """
    Figure 2.10: Illustration of how bias arises when using maximum likelihood to
    determine the mean and variance of a Gaussian (3 subplots, N=2 data points).
    """
    true_mu = 0.0
    true_sigma = 1.0
    g_true = Gaussian1D(mu=true_mu, sigma2=true_sigma**2)

    x = np.linspace(-3.2, 3.2, 600)
    y_true = g_true.pdf(x)

    # 3 data sets of 2 points each (N=2) matching Figure 2.10
    datasets = [
        np.array([-1.7, -0.9]),   # Both points left of mu
        np.array([-0.42, 0.42]),  # Points straddle mu symmetrically
        np.array([0.9, 1.7])      # Both points right of mu
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), dpi=300)

    for i, (ax, data) in enumerate(zip(axes, datasets)):
        mle = gaussian_maximum_likelihood(data)
        mu_ml = mle["mu_ML"]
        sigma_ml = mle["sigma_ML"]
        g_fit = Gaussian1D(mu=mu_ml, sigma2=sigma_ml**2)
        y_fit = g_fit.pdf(x)

        # Baseline horizontal axis
        ax.axhline(0, color='blue', linewidth=1.2, zorder=1)

        # Dashed vertical line at true mean mu
        ax.axvline(true_mu, color='#7f7f7f', linestyle='--', linewidth=1.2, zorder=2)
        ax.text(true_mu, -0.08, r'$\mu$', fontsize=13, ha='center', va='top')

        # True distribution in red (broad curve)
        ax.plot(x, y_true, color='red', linewidth=1.8, zorder=3)

        # Fitted distribution in blue (narrow, taller curve)
        ax.plot(x, y_fit, color='blue', linewidth=1.8, zorder=4)

        # Two data points in green on baseline
        ax.scatter(data, [0, 0], color='#2ca02c', s=45, zorder=5)

        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-0.12, 1.25)
        ax.set_xticks([])
        ax.set_yticks([])

        for s in ['top', 'right', 'left', 'bottom']:
            ax.spines[s].set_visible(False)

        ax.grid(False)

    plt.tight_layout()

    for sdir in save_dirs:
        out_path = os.path.join(sdir, "fig2_10_mle_bias.png")
        save_plot(fig, out_path)

    plt.close(fig)


def generate_figure_2_11(save_dirs=("result", "2/result")):
    """
    Figure 2.11: Schematic illustration of a Gaussian conditional distribution for t
    given x, in which the mean is given by the polynomial function y(x, w) and
    the variance is given by sigma^2.
    """
    rng = np.random.default_rng(42)

    # Generate synthetic training points around a smooth polynomial
    N = 35
    x_train = np.sort(rng.uniform(0.1, 2.9, N))
    # True underlying curve: smooth cubic polynomial
    true_poly = lambda x: 0.2 + 0.9 * x - 0.45 * x**2 + 0.15 * x**3
    noise_sigma = 0.14
    t_train = true_poly(x_train) + rng.normal(0, noise_sigma, N)

    # Fit polynomial model of degree 3
    model = GaussianLinearRegression(degree=3)
    model.fit(x_train, t_train)

    x_dense = np.linspace(0.05, 3.0, 400)
    y_dense, _ = model.predict(x_dense)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Axes with arrows
    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_visible(False)

    ax.annotate(
        '', xy=(3.3, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=14)
    )
    ax.text(3.35, 0, r'$x$', fontsize=13, ha='left', va='center')

    ax.annotate(
        '', xy=(0, 2.05), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=14)
    )
    ax.text(0.02, 2.08, r'$t$', fontsize=13, ha='center', va='bottom')

    # Data points in blue
    ax.scatter(x_train, t_train, color='#1f77b4', s=25, zorder=3)

    # Regression curve y(x, w) in red
    ax.plot(x_dense, y_dense, color='red', linewidth=2.0, zorder=4)
    # Place label y(x, w) slightly above the curve near x=2.5
    ax.text(2.55, 1.72, r'$y(x, \mathbf{w})$', color='black', fontsize=12, ha='left', va='bottom')

    # Vertical conditional slice at x0
    x0 = 2.05
    y0, _ = model.predict(np.array([x0]))
    y0 = float(y0[0])

    # Vertical slice line from baseline y=0 up to y=1.75
    ax.plot([x0, x0], [0, 1.75], color='#555555', linewidth=1.2, zorder=2)
    ax.text(x0, -0.1, r'$x_0$', fontsize=13, ha='center', va='top')

    # Draw Gaussian bell curve along vertical line at x0 (rotated 90 deg)
    t_slice = np.linspace(y0 - 3.2 * noise_sigma, y0 + 3.2 * noise_sigma, 400)
    gauss_slice = Gaussian1D(mu=y0, sigma2=noise_sigma**2)
    pdf_vals = gauss_slice.pdf(t_slice)
    # Scale density horizontally for visual display
    h_scale = 0.12 / np.max(pdf_vals)
    x_curve = x0 + pdf_vals * h_scale

    # Plot rotated Gaussian in blue
    ax.plot(x_curve, t_slice, color='blue', linewidth=1.8, zorder=5)

    # Label for conditional distribution
    ax.text(x0 + 0.14, y0 - 0.28, r'$p(t|x_0, \mathbf{w}, \sigma^2)$', fontsize=11, color='black', ha='left', va='center')

    ax.set_xlim(-0.1, 3.5)
    ax.set_ylim(-0.15, 2.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    plt.tight_layout()

    for sdir in save_dirs:
        out_path = os.path.join(sdir, "fig2_11_linear_regression_conditional_gaussian.png")
        save_plot(fig, out_path)

    plt.close(fig)


def generate_all_figures():
    """Generate all figures for Section 2.3."""
    print("Generating Figure 2.8...")
    generate_figure_2_8()
    print("Generating Figure 2.9...")
    generate_figure_2_9()
    print("Generating Figure 2.10...")
    generate_figure_2_10()
    print("Generating Figure 2.11...")
    generate_figure_2_11()
    print("All Chapter 2 Section 2.3 figures generated successfully!")


if __name__ == "__main__":
    generate_all_figures()
