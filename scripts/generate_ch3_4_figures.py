"""
Generate Section 3.4 figures:
- fig3_13_exp_family_geometry.png
- fig3_14_exp_family_members.png
- fig3_15_sufficient_statistics_online.png
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from common.plot_utils import setup_style

setup_style()


def plot_figure_3_13(save_paths=None, show=False):
    """
    Figure 3.13: Geometry of the Exponential Family:
    - Convex log-partition function A(eta)
    - Gradient mapping nabla A(eta) = E[u(x)] (Legendre transformation)
    - Hessian nabla^2 A(eta) = Var[u(x)] (Fisher information metric)
    """
    eta = np.linspace(-5.0, 5.0, 300)
    # Bernoulli exponential family: A(eta) = ln(1 + e^eta)
    A = np.log1p(np.exp(eta))
    mu = 1.0 / (1.0 + np.exp(-eta))
    var = mu * (1.0 - mu)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    
    # 1. Log-partition function A(eta)
    ax1 = axes[0]
    ax1.plot(eta, A, color='#1E56A0', linewidth=2.5, label=r'$A(\eta) = \ln(1 + e^{\eta})$')
    # Tangents at eta = -1.5, 0, 1.5
    for et0, col in zip([-2.0, 0.0, 2.0], ['#E02020', '#2E8B57', '#9370DB']):
        A0 = np.log1p(np.exp(et0))
        slope0 = 1.0 / (1.0 + np.exp(-et0))
        tangent = A0 + slope0 * (eta - et0)
        ax1.plot(eta, tangent, linestyle='--', color=col, alpha=0.8,
                 label=rf'Tangent at $\eta={et0:+.1f}$')
        ax1.scatter([et0], [A0], color=col, s=40, zorder=5)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-0.5, 5.5)
    ax1.set_xlabel(r'Natural Parameter $\eta$', fontsize=11)
    ax1.set_ylabel(r'Log-partition $A(\eta) = -\ln g(\eta)$', fontsize=11)
    ax1.set_title(r'(a) Strictly Convex $A(\eta)$ & Supporting Tangents', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9, frameon=True)
    ax1.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for s in ax1.spines.values():
        s.set_linewidth(0.8)

    # 2. Gradient mapping nabla A(eta) = E[u]
    ax2 = axes[1]
    ax2.plot(eta, mu, color='#E02020', linewidth=2.5, label=r'$\nabla A(\eta) = \sigma(\eta) = \mathbb{E}[x]$')
    ax2.axhline(0.0, color='gray', linestyle=':', alpha=0.6)
    ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax2.axvline(0.0, color='gray', linestyle=':', alpha=0.6)
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel(r'Natural Parameter $\eta$', fontsize=11)
    ax2.set_ylabel(r'Expectation Parameter $\mu = \mathbb{E}[u(x)]$', fontsize=11)
    ax2.set_title(r'(b) Dual Mapping $\nabla A(\eta): \mathcal{H} \to \mathcal{M}$', fontsize=11)
    ax2.legend(loc='lower right', fontsize=10, frameon=True)
    ax2.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for s in ax2.spines.values():
        s.set_linewidth(0.8)

    # 3. Hessian nabla^2 A(eta) = Var[u]
    ax3 = axes[2]
    ax3.plot(eta, var, color='#2E8B57', linewidth=2.5, label=r'$\nabla^2 A(\eta) = \mu(1-\mu) = \mathrm{Var}[x]$')
    ax3.fill_between(eta, var, color='#2E8B57', alpha=0.2)
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(-0.02, 0.3)
    ax3.set_xlabel(r'Natural Parameter $\eta$', fontsize=11)
    ax3.set_ylabel(r'Curvature / Variance $\nabla^2 A(\eta)$', fontsize=11)
    ax3.set_title(r'(c) Curvature as Variance / Fisher Information', fontsize=11)
    ax3.legend(loc='upper right', fontsize=10, frameon=True)
    ax3.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for s in ax3.spines.values():
        s.set_linewidth(0.8)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.13 saved successfully to: {p}")
    if show:
        plt.show()
    plt.close(fig)
    return fig, axes


def plot_figure_3_14(save_paths=None, show=False):
    """
    Figure 3.14: Major Exponential Family Members and their Canonical Mappings.
    - (a) Bernoulli: mu in (0, 1) <-> eta in R
    - (b) Multinomial (3 categories): simplex <-> R^2
    - (c) Univariate Gaussian: (mu, sigma^2) <-> (eta1, eta2)
    - (d) Von Mises: (theta0, m) <-> (eta1, eta2) on unit disk / plane
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    
    # (a) Bernoulli
    ax_a = axes[0, 0]
    mu_vals = np.linspace(0.01, 0.99, 200)
    eta_vals = np.log(mu_vals / (1.0 - mu_vals))
    ax_a.plot(mu_vals, eta_vals, color='#1E56A0', linewidth=2.2)
    ax_a.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax_a.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax_a.set_xlabel(r'Mean Parameter $\mu \in (0, 1)$', fontsize=11)
    ax_a.set_ylabel(r'Natural Parameter $\eta = \ln(\mu / (1-\mu))$', fontsize=11)
    ax_a.set_title(r'(a) Bernoulli: Logit Link ($\mathbf{u}(x)=x$)', fontsize=12)
    ax_a.tick_params(direction='in', top=True, right=True)

    # (b) Multinomial (3-state simplex to natural coordinates)
    ax_b = axes[0, 1]
    # Plot simplex in 2D barycentric coords
    # Simplex vertices
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, np.sqrt(3)/2])
    tri = np.array([v1, v2, v3, v1])
    ax_b.plot(tri[:, 0], tri[:, 1], 'k-', linewidth=1.5)
    
    # Scatter grid of points inside simplex
    grid_pts = []
    colors = []
    for p1 in np.linspace(0.05, 0.9, 18):
        for p2 in np.linspace(0.05, 0.9 - p1, 18):
            p3 = 1.0 - p1 - p2
            if p3 > 0.02:
                xy = p1 * v1 + p2 * v2 + p3 * v3
                grid_pts.append(xy)
                # Color by eta1 = ln(p1 / p3)
                colors.append(np.log(p1 / p3))
    grid_pts = np.array(grid_pts)
    sc = ax_b.scatter(grid_pts[:, 0], grid_pts[:, 1], c=colors, cmap='coolwarm', s=25, alpha=0.85)
    plt.colorbar(sc, ax=ax_b, label=r'$\eta_1 = \ln(\mu_1 / \mu_3)$', pad=0.02)
    ax_b.text(v1[0] - 0.06, v1[1] - 0.04, r'$\mu_1=1$', fontsize=10, color='#1E56A0', fontweight='bold')
    ax_b.text(v2[0] + 0.02, v2[1] - 0.04, r'$\mu_2=1$', fontsize=10, color='#E02020', fontweight='bold')
    ax_b.text(v3[0] - 0.05, v3[1] + 0.04, r'$\mu_3=1$', fontsize=10, color='#2E8B57', fontweight='bold')
    ax_b.set_title(r'(b) Multinomial: Probability Simplex & Softmax', fontsize=12)
    ax_b.set_aspect('equal')
    ax_b.axis('off')

    # (c) Univariate Gaussian Natural Parameter space (eta1, eta2) with eta2 < 0
    ax_c = axes[1, 0]
    eta1_grid = np.linspace(-3.0, 3.0, 150)
    eta2_grid = np.linspace(-3.0, -0.1, 150)
    E1, E2 = np.meshgrid(eta1_grid, eta2_grid)
    # sigma^2 = -1 / (2*eta2), mu = -eta1 / (2*eta2)
    MU = -E1 / (2.0 * E2)
    SIG2 = -1.0 / (2.0 * E2)
    cs = ax_c.contour(E1, E2, MU, levels=np.linspace(-2, 2, 9), cmap='coolwarm', linewidths=1.2)
    ax_c.clabel(cs, inline=True, fontsize=8, fmt=r'$\mu=%.1f$')
    ax_c.axhline(0, color='black', linewidth=1.5)
    ax_c.fill_between(eta1_grid, 0, 0.5, color='gray', alpha=0.3)
    ax_c.text(0, 0.15, r'Invalid domain ($\eta_2 \geq 0$)', ha='center', fontsize=10, color='darkred')
    ax_c.set_xlim(-3, 3)
    ax_c.set_ylim(-3, 0.5)
    ax_c.set_xlabel(r'Natural Parameter $\eta_1 = \mu / \sigma^2$', fontsize=11)
    ax_c.set_ylabel(r'Natural Parameter $\eta_2 = -1 / (2\sigma^2)$', fontsize=11)
    ax_c.set_title(r'(c) Gaussian: $\mathbf{u}(x) = (x, x^2)^T$ Domain ($\eta_2 < 0$)', fontsize=12)
    ax_c.tick_params(direction='in', top=True, right=True)

    # (d) Von Mises Natural Parameter space eta = (m cos theta0, m sin theta0)
    ax_d = axes[1, 1]
    radii = [1.0, 2.5, 4.0]
    circle_theta = np.linspace(0, 2*np.pi, 200)
    for r in radii:
        ax_d.plot(r * np.cos(circle_theta), r * np.sin(circle_theta), 'k--', alpha=0.4)
        ax_d.text(r * 0.707 + 0.1, r * 0.707 + 0.1, f'$m={r:.1f}$', fontsize=8, color='gray')
    
    # Vector arrows
    test_params = [(np.pi/4, 4.0, '#E02020', r'$\theta_0=\frac{\pi}{4}, m=4$'),
                   (3*np.pi/4, 2.5, '#1E56A0', r'$\theta_0=\frac{3\pi}{4}, m=2.5$'),
                   (-np.pi/3, 3.2, '#2E8B57', r'$\theta_0=-\frac{\pi}{3}, m=3.2$')]
    for th0, m_val, col, lbl in test_params:
        e1 = m_val * np.cos(th0)
        e2 = m_val * np.sin(th0)
        ax_d.annotate('', xy=(e1, e2), xytext=(0, 0),
                      arrowprops=dict(facecolor=col, edgecolor=col, width=1.8, headwidth=8))
        ax_d.scatter([e1], [e2], color=col, s=40, zorder=5, label=lbl)
    
    ax_d.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax_d.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax_d.set_xlim(-5, 5)
    ax_d.set_ylim(-5, 5)
    ax_d.set_aspect('equal')
    ax_d.set_xlabel(r'Natural Parameter $\eta_1 = m \cos \theta_0$', fontsize=11)
    ax_d.set_ylabel(r'Natural Parameter $\eta_2 = m \sin \theta_0$', fontsize=11)
    ax_d.set_title(r'(d) Von Mises: $\mathbf{u}(\theta) = (\cos\theta, \sin\theta)^T$', fontsize=12)
    ax_d.legend(loc='lower left', fontsize=9, frameon=True)
    ax_d.tick_params(direction='in', top=True, right=True)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.14 saved successfully to: {p}")
    if show:
        plt.show()
    plt.close(fig)
    return fig, axes


def plot_figure_3_15(save_paths=None, show=False):
    """
    Figure 3.15: Sufficient Statistics and Online / Stream Estimation (Bishop Section 3.4.1).
    - Sequential MLE update of Gaussian mean and variance using only running sums sum(x_n), sum(x_n^2).
    - Memory comparison: O(1) fixed space vs O(N) linear storage.
    """
    np.random.seed(42)
    true_mu = 3.5
    true_sigma2 = 2.0
    N = 500
    X = np.random.normal(true_mu, np.sqrt(true_sigma2), size=N)
    
    # Online accumulation of sufficient statistics
    sum_x = np.cumsum(X)
    sum_x2 = np.cumsum(X**2)
    n_arr = np.arange(1, N + 1)
    
    # Online MLE
    mu_mle_seq = sum_x / n_arr
    sigma2_mle_seq = (sum_x2 / n_arr) - (mu_mle_seq**2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300)
    
    # (a) Online convergence of parameter estimates
    ax1.plot(n_arr, mu_mle_seq, color='#1E56A0', linewidth=1.8, label=r'$\mu^{\mathrm{ML}}_N = \frac{1}{N}\sum x_n$')
    ax1.axhline(true_mu, color='#1E56A0', linestyle='--', alpha=0.7, label=r'True $\mu = 3.5$')
    
    ax1.plot(n_arr, sigma2_mle_seq, color='#E02020', linewidth=1.8, label=r'${\sigma^2}^{\mathrm{ML}}_N = \frac{1}{N}\sum x_n^2 - (\mu^{\mathrm{ML}}_N)^2$')
    ax1.axhline(true_sigma2, color='#E02020', linestyle='--', alpha=0.7, label=r'True $\sigma^2 = 2.0$')
    
    ax1.set_xlim(1, N)
    ax1.set_ylim(0, 5.5)
    ax1.set_xlabel(r'Sample Count $N$ (Stream Step)', fontsize=11)
    ax1.set_ylabel('Parameter Estimate', fontsize=11)
    ax1.set_title('(a) Online Stream MLE via Running Sufficient Statistics', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9, frameon=True)
    ax1.tick_params(direction='in', top=True, right=True)
    for s in ax1.spines.values():
        s.set_linewidth(0.8)

    # (b) Memory Footprint: Buffer O(N) vs Sufficient Statistics O(1)
    bytes_per_float = 8
    raw_buffer_bytes = n_arr * bytes_per_float
    # Sufficient statistics: just 2 numbers (sum_x, sum_x2) and 1 counter (N) -> 24 bytes
    suff_stat_bytes = np.full_like(n_arr, 24)
    
    ax2.plot(n_arr, raw_buffer_bytes / 1024.0, color='#E02020', linewidth=2.0,
             label=r'Raw Data Buffering: $\mathcal{O}(N)$ Storage')
    ax2.plot(n_arr, suff_stat_bytes / 1024.0, color='#2E8B57', linewidth=2.2,
             label=r'Sufficient Statistics $\sum \mathbf{u}(x_n)$: $\mathcal{O}(1)$ Storage')
    ax2.fill_between(n_arr, suff_stat_bytes / 1024.0, raw_buffer_bytes / 1024.0,
                     color='#E02020', alpha=0.1)
    ax2.set_xlim(1, N)
    ax2.set_xlabel(r'Sample Count $N$', fontsize=11)
    ax2.set_ylabel('Memory Footprint (KB)', fontsize=11)
    ax2.set_title(r'(b) Memory Efficiency of Sufficient Statistics', fontsize=11)
    ax2.legend(loc='upper left', fontsize=10, frameon=True)
    ax2.tick_params(direction='in', top=True, right=True)
    for s in ax2.spines.values():
        s.set_linewidth(0.8)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.15 saved successfully to: {p}")
    if show:
        plt.show()
    plt.close(fig)
    return fig, (ax1, ax2)


if __name__ == '__main__':
    targets = [
        ('fig3_13_exp_family_geometry.png', plot_figure_3_13),
        ('fig3_14_exp_family_members.png', plot_figure_3_14),
        ('fig3_15_sufficient_statistics_online.png', plot_figure_3_15)
    ]
    for fname, func in targets:
        paths = [f"3/result/{fname}", f"result/{fname}"]
        func(save_paths=paths, show=False)
