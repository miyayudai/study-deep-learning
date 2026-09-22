import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd

# Paths to save
SAVE_DIRS = ['3/result', 'result']

def save_fig(fig, filename):
    for d in SAVE_DIRS:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, filename)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")

# -------------------------------------------------------------
# Figure 3.2: Central Limit Theorem (N = 1, 2, 10)
# -------------------------------------------------------------
def plot_figure_3_2():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=300, sharey=True)
    N_values = [1, 2, 10]
    num_samples = 200000

    for ax, N in zip(axes, N_values):
        # Draw N uniform samples and compute mean
        samples = np.mean(np.random.uniform(0.0, 1.0, size=(num_samples, N)), axis=1)
        # Bishop textbook style: yellow bars with thin black borders
        counts, bins, patches = ax.hist(
            samples, bins=50, range=(0.0, 1.0), density=True,
            color='#E8BA3A', edgecolor='black', linewidth=0.5
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 3.5)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0, 1, 2, 3])
        ax.text(0.5, 3.1, f"$N = {N}$", ha='center', fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)

    plt.tight_layout()
    save_fig(fig, 'fig3_02_central_limit_theorem.png')
    plt.close(fig)

# -------------------------------------------------------------
# Figure 3.3: Geometry of the Gaussian
# -------------------------------------------------------------
def plot_figure_3_3():
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=300)
    mu = np.array([2.5, 2.5])
    # Tilted covariance
    theta = np.radians(35)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    lambda1, lambda2 = 2.2, 0.6
    Sigma = R @ np.diag([lambda1, lambda2]) @ R.T

    # Eigenvalues and eigenvectors
    eigvals, eigvecs = la.eigh(Sigma)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    u1, u2 = eigvecs[:, 0], eigvecs[:, 1]
    l1, l2 = eigvals[0], eigvals[1]

    # Ellipse points where (x - mu)^T Sigma^-1 (x - mu) = 1
    t = np.linspace(0, 2*np.pi, 200)
    circle = np.array([np.cos(t), np.sin(t)]) # shape (2, 200)
    ellipse = mu[:, None] + eigvecs @ (np.diag(np.sqrt(eigvals)) @ circle)

    # Plot constant density ellipse in red
    ax.plot(ellipse[0], ellipse[1], color='#E02020', linewidth=2.0, zorder=4)

    # Plot axes lines passing through mu
    axis_len1 = 2.0 * np.sqrt(l1)
    axis_len2 = 2.0 * np.sqrt(l2)
    ax.plot([mu[0] - axis_len1*u1[0], mu[0] + axis_len1*u1[0]],
            [mu[1] - axis_len1*u1[1], mu[1] + axis_len1*u1[1]],
            color='gray', linestyle='--', linewidth=1.0, zorder=2)
    ax.plot([mu[0] - axis_len2*u2[0], mu[0] + axis_len2*u2[0]],
            [mu[1] - axis_len2*u2[1], mu[1] + axis_len2*u2[1]],
            color='gray', linestyle='--', linewidth=1.0, zorder=2)

    # Arrows for eigenvectors scaled by sqrt(lambda)
    ax.annotate('', xy=mu + np.sqrt(l1)*u1, xytext=mu,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=mu + np.sqrt(l2)*u2, xytext=mu,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Center point mu
    ax.scatter([mu[0]], [mu[1]], color='black', s=30, zorder=5)
    ax.text(mu[0] - 0.25, mu[1] - 0.35, r'$\boldsymbol{\mu}$', fontsize=13)

    # Labels for arrows
    p1 = mu + 0.65 * np.sqrt(l1)*u1
    ax.text(p1[0] + 0.1, p1[1] - 0.25, r'$\lambda_1^{1/2}\mathbf{u}_1$', fontsize=12)
    p2 = mu + 0.65 * np.sqrt(l2)*u2
    ax.text(p2[0] - 0.55, p2[1] + 0.15, r'$\lambda_2^{1/2}\mathbf{u}_2$', fontsize=12)

    # u1, u2 labels at axis ends
    end1 = mu + axis_len1 * u1
    ax.text(end1[0] + 0.1, end1[1], r'$\mathbf{u}_1$', fontsize=12, fontweight='bold')
    end2 = mu + axis_len2 * u2
    ax.text(end2[0], end2[1] + 0.1, r'$\mathbf{u}_2$', fontsize=12, fontweight='bold')

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)

    save_fig(fig, 'fig3_03_gaussian_geometry.png')
    plt.close(fig)

# -------------------------------------------------------------
# Figure 3.4: Contours for General, Diagonal, and Isotropic Covariances
# -------------------------------------------------------------
def plot_figure_3_4():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), dpi=300)
    titles = [
        "(a) General form\n(arbitrary covariance)",
        "(b) Diagonal\n(axis-aligned)",
        "(c) Spherical / Isotropic\n" + r"($\mathbf{\Sigma} = \sigma^2 \mathbf{I}$)"
    ]

    # Grid
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Three covariances
    cov_general = np.array([[1.5, 0.9], [0.9, 1.0]])
    cov_diag = np.array([[1.8, 0.0], [0.0, 0.6]])
    cov_spherical = np.array([[1.0, 0.0], [0.0, 1.0]])

    covs = [cov_general, cov_diag, cov_spherical]

    for ax, cov, title in zip(axes, covs, titles):
        inv_cov = la.inv(cov)
        # Compute Mahalanobis distance squared on grid
        # (pos) @ inv_cov * pos
        quad = np.einsum('...i,ij,...j->...', pos, inv_cov, pos)
        density = np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(la.det(cov)))

        levels = np.linspace(0.02, density.max() * 0.9, 6)
        ax.contour(X, Y, density, levels=levels, colors='#E02020', linewidths=1.5)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel(r'$x_1$', fontsize=11)
        ax.set_ylabel(r'$x_2$', fontsize=11)
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='major', direction='in', top=True, right=True)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)

    plt.tight_layout()
    save_fig(fig, 'fig3_04_covariance_geometries.png')
    plt.close(fig)

# -------------------------------------------------------------
# Figure 3.5: Conditional and Marginal Gaussians
# -------------------------------------------------------------
def plot_figure_3_5():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)
    
    # 2D Gaussian over (xa, xb)
    mu = np.array([0.5, 0.5])
    # sigma_a^2 = 0.05, sigma_b^2 = 0.05, correlation rho = 0.8
    sigma = np.array([[0.04, 0.032], [0.032, 0.04]])
    inv_sigma = la.inv(sigma)

    xa = np.linspace(0.0, 1.0, 200)
    xb = np.linspace(0.0, 1.0, 200)
    XA, XB = np.meshgrid(xa, xb)
    pos = np.dstack((XA - mu[0], XB - mu[1]))
    quad = np.einsum('...i,ij,...j->...', pos, inv_sigma, pos)
    density = np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(la.det(sigma)))

    # (a) Contours of p(xa, xb)
    levels = np.linspace(0.5, density.max() * 0.95, 7)
    ax1.contour(XA, XB, density, levels=levels, colors='#E02020', linewidths=1.4)
    # Horizontal line at xb = 0.7
    xb_val = 0.7
    ax1.axhline(xb_val, color='gray', linestyle='--', linewidth=1.2)
    ax1.text(0.1, xb_val + 0.03, r'$x_b = 0.7$', fontsize=11)
    ax1.text(0.65, 0.35, r'$p(x_a, x_b)$', fontsize=12, color='#E02020')

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel(r'$x_a$', fontsize=12)
    ax1.set_ylabel(r'$x_b$', fontsize=12)
    ax1.set_title('(a) Joint distribution contours', fontsize=12)
    ax1.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for spine in ax1.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)

    # (b) Marginal p(xa) and Conditional p(xa | xb = 0.7)
    # Marginal p(xa): N(mu_a, sigma_aa)
    mu_a = mu[0]
    var_a = sigma[0, 0]
    p_xa_marginal = (1.0 / np.sqrt(2 * np.pi * var_a)) * np.exp(-0.5 * (xa - mu_a)**2 / var_a)

    # Conditional p(xa | xb = 0.7):
    # mu_{a|b} = mu_a + sigma_ab / sigma_bb * (xb - mu_b)
    # var_{a|b} = sigma_aa - sigma_ab^2 / sigma_bb
    mu_cond = mu_a + (sigma[0, 1] / sigma[1, 1]) * (xb_val - mu[1])
    var_cond = sigma[0, 0] - (sigma[0, 1]**2 / sigma[1, 1])
    p_xa_conditional = (1.0 / np.sqrt(2 * np.pi * var_cond)) * np.exp(-0.5 * (xa - mu_cond)**2 / var_cond)

    ax2.plot(xa, p_xa_marginal, color='#1E56A0', linewidth=2.0, label=r'$p(x_a)$ (marginal)')
    ax2.plot(xa, p_xa_conditional, color='#E02020', linewidth=2.0, label=r'$p(x_a | x_b = 0.7)$ (conditional)')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 7)
    ax2.set_xlabel(r'$x_a$', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('(b) Marginal and conditional distributions', fontsize=12)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax2.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for spine in ax2.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)

    plt.tight_layout()
    save_fig(fig, 'fig3_05_conditional_marginal.png')
    plt.close(fig)

# -------------------------------------------------------------
# Figure 3.6: Old Faithful Single Gaussian vs GMM
# -------------------------------------------------------------
def plot_figure_3_6():
    df = pd.read_csv('common/data/faithful.csv')
    X = df[['duration', 'waiting']].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300, sharex=True, sharey=True)

    # Grid for contour
    x_grid = np.linspace(1.2, 5.8, 200)
    y_grid = np.linspace(35, 100, 200)
    XX, YY = np.meshgrid(x_grid, y_grid)
    grid_pos = np.dstack((XX, YY))

    # (a) Single Gaussian MLE fit
    mu_mle = np.mean(X, axis=0)
    diff = X - mu_mle
    sigma_mle = (diff.T @ diff) / len(X)
    inv_mle = la.inv(sigma_mle)

    pos_mle = grid_pos - mu_mle
    quad_mle = np.einsum('...i,ij,...j->...', pos_mle, inv_mle, pos_mle)
    dens_mle = np.exp(-0.5 * quad_mle) / (2 * np.pi * np.sqrt(la.det(sigma_mle)))

    ax1.scatter(X[:, 0], X[:, 1], s=25, facecolors='none', edgecolors='#1E56A0', linewidth=0.9, alpha=0.8)
    levels1 = np.linspace(0.0005, dens_mle.max() * 0.9, 6)
    ax1.contour(XX, YY, dens_mle, levels=levels1, colors='#E02020', linewidths=1.4)
    ax1.set_xlabel('Eruption duration (min)', fontsize=11)
    ax1.set_ylabel('Waiting time to next eruption (min)', fontsize=11)
    ax1.set_title('(a) Single Gaussian fit (MLE)', fontsize=12)

    # (b) 2-Component GMM fit (EM)
    np.random.seed(42)
    # Fit 2 components
    N, D = X.shape
    idx0 = 0
    mu = np.zeros((2, D))
    mu[0] = X[idx0]
    dist = np.sum((X - mu[0])**2, axis=1)
    mu[1] = X[np.argmax(dist)]
    pi = np.array([0.5, 0.5])
    sigma = np.array([np.cov(X, rowvar=False), np.cov(X, rowvar=False)])

    for _ in range(50):
        # E-step
        gamma = np.zeros((N, 2))
        for k in range(2):
            d = X - mu[k]
            sol = la.solve(sigma[k], d.T).T
            q = np.sum(d * sol, axis=1)
            det = la.det(sigma[k])
            gamma[:, k] = pi[k] * np.exp(-0.5 * q) / np.sqrt((2*np.pi)**D * det)
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        # M-step
        Nk = np.sum(gamma, axis=0)
        pi = Nk / N
        for k in range(2):
            mu[k] = np.sum(gamma[:, k:k+1] * X, axis=0) / Nk[k]
            d = X - mu[k]
            sigma[k] = (gamma[:, k:k+1] * d).T @ d / Nk[k] + 1e-5 * np.eye(D)

    # Compute mixture density on grid
    dens_gmm = np.zeros_like(XX)
    for k in range(2):
        pos_k = grid_pos - mu[k]
        inv_k = la.inv(sigma[k])
        quad_k = np.einsum('...i,ij,...j->...', pos_k, inv_k, pos_k)
        dens_gmm += pi[k] * np.exp(-0.5 * quad_k) / (2 * np.pi * np.sqrt(la.det(sigma[k])))

    ax2.scatter(X[:, 0], X[:, 1], s=25, facecolors='none', edgecolors='#1E56A0', linewidth=0.9, alpha=0.8)
    levels2 = np.linspace(0.0005, dens_gmm.max() * 0.9, 7)
    ax2.contour(XX, YY, dens_gmm, levels=levels2, colors='#E02020', linewidths=1.4)
    ax2.set_xlabel('Eruption duration (min)', fontsize=11)
    ax2.set_title('(b) Two-component Gaussian mixture (EM)', fontsize=12)

    for ax in (ax1, ax2):
        ax.set_xlim(1.2, 5.8)
        ax.set_ylim(35, 100)
        ax.tick_params(axis='both', which='major', direction='in', top=True, right=True)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)

    plt.tight_layout()
    save_fig(fig, 'fig3_06_old_faithful_gaussian_and_mixture.png')
    plt.close(fig)

# -------------------------------------------------------------
# Figure 3.7: 1D Gaussian Mixture
# -------------------------------------------------------------
def plot_figure_3_7():
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    
    # 3 components
    x = np.linspace(-4, 6, 500)
    means = [-1.5, 0.5, 3.0]
    variances = [0.4, 0.25, 0.7]
    weights = [0.35, 0.40, 0.25]

    mixture = np.zeros_like(x)
    for k, (mu, var, pi) in enumerate(zip(means, variances, weights)):
        comp = pi * (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * (x - mu)**2 / var)
        mixture += comp
        ax.plot(x, comp, color='#1E56A0', linestyle='--', linewidth=1.5,
                label=f'Component {k+1}' if k == 0 else f'Component {k+1}')

    ax.plot(x, mixture, color='#E02020', linewidth=2.2, label=r'Sum $p(x) = \sum \pi_k \mathcal{N}_k$')

    ax.set_xlim(-4, 6)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$p(x)$', fontsize=12)
    ax.set_title('Gaussian mixture distribution in one dimension', fontsize=12)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)

    plt.tight_layout()
    save_fig(fig, 'fig3_07_gaussian_mixture_1d.png')
    plt.close(fig)

# -------------------------------------------------------------
# Figure 3.8: 2D Gaussian Mixture with 3 Components
# -------------------------------------------------------------
def plot_figure_3_8():
    fig = plt.figure(figsize=(15, 4.5), dpi=300)

    # 3 Components in 2D
    # (a) pi1 = 0.5, pi2 = 0.3, pi3 = 0.2 (matching Bishop 2024 page 107!)
    pi = [0.5, 0.3, 0.2]
    mu1 = np.array([-1.0, -0.8])
    Sigma1 = np.array([[0.6, 0.3], [0.3, 0.5]])

    mu2 = np.array([1.2, 0.2])
    Sigma2 = np.array([[0.4, -0.2], [-0.2, 0.6]])

    mu3 = np.array([-0.5, 1.5])
    Sigma3 = np.array([[0.5, 0.1], [0.1, 0.4]])

    mus = [mu1, mu2, mu3]
    Sigmas = [Sigma1, Sigma2, Sigma3]
    colors = ['#E02020', '#1E56A0', '#2CA02C']

    x = np.linspace(-3.5, 3.5, 200)
    y = np.linspace(-3.0, 3.5, 200)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Subplot (a): Contours of individual components
    ax1 = fig.add_subplot(1, 3, 1)
    for k in range(3):
        diff = pos - mus[k]
        inv = la.inv(Sigmas[k])
        quad = np.einsum('...i,ij,...j->...', diff, inv, diff)
        comp_dens = np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(la.det(Sigmas[k])))
        levels = np.linspace(comp_dens.max() * 0.15, comp_dens.max() * 0.95, 4)
        ax1.contour(X, Y, comp_dens, levels=levels, colors=colors[k], linewidths=1.4)
        # Label pi_k
        ax1.text(mus[k][0], mus[k][1] - 0.9, rf'$\pi_{k+1} = {pi[k]}$',
                 ha='center', fontsize=11, color=colors[k], fontweight='bold')

    ax1.set_xlim(-3.5, 3.5)
    ax1.set_ylim(-3.0, 3.5)
    ax1.set_xlabel(r'$x_1$', fontsize=11)
    ax1.set_ylabel(r'$x_2$', fontsize=11)
    ax1.set_title('(a) Component contours', fontsize=12)
    ax1.set_aspect('equal')
    ax1.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for spine in ax1.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)

    # Subplot (b): Marginal density p(x) contours
    ax2 = fig.add_subplot(1, 3, 2)
    mixture_dens = np.zeros_like(X)
    for k in range(3):
        diff = pos - mus[k]
        inv = la.inv(Sigmas[k])
        quad = np.einsum('...i,ij,...j->...', diff, inv, diff)
        comp_dens = np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(la.det(Sigmas[k])))
        mixture_dens += pi[k] * comp_dens

    levels_mix = np.linspace(0.015, mixture_dens.max() * 0.95, 8)
    ax2.contour(X, Y, mixture_dens, levels=levels_mix, colors='#E02020', linewidths=1.4)
    ax2.set_xlim(-3.5, 3.5)
    ax2.set_ylim(-3.0, 3.5)
    ax2.set_xlabel(r'$x_1$', fontsize=11)
    ax2.set_ylabel(r'$x_2$', fontsize=11)
    ax2.set_title('(b) Marginal density $p(\mathbf{x})$ contours', fontsize=12)
    ax2.set_aspect('equal')
    ax2.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for spine in ax2.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)

    # Subplot (c): 3D Surface plot
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    surf = ax3.plot_surface(X, Y, mixture_dens, cmap='viridis', edgecolor='none', alpha=0.9, antialiased=True)
    ax3.set_xlabel(r'$x_1$', fontsize=10, labelpad=5)
    ax3.set_ylabel(r'$x_2$', fontsize=10, labelpad=5)
    ax3.set_zlabel(r'$p(\mathbf{x})$', fontsize=10, labelpad=5)
    ax3.set_title('(c) 3D surface of $p(\mathbf{x})$', fontsize=12)
    ax3.view_init(elev=40, azim=-60)
    ax3.tick_params(labelsize=8)

    plt.tight_layout()
    save_fig(fig, 'fig3_08_gaussian_mixture_2d.png')
    plt.close(fig)

if __name__ == '__main__':
    plot_figure_3_2()
    plot_figure_3_3()
    plot_figure_3_4()
    plot_figure_3_5()
    plot_figure_3_6()
    plot_figure_3_7()
    plot_figure_3_8()
    print("All Figure 3.2 to 3.8 generated successfully!")
