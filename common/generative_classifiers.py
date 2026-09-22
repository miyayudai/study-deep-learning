"""
Generative Classifiers for Single-layer Networks: Classification.
Bishop & Bishop (2024), Chapter 5, Section 5.3.

Covers:
- Logistic sigmoid & scaled probit comparison (Section 5.3, Eq 5.40 - 5.44, Figure 5.12)
- Gaussian Discriminant Analysis (GDA / LDA / QDA) for continuous inputs (Section 5.3.1, Eq 5.47 - 5.53, Figure 5.13 - 5.14)
- Maximum Likelihood Estimation for Gaussian class-conditionals (Section 5.3.2, Eq 5.54 - 5.63)
- Discrete Features & Naive Bayes (Section 5.3.3, Eq 5.64 - 5.65)
- General Exponential Family Class-Conditionals (Section 5.3.4, Eq 5.66)
"""
import os
from typing import Optional, Union, Tuple, List, Dict
import numpy as np
import scipy.linalg as la
import scipy.stats as stats
from scipy.special import expit, erf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from common.plot_utils import setup_style, save_plot


# =====================================================================
# 1. Activation Functions & Probit Approximation (Section 5.3)
# =====================================================================

def sigmoid(a: np.ndarray) -> np.ndarray:
    """
    Logistic sigmoid function (Eq 5.42):
        sigma(a) = 1 / (1 + exp(-a))
    """
    return expit(a)


def logit(p: np.ndarray) -> np.ndarray:
    """
    Logit / log-odds function (Eq 5.44):
        a = ln(p / (1 - p))
    """
    p_clamped = np.clip(p, 1e-15, 1.0 - 1e-15)
    return np.log(p_clamped / (1.0 - p_clamped))


def probit(a: np.ndarray) -> np.ndarray:
    """
    Standard normal cumulative distribution function Phi(a) (Eq 5.86).
    """
    return stats.norm.cdf(a)


def scaled_probit(a: np.ndarray, lambda_sq: float = np.pi / 8.0) -> np.ndarray:
    """
    Scaled probit function Phi(lambda * a) where lambda^2 = pi / 8 (Figure 5.12).
    Derivatives at a = 0 match the logistic sigmoid:
        d/da sigma(a)|_{a=0} = 1/4
        d/da Phi(lambda a)|_{a=0} = lambda / sqrt(2*pi) = sqrt(pi/8) / sqrt(2*pi) = 1/4.
    """
    lam = np.sqrt(lambda_sq)
    return stats.norm.cdf(lam * a)


# =====================================================================
# 2. Gaussian Discriminant Analysis (LDA & QDA) (Section 5.3.1 - 5.3.2)
# =====================================================================

class GaussianDiscriminantAnalysis:
    """
    Gaussian class-conditional generative classifier:
        p(x | C_k) = N(x | mu_k, Sigma_k)
    Supports:
    - shared_cov=True: Linear Discriminant Analysis (LDA) with shared Sigma.
      Posterior probabilities are given by linear softmax / sigmoid (Eq 5.48 - 5.53).
    - shared_cov=False: Quadratic Discriminant Analysis (QDA) with class-specific Sigma_k.
      Decision boundaries are quadratic surfaces.
    """
    def __init__(self, shared_cov: bool = True):
        self.shared_cov = bool(shared_cov)
        self.num_classes = None
        self.dim = None
        self.priors = None       # shape (K,)
        self.means = None        # shape (K, D)
        self.covs = None         # shape (K, D, D) if not shared, else (D, D)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianDiscriminantAnalysis':
        """
        Maximum likelihood estimation (Section 5.3.2, Eq 5.54 - 5.63).
        X: shape (N, D)
        y: shape (N,) integer class labels in {0, ..., K-1}
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=int).ravel()
        N, D = X_arr.shape
        classes = np.unique(y_arr)
        K = len(classes)
        self.num_classes = K
        self.dim = D

        self.priors = np.zeros(K)
        self.means = np.zeros((K, D))
        class_covs = np.zeros((K, D, D))

        for k in range(K):
            mask = (y_arr == k)
            N_k = np.sum(mask)
            self.priors[k] = N_k / N  # Eq 5.56
            X_k = X_arr[mask]
            self.means[k] = np.mean(X_k, axis=0)  # Eq 5.58 - 5.59
            diff = X_k - self.means[k]
            # Unbiased or ML covariance: ML uses 1/N_k
            class_covs[k] = (diff.T @ diff) / max(N_k, 1)

        if self.shared_cov:
            # Shared covariance S = sum_k (N_k / N) * S_k (Eq 5.61)
            shared = np.zeros((D, D))
            for k in range(K):
                shared += self.priors[k] * class_covs[k]
            self.covs = shared
        else:
            self.covs = class_covs

        return self

    def log_likelihood_terms(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log p(x | C_k) + ln p(C_k) for all classes k.
        Returns: shape (N, K)
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        N, D = X_arr.shape
        K = self.num_classes
        log_terms = np.zeros((N, K))

        for k in range(K):
            cov_k = self.covs if self.shared_cov else self.covs[k]
            mean_k = self.means[k]
            # Use multivariate_normal logpdf
            sign, logdet = np.linalg.slogdet(cov_k)
            cov_inv = np.linalg.pinv(cov_k)
            diff = X_arr - mean_k
            quad = np.sum(diff @ cov_inv * diff, axis=1)
            log_density = -0.5 * (D * np.log(2.0 * np.pi) + logdet + quad)
            log_terms[:, k] = log_density + np.log(max(self.priors[k], 1e-15))

        return log_terms

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior probabilities p(C_k | x) via Softmax (Eq 5.45).
        """
        log_terms = self.log_likelihood_terms(X)
        # Numerically stable softmax
        max_log = np.max(log_terms, axis=1, keepdims=True)
        exp_terms = np.exp(log_terms - max_log)
        return exp_terms / np.sum(exp_terms, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign to class with maximum posterior probability."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def get_linear_weights_2class(self) -> Tuple[np.ndarray, float]:
        """
        For 2-class LDA (shared_cov=True), return linear parameters w and w0 (Eq 5.49 - 5.50):
            p(C_1 | x) = sigma(w^T x + w0)
        """
        if not self.shared_cov or self.num_classes != 2:
            raise ValueError("Only available for 2-class LDA with shared covariance.")
        inv_cov = np.linalg.pinv(self.covs)
        mu1, mu2 = self.means[0], self.means[1]
        w = inv_cov @ (mu1 - mu2)
        w0 = -0.5 * (mu1 @ inv_cov @ mu1) + 0.5 * (mu2 @ inv_cov @ mu2) + np.log(self.priors[0] / self.priors[1])
        return w, float(w0)


# =====================================================================
# 3. Discrete Features: Naive Bayes (Section 5.3.3)
# =====================================================================

class BernoulliNaiveBayes:
    """
    Naive Bayes classifier for binary discrete features x_i in {0, 1} (Eq 5.64 - 5.65):
        p(x | C_k) = prod_{i=1}^D mu_{ki}^{x_i} (1 - mu_{ki})^{1 - x_i}
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha) # Laplace smoothing
        self.num_classes = None
        self.dim = None
        self.priors = None
        self.mu = None            # shape (K, D) feature probabilities

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BernoulliNaiveBayes':
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=int).ravel()
        N, D = X_arr.shape
        classes = np.unique(y_arr)
        K = len(classes)
        self.num_classes = K
        self.dim = D

        self.priors = np.zeros(K)
        self.mu = np.zeros((K, D))

        for k in range(K):
            mask = (y_arr == k)
            N_k = np.sum(mask)
            self.priors[k] = N_k / N
            X_k = X_arr[mask]
            # Laplace smoothing: (count + alpha) / (N_k + 2 * alpha)
            self.mu[k] = (np.sum(X_k, axis=0) + self.alpha) / (N_k + 2.0 * self.alpha)

        return self

    def log_likelihood_terms(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Eq 5.65: a_k(x) = sum_i {x_i ln mu_ki + (1-x_i) ln(1-mu_ki)} + ln p(C_k)."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        N, D = X_arr.shape
        K = self.num_classes
        log_terms = np.zeros((N, K))

        for k in range(K):
            mu_k = np.clip(self.mu[k], 1e-15, 1.0 - 1e-15)
            log_prob = X_arr @ np.log(mu_k) + (1.0 - X_arr) @ np.log(1.0 - mu_k)
            log_terms[:, k] = log_prob + np.log(max(self.priors[k], 1e-15))

        return log_terms

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_terms = self.log_likelihood_terms(X)
        max_log = np.max(log_terms, axis=1, keepdims=True)
        exp_terms = np.exp(log_terms - max_log)
        return exp_terms / np.sum(exp_terms, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# =====================================================================
# 4. General Exponential Family Class-Conditionals (Section 5.3.4)
# =====================================================================

class ExponentialFamilyClassifier:
    """
    Generative classifier with class-conditional densities from the exponential family (Eq 5.66):
        p(x | lambda_k, s) = (1/s) h(x/s) g(lambda_k) exp( (1/s) lambda_k^T x )
    
    The base measure h(x/s) is shared and cancels out, leading to linear activations (Eq 5.68):
        a_k(x) = (1/s) lambda_k^T x + ln g(lambda_k) + ln p(C_k) = w_k^T x + w_{k0}
    
    And for 2 classes (Eq 5.67):
        a(x) = (1/s) (lambda_1 - lambda_2)^T x + ln(g(lambda_1)/g(lambda_2)) + ln(p(C_1)/p(C_2))
    """
    def __init__(self, natural_params: np.ndarray, log_g: np.ndarray, priors: np.ndarray, scale: float = 1.0):
        """
        natural_params: shape (K, D) natural parameter vectors lambda_k
        log_g: shape (K,) log partition / normalization terms ln g(lambda_k)
        priors: shape (K,) class prior probabilities p(C_k)
        scale: dispersion / scale parameter s > 0
        """
        self.natural_params = np.asarray(natural_params, dtype=np.float64)
        self.log_g = np.asarray(log_g, dtype=np.float64)
        self.priors = np.asarray(priors, dtype=np.float64)
        self.scale = float(scale)
        self.num_classes, self.dim = self.natural_params.shape

        # Linear weights and biases: Eq 5.68
        self.weights = self.natural_params / self.scale  # shape (K, D)
        self.biases = self.log_g + np.log(np.maximum(self.priors, 1e-15))  # shape (K,)

    def activations(self, X: np.ndarray) -> np.ndarray:
        """Compute linear activations a_k(x) = w_k^T x + w_{k0} (Eq 5.68)."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return X_arr @ self.weights.T + self.biases

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute posterior probabilities p(C_k | x) via Softmax (Eq 5.45) or Sigmoid (Eq 5.40)."""
        A = self.activations(X)
        max_A = np.max(A, axis=1, keepdims=True)
        exp_A = np.exp(A - max_A)
        return exp_A / np.sum(exp_A, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def get_2class_linear_weights(self) -> Tuple[np.ndarray, float]:
        """For 2-class case, return w and w0 for a(x) = w^T x + w0 (Eq 5.67)."""
        if self.num_classes != 2:
            raise ValueError("Only valid for 2-class problems.")
        w = (self.natural_params[0] - self.natural_params[1]) / self.scale
        w0 = (self.log_g[0] - self.log_g[1]) + np.log(self.priors[0] / self.priors[1])
        return w, float(w0)


# =====================================================================
# 5. Figure Reproduction (Figures 5.12 to 5.14)
# =====================================================================

def plot_figure_5_12_sigmoid_and_probit(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """
    Faithful reproduction of Figure 5.12 (Book page 151):
    Plot of the logistic sigmoid function sigma(a) (red solid) together with
    the scaled probit function Phi(lambda a) for lambda^2 = pi / 8 (blue dashed).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    a = np.linspace(-8.0, 8.0, 300)
    sig = sigmoid(a)
    prb = scaled_probit(a, lambda_sq=np.pi / 8.0)

    # Plot curves
    ax.plot(a, sig, color='#E41A1C', lw=2.5, label=r'$\sigma(a)$')
    ax.plot(a, prb, color='#0044FF', lw=2.2, linestyle='--', label=r'$\Phi(\lambda a)$')

    # Reference lines at a=0 and y=0.5
    ax.axvline(0, color='gray', linestyle=':', lw=1.2)
    ax.axhline(0.5, color='gray', linestyle=':', lw=1.2)

    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([-5, 0, 5])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(direction='in', top=True, right=True)

    ax.set_xlabel(r'$a$', fontsize=13)
    ax.set_ylabel('Probability', fontsize=13)
    ax.legend(frameon=True, loc='lower right', fontsize=12)
    ax.set_title("Figure 5.12: Logistic Sigmoid and Scaled Probit", fontsize=12, pad=10)

    fig.tight_layout()
    if filepath:
        save_plot(fig, filepath)
    if save_both and filepath:
        if "5/result" in filepath:
            alt_path = filepath.replace("5/result", "result")
            save_plot(fig, alt_path)
        elif "result" in filepath:
            alt_path = os.path.join("5", filepath)
            save_plot(fig, alt_path)
    return fig


def plot_figure_5_13_two_class_gaussian_posteriors(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """
    Faithful reproduction of Figure 5.13 (Book page 153):
    Left: Class-conditional densities p(x | C1) (red) and p(x | C2) (blue) with shared covariance.
    Right: Corresponding posterior probability surface p(C1 | x) coloured by ink proportions.
    """
    setup_style()
    fig = plt.figure(figsize=(12, 5.5))

    # Grid in 2D
    x1 = np.linspace(-3.5, 3.5, 80)
    x2 = np.linspace(-3.5, 3.5, 80)
    X1, X2 = np.meshgrid(x1, x2)
    pos = np.dstack((X1, X2))

    # Distributions with shared covariance
    mu1 = np.array([-1.2, 0.0])
    mu2 = np.array([1.2, 0.0])
    cov = np.array([[1.1, 0.2], [0.2, 1.1]])

    rv1 = stats.multivariate_normal(mu1, cov)
    rv2 = stats.multivariate_normal(mu2, cov)
    Z1 = rv1.pdf(pos)
    Z2 = rv2.pdf(pos)

    # Posteriors with equal priors
    P1 = Z1 / (Z1 + Z2)

    # Subplot 1: Class-conditional densities (3D)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X1, X2, Z1, color='#E41A1C', alpha=0.68, edgecolor='none', rstride=2, cstride=2)
    ax1.plot_surface(X1, X2, Z2, color='#0044FF', alpha=0.68, edgecolor='none', rstride=2, cstride=2)
    ax1.set_xlabel(r'$x_1$', fontsize=13, labelpad=-5)
    ax1.set_ylabel(r'$x_2$', fontsize=13, labelpad=-5)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.view_init(elev=26, azim=-65)
    ax1.set_title(r"Class-conditional densities $p(\mathbf{x}|\mathcal{C}_k)$", fontsize=12, pad=5)

    # Subplot 2: Posterior probability surface (3D)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # Custom colormap from blue (0.0) to red (1.0)
    colors = np.zeros(X1.shape + (4,))
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            p = P1[i, j]
            # Red proportion p, blue proportion (1-p)
            colors[i, j] = [p, 0.0, 1.0 - p, 0.88]

    ax2.plot_surface(X1, X2, P1, facecolors=colors, shade=False, rstride=2, cstride=2)
    ax2.set_xlabel(r'$x_1$', fontsize=13, labelpad=-5)
    ax2.set_ylabel(r'$x_2$', fontsize=13, labelpad=-5)
    ax2.set_zlabel(r'$p(\mathcal{C}_1|\mathbf{x})$', fontsize=12, labelpad=-2)
    ax2.set_zlim(0, 1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([0, 1])
    ax2.set_zticklabels(['0', '1'], fontsize=11)
    ax2.view_init(elev=26, azim=-65)
    ax2.set_title(r"Posterior probability $p(\mathcal{C}_1|\mathbf{x})$", fontsize=12, pad=5)

    fig.suptitle("Figure 5.13: Class-Conditional Densities and Posterior Probability Surface", fontsize=14, y=0.98)
    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.02, right=0.96, wspace=0.08)

    if filepath:
        save_plot(fig, filepath)
    if save_both and filepath:
        if "5/result" in filepath:
            alt_path = filepath.replace("5/result", "result")
            save_plot(fig, alt_path)
        elif "result" in filepath:
            alt_path = os.path.join("5", filepath)
            save_plot(fig, alt_path)
    return fig


def plot_figure_5_14_multiclass_gaussian_boundaries(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """
    Faithful reproduction of Figure 5.14 (Book page 154):
    Left: Class-conditional densities for 3 classes (red, green, blue).
          Red and blue classes have the same covariance matrix.
    Right: Posterior probabilities RGB blend map with decision boundaries.
           Boundary between red & blue (shared cov) is linear.
           Boundaries with green (different cov) are quadratic.
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2))

    # Grid
    x1 = np.linspace(-4.0, 4.0, 250)
    x2 = np.linspace(-4.0, 4.0, 250)
    X1, X2 = np.meshgrid(x1, x2)
    pos = np.dstack((X1, X2))

    # Parameters aligned with Bishop Fig 5.14:
    # Class 1 (Red): lower-left, horizontal ellipse
    mu1 = np.array([-1.4, -1.2])
    cov1 = np.array([[2.2, 0.4], [0.4, 1.0]])

    # Class 2 (Blue): upper-left, identical covariance to Class 1 (shared)
    mu2 = np.array([-1.2, 2.0])
    cov2 = cov1.copy()

    # Class 3 (Green): right, tilted covariance (different)
    mu3 = np.array([1.8, 0.3])
    cov3 = np.array([[1.0, -0.6], [-0.6, 1.8]])

    rv1 = stats.multivariate_normal(mu1, cov1)
    rv2 = stats.multivariate_normal(mu2, cov2)
    rv3 = stats.multivariate_normal(mu3, cov3)

    Z1 = rv1.pdf(pos)
    Z2 = rv2.pdf(pos)
    Z3 = rv3.pdf(pos)

    # Left Plot: Contours of class-conditional densities
    levels1 = np.linspace(0.015, Z1.max(), 7)
    levels2 = np.linspace(0.015, Z2.max(), 7)
    levels3 = np.linspace(0.015, Z3.max(), 7)

    ax1.contour(X1, X2, Z1, levels=levels1, colors='#E41A1C', alpha=0.85, linewidths=1.5)
    ax1.contour(X1, X2, Z2, levels=levels2, colors='#0044FF', alpha=0.85, linewidths=1.5)
    ax1.contour(X1, X2, Z3, levels=levels3, colors='#2CA02C', alpha=0.85, linewidths=1.5)

    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_xlabel(r'$x_1$', fontsize=13)
    ax1.set_ylabel(r'$x_2$', fontsize=13)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Class-conditional Densities (Left)", fontsize=12)

    # Right Plot: RGB Posterior Blend Map
    total_Z = Z1 + Z2 + Z3
    P1 = Z1 / total_Z
    P2 = Z2 / total_Z
    P3 = Z3 / total_Z

    # RGB image: R=P1, G=P3, B=P2
    rgb_img = np.zeros(X1.shape + (3,))
    rgb_img[:, :, 0] = np.clip(P1, 0, 1) # Red = Class 1
    rgb_img[:, :, 1] = np.clip(P3, 0, 1) # Green = Class 3
    rgb_img[:, :, 2] = np.clip(P2, 0, 1) # Blue = Class 2

    ax2.imshow(rgb_img, origin='lower', extent=[-4, 4, -4, 4], aspect='auto')

    # Decision boundaries: where the two largest posteriors are equal
    # Class 1 vs 2: P1 == P2 (and both > P3) -> Linear
    # Class 1 vs 3: P1 == P3 (and both > P2) -> Quadratic
    # Class 2 vs 3: P2 == P3 (and both > P1) -> Quadratic
    max_classes = np.argmax(np.stack([P1, P2, P3], axis=-1), axis=-1)

    # Draw decision boundary contours in white
    # P1 - P2 = 0 where class is 0 or 1
    diff_12 = P1 - P2
    diff_13 = P1 - P3
    diff_23 = P2 - P3

    mask_12 = np.logical_or(max_classes == 0, max_classes == 1)
    mask_13 = np.logical_or(max_classes == 0, max_classes == 2)
    mask_23 = np.logical_or(max_classes == 1, max_classes == 2)

    diff_12_masked = np.where(mask_12, diff_12, np.nan)
    diff_13_masked = np.where(mask_13, diff_13, np.nan)
    diff_23_masked = np.where(mask_23, diff_23, np.nan)

    ax2.contour(X1, X2, diff_12_masked, levels=[0], colors='white', linewidths=2.2)
    ax2.contour(X1, X2, diff_13_masked, levels=[0], colors='white', linewidths=2.2)
    ax2.contour(X1, X2, diff_23_masked, levels=[0], colors='white', linewidths=2.2)

    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_xlabel(r'$x_1$', fontsize=13)
    ax2.set_ylabel(r'$x_2$', fontsize=13)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Posterior Probabilities & Boundaries (Right)", fontsize=12)

    fig.suptitle("Figure 5.14: Linear and Quadratic Decision Boundaries in Generative Classifiers",
                 fontsize=13, y=0.98)
    fig.tight_layout()

    if filepath:
        save_plot(fig, filepath)
    if save_both and filepath:
        if "5/result" in filepath:
            alt_path = filepath.replace("5/result", "result")
            save_plot(fig, alt_path)
        elif "result" in filepath:
            alt_path = os.path.join("5", filepath)
            save_plot(fig, alt_path)
    return fig


def generate_all_section_5_3_figures(result_dirs: Optional[List[str]] = None) -> Dict[str, plt.Figure]:
    """Generate and save Figures 5.12 to 5.14 to result/ and 5/result/."""
    if result_dirs is None:
        result_dirs = ["5/result", "result"]
    for d in result_dirs:
        os.makedirs(d, exist_ok=True)

    figs = {}
    figs['fig_5_12'] = plot_figure_5_12_sigmoid_and_probit(filepath=os.path.join(result_dirs[0], "fig_5_12_sigmoid_and_probit.png"))
    figs['fig_5_13'] = plot_figure_5_13_two_class_gaussian_posteriors(filepath=os.path.join(result_dirs[0], "fig_5_13_gaussian_densities_posteriors.png"))
    figs['fig_5_14'] = plot_figure_5_14_multiclass_gaussian_boundaries(filepath=os.path.join(result_dirs[0], "fig_5_14_linear_quadratic_boundaries.png"))
    return figs
