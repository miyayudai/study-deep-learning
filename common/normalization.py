"""Chapter 7: Gradient Descent
Section 7.4: Normalization

This module implements data normalization, batch normalization, and layer normalization
for Section 7.4 of:
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.

Contents:
- Section 7.4.1: Data Normalization
  * Input feature mean: mu_i = (1/N) * sum_n x_ni (Eq 7.48)
  * Input feature variance: sigma_i^2 = (1/N) * sum_n (x_ni - mu_i)^2 (Eq 7.49)
  * Scaled input: x_tilde_ni = (x_ni - mu_i) / sigma_i (Eq 7.50)
  * Figure 7.7: Effect of input data normalization
- Section 7.4.2: Batch Normalization (Ioffe & Szegedy, 2015)
  * Jacobian product in deep networks: prod_k (del z^(k) / del z^(k-1)) (Eq 7.51)
  * Mini-batch mean: mu_i = (1/K) * sum_n a_ni (Eq 7.52)
  * Mini-batch variance: sigma_i^2 = (1/K) * sum_n (a_ni - mu_i)^2 (Eq 7.53)
  * Standardized pre-activation: a_hat_ni = (a_ni - mu_i) / sqrt(sigma_i^2 + delta) (Eq 7.54)
  * Affine transformation: a_tilde_ni = gamma_i * a_hat_ni + beta_i (Eq 7.55)
  * Running statistics for inference:
    mu_i^(tau) = alpha * mu_i^(tau-1) + (1 - alpha) * mu_i (Eq 7.56)
    sigma_i^(tau) = alpha * sigma_i^(tau-1) + (1 - alpha) * sigma_i (Eq 7.57)
- Section 7.4.3: Layer Normalization (Ba, Kiros, & Hinton, 2016)
  * Per-sample mean across hidden units: mu_n = (1/M) * sum_i a_ni (Eq 7.58)
  * Per-sample variance across hidden units: sigma_n^2 = (1/M) * sum_i (a_ni - mu_n)^2 (Eq 7.59)
  * Standardized pre-activation: a_hat_ni = (a_ni - mu_n) / sqrt(sigma_n^2 + delta) (Eq 7.60)
  * Figure 7.8: Comparison between Batch Normalization (a) and Layer Normalization (b)
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from common.plot_utils import save_plot, setup_style


def _get_project_root() -> str:
    """Return the absolute path to the repository root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _save_figure(
    fig: plt.Figure,
    filename: str,
    filepath: Optional[str] = None,
    save_both: bool = True,
) -> Tuple[str, Optional[str]]:
    """Save figure to target filepath, and/or both 7/result and root result directory."""
    if filepath:
        save_plot(fig, filepath)
    root = _get_project_root()
    path_ch7 = os.path.join(root, "7", "result", filename)
    path_root = os.path.join(root, "result", filename)
    if save_both or not filepath:
        if not filepath or os.path.abspath(filepath) != os.path.abspath(path_ch7):
            save_plot(fig, path_ch7)
        if not filepath or os.path.abspath(filepath) != os.path.abspath(path_root):
            save_plot(fig, path_root)
        return path_ch7, path_root
    return filepath, None


# ===========================================================================
# 7.4.1 Data Normalization
# ===========================================================================

class DataStandardizer:
    """Standardize input features to have zero mean and unit variance.
    
    Equations:
        mu_i = (1 / N) * sum_{n=1}^N x_{ni}       (Eq 7.48)
        sigma_i^2 = (1 / N) * sum_{n=1}^N (x_{ni} - mu_i)^2  (Eq 7.49)
        x_tilde_{ni} = (x_{ni} - mu_i) / sigma_i  (Eq 7.50)
    
    The same mu_i and sigma_i evaluated on training data must be used to
    preprocess validation and test data (Bishop & Bishop 2024, p. 226).
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "DataStandardizer":
        """Compute the mean and standard deviation from training data X."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X_arr, axis=0)  # Eq 7.48
        self.var_ = np.var(X_arr, axis=0)   # Eq 7.49
        self.std_ = np.sqrt(self.var_ + self.eps)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize X using stored training mean and standard deviation."""
        if not self.is_fitted or self.mean_ is None or self.std_ is None:
            raise ValueError("DataStandardizer must be fitted before transforming data.")
        X_arr = np.asarray(X, dtype=np.float64)
        return (X_arr - self.mean_) / self.std_  # Eq 7.50

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to X and return standardized version."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Transform standardized data back to the original scale."""
        if not self.is_fitted or self.mean_ is None or self.std_ is None:
            raise ValueError("DataStandardizer must be fitted before inverse transform.")
        X_arr = np.asarray(X_scaled, dtype=np.float64)
        return X_arr * self.std_ + self.mean_


# ===========================================================================
# 7.4.2 Batch Normalization
# ===========================================================================

class BatchNormalization:
    """Batch Normalization layer for deep neural networks (Ioffe & Szegedy, 2015).
    
    Normalizes pre-activations across the mini-batch separately for each hidden unit:
        mu_i = (1 / K) * sum_{n=1}^K a_{ni}                        (Eq 7.52)
        sigma_i^2 = (1 / K) * sum_{n=1}^K (a_{ni} - mu_i)^2        (Eq 7.53)
        a_hat_{ni} = (a_{ni} - mu_i) / sqrt(sigma_i^2 + delta)     (Eq 7.54)
        a_tilde_{ni} = gamma_i * a_hat_{ni} + beta_i               (Eq 7.55)
        
    During inference, running exponential moving averages are used:
        mu_i^(tau) = alpha * mu_i^(tau-1) + (1 - alpha) * mu_i     (Eq 7.56)
        sigma_i^(tau) = alpha * sigma_i^(tau-1) + (1 - alpha) * sigma_i (Eq 7.57)
    """

    def __init__(
        self,
        num_features: int,
        delta: float = 1e-5,
        momentum: float = 0.9,
    ):
        self.num_features = num_features
        self.delta = delta
        self.momentum = momentum  # alpha in Eq (7.56, 7.57)
        
        # Learnable affine parameters gamma and beta (Eq 7.55)
        self.gamma = np.ones(num_features, dtype=np.float64)
        self.beta = np.zeros(num_features, dtype=np.float64)
        
        # Gradients
        self.dgamma = np.zeros(num_features, dtype=np.float64)
        self.dbeta = np.zeros(num_features, dtype=np.float64)
        
        # Running statistics for test-time inference
        self.running_mean = np.zeros(num_features, dtype=np.float64)
        self.running_var = np.ones(num_features, dtype=np.float64)
        self.initialized_running_stats = False
        
        # Cache for backprop
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, a: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass for Batch Normalization.
        
        Args:
            a: Input pre-activations of shape (K, M), where K is mini-batch size
               and M is number of hidden units.
            training: If True, uses mini-batch statistics and updates running averages.
                      If False, uses accumulated running statistics.
                      
        Returns:
            Normalized and scaled pre-activations a_tilde of shape (K, M).
        """
        a_arr = np.asarray(a, dtype=np.float64)
        K, M = a_arr.shape
        if M != self.num_features:
            raise ValueError(f"Expected feature dim {self.num_features}, got {M}")
            
        if training:
            # 1. Mini-batch mean (Eq 7.52)
            mu = np.mean(a_arr, axis=0)  # Shape: (M,)
            # 2. Mini-batch variance (Eq 7.53)
            x_minus_mu = a_arr - mu
            var = np.mean(x_minus_mu ** 2, axis=0)  # Shape: (M,)
            # 3. Standardize (Eq 7.54)
            inv_std = 1.0 / np.sqrt(var + self.delta)
            a_hat = x_minus_mu * inv_std  # Shape: (K, M)
            # 4. Affine scale and shift (Eq 7.55)
            out = self.gamma * a_hat + self.beta
            
            # 5. Update running statistics (Eq 7.56, 7.57)
            if not self.initialized_running_stats:
                self.running_mean = mu.copy()
                self.running_var = var.copy()
                self.initialized_running_stats = True
            else:
                self.running_mean = (
                    self.momentum * self.running_mean + (1.0 - self.momentum) * mu
                )
                self.running_var = (
                    self.momentum * self.running_var + (1.0 - self.momentum) * var
                )
                
            self.cache = {
                "a": a_arr,
                "a_hat": a_hat,
                "x_minus_mu": x_minus_mu,
                "var": var,
                "inv_std": inv_std,
            }
            return out
        else:
            # Test-time inference using running statistics
            a_hat = (a_arr - self.running_mean) / np.sqrt(self.running_var + self.delta)
            return self.gamma * a_hat + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass computing gradients with respect to input and parameters.
        
        Args:
            dout: Upstream gradient dE / d(a_tilde) of shape (K, M).
            
        Returns:
            Gradient dE / da of shape (K, M).
        """
        K, M = dout.shape
        a_hat = self.cache["a_hat"]
        x_minus_mu = self.cache["x_minus_mu"]
        inv_std = self.cache["inv_std"]
        
        # dE / dgamma = sum_n (dout * a_hat)
        self.dgamma = np.sum(dout * a_hat, axis=0)
        # dE / dbeta = sum_n (dout)
        self.dbeta = np.sum(dout, axis=0)
        
        # dE / da_hat = dout * gamma
        da_hat = dout * self.gamma
        
        # Backprop through variance, mean, and standardization
        # dE / dvar = sum_n (da_hat * (a - mu) * -0.5 * (var + delta)^(-1.5))
        dvar = np.sum(da_hat * x_minus_mu * -0.5 * (inv_std ** 3), axis=0)
        # dE / dmu = sum_n (da_hat * -inv_std) + dvar * mean(-2 * (a - mu))
        dmu = np.sum(da_hat * -inv_std, axis=0) + dvar * np.mean(-2.0 * x_minus_mu, axis=0)
        # dE / da = da_hat * inv_std + dvar * 2 * (a - mu) / K + dmu / K
        da = da_hat * inv_std + dvar * 2.0 * x_minus_mu / K + dmu / K
        return da


# ===========================================================================
# 7.4.3 Layer Normalization
# ===========================================================================

class LayerNormalization:
    """Layer Normalization for neural networks (Ba, Kiros, & Hinton, 2016).
    
    Normalizes across the hidden units separately for each data point:
        mu_n = (1 / M) * sum_{i=1}^M a_{ni}                        (Eq 7.58)
        sigma_n^2 = (1 / M) * sum_{i=1}^M (a_{ni} - mu_n)^2        (Eq 7.59)
        a_hat_{ni} = (a_{ni} - mu_n) / sqrt(sigma_n^2 + delta)     (Eq 7.60)
        a_tilde_{ni} = gamma_i * a_hat_{ni} + beta_i               (Eq 7.55)
        
    Training and inference use the exact same calculation; no running averages.
    """

    def __init__(self, num_features: int, delta: float = 1e-5):
        self.num_features = num_features
        self.delta = delta
        
        # Learnable affine parameters gamma and beta per feature
        self.gamma = np.ones(num_features, dtype=np.float64)
        self.beta = np.zeros(num_features, dtype=np.float64)
        
        # Gradients
        self.dgamma = np.zeros(num_features, dtype=np.float64)
        self.dbeta = np.zeros(num_features, dtype=np.float64)
        
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, a: np.ndarray) -> np.ndarray:
        """Forward pass for Layer Normalization.
        
        Args:
            a: Input pre-activations of shape (K, M).
            
        Returns:
            Normalized pre-activations of shape (K, M).
        """
        a_arr = np.asarray(a, dtype=np.float64)
        K, M = a_arr.shape
        if M != self.num_features:
            raise ValueError(f"Expected feature dim {self.num_features}, got {M}")
            
        # 1. Per-sample mean across features (Eq 7.58)
        mu = np.mean(a_arr, axis=1, keepdims=True)  # Shape: (K, 1)
        # 2. Per-sample variance across features (Eq 7.59)
        x_minus_mu = a_arr - mu
        var = np.mean(x_minus_mu ** 2, axis=1, keepdims=True)  # Shape: (K, 1)
        # 3. Standardize (Eq 7.60)
        inv_std = 1.0 / np.sqrt(var + self.delta)
        a_hat = x_minus_mu * inv_std  # Shape: (K, M)
        # 4. Affine scale and shift
        out = self.gamma * a_hat + self.beta
        
        self.cache = {
            "a": a_arr,
            "a_hat": a_hat,
            "x_minus_mu": x_minus_mu,
            "var": var,
            "inv_std": inv_std,
        }
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass for Layer Normalization.
        
        Args:
            dout: Upstream gradient of shape (K, M).
            
        Returns:
            Gradient dE / da of shape (K, M).
        """
        K, M = dout.shape
        a_hat = self.cache["a_hat"]
        x_minus_mu = self.cache["x_minus_mu"]
        inv_std = self.cache["inv_std"]
        
        self.dgamma = np.sum(dout * a_hat, axis=0)
        self.dbeta = np.sum(dout, axis=0)
        
        da_hat = dout * self.gamma
        
        dvar = np.sum(da_hat * x_minus_mu * -0.5 * (inv_std ** 3), axis=1, keepdims=True)
        dmu = np.sum(da_hat * -inv_std, axis=1, keepdims=True) + dvar * np.mean(-2.0 * x_minus_mu, axis=1, keepdims=True)
        da = da_hat * inv_std + dvar * 2.0 * x_minus_mu / M + dmu / M
        return da


# ===========================================================================
# 7.4.2 Vanishing/Exploding Gradient Jacobian Product Simulation (Eq 7.51)
# ===========================================================================

def simulate_jacobian_norm_propagation(
    depth: int = 15,
    width: int = 32,
    weight_scale: float = 1.2,
    normalize: bool = False,
    num_trials: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Simulate the product of layer Jacobians prod_{k=1}^L (del z^(k) / del z^(k-1)) (Eq 7.51).
    
    Demonstrates that without normalization, products of Jacobians either vanish
    to 0 (weight_scale < 1) or explode to infinity (weight_scale > 1), whereas
    normalization controls signal bounds.
    
    Args:
        depth: Number of layers L.
        width: Hidden unit dimension M.
        weight_scale: Scale of random weight matrices.
        normalize: If True, applies layer normalization after each linear transformation.
        num_trials: Number of repeated runs.
        seed: Random seed.
        
    Returns:
        Array of mean gradient norm through depth L.
    """
    rng = np.random.default_rng(seed)
    norms_per_layer = np.zeros(depth + 1)
    
    for _ in range(num_trials):
        # Initial gradient vector at top layer
        g = rng.normal(size=width)
        g /= np.linalg.norm(g)
        norms = [float(np.linalg.norm(g))]
        
        for _ in range(depth):
            # Weight matrix representing Jacobian component
            W = rng.normal(0, weight_scale / np.sqrt(width), size=(width, width))
            g = W.T @ g
            if normalize:
                g = (g - np.mean(g)) / (np.std(g) + 1e-5)
            norms.append(float(np.linalg.norm(g)))
            
        norms_per_layer += np.array(norms)
        
    return norms_per_layer / num_trials


# ===========================================================================
# Figure Reproductions (Figures 7.7 and 7.8)
# ===========================================================================

def generate_figure_7_7(
    filepath: Optional[str] = None,
    save_both: bool = True,
) -> Tuple[str, Optional[str]]:
    """Reproduce Figure 7.7: Illustration of the effect of input data normalization.
    
    The red circles show the original data points for a dataset with two variables.
    The blue crosses show the dataset after normalization such that each variable
    now has zero mean and unit variance across the dataset.
    (Bishop & Bishop 2024, p. 226)
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    
    rng = np.random.default_rng(42)
    mean_orig = np.array([3.5, 1.5])
    cov_orig = np.array([[6.0, 4.0], [4.0, 18.0]])
    raw_data = rng.multivariate_normal(mean_orig, cov_orig, size=22)
    raw_data = np.clip(raw_data, -9.0, 9.0)
    
    scaler = DataStandardizer()
    norm_data = scaler.fit_transform(raw_data)
    
    # Plot original data (red circles)
    ax.scatter(raw_data[:, 0], raw_data[:, 1], color="red", s=50, label="Original data", zorder=3)
    # Plot normalized data (blue crosses)
    ax.scatter(norm_data[:, 0], norm_data[:, 1], color="blue", marker="x", s=60, lw=2, label="Normalized data", zorder=4)
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12, rotation=0, labelpad=10)
    
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)
        
    fig.tight_layout()
    filename = "fig_7_7_data_normalization.png"
    return _save_figure(fig, filename, filepath, save_both=save_both)


def generate_figure_7_8(
    filepath: Optional[str] = None,
    save_both: bool = True,
) -> Tuple[str, Optional[str]]:
    """Reproduce Figure 7.8: Illustration of batch and layer normalization in a neural network.
    
    In batch normalization, shown in (a), the mean and variance are computed
    across the mini-batch separately for each hidden unit.
    In layer normalization, shown in (b), the mean and variance are computed
    across the hidden units separately for each data point.
    (Bishop & Bishop 2024, p. 228)
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.8))
    
    cream_color = "#fff9e6"
    edge_color = "#333333"
    
    # (a) Batch Normalization
    nrows, ncols = 6, 4
    dx, dy = 0.8, 0.8
    x0, y0 = 1.0, 1.0
    
    for r in range(nrows):
        for c in range(ncols):
            rect = patches.Rectangle(
                (x0 + c * dx, y0 + (nrows - 1 - r) * dy),
                dx, dy, facecolor=cream_color, edgecolor=edge_color, lw=1.2
            )
            ax1.add_patch(rect)
            
    x_mu = x0 + ncols * dx + 1.2
    x_sigma = x_mu + dx + 0.1
    for r in range(nrows):
        y_r = y0 + (nrows - 1 - r) * dy
        ax1.annotate("", xy=(x_mu - 0.2, y_r + dy / 2), xytext=(x0 + ncols * dx + 0.1, y_r + dy / 2),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        rect_mu = patches.Rectangle((x_mu, y_r), dx, dy, facecolor=cream_color, edgecolor=edge_color, lw=1.2)
        ax1.add_patch(rect_mu)
        rect_sig = patches.Rectangle((x_sigma, y_r), dx, dy, facecolor=cream_color, edgecolor=edge_color, lw=1.2)
        ax1.add_patch(rect_sig)
        
    ax1.text(x0 + ncols * dx / 2, y0 + nrows * dy + 0.6, "Mini-batch", fontsize=12, ha="center")
    ax1.plot([x0, x0, x0 + ncols * dx / 2 - 0.2], [y0 + nrows * dy + 0.1, y0 + nrows * dy + 0.3, y0 + nrows * dy + 0.3], "k-", lw=1.2)
    ax1.plot([x0 + ncols * dx, x0 + ncols * dx, x0 + ncols * dx / 2 + 0.2], [y0 + nrows * dy + 0.1, y0 + nrows * dy + 0.3, y0 + nrows * dy + 0.3], "k-", lw=1.2)
    
    ax1.text(x0 - 0.8, y0 + nrows * dy / 2, "Hidden units", fontsize=12, ha="right", va="center")
    ax1.plot([x0 - 0.2, x0 - 0.4, x0 - 0.4], [y0, y0, y0 + nrows * dy / 2 - 0.2], "k-", lw=1.2)
    ax1.plot([x0 - 0.2, x0 - 0.4, x0 - 0.4], [y0 + nrows * dy, y0 + nrows * dy, y0 + nrows * dy / 2 + 0.2], "k-", lw=1.2)
    
    ax1.text(x_mu + dx / 2, y0 + nrows * dy + 0.4, r"$\mu$", fontsize=13, ha="center")
    ax1.text(x_sigma + dx / 2, y0 + nrows * dy + 0.4, r"$\sigma$", fontsize=13, ha="center")
    ax1.text(x0 + (x_sigma + dx - x0) / 2, 0.2, "(a)", fontsize=13, ha="center")
    ax1.set_xlim(-0.5, x_sigma + dx + 0.8)
    ax1.set_ylim(0.0, y0 + nrows * dy + 1.2)
    ax1.axis("off")
    
    # (b) Layer Normalization
    x0_b = 1.8
    for r in range(nrows):
        for c in range(ncols):
            rect = patches.Rectangle(
                (x0_b + c * dx, y0 + (nrows - 1 - r) * dy),
                dx, dy, facecolor=cream_color, edgecolor=edge_color, lw=1.2
            )
            ax2.add_patch(rect)
            
    y_mu = y0 - 1.2
    y_sigma = y_mu - dy - 0.1
    for c in range(ncols):
        x_c = x0_b + c * dx
        ax2.annotate("", xy=(x_c + dx / 2, y_mu + dy + 0.2), xytext=(x_c + dx / 2, y0 - 0.1),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        rect_mu = patches.Rectangle((x_c, y_mu), dx, dy, facecolor=cream_color, edgecolor=edge_color, lw=1.2)
        ax2.add_patch(rect_mu)
        rect_sig = patches.Rectangle((x_c, y_sigma), dx, dy, facecolor=cream_color, edgecolor=edge_color, lw=1.2)
        ax2.add_patch(rect_sig)
        
    ax2.text(x0_b - 0.4, y_mu + dy / 2, r"$\mu$", fontsize=13, ha="right", va="center")
    ax2.text(x0_b - 0.4, y_sigma + dy / 2, r"$\sigma$", fontsize=13, ha="right", va="center")
    
    ax2.text(x0_b + ncols * dx / 2, y0 + nrows * dy + 0.6, "Mini-batch", fontsize=12, ha="center")
    ax2.plot([x0_b, x0_b, x0_b + ncols * dx / 2 - 0.2], [y0 + nrows * dy + 0.1, y0 + nrows * dy + 0.3, y0 + nrows * dy + 0.3], "k-", lw=1.2)
    ax2.plot([x0_b + ncols * dx, x0_b + ncols * dx, x0_b + ncols * dx / 2 + 0.2], [y0 + nrows * dy + 0.1, y0 + nrows * dy + 0.3, y0 + nrows * dy + 0.3], "k-", lw=1.2)
    
    ax2.text(x0_b - 0.8, y0 + nrows * dy / 2, "Hidden units", fontsize=12, ha="right", va="center")
    ax2.plot([x0_b - 0.2, x0_b - 0.4, x0_b - 0.4], [y0, y0, y0 + nrows * dy / 2 - 0.2], "k-", lw=1.2)
    ax2.plot([x0_b - 0.2, x0_b - 0.4, x0_b - 0.4], [y0 + nrows * dy, y0 + nrows * dy, y0 + nrows * dy / 2 + 0.2], "k-", lw=1.2)
    
    ax2.text(x0_b + ncols * dx / 2, y_sigma - 0.8, "(b)", fontsize=13, ha="center")
    ax2.set_xlim(0.0, x0_b + ncols * dx + 1.2)
    ax2.set_ylim(y_sigma - 1.2, y0 + nrows * dy + 1.2)
    ax2.axis("off")
    
    fig.tight_layout()
    filename = "fig_7_8_batch_vs_layer_norm.png"
    return _save_figure(fig, filename, filepath, save_both=save_both)
