"""
Mixture Density Networks (MDN)
===============================
Bishop & Bishop (2024), Chapter 6: Deep Neural Networks, Section 6.5.

This module implements:
1. Robot arm kinematics (Section 6.5.1, Figure 6.16)
   - Forward kinematics: mapping joint angles (theta_1, theta_2) to end effector position (x_1, x_2).
   - Inverse kinematics: multiple solutions ('elbow up' and 'elbow down').
2. Synthetic forward and inverse toy problems (Section 6.5.1, Figure 6.17).
3. Standard MLP Regressor trained via sum-of-squares error (least squares / Gaussian conditional assumption).
4. Mixture Density Network (MDN) architecture and gradient optimization (Sections 6.5.2 & 6.5.3, Figure 6.18):
   - Conditional Gaussian mixture distribution: p(t|x) = sum_k pi_k(x) N(t | mu_k(x), sigma_k^2(x) I) (Eq 6.38)
   - Softmax output for mixing coefficients pi_k(x) (Eq 6.40)
   - Exponential output for variances sigma_k(x) (Eq 6.41)
   - Linear output for means mu_k(x) (Eq 6.42)
   - Negative log-likelihood error function E(w) (Eq 6.43)
   - Exact analytical pre-activation gradients (Eq 6.44 - 6.47) and backpropagation.
5. Predictive distribution calculations (Section 6.5.4, Figure 6.19):
   - Conditional mean E[t|x] = sum_k pi_k(x) mu_k(x) (Eq 6.48)
   - Conditional variance s^2(x) (Eq 6.50)
   - Approximate conditional mode t* = mu_{k*}(x) where k* = argmax_k pi_k(x)
6. Publication-quality figure generation functions for Figures 6.16, 6.17, 6.18, and 6.19.
"""

import os
from typing import Tuple, Optional, Union
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle, Circle

from common.plot_utils import save_plot


# =====================================================================
# 1. Robot Kinematics (Section 6.5.1, Figure 6.16)
# =====================================================================

def forward_kinematics(
    theta1: Union[float, np.ndarray],
    theta2: Union[float, np.ndarray],
    L1: float = 1.0,
    L2: float = 0.65
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Compute end-effector Cartesian coordinates (x1, x2) from joint angles.
    
    x_1 = L_1 * cos(theta_1) + L_2 * cos(theta_1 + theta_2)
    x_2 = L_1 * sin(theta_1) + L_2 * sin(theta_1 + theta_2)
    """
    x1 = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    x2 = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return x1, x2


def inverse_kinematics(
    x1: float,
    x2: float,
    L1: float = 1.0,
    L2: float = 0.65
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute the two inverse kinematics solutions ('elbow up', 'elbow down')
    for a planar two-link manipulator reaching target (x1, x2).
    
    Returns:
        (theta_up, theta_down) where each is (theta1, theta2) in radians.
    """
    r_sq = x1**2 + x2**2
    cos_theta2 = (r_sq - L1**2 - L2**2) / (2.0 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    
    theta2_1 = float(np.arccos(cos_theta2))
    theta2_2 = -theta2_1
    
    beta = float(np.arctan2(x2, x1))
    
    alpha1 = float(np.arctan2(L2 * np.sin(theta2_1), L1 + L2 * np.cos(theta2_1)))
    theta1_1 = beta - alpha1
    
    alpha2 = float(np.arctan2(L2 * np.sin(theta2_2), L1 + L2 * np.cos(theta2_2)))
    theta1_2 = beta - alpha2
    
    elbow1_x = L1 * np.cos(theta1_1)
    elbow2_x = L1 * np.cos(theta1_2)
    
    if elbow1_x < elbow2_x:
        return (theta1_1, theta2_1), (theta1_2, theta2_2)
    else:
        return (theta1_2, theta2_2), (theta1_1, theta2_1)


# =====================================================================
# 2. Toy Dataset Generation (Section 6.5.1, Figure 6.17)
# =====================================================================

def generate_forward_data(n_samples: int = 250, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample synthetic toy forward data:
      x ~ Uniform(0, 1)
      t = x + 0.3 * sin(2 * pi * x) + Uniform(-0.1, 0.1)
    """
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.0, 1.0, n_samples)
    noise = rng.uniform(-0.1, 0.1, n_samples)
    t = x + 0.3 * np.sin(2.0 * np.pi * x) + noise
    return x, t


def generate_inverse_data(n_samples: int = 250, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample synthetic toy inverse data:
      Obtained by exchanging the roles of x and t in forward data.
      Returns (x_inv, t_inv) where x_inv = t_fwd, t_inv = x_fwd.
    """
    x_fwd, t_fwd = generate_forward_data(n_samples=n_samples, seed=seed)
    return t_fwd, x_fwd


# =====================================================================
# 3. Standard Two-Layer MLP Least Squares Regressor
# =====================================================================

class StandardMLPRegressor:
    """
    Standard two-layer neural network with tanh hidden units and linear output
    trained by minimizing sum-of-squares error (Gaussian conditional assumption).
    """
    def __init__(self, n_in: int = 1, n_hidden: int = 6, n_out: int = 1, seed: int = 42):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        rng = np.random.RandomState(seed)
        self.W1 = rng.normal(0, 0.5, (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 0.5, (n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def pack(self) -> np.ndarray:
        return np.concatenate([
            self.W1.ravel(),
            self.b1.ravel(),
            self.W2.ravel(),
            self.b2.ravel()
        ])

    def unpack(self, params: np.ndarray) -> None:
        idx = 0
        w1_s = self.n_in * self.n_hidden
        self.W1 = params[idx:idx + w1_s].reshape(self.n_in, self.n_hidden)
        idx += w1_s
        self.b1 = params[idx:idx + self.n_hidden]
        idx += self.n_hidden
        w2_s = self.n_hidden * self.n_out
        self.W2 = params[idx:idx + w2_s].reshape(self.n_hidden, self.n_out)
        idx += w2_s
        self.b2 = params[idx:idx + self.n_out]

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X.ndim == 1:
            X = X[:, None]
        a1 = X @ self.W1 + self.b1
        z = np.tanh(a1)
        a2 = z @ self.W2 + self.b2
        return a2, z

    def loss_and_grad(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        self.unpack(params)
        y_pred, z = self.forward(X)
        diff = y_pred - y
        loss = 0.5 * np.sum(diff**2)
        
        delta2 = diff
        dW2 = z.T @ delta2
        db2 = np.sum(delta2, axis=0)
        
        dz = delta2 @ self.W2.T
        da1 = dz * (1.0 - z**2)
        dW1 = X.T @ da1
        db1 = np.sum(da1, axis=0)
        
        grad = np.concatenate([
            dW1.ravel(),
            db1.ravel(),
            dW2.ravel(),
            db2.ravel()
        ])
        return loss, grad

    def fit(self, X: np.ndarray, y: np.ndarray, maxiter: int = 500) -> "StandardMLPRegressor":
        if X.ndim == 1:
            X = X[:, None]
        if y.ndim == 1:
            y = y[:, None]
        p0 = self.pack()
        res = opt.minimize(
            self.loss_and_grad,
            p0,
            args=(X, y),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter}
        )
        self.unpack(res.x)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X[:, None]
        y_pred, _ = self.forward(X)
        return y_pred.ravel() if self.n_out == 1 else y_pred


# =====================================================================
# 4. Mixture Density Network (MDN)
# =====================================================================

class MixtureDensityNetwork:
    """
    Mixture Density Network (Bishop & Bishop 2024, Section 6.5).
    
    Predicts conditional mixture density:
      p(t|x) = sum_{k=1}^K pi_k(x) N(t | mu_k(x), sigma_k^2(x) I)
      
    Network output dimensions: (L + 2) * K
      - a^{pi}_k: mixing coefficient pre-activations (softmax, Eq 6.40)
      - a^{sigma}_k: variance pre-activations (exponential, Eq 6.41)
      - a^{mu}_{kj}: component mean pre-activations (linear, Eq 6.42)
    """
    def __init__(
        self,
        n_in: int = 1,
        n_hidden: int = 5,
        n_components: int = 3,
        target_dim: int = 1,
        seed: int = 42
    ):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_components = n_components
        self.target_dim = target_dim
        self.n_out = (target_dim + 2) * n_components  # (L + 2) * K
        
        rng = np.random.RandomState(seed)
        self.W1 = rng.normal(0, 0.5, (n_in, n_hidden))
        self.b1 = rng.normal(0, 0.5, (n_hidden,))
        
        self.W2 = rng.normal(0, 0.5, (n_hidden, self.n_out))
        self.b2 = np.zeros(self.n_out)
        
        # Smart initialization for toy inverse problem
        if target_dim == 1 and n_components == 3:
            # Spread means across [0.15, 0.50, 0.85]
            self.b2[6:9] = np.array([0.15, 0.50, 0.85])
            # Initialize std deviations to ~0.1
            self.b2[3:6] = np.log(0.1)
            # Uniform mixing prior
            self.b2[0:3] = 0.0

    def pack(self) -> np.ndarray:
        return np.concatenate([
            self.W1.ravel(),
            self.b1.ravel(),
            self.W2.ravel(),
            self.b2.ravel()
        ])

    def unpack(self, params: np.ndarray) -> None:
        idx = 0
        w1_s = self.n_in * self.n_hidden
        self.W1 = params[idx:idx + w1_s].reshape(self.n_in, self.n_hidden)
        idx += w1_s
        self.b1 = params[idx:idx + self.n_hidden]
        idx += self.n_hidden
        w2_s = self.n_hidden * self.n_out
        self.W2 = params[idx:idx + w2_s].reshape(self.n_hidden, self.n_out)
        idx += w2_s
        self.b2 = params[idx:idx + self.n_out]

    def forward(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple]:
        """
        Forward pass computing mixing coefficients, variances, and means.
        
        Returns:
            pi: (N, K)
            sigma: (N, K)
            mu: (N, K) for L=1 or (N, K, L) for L>1
            cache: intermediate activations for backprop
        """
        if X.ndim == 1:
            X = X[:, None]
            
        a1 = X @ self.W1 + self.b1
        z = np.tanh(a1)
        a2 = z @ self.W2 + self.b2
        
        K = self.n_components
        L = self.target_dim
        
        a_pi = a2[:, :K]
        a_sigma = a2[:, K:2 * K]
        a_mu = a2[:, 2 * K:]
        
        # Softmax for mixing coefficients (Eq 6.40)
        a_pi_max = np.max(a_pi, axis=1, keepdims=True)
        exp_pi = np.exp(a_pi - a_pi_max)
        pi = exp_pi / np.sum(exp_pi, axis=1, keepdims=True)
        
        # Exponential for variances (Eq 6.41), clamped for numerical stability
        sigma = np.exp(np.clip(a_sigma, -7.0, 5.0))
        
        # Linear for means (Eq 6.42)
        if L == 1:
            mu = a_mu
        else:
            mu = a_mu.reshape(-1, K, L)
            
        cache = (X, a1, z, a2, pi, sigma, mu, a_sigma)
        return pi, sigma, mu, cache

    def loss_and_grad(
        self,
        params: np.ndarray,
        X: np.ndarray,
        T: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Evaluate negative log likelihood error E(w) and analytical gradient dE/dw.
        
        Equations 6.43, 6.44, 6.45, 6.46, 6.47.
        """
        self.unpack(params)
        pi, sigma, mu, cache = self.forward(X)
        X, a1, z, a2, _, _, _, a_sigma = cache
        
        N = X.shape[0]
        K = self.n_components
        L = self.target_dim
        
        if T.ndim == 1:
            T = T[:, None]
            
        if L == 1:
            diff = T - mu  # (N, K)
            norm_sq = diff**2
            log_prob_k = -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * (norm_sq / (sigma**2))
        else:
            diff = T[:, None, :] - mu  # (N, K, L)
            norm_sq = np.sum(diff**2, axis=2)  # (N, K)
            log_prob_k = -0.5 * L * np.log(2.0 * np.pi) - L * np.log(sigma) - 0.5 * (norm_sq / (sigma**2))
            
        log_joint = np.log(np.clip(pi, 1e-15, 1.0)) + log_prob_k
        max_log_joint = np.max(log_joint, axis=1, keepdims=True)
        log_p_t = max_log_joint + np.log(np.sum(np.exp(log_joint - max_log_joint), axis=1, keepdims=True))
        
        loss = -float(np.sum(log_p_t))
        
        # Responsibilities gamma_nk (Eq 6.44)
        gamma = np.exp(log_joint - log_p_t)  # (N, K)
        
        # Analytical pre-activation derivatives:
        # Eq 6.45: dE_n / da_pi_k = pi_k - gamma_nk
        d_a_pi = pi - gamma  # (N, K)
        
        # Eq 6.47: dE_n / da_sigma_k = gamma_nk * (L - ||t_n - mu_k||^2 / sigma_k^2)
        d_a_sigma = gamma * (L - norm_sq / (sigma**2))  # (N, K)
        
        # Eq 6.46: dE_n / da_mu_{kl} = gamma_nk * (mu_{kl} - t_{nl}) / sigma_k^2
        if L == 1:
            d_a_mu = -gamma * (diff / (sigma**2))  # (N, K)
        else:
            d_a_mu = -gamma[:, :, None] * (diff / (sigma[:, :, None]**2))  # (N, K, L)
            d_a_mu = d_a_mu.reshape(N, K * L)
            
        delta2 = np.zeros((N, self.n_out))
        delta2[:, :K] = d_a_pi
        delta2[:, K:2 * K] = d_a_sigma
        delta2[:, 2 * K:] = d_a_mu
        
        # Backpropagation
        dW2 = z.T @ delta2
        db2 = np.sum(delta2, axis=0)
        
        dz = delta2 @ self.W2.T
        da1 = dz * (1.0 - z**2)
        
        dW1 = X.T @ da1
        db1 = np.sum(da1, axis=0)
        
        grad = np.concatenate([
            dW1.ravel(),
            db1.ravel(),
            dW2.ravel(),
            db2.ravel()
        ])
        
        return loss, grad

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        maxiter: int = 1500,
        seed: Optional[int] = None
    ) -> "MixtureDensityNetwork":
        """
        Train the Mixture Density Network using L-BFGS-B optimization.
        """
        if seed is not None:
            self.__init__(
                n_in=self.n_in,
                n_hidden=self.n_hidden,
                n_components=self.n_components,
                target_dim=self.target_dim,
                seed=seed
            )
            
        if X.ndim == 1:
            X = X[:, None]
        if T.ndim == 1:
            T = T[:, None]
            
        p0 = self.pack()
        res = opt.minimize(
            self.loss_and_grad,
            p0,
            args=(X, T),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter}
        )
        self.unpack(res.x)
        self.final_loss_ = res.fun
        return self

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate conditional mean E[t|x] = sum_k pi_k(x) mu_k(x) (Eq 6.48).
        """
        pi, _, mu, _ = self.forward(X)
        if self.target_dim == 1:
            return np.sum(pi * mu, axis=1)
        else:
            return np.sum(pi[:, :, None] * mu, axis=1)

    def predict_variance(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate conditional variance s^2(x) (Eq 6.50).
        """
        pi, sigma, mu, _ = self.forward(X)
        mean = self.predict_mean(X)
        if self.target_dim == 1:
            mean = mean[:, None]
            diff_sq = (mu - mean)**2
            return np.sum(pi * (sigma**2 + diff_sq), axis=1)
        else:
            mean = mean[:, None, :]
            diff_sq = np.sum((mu - mean)**2, axis=2)
            return np.sum(pi * (self.target_dim * (sigma**2) + diff_sq), axis=1)

    def predict_mode_approx(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate approximate conditional mode t* = mu_{k*}(x)
        where k* = argmax_k pi_k(x) (Section 6.5.4, Figure 6.19d).
        """
        pi, _, mu, _ = self.forward(X)
        k_star = np.argmax(pi, axis=1)
        if self.target_dim == 1:
            return mu[np.arange(len(X)), k_star]
        else:
            return mu[np.arange(len(X)), k_star, :]

    def predict_density(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Evaluate conditional mixture density p(t|x) at given pairs (x, t).
        """
        pi, sigma, mu, _ = self.forward(X)
        if T.ndim == 1:
            T = T[:, None]
            
        if self.target_dim == 1:
            diff = T - mu
            dens_k = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * (diff / sigma)**2)
            return np.sum(pi * dens_k, axis=1)
        else:
            diff = T[:, None, :] - mu
            norm_sq = np.sum(diff**2, axis=2)
            dens_k = (1.0 / ((2.0 * np.pi)**(self.target_dim / 2.0) * (sigma**self.target_dim))) * np.exp(-0.5 * norm_sq / (sigma**2))
            return np.sum(pi * dens_k, axis=1)

    def sample(self, X: np.ndarray, n_samples: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """
        Sample targets t ~ p(t|x) from the conditional mixture model.
        """
        rng = np.random.RandomState(seed)
        pi, sigma, mu, _ = self.forward(X)
        N = X.shape[0]
        K = self.n_components
        L = self.target_dim
        
        samples = np.zeros((N, n_samples) if L == 1 else (N, n_samples, L))
        for i in range(N):
            comp_indices = rng.choice(K, size=n_samples, p=pi[i])
            for s, k in enumerate(comp_indices):
                if L == 1:
                    samples[i, s] = rng.normal(mu[i, k], sigma[i, k])
                else:
                    samples[i, s] = rng.normal(mu[i, k], sigma[i, k], size=L)
        return samples


def _get_project_root() -> str:
    """Return the absolute path to the repository root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _save_figure(
    fig: plt.Figure,
    filename: str,
    filepath: Optional[str] = None,
    save_both: bool = True
) -> Tuple[str, Optional[str]]:
    """Save figure to target filepath, and/or both 6/result and root result directory."""
    root = _get_project_root()
    path_ch6 = os.path.join(root, "6", "result", filename)
    path_root = os.path.join(root, "result", filename)
    if filepath:
        save_plot(fig, filepath)
        if save_both:
            if os.path.abspath(filepath) != os.path.abspath(path_root):
                save_plot(fig, path_root)
            if os.path.abspath(filepath) != os.path.abspath(path_ch6):
                save_plot(fig, path_ch6)
        return filepath, path_root if save_both else None
    else:
        save_plot(fig, path_ch6)
        if save_both:
            save_plot(fig, path_root)
        return path_ch6, path_root if save_both else None


# =====================================================================
# 5. Figure Generators (Figures 6.16 - 6.19)
# =====================================================================

def generate_figure_6_16(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """
    Figure 6.16: Robot kinematics example.
    (a) Forward kinematics: End effector Cartesian coordinates (x1, x2) determined
        uniquely by joint angles theta1, theta2.
    (b) Inverse kinematics: Two solutions ('elbow up' and 'elbow down').
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    
    def draw_ground(ax, x_center=0.0):
        ax.plot([x_center - 0.7, x_center + 0.7], [0, 0], color='black', lw=2.5, zorder=2)
        rect = Rectangle((x_center - 0.7, -0.08), 1.4, 0.08, facecolor='#cccccc', edgecolor='none', zorder=1)
        ax.add_patch(rect)

    L1 = 1.0
    L2 = 0.65
    theta1 = np.radians(115)
    theta2 = np.radians(-75)
    
    x_end, y_end = forward_kinematics(theta1, theta2, L1, L2)
    elbow_x = L1 * np.cos(theta1)
    elbow_y = L1 * np.sin(theta1)
    
    # --- Panel (a): Forward Kinematics ---
    ax = axes[0]
    draw_ground(ax)
    ax.plot([0, elbow_x, x_end], [0, elbow_y, y_end], color='red', lw=5.0, zorder=3, solid_capstyle='round')
    ax.scatter([x_end], [y_end], color='blue', s=120, zorder=4)
    ax.text(x_end + 0.05, y_end, r'$(x_1, x_2)$', fontsize=13, verticalalignment='center')
    
    ax.text(elbow_x * 0.5 - 0.12, elbow_y * 0.5, r'$L_1$', fontsize=13, horizontalalignment='center')
    ax.text((elbow_x + x_end) * 0.5 - 0.08, (elbow_y + y_end) * 0.5 + 0.08, r'$L_2$', fontsize=13, horizontalalignment='center')
    
    # Angle theta 1 arc
    arc_theta1 = Arc((0, 0), 0.5, 0.5, angle=0, theta1=0, theta2=115, color='black', lw=1.5)
    ax.add_patch(arc_theta1)
    arrow_t1_x = 0.25 * np.cos(np.radians(115))
    arrow_t1_y = 0.25 * np.sin(np.radians(115))
    ax.annotate('', xy=(arrow_t1_x, arrow_t1_y), xytext=(arrow_t1_x + 0.04, arrow_t1_y - 0.03),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(0.18, 0.22, r'$\theta_1$', fontsize=13)
    
    # Angle theta 2 arc (interior angle)
    arc_theta2 = Arc((elbow_x, elbow_y), 0.35, 0.35, angle=0, theta1=295, theta2=400, color='black', lw=1.5)
    ax.add_patch(arc_theta2)
    arrow_t2_x = elbow_x + 0.175 * np.cos(np.radians(40))
    arrow_t2_y = elbow_y + 0.175 * np.sin(np.radians(40))
    ax.annotate('', xy=(arrow_t2_x, arrow_t2_y), xytext=(arrow_t2_x + 0.02, arrow_t2_y - 0.04),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(elbow_x + 0.14, elbow_y - 0.08, r'$\theta_2$', fontsize=13)
    
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.15, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(0.0, -0.22, '(a)', fontsize=14, horizontalalignment='center')
    
    # --- Panel (b): Inverse Kinematics ---
    ax = axes[1]
    draw_ground(ax)
    
    (up, down) = inverse_kinematics(x_end, y_end, L1, L2)
    t1_up, t2_up = up
    t1_down, t2_down = down
    
    e_up_x = L1 * np.cos(t1_up)
    e_up_y = L1 * np.sin(t1_up)
    
    e_down_x = L1 * np.cos(t1_down)
    e_down_y = L1 * np.sin(t1_down)
    
    # Solid: elbow up
    ax.plot([0, e_up_x, x_end], [0, e_up_y, y_end], color='red', lw=5.0, zorder=3, solid_capstyle='round')
    # Dashed: elbow down
    ax.plot([0, e_down_x, x_end], [0, e_down_y, y_end], color='red', lw=3.5, linestyle='--', zorder=3)
    
    ax.scatter([x_end], [y_end], color='blue', s=120, zorder=4)
    ax.text(x_end + 0.05, y_end, r'$(x_1, x_2)$', fontsize=13, verticalalignment='center')
    
    ax.text(e_up_x - 0.12, e_up_y * 0.75, 'elbow\nup', fontsize=11, horizontalalignment='right', verticalalignment='center')
    ax.text(e_down_x + 0.08, e_down_y * 0.65, 'elbow\ndown', fontsize=11, horizontalalignment='left', verticalalignment='center')
    
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.15, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(0.0, -0.22, '(b)', fontsize=14, horizontalalignment='center')
    
    plt.tight_layout()
    _save_figure(fig, "fig_6_16_robot_kinematics.png", filepath=filepath, save_both=save_both)
    return fig



def generate_figure_6_17(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """
    Figure 6.17: Forward and inverse toy problems with least-squares MLP fit.
    Left: Forward problem (good fit).
    Right: Inverse problem (poor fit due to multimodality).
    """
    x_fwd, t_fwd = generate_forward_data(n_samples=250, seed=42)
    x_inv, t_inv = generate_inverse_data(n_samples=250, seed=42)
    
    mlp_fwd = StandardMLPRegressor(n_hidden=6, seed=42).fit(x_fwd, t_fwd)
    mlp_inv = StandardMLPRegressor(n_hidden=6, seed=42).fit(x_inv, t_inv)
    
    x_plot = np.linspace(-0.1, 1.1, 250)
    pred_fwd = mlp_fwd.predict(x_plot)
    pred_inv = mlp_inv.predict(x_plot)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Left: Forward problem
    axes[0].scatter(x_fwd, t_fwd, facecolors='none', edgecolors='#00e600', s=25, alpha=0.85)
    axes[0].plot(x_plot, pred_fwd, color='red', lw=2.0)
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    
    # Right: Inverse problem
    axes[1].scatter(x_inv, t_inv, facecolors='none', edgecolors='#00e600', s=25, alpha=0.85)
    axes[1].plot(x_plot, pred_inv, color='red', lw=2.0)
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    
    plt.tight_layout()
    _save_figure(fig, "fig_6_17_forward_inverse_problems.png", filepath=filepath, save_both=save_both)
    return fig


def generate_figure_6_18(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """
    Figure 6.18: Mixture Density Network architecture diagram and multimodal
    conditional probability density p(t|x).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={'width_ratios': [1.1, 1.0]})
    
    # --- Left: Network diagram ---
    ax1.set_xlim(-0.2, 1.8)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    node_radius = 0.08
    node_face = '#e6edff'
    node_edge = '#0000cc'
    node_lw = 1.8
    
    inputs = [('x_1', 0.0, 0.3), ('x_D', 0.0, 0.7)]
    hidden = [('', 0.5, 0.15), ('', 0.5, 0.85)]
    outputs = [(r'$\theta_1$', 1.0, 0.3), (r'$\theta_K$', 1.0, 0.7)]
    
    for _, ix, iy in inputs:
        for _, hx, hy in hidden:
            dx = hx - ix
            dy = hy - iy
            dist = np.hypot(dx, dy)
            start_x = ix + dx * (node_radius / dist)
            start_y = iy + dy * (node_radius / dist)
            end_x = hx - dx * (node_radius / dist)
            end_y = hy - dy * (node_radius / dist)
            ax1.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                         arrowprops=dict(arrowstyle="->", color="black", lw=1.3, mutation_scale=12))
            
    for _, hx, hy in hidden:
        for _, ox, oy in outputs:
            dx = ox - hx
            dy = oy - hy
            dist = np.hypot(dx, dy)
            start_x = hx + dx * (node_radius / dist)
            start_y = hy + dy * (node_radius / dist)
            end_x = ox - dx * (node_radius / dist)
            end_y = oy - dy * (node_radius / dist)
            ax1.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                         arrowprops=dict(arrowstyle="->", color="black", lw=1.3, mutation_scale=12))
            
    for label, x, y in inputs:
        c = Circle((x, y), node_radius, facecolor=node_face, edgecolor=node_edge, lw=node_lw, zorder=5)
        ax1.add_patch(c)
        ax1.text(x, y, r'$' + label + '$', fontsize=13, ha='center', va='center', zorder=6)
        
    for label, x, y in hidden:
        c = Circle((x, y), node_radius, facecolor=node_face, edgecolor=node_edge, lw=node_lw, zorder=5)
        ax1.add_patch(c)
        
    for label, x, y in outputs:
        c = Circle((x, y), node_radius, facecolor=node_face, edgecolor=node_edge, lw=node_lw, zorder=5)
        ax1.add_patch(c)
        ax1.text(x, y, label, fontsize=13, ha='center', va='center', zorder=6)
        
    ax1.text(0.0, 0.5, r'$\vdots$', fontsize=16, ha='center', va='center')
    ax1.text(0.5, 0.5, r'$\vdots$', fontsize=16, ha='center', va='center')
    ax1.text(1.0, 0.5, r'$\vdots$', fontsize=16, ha='center', va='center')
    
    ax1.annotate('', xy=(1.55, 0.5), xytext=(1.2, 0.5),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.8, mutation_scale=15))
    ax1.text(1.375, 0.57, r'$\boldsymbol{\theta}$', fontsize=15, ha='center')

    # --- Right: Density plot p(t|x) ---
    t = np.linspace(-3, 8, 500)
    
    pi1, mu1, sig1 = 0.30, -0.5, 0.7
    pi2, mu2, sig2 = 0.25, 2.5, 1.0
    pi3, mu3, sig3 = 0.45, 5.0, 0.5
    
    comp1 = pi1 * (1.0 / (np.sqrt(2.0 * np.pi) * sig1)) * np.exp(-0.5 * ((t - mu1) / sig1)**2)
    comp2 = pi2 * (1.0 / (np.sqrt(2.0 * np.pi) * sig2)) * np.exp(-0.5 * ((t - mu2) / sig2)**2)
    comp3 = pi3 * (1.0 / (np.sqrt(2.0 * np.pi) * sig3)) * np.exp(-0.5 * ((t - mu3) / sig3)**2)
    mixture = comp1 + comp2 + comp3
    
    ax2.plot(t, comp1, color='#0033cc', lw=2.0)
    ax2.plot(t, comp2, color='#0033cc', lw=2.0)
    ax2.plot(t, comp3, color='#0033cc', lw=2.0)
    ax2.plot(t, mixture, color='red', lw=2.2)
    
    ax2.set_xlim(-2.5, 7.5)
    ax2.set_ylim(-0.02, 0.45)
    ax2.set_xlabel(r'$t$', fontsize=14)
    ax2.text(-2.2, 0.38, r'$p(t|x)$', fontsize=14)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.2)
        
    plt.tight_layout()
    _save_figure(fig, "fig_6_18_mixture_density_network.png", filepath=filepath, save_both=save_both)
    return fig


def generate_figure_6_19(
    filepath: Optional[str] = None,
    save_both: bool = True,
    mdn_model: Optional[MixtureDensityNetwork] = None
) -> plt.Figure:
    """
    Figure 6.19: Mixture density network predictions for the inverse problem.
    (a) Mixing coefficients pi_k(x) as a function of x.
    (b) Component means mu_k(x) as a function of x.
    (c) Contours of conditional density p(t|x).
    (d) Approximate conditional mode t* (red dots) overlaid on data.
    """
    x_inv, t_inv = generate_inverse_data(n_samples=250, seed=42)
    
    if mdn_model is None:
        mdn_model = MixtureDensityNetwork(n_in=1, n_hidden=5, n_components=3, seed=9)
        mdn_model.fit(x_inv, t_inv, maxiter=1500)
        
    x_test = np.linspace(0.0, 1.0, 300)[:, None]
    pi_test, sigma_test, mu_test, _ = mdn_model.forward(x_test)
    
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # blue, green, red
    
    # --- (a) Mixing coefficients ---
    for k in range(3):
        axes[0, 0].plot(x_test, pi_test[:, k], color=colors[k], lw=2.0)
    axes[0, 0].set_xlim(-0.05, 1.05)
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_yticks([0, 1])
    axes[0, 0].text(0.5, -0.15, '(a)', fontsize=13, ha='center', transform=axes[0, 0].transAxes)
    
    # --- (b) Means ---
    for k in range(3):
        axes[0, 1].plot(x_test, mu_test[:, k], color=colors[k], lw=2.0)
    axes[0, 1].set_xlim(-0.05, 1.05)
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_yticks([0, 1])
    axes[0, 1].text(0.5, -0.15, '(b)', fontsize=13, ha='center', transform=axes[0, 1].transAxes)
    
    # --- (c) Density contours ---
    grid_x = np.linspace(-0.05, 1.05, 200)
    grid_t = np.linspace(-0.05, 1.05, 200)
    GX, GT = np.meshgrid(grid_x, grid_t)
    
    pi_g, sigma_g, mu_g, _ = mdn_model.forward(GX.ravel()[:, None])
    T_g = GT.ravel()[:, None]
    diff_g = T_g - mu_g
    dens_k = (1.0 / (np.sqrt(2.0 * np.pi) * sigma_g)) * np.exp(-0.5 * (diff_g / sigma_g)**2)
    joint_dens = np.sum(pi_g * dens_k, axis=1).reshape(GX.shape)
    
    levels = [0.15, 0.35, 0.8, 1.8, 3.5, 6.0, 10.0, 15.0]
    axes[1, 0].contour(GX, GT, joint_dens, levels=[l for l in levels if l < np.max(joint_dens)], cmap='jet')
    axes[1, 0].set_xlim(-0.05, 1.05)
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].text(0.5, -0.15, '(c)', fontsize=13, ha='center', transform=axes[1, 0].transAxes)
    
    # --- (d) Approximate conditional mode ---
    max_k = np.argmax(pi_test, axis=1)
    mode_t = mu_test[np.arange(len(x_test)), max_k]
    
    axes[1, 1].scatter(x_inv, t_inv, facecolors='none', edgecolors='#00e600', s=25, alpha=0.85)
    axes[1, 1].scatter(x_test, mode_t, color='red', s=8, zorder=5)
    axes[1, 1].set_xlim(-0.05, 1.05)
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].text(0.5, -0.15, '(d)', fontsize=13, ha='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    _save_figure(fig, "fig_6_19_mdn_predictions.png", filepath=filepath, save_both=save_both)
    return fig

