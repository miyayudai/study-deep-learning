"""
common/parameter_sharing.py
===========================
Section 9.4: Parameter Sharing & Soft Weight Sharing
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts
Nowlan & Hinton (1992), Simplifying Neural Networks by Soft Weight-Sharing

This module implements:
1. SoftWeightSharingGMM: Gaussian Mixture Model (GMM) prior over network weights,
   computing analytical penalties and exact gradients with respect to weights,
   component centers, log-variances, and softmax logits (Eqs 9.21 - 9.31).
2. SoftWeightSharingMLP: A 2-layer neural network trained jointly with soft weight
   sharing regularization, demonstrating weight clustering and parameter compression.
3. HardWeightSharingMLP: A neural network enforcing strict equality constraints
   across parameter groups via pooled gradients.
4. Publication-quality figure generation functions for prior densities, energy
   surfaces, restoring forces, clustering evolution, and comparative analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .plot_utils import setup_style, save_plot


class SoftWeightSharingGMM:
    """
    Gaussian Mixture Model (GMM) prior over neural network weights.

    Textbook equations:
    -------------------
    1. Prior density over weights (Eq 9.21):
       p(w) = prod_i [ sum_{j=1}^K pi_j N(w_i | mu_j, sigma_j^2) ]

    2. Regularization penalty (Eq 9.22):
       Omega(w) = -ln p(w) = -sum_i ln [ sum_{j=1}^K pi_j N(w_i | mu_j, sigma_j^2) ]

    3. Total objective (Eq 9.23):
       \\widetilde{E}(w) = E(w) + lambda * Omega(w)

    4. Posterior responsibilities (Eq 9.24):
       gamma_j(w_i) = pi_j N(w_i | mu_j, sigma_j^2) / [ sum_k pi_k N(w_i | mu_k, sigma_k^2) ]

    5. Derivative with respect to weights w_i (Eq 9.25):
       d\\widetilde{E} / dw_i = dE / dw_i + lambda * sum_j gamma_j(w_i) (w_i - mu_j) / sigma_j^2

    6. Derivative with respect to centers mu_j (Eq 9.26):
       d\\widetilde{E} / dmu_j = lambda * sum_i gamma_j(w_i) (mu_j - w_i) / sigma_j^2

    7. Derivative with respect to log-variances beta_j = ln(sigma_j^2) (Eq 9.27 - 9.28):
       d\\widetilde{E} / dbeta_j = (lambda / 2) * sum_i gamma_j(w_i) [ 1 - (w_i - mu_j)^2 / sigma_j^2 ]

    8. Derivative with respect to softmax logits eta_j (Eq 9.30 - 9.31):
       pi_j = exp(eta_j) / sum_k exp(eta_k)
       d\\widetilde{E} / deta_j = lambda * sum_i [ pi_j - gamma_j(w_i) ]
    """

    def __init__(
        self,
        n_components: int = 3,
        mu: Optional[Union[List[float], np.ndarray]] = None,
        beta: Optional[Union[List[float], np.ndarray]] = None,
        logits: Optional[Union[List[float], np.ndarray]] = None,
        min_beta: float = -6.0,
        max_beta: float = 4.0,
    ) -> None:
        self.K = n_components
        self.min_beta = min_beta
        self.max_beta = max_beta

        if mu is not None:
            self.mu = np.array(mu, dtype=float)
        else:
            self.mu = np.linspace(-1.5, 1.5, self.K)

        if beta is not None:
            self.beta = np.array(beta, dtype=float)
        else:
            self.beta = np.full(self.K, np.log(0.2))

        if logits is not None:
            self.logits = np.array(logits, dtype=float)
        else:
            self.logits = np.zeros(self.K)

        self._clip_beta()

    def _clip_beta(self) -> None:
        """Clip log-variances to prevent singularity/collapse."""
        self.beta = np.clip(self.beta, self.min_beta, self.max_beta)

    @property
    def sigma2(self) -> np.ndarray:
        """Component variances sigma_j^2 = exp(beta_j) (Eq 9.27)."""
        return np.exp(self.beta)

    @property
    def sigma(self) -> np.ndarray:
        """Component standard deviations."""
        return np.sqrt(self.sigma2)

    @property
    def pi(self) -> np.ndarray:
        """Mixing proportions pi_j via stable softmax (Eq 9.30)."""
        shifted = self.logits - np.max(self.logits)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals)

    def compute_log_joint(self, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute log pi_j + log N(w_i | mu_j, sigma_j^2) for all weights and components.
        Returns:
            log_joint: (W, K) matrix of component log densities.
            log_marginal: (W, 1) vector of log marginal densities ln p(w_i).
        """
        w_flat = np.asarray(w).reshape(-1, 1)  # (W, 1)
        mu = self.mu.reshape(1, -1)           # (1, K)
        s2 = self.sigma2.reshape(1, -1)       # (1, K)
        pi = self.pi.reshape(1, -1)           # (1, K)

        log_norm = -0.5 * np.log(2.0 * np.pi) - 0.5 * self.beta.reshape(1, -1)
        log_quad = -0.5 * ((w_flat - mu) ** 2) / s2
        log_gauss = log_norm + log_quad

        log_joint = np.log(np.maximum(pi, 1e-15)) + log_gauss  # (W, K)

        max_log = np.max(log_joint, axis=1, keepdims=True)
        log_marginal = max_log + np.log(np.sum(np.exp(log_joint - max_log), axis=1, keepdims=True))

        return log_joint, log_marginal

    def responsibilities(self, w: np.ndarray) -> np.ndarray:
        """
        Posterior probability gamma_j(w_i) that component j generated weight w_i (Eq 9.24).
        """
        log_joint, log_marginal = self.compute_log_joint(w)
        log_gamma = log_joint - log_marginal
        return np.exp(log_gamma)

    def penalty(self, w: np.ndarray) -> float:
        """
        Negative log-likelihood regularizer Omega(w) = -ln p(w) (Eq 9.22).
        """
        _, log_marginal = self.compute_log_joint(w)
        return float(-np.sum(log_marginal))

    def grad_w(
        self,
        w: np.ndarray,
        grad_E: Optional[np.ndarray] = None,
        lambda_reg: float = 1.0,
    ) -> np.ndarray:
        """
        Analytical derivative of total error with respect to weights (Eq 9.25).
        """
        w_arr = np.asarray(w)
        w_flat = w_arr.ravel()
        gamma = self.responsibilities(w_flat)  # (W, K)

        mu = self.mu.reshape(1, -1)
        s2 = self.sigma2.reshape(1, -1)

        diff = (w_flat.reshape(-1, 1) - mu) / s2
        grad_omega = np.sum(gamma * diff, axis=1)

        total_grad = lambda_reg * grad_omega
        if grad_E is not None:
            total_grad += grad_E.ravel()

        return total_grad.reshape(w_arr.shape)

    def grad_mu(self, w: np.ndarray, lambda_reg: float = 1.0) -> np.ndarray:
        """
        Analytical derivative with respect to Gaussian centres mu_j (Eq 9.26).
        """
        w_flat = np.asarray(w).ravel()
        gamma = self.responsibilities(w_flat)  # (W, K)

        mu = self.mu.reshape(1, -1)
        s2 = self.sigma2.reshape(1, -1)

        diff = (mu - w_flat.reshape(-1, 1)) / s2
        grad_mu = np.sum(gamma * diff, axis=0)

        return lambda_reg * grad_mu

    def grad_beta(self, w: np.ndarray, lambda_reg: float = 1.0) -> np.ndarray:
        """
        Analytical derivative with respect to log-variances beta_j (Eq 9.28).
        """
        w_flat = np.asarray(w).ravel()
        gamma = self.responsibilities(w_flat)  # (W, K)

        mu = self.mu.reshape(1, -1)
        s2 = self.sigma2.reshape(1, -1)

        quad = ((w_flat.reshape(-1, 1) - mu) ** 2) / s2
        grad_beta = 0.5 * np.sum(gamma * (1.0 - quad), axis=0)

        return lambda_reg * grad_beta

    def grad_logits(self, w: np.ndarray, lambda_reg: float = 1.0) -> np.ndarray:
        """
        Analytical derivative with respect to softmax logits eta_j (Eq 9.31).
        """
        w_flat = np.asarray(w).ravel()
        gamma = self.responsibilities(w_flat)  # (W, K)
        pi = self.pi.reshape(1, -1)

        grad_logits = np.sum(pi - gamma, axis=0)
        return lambda_reg * grad_logits

    def effective_force(self, w: np.ndarray) -> np.ndarray:
        """
        Effective restoring force -dOmega / dw pulling weights towards centers.
        """
        w_flat = np.asarray(w).ravel()
        gamma = self.responsibilities(w_flat)
        mu = self.mu.reshape(1, -1)
        s2 = self.sigma2.reshape(1, -1)

        force = np.sum(gamma * (mu - w_flat.reshape(-1, 1)) / s2, axis=1)
        return force.reshape(np.asarray(w).shape)

    def check_gradients(
        self,
        w: np.ndarray,
        lambda_reg: float = 1.0,
        eps: float = 1e-6,
    ) -> Dict[str, float]:
        """
        Verify analytical gradients against finite-difference numerical approximations.
        """
        w_arr = np.asarray(w, dtype=float).ravel()

        analytic_gw = self.grad_w(w_arr, lambda_reg=lambda_reg).ravel()
        num_gw = np.zeros_like(w_arr)
        for i in range(len(w_arr)):
            w_plus = w_arr.copy()
            w_minus = w_arr.copy()
            w_plus[i] += eps
            w_minus[i] -= eps
            num_gw[i] = lambda_reg * (self.penalty(w_plus) - self.penalty(w_minus)) / (2.0 * eps)
        err_w = float(np.max(np.abs(analytic_gw - num_gw)))

        analytic_gmu = self.grad_mu(w_arr, lambda_reg=lambda_reg)
        num_gmu = np.zeros_like(self.mu)
        for j in range(self.K):
            self.mu[j] += eps
            p_plus = self.penalty(w_arr)
            self.mu[j] -= 2.0 * eps
            p_minus = self.penalty(w_arr)
            self.mu[j] += eps
            num_gmu[j] = lambda_reg * (p_plus - p_minus) / (2.0 * eps)
        err_mu = float(np.max(np.abs(analytic_gmu - num_gmu)))

        analytic_gbeta = self.grad_beta(w_arr, lambda_reg=lambda_reg)
        num_gbeta = np.zeros_like(self.beta)
        for j in range(self.K):
            self.beta[j] += eps
            p_plus = self.penalty(w_arr)
            self.beta[j] -= 2.0 * eps
            p_minus = self.penalty(w_arr)
            self.beta[j] += eps
            num_gbeta[j] = lambda_reg * (p_plus - p_minus) / (2.0 * eps)
        err_beta = float(np.max(np.abs(analytic_gbeta - num_gbeta)))

        analytic_glogits = self.grad_logits(w_arr, lambda_reg=lambda_reg)
        num_glogits = np.zeros_like(self.logits)
        for j in range(self.K):
            self.logits[j] += eps
            p_plus = self.penalty(w_arr)
            self.logits[j] -= 2.0 * eps
            p_minus = self.penalty(w_arr)
            self.logits[j] += eps
            num_glogits[j] = lambda_reg * (p_plus - p_minus) / (2.0 * eps)
        err_logits = float(np.max(np.abs(analytic_glogits - num_glogits)))

        return {
            "error_w": err_w,
            "error_mu": err_mu,
            "error_beta": err_beta,
            "error_logits": err_logits,
        }

    def update(
        self,
        grad_mu: np.ndarray,
        grad_beta: np.ndarray,
        grad_logits: np.ndarray,
        lr_mu: float = 0.01,
        lr_beta: float = 0.01,
        lr_logits: float = 0.01,
    ) -> None:
        """Perform gradient descent update on GMM parameters."""
        self.mu -= lr_mu * grad_mu
        self.beta -= lr_beta * grad_beta
        self._clip_beta()
        self.logits -= lr_logits * grad_logits
        self.logits -= np.mean(self.logits)


class SoftWeightSharingMLP:
    """
    2-Layer Multi-Layer Perceptron (MLP) with Soft Weight Sharing.
    Jointly trains network weights and GMM mixture components.
    """

    def __init__(
        self,
        d_in: int = 1,
        d_hidden: int = 64,
        d_out: int = 1,
        n_components: int = 3,
        seed: int = 42,
    ) -> None:
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.n_components = n_components

        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(d_in, d_hidden) * 0.7
        self.b1 = np.zeros((1, d_hidden))
        self.W2 = rng.randn(d_hidden, d_out) * 0.7
        self.b2 = np.zeros((1, d_out))

        if n_components == 1:
            mu_init = [0.0]
            beta_init = [0.0]
            logits_init = [0.0]
        elif n_components == 3:
            mu_init = [-1.0, 0.0, 1.0]
            beta_init = [np.log(0.1), np.log(0.05), np.log(0.1)]
            logits_init = [0.0, 0.0, 0.0]
        else:
            mu_init = np.linspace(-1.0, 1.0, n_components)
            beta_init = np.full(n_components, np.log(0.1))
            logits_init = np.zeros(n_components)

        self.gmm = SoftWeightSharingGMM(
            n_components=n_components,
            mu=mu_init,
            beta=beta_init,
            logits=logits_init,
        )

    def get_all_weights(self) -> np.ndarray:
        """Concatenate all network weights (excluding biases)."""
        return np.concatenate([self.W1.ravel(), self.W2.ravel()])

    def set_all_weights(self, w: np.ndarray) -> None:
        """Unpack concatenated weights back into layer weight matrices."""
        n_w1 = self.W1.size
        self.W1 = w[:n_w1].reshape(self.W1.shape)
        self.W2 = w[n_w1:].reshape(self.W2.shape)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through MLP: h = tanh(X @ W1 + b1), y = h @ W2 + b2."""
        a1 = X @ self.W1 + self.b1
        h = np.tanh(a1)
        y = h @ self.W2 + self.b2
        return h, y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute network predictions."""
        _, y = self.forward(X)
        return y

    def train_step(
        self,
        X: np.ndarray,
        y_target: np.ndarray,
        lr_w: float = 0.03,
        lr_gmm: float = 0.01,
        lambda_reg: float = 0.005,
    ) -> Dict[str, float]:
        """Perform a single joint optimization step on data batch."""
        N = X.shape[0]
        h, y_pred = self.forward(X)

        error = y_pred - y_target
        mse_loss = float(np.mean(error ** 2))

        dy = 2.0 * error / N
        dW2_mse = h.T @ dy
        db2 = np.sum(dy, axis=0, keepdims=True)

        dh = dy @ self.W2.T * (1.0 - h ** 2)
        dW1_mse = X.T @ dh
        db1 = np.sum(dh, axis=0, keepdims=True)

        all_w = self.get_all_weights()
        reg_penalty = self.gmm.penalty(all_w)

        reg_gw = self.gmm.grad_w(all_w, lambda_reg=lambda_reg)
        reg_gmu = self.gmm.grad_mu(all_w, lambda_reg=lambda_reg)
        reg_gbeta = self.gmm.grad_beta(all_w, lambda_reg=lambda_reg)
        reg_glogits = self.gmm.grad_logits(all_w, lambda_reg=lambda_reg)

        n_w1 = self.W1.size
        reg_dW1 = reg_gw[:n_w1].reshape(self.W1.shape)
        reg_dW2 = reg_gw[n_w1:].reshape(self.W2.shape)

        self.W1 -= lr_w * (dW1_mse + reg_dW1)
        self.W2 -= lr_w * (dW2_mse + reg_dW2)
        self.b1 -= lr_w * db1
        self.b2 -= lr_w * db2

        self.gmm.update(
            reg_gmu,
            reg_gbeta,
            reg_glogits,
            lr_mu=lr_gmm,
            lr_beta=lr_gmm,
            lr_logits=lr_gmm,
        )

        total_loss = mse_loss + lambda_reg * reg_penalty

        return {
            "mse_loss": mse_loss,
            "reg_penalty": reg_penalty,
            "total_loss": total_loss,
        }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1200,
        lr_w: float = 0.03,
        lr_gmm: float = 0.01,
        lambda_reg: float = 0.005,
        record_epochs: Optional[List[int]] = None,
    ) -> Dict[str, Union[List[float], Dict[int, Dict[str, np.ndarray]]]]:
        """
        Train the network and optionally snapshot weight and GMM distributions.
        """
        if record_epochs is None:
            record_epochs = [0, 100, 500, epochs - 1]

        history: Dict[str, List[float]] = {
            "mse_loss": [],
            "reg_penalty": [],
            "total_loss": [],
        }
        snapshots: Dict[int, Dict[str, np.ndarray]] = {}

        for ep in range(epochs):
            if ep in record_epochs:
                snapshots[ep] = {
                    "weights": self.get_all_weights().copy(),
                    "mu": self.gmm.mu.copy(),
                    "sigma2": self.gmm.sigma2.copy(),
                    "pi": self.gmm.pi.copy(),
                }

            metrics = self.train_step(
                X,
                y,
                lr_w=lr_w,
                lr_gmm=lr_gmm,
                lambda_reg=lambda_reg,
            )

            history["mse_loss"].append(metrics["mse_loss"])
            history["reg_penalty"].append(metrics["reg_penalty"])
            history["total_loss"].append(metrics["total_loss"])

        if (epochs - 1) not in snapshots:
            snapshots[epochs - 1] = {
                "weights": self.get_all_weights().copy(),
                "mu": self.gmm.mu.copy(),
                "sigma2": self.gmm.sigma2.copy(),
                "pi": self.gmm.pi.copy(),
            }

        return {"history": history, "snapshots": snapshots}


class HardWeightSharingMLP:
    """
    MLP with Hard Weight Sharing (strict equality constraints).
    Weights are mapped to groups, and gradients are pooled across identical weights.
    """

    def __init__(
        self,
        d_in: int = 1,
        d_hidden: int = 64,
        d_out: int = 1,
        n_groups: int = 8,
        seed: int = 42,
    ) -> None:
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.n_groups = n_groups

        rng = np.random.RandomState(seed)
        total_weights = d_in * d_hidden + d_hidden * d_out

        self.unique_weights = rng.randn(n_groups) * 0.7
        self.weight_indices = np.arange(total_weights) % n_groups
        rng.shuffle(self.weight_indices)

        self.b1 = np.zeros((1, d_hidden))
        self.b2 = np.zeros((1, d_out))

    @property
    def W1(self) -> np.ndarray:
        n_w1 = self.d_in * self.d_hidden
        w1_flat = self.unique_weights[self.weight_indices[:n_w1]]
        return w1_flat.reshape(self.d_in, self.d_hidden)

    @property
    def W2(self) -> np.ndarray:
        n_w1 = self.d_in * self.d_hidden
        w2_flat = self.unique_weights[self.weight_indices[n_w1:]]
        return w2_flat.reshape(self.d_hidden, self.d_out)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = np.tanh(X @ self.W1 + self.b1)
        y = h @ self.W2 + self.b2
        return h, y

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, y = self.forward(X)
        return y

    def fit(
        self,
        X: np.ndarray,
        y_target: np.ndarray,
        epochs: int = 1200,
        lr: float = 0.03,
    ) -> List[float]:
        """Train hard weight sharing MLP using pooled gradients."""
        N = X.shape[0]
        loss_hist = []

        for _ in range(epochs):
            h, y_pred = self.forward(X)
            err = y_pred - y_target
            loss = float(np.mean(err ** 2))
            loss_hist.append(loss)

            dy = 2.0 * err / N
            dW2 = h.T @ dy
            db2 = np.sum(dy, axis=0, keepdims=True)

            dh = dy @ self.W2.T * (1.0 - h ** 2)
            dW1 = X.T @ dh
            db1 = np.sum(dh, axis=0, keepdims=True)

            full_grad = np.concatenate([dW1.ravel(), dW2.ravel()])
            pooled_grad = np.zeros(self.n_groups)
            for g in range(self.n_groups):
                mask = (self.weight_indices == g)
                count = np.sum(mask)
                if count > 0:
                    pooled_grad[g] = np.sum(full_grad[mask]) / np.sqrt(count)

            # Gradient clipping for numerical stability
            pooled_grad = np.clip(pooled_grad, -5.0, 5.0)
            self.unique_weights -= lr * pooled_grad
            self.b1 -= lr * db1
            self.b2 -= lr * db2

        return loss_hist


# =============================================================================
# Figure Generation Functions
# =============================================================================

def generate_figure_soft_weight_sharing_prior(
    save_dir: Optional[str] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """
    Generate 3-panel publication figure illustrating the mechanics of Soft Weight Sharing:
    (a) GMM prior density p(w) with individual components
    (b) Regularization potential energy Omega(w) = -ln p(w)
    (c) Restoring force -dOmega / dw pulling weights toward cluster centers.
    """
    setup_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    mu = np.array([-1.5, 0.0, 1.5])
    beta = np.array([np.log(0.15), np.log(0.08), np.log(0.20)])
    logits = np.array([0.0, 0.5, 0.2])
    gmm = SoftWeightSharingGMM(n_components=3, mu=mu, beta=beta, logits=logits)

    w_grid = np.linspace(-3.0, 3.0, 500)

    _, log_marginal = gmm.compute_log_joint(w_grid)
    p_w = np.exp(log_marginal).ravel()

    s2 = gmm.sigma2
    pi = gmm.pi
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    for j in range(3):
        comp_gauss = (1.0 / np.sqrt(2 * np.pi * s2[j])) * np.exp(-0.5 * (w_grid - mu[j])**2 / s2[j])
        weighted_comp = pi[j] * comp_gauss
        ax1.plot(
            w_grid, weighted_comp, "--", color=colors[j], alpha=0.85,
            label=f"Comp {j+1} ($\\mu={mu[j]:.1f}, \\pi={pi[j]:.2f}$)"
        )

    ax1.plot(w_grid, p_w, "k-", lw=2.2, label=r"Total Prior $p(w) = \sum_j \pi_j \mathcal{N}$")
    ax1.set_title("(a) GMM Prior Density $p(w)$", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Weight value $w$", fontsize=11)
    ax1.set_ylabel("Probability density", fontsize=11)
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.set_xlim(-3, 3)

    omega = -log_marginal.ravel()
    omega_norm = omega - np.min(omega)

    ax2.plot(w_grid, omega_norm, color="#8c564b", lw=2.2, label=r"$\Omega(w) = -\ln p(w)$")
    for j in range(3):
        ax2.axvline(mu[j], color=colors[j], linestyle=":", alpha=0.7)
    ax2.set_title(r"(b) Regularization Potential $\Omega(w)$", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Weight value $w$", fontsize=11)
    ax2.set_ylabel(r"Potential energy $\Omega(w) - \min$", fontsize=11)
    ax2.legend(loc="upper center", fontsize=9.5)
    ax2.set_xlim(-3, 3)

    force = gmm.effective_force(w_grid)
    ax3.axhline(0, color="gray", linestyle="-", lw=0.8, alpha=0.7)
    ax3.plot(w_grid, force, color="#9467bd", lw=2.2, label=r"Restoring Force $-\frac{\partial \Omega}{\partial w}$")
    for j in range(3):
        ax3.axvline(mu[j], color=colors[j], linestyle=":", alpha=0.7, label=f"Center $\\mu_{j+1}$" if j == 0 else "")
        ax3.plot(mu[j], 0, "o", color=colors[j], markersize=6)

    ax3.set_title(r"(c) Weight Pulling Force $-\frac{\partial \Omega}{\partial w}$", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Weight value $w$", fontsize=11)
    ax3.set_ylabel("Force towards centers", fontsize=11)
    ax3.legend(loc="lower right", fontsize=9)
    ax3.set_xlim(-3, 3)

    plt.tight_layout()

    if save_dir is not None:
        save_plot(fig, os.path.join(save_dir, "fig_9_soft_weight_sharing_prior.png"))
    else:
        repo_root = Path(__file__).resolve().parent.parent
        save_plot(fig, repo_root / "9" / "result" / "fig_9_soft_weight_sharing_prior.png")
        save_plot(fig, repo_root / "result" / "fig_9_soft_weight_sharing_prior.png")

    return fig, (ax1, ax2, ax3)


def generate_figure_weight_clustering(
    save_dir: Optional[str] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """
    Generate 3-panel figure showing the dynamic evolution of weight clustering
    during training:
    (a) Initial weight distribution (Epoch 0)
    (b) Intermediate training state (Epoch 100)
    (c) Converged state with discrete weight clusters (Epoch 1200).
    """
    setup_style()

    np.random.seed(42)
    N = 100
    X = np.sort(np.random.uniform(-1, 1, (N, 1)), axis=0)
    y = np.sin(np.pi * X) + 0.3 * np.cos(2 * np.pi * X) + 0.02 * np.random.randn(N, 1)

    model = SoftWeightSharingMLP(d_in=1, d_hidden=64, d_out=1, n_components=3, seed=42)
    record_epochs = [0, 100, 1199]
    res = model.fit(
        X,
        y,
        epochs=1200,
        lr_w=0.03,
        lr_gmm=0.01,
        lambda_reg=0.005,
        record_epochs=record_epochs,
    )
    snapshots = res["snapshots"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    w_grid = np.linspace(-2.5, 2.5, 300)

    titles = [
        "(a) Epoch 0: Initial Weights",
        "(b) Epoch 100: Transition Phase",
        "(c) Epoch 1200: Soft Clustering",
    ]

    for idx, (ep, ax) in enumerate(zip([0, 100, 1199], axes)):
        data = snapshots[ep]
        w = data["weights"]
        mu = data["mu"]
        s2 = data["sigma2"]
        pi = data["pi"]

        ax.hist(
            w,
            bins=25,
            range=(-2.5, 2.5),
            density=True,
            alpha=0.45,
            color="#1f77b4",
            edgecolor="black",
            label=f"Weights ($N_w={len(w)}$)",
        )

        p_gmm = np.zeros_like(w_grid)
        for j in range(len(mu)):
            comp = (1.0 / np.sqrt(2 * np.pi * s2[j])) * np.exp(-0.5 * (w_grid - mu[j])**2 / s2[j])
            p_gmm += pi[j] * comp
            ax.plot(w_grid, pi[j] * comp, ":", color=["#2ca02c", "#d62728", "#ff7f0e"][j], lw=1.5)

        ax.plot(w_grid, p_gmm, "r-", lw=2.2, label=r"Learned Prior $p(w)$")
        ax.set_title(titles[idx], fontsize=12, fontweight="bold")
        ax.set_xlabel("Weight value $w$", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Density", fontsize=11)
        ax.legend(loc="upper right", fontsize=8.5)
        ax.set_xlim(-2.5, 2.5)

    plt.tight_layout()

    if save_dir is not None:
        save_plot(fig, os.path.join(save_dir, "fig_9_soft_weight_sharing_clustering.png"))
    else:
        repo_root = Path(__file__).resolve().parent.parent
        save_plot(fig, repo_root / "9" / "result" / "fig_9_soft_weight_sharing_clustering.png")
        save_plot(fig, repo_root / "result" / "fig_9_soft_weight_sharing_clustering.png")

    return fig, tuple(axes)


def generate_figure_hard_vs_soft_comparison(
    save_dir: Optional[str] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Generate 2-panel comparison of regression fits and weight distributions
    under No Regularization, L2 Weight Decay, and Soft Weight Sharing.
    """
    setup_style()

    np.random.seed(42)
    N = 80
    X = np.sort(np.random.uniform(-1, 1, (N, 1)), axis=0)
    y_true = np.sin(np.pi * X) + 0.3 * np.cos(2 * np.pi * X)
    y = y_true + 0.08 * np.random.randn(N, 1)

    X_test = np.linspace(-1.1, 1.1, 200).reshape(-1, 1)
    y_test_true = np.sin(np.pi * X_test) + 0.3 * np.cos(2 * np.pi * X_test)

    m_noreg = SoftWeightSharingMLP(d_in=1, d_hidden=64, d_out=1, n_components=3, seed=42)
    m_noreg.fit(X, y, epochs=1200, lr_w=0.03, lr_gmm=0.0, lambda_reg=0.0)

    m_l2 = SoftWeightSharingMLP(d_in=1, d_hidden=64, d_out=1, n_components=1, seed=42)
    m_l2.gmm.mu = np.array([0.0])
    m_l2.gmm.beta = np.array([np.log(1.0)])
    m_l2.fit(X, y, epochs=1200, lr_w=0.03, lr_gmm=0.0, lambda_reg=0.01)

    m_sws = SoftWeightSharingMLP(d_in=1, d_hidden=64, d_out=1, n_components=3, seed=42)
    m_sws.fit(X, y, epochs=1200, lr_w=0.03, lr_gmm=0.01, lambda_reg=0.005)

    m_hws = HardWeightSharingMLP(d_in=1, d_hidden=64, d_out=1, n_groups=8, seed=42)
    m_hws.fit(X, y, epochs=1200, lr=0.03)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))

    ax1.scatter(X, y, color="black", s=20, alpha=0.5, label="Training Data")
    ax1.plot(X_test, y_test_true, "k--", alpha=0.7, lw=1.5, label="Ground Truth")
    ax1.plot(X_test, m_noreg.predict(X_test), color="#d62728", lw=1.8, label="No Regularization")
    ax1.plot(X_test, m_l2.predict(X_test), color="#1f77b4", lw=1.8, label=r"$L_2$ Weight Decay")
    ax1.plot(X_test, m_sws.predict(X_test), color="#2ca02c", lw=2.2, label="Soft Weight Sharing")
    ax1.plot(X_test, m_hws.predict(X_test), color="#ff7f0e", lw=1.8, linestyle="-.", label="Hard Weight Sharing (8 groups)")

    ax1.set_title("(a) Regression Function Approximations", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Input $x$", fontsize=11)
    ax1.set_ylabel("Output $y$", fontsize=11)
    ax1.legend(loc="lower center", fontsize=8.5)
    ax1.set_xlim(-1.15, 1.15)

    w_noreg = m_noreg.get_all_weights()
    w_l2 = m_l2.get_all_weights()
    w_sws = m_sws.get_all_weights()

    ax2.hist(w_noreg, bins=25, alpha=0.35, color="#d62728", label="No Reg (wide spread)", density=True)
    ax2.hist(w_l2, bins=25, alpha=0.45, color="#1f77b4", label=r"$L_2$ (shrunk to 0)", density=True)
    ax2.hist(w_sws, bins=25, alpha=0.55, color="#2ca02c", label="Soft Weight Sharing (multimodal)", density=True)

    ax2.scatter(
        m_hws.unique_weights,
        np.zeros_like(m_hws.unique_weights) + 0.1,
        color="#ff7f0e",
        marker="x",
        s=80,
        lw=2.5,
        label="Hard Sharing (8 discrete values)",
        zorder=5,
    )

    ax2.set_title("(b) Learned Weight Distributions", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Weight value $w$", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()

    if save_dir is not None:
        save_plot(fig, os.path.join(save_dir, "fig_9_hard_vs_soft_comparison.png"))
    else:
        repo_root = Path(__file__).resolve().parent.parent
        save_plot(fig, repo_root / "9" / "result" / "fig_9_hard_vs_soft_comparison.png")
        save_plot(fig, repo_root / "result" / "fig_9_hard_vs_soft_comparison.png")

    return fig, (ax1, ax2)
