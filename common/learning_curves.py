"""
Learning Curves, Early Stopping, and Double Descent module for Deep Learning: Foundations and Concepts (Bishop & Bishop 2024).

Section 9.3:
- 9.3.1 Early stopping:
  * Behavior of training set vs validation set error (Figure 9.7).
  * Equivalence to weight decay on quadratic error surfaces (Figure 9.8).
  * Relationship tau * eta ~ 1 / lambda (Bishop 1995a, Exercise 9.6).
- 9.3.2 Double descent:
  * Model-wise double descent across width parameters (Figure 9.9).
  * Interpolation threshold, classical bias-variance regime, and modern overparameterized regime.
  * Epoch-wise double descent in deep networks (Figure 9.10).
  * Sample-wise non-monotonicity and the sample size paradox (Figure 9.11).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


class EarlyStoppingAnalysis:
    """Analysis of early stopping on iterative learning curves (Section 9.3.1, Figure 9.7).

    Simulates and evaluates the progression of training error and validation error
    over iteration steps tau, identifying the optimal early stopping checkpoint.
    """

    def __init__(self, n_steps: int = 51) -> None:
        self.n_steps = int(n_steps)
        self.steps = np.arange(self.n_steps)

        # Realistic learning curves matching textbook Figure 9.7 on sinusoidal data:
        # Training error monotonically decreases
        self.train_error = 0.17 + (0.28 - 0.17) / (1.0 + np.exp(0.25 * (self.steps - 18)))
        self.train_error[0] = 0.285

        # Validation error decreases to minimum around step 28, then slowly creeps upward
        self.val_error = (
            0.372
            + 0.058 * np.exp(-0.12 * self.steps)
            + 0.00008 * np.maximum(0, self.steps - 28) ** 1.3
        )

        self.best_step = int(np.argmin(self.val_error))
        self.best_val_error = float(self.val_error[self.best_step])
        self.final_val_error = float(self.val_error[-1])

    def get_curves(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (steps, train_error, val_error)."""
        return self.steps, self.train_error, self.val_error

    def error_reduction_from_early_stopping(self) -> float:
        """Absolute reduction in validation error achieved by early stopping vs training to end."""
        return self.final_val_error - self.best_val_error


class GradientFlowShrinkage:
    """Continuous gradient flow on quadratic error surface and equivalence to weight decay (Section 9.3.1, Figure 9.8).

    dw/dt = -nabla E(w) = -H (w - w_star) with w(0) = 0
    Exact solution: w(t) = w_star - exp(-H t) w_star
    In eigenvector coordinates: w_i(t) = (1 - exp(-eta_i * t)) w_star_i
    Weight decay solution: w_hat_i = (eta_i / (eta_i + lambda)) w_star_i
    """

    def __init__(
        self,
        H: np.ndarray,
        w_star: np.ndarray,
    ) -> None:
        self.H = np.asarray(H, dtype=float)
        self.w_star = np.asarray(w_star, dtype=float)
        self.dim = len(self.w_star)

        eigvals, eigvecs = np.linalg.eigh(self.H)
        self.eigenvalues = eigvals
        self.eigenvectors = eigvecs

    def trajectory(self, t_max: float = 12.0, n_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Compute continuous gradient flow trajectory w(t) for t in [0, t_max]."""
        t = np.linspace(0.0, float(t_max), n_points)
        # Compute w(t) along each eigenvector
        traj = np.zeros((n_points, self.dim))
        for i in range(self.dim):
            traj[:, i] = (1.0 - np.exp(-self.eigenvalues[i] * t)) * self.w_star[i]
        return t, traj

    def gradient_flow_point(self, t: float) -> np.ndarray:
        """Position w(t) at time t."""
        t_val = float(t)
        return (1.0 - np.exp(-self.eigenvalues * t_val)) * self.w_star

    def weight_decay_point(self, lambda_reg: float) -> np.ndarray:
        """Analytical weight decay optimum w_hat = (H + lambda * I)^{-1} H w_star."""
        lam = float(lambda_reg)
        return (self.eigenvalues / (self.eigenvalues + lam)) * self.w_star

    def effective_lambda(self, tau: int, eta: float) -> float:
        """Effective weight decay parameter lambda ~ 1 / (tau * eta) (Bishop 1995a)."""
        return 1.0 / (float(tau) * float(eta))


class DoubleDescentModel:
    """Model-wise double descent model (Section 9.3.2, Figure 9.9, Nakkiran et al. 2019).

    Tracks test and train error across model complexity (e.g. ResNet18 width parameter).
    """

    def __init__(self, max_width: int = 64, interp_width: int = 11) -> None:
        self.max_width = int(max_width)
        self.interp_width = int(interp_width)
        self.widths = np.arange(1, self.max_width + 1)

        # Train error: drops monotonically to 0 at interpolation threshold
        train_raw = 0.45 * np.maximum(0.0, 1.0 - self.widths / float(self.interp_width)) ** 1.8
        self.train_error = train_raw

        # Test error: classical U-shape followed by peak at interpolation threshold and modern descent
        test_curve = np.zeros(len(self.widths))
        for idx, k in enumerate(self.widths):
            if k < self.interp_width:
                # Underparameterized regime: drops then rises to peak
                # U-curve with minimum around k=6
                u_curve = 0.33 + 0.17 * np.exp(-0.6 * (k - 1)) + 0.08 * ((k - 6) / 5.0) ** 2
                test_curve[idx] = u_curve
            else:
                # Overparameterized regime: peaks at k=11, then decreases monotonically
                peak = 0.41
                decay = 0.26 + (peak - 0.26) * np.exp(-0.06 * (k - self.interp_width))
                test_curve[idx] = decay

        self.test_error = test_curve

    def get_curves(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (widths, train_error, test_error)."""
        return self.widths, self.train_error, self.test_error


def _save_figure(fig: plt.Figure, filename: str, custom_path: Optional[Union[str, Path]] = None) -> None:
    """Save figure to Chapter 9 result directory and repository root result directory."""
    if custom_path is not None:
        out_path = Path(custom_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        return

    repo_root = Path(__file__).resolve().parent.parent
    dir_ch9 = repo_root / "9" / "result"
    dir_root = repo_root / "result"
    dir_ch9.mkdir(parents=True, exist_ok=True)
    dir_root.mkdir(parents=True, exist_ok=True)

    fig.savefig(dir_ch9 / filename, dpi=300, bbox_inches="tight")
    fig.savefig(dir_root / filename, dpi=300, bbox_inches="tight")


def generate_figure_9_7(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Generate Figure 9.7: Behavior of training set error (left) and validation set error (right).

    Illustrates optimal early stopping at minimum of validation set error (dashed vertical line).
    """
    analyzer = EarlyStoppingAnalysis(n_steps=51)
    steps, train_err, val_err = analyzer.get_curves()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.8))

    # Left: Training set error
    ax1.plot(
        steps,
        train_err,
        "r-o",
        markersize=3.8,
        markerfacecolor="none",
        markeredgecolor="red",
        markeredgewidth=1.1,
        lw=1.2,
    )
    ax1.axvline(analyzer.best_step, color="black", linestyle="--", dashes=(6, 6), lw=1.1)
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0.15, 0.29)
    ax1.set_xticks([0, 10, 20, 30, 40, 50])
    ax1.set_yticks([0.15, 0.20, 0.25])
    ax1.tick_params(direction="in", length=5)

    # Right: Validation set error
    ax2.plot(
        steps,
        val_err,
        "b-o",
        markersize=3.8,
        markerfacecolor="none",
        markeredgecolor="blue",
        markeredgewidth=1.1,
        lw=1.2,
    )
    ax2.axvline(analyzer.best_step, color="black", linestyle="--", dashes=(6, 6), lw=1.1)
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0.35, 0.45)
    ax2.set_xticks([0, 10, 20, 30, 40, 50])
    ax2.set_yticks([0.35, 0.40, 0.45])
    ax2.tick_params(direction="in", length=5)

    plt.tight_layout()

    _save_figure(fig, "fig_9_7_early_stopping_curves.png", save_path)
    _save_figure(fig, "Figure_9_7.png", None if save_path is None else Path(save_path).parent / "Figure_9_7.png")
    return fig


def generate_figure_9_8(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Generate Figure 9.8: Schematic illustration of why early stopping gives similar results to weight decay."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    w_star = np.array([2.8, 1.8])
    eta1, eta2 = 0.4, 4.0

    w1 = np.linspace(-0.8, 4.8, 400)
    w2 = np.linspace(-0.8, 3.2, 400)
    W1, W2 = np.meshgrid(w1, w2)
    E = 0.5 * (eta1 * (W1 - w_star[0]) ** 2 + eta2 * (W2 - w_star[1]) ** 2)

    levels = [0.08, 0.32, 0.72, 1.28, 2.0, 2.88, 3.92]
    alphas = np.linspace(0.85, 0.3, len(levels))
    for lev, alpha in zip(levels, alphas):
        ax.contour(W1, W2, E, levels=[lev], colors=["#e74c3c"], linewidths=1.2, alpha=alpha)

    # Gradient flow trajectory
    t = np.linspace(0, 12, 500)
    traj_w1 = (1.0 - np.exp(-eta1 * t)) * w_star[0]
    traj_w2 = (1.0 - np.exp(-eta2 * t)) * w_star[1]
    ax.plot(traj_w1, traj_w2, color="#2980b9", lw=2.4, zorder=5)

    # Point w_hat along the curve (stopping early where w2 is converged but w1 is small)
    t_hat = 0.65
    w_hat = np.array([(1.0 - np.exp(-eta1 * t_hat)) * w_star[0], (1.0 - np.exp(-eta2 * t_hat)) * w_star[1]])
    ax.plot(w_hat[0], w_hat[1], "o", color="#2980b9", markersize=6.5, zorder=6)
    ax.text(w_hat[0] + 0.15, w_hat[1] - 0.22, r"$\widehat{\mathbf{w}}$", fontsize=13, ha="left", va="top")

    # Point w_star
    ax.plot(w_star[0], w_star[1], "ro", markersize=6.5, zorder=6)
    ax.text(w_star[0] + 0.18, w_star[1], r"$\mathbf{w}^\star$", fontsize=13, ha="left", va="center")

    # Arrow into w_star
    ax.annotate(
        "",
        xy=(w_star[0], w_star[1]),
        xytext=(traj_w1[-18], traj_w2[-18]),
        arrowprops=dict(arrowstyle="-|>", color="#2980b9", lw=2.2, mutation_scale=15),
        zorder=5,
    )

    # Axes
    ax.annotate(
        "",
        xy=(4.6, 0),
        xytext=(-0.5, 0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=14),
    )
    ax.annotate(
        "",
        xy=(0, 2.9),
        xytext=(0, -0.5),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=14),
    )
    ax.text(4.45, -0.22, r"$w_1$", fontsize=12, va="top")
    ax.text(-0.22, 2.75, r"$w_2$", fontsize=12, ha="right")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.7, 4.9)
    ax.set_ylim(-0.7, 3.1)
    plt.tight_layout()

    _save_figure(fig, "fig_9_8_early_stopping_quadratic.png", save_path)
    _save_figure(fig, "Figure_9_8.png", None if save_path is None else Path(save_path).parent / "Figure_9_8.png")
    return fig


def generate_figure_9_9(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Generate Figure 9.9: Plot of training set and test set errors for ResNet18 versus model complexity.

    Shows interpolation threshold, classical regime (bias-variance tradeoff), and modern overparameterized regime.
    """
    model = DoubleDescentModel(max_width=64, interp_width=11)
    widths, train_err, test_err = model.get_curves()

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Shaded critical regime
    ax.axvspan(6, 20, color="#f5cba7", alpha=0.6, zorder=1)

    # Curves
    ax.plot(widths, test_err, color="#2554c7", lw=2.2, label="Test", zorder=4)
    ax.plot(widths, train_err, color="#9bb7d4", lw=2.0, linestyle="--", label="Train", zorder=4)

    # Vertical interpolation threshold line
    ax.axvline(11, color="black", linestyle="--", dashes=(6, 6), lw=1.3, zorder=3)

    # Annotations
    ax.annotate(
        "Interpolation\nThreshold",
        xy=(11, 0.12),
        xytext=(26, 0.18),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.3, connectionstyle="arc3,rad=-0.2"),
        fontsize=11,
        ha="left",
    )

    ax.annotate(
        "Critical\nRegime",
        xy=(17, 0.42),
        xytext=(27, 0.43),
        arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.3, connectionstyle="arc3,rad=0.2"),
        fontsize=11,
        color="#d35400",
        ha="left",
    )

    # Text above the axes using transAxes
    ax.text(
        0.12,
        1.03,
        "Classical Regime:\nBias-Variance Tradeoff",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
    )
    ax.text(
        0.60,
        1.03,
        "Modern Regime:\nLarger Model is Better",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
    )

    ax.set_xlabel("ResNet18 width parameter", fontsize=12, labelpad=8)
    ax.set_ylabel("Test / Train Error", fontsize=12, labelpad=8)
    ax.set_xlim(1, 64)
    ax.set_ylim(0.0, 0.55)
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.tick_params(direction="in", length=4)
    ax.legend(frameon=False, fontsize=11, loc="upper right")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    _save_figure(fig, "fig_9_9_double_descent_resnet18.png", save_path)
    _save_figure(fig, "Figure_9_9.png", None if save_path is None else Path(save_path).parent / "Figure_9_9.png")
    return fig


def generate_figure_9_10(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Generate Figure 9.10: Plot of test set error versus number of epochs of gradient descent for ResNet18 models."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    epochs = np.logspace(0, 3.6, 300)  # 1 to ~4000 epochs
    log_ep = np.log10(epochs)

    # 1. Small model (width parameter = 3): monotonic decrease
    # decreases from 0.70 at epoch 1 to 0.25 at epoch 4000
    err_small = 0.24 + 0.46 * np.exp(-0.75 * log_ep)

    # 2. Intermediate model (width parameter = 12): decreases to epoch ~30 then increases to 0.34
    # U-shaped in log epochs
    err_inter = 0.22 + 0.31 * np.exp(-1.8 * log_ep) + 0.12 / (1.0 + np.exp(-2.5 * (log_ep - 1.6)))
    # Add minor noise in later epochs
    rng = np.random.RandomState(42)
    err_inter += rng.normal(0, 0.005, size=len(epochs)) * (log_ep > 1.8)

    # 3. Large model (width parameter = 64): double descent across epochs
    # Drops from 0.38 to 0.19 at epoch 10 (log_ep=1.0), rises with noisy peak to 0.33 around epoch 60-100 (log_ep=1.8), then drops to 0.18 at epoch 4000
    err_large_base = (
        0.18
        + 0.20 * np.exp(-2.5 * log_ep)
        + 0.14 * np.exp(-0.5 * ((log_ep - 1.85) / 0.38) ** 2)
    )
    # Add realistic SGD fluctuation noise around peak
    noise_large = rng.normal(0, 0.015, size=len(epochs)) * np.exp(-0.5 * ((log_ep - 1.9) / 0.5) ** 2)
    err_large = err_large_base + noise_large

    ax.plot(epochs, err_small, color="#f1c40f", lw=2.2, label="width parameter = 3")
    ax.plot(epochs, err_inter, color="#16a085", lw=2.2, label="width parameter = 12")
    ax.plot(epochs, err_large, color="#922b21", lw=2.0, label="width parameter = 64")

    # Text annotations along curves
    ax.text(10, 0.58, "Small Model", color="#d4ac0d", fontsize=11, rotation=-25)
    ax.text(4, 0.48, "Intermediate Model", color="#117a65", fontsize=11, rotation=-25)
    ax.text(2, 0.36, "Large Model", color="#78281f", fontsize=11, rotation=-25)

    ax.set_xscale("log")
    ax.set_xlabel("Epochs", fontsize=12, labelpad=8)
    ax.set_ylabel("Test Error", fontsize=12, labelpad=8)
    ax.set_xlim(1, 4000)
    ax.set_ylim(0.1, 0.7)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1k"])
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.tick_params(direction="in", length=4, which="both")
    ax.legend(frameon=False, fontsize=10.5, loc="upper right")

    plt.tight_layout()

    _save_figure(fig, "fig_9_10_epoch_double_descent.png", save_path)
    _save_figure(fig, "Figure_9_10.png", None if save_path is None else Path(save_path).parent / "Figure_9_10.png")
    return fig


def generate_figure_9_11(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Generate Figure 9.11: Plot of test set error for a large transformer model versus embedding dimension."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # 4k samples curve
    d_4k = np.array([10, 18, 25, 33, 42, 52, 63, 73, 85, 98, 110, 122, 134, 146, 158, 170, 182, 195, 205])
    loss_4k = np.array([9.6, 15.2, 22.1, 21.0, 18.1, 16.5, 14.8, 13.9, 13.1, 12.3, 11.8, 11.4, 11.0, 10.7, 10.5, 10.3, 10.1, 9.9, 9.8])

    # 18k samples curve
    d_18k = np.array([10, 25, 33, 42, 50, 58, 65, 73, 83, 94, 104, 115, 126, 138, 150, 163, 175, 188, 200])
    loss_18k = np.array([8.3, 9.2, 10.7, 12.6, 15.0, 17.2, 16.8, 15.6, 14.4, 13.5, 12.6, 12.0, 11.4, 10.8, 10.3, 9.9, 9.6, 9.3, 9.1])

    ax.plot(
        d_4k,
        loss_4k,
        "-o",
        color="#76d7c4",
        markersize=4.5,
        lw=1.5,
        label="4k Samples",
    )
    ax.plot(
        d_18k,
        loss_18k,
        "-o",
        color="#2980b9",
        markersize=4.5,
        lw=1.5,
        label="18k Samples",
    )

    # Red vertical arrows in the range where 18k samples gives higher loss than 4k samples
    arrow_d = [58, 65, 73, 83, 94]
    arrow_y4k = [15.4, 14.6, 13.9, 13.2, 12.5]
    arrow_y18k = [17.2, 16.8, 15.6, 14.4, 13.5]

    for x_d, y_bot, y_top in zip(arrow_d, arrow_y4k, arrow_y18k):
        ax.annotate(
            "",
            xy=(x_d, y_top),
            xytext=(x_d, y_bot),
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.3),
        )

    ax.text(
        78,
        18.5,
        "For models in this range\n4.5x more samples harm test loss",
        fontsize=9.5,
        ha="center",
    )

    ax.set_xlabel(r"Transformer Embedding Dimension ($d_{\mathrm{model}}$)", fontsize=12, labelpad=8)
    ax.set_ylabel("Cross-Entropy Test Loss", fontsize=12, labelpad=8)
    ax.set_xlim(0, 220)
    ax.set_ylim(7.5, 23.0)
    ax.set_xticks([0, 25, 50, 75, 100, 125, 150, 175, 200])
    ax.set_yticks([8, 10, 12, 14, 16, 18, 20, 22])
    ax.tick_params(direction="in", length=4)
    ax.legend(frameon=False, fontsize=11, loc="upper right")

    plt.tight_layout()

    _save_figure(fig, "fig_9_11_sample_wise_non_monotonicity.png", save_path)
    _save_figure(fig, "Figure_9_11.png", None if save_path is None else Path(save_path).parent / "Figure_9_11.png")
    return fig
