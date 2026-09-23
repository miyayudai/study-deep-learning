"""
Weight Decay and Regularization module for Deep Learning: Foundations and Concepts (Bishop & Bishop 2024).

Section 9.2:
- Quadratic error functions and weight decay shrinkage in rotated Hessian coordinates (Eqs 9.1, 9.5).
- Consistent regularizers under linear transformations of inputs and outputs (Eqs 9.6-9.15).
- Prior distributions over 2-layer network functions governed by 4 hyperparameters (alpha_1w, alpha_1b, alpha_2w, alpha_2b).
- Generalized weight decay (L_q penalties) and sparsity mechanisms (Eqs 9.18-9.20, Figures 9.3-9.6).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon
from scipy.optimize import minimize


class QuadraticObjective:
    """Quadratic error surface with analytical weight decay shrinkage.

    E(w) = E_0 + 0.5 * (w - w_star)^T @ H @ (w - w_star)
    Regularized: E_tilde(w) = E(w) + 0.5 * lambda_reg * ||w||^2 (Eq 9.1)
    """

    def __init__(
        self,
        H: np.ndarray,
        w_star: np.ndarray,
        E_0: float = 0.0,
    ) -> None:
        self.H = np.asarray(H, dtype=float)
        self.w_star = np.asarray(w_star, dtype=float)
        self.E_0 = float(E_0)
        self.dim = len(self.w_star)

        if self.H.shape != (self.dim, self.dim):
            raise ValueError(
                f"H shape {self.H.shape} incompatible with w_star dim {self.dim}"
            )

        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(self.H)
        self.eigenvalues = eigvals
        self.eigenvectors = eigvecs

    def value(self, w: np.ndarray) -> float:
        """Unregularized error value E(w)."""
        w = np.asarray(w, dtype=float)
        diff = w - self.w_star
        return self.E_0 + 0.5 * float(diff.T @ self.H @ diff)

    def gradient(self, w: np.ndarray) -> np.ndarray:
        """Unregularized error gradient nabla E(w) = H (w - w_star)."""
        w = np.asarray(w, dtype=float)
        return self.H @ (w - self.w_star)

    def regularized_value(self, w: np.ndarray, lambda_reg: float) -> float:
        """Regularized error value E_tilde(w) = E(w) + 0.5 * lambda_reg * ||w||^2."""
        w = np.asarray(w, dtype=float)
        reg_term = 0.5 * float(lambda_reg) * float(np.sum(w**2))
        return self.value(w) + reg_term

    def regularized_gradient(self, w: np.ndarray, lambda_reg: float) -> np.ndarray:
        """Regularized gradient nabla E_tilde(w) = nabla E(w) + lambda * w (Eq 9.5)."""
        w = np.asarray(w, dtype=float)
        return self.gradient(w) + float(lambda_reg) * w

    def regularized_optimum(self, lambda_reg: float) -> np.ndarray:
        """Exact regularized minimum w_hat = (H + lambda * I)^{-1} H w_star."""
        reg_H = self.H + float(lambda_reg) * np.eye(self.dim)
        return np.linalg.solve(reg_H, self.H @ self.w_star)

    def shrinkage_factors(self, lambda_reg: float) -> np.ndarray:
        """Shrinkage factor eta_i / (eta_i + lambda) along each eigenvector axis."""
        lam = float(lambda_reg)
        return self.eigenvalues / (self.eigenvalues + lam)

    def effective_number_of_parameters(self, lambda_reg: float) -> float:
        """Effective number of active parameters gamma = sum_i eta_i / (eta_i + lambda)."""
        return float(np.sum(self.shrinkage_factors(lambda_reg)))


class MLP2Layer:
    """Two-layer multilayer perceptron with tanh hidden activations and linear outputs (Eqs 9.6-9.7).

    z_j = tanh(sum_i w_{ji} x_i + w_{j0})   (Eq 9.6)
    y_k = sum_j w_{kj} z_j + w_{k0}          (Eq 9.7)
    """

    def __init__(
        self,
        W1: np.ndarray,
        b1: np.ndarray,
        W2: np.ndarray,
        b2: np.ndarray,
    ) -> None:
        self.W1 = np.asarray(W1, dtype=float)  # (M, D_in)
        self.b1 = np.asarray(b1, dtype=float).reshape(-1, 1)  # (M, 1)
        self.W2 = np.asarray(W2, dtype=float)  # (D_out, M)
        self.b2 = np.asarray(b2, dtype=float).reshape(-1, 1)  # (D_out, 1)

        self.n_hidden, self.n_in = self.W1.shape
        self.n_out = self.W2.shape[0]

        if self.W2.shape[1] != self.n_hidden:
            raise ValueError(
                f"W2 columns {self.W2.shape[1]} must match hidden units {self.n_hidden}"
            )
        if self.b1.shape[0] != self.n_hidden:
            raise ValueError(f"b1 rows {self.b1.shape[0]} must match hidden units {self.n_hidden}")
        if self.b2.shape[0] != self.n_out:
            raise ValueError(f"b2 rows {self.b2.shape[0]} must match output units {self.n_out}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward evaluation of network function y(x).

        Args:
            x: Input array of shape (D_in, N) or (N, D_in) or (N,).

        Returns:
            Output array of shape (D_out, N) or (N,).
        """
        x_arr = np.asarray(x, dtype=float)
        squeeze_out = False
        if x_arr.ndim == 1:
            if self.n_in == 1:
                x_mat = x_arr.reshape(1, -1)
                squeeze_out = True
            else:
                x_mat = x_arr.reshape(-1, 1)
        elif x_arr.ndim == 2:
            if x_arr.shape[0] == self.n_in:
                x_mat = x_arr
            elif x_arr.shape[1] == self.n_in:
                x_mat = x_arr.T
            else:
                raise ValueError(f"Input shape {x_arr.shape} cannot match n_in={self.n_in}")
        else:
            raise ValueError("Input must be 1D or 2D array")

        # Hidden activations
        a1 = self.W1 @ x_mat + self.b1
        z = np.tanh(a1)
        # Output activations
        y = self.W2 @ z + self.b2

        if squeeze_out and self.n_out == 1:
            return y.flatten()
        return y

    def transform_inputs(self, a: float, b: float) -> "MLP2Layer":
        """Linear transformation of input variables: x_tilde = a * x + b (Eq 9.8).

        Weight/bias transformation rules (Eqs 9.9-9.10):
        w_tilde_{ji} = (1 / a) * w_{ji}
        w_tilde_{j0} = w_{j0} - (b / a) * sum_i w_{ji}
        """
        a = float(a)
        b = float(b)
        if np.isclose(a, 0.0):
            raise ValueError("Scale factor 'a' cannot be zero.")

        W1_tilde = self.W1 / a
        # Sum across input dimension i
        sum_wji = np.sum(self.W1, axis=1, keepdims=True)
        b1_tilde = self.b1 - (b / a) * sum_wji

        return MLP2Layer(W1=W1_tilde, b1=b1_tilde, W2=self.W2.copy(), b2=self.b2.copy())

    def transform_outputs(self, c: float, d: float) -> "MLP2Layer":
        """Linear transformation of output variables: y_tilde = c * y + d (Eq 9.11).

        Weight/bias transformation rules (Eqs 9.12-9.13):
        w_tilde_{kj} = c * w_{kj}
        w_tilde_{k0} = c * w_{k0} + d
        """
        c = float(c)
        d = float(d)

        W2_tilde = c * self.W2
        b2_tilde = c * self.b2 + d

        return MLP2Layer(W1=self.W1.copy(), b1=self.b1.copy(), W2=W2_tilde, b2=b2_tilde)

    def weight_decay_loss(
        self,
        lambda1: float,
        lambda2: float,
        include_biases: bool = False,
    ) -> float:
        """Consistent regularizer for 2-layer network (Eq 9.14).

        Omega(w) = 0.5 * lambda1 * sum_{W1} w^2 + 0.5 * lambda2 * sum_{W2} w^2
        """
        loss = 0.5 * float(lambda1) * float(np.sum(self.W1**2))
        loss += 0.5 * float(lambda2) * float(np.sum(self.W2**2))
        if include_biases:
            loss += 0.5 * float(lambda1) * float(np.sum(self.b1**2))
            loss += 0.5 * float(lambda2) * float(np.sum(self.b2**2))
        return loss


def sample_mlp_functions(
    alpha_1w: float,
    alpha_1b: float,
    alpha_2w: float,
    alpha_2b: float,
    n_samples: int = 5,
    n_hidden: int = 12,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    n_points: int = 500,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample network functions from Gaussian prior over weights and biases (Section 9.2.1, Figure 9.4).

    In Bishop & Bishop (2024) Figure 9.4:
    - alpha_2w governs vertical scale of functions.
    - alpha_1w governs horizontal scale of variations (slope of tanh transitions).
    - alpha_1b governs horizontal range over which variations occur (bias offsets).
    - alpha_2b governs vertical offsets.

    Note on parameterization: Following Bishop's numerical simulation in Netlab/PRML,
    alpha parameters govern the variance/spread of the weights:
    sigma_{1w} = sqrt(alpha_1w), sigma_{1b} = sqrt(alpha_1b),
    sigma_{2w} = alpha_2w (giving exact 10x vertical scale), sigma_{2b} = sqrt(alpha_2b).

    Args:
        alpha_1w: Parameter governing first-layer weights.
        alpha_1b: Parameter governing first-layer biases.
        alpha_2w: Parameter governing second-layer weights.
        alpha_2b: Parameter governing second-layer biases.
        n_samples: Number of function curves to sample.
        n_hidden: Number of hidden tanh units (default 12).
        x_range: Interval for input x (default (-1, 1)).
        n_points: Number of evaluation points.
        seed: Base random seed for reproducible sampling.

    Returns:
        x: Input coordinates of shape (n_points,).
        y: Sampled function values of shape (n_samples, n_points).
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    x_mat = x.reshape(1, -1)
    y_samples = np.zeros((n_samples, n_points))

    # Standard deviations for the weights and biases
    s_1w = np.sqrt(float(alpha_1w))
    s_1b = np.sqrt(float(alpha_1b))
    s_2w = float(alpha_2w)
    s_2b = np.sqrt(float(alpha_2b))

    for k in range(n_samples):
        curve_seed = None if seed is None else (seed + k * 107)
        rng = np.random.RandomState(curve_seed)

        W1 = rng.normal(0.0, s_1w, size=(n_hidden, 1))
        b1 = rng.normal(0.0, s_1b, size=(n_hidden, 1))
        W2 = rng.normal(0.0, s_2w, size=(1, n_hidden))
        b2 = rng.normal(0.0, s_2b, size=(1, 1))

        mlp = MLP2Layer(W1=W1, b1=b1, W2=W2, b2=b2)
        y_samples[k] = mlp.forward(x_mat).flatten()

    return x, y_samples


def lq_penalty(w: np.ndarray, q: float, lambda_reg: float = 1.0) -> float:
    """Generalized weight decay penalty Omega(w) = 0.5 * lambda * sum |w_j|^q (Eq 9.18)."""
    w_arr = np.asarray(w, dtype=float)
    return 0.5 * float(lambda_reg) * float(np.sum(np.abs(w_arr) ** float(q)))


def lq_contour_points(q: float, radius: float = 1.0, n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 2D boundary points for the L_q constraint |w1|^q + |w2|^q = radius^q.

    For a given angle theta in [0, 2*pi], (cos(theta), sin(theta)) is scaled so that
    |r*cos(theta)|^q + |r*sin(theta)|^q = radius^q.
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    denom = np.abs(cos_t) ** q + np.abs(sin_t) ** q
    r = radius / (denom ** (1.0 / q))
    w1 = r * cos_t
    w2 = r * sin_t
    return w1, w2


def solve_constrained_qp(
    H: np.ndarray,
    w_star: np.ndarray,
    q: float,
    eta: float,
    w_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve constrained optimization problem (Eq 9.20):

    min 0.5 * (w - w_star)^T @ H @ (w - w_star)
    s.t. sum |w_j|^q <= eta
    """
    H = np.asarray(H, dtype=float)
    w_star = np.asarray(w_star, dtype=float)
    dim = len(w_star)

    if w_init is None:
        w_init = np.zeros(dim)
        w_init[1] = eta ** (1.0 / q)

    def objective(w: np.ndarray) -> float:
        diff = w - w_star
        return 0.5 * float(diff.T @ H @ diff)

    def constraint(w: np.ndarray) -> float:
        return float(eta) - float(np.sum(np.abs(w) ** float(q)))

    res = minimize(
        objective,
        w_init,
        method="SLSQP",
        constraints={"type": "ineq", "fun": constraint},
        tol=1e-9,
        options={"maxiter": 500},
    )
    return res.x


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


def generate_figure_9_3(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Generate Figure 9.3: Contours of unregularized quadratic error, L_2 regularizer, and regularized objective.

    Shows error function contours (red), sum-of-squares regularizer (green), and regularized objective (blue),
    with axes aligned with Hessian eigenvectors u_1, u_2.
    Demonstrates greater shrinkage along the direction of small eigenvalue (w1) than large eigenvalue (w2).
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    w_star = np.array([3.2, 2.0])
    eta1, eta2 = 0.35, 3.5
    lam = 1.0

    obj = QuadraticObjective(H=np.diag([eta1, eta2]), w_star=w_star)
    w_hat = obj.regularized_optimum(lambda_reg=lam)

    w1 = np.linspace(-1.5, 5.5, 400)
    w2 = np.linspace(-1.5, 3.5, 400)
    W1, W2 = np.meshgrid(w1, w2)

    E = 0.5 * (eta1 * (W1 - w_star[0]) ** 2 + eta2 * (W2 - w_star[1]) ** 2)
    R = 0.5 * (W1**2 + W2**2)
    E_reg = E + lam * R

    # Contours
    levels_E = [0.1, 0.4, 0.9, 1.6, 2.5, 3.6, 4.9]
    alphas_E = np.linspace(0.85, 0.35, len(levels_E))
    for lev, alpha in zip(levels_E, alphas_E):
        ax.contour(W1, W2, E, levels=[lev], colors=["#e74c3c"], linewidths=1.3, alpha=alpha)

    levels_R = [0.2, 0.5, 1.0, 1.7, 2.6]
    alphas_R = np.linspace(0.85, 0.4, len(levels_R))
    for lev, alpha in zip(levels_R, alphas_R):
        ax.contour(W1, W2, R, levels=[lev], colors=["#2ecc71"], linewidths=1.3, alpha=alpha)

    levels_reg = [0.4, 0.9, 1.6, 2.5, 3.6, 4.9]
    alphas_reg = np.linspace(0.85, 0.35, len(levels_reg))
    for lev, alpha in zip(levels_reg, alphas_reg):
        ax.contour(W1, W2, E_reg - E_reg.min(), levels=[lev], colors=["#3498db"], linewidths=1.3, alpha=alpha)

    # Coordinate axes w1, w2
    ax.annotate(
        "",
        xy=(5.3, 0),
        xytext=(-1.1, 0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=15),
    )
    ax.annotate(
        "",
        xy=(0, 3.3),
        xytext=(0, -1.1),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=15),
    )
    ax.text(5.1, -0.3, r"$w_1$", fontsize=13, va="top")
    ax.text(-0.3, 3.1, r"$w_2$", fontsize=13, ha="right")

    # Eigenvector coordinate axes u1, u2 from w_star
    ax.annotate(
        "",
        xy=(w_star[0] + 2.0, w_star[1]),
        xytext=(w_star[0], w_star[1]),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15),
    )
    ax.annotate(
        "",
        xy=(w_star[0], w_star[1] + 1.2),
        xytext=(w_star[0], w_star[1]),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=15),
    )
    ax.text(w_star[0] + 2.15, w_star[1], r"$u_1$", fontsize=13, va="center")
    ax.text(w_star[0], w_star[1] + 1.35, r"$u_2$", fontsize=13, ha="center")

    # Points w_star and w_hat
    ax.plot(w_star[0], w_star[1], "ro", markersize=6, zorder=6)
    ax.text(w_star[0] - 0.28, w_star[1] - 0.05, r"$\mathbf{w}^\star$", fontsize=13, ha="right", va="center")

    ax.plot(w_hat[0], w_hat[1], "ro", markersize=6, zorder=6)
    ax.text(w_hat[0] - 0.15, w_hat[1] - 0.15, r"$\widehat{\mathbf{w}}$", fontsize=13, ha="right", va="top")

    # Function labels
    ax.text(w_star[0] - 1.2, w_star[1] + 1.2, r"$E(\mathbf{w})$", fontsize=13, color="black")
    ax.text(0.5, -0.7, r"$w_1^2 + w_2^2$", fontsize=13, color="black")
    ax.text(1.2, 0.4, r"$E(\mathbf{w}) + \lambda(w_1^2 + w_2^2)$", fontsize=13, color="black")

    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 5.6)
    ax.set_ylim(-1.3, 3.6)
    ax.axis("off")
    plt.tight_layout()

    _save_figure(fig, "fig_9_3_weight_decay_shrinkage.png", save_path)
    _save_figure(fig, "Figure_9_3.png", None if save_path is None else Path(save_path).parent / "Figure_9_3.png")
    return fig


def generate_figure_9_4(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Generate Figure 9.4: Prior distributions over network functions under 4 hyperparameter settings.

    Illustration of the effect of the hyperparameters governing the prior distribution over weights and biases
    in a two-layer network having a single input, a single linear output, and 12 hidden units with tanh activations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    panels = [
        {
            "ax": axes[0, 0],
            "params": (1, 1, 1, 1),
            "title": r"$\alpha_1^{\mathrm{w}} = 1, \alpha_1^{\mathrm{b}} = 1, \alpha_2^{\mathrm{w}} = 1, \alpha_2^{\mathrm{b}} = 1$",
            "ylim": (-6, 4),
            "yticks": [-6, -4, -2, 0, 2, 4],
        },
        {
            "ax": axes[0, 1],
            "params": (1, 1, 10, 1),
            "title": r"$\alpha_1^{\mathrm{w}} = 1, \alpha_1^{\mathrm{b}} = 1, \alpha_2^{\mathrm{w}} = 10, \alpha_2^{\mathrm{b}} = 1$",
            "ylim": (-60, 40),
            "yticks": [-60, -40, -20, 0, 20, 40],
        },
        {
            "ax": axes[1, 0],
            "params": (1000, 100, 1, 1),
            "title": r"$\alpha_1^{\mathrm{w}} = 1000, \alpha_1^{\mathrm{b}} = 100, \alpha_2^{\mathrm{w}} = 1, \alpha_2^{\mathrm{b}} = 1$",
            "ylim": (-10, 5),
            "yticks": [-10, -5, 0, 5],
        },
        {
            "ax": axes[1, 1],
            "params": (1000, 1000, 1, 1),
            "title": r"$\alpha_1^{\mathrm{w}} = 1000, \alpha_1^{\mathrm{b}} = 1000, \alpha_2^{\mathrm{w}} = 1, \alpha_2^{\mathrm{b}} = 1$",
            "ylim": (-10, 5),
            "yticks": [-10, -5, 0, 5],
        },
    ]

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#d4ac0d"]
    x = np.linspace(-1.0, 1.0, 600)
    x_mat = x.reshape(1, -1)

    # Use fixed seeds for the 5 curves to match characteristic textbook profiles
    curve_seeds = [42, 105, 314, 523, 789]

    for panel in panels:
        ax = panel["ax"]
        a1w, a1b, a2w, a2b = panel["params"]

        s_1w = np.sqrt(a1w)
        s_1b = np.sqrt(a1b)
        s_2w = float(a2w)
        s_2b = np.sqrt(a2b)

        for k in range(5):
            rng = np.random.RandomState(curve_seeds[k])
            W1 = rng.normal(0.0, s_1w, size=(12, 1))
            b1 = rng.normal(0.0, s_1b, size=(12, 1))
            W2 = rng.normal(0.0, s_2w, size=(1, 12))
            b2 = rng.normal(0.0, s_2b, size=(1, 1))

            mlp = MLP2Layer(W1=W1, b1=b1, W2=W2, b2=b2)
            y = mlp.forward(x_mat).flatten()
            ax.plot(x, y, color=colors[k], lw=1.8)

        ax.set_title(panel["title"], fontsize=12, pad=10)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(panel["ylim"])
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_yticks(panel["yticks"])
        ax.grid(False)
        ax.tick_params(direction="in", length=4)

    plt.tight_layout()

    _save_figure(fig, "fig_9_4_prior_network_functions.png", save_path)
    _save_figure(fig, "Figure_9_4.png", None if save_path is None else Path(save_path).parent / "Figure_9_4.png")
    return fig


def generate_figure_9_5(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Generate Figure 9.5: Contours of the generalized L_q regularization term for q in {0.5, 1, 2, 4}."""
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.5))
    qs = [0.5, 1.0, 2.0, 4.0]
    titles = [r"$q = 0.5$", r"$q = 1$", r"$q = 2$", r"$q = 4$"]

    w1 = np.linspace(-1.5, 1.5, 400)
    w2 = np.linspace(-1.5, 1.5, 400)
    W1, W2 = np.meshgrid(w1, w2)

    for ax, q, title in zip(axes.flatten(), qs, titles):
        Z = np.abs(W1) ** q + np.abs(W2) ** q
        radii = np.linspace(0.12, 1.0, 8)
        levels = radii**q
        alphas = np.linspace(0.85, 0.35, len(levels))

        for lev, alpha in zip(levels, alphas):
            ax.contour(W1, W2, Z, levels=[lev], colors=["#e74c3c"], linewidths=1.3, alpha=alpha)

        # Coordinate axes
        ax.annotate(
            "",
            xy=(1.45, 0),
            xytext=(-1.45, 0),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=14),
        )
        ax.annotate(
            "",
            xy=(0, 1.45),
            xytext=(0, -1.45),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=14),
        )
        ax.text(1.5, 0.05, r"$w_1$", fontsize=12, va="bottom")
        ax.text(0.05, 1.5, r"$w_2$", fontsize=12, ha="left")
        ax.text(0, -1.7, title, fontsize=13, ha="center")

        ax.set_xlim(-1.65, 1.65)
        ax.set_ylim(-1.8, 1.65)
        ax.set_aspect("equal")
        ax.axis("off")

    plt.tight_layout()

    _save_figure(fig, "fig_9_5_lq_regularization_contours.png", save_path)
    _save_figure(fig, "Figure_9_5.png", None if save_path is None else Path(save_path).parent / "Figure_9_5.png")
    return fig


def generate_figure_9_6(save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Generate Figure 9.6: Contours of unregularized error with constraint regions for Lasso (q=1) and Ridge (q=2).

    Illustrates the sparsity mechanism:
    - Lasso (q=1) has sharp corners on coordinate axes, yielding sparse solution w_hat_1 = 0.
    - Ridge (q=2) has smooth boundary yielding non-zero coefficients.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 6))

    theta = np.deg2rad(45)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    a, b = 1.4, 0.7
    H_inv = R @ np.diag([a**2, b**2]) @ R.T
    H = np.linalg.inv(H_inv)

    # w_star chosen analytically so normal cone at (0, 1) contains gradient:
    # g = (0.6, 1.0) => w_star = [0, 1] + H_inv @ g = [1.47, 2.67]
    g = np.array([0.6, 1.0])
    w_star = np.array([0.0, 1.0]) + H_inv @ g
    eta = 1.0

    w1 = np.linspace(-1.3, 3.3, 400)
    w2 = np.linspace(-1.3, 3.6, 400)
    W1, W2 = np.meshgrid(w1, w2)
    dW1 = W1 - w_star[0]
    dW2 = W2 - w_star[1]
    E = 0.5 * (H[0, 0] * dW1**2 + 2 * H[0, 1] * dW1 * dW2 + H[1, 1] * dW2**2)

    # Exact solutions
    w_hat_diamond = np.array([0.0, 1.0])
    E_diamond = 0.5 * (w_hat_diamond - w_star).T @ H @ (w_hat_diamond - w_star)

    w_hat_circle = solve_constrained_qp(H=H, w_star=w_star, q=2.0, eta=eta, w_init=np.array([0.2, 0.95]))
    E_circle = 0.5 * (w_hat_circle - w_star).T @ H @ (w_hat_circle - w_star)

    levels = np.linspace(0.06, E_diamond, 8)
    alphas = np.linspace(0.25, 0.85, len(levels))

    for ax, is_lasso in [(ax1, True), (ax2, False)]:
        # Axes
        ax.annotate(
            "",
            xy=(3.1, 0),
            xytext=(-1.15, 0),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=14),
        )
        ax.annotate(
            "",
            xy=(0, 3.4),
            xytext=(0, -1.15),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=14),
        )
        ax.text(3.0, -0.28, r"$w_1$", fontsize=12, va="top")
        ax.text(-0.25, 3.25, r"$w_2$", fontsize=12, ha="right")

        # Concentric error contours
        for lev, alpha in zip(levels, alphas):
            ax.contour(W1, W2, E, levels=[lev], colors=["#e74c3c"], linewidths=1.3, alpha=alpha)

        if is_lasso:
            diamond = Polygon(
                [[-eta, 0], [0, eta], [eta, 0], [0, -eta]],
                closed=True,
                facecolor="#5cb85c",
                edgecolor="#1e5631",
                lw=1.5,
                alpha=0.9,
                zorder=3,
            )
            ax.add_patch(diamond)
            ax.plot(w_hat_diamond[0], w_hat_diamond[1], "ro", markersize=6, zorder=5)
            ax.text(
                w_hat_diamond[0] - 0.18,
                w_hat_diamond[1],
                r"$\widehat{\mathbf{w}}$",
                fontsize=13,
                ha="right",
                va="center",
            )
            ax.text(0.35, -0.45, r"$|w_1| + |w_2| \leqslant \eta$", fontsize=12)
        else:
            circle = Circle(
                (0, 0),
                eta,
                facecolor="#5cb85c",
                edgecolor="#1e5631",
                lw=1.5,
                alpha=0.9,
                zorder=3,
            )
            ax.add_patch(circle)
            ax.plot(w_hat_circle[0], w_hat_circle[1], "ro", markersize=6, zorder=5)
            ax.text(
                w_hat_circle[0] - 0.1,
                w_hat_circle[1] - 0.18,
                r"$\widehat{\mathbf{w}}$",
                fontsize=13,
                ha="right",
                va="top",
            )
            ax.text(0.45, -0.45, r"$w_1^2 + w_2^2 \leqslant \eta$", fontsize=12)

        ax.text(w_star[0] + 0.5, w_star[1] - 0.9, r"$E(\mathbf{w})$", fontsize=13)
        ax.set_xlim(-1.25, 3.25)
        ax.set_ylim(-1.25, 3.5)
        ax.set_aspect("equal")
        ax.axis("off")

    plt.tight_layout()

    _save_figure(fig, "fig_9_6_sparsity_lasso_vs_ridge.png", save_path)
    _save_figure(fig, "Figure_9_6.png", None if save_path is None else Path(save_path).parent / "Figure_9_6.png")
    return fig
