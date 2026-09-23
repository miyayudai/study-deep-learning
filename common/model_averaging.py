"""
common/model_averaging.py
=========================
Section 9.6: Model Averaging & Section 9.6.1: Dropout
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

References:
- Breiman (1996), Bagging Predictors
- Freund & Schapire (1996), Experiments with a New Boosting Algorithm
- Srivastava et al. (2014), Dropout: A Simple Way to Prevent Neural Networks from Overfitting
- Gal & Ghahramani (2016), Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
- Lakshminarayanan et al. (2017), Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles

This module provides:
1. EnsembleCommittee: Mathematical committee averaging, expected squared error decomposition,
   Jensen's inequality bound, and optimal weighted combination (Eqs 9.42 - 9.50, 9.64 - 9.67).
2. DropoutMLP: Deep MLP with Inverted and Standard Dropout, exact analytical gradients,
   minibatch SGD training, and Monte Carlo Dropout for epistemic uncertainty estimation (Eq 9.51).
3. LinearRegressionDropout: Mathematical and empirical equivalence between dropout on linear regression
   and input-variance-weighted L2 regularization (Exercise 9.18).
4. High-resolution figure generators for Figure 9.17 and theoretical/pedagogical figures.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .plot_utils import setup_style, save_plot


def _save_figure(fig: plt.Figure, filename_base: str, save_dir: Optional[str] = None) -> None:
    """Save figure to Chapter 9 result directory and root result directory."""
    if save_dir is not None:
        save_plot(fig, os.path.join(save_dir, f"{filename_base}.png"))
        return

    repo_root = Path(__file__).resolve().parent.parent
    dir_ch9 = repo_root / "9" / "result"
    dir_root = repo_root / "result"
    dir_ch9.mkdir(parents=True, exist_ok=True)
    dir_root.mkdir(parents=True, exist_ok=True)

    save_plot(fig, str(dir_ch9 / f"{filename_base}.png"))
    save_plot(fig, str(dir_root / f"{filename_base}.png"))


class EnsembleCommittee:
    """
    Mathematical committee model averaging (Section 9.6, Eqs 9.42 - 9.50).
    Given M trained models, computes committee predictions and theoretical error reductions.
    """

    def __init__(
        self,
        models: List[Any],
        weights: Optional[np.ndarray] = None,
    ) -> None:
        self.models = models
        self.M = len(models)
        if weights is None:
            self.weights = np.ones(self.M) / self.M
        else:
            weights = np.asarray(weights, dtype=float)
            assert len(weights) == self.M
            assert np.all(weights >= 0), "Weights must be non-negative (Exercise 9.17)"
            self.weights = weights / np.sum(weights)

    def predict_individual(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate individual predictions from each member model.
        Returns:
            preds: (M, N) array where preds[m, n] is prediction of model m on sample n.
        """
        preds = []
        for model in self.models:
            if hasattr(model, "predict"):
                p = model.predict(X)
            elif callable(model):
                p = model(X)
            else:
                raise ValueError("Model must have a predict method or be callable")
            preds.append(np.asarray(p).ravel())
        return np.array(preds)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Committee weighted average prediction (Eq 9.43 / Eq 9.50):
        y_COM(x) = sum_{m=1}^M alpha_m y_m(x)
        """
        ind_preds = self.predict_individual(X)  # (M, N)
        return np.sum(self.weights[:, np.newaxis] * ind_preds, axis=0)

    def evaluate_errors(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Evaluate empirical and theoretical error decomposition (Eqs 9.44 - 9.50).
        Returns:
            E_m: list of individual mean squared errors (Eq 9.45)
            E_AV: average individual error (Eq 9.46)
            E_COM: committee squared error (Eq 9.47)
            error_matrix: (M, N) residual matrix eps_m(x_n) = y_m(x_n) - y_true(x_n)
            cov_matrix: (M, M) sample covariance of residuals
            corr_matrix: (M, M) correlation matrix of residuals
            mean_corr: average off-diagonal error correlation r_bar
            theoretical_ratio: (1 + (M-1)*r_bar) / M
            empirical_ratio: E_COM / E_AV
        """
        y_true = np.asarray(y_true).ravel()
        ind_preds = self.predict_individual(X)  # (M, N)
        N = len(y_true)

        # Residuals eps_m(x) = y_m(x) - y(x) (Eq 9.44)
        errors = ind_preds - y_true[np.newaxis, :]  # (M, N)

        # Individual MSEs E_m (Eq 9.45)
        E_m = np.mean(errors ** 2, axis=1)

        # Average individual error E_AV (Eq 9.46)
        E_AV = np.mean(E_m)

        # Committee prediction and error E_COM (Eq 9.47)
        y_com = self.predict(X)
        E_COM = float(np.mean((y_com - y_true) ** 2))

        # Covariance and correlation of model errors
        cov_matrix = (errors @ errors.T) / N
        stds = np.sqrt(np.diag(cov_matrix))
        outer_stds = np.outer(stds, stds)
        outer_stds[outer_stds == 0] = 1.0
        corr_matrix = cov_matrix / outer_stds

        # Mean off-diagonal correlation
        if self.M > 1:
            mask = ~np.eye(self.M, dtype=bool)
            r_bar = float(np.mean(corr_matrix[mask]))
        else:
            r_bar = 1.0

        theoretical_ratio = (1.0 + (self.M - 1) * r_bar) / self.M

        return {
            "E_m": E_m,
            "E_AV": float(E_AV),
            "E_COM": float(E_COM),
            "error_matrix": errors,
            "cov_matrix": cov_matrix,
            "corr_matrix": corr_matrix,
            "mean_corr": r_bar,
            "theoretical_ratio": float(theoretical_ratio),
            "empirical_ratio": float(E_COM / E_AV) if E_AV > 0 else 1.0,
        }

    @staticmethod
    def compute_optimal_weights(error_matrix: np.ndarray) -> np.ndarray:
        """
        Compute optimal non-negative committee weights minimizing E_COM (Exercise 9.17).
        Solves: min alpha^T C alpha  s.t. sum(alpha) = 1, alpha >= 0.
        Uses quadratic programming via projected gradient or pseudo-inverse.
        """
        M, N = error_matrix.shape
        C = (error_matrix @ error_matrix.T) / N
        C += np.eye(M) * 1e-7  # Numerical stabilization

        # Projected gradient descent on simplex
        alpha = np.ones(M) / M
        lr = 0.5 / (np.linalg.norm(C, ord=2) + 1e-6)

        for _ in range(500):
            grad = 2.0 * (C @ alpha)
            alpha_new = alpha - lr * grad
            # Project onto probability simplex (Duchi et al., 2008)
            u = np.sort(alpha_new)[::-1]
            cssv = np.cumsum(u)
            rho = np.nonzero(u * np.arange(1, M + 1) > (cssv - 1))[0][-1]
            theta = (cssv[rho] - 1.0) / (rho + 1.0)
            alpha = np.maximum(alpha_new - theta, 0.0)

        return alpha / np.sum(alpha)


class DropoutMLP:
    """
    Multilayer Perceptron supporting Inverted and Standard Dropout (Srivastava et al., 2014)
    and Monte Carlo Dropout for epistemic uncertainty estimation (Gal & Ghahramani, 2016).
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "relu",
        seed: int = 42,
    ) -> None:
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.activation_name = activation.lower()
        self.rng = np.random.RandomState(seed)

        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []

        # He / Xavier initialization
        for l in range(self.n_layers):
            d_in = layer_sizes[l]
            d_out = layer_sizes[l + 1]
            if self.activation_name == "relu":
                scale = np.sqrt(2.0 / d_in)
            else:
                scale = np.sqrt(1.0 / d_in)
            self.W.append(self.rng.randn(d_in, d_out) * scale)
            self.b.append(np.zeros(d_out))

    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation_name == "relu":
            return np.maximum(0.0, z)
        elif self.activation_name == "tanh":
            return np.tanh(z)
        elif self.activation_name == "linear":
            return z
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

    def _activate_grad(self, z: np.ndarray) -> np.ndarray:
        if self.activation_name == "relu":
            return (z > 0).astype(float)
        elif self.activation_name == "tanh":
            return 1.0 - np.tanh(z) ** 2
        elif self.activation_name == "linear":
            return np.ones_like(z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

    def forward(
        self,
        X: np.ndarray,
        training: bool = False,
        p_hidden: float = 0.5,
        p_input: float = 0.0,
        mode: str = "inverted",
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass with dropout.
        Args:
            X: (N, d_in) input batch
            training: whether in training mode (applies Bernoulli dropout mask)
            p_hidden: probability of dropping hidden nodes (retention prob rho_h = 1 - p_hidden)
            p_input: probability of dropping input nodes (retention prob rho_in = 1 - p_input)
            mode: "inverted" (scales by 1/rho during training) or "standard" (scales by rho at inference)
        Returns:
            y_pred: (N, d_out) output
            activations: list of activations at each layer [a_0, a_1, ..., a_L]
            pre_activations: list of pre-activations [z_1, ..., z_L]
            masks: list of binary dropout masks applied [m_0, m_1, ..., m_{L-1}]
        """
        X = np.atleast_2d(X)
        N = X.shape[0]

        rho_in = 1.0 - p_input
        rho_h = 1.0 - p_hidden

        masks: List[np.ndarray] = []
        activations: List[np.ndarray] = []
        pre_activations: List[np.ndarray] = []

        # Layer 0 (Input)
        a = X.copy()
        if training and p_input > 0.0:
            m0 = (self.rng.rand(*a.shape) < rho_in).astype(float)
            if mode == "inverted":
                a = (a * m0) / rho_in
            else:
                a = a * m0
            masks.append(m0)
        else:
            if not training and mode == "standard" and p_input > 0.0:
                a = a * rho_in
            masks.append(np.ones_like(a))
        activations.append(a)

        # Hidden and output layers
        for l in range(self.n_layers):
            z = a @ self.W[l] + self.b[l]
            pre_activations.append(z)

            is_output = (l == self.n_layers - 1)
            if is_output:
                # Output layer has no activation (linear regression) and NO dropout
                a = z
            else:
                h = self._activate(z)
                if training and p_hidden > 0.0:
                    ml = (self.rng.rand(*h.shape) < rho_h).astype(float)
                    if mode == "inverted":
                        a = (h * ml) / rho_h
                    else:
                        a = h * ml
                    masks.append(ml)
                else:
                    if not training and mode == "standard" and p_hidden > 0.0:
                        a = h * rho_h
                    else:
                        a = h
                    masks.append(np.ones_like(h))
            activations.append(a)

        return a, activations, pre_activations, masks

    def backward(
        self,
        y_true: np.ndarray,
        activations: List[np.ndarray],
        pre_activations: List[np.ndarray],
        masks: List[np.ndarray],
        mode: str = "inverted",
        p_hidden: float = 0.5,
        p_input: float = 0.0,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backpropagation through the pruned network with dropout masks.
        Returns:
            grad_W: list of weight gradients [dE/dW_0, ..., dE/dW_{L-1}]
            grad_b: list of bias gradients [dE/db_0, ..., dE/db_{L-1}]
        """
        y_true = np.atleast_2d(y_true)
        N = y_true.shape[0]
        rho_h = 1.0 - p_hidden

        # Loss: 0.5 * MSE
        y_pred = activations[-1]
        d_out = y_true.shape[1]
        delta = (y_pred - y_true) / (N * d_out)  # (N, d_out)

        grad_W: List[np.ndarray] = []
        grad_b: List[np.ndarray] = []

        for l in reversed(range(self.n_layers)):
            a_prev = activations[l]
            gW = a_prev.T @ delta
            gb = np.sum(delta, axis=0)

            grad_W.append(gW)
            grad_b.append(gb)

            if l > 0:
                delta_prev = delta @ self.W[l].T
                # Mask from previous hidden layer
                m_prev = masks[l]
                if mode == "inverted" and p_hidden > 0.0:
                    delta_prev = (delta_prev * m_prev) / rho_h
                else:
                    delta_prev = delta_prev * m_prev
                # Activation derivative
                act_deriv = self._activate_grad(pre_activations[l - 1])
                delta = delta_prev * act_deriv

        grad_W.reverse()
        grad_b.reverse()
        return grad_W, grad_b

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 500,
        lr: float = 0.02,
        batch_size: int = 32,
        p_hidden: float = 0.5,
        p_input: float = 0.0,
        mode: str = "inverted",
        verbose: bool = False,
    ) -> List[float]:
        """
        Train the MLP with SGD using Inverted Dropout.
        """
        X = np.atleast_2d(X)
        y = np.atleast_2d(y)
        N = X.shape[0]
        history = []

        for epoch in range(epochs):
            indices = self.rng.permutation(N)
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_idx = indices[start:end]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                _, acts, pre_acts, masks = self.forward(
                    X_batch, training=True, p_hidden=p_hidden, p_input=p_input, mode=mode
                )
                gW, gb = self.backward(
                    y_batch, acts, pre_acts, masks, mode=mode, p_hidden=p_hidden, p_input=p_input
                )

                for l in range(self.n_layers):
                    self.W[l] -= lr * gW[l]
                    self.b[l] -= lr * gb[l]

            # Track deterministic training error
            y_eval = self.predict(X, mode=mode, p_hidden=p_hidden, p_input=p_input)
            loss = 0.5 * np.mean((y_eval - y) ** 2)
            history.append(float(loss))
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6f}")

        return history

    def predict(
        self,
        X: np.ndarray,
        mode: str = "inverted",
        p_hidden: float = 0.5,
        p_input: float = 0.0,
    ) -> np.ndarray:
        """
        Deterministic prediction without dropout (expected activations).
        """
        out, _, _, _ = self.forward(
            X, training=False, p_hidden=p_hidden, p_input=p_input, mode=mode
        )
        return out

    def predict_mc_dropout(
        self,
        X: np.ndarray,
        n_samples: int = 50,
        p_hidden: float = 0.5,
        p_input: float = 0.0,
        mode: str = "inverted",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout inference (Gal & Ghahramani, 2016 / Section 9.6.1, Eq 9.51).
        Performs n_samples forward passes with dropout active during inference.
        Returns:
            mean: (N, d_out) predictive mean
            var: (N, d_out) epistemic predictive variance
            all_samples: (n_samples, N, d_out) raw ensemble predictions
        """
        samples = []
        for _ in range(n_samples):
            pred, _, _, _ = self.forward(
                X, training=True, p_hidden=p_hidden, p_input=p_input, mode=mode
            )
            samples.append(pred)
        samples_arr = np.array(samples)  # (T, N, d_out)
        mean = np.mean(samples_arr, axis=0)
        var = np.var(samples_arr, axis=0)
        return mean, var, samples_arr


class LinearRegressionDropout:
    """
    Mathematical and empirical analysis of Dropout applied to Linear Regression (Exercise 9.18).
    Demonstrates that least-squares regression with dropout on inputs is equivalent to
    L2 weight decay with a data-dependent diagonal regularizer.
    """

    @staticmethod
    def regularized_analytical_weights(
        X: np.ndarray,
        y: np.ndarray,
        rho: float,
    ) -> np.ndarray:
        """
        Exact expected analytical solution under dropout (retention probability rho = 1 - p).
        Under Eq (9.70) and Eq (9.71):
        E[R_ni] = rho, E[R_ni R_nj] = rho if i=j, rho^2 if i != j.
        The effective regularizer adds (1 - rho) * diag(X^T X) / rho.
        w* = (X^T X + ((1 - rho) / rho) * diag(X^T X))^{-1} X^T y
        """
        X = np.atleast_2d(X)
        y = np.asarray(y).ravel()
        XTX = X.T @ X
        XTy = X.T @ y
        diag_XTX = np.diag(np.diag(XTX))

        regularizer = ((1.0 - rho) / rho) * diag_XTX
        w_star = np.linalg.solve(XTX + regularizer, XTy)
        return w_star

    @staticmethod
    def empirical_mc_weights(
        X: np.ndarray,
        y: np.ndarray,
        rho: float,
        n_trials: int = 500,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Empirical average of weights obtained by fitting linear models on dropout-masked inputs.
        """
        rng = np.random.RandomState(seed)
        N, D = X.shape
        w_list = []
        for _ in range(n_trials):
            R = (rng.rand(N, D) < rho).astype(float)
            X_drop = (X * R) / rho
            # Ordinary least squares with pseudo-inverse
            w = np.linalg.pinv(X_drop.T @ X_drop + np.eye(D) * 1e-8) @ (X_drop.T @ y)
            w_list.append(w)
        return np.mean(w_list, axis=0)


def generate_figure_9_17(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 9.17: A neural network on the left along with two examples of
    pruned networks in which a random subset of nodes have been omitted.
    Faithful reproduction of Bishop & Bishop (2024), Chapter 9, page 280.
    """
    setup_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.5, 4.0))

    # Architecture specs:
    # Layer 0 (input): 3 nodes
    # Layer 1 (hidden 1): 4 nodes
    # Layer 2 (hidden 2): 4 nodes
    # Layer 3 (output): 3 nodes
    layer_sizes = [3, 4, 4, 3]
    layer_xs = [0.15, 0.40, 0.65, 0.90]

    # Node coordinates
    node_coords = []
    for l_idx, count in enumerate(layer_sizes):
        ys = np.linspace(0.85, 0.15, count)
        coords = [(layer_xs[l_idx], y) for y in ys]
        node_coords.append(coords)

    # Color definitions matching textbook vector art
    node_face_active = '#c8d4ff'     # Soft lavender blue
    node_edge_active = '#000080'     # Navy blue
    node_edge_omitted = '#000000'    # Black dashed

    def draw_network(
        ax: plt.Axes,
        active_nodes: List[List[bool]],
        show_labels: bool = False,
    ) -> None:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.0, 1.05)
        ax.axis('off')

        # 1. Draw connecting arrows between consecutive layers
        for l in range(len(layer_sizes) - 1):
            src_coords = node_coords[l]
            dst_coords = node_coords[l + 1]
            for s_idx, (sx, sy) in enumerate(src_coords):
                if not active_nodes[l][s_idx]:
                    continue
                for d_idx, (dx, dy) in enumerate(dst_coords):
                    if not active_nodes[l + 1][d_idx]:
                        continue
                    # Arrow geometry with circle offset
                    radius = 0.042
                    angle = np.arctan2(dy - sy, dx - sx)
                    x_start = sx + radius * np.cos(angle)
                    y_start = sy + radius * np.sin(angle)
                    x_end = dx - radius * np.cos(angle)
                    y_end = dy - radius * np.sin(angle)

                    ax.annotate(
                        '', xy=(x_end, y_end), xytext=(x_start, y_start),
                        arrowprops=dict(
                            arrowstyle='->',
                            lw=1.0,
                            color='black',
                            mutation_scale=10,
                        )
                    )

        # 2. Draw nodes (circles)
        for l in range(len(layer_sizes)):
            for n_idx, (nx, ny) in enumerate(node_coords[l]):
                is_active = active_nodes[l][n_idx]
                radius = 0.044
                if is_active:
                    c = patches.Circle(
                        (nx, ny), radius,
                        facecolor=node_face_active,
                        edgecolor=node_edge_active,
                        lw=1.6,
                        zorder=4
                    )
                else:
                    c = patches.Circle(
                        (nx, ny), radius,
                        facecolor='none',
                        edgecolor=node_edge_omitted,
                        lw=1.4,
                        linestyle='--',
                        zorder=4
                    )
                ax.add_patch(c)

        # 3. Layer labels (only on the left diagram)
        if show_labels:
            ax.text(layer_xs[0], 0.96, 'inputs', fontsize=11, ha='center', va='bottom')
            ax.text((layer_xs[1] + layer_xs[2]) / 2, 0.98, 'hidden units', fontsize=11, ha='center', va='bottom')
            ax.text(layer_xs[3], 0.96, 'outputs', fontsize=11, ha='center', va='bottom')

    # Panel 1: Full network (all nodes active)
    active_full = [[True] * count for count in layer_sizes]
    draw_network(ax1, active_full, show_labels=True)

    # Panel 2: Pruned network 1 (textbook page 280 middle diagram)
    # Inputs: node 1 dropped (index 1)
    # Hidden 1: node 0 and node 2 dropped (indices 0, 2)
    # Hidden 2: all 4 active
    # Outputs: all 3 active
    active_pruned1 = [
        [True, False, True],          # Input: middle dropped
        [False, True, False, True],   # Hidden 1: top and 3rd dropped
        [True, True, True, True],     # Hidden 2: all present
        [True, True, True],           # Output: all present
    ]
    draw_network(ax2, active_pruned1, show_labels=False)

    # Panel 3: Pruned network 2 (textbook page 280 right diagram)
    # Inputs: all 3 active
    # Hidden 1: node 1 dropped (index 1)
    # Hidden 2: node 1 and node 2 dropped (indices 1, 2)
    # Outputs: all 3 active
    active_pruned2 = [
        [True, True, True],           # Input: all present
        [True, False, True, True],    # Hidden 1: 2nd dropped
        [True, False, False, True],   # Hidden 2: 2nd and 3rd dropped
        [True, True, True],           # Output: all present
    ]
    draw_network(ax3, active_pruned2, show_labels=False)

    plt.tight_layout()

    _save_figure(fig, "Figure_9_17", save_dir)
    _save_figure(fig, "fig_9_17_dropout_architecture", save_dir)
    return fig


def generate_figure_committee_theory(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Theoretical Committee Error Ratio E_COM / E_AV as a function of ensemble size M
    and error correlation r (Eqs 9.47 - 9.50, 9.64).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    M_vals = np.arange(1, 31)
    correlations = [0.0, 0.2, 0.5, 0.8, 1.0]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#7f7f7f']

    for r, col in zip(correlations, colors):
        # E_COM / E_AV = (1 + (M - 1) * r) / M (Eq 9.47 / 9.50)
        ratio = (1.0 + (M_vals - 1) * r) / M_vals
        label = f"$r = {r:.1f}$"
        if r == 0.0:
            label += " (Uncorrelated: $1/M$)"
        elif r == 1.0:
            label += " (Identical: $E_{\\mathrm{COM}} = E_{\\mathrm{AV}}$)"
        ax.plot(M_vals, ratio, marker='o' if r in [0.0, 1.0] else None,
                lw=2.0, color=col, label=label, markersize=4)

    # Highlight Jensen's inequality bound (Eq 9.64)
    ax.axhline(1.0, color='black', linestyle=':', lw=1.2, label=r"Jensen's bound: $E_{\mathrm{COM}} \leq E_{\mathrm{AV}}$")

    ax.set_xlim(1, 30)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Number of Committee Models $M$", fontsize=11)
    ax.set_ylabel("Error Ratio $E_{\\mathrm{COM}} / E_{\\mathrm{AV}}$", fontsize=11)
    ax.set_title("Committee Error Reduction vs. Error Correlation $r$", fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fontsize=10, loc='center right')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    _save_figure(fig, "fig_9_ensemble_bias_variance_reduction", save_dir)
    return fig


def generate_figure_mc_dropout(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Monte Carlo Dropout (Gal & Ghahramani, 2016) epistemic uncertainty estimation
    on non-linear regression with an out-of-distribution / missing data gap.
    """
    setup_style()
    rng = np.random.RandomState(42)

    # Generate synthetic regression data with missing data interval [0.3, 0.8]
    x1 = rng.uniform(-1.8, 0.3, size=35)
    x2 = rng.uniform(0.8, 2.0, size=35)
    X_train = np.sort(np.concatenate([x1, x2]))[:, np.newaxis]
    y_true_fn = lambda x: np.sin(2.5 * x) + 0.3 * x
    y_train = y_true_fn(X_train.ravel()) + rng.randn(len(X_train)) * 0.12

    X_test = np.linspace(-2.2, 2.4, 200)[:, np.newaxis]
    y_test_true = y_true_fn(X_test.ravel())

    # 1. Train MLP with Inverted Dropout
    mlp_drop = DropoutMLP([1, 48, 48, 1], activation="relu", seed=42)
    mlp_drop.fit(X_train, y_train[:, np.newaxis], epochs=900, lr=0.03, p_hidden=0.3, mode="inverted")

    # MC Dropout inference: 60 forward passes
    mean_drop, var_drop, samples = mlp_drop.predict_mc_dropout(X_test, n_samples=60, p_hidden=0.3)
    std_drop = np.sqrt(var_drop).ravel()
    mean_drop = mean_drop.ravel()

    # 2. Train Standard MLP without Dropout
    mlp_std = DropoutMLP([1, 48, 48, 1], activation="relu", seed=42)
    mlp_std.fit(X_train, y_train[:, np.newaxis], epochs=900, lr=0.03, p_hidden=0.0, mode="inverted")
    pred_std = mlp_std.predict(X_test, p_hidden=0.0).ravel()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # Panel 1: Standard Deterministic Network
    ax1.plot(X_test, y_test_true, 'k--', lw=1.5, label="True function $h(x)$")
    ax1.scatter(X_train, y_train, c='crimson', s=25, alpha=0.8, label="Training data $(N=70)$", zorder=3)
    ax1.plot(X_test, pred_std, color='mediumblue', lw=2.0, label="Standard MLP (No Dropout)")
    ax1.axvspan(0.3, 0.8, color='gray', alpha=0.15, label="Missing data gap (OOD)")
    ax1.set_xlim(-2.2, 2.4)
    ax1.set_ylim(-1.6, 1.8)
    ax1.set_xlabel("$x$", fontsize=11)
    ax1.set_ylabel("$y$", fontsize=11)
    ax1.set_title("(a) Standard MLP (Overconfident in gap)", fontsize=12, fontweight='bold')
    ax1.legend(frameon=True, fontsize=9, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel 2: MC Dropout with Epistemic Uncertainty
    ax2.plot(X_test, y_test_true, 'k--', lw=1.5, label="True function $h(x)$")
    ax2.scatter(X_train, y_train, c='crimson', s=25, alpha=0.8, label="Training data $(N=70)$", zorder=3)
    ax2.plot(X_test, mean_drop, color='darkgreen', lw=2.0, label="MC Dropout Mean (60 passes)")
    # 95% Confidence Interval (+- 2 std)
    ax2.fill_between(
        X_test.ravel(),
        mean_drop - 2 * std_drop,
        mean_drop + 2 * std_drop,
        color='lightgreen', alpha=0.45,
        label=r"Epistemic Uncertainty ($\pm 2\sigma_*$)"
    )
    ax2.axvspan(0.3, 0.8, color='gray', alpha=0.15, label="Missing data gap (OOD)")
    ax2.set_xlim(-2.2, 2.4)
    ax2.set_ylim(-1.6, 1.8)
    ax2.set_xlabel("$x$", fontsize=11)
    ax2.set_ylabel("$y$", fontsize=11)
    ax2.set_title("(b) MC Dropout (Adaptive Uncertainty)", fontsize=12, fontweight='bold')
    ax2.legend(frameon=True, fontsize=9, loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    _save_figure(fig, "fig_9_mc_dropout_uncertainty", save_dir)
    return fig


def generate_figure_dropout_coadaptation(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Visualization of hidden unit co-adaptation with and without Dropout.
    Plots the hidden layer activation correlation matrix Corr(h_i, h_j).
    """
    setup_style()
    rng = np.random.RandomState(42)
    N = 150
    X = rng.randn(N, 10)
    y = np.sin(X[:, :3]).sum(axis=1)[:, np.newaxis] + 0.1 * rng.randn(N, 1)

    # Model without dropout
    mlp_no_drop = DropoutMLP([10, 20, 1], activation="relu", seed=42)
    mlp_no_drop.fit(X, y, epochs=400, lr=0.03, p_hidden=0.0, mode="inverted")
    _, acts_no, _, _ = mlp_no_drop.forward(X, training=False, p_hidden=0.0)
    H_no = acts_no[1]  # (N, 20)
    corr_no = np.corrcoef(H_no.T)
    corr_no = np.nan_to_num(corr_no, nan=0.0)

    # Model with dropout
    mlp_drop = DropoutMLP([10, 20, 1], activation="relu", seed=42)
    mlp_drop.fit(X, y, epochs=400, lr=0.03, p_hidden=0.5, mode="inverted")
    _, acts_drop, _, _ = mlp_drop.forward(X, training=False, p_hidden=0.5)
    H_drop = acts_drop[1]  # (N, 20)
    corr_drop = np.corrcoef(H_drop.T)
    corr_drop = np.nan_to_num(corr_drop, nan=0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))

    im1 = ax1.imshow(corr_no, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax1.set_title("(a) Without Dropout (High Co-adaptation)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Hidden Unit Index", fontsize=10)
    ax1.set_ylabel("Hidden Unit Index", fontsize=10)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(corr_drop, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax2.set_title("(b) With Dropout $p=0.5$ (Decorrelated Features)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Hidden Unit Index", fontsize=10)
    ax2.set_ylabel("Hidden Unit Index", fontsize=10)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    _save_figure(fig, "fig_9_dropout_coadaptation", save_dir)
    return fig
