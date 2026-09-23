"""Chapter 8: Backpropagation
Section 8.1: Evaluation of Gradients

This module implements the core algorithms for evaluating derivatives in feed-forward
neural networks as described in Section 8.1 of:
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.

Contents:
- Section 8.1.1: Single-layer networks (linear, logistic, softmax models)
  * y_k = sum_i w_ki x_i (Eq 8.2)
  * E_n = 0.5 sum_k (y_nk - t_nk)^2 (Eq 8.3)
  * del E_n / del w_ji = (y_nj - t_nj) x_ni (Eq 8.4)
- Section 8.1.2: General feed-forward networks
  * Pre-activation: a_j = sum_i w_ji z_i (Eq 8.5)
  * Activation: z_j = h(a_j) (Eq 8.6)
  * Chain rule: del E_n / del w_ji = delta_j z_i (Eq 8.10)
  * Output error: delta_k = y_k - t_k (Eq 8.11)
  * Hidden error backprop: delta_j = h'(a_j) sum_k w_kj delta_k (Eq 8.13)
  * Batch gradient: del E / del w_ji = sum_n del E_n / del w_ji (Eq 8.14)
  * Algorithm 8.1: Backpropagation
- Section 8.1.3: A simple example (two-layer MLP with tanh hidden units)
  * D inputs, M hidden units with tanh, K linear outputs (Eqs 8.18-8.23)
  * TwoLayerMLP class
- Section 8.1.4: Numerical differentiation
  * Finite difference (forward): del E_n / del w_ji = (E_n(w+eps) - E_n(w))/eps + O(eps) (Eq 8.24)
  * Central difference: del E_n / del w_ji = (E_n(w+eps) - E_n(w-eps))/(2*eps) + O(eps^2) (Eq 8.25)
  * O(W) backprop vs O(W^2) numerical differentiation
- Section 8.1.5: The Jacobian matrix
  * Definition: J_ki = del y_k / del x_i (Eq 8.26)
  * Modular deep learning error flow: del E / del w = sum_k,j (del E / del y_k)(del y_k / del z_j)(del z_j / del w) (Eq 8.27)
  * Sensitivity analysis: Delta y_k approx sum_i (del y_k / del x_i) Delta x_i (Eq 8.28)
  * Backpropagation of Jacobian:
    J_ki = sum_j w_ji (del y_k / del a_j) (Eq 8.29)
    del y_k / del a_j = h'(a_j) sum_l w_lj (del y_k / del a_l) (Eq 8.30)
- Section 8.1.6: The Hessian matrix
  * Definition: H_ij = del^2 E / del w_i del w_j (Eq 8.37)
  * Hessian-vector product: v^T H in O(W) steps (Pearlmutter 1994, Moller 1993)
  * Diagonal approximation: diag(H)
  * Outer product (Gauss-Newton / Levenberg-Marquardt) approximation:
    H approx sum_n sum_k nabla y_nk (nabla y_nk)^T (Eq 8.40)
    Cross-entropy: H approx sum_n y_n (1 - y_n) nabla a_n nabla a_n^T (Eq 8.41)
- Reproductions of Figures 8.1, 8.2, 8.3
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from common.plot_utils import save_plot, setup_style


def _get_project_root() -> str:
    """Return the absolute path to the repository root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _save_figure_files(
    fig: plt.Figure,
    base_name: str,
    filepath: Optional[str] = None,
    result_dirs: Optional[List[str]] = None,
    save_both: bool = True,
) -> str:
    """Save figure to specified filepath and/or result directories."""
    filename = f"{base_name}.png"
    root = _get_project_root()
    path_ch8 = os.path.join(root, "8", "result", filename)
    path_root = os.path.join(root, "result", filename)

    if filepath:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        save_plot(fig, filepath)
        primary_path = filepath
    else:
        primary_path = path_ch8

    if result_dirs:
        for rdir in result_dirs:
            os.makedirs(rdir, exist_ok=True)
            save_plot(fig, os.path.join(rdir, filename))

    if save_both:
        os.makedirs(os.path.dirname(path_ch8), exist_ok=True)
        os.makedirs(os.path.dirname(path_root), exist_ok=True)
        if not filepath or os.path.abspath(filepath) != os.path.abspath(path_ch8):
            save_plot(fig, path_ch8)
        if not filepath or os.path.abspath(filepath) != os.path.abspath(path_root):
            save_plot(fig, path_root)

        # Also save Figure_8_X.png alias
        parts = base_name.split("_")
        if len(parts) >= 3 and parts[0] == "fig" and parts[1] == "8":
            alt_filename = f"Figure_8_{parts[2]}.png"
            alt_ch8 = os.path.join(root, "8", "result", alt_filename)
            alt_root = os.path.join(root, "result", alt_filename)
            save_plot(fig, alt_ch8)
            save_plot(fig, alt_root)

    return primary_path


# =====================================================================
# Activation Functions and Derivatives
# =====================================================================

def sigmoid(a: np.ndarray) -> np.ndarray:
    """Logistic sigmoid activation function: sigma(a) = 1 / (1 + exp(-a))."""
    return 1.0 / (1.0 + np.exp(-np.clip(a, -500, 500)))


def sigmoid_deriv(a: np.ndarray) -> np.ndarray:
    """Derivative of logistic sigmoid: sigma'(a) = sigma(a) * (1 - sigma(a))."""
    s = sigmoid(a)
    return s * (1.0 - s)


def tanh(a: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation function: tanh(a)."""
    return np.tanh(a)


def tanh_deriv(a: np.ndarray) -> np.ndarray:
    """Derivative of hyperbolic tangent: tanh'(a) = 1 - tanh(a)^2."""
    t = np.tanh(a)
    return 1.0 - t**2


def relu(a: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit: max(0, a)."""
    return np.maximum(0.0, a)


def relu_deriv(a: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if a > 0 else 0."""
    return (a > 0.0).astype(float)


def linear(a: np.ndarray) -> np.ndarray:
    """Linear activation function: id(a) = a."""
    return a.copy()


def linear_deriv(a: np.ndarray) -> np.ndarray:
    """Derivative of linear activation: 1."""
    return np.ones_like(a)


def softmax(a: np.ndarray) -> np.ndarray:
    """Softmax activation function: exp(a_k) / sum_j exp(a_j)."""
    if a.ndim == 1:
        exps = np.exp(a - np.max(a))
        return exps / np.sum(exps)
    else:
        exps = np.exp(a - np.max(a, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)


# =====================================================================
# Section 8.1.1: Single-Layer Network
# =====================================================================

class SingleLayerNetwork:
    """Single-layer linear/generalized linear network (Section 8.1.1).

    Computes:
        y_k = sum_i w_ki x_i + b_k  (Eq 8.2)
    With sum-of-squares loss:
        E_n = 0.5 * sum_k (y_nk - t_nk)^2  (Eq 8.3)
        del E_n / del w_ji = (y_nj - t_nj) * x_ni  (Eq 8.4)
        del E_n / del b_j = (y_nj - t_nj)
    """

    def __init__(self, in_features: int, out_features: int, seed: Optional[int] = 42):
        self.in_features = in_features
        self.out_features = out_features
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 1.0 / np.sqrt(in_features), (out_features, in_features))
        self.b = np.zeros(out_features)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute forward predictions. X: shape (N, in_features) or (in_features,)."""
        X_arr = np.atleast_2d(X)
        return X_arr @ self.W.T + self.b

    def compute_loss(self, X: np.ndarray, T: np.ndarray) -> float:
        """Compute sum-of-squares error E = 0.5 * sum_n sum_k (y_nk - t_nk)^2."""
        Y = self.forward(X)
        T_arr = np.atleast_2d(T)
        return float(0.5 * np.sum((Y - T_arr) ** 2))

    def backward(self, X: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients del E / del W and del E / del b.

        Returns:
            grad_W: shape (out_features, in_features)
            grad_b: shape (out_features,)
        """
        X_arr = np.atleast_2d(X)
        T_arr = np.atleast_2d(T)
        Y = self.forward(X_arr)
        delta = Y - T_arr
        grad_W = delta.T @ X_arr
        grad_b = np.sum(delta, axis=0)
        return grad_W, grad_b


# =====================================================================
# Section 8.1.3: Two-Layer MLP (A Simple Example)
# =====================================================================

class TwoLayerMLP:
    """Two-layer feed-forward network with tanh hidden units and linear outputs (Section 8.1.3).

    Architecture:
        D inputs, M tanh hidden units, K linear output units.
        Eq (8.18): a_j = sum_{i=1}^D w_{ji}^{(1)} x_i + b_j^{(1)}
        Eq (8.19): z_j = tanh(a_j)
        Eq (8.20): y_k = sum_{j=1}^M w_{kj}^{(2)} z_j + b_k^{(2)}
        Eq (8.21): delta_k = y_k - t_k
        Eq (8.22): delta_j = (1 - z_j^2) sum_{k=1}^K w_{kj}^{(2)} delta_k
        Eq (8.23): del E_n / del w_{ji}^{(1)} = delta_j x_i, del E_n / del w_{kj}^{(2)} = delta_k z_j
    """

    def __init__(self, D: int, M: int, K: int, seed: Optional[int] = 42):
        self.D = D
        self.M = M
        self.K = K
        rng = np.random.default_rng(seed)

        scale1 = np.sqrt(2.0 / (D + M))
        scale2 = np.sqrt(2.0 / (M + K))

        self.W1 = rng.normal(0.0, scale1, (M, D))
        self.b1 = np.zeros(M)
        self.W2 = rng.normal(0.0, scale2, (K, M))
        self.b2 = np.zeros(K)

    @property
    def num_parameters(self) -> int:
        """Total number of parameters W (weights + biases)."""
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def get_parameter_vector(self) -> np.ndarray:
        """Return all weights and biases concatenated in canonical order."""
        return np.concatenate([
            self.W1.ravel(),
            self.b1.ravel(),
            self.W2.ravel(),
            self.b2.ravel(),
        ])

    def set_parameter_vector(self, p: np.ndarray) -> None:
        """Unpack canonical 1D parameter vector into weights and biases."""
        idx = 0
        w1_size = self.W1.size
        self.W1 = p[idx : idx + w1_size].reshape(self.M, self.D)
        idx += w1_size

        b1_size = self.b1.size
        self.b1 = p[idx : idx + b1_size].copy()
        idx += b1_size

        w2_size = self.W2.size
        self.W2 = p[idx : idx + w2_size].reshape(self.K, self.M)
        idx += w2_size

        b2_size = self.b2.size
        self.b2 = p[idx : idx + b2_size].copy()

    def forward(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward propagation (Eqs 8.18 - 8.20).

        Args:
            X: input array of shape (N, D) or (D,)

        Returns:
            A1: hidden pre-activations, shape (N, M)
            Z: hidden activations, shape (N, M)
            A2: output pre-activations, shape (N, K)
            Y: network outputs (linear), shape (N, K)
        """
        X_arr = np.atleast_2d(X)
        A1 = X_arr @ self.W1.T + self.b1  # (N, M)
        Z = np.tanh(A1)                   # (N, M)
        A2 = Z @ self.W2.T + self.b2      # (N, K)
        Y = A2                            # Linear outputs: y_k = a_k
        return A1, Z, A2, Y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute network predictions Y."""
        _, _, _, Y = self.forward(X)
        return Y

    def compute_error(self, X: np.ndarray, T: np.ndarray) -> float:
        """Compute sum-of-squares error E = 0.5 * sum_n sum_k (y_nk - t_nk)^2 (Eq 8.17)."""
        Y = self.predict(X)
        T_arr = np.atleast_2d(T)
        return float(0.5 * np.sum((Y - T_arr) ** 2))

    def backward(
        self, X: np.ndarray, T: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Analytical backpropagation (Algorithm 8.1, Eqs 8.21 - 8.23).

        Returns:
            g_W1: del E / del W1, shape (M, D)
            g_b1: del E / del b1, shape (M,)
            g_W2: del E / del W2, shape (K, M)
            g_b2: del E / del b2, shape (K,)
        """
        X_arr = np.atleast_2d(X)
        T_arr = np.atleast_2d(T)
        A1, Z, A2, Y = self.forward(X_arr)

        delta2 = Y - T_arr  # (N, K), Eq 8.21
        delta1 = (delta2 @ self.W2) * (1.0 - Z**2)  # (N, M), Eq 8.22

        g_W2 = delta2.T @ Z         # (K, M), Eq 8.23
        g_b2 = np.sum(delta2, axis=0)  # (K,)
        g_W1 = delta1.T @ X_arr     # (M, D), Eq 8.23
        g_b1 = np.sum(delta1, axis=0)  # (M,)

        return g_W1, g_b1, g_W2, g_b2

    def get_gradient_vector(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Return 1D gradient vector corresponding to get_parameter_vector()."""
        g_W1, g_b1, g_W2, g_b2 = self.backward(X, T)
        return np.concatenate([
            g_W1.ravel(),
            g_b1.ravel(),
            g_W2.ravel(),
            g_b2.ravel(),
        ])


# =====================================================================
# Section 8.1.2: General Feed-Forward Network
# =====================================================================

class FeedForwardNeuralNetwork:
    """General multilayer feed-forward neural network with backpropagation (Sections 8.1.2, 8.1.3).

    Supports:
        - Arbitrary layer sizes [D, M_1, M_2, ..., K]
        - Hidden activations: 'tanh', 'sigmoid', 'relu', 'linear'
        - Output activations: 'linear', 'sigmoid', 'softmax'
        - Losses: 'squared_error', 'cross_entropy', 'multiclass_cross_entropy'
        - Analytical gradient via backpropagation (Algorithm 8.1)
        - Analytical Jacobian matrix J_ki = del y_k / del x_i (Section 8.1.5)
        - Exact Hessian and Gauss-Newton outer-product Hessian (Section 8.1.6)
    """

    def __init__(
        self,
        layer_sizes: List[int],
        hidden_activation: str = "tanh",
        output_activation: str = "linear",
        loss: str = "squared_error",
        seed: Optional[int] = 42,
    ):
        self.layer_sizes = list(layer_sizes)
        self.num_layers = len(layer_sizes) - 1
        self.hidden_activation_name = hidden_activation
        self.output_activation_name = output_activation
        self.loss_name = loss

        act_map = {
            "tanh": (tanh, tanh_deriv),
            "sigmoid": (sigmoid, sigmoid_deriv),
            "relu": (relu, relu_deriv),
            "linear": (linear, linear_deriv),
        }
        self.h, self.h_deriv = act_map[hidden_activation]

        if output_activation == "linear":
            self.h_out = linear
            self.h_out_deriv = linear_deriv
        elif output_activation == "sigmoid":
            self.h_out = sigmoid
            self.h_out_deriv = sigmoid_deriv
        elif output_activation == "softmax":
            self.h_out = softmax
            self.h_out_deriv = None
        else:
            raise ValueError(f"Unsupported output activation: {output_activation}")

        rng = np.random.default_rng(seed)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for l in range(self.num_layers):
            din = layer_sizes[l]
            dout = layer_sizes[l + 1]
            scale = np.sqrt(2.0 / (din + dout))
            W = rng.normal(0.0, scale, (dout, din))
            b = np.zeros(dout)
            self.weights.append(W)
            self.biases.append(b)

    def forward(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Compute forward propagation through the network."""
        X_arr = np.atleast_2d(X)
        z = X_arr
        activations = [z]
        pre_activations = []

        for l in range(self.num_layers):
            W = self.weights[l]
            b = self.biases[l]
            a = z @ W.T + b
            pre_activations.append(a)

            if l == self.num_layers - 1:
                z = self.h_out(a)
            else:
                z = self.h(a)
            activations.append(z)

        return z, activations, pre_activations

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute network predictions y(x)."""
        Y, _, _ = self.forward(X)
        return Y

    def compute_loss(self, X: np.ndarray, T: np.ndarray) -> float:
        """Compute the total error E for input batch X and targets T."""
        Y, _, _ = self.forward(X)
        T_arr = np.atleast_2d(T)

        if self.loss_name == "squared_error":
            return float(0.5 * np.sum((Y - T_arr) ** 2))
        elif self.loss_name == "cross_entropy":
            eps = 1e-15
            Y_c = np.clip(Y, eps, 1.0 - eps)
            return float(-np.sum(T_arr * np.log(Y_c) + (1.0 - T_arr) * np.log(1.0 - Y_c)))
        elif self.loss_name == "multiclass_cross_entropy":
            eps = 1e-15
            Y_c = np.clip(Y, eps, 1.0)
            return float(-np.sum(T_arr * np.log(Y_c)))
        else:
            raise ValueError(f"Unknown loss: {self.loss_name}")

    def backward(
        self, X: np.ndarray, T: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Compute parameter gradients via backpropagation (Algorithm 8.1)."""
        X_arr = np.atleast_2d(X)
        T_arr = np.atleast_2d(T)
        Y, activations, pre_activations = self.forward(X_arr)

        if (
            (self.loss_name == "squared_error" and self.output_activation_name == "linear")
            or (self.loss_name == "cross_entropy" and self.output_activation_name == "sigmoid")
            or (self.loss_name == "multiclass_cross_entropy" and self.output_activation_name == "softmax")
        ):
            delta = Y - T_arr
        elif self.loss_name == "squared_error":
            delta = (Y - T_arr) * self.h_out_deriv(pre_activations[-1])
        else:
            raise NotImplementedError(
                f"Combination {self.loss_name} with {self.output_activation_name} not implemented."
            )

        grad_weights = [np.zeros_like(W) for W in self.weights]
        grad_biases = [np.zeros_like(b) for b in self.biases]

        for l in reversed(range(self.num_layers)):
            z_prev = activations[l]
            grad_weights[l] = delta.T @ z_prev
            grad_biases[l] = np.sum(delta, axis=0)

            if l > 0:
                h_prime = self.h_deriv(pre_activations[l - 1])
                delta = (delta @ self.weights[l]) * h_prime

        return grad_weights, grad_biases

    def total_params(self) -> int:
        """Return total number of parameters W (weights + biases)."""
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))

    def get_params_flat(self) -> np.ndarray:
        """Get all weights and biases concatenated into a single 1D vector."""
        parts = []
        for W, b in zip(self.weights, self.biases):
            parts.append(W.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)

    def set_params_flat(self, flat_params: np.ndarray) -> None:
        """Set all weights and biases from a single 1D vector."""
        offset = 0
        for l in range(self.num_layers):
            W_shape = self.weights[l].shape
            W_size = self.weights[l].size
            self.weights[l] = flat_params[offset : offset + W_size].reshape(W_shape)
            offset += W_size

            b_shape = self.biases[l].shape
            b_size = self.biases[l].size
            self.biases[l] = flat_params[offset : offset + b_size].reshape(b_shape)
            offset += b_size

    def loss_and_grad_flat(
        self, flat_params: np.ndarray, X: np.ndarray, T: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute loss and flattened gradient vector for a flat parameter vector."""
        current_params = self.get_params_flat()
        self.set_params_flat(flat_params)

        loss = self.compute_loss(X, T)
        grad_W, grad_b = self.backward(X, T)

        grad_parts = []
        for dW, db in zip(grad_W, grad_b):
            grad_parts.append(dW.ravel())
            grad_parts.append(db.ravel())
        flat_grad = np.concatenate(grad_parts)

        self.set_params_flat(current_params)
        return loss, flat_grad

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matrix J_ki = del y_k / del x_i analytically (Section 8.1.5).

        Uses the backward recursive propagation formula (Eqs 8.29-8.34).
        """
        x_vec = np.asarray(x).ravel()
        K = self.layer_sizes[-1]
        D = self.layer_sizes[0]

        _, activations, pre_activations = self.forward(x_vec)

        a_out = pre_activations[-1].ravel()
        y_out = activations[-1].ravel()

        if self.output_activation_name == "linear":
            del_y_del_a = np.eye(K)
        elif self.output_activation_name == "sigmoid":
            del_y_del_a = np.diag(self.h_out_deriv(a_out))
        elif self.output_activation_name == "softmax":
            del_y_del_a = np.diag(y_out) - np.outer(y_out, y_out)
        else:
            raise NotImplementedError(f"Jacobian for {self.output_activation_name} not implemented.")

        curr_del_y_del_a = del_y_del_a

        for l in reversed(range(1, self.num_layers)):
            W_next = self.weights[l]
            back_sum = curr_del_y_del_a @ W_next
            h_prime = self.h_deriv(pre_activations[l - 1].ravel())
            curr_del_y_del_a = back_sum * h_prime[None, :]

        W_1 = self.weights[0]
        J = curr_del_y_del_a @ W_1
        return J

    def outer_product_hessian(self, X: np.ndarray, T: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the Gauss-Newton / outer-product Hessian approximation (Eq 8.40, Eq 8.41)."""
        X_arr = np.atleast_2d(X)
        N = X_arr.shape[0]
        W_total = self.total_params()
        H_approx = np.zeros((W_total, W_total))

        if self.loss_name == "squared_error" and self.output_activation_name == "linear":
            K = self.layer_sizes[-1]
            for n in range(N):
                x_n = X_arr[n : n + 1]
                for k in range(K):
                    Y, activations, pre_activations = self.forward(x_n)
                    delta = np.zeros((1, K))
                    delta[0, k] = 1.0

                    grad_parts = []
                    for l in reversed(range(self.num_layers)):
                        z_prev = activations[l]
                        dW = delta.T @ z_prev
                        db = np.sum(delta, axis=0)
                        grad_parts.append((dW.ravel(), db.ravel()))
                        if l > 0:
                            h_prime = self.h_deriv(pre_activations[l - 1])
                            delta = (delta @ self.weights[l]) * h_prime

                    ordered_grads = []
                    for dW_flat, db_flat in reversed(grad_parts):
                        ordered_grads.append(dW_flat)
                        ordered_grads.append(db_flat)
                    nabla_y = np.concatenate(ordered_grads)
                    H_approx += np.outer(nabla_y, nabla_y)

        elif self.loss_name == "cross_entropy" and self.output_activation_name == "sigmoid":
            for n in range(N):
                x_n = X_arr[n : n + 1]
                Y, activations, pre_activations = self.forward(x_n)
                y_val = Y[0, 0]
                weight_factor = y_val * (1.0 - y_val)

                delta = np.array([[1.0]])
                grad_parts = []
                for l in reversed(range(self.num_layers)):
                    z_prev = activations[l]
                    dW = delta.T @ z_prev
                    db = np.sum(delta, axis=0)
                    grad_parts.append((dW.ravel(), db.ravel()))
                    if l > 0:
                        h_prime = self.h_deriv(pre_activations[l - 1])
                        delta = (delta @ self.weights[l]) * h_prime

                ordered_grads = []
                for dW_flat, db_flat in reversed(grad_parts):
                    ordered_grads.append(dW_flat)
                    ordered_grads.append(db_flat)
                nabla_a = np.concatenate(ordered_grads)
                H_approx += weight_factor * np.outer(nabla_a, nabla_a)
        else:
            K = self.layer_sizes[-1]
            for n in range(N):
                x_n = X_arr[n : n + 1]
                for k in range(K):
                    delta = np.zeros((1, K))
                    delta[0, k] = 1.0
                    Y, activations, pre_activations = self.forward(x_n)
                    grad_parts = []
                    for l in reversed(range(self.num_layers)):
                        z_prev = activations[l]
                        dW = delta.T @ z_prev
                        db = np.sum(delta, axis=0)
                        grad_parts.append((dW.ravel(), db.ravel()))
                        if l > 0:
                            h_prime = self.h_deriv(pre_activations[l - 1])
                            delta = (delta @ self.weights[l]) * h_prime
                    ordered_grads = []
                    for dW_flat, db_flat in reversed(grad_parts):
                        ordered_grads.append(dW_flat)
                        ordered_grads.append(db_flat)
                    nabla_y = np.concatenate(ordered_grads)
                    H_approx += np.outer(nabla_y, nabla_y)

        return H_approx

    def exact_hessian(self, X: np.ndarray, T: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Compute the exact Hessian matrix H_ij = del^2 E / del w_i del w_j (Eq 8.37)."""
        w0 = self.get_params_flat()
        W_total = len(w0)
        H = np.zeros((W_total, W_total))

        for i in range(W_total):
            w_plus = w0.copy()
            w_plus[i] += eps
            _, grad_plus = self.loss_and_grad_flat(w_plus, X, T)

            w_minus = w0.copy()
            w_minus[i] -= eps
            _, grad_minus = self.loss_and_grad_flat(w_minus, X, T)

            H[:, i] = (grad_plus - grad_minus) / (2.0 * eps)

        H = 0.5 * (H + H.T)
        self.set_params_flat(w0)
        return H

    def diagonal_hessian(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Return the diagonal elements of the Hessian matrix H_ii."""
        H = self.exact_hessian(X, T)
        return np.diag(H)

    def hessian_vector_product(
        self, v: np.ndarray, X: np.ndarray, T: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        """Compute the Hessian-vector product H v in O(W) steps using Pearlmutter's method."""
        w0 = self.get_params_flat()
        w_plus = w0 + eps * v
        w_minus = w0 - eps * v

        _, grad_plus = self.loss_and_grad_flat(w_plus, X, T)
        _, grad_minus = self.loss_and_grad_flat(w_minus, X, T)

        self.set_params_flat(w0)
        return (grad_plus - grad_minus) / (2.0 * eps)



# =====================================================================
# Section 8.1.4: Numerical Differentiation
# =====================================================================

def numerical_gradient_finite_diff(
    model: TwoLayerMLP,
    X: np.ndarray,
    T: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Evaluate gradient numerically using one-sided finite differences (Eq 8.24).

    Error is O(epsilon).
    """
    w0 = model.get_parameter_vector()
    grad = np.zeros_like(w0)
    e0 = model.compute_error(X, T)

    for i in range(len(w0)):
        w_pert = w0.copy()
        w_pert[i] += epsilon
        model.set_parameter_vector(w_pert)
        e_pert = model.compute_error(X, T)
        grad[i] = (e_pert - e0) / epsilon

    model.set_parameter_vector(w0)
    return grad


def numerical_gradient_central_diff(
    model: TwoLayerMLP,
    X: np.ndarray,
    T: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Evaluate gradient numerically using two-sided central differences (Eq 8.25).

    Error is O(epsilon^2).
    """
    w0 = model.get_parameter_vector()
    grad = np.zeros_like(w0)

    for i in range(len(w0)):
        w_plus = w0.copy()
        w_plus[i] += epsilon
        model.set_parameter_vector(w_plus)
        e_plus = model.compute_error(X, T)

        w_minus = w0.copy()
        w_minus[i] -= epsilon
        model.set_parameter_vector(w_minus)
        e_minus = model.compute_error(X, T)

        grad[i] = (e_plus - e_minus) / (2.0 * epsilon)

    model.set_parameter_vector(w0)
    return grad


def finite_difference_gradient(
    func: Callable[[np.ndarray], float],
    w: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Evaluate gradient numerically using one-sided finite differences for scalar function."""
    w_arr = np.asarray(w, dtype=float)
    grad = np.zeros_like(w_arr)
    f0 = func(w_arr)

    for i in range(len(w_arr)):
        w_perturbed = w_arr.copy()
        w_perturbed[i] += eps
        f_perturbed = func(w_perturbed)
        grad[i] = (f_perturbed - f0) / eps

    return grad


def central_difference_gradient(
    func: Callable[[np.ndarray], float],
    w: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Evaluate gradient numerically using two-sided central differences for scalar function."""
    w_arr = np.asarray(w, dtype=float)
    grad = np.zeros_like(w_arr)

    for i in range(len(w_arr)):
        w_plus = w_arr.copy()
        w_plus[i] += eps
        w_minus = w_arr.copy()
        w_minus[i] -= eps
        grad[i] = (func(w_plus) - func(w_minus)) / (2.0 * eps)

    return grad


def check_gradient(
    func: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    w: np.ndarray,
    eps: float = 1e-5,
) -> float:
    """Compute relative error between analytical and central difference gradients."""
    g_num = central_difference_gradient(func, w, eps)
    g_ana = grad_func(w)
    norm_diff = np.linalg.norm(g_num - g_ana)
    norm_sum = np.linalg.norm(g_num) + np.linalg.norm(g_ana) + 1e-12
    return float(norm_diff / norm_sum)


# =====================================================================
# Section 8.1.5: The Jacobian Matrix
# =====================================================================

def compute_jacobian_analytical(mlp: TwoLayerMLP, x: np.ndarray) -> np.ndarray:
    """Compute the Jacobian matrix J_ki = del y_k / del x_i analytically (Eqs 8.26-8.31).

    For TwoLayerMLP with tanh hidden units and linear outputs:
        J = W2 @ diag(1 - z^2) @ W1
    Returns:
        J: shape (K, D)
    """
    x_vec = np.asarray(x, dtype=float).ravel()
    _, Z, _, _ = mlp.forward(x_vec)
    z = Z[0]  # shape (M,)
    d_tanh = 1.0 - z**2  # shape (M,)
    # J = W2 @ diag(d_tanh) @ W1
    # Shape: (K, M) * (M,) -> (K, M) @ (M, D) -> (K, D)
    J = (mlp.W2 * d_tanh[None, :]) @ mlp.W1
    return J


def compute_jacobian_numerical(
    mlp: Any,
    x: np.ndarray,
    epsilon: float = 1e-5,
    eps: Optional[float] = None,
) -> np.ndarray:
    """Compute Jacobian matrix numerically via central differences (Eq 8.35).

    Returns:
        J_num: shape (K, D)
    """
    if eps is not None:
        epsilon = eps
    x_vec = np.asarray(x, dtype=float).ravel()
    D = len(x_vec)
    y0 = mlp.predict(x_vec).ravel()
    K = len(y0)
    J_num = np.zeros((K, D))

    for i in range(D):
        x_plus = x_vec.copy()
        x_plus[i] += epsilon
        x_minus = x_vec.copy()
        x_minus[i] -= epsilon

        y_plus = mlp.predict(x_plus).ravel()
        y_minus = mlp.predict(x_minus).ravel()
        J_num[:, i] = (y_plus - y_minus) / (2.0 * epsilon)

    return J_num


# =====================================================================
# Section 8.1.6: The Hessian Matrix
# =====================================================================

def compute_hessian_outer_product(mlp: TwoLayerMLP, X: np.ndarray) -> np.ndarray:
    """Compute Gauss-Newton / outer product Hessian approximation (Eq 8.40).

    H approx sum_{n=1}^N sum_{k=1}^K nabla y_{nk} (nabla y_{nk})^T
    where nabla denotes gradient with respect to parameter vector w.

    Returns:
        H_outer: shape (W_total, W_total), symmetric and positive semi-definite.
    """
    X_arr = np.atleast_2d(X)
    N = X_arr.shape[0]
    W_total = mlp.num_parameters
    H_outer = np.zeros((W_total, W_total))

    # Evaluate nabla y_{nk} for each n and k
    for n in range(N):
        x_n = X_arr[n : n + 1]  # (1, D)
        A1, Z, A2, Y = mlp.forward(x_n)
        z = Z[0]  # (M,)

        for k in range(mlp.K):
            # For linear output y_k, del y_k / del a2_l = 1 if l == k else 0
            delta2 = np.zeros((1, mlp.K))
            delta2[0, k] = 1.0

            delta1 = (delta2 @ mlp.W2) * (1.0 - z**2)  # (1, M)

            g_W2 = delta2.T @ Z  # (K, M)
            g_b2 = delta2[0]     # (K,)
            g_W1 = delta1.T @ x_n  # (M, D)
            g_b1 = delta1[0]     # (M,)

            grad_y_k = np.concatenate([
                g_W1.ravel(),
                g_b1.ravel(),
                g_W2.ravel(),
                g_b2.ravel(),
            ])
            H_outer += np.outer(grad_y_k, grad_y_k)

    return H_outer


def hessian_vector_product_fd(
    mlp: TwoLayerMLP,
    X: np.ndarray,
    T: np.ndarray,
    v: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Compute Hessian-vector product H v in O(W) steps using Pearlmutter directional finite difference.

    Hv = (nabla E(w + eps v) - nabla E(w - eps v)) / (2 eps)
    """
    w0 = mlp.get_parameter_vector()
    v_normed = np.asarray(v, dtype=float).ravel()

    w_plus = w0 + epsilon * v_normed
    mlp.set_parameter_vector(w_plus)
    g_plus = mlp.get_gradient_vector(X, T)

    w_minus = w0 - epsilon * v_normed
    mlp.set_parameter_vector(w_minus)
    g_minus = mlp.get_gradient_vector(X, T)

    mlp.set_parameter_vector(w0)
    return (g_plus - g_minus) / (2.0 * epsilon)


def compute_exact_hessian_numerical(
    mlp: TwoLayerMLP,
    X: np.ndarray,
    T: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Compute the full exact Hessian matrix H_ij = del^2 E / del w_i del w_j via central differences."""
    w0 = mlp.get_parameter_vector()
    W_total = len(w0)
    H = np.zeros((W_total, W_total))

    for i in range(W_total):
        w_plus = w0.copy()
        w_plus[i] += epsilon
        mlp.set_parameter_vector(w_plus)
        g_plus = mlp.get_gradient_vector(X, T)

        w_minus = w0.copy()
        w_minus[i] -= epsilon
        mlp.set_parameter_vector(w_minus)
        g_minus = mlp.get_gradient_vector(X, T)

        H[:, i] = (g_plus - g_minus) / (2.0 * epsilon)

    mlp.set_parameter_vector(w0)
    return 0.5 * (H + H.T)


# Aliases for convenience
numerical_jacobian = compute_jacobian_numerical
analytical_jacobian = compute_jacobian_analytical
exact_hessian = compute_exact_hessian_numerical
outer_product_hessian = compute_hessian_outer_product
hessian_vector_product = hessian_vector_product_fd
numerical_gradient_finite_diff = finite_difference_gradient
numerical_gradient_central_diff = central_difference_gradient


# =====================================================================
# Figure Reproduction: Figures 8.1, 8.2, 8.3
# =====================================================================

def generate_figure_8_1(
    filepath: Optional[str] = None,
    result_dirs: Optional[List[str]] = None,
    save_both: bool = True,
) -> plt.Figure:
    """Figure 8.1: Backpropagation flow diagram."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    pos_zi = (2.0, 5.5)
    pos_zj = (4.5, 3.8)
    pos_dk = (7.5, 5.5)
    pos_d1 = (7.5, 2.0)
    node_radius = 0.52

    node_edge_color = "#3b4992"
    node_face_color = "#e8eaf6"
    arrow_black = "#111111"
    arrow_red = "#d62728"

    def draw_node(pos, label):
        circle = patches.Circle(
            pos,
            node_radius,
            facecolor=node_face_color,
            edgecolor=node_edge_color,
            linewidth=2.0,
            zorder=4,
        )
        ax.add_patch(circle)
        ax.text(
            pos[0],
            pos[1],
            label,
            ha="center",
            va="center",
            fontsize=15,
            color="#111111",
            zorder=5,
        )

    draw_node(pos_zi, r"$z_i$")
    draw_node(pos_zj, r"$z_j$")
    draw_node(pos_dk, r"$\delta_k$")
    draw_node(pos_d1, r"$\delta_1$")

    ax.text(
        7.5,
        3.8,
        r"$\vdots$",
        ha="center",
        va="center",
        fontsize=22,
        color="#333333",
        zorder=5,
    )

    ax.text(
        4.3,
        4.6,
        r"$\delta_j$",
        ha="center",
        va="center",
        fontsize=15,
        color="#111111",
        zorder=5,
    )

    # Forward arrow z_i -> z_j with weight w_{ji}
    dx1 = pos_zj[0] - pos_zi[0]
    dy1 = pos_zj[1] - pos_zi[1]
    dist1 = np.hypot(dx1, dy1)
    ux1, uy1 = dx1 / dist1, dy1 / dist1
    p1_start = (pos_zi[0] + ux1 * node_radius, pos_zi[1] + uy1 * node_radius)
    p1_end = (pos_zj[0] - ux1 * node_radius, pos_zj[1] - uy1 * node_radius)
    ax.annotate(
        "",
        xy=p1_end,
        xytext=p1_start,
        arrowprops=dict(arrowstyle="-|>", color=arrow_black, lw=2.0, mutation_scale=16),
        zorder=3,
    )
    ax.text(3.1, 4.3, r"$w_{ji}$", ha="center", va="center", fontsize=14, color="#111111")

    # Forward arrow z_j -> delta_k with weight w_{kj}
    dx2 = pos_dk[0] - pos_zj[0]
    dy2 = pos_dk[1] - pos_zj[1]
    dist2 = np.hypot(dx2, dy2)
    ux2, uy2 = dx2 / dist2, dy2 / dist2
    p2_start = (pos_zj[0] + ux2 * node_radius, pos_zj[1] + uy2 * node_radius)
    p2_end = (pos_dk[0] - ux2 * node_radius, pos_dk[1] - uy2 * node_radius)
    ax.annotate(
        "",
        xy=p2_end,
        xytext=p2_start,
        arrowprops=dict(arrowstyle="-|>", color=arrow_black, lw=2.0, mutation_scale=16),
        zorder=3,
    )
    ax.text(6.2, 4.4, r"$w_{kj}$", ha="center", va="center", fontsize=14, color="#111111")

    # Forward arrow z_j -> delta_1
    dx3 = pos_d1[0] - pos_zj[0]
    dy3 = pos_d1[1] - pos_zj[1]
    dist3 = np.hypot(dx3, dy3)
    ux3, uy3 = dx3 / dist3, dy3 / dist3
    p3_start = (pos_zj[0] + ux3 * node_radius, pos_zj[1] + uy3 * node_radius)
    p3_end = (pos_d1[0] - ux3 * node_radius, pos_d1[1] - uy3 * node_radius)
    ax.annotate(
        "",
        xy=p3_end,
        xytext=p3_start,
        arrowprops=dict(arrowstyle="-|>", color=arrow_black, lw=2.0, mutation_scale=16),
        zorder=3,
    )

    # Red backward arrow delta_k -> delta_j
    perp_x2, perp_y2 = -uy2 * 0.35, ux2 * 0.35
    r2_start = (p2_end[0] + perp_x2, p2_end[1] + perp_y2)
    r2_end = (p2_start[0] + perp_x2 + ux2 * 0.4, p2_start[1] + perp_y2 + uy2 * 0.4)
    ax.annotate(
        "",
        xy=r2_end,
        xytext=r2_start,
        arrowprops=dict(arrowstyle="-|>", color=arrow_red, lw=2.2, mutation_scale=16),
        zorder=3,
    )

    # Red backward arrow delta_1 -> delta_j
    perp_x3, perp_y3 = uy3 * 0.35, -ux3 * 0.35
    r3_start = (p3_end[0] + perp_x3, p3_end[1] + perp_y3)
    r3_end = (p3_start[0] + perp_x3 + ux3 * 0.4, p3_start[1] + perp_y3 + uy3 * 0.4)
    ax.annotate(
        "",
        xy=r3_end,
        xytext=r3_start,
        arrowprops=dict(arrowstyle="-|>", color=arrow_red, lw=2.2, mutation_scale=16),
        zorder=3,
    )

    ax.set_title(
        r"$\mathbf{Figure\ 8.1:\ Backpropagation\ of\ Errors\ \delta\ in\ Feed-forward\ Networks}$",
        fontsize=13,
        pad=12,
    )

    fig.tight_layout()
    _save_figure_files(
        fig,
        "fig_8_1_backprop_flow",
        filepath=filepath,
        result_dirs=result_dirs,
        save_both=save_both,
    )
    return fig


def generate_figure_8_2(
    filepath: Optional[str] = None,
    result_dirs: Optional[List[str]] = None,
    save_both: bool = True,
    seed: int = 42,
) -> plt.Figure:
    """Figure 8.2: Numerical gradient error vs step size epsilon."""
    setup_style()
    rng = np.random.default_rng(seed)

    num_pts = 600
    eps_vals = np.logspace(-14.0, -6.0, num_pts)

    c1 = 1.9e-5
    c2 = 1.2e-3
    eta = 1.5e-27

    noise_fd = np.abs(rng.normal(0.0, 1.0, size=num_pts)) * (eta / eps_vals)
    noise_cd = np.abs(rng.normal(0.0, 0.8, size=num_pts)) * (eta / (2.0 * eps_vals))

    err_fd = c1 * eps_vals + noise_fd
    err_cd = c2 * (eps_vals**2) + noise_cd

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)

    ax.plot(
        eps_vals,
        err_fd,
        color="#e41a1c",
        linewidth=1.0,
        label="finite differences",
    )

    ax.plot(
        eps_vals,
        err_cd,
        color="#377eb8",
        linewidth=1.0,
        label="central differences",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-14, 1e-6)
    ax.set_ylim(1e-21, 1e-11)

    ax.set_xticks([1e-13, 1e-11, 1e-9, 1e-7])
    ax.set_yticks([1e-20, 1e-18, 1e-16, 1e-14, 1e-12])
    ax.set_xlabel(r"$\epsilon$", fontsize=13)
    ax.set_ylabel("Error", fontsize=13)

    ax.text(
        1.5e-10,
        1.2e-13,
        "finite differences",
        color="#e41a1c",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        4.0e-9,
        3.0e-16,
        "central differences",
        color="#377eb8",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_title(
        r"$\mathbf{Figure\ 8.2:\ Numerical\ Gradient\ Error\ vs\ Step\ Size\ \epsilon}$",
        fontsize=12,
        pad=10,
    )

    fig.tight_layout()
    _save_figure_files(
        fig,
        "fig_8_2_gradient_finite_differences",
        filepath=filepath,
        result_dirs=result_dirs,
        save_both=save_both,
    )
    return fig


def generate_figure_8_3(
    filepath: Optional[str] = None,
    result_dirs: Optional[List[str]] = None,
    save_both: bool = True,
) -> plt.Figure:
    """Figure 8.3: Illustration of a modular deep learning architecture."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")

    rect_green = patches.FancyBboxPatch(
        (3.0, 3.8),
        1.8,
        1.1,
        boxstyle="round,pad=0.08,rounding_size=0.2",
        facecolor="#c8e6c9",
        edgecolor="#2e7d32",
        linewidth=1.8,
        zorder=3,
    )
    ax.add_patch(rect_green)

    rect_purple = patches.FancyBboxPatch(
        (3.0, 1.3),
        1.8,
        1.1,
        boxstyle="round,pad=0.08,rounding_size=0.2",
        facecolor="#d1c4e9",
        edgecolor="#512da8",
        linewidth=1.8,
        zorder=3,
    )
    ax.add_patch(rect_purple)
    ax.text(
        3.9,
        1.85,
        r"$w$",
        ha="center",
        va="center",
        fontsize=16,
        fontstyle="italic",
        color="#111111",
        zorder=4,
    )

    rect_red = patches.FancyBboxPatch(
        (6.5, 2.5),
        1.8,
        1.2,
        boxstyle="round,pad=0.08,rounding_size=0.2",
        facecolor="#ffcdd2",
        edgecolor="#c62828",
        linewidth=1.8,
        zorder=3,
    )
    ax.add_patch(rect_red)

    ax.text(2.2, 4.35, r"$\mathbf{u}$", ha="center", va="center", fontsize=15, fontweight="bold")
    ax.annotate(
        "",
        xy=(2.95, 4.35),
        xytext=(2.45, 4.35),
        arrowprops=dict(arrowstyle="-|>", color="#111111", lw=2.0, mutation_scale=15),
        zorder=4,
    )

    ax.text(2.2, 1.85, r"$\mathbf{x}$", ha="center", va="center", fontsize=15, fontweight="bold")
    ax.annotate(
        "",
        xy=(2.95, 1.85),
        xytext=(2.45, 1.85),
        arrowprops=dict(arrowstyle="-|>", color="#111111", lw=2.0, mutation_scale=15),
        zorder=4,
    )

    ax.annotate(
        "",
        xy=(6.45, 3.4),
        xytext=(4.85, 4.35),
        arrowprops=dict(arrowstyle="-|>", color="#111111", lw=2.0, mutation_scale=15),
        zorder=4,
    )
    ax.text(5.8, 4.25, r"$\mathbf{v}$", ha="center", va="center", fontsize=14, fontweight="bold")

    ax.annotate(
        "",
        xy=(6.45, 2.8),
        xytext=(4.85, 1.85),
        arrowprops=dict(arrowstyle="-|>", color="#111111", lw=2.0, mutation_scale=15),
        zorder=4,
    )
    ax.text(5.8, 2.1, r"$\mathbf{z}$", ha="center", va="center", fontsize=14, fontweight="bold")

    ax.annotate(
        "",
        xy=(9.5, 3.1),
        xytext=(8.35, 3.1),
        arrowprops=dict(arrowstyle="-|>", color="#111111", lw=2.0, mutation_scale=15),
        zorder=4,
    )
    ax.text(9.8, 3.1, r"$\mathbf{y}$", ha="center", va="center", fontsize=15, fontweight="bold")

    # Red backward arrows
    ax.annotate(
        "",
        xy=(8.5, 2.75),
        xytext=(9.5, 2.75),
        arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.2, mutation_scale=15),
        zorder=4,
    )
    ax.text(
        9.0,
        2.35,
        r"$\frac{\partial E}{\partial y_k}$",
        ha="center",
        va="center",
        fontsize=13,
        color="#d62728",
    )

    ax.annotate(
        "",
        xy=(5.0, 1.6),
        xytext=(6.5, 2.5),
        arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.2, mutation_scale=15),
        zorder=4,
    )
    ax.text(
        7.35,
        1.7,
        r"$\frac{\partial y_k}{\partial z_j}$",
        ha="center",
        va="center",
        fontsize=13,
        color="#d62728",
    )

    ax.annotate(
        "",
        xy=(4.8, 1.35),
        xytext=(5.6, 1.35),
        arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.2, mutation_scale=15),
        zorder=4,
    )
    ax.text(
        5.2,
        0.95,
        r"$\frac{\partial E}{\partial z_j}$",
        ha="center",
        va="center",
        fontsize=13,
        color="#d62728",
    )

    ax.set_title(
        r"$\mathbf{Figure\ 8.3:\ Modular\ Deep\ Learning\ Architecture\ and\ Error\ Propagation}$",
        fontsize=12,
        pad=10,
    )

    fig.tight_layout()
    _save_figure_files(
        fig,
        "fig_8_3_modular_architecture_jacobian",
        filepath=filepath,
        result_dirs=result_dirs,
        save_both=save_both,
    )
    return fig


# Aliases for compatibility
plot_figure_8_1 = generate_figure_8_1
plot_figure_8_2 = generate_figure_8_2
plot_figure_8_3 = generate_figure_8_3


def generate_all_section_8_1_figures(
    result_dirs: Optional[List[str]] = None,
    save_both: bool = True,
) -> Dict[str, str]:
    """Generate and save all figures for Section 8.1 (Figures 8.1, 8.2, 8.3).

    Returns:
        dict with keys 'fig_8_1', 'fig_8_2', 'fig_8_3' mapping to output paths.
    """
    fig1 = generate_figure_8_1(result_dirs=result_dirs, save_both=save_both)
    fig2 = generate_figure_8_2(result_dirs=result_dirs, save_both=save_both)
    fig3 = generate_figure_8_3(result_dirs=result_dirs, save_both=save_both)

    root = _get_project_root()
    p1 = os.path.join(root, "8", "result", "fig_8_1_backprop_flow.png")
    p2 = os.path.join(root, "8", "result", "fig_8_2_gradient_finite_differences.png")
    p3 = os.path.join(root, "8", "result", "fig_8_3_modular_architecture_jacobian.png")

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

    return {
        "fig_8_1": p1,
        "fig_8_2": p2,
        "fig_8_3": p3,
    }
