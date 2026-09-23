"""
Chapter 6: Deep Neural Networks
Section 6.2: Multilayer Networks

Bishop & Bishop (2024), Deep Learning: Foundations and Concepts, pp. 180-186.

Covers:
- Forward propagation & parameter matrices (Section 6.2.1, Eq 6.7 - 6.12, Figure 6.9)
- Universal approximation theorem & multi-task fitting (Section 6.2.2, Figure 6.10)
- Classification decision surfaces & hidden unit hyperplanes (Section 6.2.2, Figure 6.11)
- Hidden unit activation functions & vanishing gradients (Section 6.2.3, Eq 6.13 - 6.18, Figure 6.12)
- Weight-space symmetries: sign-flip (2^M) and permutation (M!) symmetries (Section 6.2.4)
"""

import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import multivariate_normal

from common.plot_utils import save_plot, setup_style


def _get_project_root() -> str:
    """Return the absolute path to the repository root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _save_figure(fig: plt.Figure, filename: str, filepath: Optional[str] = None, save_both: bool = True) -> Tuple[str, Optional[str]]:
    """Save figure to target filepath, and/or both 6/result and root result directory."""
    if filepath:
        save_plot(fig, filepath)
    root = _get_project_root()
    path_ch6 = os.path.join(root, "6", "result", filename)
    path_root = os.path.join(root, "result", filename)
    if save_both or not filepath:
        if not filepath or os.path.abspath(filepath) != os.path.abspath(path_ch6):
            save_plot(fig, path_ch6)
        if not filepath or os.path.abspath(filepath) != os.path.abspath(path_root):
            save_plot(fig, path_root)
        return path_ch6, path_root
    return filepath, None


# ==============================================================================
# 1. Activation Functions & Derivatives (Section 6.2.3)
# ==============================================================================

def sigmoid(a: np.ndarray) -> np.ndarray:
    """Logistic sigmoid activation function: sigma(a) = 1 / (1 + exp(-a)).
    
    Eq (6.13).
    """
    a_arr = np.asarray(a, dtype=np.float64)
    return expit(a_arr)


def sigmoid_deriv(a: np.ndarray) -> np.ndarray:
    """Derivative of logistic sigmoid: d/da sigma(a) = sigma(a) * (1 - sigma(a))."""
    s = sigmoid(a)
    return s * (1.0 - s)


def tanh_act(a: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation function: tanh(a) = (e^a - e^-a) / (e^a + e^-a).
    
    Eq (6.14).
    """
    a_arr = np.asarray(a, dtype=np.float64)
    return np.tanh(a_arr)


def tanh_deriv(a: np.ndarray) -> np.ndarray:
    """Derivative of tanh: d/da tanh(a) = 1 - tanh^2(a)."""
    t = np.tanh(np.asarray(a, dtype=np.float64))
    return 1.0 - t**2


def hard_tanh(a: np.ndarray) -> np.ndarray:
    """Hard tanh activation function: h(a) = max(-1, min(1, a)).
    
    Collobert (2004), Eq (6.15).
    """
    a_arr = np.asarray(a, dtype=np.float64)
    return np.clip(a_arr, -1.0, 1.0)


def hard_tanh_deriv(a: np.ndarray) -> np.ndarray:
    """Derivative of hard tanh: 1 if |a| < 1 else 0."""
    a_arr = np.asarray(a, dtype=np.float64)
    return np.where(np.abs(a_arr) < 1.0, 1.0, 0.0)


def softplus(a: np.ndarray) -> np.ndarray:
    """Softplus activation function: h(a) = ln(1 + exp(a)).
    
    Eq (6.16). Smoothed version of ReLU (soft ReLU).
    """
    a_arr = np.asarray(a, dtype=np.float64)
    # Numerically stable: log(1 + exp(a)) = a + log(1 + exp(-a)) for a > 0
    return np.where(a_arr > 30.0, a_arr, np.log1p(np.exp(np.clip(a_arr, -50.0, 30.0))))


def softplus_deriv(a: np.ndarray) -> np.ndarray:
    """Derivative of softplus: d/da ln(1 + exp(a)) = exp(a) / (1 + exp(a)) = sigmoid(a)."""
    return sigmoid(a)


def relu(a: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit (ReLU): h(a) = max(0, a).
    
    Eq (6.17).
    """
    a_arr = np.asarray(a, dtype=np.float64)
    return np.maximum(0.0, a_arr)


def relu_deriv(a: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if a > 0 else 0 (subgradient 0 at a=0)."""
    a_arr = np.asarray(a, dtype=np.float64)
    return np.where(a_arr > 0.0, 1.0, 0.0)


def leaky_relu(a: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Leaky ReLU: h(a) = max(0, a) + min(0, alpha * a).
    
    Eq (6.18) with parameter alpha in (0, 1).
    """
    a_arr = np.asarray(a, dtype=np.float64)
    return np.maximum(0.0, a_arr) + np.minimum(0.0, float(alpha) * a_arr)


def leaky_relu_deriv(a: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Derivative of Leaky ReLU: 1 if a > 0 else alpha."""
    a_arr = np.asarray(a, dtype=np.float64)
    return np.where(a_arr > 0.0, 1.0, float(alpha))


def absolute_act(a: np.ndarray) -> np.ndarray:
    """Absolute value activation function: h(a) = |a|.
    
    Eq (6.18) with alpha = -1.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    return np.abs(a_arr)


def absolute_deriv(a: np.ndarray) -> np.ndarray:
    """Derivative of absolute value activation: sign(a)."""
    a_arr = np.asarray(a, dtype=np.float64)
    return np.sign(a_arr)


ACTIVATION_MAP: Dict[str, Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
    "tanh": (tanh_act, tanh_deriv),
    "hard_tanh": (hard_tanh, hard_tanh_deriv),
    "softplus": (softplus, softplus_deriv),
    "relu": (relu, relu_deriv),
    "leaky_relu": (leaky_relu, leaky_relu_deriv),
    "absolute": (absolute_act, absolute_deriv),
    "sigmoid": (sigmoid, sigmoid_deriv),
    "linear": (lambda a: np.asarray(a, dtype=np.float64), lambda a: np.ones_like(a, dtype=np.float64)),
}


# ==============================================================================
# 2. Two-Layer Multi-Layer Perceptron (Section 6.2.1, Eq 6.7 - 6.12)
# ==============================================================================

class TwoLayerMLP:
    """Two-layer Feed-Forward Neural Network (Bishop & Bishop 2024, Section 6.2).
    
    Computes:
        a_j^(1) = sum_{i=1}^D w_{ji}^{(1)} x_i + w_{j0}^{(1)}    (Eq 6.7)
        z_j^(1) = h(a_j^(1))                                      (Eq 6.8)
        a_k^(2) = sum_{j=1}^M w_{kj}^{(2)} z_j^(1) + w_{k0}^{(2)} (Eq 6.9)
        y_k     = f(a_k^(2))                                      (Eq 6.11)
        
    Vector form:
        y(x, w) = f(W^(2) h(W^(1) x))                            (Eq 6.12)
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: int,
        n_out: int,
        hidden_activation: str = "tanh",
        output_activation: str = "linear",
        random_state: Optional[int] = None,
    ):
        if n_in < 1 or n_hidden < 1 or n_out < 1:
            raise ValueError(f"Dimensions must be positive integers: n_in={n_in}, n_hidden={n_hidden}, n_out={n_out}")

        self.n_in = int(n_in)
        self.n_hidden = int(n_hidden)
        self.n_out = int(n_out)
        self.hidden_act_name = hidden_activation.lower()
        self.output_act_name = output_activation.lower()

        if self.hidden_act_name not in ACTIVATION_MAP:
            raise ValueError(f"Unsupported hidden activation: {hidden_activation}")
        if self.output_act_name not in ACTIVATION_MAP:
            raise ValueError(f"Unsupported output activation: {output_activation}")

        self.h_func, self.h_deriv = ACTIVATION_MAP[self.hidden_act_name]
        self.f_func, self.f_deriv = ACTIVATION_MAP[self.output_act_name]

        # Parameters
        self.W1: np.ndarray = np.zeros((self.n_hidden, self.n_in), dtype=np.float64)
        self.b1: np.ndarray = np.zeros(self.n_hidden, dtype=np.float64)
        self.W2: np.ndarray = np.zeros((self.n_out, self.n_hidden), dtype=np.float64)
        self.b2: np.ndarray = np.zeros(self.n_out, dtype=np.float64)

        self.init_weights(random_state=random_state)

    def init_weights(self, scale: float = 0.5, random_state: Optional[int] = None) -> "TwoLayerMLP":
        """Initialize weights with Gaussian random noise."""
        rng = np.random.default_rng(random_state)
        # Xavier/Glorot-like initialization
        s1 = np.sqrt(2.0 / (self.n_in + self.n_hidden)) * scale
        s2 = np.sqrt(2.0 / (self.n_hidden + self.n_out)) * scale
        self.W1 = rng.normal(0.0, s1, (self.n_hidden, self.n_in))
        self.b1 = np.zeros(self.n_hidden, dtype=np.float64)
        self.W2 = rng.normal(0.0, s2, (self.n_out, self.n_hidden))
        self.b2 = np.zeros(self.n_out, dtype=np.float64)
        return self

    def count_parameters(self) -> int:
        """Count total learnable parameters in the network.
        
        Layer 1: M * D weights + M biases = M * (D + 1)
        Layer 2: K * M weights + K biases = K * (M + 1)
        Total: (D + 1) * M + (M + 1) * K
        """
        return (self.n_in + 1) * self.n_hidden + (self.n_hidden + 1) * self.n_out

    def get_augmented_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return augmented weight matrices absorbing biases (Eq 6.10, 6.11).
        
        W1_aug: shape (M, D + 1) where column 0 is bias b1
        W2_aug: shape (K, M + 1) where column 0 is bias b2
        """
        W1_aug = np.column_stack([self.b1, self.W1])
        W2_aug = np.column_stack([self.b2, self.W2])
        return W1_aug, W2_aug

    def set_augmented_matrices(self, W1_aug: np.ndarray, W2_aug: np.ndarray) -> "TwoLayerMLP":
        """Set weights and biases from augmented parameter matrices."""
        W1_arr = np.asarray(W1_aug, dtype=np.float64)
        W2_arr = np.asarray(W2_aug, dtype=np.float64)
        if W1_arr.shape != (self.n_hidden, self.n_in + 1):
            raise ValueError(f"W1_aug shape {W1_arr.shape} does not match ({self.n_hidden}, {self.n_in + 1})")
        if W2_arr.shape != (self.n_out, self.n_hidden + 1):
            raise ValueError(f"W2_aug shape {W2_arr.shape} does not match ({self.n_out}, {self.n_hidden + 1})")
        self.b1 = W1_arr[:, 0].copy()
        self.W1 = W1_arr[:, 1:].copy()
        self.b2 = W2_arr[:, 0].copy()
        self.W2 = W2_arr[:, 1:].copy()
        return self

    def get_params_flat(self) -> np.ndarray:
        """Flatten all parameters into a 1D vector."""
        return np.concatenate([
            self.W1.ravel(),
            self.b1.ravel(),
            self.W2.ravel(),
            self.b2.ravel(),
        ])

    def set_params_flat(self, params: np.ndarray) -> "TwoLayerMLP":
        """Load parameters from a 1D vector."""
        p = np.asarray(params, dtype=np.float64).ravel()
        expected = self.count_parameters()
        if len(p) != expected:
            raise ValueError(f"Expected {expected} parameters, got {len(p)}")

        idx = 0
        w1_size = self.n_hidden * self.n_in
        self.W1 = p[idx : idx + w1_size].reshape(self.n_hidden, self.n_in).copy()
        idx += w1_size

        b1_size = self.n_hidden
        self.b1 = p[idx : idx + b1_size].copy()
        idx += b1_size

        w2_size = self.n_out * self.n_hidden
        self.W2 = p[idx : idx + w2_size].reshape(self.n_out, self.n_hidden).copy()
        idx += w2_size

        b2_size = self.n_out
        self.b2 = p[idx : idx + b2_size].copy()
        return self

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Forward propagation through the network (Eq 6.7 - 6.12).
        
        Args:
            X: Input array of shape (N, D) or (D,)
            
        Returns:
            y: Output array of shape (N, K) or (K,)
            cache: Tuple of (a1, z1, a2) internal activations
        """
        X_arr = np.asarray(X, dtype=np.float64)
        is_1d = (X_arr.ndim == 1)
        if is_1d:
            X_arr = X_arr.reshape(1, -1)

        if X_arr.shape[1] != self.n_in:
            raise ValueError(f"Input feature dimension {X_arr.shape[1]} does not match n_in={self.n_in}")

        # Layer 1: pre-activations a1 = X @ W1.T + b1 (Eq 6.7)
        a1 = X_arr @ self.W1.T + self.b1  # (N, M)
        # Layer 1: hidden unit activations z1 = h(a1) (Eq 6.8)
        z1 = self.h_func(a1)              # (N, M)

        # Layer 2: pre-activations a2 = z1 @ W2.T + b2 (Eq 6.9)
        a2 = z1 @ self.W2.T + self.b2     # (N, K)
        # Layer 2: network outputs y = f(a2) (Eq 6.11)
        y = self.f_func(a2)               # (N, K)

        if is_1d:
            return y.squeeze(0), (a1.squeeze(0), z1.squeeze(0), a2.squeeze(0))
        return y, (a1, z1, a2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output y for given inputs X."""
        y, _ = self.forward(X)
        return y

    def compute_loss_and_gradients(
        self,
        X: np.ndarray,
        t: np.ndarray,
        reg: float = 0.0,
    ) -> Tuple[float, np.ndarray]:
        """Compute objective function and analytical parameter gradients.
        
        Supports:
        - linear output: sum-of-squares error E = 0.5 * sum ||y - t||^2
        - sigmoid output: binary cross-entropy E = - sum [t ln y + (1 - t) ln(1 - y)]
        """
        X_arr = np.asarray(X, dtype=np.float64)
        t_arr = np.asarray(t, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if t_arr.ndim == 1:
            t_arr = t_arr.reshape(-1, self.n_out)

        N = len(X_arr)
        y, (a1, z1, a2) = self.forward(X_arr)

        if self.output_act_name == "linear":
            # Squared error loss: 0.5 * sum ||y - t||^2
            diff = y - t_arr
            loss = 0.5 * float(np.sum(diff**2))
            delta2 = diff  # dE / da2 = y - t
        elif self.output_act_name == "sigmoid":
            # Cross-entropy loss
            eps = 1e-12
            y_clamped = np.clip(y, eps, 1.0 - eps)
            loss = -float(np.sum(t_arr * np.log(y_clamped) + (1.0 - t_arr) * np.log(1.0 - y_clamped)))
            delta2 = y - t_arr  # dE / da2 = y - t (Eq 6.31)
        else:
            diff = y - t_arr
            loss = 0.5 * float(np.sum(diff**2))
            delta2 = diff * self.f_deriv(a2)

        # Regularization (L2 weight decay)
        if reg > 0:
            loss += 0.5 * reg * (float(np.sum(self.W1**2)) + float(np.sum(self.W2**2)))

        # Gradients for Layer 2:
        grad_W2 = delta2.T @ z1  # (K, M)
        grad_b2 = np.sum(delta2, axis=0)  # (K,)
        if reg > 0:
            grad_W2 += reg * self.W2

        # Backpropagation to Layer 1:
        # delta1 = (delta2 @ W2) * h'(a1)
        delta1 = (delta2 @ self.W2) * self.h_deriv(a1)  # (N, M)
        grad_W1 = delta1.T @ X_arr  # (M, D)
        grad_b1 = np.sum(delta1, axis=0)  # (M,)
        if reg > 0:
            grad_W1 += reg * self.W1

        grad_flat = np.concatenate([
            grad_W1.ravel(),
            grad_b1.ravel(),
            grad_W2.ravel(),
            grad_b2.ravel(),
        ])
        return loss, grad_flat

    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        reg: float = 0.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        n_restarts: int = 1,
        random_state: Optional[int] = None,
    ) -> "TwoLayerMLP":
        """Fit the two-layer network using L-BFGS optimization."""
        best_loss = float("inf")
        best_params = self.get_params_flat()

        rng = np.random.default_rng(random_state)

        for restart in range(n_restarts):
            if restart > 0:
                self.init_weights(random_state=int(rng.integers(0, 1000000)))

            p0 = self.get_params_flat()

            def obj_func(p: np.ndarray) -> Tuple[float, np.ndarray]:
                self.set_params_flat(p)
                return self.compute_loss_and_gradients(X, t, reg=reg)

            res = minimize(
                obj_func,
                p0,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
            )

            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x.copy()

        self.set_params_flat(best_params)
        return self


# ==============================================================================
# 3. Weight-Space Symmetries (Section 6.2.4, pp. 185-186)
# ==============================================================================

def compute_weight_space_symmetries(
    n_hidden: Union[int, Sequence[int]],
    is_odd_activation: bool = True,
) -> int:
    """Compute total weight-space symmetry factor (Bishop & Bishop 2024, pp. 185-186).
    
    For a single hidden layer of M units:
        Permutations: M!
        Sign-flips (for odd activation functions like tanh): 2^M
        Total: M! * 2^M
        
    For L hidden layers:
        Total = prod_{l=1}^L [ M_l! * 2^{M_l} ]
    """
    if isinstance(n_hidden, int):
        layers = [n_hidden]
    else:
        layers = list(n_hidden)

    total_factor = 1
    for M in layers:
        if M < 1:
            raise ValueError("Hidden units count must be at least 1")
        perms = math.factorial(M)
        flips = (2**M) if is_odd_activation else 1
        total_factor *= perms * flips
    return total_factor


def apply_hidden_unit_sign_flip(model: TwoLayerMLP, hidden_idx: int) -> TwoLayerMLP:
    """Apply sign-flip symmetry to hidden unit j in a two-layer network (p. 185).
    
    Transforms:
        w_{ji}^{(1)} -> -w_{ji}^{(1)}  (for all i=0..D)
        w_{kj}^{(2)} -> -w_{kj}^{(2)}  (for all k=1..K)
        
    For an odd activation function h(-a) = -h(a), this leaves the input-output mapping
    y(x, w) completely unchanged.
    """
    if hidden_idx < 0 or hidden_idx >= model.n_hidden:
        raise IndexError(f"Hidden unit index {hidden_idx} out of range [0, {model.n_hidden - 1}]")

    new_model = TwoLayerMLP(
        model.n_in, model.n_hidden, model.n_out,
        hidden_activation=model.hidden_act_name,
        output_activation=model.output_act_name,
    )
    new_model.W1 = model.W1.copy()
    new_model.b1 = model.b1.copy()
    new_model.W2 = model.W2.copy()
    new_model.b2 = model.b2.copy()

    # Flip incoming weights and bias
    new_model.W1[hidden_idx, :] *= -1.0
    new_model.b1[hidden_idx] *= -1.0
    # Flip outgoing weights
    new_model.W2[:, hidden_idx] *= -1.0

    return new_model


def apply_hidden_unit_permutation(model: TwoLayerMLP, perm: Sequence[int]) -> TwoLayerMLP:
    """Apply permutation symmetry to hidden units in a two-layer network (p. 186).
    
    Permutes hidden units according to permutation sequence `perm` of length M.
    """
    perm_list = list(perm)
    M = model.n_hidden
    if len(perm_list) != M or set(perm_list) != set(range(M)):
        raise ValueError(f"Invalid permutation of length {M}: {perm}")

    new_model = TwoLayerMLP(
        model.n_in, model.n_hidden, model.n_out,
        hidden_activation=model.hidden_act_name,
        output_activation=model.output_act_name,
    )
    new_model.W1 = model.W1[perm_list, :].copy()
    new_model.b1 = model.b1[perm_list].copy()
    new_model.W2 = model.W2[:, perm_list].copy()
    new_model.b2 = model.b2.copy()

    return new_model


def verify_weight_space_symmetry(
    model: TwoLayerMLP,
    X: np.ndarray,
    tol: float = 1e-12,
) -> Dict[str, Union[bool, float]]:
    """Verify that sign-flip and permutation transformations preserve output identically."""
    y_orig = model.predict(X)

    # 1. Test all sign-flips
    max_flip_err = 0.0
    for j in range(model.n_hidden):
        model_flipped = apply_hidden_unit_sign_flip(model, j)
        y_flipped = model_flipped.predict(X)
        err = float(np.max(np.abs(y_orig - y_flipped)))
        if err > max_flip_err:
            max_flip_err = err

    # 2. Test cyclic permutation
    M = model.n_hidden
    perm = [(i + 1) % M for i in range(M)]
    model_perm = apply_hidden_unit_permutation(model, perm)
    y_perm = model_perm.predict(X)
    perm_err = float(np.max(np.abs(y_orig - y_perm)))

    return {
        "sign_flip_max_diff": max_flip_err,
        "sign_flip_invariant": bool(max_flip_err < tol),
        "permutation_diff": perm_err,
        "permutation_invariant": bool(perm_err < tol),
        "total_symmetries": compute_weight_space_symmetries(M, is_odd_activation=(model.hidden_act_name == "tanh")),
    }


# ==============================================================================
# 4. Universal Approximation Helpers (Section 6.2.2)
# ==============================================================================

def generate_approximation_data(
    func_name: str,
    n_samples: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 50 uniformly sampled data points over (-1, 1) as in Figure 6.10.
    
    Functions:
    - 'x^2': f(x) = x^2
    - 'sin': f(x) = sin(0.8 * pi * x) (sine wave peaking at x ~ 0.625)
    - 'abs': f(x) = |x|
    - 'heaviside': f(x) = H(x) = 1 if x >= 0 else 0
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1.0, 1.0, n_samples))

    fn = func_name.lower()
    if fn in ["x^2", "x2", "quadratic"]:
        y = x**2
    elif fn in ["sin", "sine", "sin(x)"]:
        y = np.sin(0.8 * np.pi * x)
    elif fn in ["abs", "|x|", "absolute"]:
        y = np.abs(x)
    elif fn in ["heaviside", "h(x)", "step"]:
        y = np.where(x >= 0.0, 1.0, 0.0)
    else:
        raise ValueError(f"Unknown target function: {func_name}")

    return x.reshape(-1, 1), y.reshape(-1, 1)


# ==============================================================================
# 5. Figure Reproductions (Figures 6.9, 6.10, 6.11, 6.12)
# ==============================================================================

def generate_figure_6_9(filepath: Optional[str] = None, save_both: bool = True) -> Tuple[plt.Figure, Tuple[str, Optional[str]]]:
    """Faithful reproduction of Figure 6.9: Network diagram for a two-layer neural network.
    
    Displays:
    - Input nodes x0, x1, ..., xD (light blue fill, dark blue border)
    - Hidden nodes z0, z1, ..., zM (z0 is solid dark blue fill)
    - Output nodes y1, ..., yK
    - Information flow arrows from inputs to hidden and hidden to outputs
    - Specific parameter labels: w_{MD}^{(1)}, w_{10}^{(1)}, w_{KM}^{(2)}, w_{10}^{(2)}
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 8.5)
    ax.axis("off")

    x_in = 1.8
    x_hid = 5.2
    x_out = 8.6

    # Column headers
    ax.text(x_in, 8.0, "Inputs", ha="center", va="center", fontsize=15, fontweight="medium")
    ax.text(x_hid, 8.0, "Hidden units", ha="center", va="center", fontsize=15, fontweight="medium")
    ax.text(x_out, 8.0, "Outputs", ha="center", va="center", fontsize=15, fontweight="medium")

    # Node positions
    y_in = {"xD": 6.5, "x1": 3.8, "x0": 2.0}
    y_hid = {"zM": 6.8, "z1": 2.6, "z0": 0.8}
    y_out = {"yK": 6.5, "y1": 3.8}

    r = 0.44

    # Connect inputs to hidden (z1 and zM)
    for yin in y_in.values():
        for yhid in [y_hid["zM"], y_hid["z1"]]:
            ax.annotate("", xy=(x_hid - r, yhid), xytext=(x_in + r, yin),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=12))

    # Connect hidden (z0, z1, zM) to outputs (y1 and yK)
    for yhid in y_hid.values():
        for yout in y_out.values():
            ax.annotate("", xy=(x_out - r, yout), xytext=(x_hid + r, yhid),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=12))

    # Dots
    ax.text(x_in, 5.15, r"$\vdots$", ha="center", va="center", fontsize=18)
    ax.text(x_hid, 4.7, r"$\vdots$", ha="center", va="center", fontsize=18)
    ax.text(x_out, 5.15, r"$\vdots$", ha="center", va="center", fontsize=18)

    # Input nodes
    for name, yin in y_in.items():
        c = plt.Circle((x_in, yin), r, edgecolor="#0000CC", facecolor="#E6ECFF", lw=1.8, zorder=5)
        ax.add_patch(c)
        sub = name[1:]
        ax.text(x_in, yin, rf"$x_{{{sub}}}$", ha="center", va="center", fontsize=14, zorder=6)

    # Hidden nodes
    for name, yhid in y_hid.items():
        if name == "z0":
            c = plt.Circle((x_hid, yhid), r, edgecolor="#000088", facecolor="#0000CC", lw=1.8, zorder=5)
            ax.add_patch(c)
            ax.text(x_hid, yhid - 0.72, r"$z_0$", ha="center", va="center", fontsize=14, zorder=6)
        else:
            c = plt.Circle((x_hid, yhid), r, edgecolor="#0000CC", facecolor="#E6ECFF", lw=1.8, zorder=5)
            ax.add_patch(c)
            sub = name[1:]
            ax.text(x_hid, yhid, rf"$z_{{{sub}}}$", ha="center", va="center", fontsize=14, zorder=6)

    # Output nodes
    for name, yout in y_out.items():
        c = plt.Circle((x_out, yout), r, edgecolor="#0000CC", facecolor="#E6ECFF", lw=1.8, zorder=5)
        ax.add_patch(c)
        sub = name[1:]
        ax.text(x_out, yout, rf"$y_{{{sub}}}$", ha="center", va="center", fontsize=14, zorder=6)

    # Weight labels
    ax.text(3.35, 7.05, r"$w_{MD}^{(1)}$", fontsize=13, ha="center")
    ax.text(3.5, 1.8, r"$w_{10}^{(1)}$", fontsize=13, ha="center")
    ax.text(7.0, 7.05, r"$w_{KM}^{(2)}$", fontsize=13, ha="center")
    ax.text(7.0, 1.8, r"$w_{10}^{(2)}$", fontsize=13, ha="center")

    fig.tight_layout()
    paths = _save_figure(fig, "fig_6_9_two_layer_network.png", filepath=filepath, save_both=save_both)
    return fig, paths


def generate_figure_6_10(filepath: Optional[str] = None, save_both: bool = True) -> Tuple[plt.Figure, Tuple[str, Optional[str]]]:
    """Faithful reproduction of Figure 6.10: Universal approximation capability.
    
    Approximates 4 functions:
    (a) f(x) = x^2
    (b) f(x) = sin(x) (sin(0.8 * pi * x))
    (c) f(x) = |x|
    (d) f(x) = H(x) (Heaviside step function)
    with N=50 data points (blue dots) and 2-layer network with 3 hidden units (tanh).
    Plots:
    - Red curve: Network prediction y(x)
    - Three dashed curves: individual hidden unit outputs z_j(x) in [-1, 1]
    - Blue dots: training data
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2))

    x_plot = np.linspace(-1.0, 1.0, 300).reshape(-1, 1)
    func_configs = [
        ("x^2", "(a)", 42),
        ("sin", "(b)", 43),
        ("abs", "(c)", 44),
        ("heaviside", "(d)", 45),
    ]

    hidden_colors = ["#2CA02C", "#E377C2", "#BCBD22"]  # green, magenta, yellow-olive

    for ax, (fname, sublabel, s) in zip(axes.flat, func_configs):
        X_tr, y_tr = generate_approximation_data(fname, n_samples=50, seed=s)

        mlp = TwoLayerMLP(n_in=1, n_hidden=3, n_out=1, hidden_activation="tanh", output_activation="linear")
        
        # Fit network with high convergence accuracy
        if fname == "heaviside":
            # Direct sharp step initialization
            mlp.W1 = np.array([[30.0], [0.1], [-0.1]])
            mlp.b1 = np.array([0.0, 5.0, -5.0])
            mlp.W2 = np.array([[0.5, 0.0, 0.0]])
            mlp.b2 = np.array([0.5])
            mlp.fit(X_tr, y_tr, max_iter=200, tol=1e-7, n_restarts=1)
        else:
            mlp.fit(X_tr, y_tr, max_iter=800, tol=1e-8, n_restarts=20, random_state=s)

        y_pred, (a1_pl, z1_pl, a2_pl) = mlp.forward(x_plot)

        # Plot data points
        ax.scatter(X_tr.ravel(), y_tr.ravel(), color="blue", s=10, zorder=5)
        # Plot network function
        ax.plot(x_plot.ravel(), y_pred.ravel(), color="red", lw=2.0, zorder=4)
        # Plot individual hidden units (dashed lines)
        for j in range(3):
            ax.plot(x_plot.ravel(), z1_pl[:, j], color=hidden_colors[j], linestyle="--", lw=1.5, zorder=3)

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(sublabel, fontsize=13)
        ax.set_aspect("equal")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)

    fig.tight_layout()
    paths = _save_figure(fig, "fig_6_10_universal_approximation.png", filepath=filepath, save_both=save_both)
    return fig, paths


def generate_figure_6_11(filepath: Optional[str] = None, save_both: bool = True) -> Tuple[plt.Figure, Tuple[str, Optional[str]]]:
    """Faithful reproduction of Figure 6.11: Two-class classification with 2 hidden units.
    
    Shows:
    - Synthetic 2D data: Class 0 (blue circles), Class 1 (red crosses)
    - Dashed blue lines: z_j = 0.5 contours for each of the two hidden units
    - Red line: y = 0.5 decision surface of the two-layer network
    - Green lines: optimal Bayes decision boundary from generative distribution
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.grid(False)

    # Class 0: mixture of two Gaussians
    mu0_1 = np.array([-0.6, 0.8])
    cov0_1 = np.array([[0.35, -0.1], [-0.1, 0.4]])
    mu0_2 = np.array([0.2, -0.9])
    cov0_2 = np.array([[0.25, 0.05], [0.05, 0.45]])

    # Class 1: mixture of two Gaussians
    mu1_1 = np.array([0.8, 1.4])
    cov1_1 = np.array([[0.2, 0.08], [0.08, 0.35]])
    mu1_2 = np.array([1.1, -0.8])
    cov1_2 = np.array([[0.25, -0.05], [-0.05, 0.3]])

    np.random.seed(123)
    N_sub = 25
    X0 = np.vstack([
        np.random.multivariate_normal(mu0_1, cov0_1, N_sub),
        np.random.multivariate_normal(mu0_2, cov0_2, N_sub),
    ])
    X1 = np.vstack([
        np.random.multivariate_normal(mu1_1, cov1_1, N_sub),
        np.random.multivariate_normal(mu1_2, cov1_2, N_sub),
    ])

    # Grid for contour evaluation
    gx = np.linspace(-2.5, 2.5, 300)
    gy = np.linspace(-2.5, 3.0, 300)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
    pos = np.dstack((GX, GY))

    # Optimal Bayes posterior boundary
    p_c0 = 0.5 * multivariate_normal.pdf(pos, mu0_1, cov0_1) + 0.5 * multivariate_normal.pdf(pos, mu0_2, cov0_2)
    p_c1 = 0.5 * multivariate_normal.pdf(pos, mu1_1, cov1_1) + 0.5 * multivariate_normal.pdf(pos, mu1_2, cov1_2)
    bayes_post = p_c1 / (p_c0 + p_c1 + 1e-15)

    # Exact weights matching textbook Figure 6.11 geometry:
    # Hidden unit 1 defines Line 1 (slope ~ -0.34)
    # Hidden unit 2 defines Line 2 (slope ~ 0.90)
    s1, s2 = 0.70, 0.70
    W1 = np.array([[0.35 * s1, 1.0 * s1], [-0.90 * s2, 1.0 * s2]])
    b1 = np.array([0.5493 - 0.20 * s1, 0.5493 - 0.45 * s2])
    w2 = np.array([-5.2, 5.0])
    b2 = 0.65

    mlp = TwoLayerMLP(n_in=2, n_hidden=2, n_out=1, hidden_activation="tanh", output_activation="sigmoid")
    mlp.W1 = W1
    mlp.b1 = b1
    mlp.W2 = w2.reshape(1, 2)
    mlp.b2 = np.array([b2])

    y_pred, (a1_g, z1_g, a2_g) = mlp.forward(grid_pts)
    Y_surf = y_pred.reshape(GX.shape)
    Z1_surf = z1_g[:, 0].reshape(GX.shape)
    Z2_surf = z1_g[:, 1].reshape(GX.shape)

    # 1. Scatter data points
    ax.scatter(X0[:, 0], X0[:, 1], facecolors="none", edgecolors="blue", s=45, lw=1.5, zorder=5, label="Class 1")
    ax.scatter(X1[:, 0], X1[:, 1], color="red", marker="x", s=45, lw=1.8, zorder=5, label="Class 2")

    # 2. Hidden unit contours z_j = 0.5 (dashed blue lines)
    ax.contour(GX, GY, Z1_surf, levels=[0.5], colors="blue", linestyles="--", linewidths=1.3, zorder=4)
    ax.contour(GX, GY, Z2_surf, levels=[0.5], colors="blue", linestyles="--", linewidths=1.3, zorder=4)

    # 3. Network decision boundary y = 0.5 (solid red line)
    ax.contour(GX, GY, Y_surf, levels=[0.5], colors="red", linewidths=1.8, zorder=6)

    # 4. Optimal Bayes boundary (green line)
    ax.contour(GX, GY, bayes_post, levels=[0.5], colors="#00CC00", linewidths=1.5, zorder=3)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 3.0)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2, 3])
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    paths = _save_figure(fig, "fig_6_11_classification_hidden_units.png", filepath=filepath, save_both=save_both)
    return fig, paths


def generate_figure_6_12(filepath: Optional[str] = None, save_both: bool = True) -> Tuple[plt.Figure, Tuple[str, Optional[str]]]:
    """Faithful reproduction of Figure 6.12: A variety of nonlinear activation functions.
    
    Plots:
    (a) tanh: h(a) = tanh(a) (Eq 6.14)
    (b) hard tanh: h(a) = max(-1, min(1, a)) (Eq 6.15)
    (c) softplus: h(a) = ln(1 + exp(a)) (Eq 6.16)
    (d) ReLU: h(a) = max(0, a) (Eq 6.17)
    (e) leaky ReLU: h(a) = max(0, a) + min(0, 0.2 a) (Eq 6.18)
    (f) absolute: h(a) = |a| (Eq 6.18 with alpha = -1)
    """
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    a = np.linspace(-2.5, 2.5, 400)

    funcs = [
        ("tanh", tanh_act(a), "(a)"),
        ("hard tanh", hard_tanh(a), "(b)"),
        ("softplus", softplus(a), "(c)"),
        ("ReLU", relu(a), "(d)"),
        ("leaky ReLU", leaky_relu(a, alpha=0.2), "(e)"),
        ("absolute", absolute_act(a), "(f)"),
    ]

    for ax, (name, y, sublabel) in zip(axes.flat, funcs):
        ax.plot(a, y, color="red", lw=2.0)
        ax.axhline(0, color="gray", linestyle="--", lw=1.0)
        ax.axvline(0, color="gray", linestyle="--", lw=1.0)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xticks([-2.5, 0.0, 2.5])
        ax.set_yticks([-2.5, 0.0, 2.5])
        ax.tick_params(direction="in", top=True, right=True)
        ax.text(2.35, -2.25, name, ha="right", va="bottom", fontsize=12)
        ax.set_xlabel(sublabel, fontsize=12)
        ax.set_aspect("equal")

    fig.tight_layout()
    paths = _save_figure(fig, "fig_6_12_activation_functions.png", filepath=filepath, save_both=save_both)
    return fig, paths


def generate_all_section_6_2_figures() -> Dict[str, Tuple[plt.Figure, Tuple[str, Optional[str]]]]:
    """Generate and save all Section 6.2 figures (Figures 6.9 through 6.12)."""
    return {
        "fig_6_9": generate_figure_6_9(),
        "fig_6_10": generate_figure_6_10(),
        "fig_6_11": generate_figure_6_11(),
        "fig_6_12": generate_figure_6_12(),
    }
