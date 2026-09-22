"""Chapter 5: Single-layer Networks: Classification
Section 5.4: Discriminative Classifiers

This module implements the core algorithms, activation functions, link functions,
and figure reproduction functions for Section 5.4 of:
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.

Contents:
- Activation and Link functions:
  * Logistic sigmoid & logit (inverse sigmoid) (Eq 5.42, 5.44, 5.71, 5.72)
  * Softmax & its Jacobian derivative (Eq 5.76, 5.78)
  * Probit function, normal CDF, and error function (Eq 5.86, 5.87, 5.88)
- Feature transformations:
  * Gaussian radial basis functions (RBF) with bias unit (Figure 5.15)
- Models:
  * Binary Logistic Regression with IRLS (Iterative Reweighted Least Squares)
    and Gradient Descent (Eq 5.71 - 5.75)
  * Multi-class Softmax Regression (Eq 5.76 - 5.82)
  * Probit Regression for binary classification (Eq 5.83 - 5.88)
  * Canonical Link Functions analysis for Exponential Family GLMs (Eq 5.89 - 5.95)
- Figure reproductions:
  * Figure 5.15: Role of nonlinear basis functions in classification
  * Figure 5.16: Single-layer neural network architecture for multi-class classification
  * Figure 5.17: Cumulative distribution function and noisy threshold model
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from scipy.stats import norm

from common.plot_utils import save_plot, setup_style


def _get_project_root() -> str:
    """Return the absolute path to the repository root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _save_figure(fig: plt.Figure, filename: str, filepath: Optional[str] = None, save_both: bool = True) -> Tuple[str, Optional[str]]:
    """Save figure to target filepath, and/or both 5/result and root result directory."""
    if filepath:
        save_plot(fig, filepath)
    root = _get_project_root()
    path_ch5 = os.path.join(root, "5", "result", filename)
    path_root = os.path.join(root, "result", filename)
    if save_both or not filepath:
        if not filepath or os.path.abspath(filepath) != os.path.abspath(path_ch5):
            save_plot(fig, path_ch5)
        if not filepath or os.path.abspath(filepath) != os.path.abspath(path_root):
            save_plot(fig, path_root)
        return path_ch5, path_root
    return filepath, None


# ==============================================================================
# 1. Activation and Link Functions
# ==============================================================================

def sigmoid(a: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid function: sigma(a) = 1 / (1 + exp(-a)).

    Eq (5.42), (5.71).
    """
    a = np.asarray(a, dtype=np.float64)
    # Clip to prevent overflow in exp
    a_clipped = np.clip(a, -500.0, 500.0)
    return np.where(a_clipped >= 0,
                    1.0 / (1.0 + np.exp(-a_clipped)),
                    np.exp(a_clipped) / (1.0 + np.exp(a_clipped)))


def sigmoid_deriv(a: np.ndarray) -> np.ndarray:
    """Derivative of the logistic sigmoid: d sigma / da = sigma(a) * (1 - sigma(a)).

    Eq (5.72).
    """
    s = sigmoid(a)
    return s * (1.0 - s)


def logit(p: np.ndarray) -> np.ndarray:
    """Inverse of the logistic sigmoid (logit link function): ln(p / (1 - p)).

    Eq (5.44).
    """
    p = np.asarray(p, dtype=np.float64)
    p_clipped = np.clip(p, 1e-15, 1.0 - 1e-15)
    return np.log(p_clipped / (1.0 - p_clipped))


def softmax(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax activation function.

    yk = exp(ak) / sum_j exp(aj)
    Eq (5.76).
    """
    a = np.asarray(a, dtype=np.float64)
    a_max = np.max(a, axis=axis, keepdims=True)
    exp_a = np.exp(a - a_max)
    return exp_a / np.sum(exp_a, axis=axis, keepdims=True)


def softmax_jacobian(y: np.ndarray) -> np.ndarray:
    """Jacobian matrix of softmax outputs w.r.t pre-activations: dyk / daj = yk * (Ikj - yj).

    Eq (5.78).
    y: 1D array of shape (K,)
    Returns: (K, K) Jacobian matrix.
    """
    y = np.asarray(y, dtype=np.float64)
    return np.diag(y) - np.outer(y, y)


def erf_func(a: np.ndarray) -> np.ndarray:
    """Error function erf(a) = (2 / sqrt(pi)) * int_0^a exp(-theta^2) dtheta.

    Eq (5.87).
    """
    return special.erf(a)


def probit(a: np.ndarray) -> np.ndarray:
    """Probit activation function: cumulative distribution function of standard normal.

    Phi(a) = int_{-inf}^a N(theta | 0, 1) dtheta = 0.5 * (1 + erf(a / sqrt(2)))
    Eq (5.86), (5.88).
    """
    a = np.asarray(a, dtype=np.float64)
    return 0.5 * (1.0 + special.erf(a / np.sqrt(2.0)))


def probit_deriv(a: np.ndarray) -> np.ndarray:
    """Derivative of the probit function: dPhi / da = N(a | 0, 1)."""
    a = np.asarray(a, dtype=np.float64)
    return norm.pdf(a, loc=0.0, scale=1.0)


# ==============================================================================
# 2. Basis Functions
# ==============================================================================

class GaussianBasisFunctions:
    """Gaussian Radial Basis Functions (RBF) with constant bias phi_0(x) = 1.

    phi_j(x) = exp(- ||x - mu_j||^2 / (2 * s_j^2))
    Section 5.4.2, Figure 5.15.
    """

    def __init__(self, centers: np.ndarray, scales: Union[float, np.ndarray], include_bias: bool = True):
        self.centers = np.atleast_2d(np.asarray(centers, dtype=np.float64))
        self.M = len(self.centers)
        if np.isscalar(scales):
            self.scales = np.full(self.M, float(scales))
        else:
            self.scales = np.asarray(scales, dtype=np.float64)
            if len(self.scales) == 1:
                self.scales = np.full(self.M, self.scales[0])
        self.include_bias = include_bias

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform input matrix X of shape (N, D) into feature matrix Phi of shape (N, M + bias)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        N = X.shape[0]
        phi_list = []
        if self.include_bias:
            phi_list.append(np.ones((N, 1)))

        for j in range(self.M):
            diff = X - self.centers[j]
            dist_sq = np.sum(diff ** 2, axis=1, keepdims=True)
            val = np.exp(-dist_sq / (2.0 * self.scales[j] ** 2))
            phi_list.append(val)

        return np.hstack(phi_list)

    @property
    def num_features(self) -> int:
        return self.M + (1 if self.include_bias else 0)


# ==============================================================================
# 3. Models
# ==============================================================================

class LogisticRegression:
    """Binary Logistic Regression with cross-entropy error.

    Supports:
    - Iterative Reweighted Least Squares (IRLS / Newton-Raphson) (Section 5.4.3, p. 160)
    - Batch Gradient Descent (Eq 5.75)
    - L2 Regularization (weight decay) to prevent divergence on linearly separable data.
    """

    def __init__(self, reg: float = 0.0, fit_intercept: bool = True):
        self.reg = float(reg)
        self.fit_intercept = fit_intercept
        self.w: Optional[np.ndarray] = None
        self.loss_history: List[float] = []

    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self.fit_intercept:
            N = X.shape[0]
            return np.hstack([np.ones((N, 1)), X])
        return X

    def compute_loss(self, Phi: np.ndarray, t: np.ndarray, w: Optional[np.ndarray] = None) -> float:
        """Cross-entropy error function E(w) + 0.5 * reg * ||w||^2.

        Eq (5.74).
        """
        if w is None:
            w = self.w
        if w is None:
            raise ValueError("Model is not fitted yet.")

        t = np.asarray(t, dtype=np.float64)
        a = Phi @ w
        y = sigmoid(a)
        eps = 1e-15
        y = np.clip(y, eps, 1.0 - eps)
        # Negative log-likelihood
        loss = -np.sum(t * np.log(y) + (1.0 - t) * np.log(1.0 - y))
        # Regularization (exclude bias if fit_intercept is True)
        if self.reg > 0:
            reg_w = w[1:] if self.fit_intercept else w
            loss += 0.5 * self.reg * np.sum(reg_w ** 2)
        return float(loss)

    def compute_gradient(self, Phi: np.ndarray, t: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        """Gradient of cross-entropy error: grad = Phi^T (y - t) + reg * w.

        Eq (5.75).
        """
        if w is None:
            w = self.w
        if w is None:
            raise ValueError("Model is not fitted yet.")

        t = np.asarray(t, dtype=np.float64)
        y = sigmoid(Phi @ w)
        grad = Phi.T @ (y - t)
        if self.reg > 0:
            reg_vec = np.copy(w)
            if self.fit_intercept:
                reg_vec[0] = 0.0
            grad += self.reg * reg_vec
        return grad

    def compute_hessian(self, Phi: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        """Hessian of cross-entropy error: H = Phi^T R Phi + reg * I.

        Where R_nn = y_n * (1 - y_n).
        """
        if w is None:
            w = self.w
        if w is None:
            raise ValueError("Model is not fitted yet.")

        y = sigmoid(Phi @ w)
        r = y * (1.0 - y)
        # Phi^T R Phi = (Phi * r[:, None])^T Phi
        H = (Phi * r[:, np.newaxis]).T @ Phi
        if self.reg > 0:
            reg_mat = np.eye(len(w)) * self.reg
            if self.fit_intercept:
                reg_mat[0, 0] = 0.0
            H += reg_mat
        return H

    def fit(self, X: np.ndarray, t: np.ndarray,
            method: str = "irls",
            lr: float = 0.1,
            max_iter: int = 100,
            tol: float = 1e-6) -> "LogisticRegression":
        """Fit logistic regression model using IRLS or Gradient Descent."""
        Phi = self._prepare_features(X)
        N, D = Phi.shape
        t = np.asarray(t, dtype=np.float64).ravel()
        if len(t) != N:
            raise ValueError(f"Target length {len(t)} does not match samples {N}")

        self.w = np.zeros(D, dtype=np.float64)
        self.loss_history = []

        for step in range(max_iter):
            loss = self.compute_loss(Phi, t, self.w)
            self.loss_history.append(loss)

            grad = self.compute_gradient(Phi, t, self.w)
            if np.linalg.norm(grad) < tol:
                break

            if method.lower() == "irls":
                H = self.compute_hessian(Phi, self.w)
                # Solve H delta = grad with small damping for numerical stability
                try:
                    delta = np.linalg.solve(H + 1e-8 * np.eye(D), grad)
                except np.linalg.LinAlgError:
                    delta = np.linalg.pinv(H) @ grad
                w_new = self.w - delta
            elif method.lower() in ("gd", "gradient_descent"):
                w_new = self.w - lr * grad
            else:
                raise ValueError(f"Unknown method {method}. Choose 'irls' or 'gd'.")

            if np.linalg.norm(w_new - self.w) < tol:
                self.w = w_new
                break
            self.w = w_new

        self.loss_history.append(self.compute_loss(Phi, t, self.w))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior probability p(C1 | x) = sigma(w^T phi(x))."""
        if self.w is None:
            raise ValueError("Model is not fitted.")
        Phi = self._prepare_features(X)
        return sigmoid(Phi @ self.w)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels {0, 1}."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def score(self, X: np.ndarray, t: np.ndarray) -> float:
        """Compute binary classification accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(t, dtype=int).ravel()))


class SoftmaxRegression:
    """Multi-class Logistic Regression (Softmax Regression).

    Section 5.4.4, Eq (5.76) - (5.82).
    Model: y_k = exp(a_k) / sum_j exp(a_j), where a_k = w_k^T phi.
    Loss: Cross-entropy E(w_1, ..., w_K) = - sum_{n=1}^N sum_{k=1}^K t_{nk} ln y_{nk}.
    Gradient: grad_{w_j} E = sum_{n=1}^N (y_{nj} - t_{nj}) phi_n.
    """

    def __init__(self, reg: float = 0.0, fit_intercept: bool = True):
        self.reg = float(reg)
        self.fit_intercept = fit_intercept
        self.W: Optional[np.ndarray] = None  # Shape (D, K)
        self.K: Optional[int] = None
        self.loss_history: List[float] = []

    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self.fit_intercept:
            N = X.shape[0]
            return np.hstack([np.ones((N, 1)), X])
        return X

    def _to_one_hot(self, t: np.ndarray, K: int) -> np.ndarray:
        t = np.asarray(t, dtype=int).ravel()
        T = np.zeros((len(t), K), dtype=np.float64)
        for i, val in enumerate(t):
            T[i, val] = 1.0
        return T

    def compute_loss(self, Phi: np.ndarray, T: np.ndarray, W: Optional[np.ndarray] = None) -> float:
        """Compute multi-class cross-entropy error.

        Eq (5.80).
        """
        if W is None:
            W = self.W
        if W is None:
            raise ValueError("Model is not fitted.")

        A = Phi @ W
        Y = softmax(A, axis=-1)
        eps = 1e-15
        Y = np.clip(Y, eps, 1.0 - eps)
        loss = -np.sum(T * np.log(Y))
        if self.reg > 0:
            reg_W = W[1:, :] if self.fit_intercept else W
            loss += 0.5 * self.reg * np.sum(reg_W ** 2)
        return float(loss)

    def compute_gradient(self, Phi: np.ndarray, T: np.ndarray, W: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute gradient of multi-class cross-entropy w.r.t parameter matrix W (D, K).

        Eq (5.81), (5.82).
        """
        if W is None:
            W = self.W
        if W is None:
            raise ValueError("Model is not fitted.")

        A = Phi @ W
        Y = softmax(A, axis=-1)
        grad = Phi.T @ (Y - T)  # Shape (D, K)
        if self.reg > 0:
            reg_mat = np.copy(W)
            if self.fit_intercept:
                reg_mat[0, :] = 0.0
            grad += self.reg * reg_mat
        return grad

    def fit(self, X: np.ndarray, t: np.ndarray,
            lr: float = 0.05,
            max_iter: int = 500,
            tol: float = 1e-6) -> "SoftmaxRegression":
        """Fit multi-class softmax regression using gradient descent."""
        Phi = self._prepare_features(X)
        N, D = Phi.shape

        if np.asarray(t).ndim == 1:
            self.K = int(np.max(t) + 1)
            T = self._to_one_hot(t, self.K)
        else:
            T = np.asarray(t, dtype=np.float64)
            self.K = T.shape[1]

        self.W = np.zeros((D, self.K), dtype=np.float64)
        self.loss_history = []

        for step in range(max_iter):
            loss = self.compute_loss(Phi, T, self.W)
            self.loss_history.append(loss)

            grad = self.compute_gradient(Phi, T, self.W)
            if np.linalg.norm(grad) < tol:
                break

            self.W -= lr * grad

        self.loss_history.append(self.compute_loss(Phi, T, self.W))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior class probabilities p(C_k | x)."""
        if self.W is None:
            raise ValueError("Model is not fitted.")
        Phi = self._prepare_features(X)
        return softmax(Phi @ self.W, axis=-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class index k with largest probability."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=-1)

    def score(self, X: np.ndarray, t: np.ndarray) -> float:
        """Compute multi-class classification accuracy."""
        t_arr = np.asarray(t).ravel()
        return float(np.mean(self.predict(X) == t_arr))


class ProbitRegression:
    """Binary Probit Regression with Gaussian CDF activation function.

    Section 5.4.5, Eq (5.83) - (5.88).
    Model: p(t=1 | a) = Phi(a) = int_{-inf}^a N(theta | 0, 1) dtheta.
    """

    def __init__(self, reg: float = 0.0, fit_intercept: bool = True):
        self.reg = float(reg)
        self.fit_intercept = fit_intercept
        self.w: Optional[np.ndarray] = None
        self.loss_history: List[float] = []

    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self.fit_intercept:
            N = X.shape[0]
            return np.hstack([np.ones((N, 1)), X])
        return X

    def compute_loss(self, Phi: np.ndarray, t: np.ndarray, w: Optional[np.ndarray] = None) -> float:
        """Negative log-likelihood error for probit regression."""
        if w is None:
            w = self.w
        if w is None:
            raise ValueError("Model is not fitted.")

        t = np.asarray(t, dtype=np.float64)
        a = Phi @ w
        y = probit(a)
        eps = 1e-15
        y = np.clip(y, eps, 1.0 - eps)
        loss = -np.sum(t * np.log(y) + (1.0 - t) * np.log(1.0 - y))
        if self.reg > 0:
            reg_w = w[1:] if self.fit_intercept else w
            loss += 0.5 * self.reg * np.sum(reg_w ** 2)
        return float(loss)

    def compute_gradient(self, Phi: np.ndarray, t: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        """Gradient of negative log-likelihood for probit regression."""
        if w is None:
            w = self.w
        if w is None:
            raise ValueError("Model is not fitted.")

        t = np.asarray(t, dtype=np.float64)
        a = Phi @ w
        y = probit(a)
        p_deriv = probit_deriv(a)
        eps = 1e-15
        denom = np.clip(y * (1.0 - y), eps, None)
        factor = (y - t) * p_deriv / denom
        grad = Phi.T @ factor
        if self.reg > 0:
            reg_vec = np.copy(w)
            if self.fit_intercept:
                reg_vec[0] = 0.0
            grad += self.reg * reg_vec
        return grad

    def fit(self, X: np.ndarray, t: np.ndarray,
            lr: float = 0.05,
            max_iter: int = 300,
            tol: float = 1e-6) -> "ProbitRegression":
        """Fit probit regression using gradient descent."""
        Phi = self._prepare_features(X)
        N, D = Phi.shape
        t = np.asarray(t, dtype=np.float64).ravel()

        self.w = np.zeros(D, dtype=np.float64)
        self.loss_history = []

        for step in range(max_iter):
            loss = self.compute_loss(Phi, t, self.w)
            self.loss_history.append(loss)

            grad = self.compute_gradient(Phi, t, self.w)
            if np.linalg.norm(grad) < tol:
                break

            self.w -= lr * grad

        self.loss_history.append(self.compute_loss(Phi, t, self.w))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior probability p(C1 | x) = Phi(w^T phi(x))."""
        if self.w is None:
            raise ValueError("Model is not fitted.")
        Phi = self._prepare_features(X)
        return probit(Phi @ self.w)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels {0, 1}."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def score(self, X: np.ndarray, t: np.ndarray) -> float:
        """Compute classification accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(t, dtype=int).ravel()))


# ==============================================================================
# 4. Canonical Link Functions & Outlier Comparison
# ==============================================================================

def verify_canonical_link_property(
    distribution: str = 'bernoulli',
    n_samples: int = 100,
    dim: int = 2,
    seed: int = 42,
) -> Dict[str, Union[float, bool, str]]:
    """
    Verify the theoretical canonical link identity (Section 5.4.6, Eq 5.89 - 5.95):
        f^{-1}(y) = psi(y) ==> f'(a) psi'(y) = 1
    leading directly to the universal gradient form:
        nabla E(w) = (1/s) sum_{n=1}^N (y_n - t_n) phi_n.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, dim))
    w_true = rng.normal(0, 1, dim)
    a = X @ w_true

    dist = distribution.lower()
    if dist == 'bernoulli':
        y = sigmoid(a)
        f_prime = y * (1.0 - y)
        psi_prime = 1.0 / (y * (1.0 - y))
        product = f_prime * psi_prime
        max_error = float(np.max(np.abs(product - 1.0)))
        t = (y > 0.5).astype(float)
        grad_canonical = X.T @ (y - t)
        return {
            'distribution': 'Bernoulli',
            'activation': 'sigmoid',
            'link': 'logit',
            'f_prime_psi_prime_error': max_error,
            'identity_holds': bool(max_error < 1e-10),
            'gradient_norm': float(np.linalg.norm(grad_canonical)),
        }
    elif dist == 'gaussian':
        y = a
        f_prime = np.ones_like(a)
        psi_prime = np.ones_like(y)
        product = f_prime * psi_prime
        max_error = float(np.max(np.abs(product - 1.0)))
        return {
            'distribution': 'Gaussian',
            'activation': 'identity',
            'link': 'identity',
            'f_prime_psi_prime_error': max_error,
            'identity_holds': bool(max_error < 1e-10),
        }
    elif dist == 'poisson':
        y = np.exp(np.clip(a, -10, 10))
        f_prime = y
        psi_prime = 1.0 / y
        product = f_prime * psi_prime
        max_error = float(np.max(np.abs(product - 1.0)))
        return {
            'distribution': 'Poisson',
            'activation': 'exp',
            'link': 'log',
            'f_prime_psi_prime_error': max_error,
            'identity_holds': bool(max_error < 1e-10),
        }
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")


def compare_logistic_and_probit_outliers(
    n_samples: int = 50,
    outlier_distance: float = 6.0,
    seed: int = 42,
) -> Dict[str, Union[float, bool]]:
    """
    Empirically compare Logistic Regression vs Probit Regression sensitivity
    when extreme outliers are introduced on the wrong side of the boundary.
    Because probit tails drop like exp(-x^2/2) vs sigmoid exp(-|x|),
    probit requires steeper decision boundaries or shifts more under certain outliers.
    """
    rng = np.random.default_rng(seed)
    X_clean = np.r_[rng.normal(-1.5, 0.5, n_samples), rng.normal(1.5, 0.5, n_samples)].reshape(-1, 1)
    y_clean = np.array([0] * n_samples + [1] * n_samples)

    log_clean = LogisticRegression(reg=1e-4).fit(X_clean, y_clean, method='irls')
    prob_clean = ProbitRegression(reg=1e-4).fit(X_clean, y_clean, lr=0.05, max_iter=400)

    # Outliers on the wrong side
    X_outliers = np.array([[outlier_distance], [outlier_distance + 1.0]])
    y_outliers = np.array([0, 0])

    X_corrupted = np.vstack([X_clean, X_outliers])
    y_corrupted = np.concatenate([y_clean, y_outliers])

    log_corrupted = LogisticRegression(reg=1e-4).fit(X_corrupted, y_corrupted, method='irls')
    prob_corrupted = ProbitRegression(reg=1e-4).fit(X_corrupted, y_corrupted, lr=0.05, max_iter=400)

    log_shift = float(np.linalg.norm(log_corrupted.w - log_clean.w))
    prob_shift = float(np.linalg.norm(prob_corrupted.w - prob_clean.w))

    return {
        'logistic_weight_shift': log_shift,
        'probit_weight_shift': prob_shift,
        'difference': float(np.abs(prob_shift - log_shift)),
    }


# ==============================================================================
# 5. Figure Reproductions
# ==============================================================================

def generate_figure_5_15(filepath: Optional[str] = None, save_both: bool = True) -> Tuple[plt.Figure, Tuple[str, Optional[str]]]:
    """Reproduce Figure 5.15: Role of nonlinear basis functions in classification.

    Left plot: Original input space (x1, x2) with data points from two classes.
    Two Gaussian basis functions with centres marked by green crosses and contours
    by green circles. Non-linear decision boundary shown by the black curve.
    Right plot: Feature space (phi_1, phi_2) with linear decision boundary
    separating the two classes.
    """
    np.random.seed(42)

    # 1. Generate data
    N_red = 60
    N_blue1 = 40
    N_blue2 = 40

    # Red cluster at center
    X_red = np.random.randn(N_red, 2) * 0.16 + np.array([0.0, 0.15])

    # Blue clusters at bottom-left and top-right
    X_blue1 = np.random.randn(N_blue1, 2) * 0.13 + np.array([-0.65, -0.65])
    X_blue2 = np.random.randn(N_blue2, 2) * 0.15 + np.array([0.95, 0.95])
    X_blue = np.vstack([X_blue1, X_blue2])

    # 2. Gaussian basis functions
    mu1 = np.array([-0.7, -0.7])
    mu2 = np.array([0.0, 0.15])
    s1 = 0.85
    s2 = 0.58

    rbf = GaussianBasisFunctions(centers=np.array([mu1, mu2]), scales=np.array([s1, s2]), include_bias=False)
    Phi_red = rbf.transform(X_red)
    Phi_blue = rbf.transform(X_blue)

    X_all = np.vstack([X_red, X_blue])
    Phi_all = np.vstack([Phi_red, Phi_blue])
    t_all = np.concatenate([np.ones(N_red), np.zeros(len(X_blue))])

    # 3. Train logistic regression in feature space
    clf = LogisticRegression(reg=0.01, fit_intercept=True)
    clf.fit(Phi_all, t_all, method="irls")
    b = clf.w[0]
    w1, w2 = clf.w[1], clf.w[2]

    # 4. Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))

    # Grid for contour plotting
    gx = np.linspace(-1.5, 1.5, 300)
    gy = np.linspace(-1.5, 1.5, 300)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
    grid_phi = rbf.transform(grid_pts)

    P1 = grid_phi[:, 0].reshape(GX.shape)
    P2 = grid_phi[:, 1].reshape(GX.shape)

    # Left plot: x-space
    ax1.scatter(X_red[:, 0], X_red[:, 1], c='red', s=24, edgecolors='none', zorder=4)
    ax1.scatter(X_blue[:, 0], X_blue[:, 1], c='blue', s=24, edgecolors='none', zorder=4)
    ax1.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], c='limegreen', marker='+', s=140, linewidths=3.2, zorder=6)

    # Contours of basis functions
    ax1.contour(GX, GY, P1, levels=[0.5], colors='limegreen', linewidths=1.8)
    ax1.contour(GX, GY, P2, levels=[0.5], colors='limegreen', linewidths=1.8)

    # Non-linear decision boundary in x space (w1*phi1 + w2*phi2 + b = 0)
    dec_bound = (w1 * grid_phi[:, 0] + w2 * grid_phi[:, 1] + b).reshape(GX.shape)
    ax1.contour(GX, GY, dec_bound, levels=[0.0], colors='black', linewidths=1.6, zorder=5)

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_xticks([-1, 0, 1])
    ax1.set_yticks([-1, 0, 1])
    ax1.set_xlabel(r'$x_1$', fontsize=13)
    ax1.set_ylabel(r'$x_2$', fontsize=13)
    ax1.set_aspect('equal')
    ax1.set_title('Original Input Space $(x_1, x_2)$', fontsize=13)
    ax1.tick_params(direction='in', top=True, right=True)

    # Right plot: feature space (phi_1, phi_2)
    ax2.scatter(Phi_red[:, 0], Phi_red[:, 1], c='red', s=24, edgecolors='none', zorder=4)
    ax2.scatter(Phi_blue[:, 0], Phi_blue[:, 1], c='blue', s=24, edgecolors='none', zorder=4)

    phi1_vals = np.linspace(-0.05, 1.05, 200)
    phi2_vals = -(w1 * phi1_vals + b) / w2
    ax2.plot(phi1_vals, phi2_vals, c='black', linewidth=1.6, zorder=5)

    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xticks([0, 0.5, 1.0])
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_xlabel(r'$\phi_1$', fontsize=13)
    ax2.set_ylabel(r'$\phi_2$', fontsize=13)
    ax2.set_aspect('equal')
    ax2.set_title(r'Feature Space $(\phi_1, \phi_2)$', fontsize=13)
    ax2.tick_params(direction='in', top=True, right=True)

    fig.suptitle('Figure 5.15: Role of Nonlinear Basis Functions in Linear Classification', fontsize=14, y=0.98)
    plt.tight_layout()

    saved_paths = _save_figure(fig, "fig_5_15_nonlinear_basis_classification.png", filepath=filepath, save_both=save_both)
    return fig, saved_paths


def generate_figure_5_16(filepath: Optional[str] = None, save_both: bool = True) -> Tuple[plt.Figure, Tuple[str, Optional[str]]]:
    """Reproduce Figure 5.16: Multi-class linear classification as a single-layer neural network."""
    fig, ax = plt.subplots(figsize=(8.0, 6.5))

    x_in = 0.28
    x_out = 0.72
    node_r = 0.052

    # Node vertical coordinates
    y_in = [0.18, 0.44, 0.82]   # phi_0, phi_1, phi_{M-1}
    labels_in = [r'$\phi_0(\mathbf{x})$', r'$\phi_1(\mathbf{x})$', r'$\phi_{M-1}(\mathbf{x})$']

    y_out = [0.44, 0.82]        # y_1, y_K
    labels_out = [r'$y_1(\mathbf{x}, \mathbf{w})$', r'$y_K(\mathbf{x}, \mathbf{w})$']

    # Draw directed links
    for yi in y_in:
        for yo in y_out:
            ax.annotate('', xy=(x_out - node_r, yo), xytext=(x_in + node_r, yi),
                        arrowprops=dict(arrowstyle='->', color='#2b2b2b', lw=1.6, mutation_scale=14))

    # Draw input layer nodes
    for i, (yi, label) in enumerate(zip(y_in, labels_in)):
        if i == 0:
            # Bias unit: solid blue circle
            circle = patches.Circle((x_in, yi), node_r, facecolor='#0022cc', edgecolor='#001188', lw=2.2, zorder=10)
        else:
            # Standard basis unit: light blue circle with dark blue border
            circle = patches.Circle((x_in, yi), node_r, facecolor='#c8dbff', edgecolor='#0022cc', lw=2.2, zorder=10)
        ax.add_patch(circle)
        ax.text(x_in - node_r - 0.04, yi, label, ha='right', va='center', fontsize=14)

    # Input layer vertical dots
    ax.text(x_in, 0.63, r'$\vdots$', ha='center', va='center', fontsize=22, fontweight='bold')

    # Draw output layer nodes
    for yo, label in zip(y_out, labels_out):
        circle = patches.Circle((x_out, yo), node_r, facecolor='#c8dbff', edgecolor='#0022cc', lw=2.2, zorder=10)
        ax.add_patch(circle)
        ax.text(x_out + node_r + 0.04, yo, label, ha='left', va='center', fontsize=14)

    # Output layer vertical dots
    ax.text(x_out, 0.63, r'$\vdots$', ha='center', va='center', fontsize=22, fontweight='bold')

    # Add explanatory annotations
    ax.text(x_in, 0.94, 'Input Feature Units\n' + r'$\boldsymbol{\phi}(\mathbf{x})$',
            ha='center', va='center', fontsize=12, fontweight='bold', color='#001188')
    ax.text(x_out, 0.94, 'Output Predictions\n' + r'$\mathbf{y}(\mathbf{x}, \mathbf{w})$ (Softmax)',
            ha='center', va='center', fontsize=12, fontweight='bold', color='#001188')

    # Weight link annotation
    ax.text(0.50, 0.70, r'$w_{ki}$', ha='center', va='bottom', fontsize=13, color='#333333',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#cccccc', alpha=0.9))

    # Gradient annotation at bottom
    ax.text(0.50, 0.05,
            r'$\frac{\partial E}{\partial w_{ki}} = \sum_{n=1}^N (y_{nk} - t_{nk}) \phi_i(\mathbf{x}_n)$' +
            '\n' + r'Gradient = (output error) $\times$ (input activation)',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#0022cc', lw=1.2))

    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.0, 1.0)
    ax.axis('off')

    ax.set_title('Figure 5.16: Representation of Multi-class Model as a Single-layer Neural Network',
                 fontsize=13, pad=15)
    plt.tight_layout()

    saved_paths = _save_figure(fig, "fig_5_16_single_layer_network.png", filepath=filepath, save_both=save_both)
    return fig, saved_paths


def generate_figure_5_17(filepath: Optional[str] = None, save_both: bool = True) -> Tuple[plt.Figure, Tuple[str, Optional[str]]]:
    """Reproduce Figure 5.17: Cumulative distribution function and noisy threshold model."""
    pi1, mu1, s1 = 0.70, 1.45, 0.40
    pi2, mu2, s2 = 0.30, 3.00, 0.32

    theta = np.linspace(0, 4, 600)
    p_theta = pi1 * norm.pdf(theta, mu1, s1) + pi2 * norm.pdf(theta, mu2, s2)
    f_a = pi1 * norm.cdf(theta, mu1, s1) + pi2 * norm.cdf(theta, mu2, s2)

    a_val = 2.80

    fig, ax = plt.subplots(figsize=(7.0, 5.8))

    # Shaded green region under p(theta) up to a
    theta_fill = theta[theta <= a_val]
    p_fill = p_theta[theta <= a_val]
    ax.fill_between(theta_fill, 0, p_fill, color='#c8f7c5', alpha=0.9,
                    label=r'Area $= \int_{-\infty}^a p(\theta)d\theta = f(a)$')

    # Density p(theta) (blue curve)
    ax.plot(theta, p_theta, color='#1e56a0', linewidth=2.3, label=r'Density $p(\theta)$')

    # Cumulative distribution function f(a) (red curve)
    ax.plot(theta, f_a, color='#d92027', linewidth=2.3, label=r'Activation $f(a)$ (CDF)')

    # Vertical green threshold line at a
    ax.axvline(a_val, ymin=0, ymax=1.0, color='#28a745', linewidth=2.0,
               label=rf'Threshold $a = {a_val}$')

    # Intersection point with f(a)
    f_at_a = pi1 * norm.cdf(a_val, mu1, s1) + pi2 * norm.cdf(a_val, mu2, s2)
    p_at_a = pi1 * norm.pdf(a_val, mu1, s1) + pi2 * norm.pdf(a_val, mu2, s2)

    ax.scatter([a_val], [f_at_a], color='#d92027', s=45, zorder=10)
    ax.scatter([a_val], [p_at_a], color='#1e56a0', s=45, zorder=10)

    # Annotations
    ax.annotate(r'$f(a) = ' + f'{f_at_a:.2f}$', xy=(a_val, f_at_a), xytext=(a_val - 0.75, f_at_a + 0.08),
                arrowprops=dict(arrowstyle='->', color='#d92027', lw=1.5),
                fontsize=11, color='#d92027', fontweight='bold')
    ax.annotate(r'Slope $= p(a) = ' + f'{p_at_a:.2f}$', xy=(a_val, p_at_a), xytext=(a_val + 0.15, p_at_a + 0.15),
                arrowprops=dict(arrowstyle='->', color='#1e56a0', lw=1.5),
                fontsize=11, color='#1e56a0', fontweight='bold')

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(r'$\theta$, $a$', fontsize=13)
    ax.set_ylabel(r'Probability / Density', fontsize=13)
    ax.tick_params(direction='in', top=True, right=True)
    ax.legend(loc='upper left', framealpha=0.92, fontsize=10.5)

    ax.set_title('Figure 5.17: Density $p(\\theta)$ and Cumulative Distribution $f(a)$ in Probit Model',
                 fontsize=12, pad=12)
    plt.tight_layout()

    saved_paths = _save_figure(fig, "fig_5_17_probit_threshold_model.png", filepath=filepath, save_both=save_both)
    return fig, saved_paths


def generate_all_section_5_4_figures(
    result_dirs: Optional[List[str]] = None,
    save_both: bool = True,
) -> Dict[str, plt.Figure]:
    """Generate and save all figures for Chapter 5 Section 5.4."""
    if result_dirs is None:
        result_dirs = ["5/result", "result"]
    for d in result_dirs:
        os.makedirs(d, exist_ok=True)

    results = {}
    f15, _ = generate_figure_5_15(filepath=os.path.join(result_dirs[0], "fig_5_15_nonlinear_basis_classification.png"), save_both=False)
    for extra_dir in result_dirs[1:]:
        save_plot(f15, os.path.join(extra_dir, "fig_5_15_nonlinear_basis_classification.png"))
    results["fig_5_15"] = f15

    f16, _ = generate_figure_5_16(filepath=os.path.join(result_dirs[0], "fig_5_16_single_layer_network.png"), save_both=False)
    for extra_dir in result_dirs[1:]:
        save_plot(f16, os.path.join(extra_dir, "fig_5_16_single_layer_network.png"))
    results["fig_5_16"] = f16

    f17, _ = generate_figure_5_17(filepath=os.path.join(result_dirs[0], "fig_5_17_probit_threshold_model.png"), save_both=False)
    for extra_dir in result_dirs[1:]:
        save_plot(f17, os.path.join(extra_dir, "fig_5_17_probit_threshold_model.png"))
    results["fig_5_17"] = f17

    return results


# Backward compatibility aliases
plot_figure_5_15 = generate_figure_5_15
plot_figure_5_16 = generate_figure_5_16
plot_figure_5_17 = generate_figure_5_17
