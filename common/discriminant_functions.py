"""
Discriminant Functions (Chapter 5: Single-layer Networks: Classification - Section 5.1).
Covers:
- Two-class Linear Discriminant, geometry, normal vector, margin, orthogonal projection (Section 5.1.1, Eq 5.2 - 5.6)
- Multi-class Linear Discriminant, 1-vs-rest & 1-vs-1 ambiguities, convex decision regions (Section 5.1.2, Eq 5.7 - 5.10)
- 1-of-K coding (Section 5.1.3, Eq 5.11)
- Least squares for classification, normal equations, sum-of-targets constraint, outlier sensitivity (Section 5.1.4, Eq 5.12 - 5.18)
- Logistic regression for comparison with least squares (robustness to outliers)
- Faithful reproduction of Figures 5.1, 5.2, 5.3, and 5.4.
"""
import os
from typing import Optional, Union, Tuple, List, Dict
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle
import matplotlib.lines as mlines

from common.plot_utils import setup_style, save_plot


# =====================================================================
# 1. Two-class Linear Discriminant (Section 5.1.1)
# =====================================================================

class LinearDiscriminant2Class:
    """
    Two-class linear discriminant function (Eq 5.2):
        y(x) = w^T x + w_0
    where w is the weight vector and w_0 is the bias.
    Decision boundary: y(x) = 0.
    """
    def __init__(self, w: np.ndarray, w0: float):
        self.w = np.asarray(w, dtype=np.float64).ravel()
        self.w0 = float(w0)
        self.dim = len(self.w)
        self.w_norm = float(np.linalg.norm(self.w))
        if self.w_norm < 1e-12:
            raise ValueError("Weight vector w must not be zero vector.")

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate y(x) = w^T x + w_0.
        X: shape (N, D) or (D,)
        Returns: shape (N,) or float
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            return float(np.dot(self.w, X_arr) + self.w0)
        return np.dot(X_arr, self.w) + self.w0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign to class C_1 (1) if y(x) > 0 else C_2 (0)."""
        y = self.decision_function(X)
        if np.ndim(y) == 0:
            return 1 if y > 0 else 0
        return (y > 0).astype(int)

    def margin(self, X: np.ndarray) -> np.ndarray:
        """
        Signed orthogonal distance of point x from the decision surface (Eq 5.5):
            r = y(x) / ||w||
        """
        y = self.decision_function(X)
        return y / self.w_norm

    def project(self, X: np.ndarray) -> np.ndarray:
        """
        Orthogonal projection x_perp of x onto the decision surface (Eq 5.4):
            x = x_perp + r * (w / ||w||)
            => x_perp = x - r * (w / ||w||)
        """
        X_arr = np.asarray(X, dtype=np.float64)
        unit_w = self.w / self.w_norm
        if X_arr.ndim == 1:
            r = self.margin(X_arr)
            return X_arr - r * unit_w
        r = self.margin(X_arr)[:, None]
        return X_arr - r * unit_w[None, :]

    def distance_from_origin(self) -> float:
        """
        Normal distance from origin to decision surface (Eq 5.3):
            w^T x / ||w|| = - w_0 / ||w||
        """
        return -self.w0 / self.w_norm

    normal_distance_from_origin = distance_from_origin
    orthogonal_projection = project

    def decision_boundary_line(self, x1_range: Tuple[float, float], num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        For 2D input (D=2), compute (x1, x2) coordinates along the boundary:
            w_1 x_1 + w_2 x_2 + w_0 = 0 => x_2 = -(w_1 x_1 + w_0) / w_2
        """
        if self.dim != 2:
            raise ValueError("Boundary line computation is only valid for 2D inputs.")
        x1 = np.linspace(x1_range[0], x1_range[1], num_points)
        if abs(self.w[1]) < 1e-12:
            raise ValueError("w[1] is near zero; boundary is a vertical line.")
        x2 = -(self.w[0] * x1 + self.w0) / self.w[1]
        return x1, x2


# =====================================================================
# 2. Multi-class Linear Discriminant (Section 5.1.2)
# =====================================================================

class LinearDiscriminantMultiClass:
    """
    K-class linear discriminant (Eq 5.7):
        y_k(x) = w_k^T x + w_{k0}
    Assigns x to class C_k if y_k(x) > y_j(x) for all j != k.
    Decision boundary between C_k and C_j is:
        (w_k - w_j)^T x + (w_{k0} - w_{j0}) = 0 (Eq 5.8).
    Decision regions are singly connected and convex (Eq 5.9, 5.10).
    """
    def __init__(self, W: np.ndarray, w0: np.ndarray):
        """
        W: shape (D, K) where column k is w_k
        w0: shape (K,) where element k is w_{k0}
        """
        self.W = np.asarray(W, dtype=np.float64)
        self.w0 = np.asarray(w0, dtype=np.float64).ravel()
        if self.W.shape[1] != len(self.w0):
            raise ValueError(f"W columns ({self.W.shape[1]}) must match w0 length ({len(self.w0)}).")
        self.dim, self.num_classes = self.W.shape

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate y_k(x) for all classes.
        X: shape (N, D) or (D,)
        Returns: shape (N, K) or (K,)
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            return np.dot(X_arr, self.W) + self.w0
        return np.dot(X_arr, self.W) + self.w0[None, :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class index k* = argmax_k y_k(x)."""
        Y = self.decision_function(X)
        return np.argmax(Y, axis=-1)

    def pairwise_boundary(self, k: int, j: int) -> Tuple[np.ndarray, float]:
        """
        Normal vector (w_k - w_j) and offset (w_{k0} - w_{j0}) for boundary between C_k and C_j (Eq 5.8).
        """
        diff_w = self.W[:, k] - self.W[:, j]
        diff_w0 = self.w0[k] - self.w0[j]
        return diff_w, float(diff_w0)

    def verify_convexity(self, xA: np.ndarray, xB: np.ndarray, num_points: int = 50) -> bool:
        """
        Verify convexity of decision regions (Eq 5.9 - 5.10):
        If xA and xB both lie inside region R_k, then any point:
            x_hat = lambda * xA + (1 - lambda) * xB (0 <= lambda <= 1)
        must also lie inside R_k.
        """
        class_A = self.predict(xA)
        class_B = self.predict(xB)
        if class_A != class_B:
            raise ValueError(f"xA (class {class_A}) and xB (class {class_B}) must belong to the same class.")
        target_class = class_A
        lambdas = np.linspace(0.0, 1.0, num_points)
        for lam in lambdas:
            x_hat = lam * xA + (1.0 - lam) * xB
            if self.predict(x_hat) != target_class:
                return False
        return True


# =====================================================================
# 3. Heuristic Classifiers and Ambiguity Analysis (Section 5.1.2)
# =====================================================================

class OneVersusRestClassifier:
    """
    K-class classifier built from K-1 (or K) binary linear discriminants (C_k vs not C_k).
    Shows how ambiguous unclassified regions arise (Figure 5.2 left).
    """
    def __init__(self, discriminants: List[LinearDiscriminant2Class]):
        self.discriminants = discriminants
        self.num_classes = len(discriminants)

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Returns boolean matrix (N, K) where entry (n, k) is True if discriminant k accepts x_n.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            return np.array([disc.decision_function(X_arr) > 0 for disc in self.discriminants])
        return np.column_stack([disc.decision_function(X_arr) > 0 for disc in self.discriminants])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns class label (0, ..., K-1) if exactly one classifier is positive.
        If 0 or >1 classifiers are positive, returns -1 (ambiguous).
        """
        raw = self.predict_raw(X)
        if raw.ndim == 1:
            pos_indices = np.where(raw)[0]
            return int(pos_indices[0]) if len(pos_indices) == 1 else -1
        pos_counts = raw.sum(axis=1)
        res = np.full(len(X), -1, dtype=int)
        for n in range(len(X)):
            if pos_counts[n] == 1:
                res[n] = np.where(raw[n])[0][0]
        return res

    def is_ambiguous(self, X: np.ndarray) -> np.ndarray:
        """True where point falls into an ambiguous region."""
        preds = self.predict(X)
        return preds == -1


class OneVersusOneClassifier:
    """
    K-class classifier built from K(K-1)/2 pairwise binary linear discriminants.
    Shows how ambiguous tie regions arise (Figure 5.2 right).
    """
    def __init__(self, pairwise_discriminants: Dict[Tuple[int, int], LinearDiscriminant2Class], num_classes: int = 3):
        self.pairwise = pairwise_discriminants
        self.num_classes = num_classes

    def predict_votes(self, X: np.ndarray) -> np.ndarray:
        """Count votes for each class across all pairwise classifiers."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
            squeeze = True
        else:
            squeeze = False

        votes = np.zeros((len(X_arr), self.num_classes), dtype=int)
        for (c1, c2), disc in self.pairwise.items():
            # If y > 0 => class c1, else class c2
            y = disc.decision_function(X_arr)
            c1_wins = y > 0
            votes[c1_wins, c1] += 1
            votes[~c1_wins, c2] += 1

        if squeeze:
            return votes[0]
        return votes

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns winner class if there is a strict majority (> any other vote count).
        Returns -1 if tie (ambiguous).
        """
        votes = self.predict_votes(X)
        if votes.ndim == 1:
            max_v = np.max(votes)
            if np.sum(votes == max_v) == 1:
                return int(np.argmax(votes))
            return -1
        res = np.full(len(votes), -1, dtype=int)
        for n, v in enumerate(votes):
            max_v = np.max(v)
            if np.sum(v == max_v) == 1:
                res[n] = np.argmax(v)
        return res

    def is_ambiguous(self, X: np.ndarray) -> np.ndarray:
        """True where point has tied votes."""
        preds = self.predict(X)
        return preds == -1


# =====================================================================
# 4. 1-of-K Coding (Section 5.1.3)
# =====================================================================

def to_one_of_k(labels: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert integer class labels (0, ..., K-1) to 1-of-K binary coding (Eq 5.11).
    labels: shape (N,)
    Returns: T of shape (N, K)
    """
    labels_arr = np.asarray(labels, dtype=int).ravel()
    K = int(num_classes if num_classes is not None else (np.max(labels_arr) + 1))
    N = len(labels_arr)
    T = np.zeros((N, K), dtype=np.float64)
    T[np.arange(N), labels_arr] = 1.0
    return T


# =====================================================================
# 5. Least Squares for Classification (Section 5.1.4)
# =====================================================================

class LeastSquaresClassifier:
    """
    Least-squares linear classifier (Section 5.1.4).
    Each class C_k is modeled by y_k(x) = w_k^T x + w_{k0} (Eq 5.12).
    Matrix form: y(x) = W_tilde^T x_tilde (Eq 5.13).
    Closed-form solution: W_tilde = (X_tilde^T X_tilde)^(-1) X_tilde^T T = X_tilde^+ T (Eq 5.15).
    """
    def __init__(self, reg: float = 1e-6):
        self.reg = float(reg)
        self.W_tilde = None  # Shape (D+1, K)
        self.num_classes = None
        self.dim = None

    def fit(self, X: np.ndarray, T: np.ndarray) -> 'LeastSquaresClassifier':
        """
        Fit least-squares model to training inputs X and targets T.
        X: shape (N, D)
        T: shape (N, K) in 1-of-K coding, or (N,) class labels
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X_arr.shape}")
        N, D = X_arr.shape
        self.dim = D

        T_arr = np.asarray(T, dtype=np.float64)
        if T_arr.ndim == 1:
            # Check if labels are discrete
            if np.all(np.isin(T_arr, [0, 1])):
                # 2-class binary target
                T_mat = to_one_of_k(T_arr, num_classes=2)
            else:
                T_mat = to_one_of_k(T_arr)
        else:
            T_mat = T_arr

        self.num_classes = T_mat.shape[1]

        # Augment X with leading 1 for bias: x_tilde = [1, x_1, ..., x_D]
        X_tilde = np.column_stack([np.ones(N), X_arr])  # Shape (N, D+1)

        # Normal equations: (X_tilde^T X_tilde + reg * I) W_tilde = X_tilde^T T
        XtX = np.dot(X_tilde.T, X_tilde)
        if self.reg > 0:
            reg_mat = self.reg * np.eye(D + 1)
            reg_mat[0, 0] = 0.0  # Do not regularize bias
            XtX += reg_mat
        XtT = np.dot(X_tilde.T, T_mat)

        self.W_tilde = la.solve(XtX, XtT, assume_a='sym')
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate y(x) = W_tilde^T x_tilde (Eq 5.16).
        Returns shape (N, K) or (K,)
        """
        if self.W_tilde is None:
            raise RuntimeError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            x_tilde = np.concatenate([[1.0], X_arr])
            return np.dot(self.W_tilde.T, x_tilde)
        N = len(X_arr)
        X_tilde = np.column_stack([np.ones(N), X_arr])
        return np.dot(X_tilde, self.W_tilde)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class with largest output y_k(x)."""
        Y = self.decision_function(X)
        return np.argmax(Y, axis=-1)

    def verify_sum_constraint(self, X: np.ndarray, tol: float = 1e-4) -> bool:
        """
        Verify that sum_k y_k(x) = 1 for all x under 1-of-K coding (Eq 5.17 - 5.18).
        """
        Y = self.decision_function(X)
        if Y.ndim == 1:
            return abs(float(np.sum(Y)) - 1.0) < tol
        row_sums = np.sum(Y, axis=1)
        return bool(np.all(np.abs(row_sums - 1.0) < tol))

    @property
    def w0(self) -> np.ndarray:
        """Bias vector w_{k0}, shape (K,)."""
        return self.W_tilde[0, :]

    @property
    def W(self) -> np.ndarray:
        """Weight matrix, shape (D, K)."""
        return self.W_tilde[1:, :]


# =====================================================================
# 6. Logistic Regression for 2-class Outlier Comparison (Section 5.1.4)
# =====================================================================

class LogisticRegression2Class:
    """
    Two-class logistic regression classifier:
        p(C_1 | x) = sigma(w^T x + w_0) = 1 / (1 + exp(-(w^T x + w_0)))
    Used for comparing robustness against least squares in Figure 5.4.
    """
    def __init__(self, reg: float = 1e-4):
        self.reg = float(reg)
        self.w_tilde = None  # Shape (D+1,)
        self.dim = None

    def fit(self, X: np.ndarray, t: np.ndarray) -> 'LogisticRegression2Class':
        """
        Fit logistic regression using L-BFGS-B / Newton optimization.
        X: shape (N, D)
        t: shape (N,) with binary labels 0 or 1.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        t_arr = np.asarray(t, dtype=np.float64).ravel()
        N, D = X_arr.shape
        self.dim = D

        X_tilde = np.column_stack([np.ones(N), X_arr])

        def loss_and_grad(w):
            z = np.dot(X_tilde, w)
            # numerically stable log1p(exp)
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            # Negative log likelihood
            loss = -np.sum(t_arr * np.log(np.clip(p, 1e-15, 1.0)) + (1.0 - t_arr) * np.log(np.clip(1.0 - p, 1e-15, 1.0)))
            grad = np.dot(X_tilde.T, p - t_arr)
            # Regularization on weights (not bias)
            loss += 0.5 * self.reg * np.sum(w[1:] ** 2)
            grad[1:] += self.reg * w[1:]
            return loss, grad

        w_init = np.zeros(D + 1)
        res = minimize(loss_and_grad, w_init, jac=True, method='L-BFGS-B')
        self.w_tilde = res.x
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return p(C_1 | x)."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            x_tilde = np.concatenate([[1.0], X_arr])
            z = np.dot(x_tilde, self.w_tilde)
        else:
            X_tilde = np.column_stack([np.ones(len(X_arr)), X_arr])
            z = np.dot(X_tilde, self.w_tilde)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return log-odds / linear score: w^T x + w_0."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            x_tilde = np.concatenate([[1.0], X_arr])
            return np.dot(x_tilde, self.w_tilde)
        X_tilde = np.column_stack([np.ones(len(X_arr)), X_arr])
        return np.dot(X_tilde, self.w_tilde)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict 1 if p(C_1 | x) >= 0.5 else 0."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    @property
    def w0(self) -> float:
        return float(self.w_tilde[0])

    @property
    def w(self) -> np.ndarray:
        return self.w_tilde[1:]


# =====================================================================
# 7. Figure Reproduction (Figures 5.1 to 5.4)
# =====================================================================

def plot_figure_5_1_geometry(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """
    Faithful reproduction of Figure 5.1 (Book page 133):
    Illustration of the geometry of a linear discriminant function in two dimensions.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Axis limits
    xlim = (-2.0, 6.5)
    ylim = (-2.5, 5.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw coordinate axes (black arrows)
    ax.annotate('', xy=(6.2, 0), xytext=(-0.5, 0),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5, mutation_scale=15))
    ax.text(6.0, -0.4, r'$x_1$', fontsize=15)

    ax.annotate('', xy=(0, 5.2), xytext=(0, -0.5),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5, mutation_scale=15))
    ax.text(-0.4, 4.9, r'$x_2$', fontsize=15)

    # Weights and boundary
    w = np.array([0.75, 1.0])
    w_norm = np.linalg.norm(w)
    unit_w = w / w_norm
    w0 = -3.5

    d_origin = -w0 / w_norm # normal distance from origin
    p_origin_proj = d_origin * unit_w
    perp_dir = np.array([unit_w[1], -unit_w[0]]) # parallel to decision surface (pointing right-down)

    # Decision boundary line (red)
    line_x1 = np.array([-1.2, 5.8])
    line_x2 = - (w[0] * line_x1 + w0) / w[1]
    ax.plot(line_x1, line_x2, color='#E41A1C', lw=2.8, zorder=3)

    # Labels for y > 0, y = 0, y < 0 and regions R1, R2 (top left)
    ax.text(-1.8, 4.2, r'$y > 0$', fontsize=14)
    ax.text(-2.1, 3.7, r'$y = 0$', fontsize=14)
    ax.text(-2.4, 3.2, r'$y < 0$', fontsize=14)
    ax.text(-0.5, 4.1, r'$\mathcal{R}_1$', fontsize=16)
    ax.text(-1.0, 3.4, r'$\mathcal{R}_2$', fontsize=16)

    # Normal vector w from origin (green arrow)
    w_scale = 1.9
    w_end = unit_w * w_scale
    ax.annotate('', xy=(w_end[0], w_end[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color='#00C000', lw=2.5, mutation_scale=18))
    ax.text(w_end[0] - 0.45, w_end[1] + 0.1, r'$\mathbf{w}$', fontsize=15, color='black', fontweight='bold')

    # Green dashed line from w vector to decision surface
    ax.plot([w_end[0], p_origin_proj[0]], [w_end[1], p_origin_proj[1]], color='#00C000', linestyle='--', lw=1.5)

    # Green right-angle marker at p_origin_proj
    sq_size = 0.28
    c1 = p_origin_proj - unit_w * sq_size
    c2 = c1 - perp_dir * sq_size
    c3 = p_origin_proj - perp_dir * sq_size
    ax.plot([c1[0], c2[0], c3[0]], [c1[1], c2[1], c3[1]], color='#00C000', lw=1.5)

    # Normal distance from origin to decision surface (-w0 / ||w||)
    # Measured in 4th quadrant along parallel guide lines
    dist_offset = 2.6
    pt_orig_line = perp_dir * dist_offset
    pt_surf_line = p_origin_proj + perp_dir * dist_offset
    ax.plot([0, pt_orig_line[0]], [0, pt_orig_line[1]], color='black', lw=1.2)
    ax.plot([p_origin_proj[0], pt_surf_line[0]], [p_origin_proj[1], pt_surf_line[1]], color='black', lw=1.2)
    ax.annotate('', xy=(pt_surf_line[0], pt_surf_line[1]), xytext=(pt_orig_line[0], pt_orig_line[1]),
                arrowprops=dict(arrowstyle="<|-|>", color='black', lw=1.5, mutation_scale=15))
    mid_dist = 0.5 * (pt_orig_line + pt_surf_line)
    ax.text(mid_dist[0] + 0.18, mid_dist[1] - 0.25, r'$\frac{-w_0}{\|\mathbf{w}\|}$', fontsize=17)

    # Point x (blue arrow from origin)
    x_pt = np.array([5.0, 3.2])
    ax.annotate('', xy=(x_pt[0], x_pt[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color='#0044FF', lw=2.2, mutation_scale=18))
    ax.plot(x_pt[0], x_pt[1], 'o', color='#0044FF', markersize=5)
    ax.text(x_pt[0] + 0.15, x_pt[1] + 0.05, r'$\mathbf{x}$', fontsize=15, fontweight='bold', color='black')

    # Orthogonal projection x_perp
    r = (np.dot(w, x_pt) + w0) / w_norm
    x_perp = x_pt - r * unit_w
    ax.plot([x_perp[0], x_pt[0]], [x_perp[1], x_pt[1]], color='#0044FF', linestyle='--', lw=1.8)
    ax.plot(x_perp[0], x_perp[1], 'o', color='#E41A1C', markersize=5)
    ax.text(x_perp[0] - 0.7, x_perp[1] - 0.35, r'$\mathbf{x}_\perp$', fontsize=15, fontweight='bold', color='black')

    # Blue right-angle marker at x_perp
    c1_b = x_perp + unit_w * sq_size
    c2_b = c1_b + perp_dir * sq_size
    c3_b = x_perp + perp_dir * sq_size
    ax.plot([c1_b[0], c2_b[0], c3_b[0]], [c1_b[1], c2_b[1], c3_b[1]], color='#0044FF', lw=1.5)

    # Distance label y(x) / ||w||
    mid_proj = 0.5 * (x_perp + x_pt)
    ax.text(mid_proj[0] + 0.15, mid_proj[1] - 0.2, r'$\frac{y(\mathbf{x})}{\|\mathbf{w}\|}$', fontsize=17)

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


def plot_figure_5_2_ambiguities(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """
    Faithful reproduction of Figure 5.2 (Book page 134):
    Ambiguous regions (shown in green) in heuristic multi-class classifiers.
    Left: One-versus-the-rest with 2 classifiers for 3 classes.
    Right: One-versus-one with 3 pairwise classifiers for 3 classes.
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # -------------------------------------------------------------
    # Left Subplot: One-versus-the-rest
    # -------------------------------------------------------------
    ax1.set_xlim(-0.2, 5.2)
    ax1.set_ylim(-0.2, 5.2)
    ax1.axis('off')

    p_int = np.array([1.8, 2.6])
    pt1_start = np.array([0.3, 1.4])
    pt1_end = np.array([4.6, 4.6])
    pt2_start = np.array([1.0, 4.7])
    pt2_end = np.array([3.0, 0.4])

    ambig_poly_left = Polygon([p_int, [1.1, 4.6], [4.6, 4.6]],
                              facecolor='#33ee33', edgecolor='none', alpha=0.9, zorder=1)
    ax1.add_patch(ambig_poly_left)

    ax1.plot([pt1_start[0], pt1_end[0]], [pt1_start[1], pt1_end[1]], color='red', lw=2.5, zorder=3)
    ax1.plot([pt2_start[0], pt2_end[0]], [pt2_start[1], pt2_end[1]], color='red', lw=2.5, zorder=3)

    ax1.text(2.3, 3.8, '?', fontsize=18, fontweight='bold', color='black', ha='center', va='center')
    ax1.text(0.9, 2.7, r'$\mathcal{R}_1$', fontsize=16, color='black')
    ax1.text(3.1, 2.5, r'$\mathcal{R}_2$', fontsize=16, color='black')
    ax1.text(1.9, 1.3, r'$\mathcal{R}_3$', fontsize=16, color='black')

    n1 = np.array([-0.6, 0.8])
    base1 = np.array([0.7, 1.7])
    ax1.annotate('', xy=base1 + 0.6 * n1, xytext=base1 - 0.6 * n1,
                 arrowprops=dict(arrowstyle="<->", color='black', lw=1.8, mutation_scale=14), zorder=4)
    ax1.text(base1[0] + 0.6 * n1[0] - 0.4, base1[1] + 0.6 * n1[1] + 0.05, r'$\mathcal{C}_1$', fontsize=13)
    ax1.text(base1[0] - 0.6 * n1[0] + 0.1, base1[1] - 0.6 * n1[1] - 0.15, r'$\mathrm{not}\;\mathcal{C}_1$', fontsize=13)

    n2 = np.array([0.9, 0.45])
    base2 = np.array([2.5, 1.4])
    ax1.annotate('', xy=base2 + 0.55 * n2, xytext=base2 - 0.55 * n2,
                 arrowprops=dict(arrowstyle="<->", color='black', lw=1.8, mutation_scale=14), zorder=4)
    ax1.text(base2[0] + 0.55 * n2[0] + 0.1, base2[1] + 0.55 * n2[1] + 0.05, r'$\mathcal{C}_2$', fontsize=13)
    ax1.text(base2[0] - 0.55 * n2[0] - 0.65, base2[1] - 0.55 * n2[1] - 0.2, r'$\mathrm{not}\;\mathcal{C}_2$', fontsize=13)

    ax1.set_title("One-versus-the-rest (K=3, 2 classifiers)", fontsize=13, pad=10)

    # -------------------------------------------------------------
    # Right Subplot: One-versus-one
    # -------------------------------------------------------------
    ax2.set_xlim(-0.2, 5.2)
    ax2.set_ylim(-0.2, 5.2)
    ax2.axis('off')

    v1 = np.array([2.5, 3.4])
    v2 = np.array([3.4, 2.0])
    v3 = np.array([1.7, 2.2])

    tri_ambig = Polygon([v1, v2, v3], facecolor='#33ee33', edgecolor='none', alpha=0.9, zorder=1)
    ax2.add_patch(tri_ambig)

    dir13 = (v3 - v1) / np.linalg.norm(v3 - v1)
    ax2.plot([v3[0] + 0.9 * dir13[0], v1[0] - 1.2 * dir13[0]],
             [v3[1] + 0.9 * dir13[1], v1[1] - 1.2 * dir13[1]], color='red', lw=2.5, zorder=3)
    ax2.plot([v1[0], v1[0] + 1.2 * (v1[0]-v3[0])], [v1[1], v1[1] + 1.2 * (v1[1]-v3[1])],
             color='red', lw=2.0, linestyle='--', zorder=3)

    dir12 = (v2 - v1) / np.linalg.norm(v2 - v1)
    ax2.plot([v2[0] + 0.9 * dir12[0], v1[0] - 1.2 * dir12[0]],
             [v2[1] + 0.9 * dir12[1], v1[1] - 1.2 * dir12[1]], color='red', lw=2.5, zorder=3)
    ax2.plot([v2[0], v2[0] + 1.2 * (v2[0]-v1[0])], [v2[1], v2[1] + 1.2 * (v2[1]-v1[1])],
             color='red', lw=2.0, linestyle='--', zorder=3)

    dir32 = (v2 - v3) / np.linalg.norm(v2 - v3)
    ax2.plot([v3[0] - 1.3 * dir32[0], v2[0] + 1.3 * dir32[0]],
             [v3[1] - 1.3 * dir32[1], v2[1] + 1.3 * dir32[1]], color='red', lw=2.5, zorder=3)
    ax2.plot([v3[0] - 1.3 * dir32[0], v2[0] + 1.4 * dir32[0]],
             [v3[1] + 0.9, v2[1] + 1.6], color='red', lw=2.0, linestyle='--', zorder=3)

    ax2.text(2.5, 2.5, '?', fontsize=18, fontweight='bold', color='black', ha='center', va='center')
    ax2.text(1.7, 3.4, r'$\mathcal{R}_1$', fontsize=16, color='black')
    ax2.text(2.6, 1.3, r'$\mathcal{R}_2$', fontsize=16, color='black')
    ax2.text(3.7, 3.0, r'$\mathcal{R}_3$', fontsize=16, color='black')

    ax2.annotate('', xy=(0.8, 3.0), xytext=(1.2, 2.1),
                 arrowprops=dict(arrowstyle="<->", color='black', lw=1.8, mutation_scale=14), zorder=4)
    ax2.text(0.5, 3.0, r'$\mathcal{C}_1$', fontsize=13)
    ax2.text(1.1, 1.8, r'$\mathcal{C}_2$', fontsize=13)

    ax2.annotate('', xy=(2.2, 4.2), xytext=(3.0, 4.4),
                 arrowprops=dict(arrowstyle="<->", color='black', lw=1.8, mutation_scale=14), zorder=4)
    ax2.text(1.8, 4.2, r'$\mathcal{C}_1$', fontsize=13)
    ax2.text(3.1, 4.4, r'$\mathcal{C}_3$', fontsize=13)

    ax2.annotate('', xy=(4.4, 2.4), xytext=(4.1, 1.5),
                 arrowprops=dict(arrowstyle="<->", color='black', lw=1.8, mutation_scale=14), zorder=4)
    ax2.text(4.4, 2.5, r'$\mathcal{C}_3$', fontsize=13)
    ax2.text(4.1, 1.2, r'$\mathcal{C}_2$', fontsize=13)

    ax2.set_title("One-versus-one (K=3, 3 pairwise classifiers)", fontsize=13, pad=10)

    fig.suptitle("Figure 5.2: Ambiguous Regions in Heuristic Multi-Class Discriminants",
                 fontsize=14, y=0.98)
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


def plot_figure_5_3_convex_regions(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """
    Faithful reproduction of Figure 5.3 (Book page 135):
    Convex decision regions for a multi-class linear discriminant.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 6.5))

    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(-0.2, 5.2)
    ax.set_aspect('equal')
    ax.axis('off')

    center = np.array([2.5, 2.6])
    ray1 = np.array([2.1, 4.5])
    ray2 = np.array([0.9, 1.2])
    ray3 = np.array([4.7, 2.4])

    ax.plot([center[0], ray1[0]], [center[1], ray1[1]], color='red', lw=2.8, zorder=3)
    ax.plot([center[0], ray2[0]], [center[1], ray2[1]], color='red', lw=2.8, zorder=3)
    ax.plot([center[0], ray3[0]], [center[1], ray3[1]], color='red', lw=2.8, zorder=3)

    ax.text(1.5, 3.2, r'$\mathcal{R}_i$', fontsize=18, color='black')
    ax.text(3.0, 3.6, r'$\mathcal{R}_j$', fontsize=18, color='black')
    ax.text(2.6, 2.0, r'$\mathcal{R}_k$', fontsize=18, color='black')

    xA = np.array([1.2, 1.0])
    xB = np.array([4.4, 1.3])

    ax.plot([xA[0], xB[0]], [xA[1], xB[1]], color='blue', lw=2.5, zorder=4)

    ax.plot(xA[0], xA[1], 'bo', markersize=7, zorder=5)
    ax.text(xA[0] - 0.45, xA[1] - 0.05, r'$\mathbf{x}_{\mathrm{A}}$', fontsize=14, color='black', fontweight='bold')

    ax.plot(xB[0], xB[1], 'bo', markersize=7, zorder=5)
    ax.text(xB[0] + 0.12, xB[1] - 0.05, r'$\mathbf{x}_{\mathrm{B}}$', fontsize=14, color='black', fontweight='bold')

    lam = 0.5
    x_hat = lam * xA + (1.0 - lam) * xB
    ax.plot(x_hat[0], x_hat[1], 'bo', markersize=6, zorder=5)
    ax.text(x_hat[0] - 0.05, x_hat[1] - 0.35, r'$\widehat{\mathbf{x}}$', fontsize=15, color='black', fontweight='bold')

    ax.set_title("Figure 5.3: Convex Decision Regions of Multi-Class Linear Discriminant",
                 fontsize=12, pad=12)

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


def plot_figure_5_4_least_squares_outliers(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """
    Faithful reproduction of Figure 5.4 (Book page 138):
    Comparison of Least Squares (magenta line) vs Logistic Regression (green line)
    on two-class classification without and with outliers.
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.8))

    np.random.seed(42)
    N_per_class = 35

    mean1 = np.array([0.0, 1.4])
    cov1 = np.array([[1.5, 1.1],
                     [1.1, 1.4]])
    X1 = np.random.multivariate_normal(mean1, cov1, N_per_class)

    mean2 = np.array([1.2, -1.2])
    cov2 = np.array([[1.2, 0.9],
                     [0.9, 1.2]])
    X2 = np.random.multivariate_normal(mean2, cov2, N_per_class)

    X_clean = np.vstack([X1, X2])
    t_clean = np.array([1] * N_per_class + [0] * N_per_class)

    mean_outliers = np.array([7.8, -6.8])
    cov_outliers = np.array([[0.25, 0.1],
                             [0.1, 0.25]])
    X_outliers = np.random.multivariate_normal(mean_outliers, cov_outliers, 14)

    X_dirty = np.vstack([X_clean, X_outliers])
    t_dirty = np.array([1] * N_per_class + [0] * (N_per_class + len(X_outliers)))

    ls_clean = LeastSquaresClassifier().fit(X_clean, t_clean)
    lr_clean = LogisticRegression2Class().fit(X_clean, t_clean)

    ls_dirty = LeastSquaresClassifier().fit(X_dirty, t_dirty)
    lr_dirty = LogisticRegression2Class().fit(X_dirty, t_dirty)

    x_plot = np.linspace(-4, 9, 200)

    for ax, is_dirty, ls_mod, lr_mod in [(ax1, False, ls_clean, lr_clean),
                                          (ax2, True, ls_dirty, lr_dirty)]:
        ax.set_xlim(-4, 9)
        ax.set_ylim(-9, 4)
        ax.set_aspect('equal')

        ax.scatter(X1[:, 0], X1[:, 1], color='red', marker='x', s=45, lw=2.0, label=r'Class $\mathcal{C}_1$', zorder=4)
        ax.scatter(X2[:, 0], X2[:, 1], facecolors='none', edgecolors='blue', marker='o', s=45, lw=2.0, label=r'Class $\mathcal{C}_2$', zorder=4)

        if is_dirty:
            ax.scatter(X_outliers[:, 0], X_outliers[:, 1], facecolors='none', edgecolors='blue',
                       marker='o', s=45, lw=2.0, zorder=4)

        w_diff = ls_mod.W[:, 0] - ls_mod.W[:, 1]
        w0_diff = ls_mod.w0[0] - ls_mod.w0[1]
        if abs(w_diff[1]) > 1e-6:
            y_ls = -(w_diff[0] * x_plot + w0_diff) / w_diff[1]
            valid = (y_ls >= -9) & (y_ls <= 4)
            ax.plot(x_plot[valid], y_ls[valid], color='#aa00aa', lw=2.5, label='Least squares', zorder=3)

        w_lr = lr_mod.w
        w0_lr = lr_mod.w0
        if abs(w_lr[1]) > 1e-6:
            y_lr = -(w_lr[0] * x_plot + w0_lr) / w_lr[1]
            valid_lr = (y_lr >= -9) & (y_lr <= 4)
            ax.plot(x_plot[valid_lr], y_lr[valid_lr], color='#00bb00', lw=2.5, label='Logistic regression', zorder=3)

        ax.set_xticks(np.arange(-4, 10, 2))
        ax.set_yticks(np.arange(-8, 5, 2))

    ax1.set_title("Without outliers", fontsize=12)
    ax2.set_title("With outliers at bottom right", fontsize=12)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=True)

    fig.suptitle("Figure 5.4: Least Squares vs Logistic Regression Sensitivity to Outliers",
                 fontsize=14, y=1.08)
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


def generate_all_section_5_1_figures(result_dirs: Optional[List[str]] = None) -> Dict[str, plt.Figure]:
    """
    Generate and save all Figure 5.1 to 5.4 to both 5/result/ and result/.
    """
    if result_dirs is None:
        result_dirs = ["5/result", "result"]
    for d in result_dirs:
        os.makedirs(d, exist_ok=True)

    figs = {}
    figs['fig_5_1'] = plot_figure_5_1_geometry(filepath=os.path.join(result_dirs[0], "fig_5_1_discriminant_geometry.png"))
    figs['fig_5_2'] = plot_figure_5_2_ambiguities(filepath=os.path.join(result_dirs[0], "fig_5_2_multiclass_ambiguities.png"))
    figs['fig_5_3'] = plot_figure_5_3_convex_regions(filepath=os.path.join(result_dirs[0], "fig_5_3_convex_regions.png"))
    figs['fig_5_4'] = plot_figure_5_4_least_squares_outliers(filepath=os.path.join(result_dirs[0], "fig_5_4_least_squares_outliers.png"))

    return figs

