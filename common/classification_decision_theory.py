"""
Decision Theory for Classification (Chapter 5: Single-layer Networks: Classification, Section 5.2).

Covers:
- Misclassification Rate Minimization (Section 5.2.1, Eq 5.20 - 5.21, Figure 5.5)
- Expected Loss Minimization & Loss Matrix (Section 5.2.2, Eq 5.22 - 5.23, Figure 5.6)
- The Reject Option (Section 5.2.3, Figure 5.7)
- Inference and Decision Separation & Class Prior Compensation (Section 5.2.4, Eq 5.24 - 5.27, Figure 5.8)
- Classifier Accuracy & Confusion Matrix (Section 5.2.5, Eq 5.28 - 5.37, Figure 5.9)
- ROC Curves & AUC Analysis (Section 5.2.6, Eq 5.38 - 5.39, Figure 5.10 - 5.11)
"""
import os
from typing import Optional, Union, Tuple, List, Dict, Callable
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import matplotlib.lines as mlines

from common.plot_utils import setup_style, save_plot


# =====================================================================
# 1. Decision Rules and Expected Loss (Section 5.2.1 - 5.2.2)
# =====================================================================

def optimal_decision_rule_misclassification(posteriors: np.ndarray) -> np.ndarray:
    """
    Optimal decision rule to minimize misclassification rate (Eq 5.20 - 5.21):
        assign x to class k* = argmax_k p(C_k | x)
    posteriors: shape (N, K) or (K,)
    Returns: shape (N,) or int
    """
    posteriors_arr = np.asarray(posteriors, dtype=np.float64)
    return np.argmax(posteriors_arr, axis=-1)


def compute_expected_loss(posteriors: np.ndarray, loss_matrix: np.ndarray) -> np.ndarray:
    """
    Compute expected loss for choosing each class j (Eq 5.23):
        E_loss(j | x) = sum_k L_{kj} * p(C_k | x)
    posteriors: shape (N, K) or (K,)
    loss_matrix: shape (K, K) where L_{kj} is loss for true class k and decision j.
    Returns: expected_losses of shape (N, K) or (K,)
    """
    posteriors_arr = np.asarray(posteriors, dtype=np.float64)
    loss_mat = np.asarray(loss_matrix, dtype=np.float64)
    if loss_mat.shape[0] != loss_mat.shape[1]:
        raise ValueError("loss_matrix must be square (K x K).")
    # For single x: (K,) @ (K, K) -> (K,) where element j is sum_k L_{kj} p_k
    if posteriors_arr.ndim == 1:
        return np.dot(posteriors_arr, loss_mat)
    # For batch: (N, K) @ (K, K) -> (N, K)
    return np.dot(posteriors_arr, loss_mat)


def optimal_decision_rule_expected_loss(posteriors: np.ndarray, loss_matrix: np.ndarray) -> np.ndarray:
    """
    Optimal decision rule to minimize expected loss (Eq 5.22 - 5.23):
        assign x to class j* = argmin_j sum_k L_{kj} * p(C_k | x)
    """
    exp_losses = compute_expected_loss(posteriors, loss_matrix)
    return np.argmin(exp_losses, axis=-1)


# =====================================================================
# 2. Reject Option (Section 5.2.3)
# =====================================================================

def decision_rule_with_reject(
    posteriors: np.ndarray,
    theta: float,
    reject_label: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The Reject Option (Section 5.2.3):
        Assign to k* = argmax_k p(C_k | x) if max_k p(C_k | x) >= theta,
        otherwise reject (label = -1).
    posteriors: shape (N, K) or (K,)
    theta: threshold in [1/K, 1.0]
    Returns:
        decisions: shape (N,) or int
        rejected_mask: boolean mask indicating rejected points
    """
    posteriors_arr = np.asarray(posteriors, dtype=np.float64)
    if posteriors_arr.ndim == 1:
        max_p = np.max(posteriors_arr)
        if max_p >= theta:
            return int(np.argmax(posteriors_arr)), False
        return reject_label, True

    max_p = np.max(posteriors_arr, axis=1)
    k_star = np.argmax(posteriors_arr, axis=1)
    rejected = max_p < theta
    decisions = np.where(rejected, reject_label, k_star)
    return decisions, rejected


# =====================================================================
# 3. Class Prior Compensation (Section 5.2.4)
# =====================================================================

def compensate_for_class_priors(
    p_post_train: np.ndarray,
    p_train: np.ndarray,
    p_real: np.ndarray
) -> np.ndarray:
    r"""
    Compensate posterior probabilities for artificially balanced/stratified training data (Section 5.2.4, Eq 5.24 - 5.27).
     p(x | C_k) \propto p_train(C_k | x) / p_train(C_k)
        => p_target(C_k | x) \propto p_target(C_k) * p_train(C_k | x) / p_train(C_k)
    """
    train_post = np.asarray(p_post_train, dtype=np.float64)
    p_train = np.asarray(p_train, dtype=np.float64)
    p_target = np.asarray(p_real, dtype=np.float64)

    ratio = p_target / p_train
    if train_post.ndim == 1:
        unnorm = train_post * ratio
        return unnorm / np.sum(unnorm)
    unnorm = train_post * ratio[None, :]
    return unnorm / np.sum(unnorm, axis=1, keepdims=True)


# =====================================================================
# 4. Classifier Evaluation Metrics & Confusion Matrix (Section 5.2.5)
# =====================================================================

class ConfusionMatrix2Class:
    """
    Confusion Matrix and Evaluation Metrics for 2-class classification (Section 5.2.5).
    Eq 5.28 - 5.39.
    Classes: 0 (Normal / Negative), 1 (Cancer / Positive).
    """
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_t = np.asarray(y_true, dtype=int).ravel()
        y_p = np.asarray(y_pred, dtype=int).ravel()
        if len(y_t) != len(y_p):
            raise ValueError("y_true and y_pred must have same length.")

        self.N = len(y_t)
        self.TP = int(np.sum((y_t == 1) & (y_p == 1)))
        self.FP = int(np.sum((y_t == 0) & (y_p == 1)))
        self.TN = int(np.sum((y_t == 0) & (y_p == 0)))
        self.FN = int(np.sum((y_t == 1) & (y_p == 0)))

        # Matrix: [[TN, FP], [FN, TP]] matching Bishop Figure 5.9
        self.matrix = np.array([
            [self.TN, self.FP],
            [self.FN, self.TP]
        ], dtype=int)

    @property
    def accuracy(self) -> float:
        """Accuracy = (TP + TN) / N (Eq 5.29)."""
        return (self.TP + self.TN) / max(self.N, 1)

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP) (Eq 5.30)."""
        denom = self.TP + self.FP
        return self.TP / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall / Sensitivity / TPR = TP / (TP + FN) (Eq 5.31)."""
        denom = self.TP + self.FN
        return self.TP / denom if denom > 0 else 0.0

    sensitivity = recall
    true_positive_rate = recall

    @property
    def specificity(self) -> float:
        """Specificity / TNR = TN / (TN + FP) (Eq 5.32)."""
        denom = self.TN + self.FP
        return self.TN / denom if denom > 0 else 0.0

    true_negative_rate = specificity

    @property
    def false_positive_rate(self) -> float:
        """FPR = FP / (TN + FP) = 1 - Specificity (Eq 5.33)."""
        denom = self.TN + self.FP
        return self.FP / denom if denom > 0 else 0.0

    @property
    def f_score(self) -> float:
        """F-score = 2 * Precision * Recall / (Precision + Recall) (Eq 5.38 - 5.39)."""
        p, r = self.precision, self.recall
        denom = p + r
        return (2.0 * p * r) / denom if denom > 0 else 0.0

    f1_score = f_score

    @property
    def tp(self) -> int:
        return self.TP

    @property
    def fp(self) -> int:
        return self.FP

    @property
    def tn(self) -> int:
        return self.TN

    @property
    def fn(self) -> int:
        return self.FN

    def summary(self) -> Dict[str, float]:
        return {
            "N": self.N,
            "TP": self.TP,
            "FP": self.FP,
            "TN": self.TN,
            "FN": self.FN,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "false_positive_rate": self.false_positive_rate,
            "f_score": self.f_score
        }


# =====================================================================
# 5. ROC Curve and AUC (Section 5.2.6)
# =====================================================================

def compute_roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    num_thresholds: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute Receiver Operating Characteristic (ROC) curve:
    False Positive Rate (x-axis) vs True Positive Rate (y-axis), and AUC.
    y_true: shape (N,) binary labels 0 or 1
    scores: shape (N,) continuous predictive scores (e.g. p(C1 | x))
    Returns:
        fpr: array of FPR values
        tpr: array of TPR values
        thresholds: sorted thresholds
        auc: area under ROC curve
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()

    # Sort scores descending
    desc_idx = np.argsort(s)[::-1]
    s_sorted = s[desc_idx]
    y_sorted = y_t[desc_idx]

    distinct_threshold_indices = np.where(np.diff(s_sorted))[0]
    threshold_idxs = np.r_[distinct_threshold_indices, len(s_sorted) - 1]

    tps = np.cumsum(y_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_sorted == 0)[threshold_idxs]

    n_pos = np.sum(y_t == 1)
    n_neg = np.sum(y_t == 0)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both positive and negative samples are required to compute ROC.")

    tpr = tps / n_pos
    fpr = fps / n_neg

    # Include (0, 0) at top
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[s_sorted[0] + 1.0, s_sorted[threshold_idxs]]

    # Area Under Curve using trapezoidal rule
    try:
        from scipy.integrate import trapezoid
        auc = float(trapezoid(tpr, fpr))
    except ImportError:
        auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
    return fpr, tpr, thresholds, auc


# =====================================================================
# 6. Figures Reproduction (Figures 5.5 to 5.11)
# =====================================================================

def plot_figure_5_5_joint_probabilities(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """
    Faithful reproduction of Figure 5.5 (Book page 141):
    Schematic illustration of joint probabilities p(x, C_k) for two classes.
    (a) Decision boundary x = x_hat with sub-optimal threshold:
        - Green region: error from C2 misclassified as C1 (x < x0)
        - Red region: extra error from C2 misclassified as C1 (x0 < x < x_hat)
        - Blue region: error from C1 misclassified as C2 (x >= x_hat)
    (b) Decision boundary at optimal x_hat = x0 where red region vanishes.
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 9.5))

    x = np.linspace(-1.5, 6.5, 600)
    p_xC1 = 0.65 * stats.norm.pdf(x, 1.2, 0.75) + 0.35 * stats.norm.pdf(x, 3.8, 1.4)
    p_xC2 = 0.65 * stats.norm.pdf(x, 3.8, 0.95)

    # Intersection point x0 where p(x, C1) == p(x, C2)
    diff = p_xC1 - p_xC2
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    mid_idx = [i for i in sign_changes if 1.5 < x[i] < 3.5][0]
    x0 = float(x[mid_idx])

    # Subplot (a): x_hat > x0
    x_hat = 3.8

    for ax, current_xhat, is_opt in [(ax1, x_hat, False), (ax2, x0, True)]:
        ax.set_xlim(-1.2, 6.2)
        ax.set_ylim(0, 0.42)
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw curves
        ax.plot(x, p_xC1, color='black', lw=2.0, zorder=4)
        ax.plot(x, p_xC2, color='black', lw=2.0, zorder=4)

        # Region shading
        # 1. Green region: x <= x0, under p(x, C2)
        idx_green_left = x <= x0
        ax.fill_between(x[idx_green_left], 0, p_xC2[idx_green_left], color='#55bb55', alpha=0.85, zorder=2)

        if not is_opt:
            # 2. Green region: x0 < x <= x_hat, under p(x, C1)
            mid = (x > x0) & (x <= current_xhat)
            ax.fill_between(x[mid], 0, p_xC1[mid], color='#55bb55', alpha=0.85, zorder=2)

            # 3. Red region: x0 < x <= x_hat, between p(x, C1) and p(x, C2)
            ax.fill_between(x[mid], p_xC1[mid], p_xC2[mid], color='#ff6666', alpha=0.9, zorder=2)

            # 4. Blue region: x >= x_hat, under p(x, C1)
            idx_blue = x >= current_xhat
            ax.fill_between(x[idx_blue], 0, p_xC1[idx_blue], color='#6666ff', alpha=0.85, zorder=2)

            # Vertical decision line x_hat (solid black)
            ax.axvline(current_xhat, color='black', lw=2.0, zorder=5)
            # Dashed line at optimal x0
            ax.axvline(x0, color='black', linestyle='--', lw=1.5, zorder=5)

            # Red double-headed arrow showing varying x_hat
            ax.annotate('', xy=(current_xhat + 0.4, 0.35), xytext=(current_xhat - 0.4, 0.35),
                        arrowprops=dict(arrowstyle="<->", color='red', lw=2.0, mutation_scale=14), zorder=6)

            # X-axis labels
            ax.text(x0, -0.025, r'$x_0$', fontsize=14, ha='center', va='top')
            ax.text(current_xhat, -0.025, r'$\widehat{x}$', fontsize=15, ha='center', va='top')
            ax.text(2.3, -0.065, '(a)', fontsize=15, ha='center', va='top')
        else:
            # Optimal case: x_hat = x0, red region disappears!
            # Blue region: x >= x0, under p(x, C1)
            idx_blue = x >= x0
            ax.fill_between(x[idx_blue], 0, p_xC1[idx_blue], color='#6666ff', alpha=0.85, zorder=2)

            # Vertical decision line x_hat = x0 (solid black)
            ax.axvline(x0, color='black', lw=2.0, zorder=5)

            ax.text(x0, -0.025, r'$\widehat{x}$', fontsize=15, ha='center', va='top')
            ax.text(2.3, -0.065, '(b)', fontsize=15, ha='center', va='top')

        # Top arrows for R1 and R2
        div_x = current_xhat
        ax.annotate('', xy=(-1.0, 0.40), xytext=(div_x, 0.40),
                    arrowprops=dict(arrowstyle="<->", color='black', lw=1.5), zorder=6)
        ax.text((-1.0 + div_x) / 2, 0.41, r'$\mathcal{R}_1$', fontsize=15, ha='center')

        ax.annotate('', xy=(div_x, 0.40), xytext=(6.0, 0.40),
                    arrowprops=dict(arrowstyle="<->", color='black', lw=1.5), zorder=6)
        ax.text((div_x + 6.0) / 2, 0.41, r'$\mathcal{R}_2$', fontsize=15, ha='center')

        # Curve labels
        ax.text(0.4, 0.35, r'$p(x, \mathcal{C}_1)$', fontsize=14)
        ax.text(4.4, 0.28, r'$p(x, \mathcal{C}_2)$', fontsize=14)

    fig.suptitle("Figure 5.5: Joint Probabilities and Misclassification Regions",
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


def plot_figure_5_6_loss_matrix(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """
    Faithful reproduction of Figure 5.6 (Book page 142):
    Loss Matrix illustration for cancer diagnosis problem:
                  normal   cancer
        normal  (   0        1    )
        cancer  ( 100        0    )
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')

    table_data = [
        ["0", "1"],
        ["100", "0"]
    ]
    row_labels = ["normal", "cancer"]
    col_labels = ["normal", "cancer"]

    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0', '#f0f0f0'],
        rowColours=['#f0f0f0', '#f0f0f0']
    )
    table.scale(1.2, 2.0)
    table.set_fontsize(14)

    ax.text(0.5, 0.88, "Assigned Class", ha='center', va='center', fontsize=13, fontweight='bold', transform=ax.transAxes)
    ax.text(0.08, 0.5, "True Class", ha='center', va='center', rotation=90, fontsize=13, fontweight='bold', transform=ax.transAxes)
    ax.set_title("Figure 5.6: Loss Matrix for Cancer Diagnosis", fontsize=13, pad=15)

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


def plot_figure_5_7_reject_option(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """
    Faithful reproduction of Figure 5.7 (Book page 143):
    Illustration of the reject option:
    Inputs x where max(p(C1|x), p(C2|x)) <= theta are rejected.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    x = np.linspace(0.0, 4.5, 500)
    # Sigmoidal posteriors with realistic transition
    # p(C1 | x) decreases, p(C2 | x) increases
    z = -2.2 * (x - 2.25)
    p_C1 = 1.0 / (1.0 + np.exp(-z))
    p_C2 = 1.0 - p_C1

    theta = 0.88  # Rejection threshold

    # Find rejection boundaries where p_C1 == theta or p_C2 == theta
    idx_reject = np.where(np.maximum(p_C1, p_C2) <= theta)[0]
    x_left = float(x[idx_reject[0]])
    x_right = float(x[idx_reject[-1]])

    ax.set_xlim(-0.5, 4.8)
    ax.set_ylim(-0.1, 1.2)
    ax.axis('off')

    # Draw coordinate axes (black arrows)
    ax.annotate('', xy=(4.6, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5))
    ax.text(4.6, -0.06, r'$x$', fontsize=14)

    ax.annotate('', xy=(0, 1.12), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5))
    ax.text(-0.08, 0.0, '0.0', fontsize=13, ha='right', va='center')
    ax.text(-0.08, 1.0, '1.0', fontsize=13, ha='right', va='center')
    ax.text(-0.08, theta, r'$\theta$', fontsize=14, ha='right', va='center')

    # Plot posterior curves: blue for p(C1|x), red for p(C2|x)
    ax.plot(x, p_C1, color='blue', lw=2.5, label=r'$p(\mathcal{C}_1|x)$', zorder=4)
    ax.plot(x, p_C2, color='red', lw=2.5, label=r'$p(\mathcal{C}_2|x)$', zorder=4)

    # Threshold line: green dashed
    ax.plot([0, 4.6], [theta, theta], color='#00dd00', linestyle='--', lw=1.8, zorder=3)

    # Vertical green boundary lines for reject region
    ax.plot([x_left, x_left], [0, theta], color='#00dd00', lw=2.0, zorder=3)
    ax.plot([x_right, x_right], [0, theta], color='#00dd00', lw=2.0, zorder=3)

    # Two-sided arrow for reject region
    ax.annotate('', xy=(x_right, -0.02), xytext=(x_left, -0.02),
                arrowprops=dict(arrowstyle="<->", color='black', lw=2.0, mutation_scale=14))
    ax.text((x_left + x_right) / 2, -0.07, 'reject region', fontsize=13, ha='center')

    # Labels for posterior curves
    ax.text(0.3, 1.04, r'$p(\mathcal{C}_1|x)$', fontsize=14, color='black')
    ax.text(3.7, 1.04, r'$p(\mathcal{C}_2|x)$', fontsize=14, color='black')

    ax.set_title("Figure 5.7: Illustration of the Reject Option", fontsize=13, pad=15)
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


def plot_figure_5_8_class_densities_posteriors(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """
    Faithful reproduction of Figure 5.8 (Book page 145):
    Left: Class-conditional densities for two classes (bimodal Gaussian mixture for C1, unimodal for C2).
    Right: Corresponding posterior probabilities with optimal boundary at x ~ 0.56 (green vertical line).
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    x = np.linspace(0.0, 1.0, 500)

    # Class 1: Mixture of two Gaussians (blue)
    # Peak 1 at 0.2, Peak 2 at 0.5
    p_x_C1 = 0.5 * stats.norm.pdf(x, loc=0.2, scale=0.06) + 0.6 * stats.norm.pdf(x, loc=0.5, scale=0.08)

    # Class 2: Unimodal Gaussian (red) centered at 0.7
    p_x_C2 = 1.0 * stats.norm.pdf(x, loc=0.7, scale=0.09)

    # Normalize densities so they integrate to 1 (or match the scale in figure: peak around 4-5)
    # Subplot 1: Class-conditional densities
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 5.2)
    ax1.plot(x, p_x_C1, color='blue', lw=2.2, label=r'$p(x|\mathcal{C}_1)$')
    ax1.plot(x, p_x_C2, color='red', lw=2.2, label=r'$p(x|\mathcal{C}_2)$')
    ax1.set_xlabel(r'$x$', fontsize=13)
    ax1.set_ylabel('class densities', fontsize=13)
    ax1.text(0.12, 1.8, r'$p(x|\mathcal{C}_1)$', fontsize=13, color='black')
    ax1.text(0.68, 4.3, r'$p(x|\mathcal{C}_2)$', fontsize=13, color='black')

    # Subplot 2: Posterior probabilities assuming equal priors p(C1) = p(C2) = 0.5
    total = p_x_C1 + p_x_C2 + 1e-15
    post_C1 = p_x_C1 / total
    post_C2 = p_x_C2 / total

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.2)
    ax2.plot(x, post_C1, color='blue', lw=2.2)
    ax2.plot(x, post_C2, color='red', lw=2.2)
    ax2.set_xlabel(r'$x$', fontsize=13)

    # Decision boundary where post_C1 == post_C2 == 0.5
    cross_idx = np.where(np.diff(np.sign(post_C1 - post_C2)))[0]
    x_boundary = float(x[cross_idx[0]])
    ax2.axvline(x_boundary, color='#00dd00', lw=2.2, label='Decision boundary')

    ax2.text(0.2, 1.06, r'$p(\mathcal{C}_1|x)$', fontsize=13, color='black')
    ax2.text(0.8, 1.06, r'$p(\mathcal{C}_2|x)$', fontsize=13, color='black')

    fig.suptitle("Figure 5.8: Class-Conditional Densities and Posterior Probabilities",
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


def plot_figure_5_9_confusion_matrix(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """
    Faithful reproduction of Figure 5.9 (Book page 147):
    Confusion matrix for cancer treatment problem:
                  normal    cancer
        normal  (  N_TN      N_FP   )
        cancer  (  N_FN      N_TP   )
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')

    table_data = [
        [r"$N_{\mathrm{TN}}$", r"$N_{\mathrm{FP}}$"],
        [r"$N_{\mathrm{FN}}$", r"$N_{\mathrm{TP}}$"]
    ]
    row_labels = ["normal", "cancer"]
    col_labels = ["normal", "cancer"]

    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0', '#f0f0f0'],
        rowColours=['#f0f0f0', '#f0f0f0']
    )
    table.scale(1.2, 2.0)
    table.set_fontsize(14)

    ax.text(0.5, 0.88, "Decision Criterion", ha='center', va='center', fontsize=13, fontweight='bold', transform=ax.transAxes)
    ax.text(0.08, 0.5, "True Class", ha='center', va='center', rotation=90, fontsize=13, fontweight='bold', transform=ax.transAxes)
    ax.set_title("Figure 5.9: Confusion Matrix Notation", fontsize=13, pad=15)

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


def plot_figure_5_10_roc_regions(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """
    Faithful reproduction of Figure 5.10 (Book page 149):
    Joint distributions with labeled regions A, B, C, D, E for True/False Positives/Negatives.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    x = np.linspace(-1.5, 6.5, 600)
    p_xC1 = 0.65 * stats.norm.pdf(x, 1.2, 0.75) + 0.35 * stats.norm.pdf(x, 3.8, 1.4)
    p_xC2 = 0.65 * stats.norm.pdf(x, 3.8, 0.95)

    diff = p_xC1 - p_xC2
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    mid_idx = [i for i in sign_changes if 1.5 < x[i] < 3.5][0]
    x0 = float(x[mid_idx])
    x_hat = 3.8

    ax.set_xlim(-1.2, 6.2)
    ax.set_ylim(0, 0.42)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.plot(x, p_xC1, color='black', lw=2.0, zorder=4)
    ax.plot(x, p_xC2, color='black', lw=2.0, zorder=4)

    # Shading regions as in Figure 5.10
    # Region green: under p_xC2 for x <= x0, and under p_xC1 for x0 < x <= x_hat (which is Region C)
    ax.fill_between(x[x <= x0], 0, p_xC2[x <= x0], color='#55bb55', alpha=0.85, zorder=2)
    mid = (x > x0) & (x <= x_hat)
    ax.fill_between(x[mid], 0, p_xC1[mid], color='#55bb55', alpha=0.85, zorder=2)

    # Region B (red): between p_xC1 and p_xC2 for x0 < x <= x_hat
    ax.fill_between(x[mid], p_xC1[mid], p_xC2[mid], color='#ff6666', alpha=0.9, zorder=2)

    # Region E (blue): x >= x_hat under p_xC1
    right = x >= x_hat
    ax.fill_between(x[right], 0, p_xC1[right], color='#6666ff', alpha=0.85, zorder=2)

    # Vertical lines
    ax.axvline(x_hat, color='black', lw=2.0, zorder=5)
    ax.axvline(x0, color='black', linestyle='--', lw=1.5, zorder=5)

    # Region letter labels
    ax.text(1.2, 0.15, r'$A$', fontsize=16, ha='center', va='center')
    ax.text(3.3, 0.22, r'$B$', fontsize=16, ha='center', va='center')
    ax.text(3.3, 0.07, r'$C$', fontsize=16, ha='center', va='center')
    ax.text(4.2, 0.22, r'$D$', fontsize=16, ha='center', va='center')
    ax.text(4.2, 0.07, r'$E$', fontsize=16, ha='center', va='center')

    # Top arrow
    ax.annotate('', xy=(-1.0, 0.40), xytext=(x_hat, 0.40),
                arrowprops=dict(arrowstyle="<->", color='black', lw=1.5), zorder=6)
    ax.text((-1.0 + x_hat) / 2, 0.41, r'$\mathcal{R}_1$', fontsize=15, ha='center')

    ax.annotate('', xy=(x_hat, 0.40), xytext=(6.0, 0.40),
                arrowprops=dict(arrowstyle="<->", color='black', lw=1.5), zorder=6)
    ax.text((x_hat + 6.0) / 2, 0.41, r'$\mathcal{R}_2$', fontsize=15, ha='center')

    # Red double-headed arrow
    ax.annotate('', xy=(x_hat + 0.4, 0.35), xytext=(x_hat - 0.4, 0.35),
                arrowprops=dict(arrowstyle="<->", color='red', lw=2.0, mutation_scale=14), zorder=6)

    ax.text(x0, -0.025, r'$x_0$', fontsize=14, ha='center', va='top')
    ax.text(x_hat, -0.025, r'$\widehat{x}$', fontsize=15, ha='center', va='top')

    ax.text(0.4, 0.35, r'$p(x, \mathcal{C}_1)$', fontsize=14)
    ax.text(4.4, 0.28, r'$p(x, \mathcal{C}_2)$', fontsize=14)

    ax.set_title("Figure 5.10: Decision Boundary and Error Components (A, B, C, D, E)", fontsize=13, pad=15)
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


def plot_figure_5_11_roc_curve(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """
    Faithful reproduction of Figure 5.11 (Book page 150):
    Receiver Operating Characteristic (ROC) curve:
    True positive rate vs False positive rate for better (blue) and worse (red) classifiers,
    along with diagonal random classifier baseline.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 6))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')

    # Diagonal dashed line (random guessing)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1.8, label='Random classifier')

    # Synthetic ROC curves:
    # Blue curve (better classifier, AUC ~ 0.94)
    # Red curve (worse classifier, AUC ~ 0.82)
    np.random.seed(42)
    N = 100
    y_true = np.array([0] * (N // 2) + [1] * (N // 2))

    # Better classifier scores
    scores_better = np.r_[np.random.normal(0.0, 1.0, N // 2), np.random.normal(2.0, 1.0, N // 2)]
    fpr_b, tpr_b, _, auc_b = compute_roc_curve(y_true, scores_better)

    # Worse classifier scores
    scores_worse = np.r_[np.random.normal(0.0, 1.0, N // 2), np.random.normal(1.0, 1.0, N // 2)]
    fpr_w, tpr_w, _, auc_w = compute_roc_curve(y_true, scores_worse)

    # Step-like ROC curves matching Figure 5.11
    ax.step(fpr_b, tpr_b, where='post', color='blue', lw=2.2, label=f'Better classifier (AUC={auc_b:.2f})')
    ax.step(fpr_w, tpr_w, where='post', color='red', lw=2.0, label=f'Worse classifier (AUC={auc_w:.2f})')

    ax.set_xlabel('False positive rate', fontsize=13)
    ax.set_ylabel('True positive rate', fontsize=13)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.legend(loc='lower right', frameon=True)
    ax.set_title("Figure 5.11: Receiver Operating Characteristic (ROC) Curves", fontsize=13, pad=12)

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


def generate_all_section_5_2_figures(result_dirs: Optional[List[str]] = None) -> Dict[str, plt.Figure]:
    """
    Generate and save all Figure 5.5 to 5.11 to both 5/result/ and result/.
    """
    if result_dirs is None:
        result_dirs = ["5/result", "result"]
    for d in result_dirs:
        os.makedirs(d, exist_ok=True)

    figs = {}
    figs['fig_5_5'] = plot_figure_5_5_joint_probabilities(filepath=os.path.join(result_dirs[0], "fig_5_5_joint_probabilities.png"))
    figs['fig_5_6'] = plot_figure_5_6_loss_matrix(filepath=os.path.join(result_dirs[0], "fig_5_6_loss_matrix.png"))
    figs['fig_5_7'] = plot_figure_5_7_reject_option(filepath=os.path.join(result_dirs[0], "fig_5_7_reject_option.png"))
    figs['fig_5_8'] = plot_figure_5_8_class_densities_posteriors(filepath=os.path.join(result_dirs[0], "fig_5_8_class_densities_posteriors.png"))
    figs['fig_5_9'] = plot_figure_5_9_confusion_matrix(filepath=os.path.join(result_dirs[0], "fig_5_9_confusion_matrix.png"))
    figs['fig_5_10'] = plot_figure_5_10_roc_regions(filepath=os.path.join(result_dirs[0], "fig_5_10_roc_regions.png"))
    figs['fig_5_11'] = plot_figure_5_11_roc_curve(filepath=os.path.join(result_dirs[0], "fig_5_11_roc_curve.png"))

    return figs
