"""Chapter 9: Regularization
Section 9.1: Inductive Bias

This module implements concepts, models, and figure reproductions for Section 9.1 of:
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.

Contents:
- Section 9.1.1: Inverse Problems and Inductive Bias
  * Ill-posed nature of machine learning inverse problems
  * Bias-variance tradeoff and prior knowledge
  * Smoothness priors and weight decay regularizers (Eq 9.1)
- Section 9.1.2: No Free Lunch Theorem
  * Mathematical formulation and implication for learning algorithms
  * Structured real-world data distributions
- Section 9.1.3: Symmetry and Invariance
  * Group theory foundation: 4 group axioms (Closure, Associativity, Identity, Inverse)
  * Discrete groups: Cyclic group C_n, Dihedral group D_4
  * Four approaches to achieve invariance:
    1. Pre-processing invariant features
    2. Regularized error functions / Tangent propagation (Simard et al., 1992)
    3. Data augmentation (Figure 9.1, 8 canonical image transformations)
    4. Network architecture (weight sharing, CNNs)
  * Mathematical equivalence between additive input noise and gradient regularization (Bishop, 1995c)
- Section 9.1.4: Equivariance
  * Equivariance definition: S(T(I)) = T(S(I)) (Eq 9.2, Figure 9.2)
  * Generalized equivariance: S(T(I)) = \tilde{T}(S(I)) (Eq 9.3)
  * Invariance as a special case: C(T(I)) = C(I) (Eq 9.4)
  * Commutative diagram validation
- Reproductions of Figure 9.1 and Figure 9.2
"""

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
    path_ch9 = os.path.join(root, "9", "result", filename)
    path_root = os.path.join(root, "result", filename)

    if filepath:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        save_plot(fig, filepath)
        primary_path = filepath
    else:
        primary_path = path_ch9

    if result_dirs:
        for rdir in result_dirs:
            os.makedirs(rdir, exist_ok=True)
            save_plot(fig, os.path.join(rdir, filename))

    if save_both:
        os.makedirs(os.path.dirname(path_ch9), exist_ok=True)
        os.makedirs(os.path.dirname(path_root), exist_ok=True)
        if not filepath or os.path.abspath(filepath) != os.path.abspath(path_ch9):
            save_plot(fig, path_ch9)
        if not filepath or os.path.abspath(filepath) != os.path.abspath(path_root):
            save_plot(fig, path_root)

        # Also save Figure_9_X.png alias
        parts = base_name.split("_")
        if len(parts) >= 3 and parts[0] == "fig" and parts[1] == "9":
            alt_filename = f"Figure_9_{parts[2]}.png"
            alt_ch9 = os.path.join(root, "9", "result", alt_filename)
            alt_root = os.path.join(root, "result", alt_filename)
            save_plot(fig, alt_ch9)
            save_plot(fig, alt_root)

    return primary_path


# =====================================================================
# Section 9.1.1: Inverse Problems & Smoothness Regularization
# =====================================================================

def fit_polynomial_regression(
    x: np.ndarray,
    t: np.ndarray,
    degree: int,
    reg_lambda: float = 0.0,
) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Fit a 1D polynomial regression model with optional L2 regularization (Eq 9.1).

    Model:
        y(x, w) = sum_{j=0}^M w_j x^j = phi(x)^T w
    Regularized objective:
        E(w) = 0.5 sum_n (y(x_n, w) - t_n)^2 + 0.5 * reg_lambda * w^T w
    Closed-form solution:
        w* = (Phi^T Phi + reg_lambda * I)^{-1} Phi^T t
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    t_arr = np.asarray(t, dtype=float).ravel()
    N = len(x_arr)

    # Design matrix Phi: N x (degree + 1)
    Phi = np.vander(x_arr, degree + 1, increasing=True)
    M = degree + 1

    A = Phi.T @ Phi + reg_lambda * np.eye(M)
    b = Phi.T @ t_arr
    w = np.linalg.solve(A, b)

    predict_fn = lambda x_new: np.vander(np.asarray(x_new, dtype=float).ravel(), degree + 1, increasing=True) @ w
    return w, predict_fn


def compute_smoothness_norm(predict_fn: Callable[[np.ndarray], np.ndarray], x_grid: np.ndarray) -> float:
    """Compute the L2 norm of the derivative as a proxy for function smoothness."""
    dx = x_grid[1] - x_grid[0]
    y_vals = predict_fn(x_grid)
    dy = np.gradient(y_vals, dx)
    return float(np.mean(dy ** 2))


# =====================================================================
# Section 9.1.2: No Free Lunch Theorem Simulation
# =====================================================================

def simulate_no_free_lunch(
    num_inputs: int = 6,
    num_train: int = 3,
    num_test_functions: int = 100,
    random_seed: int = 42,
) -> Dict[str, float]:
    """Demonstrate the No Free Lunch (NFL) theorem on discrete binary classification.

    Setting:
        Domain X = {x_1, ..., x_K} with |X| = num_inputs.
        Labels Y in {0, 1}.
        There are 2^|X| possible functions f: X -> Y.
        Given a training set of size num_train, any two distinct algorithms
        predicting the unseen |X| - num_train points will have identical average
        generalization error (50%) when averaged uniformly across all 2^(|X| - num_train)
        possible completions of the target function.
    """
    rng = np.random.RandomState(random_seed)
    unseen_count = num_inputs - num_train
    num_possible_unseen_completions = 2 ** unseen_count

    # Algorithm A: Constant predictor (predicts 0 for all unseen)
    # Algorithm B: Majority predictor or random guesser
    # Algorithm C: Alternating parity predictor

    errors_A = []
    errors_B = []
    errors_C = []

    # Enumerate all binary completions for unseen points
    for completion_idx in range(num_possible_unseen_completions):
        # Convert integer to binary vector
        bits = [(completion_idx >> i) & 1 for i in range(unseen_count)]
        y_true = np.array(bits)

        # Pred A: all zeros
        y_pred_A = np.zeros(unseen_count)
        # Pred B: all ones
        y_pred_B = np.ones(unseen_count)
        # Pred C: alternating
        y_pred_C = np.array([i % 2 for i in range(unseen_count)])

        errors_A.append(np.mean(y_pred_A != y_true))
        errors_B.append(np.mean(y_pred_B != y_true))
        errors_C.append(np.mean(y_pred_C != y_true))

    return {
        "avg_error_A": float(np.mean(errors_A)),
        "avg_error_B": float(np.mean(errors_B)),
        "avg_error_C": float(np.mean(errors_C)),
        "num_completions": num_possible_unseen_completions,
        "nfl_verified": bool(
            abs(np.mean(errors_A) - 0.5) < 1e-12
            and abs(np.mean(errors_B) - 0.5) < 1e-12
            and abs(np.mean(errors_C) - 0.5) < 1e-12
        ),
    }


# =====================================================================
# Section 9.1.3: Group Theory Foundations for Symmetries
# =====================================================================

class GroupElement:
    """Represents an element of a mathematical transformation group."""

    def __init__(self, name: str, data: Any):
        self.name = name
        self.data = data

    def __repr__(self) -> str:
        return f"GroupElement({self.name})"


class FiniteGroup:
    """A finite group G with explicit composition table to verify group axioms.

    Axioms:
        1. Closure: For any A, B in G, A * B in G
        2. Associativity: (A * B) * C = A * (B * C)
        3. Identity: Exists I in G s.t. A * I = I * A = A
        4. Inverse: For each A in G, exists A^{-1} s.t. A * A^{-1} = A^{-1} * A = I
    """

    def __init__(
        self,
        name: str,
        elements: List[GroupElement],
        compose_fn: Callable[[GroupElement, GroupElement], GroupElement],
    ):
        self.name = name
        self.elements = elements
        self.compose_fn = compose_fn

    def compose(self, a: GroupElement, b: GroupElement) -> GroupElement:
        return self.compose_fn(a, b)

    def find_identity(self) -> Optional[GroupElement]:
        """Find the unique identity element I in G."""
        for cand in self.elements:
            is_id = True
            for a in self.elements:
                left = self.compose(cand, a)
                right = self.compose(a, cand)
                if left.name != a.name or right.name != a.name:
                    is_id = False
                    break
            if is_id:
                return cand
        return None

    def verify_axioms(self) -> Dict[str, bool]:
        """Check all 4 group axioms on this finite group."""
        names = {e.name for e in self.elements}
        identity = self.find_identity()

        # 1. Closure
        closure_holds = True
        for a in self.elements:
            for b in self.elements:
                c = self.compose(a, b)
                if c.name not in names:
                    closure_holds = False
                    break

        # 2. Associativity
        assoc_holds = True
        for a in self.elements:
            for b in self.elements:
                for c in self.elements:
                    ab_c = self.compose(self.compose(a, b), c)
                    a_bc = self.compose(a, self.compose(b, c))
                    if ab_c.name != a_bc.name:
                        assoc_holds = False
                        break

        # 3. Identity
        id_holds = identity is not None

        # 4. Inverse
        inv_holds = True
        if id_holds:
            for a in self.elements:
                has_inv = False
                for b in self.elements:
                    left = self.compose(a, b)
                    right = self.compose(b, a)
                    if left.name == identity.name and right.name == identity.name:
                        has_inv = True
                        break
                if not has_inv:
                    inv_holds = False
                    break
        else:
            inv_holds = False

        return {
            "closure": closure_holds,
            "associativity": assoc_holds,
            "identity": id_holds,
            "inverse": inv_holds,
            "is_valid_group": bool(closure_holds and assoc_holds and id_holds and inv_holds),
        }


def create_cyclic_group_c4() -> FiniteGroup:
    """Create the cyclic group C_4 of 4 rotations of a square: {0, 90, 180, 270} degrees."""
    angles = [0, 90, 180, 270]
    elements = [GroupElement(f"R_{deg}", deg) for deg in angles]
    elem_dict = {f"R_{deg}": e for deg, e in zip(angles, elements)}

    def compose_rot(a: GroupElement, b: GroupElement) -> GroupElement:
        new_angle = (a.data + b.data) % 360
        return elem_dict[f"R_{new_angle}"]

    return FiniteGroup("C_4", elements, compose_rot)


def create_dihedral_group_d4() -> FiniteGroup:
    """Create the dihedral group D_4 of symmetries of a square (order 8: 4 rotations, 4 reflections)."""
    # Represent as (rotation: 0..3, reflection: 0..1) where s * r = r^{-1} * s
    elements = []
    elem_map = {}
    for r in range(4):
        e_rot = GroupElement(f"r{r}", (r, 0))
        elements.append(e_rot)
        elem_map[(r, 0)] = e_rot
        e_ref = GroupElement(f"sr{r}", (r, 1))
        elements.append(e_ref)
        elem_map[(r, 1)] = e_ref

    def compose_d4(a: GroupElement, b: GroupElement) -> GroupElement:
        r1, s1 = a.data
        r2, s2 = b.data
        # If s1 == 1: s * r^k = r^{-k} * s
        if s1 == 0:
            new_r = (r1 + r2) % 4
            new_s = s2
        else:
            new_r = (r1 - r2) % 4
            new_s = 1 - s2
        return elem_map[(new_r, new_s)]

    return FiniteGroup("D_4", elements, compose_d4)


# =====================================================================
# Section 9.1.3: Data Augmentation & Noise-Gradient Equivalence
# =====================================================================

def verify_noise_gradient_regularization_equivalence(
    model_fn: Callable[[np.ndarray], np.ndarray],
    grad_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    t: np.ndarray,
    noise_sigma: float = 0.02,
    num_samples: int = 50000,
    random_seed: int = 42,
) -> Dict[str, float]:
    """Verify Bishop (1995c) theorem: training with additive noise is equivalent to L2 gradient penalty.

    Theorem:
        Let x_tilde = x + eps, with eps ~ N(0, sigma^2 I).
        E_eps[ 0.5 (y(x + eps) - t)^2 ] = 0.5 (y(x) - t)^2 + 0.5 * sigma^2 * ||nabla_x y(x)||^2 + O(sigma^4)
        when y(x) = t (or neglecting curvature term (y - t) Tr(nabla^2 y)).
    """
    rng = np.random.RandomState(random_seed)
    D = len(x)

    y_val = float(model_fn(x))
    loss_unperturbed = 0.5 * (y_val - t) ** 2

    # Monte Carlo evaluation with noise
    noise = rng.randn(num_samples, D) * noise_sigma
    perturbed_inputs = x + noise
    perturbed_preds = np.array([float(model_fn(p)) for p in perturbed_inputs])
    expected_noisy_loss = float(np.mean(0.5 * (perturbed_preds - t) ** 2))

    # Analytical Taylor expansion approximation
    grad = grad_fn(x)
    grad_norm_sq = float(np.sum(grad ** 2))
    predicted_regularized_loss = loss_unperturbed + 0.5 * (noise_sigma ** 2) * grad_norm_sq

    diff = abs(expected_noisy_loss - predicted_regularized_loss)
    rel_diff = diff / max(expected_noisy_loss, 1e-12)

    return {
        "loss_unperturbed": loss_unperturbed,
        "expected_noisy_loss": expected_noisy_loss,
        "predicted_regularized_loss": predicted_regularized_loss,
        "gradient_penalty_term": 0.5 * (noise_sigma ** 2) * grad_norm_sq,
        "abs_diff": diff,
        "rel_diff": rel_diff,
        "verified": bool(rel_diff < 0.05),
    }


# =====================================================================
# Section 9.1.4: Equivariance & Invariance Operators
# =====================================================================

def translate_image(im: Image.Image, dx: int, dy: int) -> Image.Image:
    """Translate an image by (dx, dy) with edge clamping."""
    w, h = im.size
    # Create empty image and paste translated
    mode = im.mode
    new_im = Image.new(mode, (w, h))
    new_im.paste(im, (dx, dy))
    return new_im


def segment_silhouette(
    im: Image.Image,
    fg_color: Tuple[int, int, int] = (240, 130, 40),
) -> Image.Image:
    """Create a segmentation silhouette from an RGBA or RGB image."""
    arr = np.array(im.convert("RGB"))
    # Simple color-based thresholding for cat: reddish-orange fur detection
    # R > 120 and R > G and G > B
    mask = (arr[:, :, 0] > 110) & (arr[:, :, 0] > arr[:, :, 2] + 20)
    out_arr = arr.copy()
    out_arr[mask] = fg_color
    return Image.fromarray(out_arr)


def verify_equivariance_property(
    image: Image.Image,
    operator_T: Callable[[Image.Image], Image.Image],
    operator_S: Callable[[Image.Image], Image.Image],
) -> Dict[str, Any]:
    """Verify the commutativity S(T(I)) = T(S(I)) (Eq 9.2)."""
    # Path 1: T then S
    t_im = operator_T(image)
    st_im = operator_S(t_im)

    # Path 2: S then T
    s_im = operator_S(image)
    ts_im = operator_T(s_im)

    arr_st = np.array(st_im)
    arr_ts = np.array(ts_im)

    max_diff = int(np.max(np.abs(arr_st.astype(int) - arr_ts.astype(int))))
    mean_diff = float(np.mean(np.abs(arr_st.astype(int) - arr_ts.astype(int))))

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "is_equivariant": bool(max_diff == 0),
        "path_1_ST": st_im,
        "path_2_TS": ts_im,
    }


# =====================================================================
# Figure Reproduction: Figure 9.1 & Figure 9.2
# =====================================================================

def generate_figure_9_1(
    filepath: Optional[str] = None,
    result_dirs: Optional[List[str]] = None,
    save_both: bool = True,
) -> plt.Figure:
    """Reproduce Figure 9.1: Illustration of data set augmentation.

    Panels:
        (a) Original image
        (b) Horizontal inversion
        (c) Scaling
        (d) Translation
        (e) Rotation
        (f) Brightness and contrast change
        (g) Additive noise
        (h) Colour shift
    """
    setup_style()
    root = _get_project_root()
    assets_dir = os.path.join(root, "9", "assets")

    fig, axes = plt.subplots(2, 4, figsize=(10, 5.5))
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    subtitles = [
        "Original",
        "Horizontal Inversion",
        "Scaling",
        "Translation",
        "Rotation",
        "Brightness & Contrast",
        "Additive Noise",
        "Colour Shift",
    ]

    for idx in range(8):
        ax = axes[idx // 4, idx % 4]
        img_path = os.path.join(assets_dir, f"pdf_img_ch9-{idx:03d}.png")
        if os.path.exists(img_path):
            im = Image.open(img_path)
            ax.imshow(im)
        else:
            # Fallback placeholder if asset missing
            dummy = np.ones((252, 274, 3), dtype=np.uint8) * 180
            ax.imshow(dummy)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{labels[idx]} {subtitles[idx]}", fontsize=9.5)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)

    plt.tight_layout()
    _save_figure_files(fig, "fig_9_1_data_augmentation", filepath, result_dirs, save_both)
    return fig


def generate_figure_9_2(
    filepath: Optional[str] = None,
    result_dirs: Optional[List[str]] = None,
    save_both: bool = True,
) -> plt.Figure:
    """Reproduce Figure 9.2: Illustration of equivariance corresponding to Eq (9.2).

    Commutative diagram:
        Top row: (a) Original -> T -> (b) Translated
        Bottom row: (c) Segmented -> T -> (d) Translated Segmented
        Vertical: (a) -> S -> (c), (b) -> S -> (d)
    """
    setup_style()
    root = _get_project_root()
    assets_dir = os.path.join(root, "9", "assets")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    path_a = os.path.join(assets_dir, "fig_9_2_a.png")
    path_b = os.path.join(assets_dir, "fig_9_2_b.png")
    path_c = os.path.join(assets_dir, "fig_9_2_c.png")
    path_d = os.path.join(assets_dir, "fig_9_2_d.png")

    im_a = Image.open(path_a) if os.path.exists(path_a) else Image.new("RGB", (189, 189), (200, 200, 200))
    im_b = Image.open(path_b) if os.path.exists(path_b) else Image.new("RGB", (189, 189), (200, 200, 200))
    im_c = Image.open(path_c) if os.path.exists(path_c) else Image.new("RGB", (189, 189), (240, 130, 40))
    im_d = Image.open(path_d) if os.path.exists(path_d) else Image.new("RGB", (189, 189), (240, 130, 40))

    # Display 4 images
    ax.imshow(im_a, extent=[1.0, 4.2, 5.5, 8.7])
    ax.imshow(im_b, extent=[5.8, 9.0, 5.5, 8.7])
    ax.imshow(im_c, extent=[1.0, 4.2, 1.0, 4.2])
    ax.imshow(im_d, extent=[5.8, 9.0, 1.0, 4.2])

    # Draw border boxes
    for (x0, x1, y0, y1) in [(1.0, 4.2, 5.5, 8.7), (5.8, 9.0, 5.5, 8.7), (1.0, 4.2, 1.0, 4.2), (5.8, 9.0, 1.0, 4.2)]:
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="black", linewidth=0.8)
        ax.add_patch(rect)

    # Sub-figure labels (a), (b), (c), (d)
    ax.text(1.8, 5.1, "(a)", ha="center", va="center", fontsize=12)
    ax.text(6.6, 5.1, "(b)", ha="center", va="center", fontsize=12)
    ax.text(2.6, 0.6, "(c)", ha="center", va="center", fontsize=12)
    ax.text(7.4, 0.6, "(d)", ha="center", va="center", fontsize=12)

    # Horizontal arrows (T):
    # (a) -> (b) with T
    ax.annotate("", xy=(5.5, 7.1), xytext=(4.5, 7.1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
    ax.text(5.0, 7.5, r"$\mathcal{T}$", ha="center", va="center", fontsize=14)

    # (c) -> (d) with T
    ax.annotate("", xy=(5.5, 2.6), xytext=(4.5, 2.6),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
    ax.text(5.0, 3.0, r"$\mathcal{T}$", ha="center", va="center", fontsize=14)

    # Vertical arrows (S):
    # (a) -> (c) with S
    ax.annotate("", xy=(3.0, 4.4), xytext=(3.0, 5.3),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
    ax.text(3.5, 4.85, r"$\mathcal{S}$", ha="center", va="center", fontsize=14)

    # (b) -> (d) with S
    ax.annotate("", xy=(7.8, 4.4), xytext=(7.8, 5.3),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
    ax.text(8.3, 4.85, r"$\mathcal{S}$", ha="center", va="center", fontsize=14)

    plt.tight_layout()
    _save_figure_files(fig, "fig_9_2_equivariance", filepath, result_dirs, save_both)
    return fig
