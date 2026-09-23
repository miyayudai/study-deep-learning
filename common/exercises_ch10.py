"""
common/exercises_ch10.py
========================
Chapter 10: Convolutional Networks - Solutions and Verifications for Exercises 10.1 - 10.13
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module provides complete analytical derivations, numerical simulations,
and self-grading verifications for all 13 exercises in Chapter 10:
- Exercise 10.1: Unit-norm linear projection maximization (x = w / ||w||)
- Exercise 10.2: 1D convolution as Toeplitz matrix multiplication
- Exercise 10.3: 2D cross-correlation vs 2D convolution (180-degree flipped filter)
- Exercise 10.4: Batch normalization in CNNs (spatial translation equivariance)
- Exercise 10.5: Commutativity of continuous 1D convolution
- Exercise 10.6: Same-padding formula P = (M - 1) / 2 for odd filter size
- Exercise 10.7: Strided convolution output dimension formula
- Exercise 10.8: VGG-16 parameter and MACs reduction (stacked 3x3 vs 5x5 / 7x7)
- Exercise 10.9: Separable 2D convolution decomposition (rank-1 filter)
- Exercise 10.10: DeepDream gradient equivalence to activation values
- Exercise 10.11: Object detection class probability parameterizations
- Exercise 10.12: Sliding window vs convolutional acceleration complexity
- Exercise 10.13: Transposed convolution as matrix transpose A^T duality
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# =============================================================================
# Exercise 10.1: Unit-Norm Linear Projection Maximization
# =============================================================================
def solve_exercise_10_1(w: np.ndarray) -> Dict[str, Any]:
    """
    Exercise 10.1: Show that the unit-norm vector x that maximizes y = w^T x
    is x* = w / ||w||, giving maximum value y* = ||w||.
    """
    w = np.asarray(w, dtype=np.float64)
    norm_w = np.linalg.norm(w)
    if norm_w == 0:
        optimal_x = np.zeros_like(w)
        max_y = 0.0
    else:
        optimal_x = w / norm_w
        max_y = norm_w

    return {
        "optimal_x": optimal_x,
        "max_y": float(max_y),
        "norm_w": float(norm_w),
        "cauchy_schwarz_bound": float(norm_w),
    }


# =============================================================================
# Exercise 10.2: 1D Convolution as Toeplitz Matrix Multiplication
# =============================================================================
def solve_exercise_10_2(D: int, w: np.ndarray) -> Dict[str, Any]:
    """
    Exercise 10.2: Construct the (D - M + 1) x D Toeplitz convolution matrix A
    such that valid convolution y = w * x equals A @ x.
    """
    w = np.asarray(w, dtype=np.float64)
    M = len(w)
    out_dim = D - M + 1
    A = np.zeros((out_dim, D), dtype=np.float64)

    for i in range(out_dim):
        A[i, i : i + M] = w

    return {
        "matrix_A": A,
        "shape": A.shape,
        "expected_shape": (out_dim, D),
    }


# =============================================================================
# Exercise 10.3: 2D Cross-Correlation vs 2D Convolution
# =============================================================================
def solve_exercise_10_3(X: np.ndarray, W: np.ndarray) -> Dict[str, Any]:
    """
    Exercise 10.3: Verify that 2D cross-correlation with filter W equals
    2D convolution with 180-degree flipped filter W' where W'_{mn} = W_{-m, -n}.
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    H, W_dim = X.shape
    M1, M2 = W.shape

    out_h = H - M1 + 1
    out_w = W_dim - M2 + 1

    # Cross-correlation: sum_{m, n} W_{m, n} * X_{i+m, j+n}
    corr_out = np.zeros((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            corr_out[i, j] = np.sum(W * X[i : i + M1, j : j + M2])

    # Flipped filter for convolution: 180-degree rotation (W'_{mn} = W_{M1-1-m, M2-1-n})
    W_flipped = np.flip(W, axis=(0, 1))

    # Mathematical 2D convolution with W_flipped:
    # (W_flipped * X)[i, j] = sum_{m, n} W_flipped[m, n] * X[i + M1 - 1 - m, j + M2 - 1 - n]
    conv_out = np.zeros((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            patch = X[i : i + M1, j : j + M2]
            conv_out[i, j] = np.sum(W_flipped * np.flip(patch, axis=(0, 1)))

    return {
        "cross_correlation": corr_out,
        "flipped_filter": W_flipped,
        "convolution_with_flipped": conv_out,
        "is_equivalent": bool(np.allclose(corr_out, conv_out)),
    }


# =============================================================================
# Exercise 10.4: Batch Normalization in CNNs
# =============================================================================
def solve_exercise_10_4() -> Dict[str, Any]:
    """
    Exercise 10.4: Spatial translation equivariance rationale for BatchNorm in CNNs.
    Returns theoretical derivation and channel-wise vs spatial statistics comparison.
    """
    explanation = (
        "In convolutional layers, feature maps are equivariant to spatial translations: "
        "if an object shifts by (delta_x, delta_y), the corresponding activations shift identically. "
        "To preserve this translation equivariance, the normalization must be invariant to spatial "
        "location. Therefore, a single mean mu_k and variance sigma_k^2 are computed per channel k "
        "across both the batch dimension N and spatial dimensions H x W (N * H * W elements total)."
    )
    return {
        "explanation": explanation,
        "shared_dimensions": ("batch_size", "height", "width"),
        "independent_dimensions": ("channels",),
    }


# =============================================================================
# Exercise 10.5: Continuous 1D Convolution Commutativity
# =============================================================================
def solve_exercise_10_5(f_vals: np.ndarray, g_vals: np.ndarray, dt: float = 0.05) -> Dict[str, Any]:
    """
    Exercise 10.5: Commutativity of continuous convolution: (f * g)(t) = (g * f)(t).
    Numerical integration verification via Simpson / Riemann sum.
    """
    # Discrete Riemann sum approximation of (f * g)(t)
    conv_fg = np.convolve(f_vals, g_vals, mode='full') * dt
    conv_gf = np.convolve(g_vals, f_vals, mode='full') * dt

    max_diff = float(np.max(np.abs(conv_fg - conv_gf)))
    return {
        "conv_fg": conv_fg,
        "conv_gf": conv_gf,
        "max_diff": max_diff,
        "is_commutative": np.allclose(conv_fg, conv_gf),
    }


# =============================================================================
# Exercise 10.6: Same Padding Formula
# =============================================================================
def solve_exercise_10_6(M: int) -> Dict[str, Any]:
    """
    Exercise 10.6: Derive zero padding P such that W_out = W for stride S = 1.
    Formula: W_out = W - M + 2P + 1 = W  ==>  2P = M - 1  ==>  P = (M - 1) / 2.
    """
    assert M % 2 == 1, "Filter size M must be odd for symmetric same-padding."
    P = (M - 1) // 2
    return {
        "filter_size_M": M,
        "same_padding_P": P,
        "formula": "P = (M - 1) / 2",
    }


# =============================================================================
# Exercise 10.7: Strided Convolution Output Dimension Formula
# =============================================================================
def solve_exercise_10_7(W: int, M: int, S: int, P: int) -> Dict[str, Any]:
    """
    Exercise 10.7: Output dimension formula: W_out = floor((W - M + 2P) / S) + 1.
    """
    numerator = W - M + 2 * P
    w_out = (numerator // S) + 1
    has_remainder = (numerator % S) != 0
    return {
        "W_out": w_out,
        "numerator": numerator,
        "has_remainder": has_remainder,
        "discarded_pixels": numerator % S,
    }


# =============================================================================
# Exercise 10.8: VGG-16 Parameter and MACs Reduction
# =============================================================================
def solve_exercise_10_8(C: int = 64) -> Dict[str, Any]:
    """
    Exercise 10.8: Compare parameter counts:
    (a) Two stacked 3x3 convs vs One 5x5 conv (both receptive field 5x5).
    (b) Three stacked 3x3 convs vs One 7x7 conv (both receptive field 7x7).
    """
    # (a)
    params_two_3x3 = 2 * (3 * 3 * C * C)      # 18 C^2
    params_one_5x5 = 5 * 5 * C * C            # 25 C^2
    ratio_a = params_two_3x3 / params_one_5x5  # 18 / 25 = 0.72

    # (b)
    params_three_3x3 = 3 * (3 * 3 * C * C)    # 27 C^2
    params_one_7x7 = 7 * 7 * C * C            # 49 C^2
    ratio_b = params_three_3x3 / params_one_7x7  # 27 / 49 approx 0.551

    return {
        "params_two_3x3": params_two_3x3,
        "params_one_5x5": params_one_5x5,
        "ratio_5x5": ratio_a,
        "params_three_3x3": params_three_3x3,
        "params_one_7x7": params_one_7x7,
        "ratio_7x7": ratio_b,
    }


# =============================================================================
# Exercise 10.9: Separable 2D Convolution Decomposition
# =============================================================================
def solve_exercise_10_9(u: np.ndarray, v: np.ndarray, X: np.ndarray) -> Dict[str, Any]:
    """
    Exercise 10.9: Rank-1 filter W = u @ v^T.
    2D convolution with W equals 1D horizontal conv with v^T followed by 1D vertical conv with u.
    Computational operations per pixel drops from M^2 to 2M (speedup M / 2).
    """
    u = np.asarray(u, dtype=np.float64).reshape(-1, 1)  # (M, 1)
    v = np.asarray(v, dtype=np.float64).reshape(-1, 1)  # (M, 1)
    M = len(u)
    W_2d = u @ v.T  # (M, M)

    H, W_dim = X.shape
    out_h = H - M + 1
    out_w = W_dim - M + 1

    # Standard 2D convolution (cross-correlation)
    out_standard = np.zeros((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            out_standard[i, j] = np.sum(W_2d * X[i : i + M, j : j + M])

    # Separable 1D horizontal then 1D vertical
    # Step 1: horizontal conv with v along each row: size (H, out_w)
    interm = np.zeros((H, out_w), dtype=np.float64)
    v_flat = v.ravel()
    for i in range(H):
        for j in range(out_w):
            interm[i, j] = np.sum(v_flat * X[i, j : j + M])

    # Step 2: vertical conv with u along columns of interm: size (out_h, out_w)
    out_separable = np.zeros((out_h, out_w), dtype=np.float64)
    u_flat = u.ravel()
    for i in range(out_h):
        for j in range(out_w):
            out_separable[i, j] = np.sum(u_flat * interm[i : i + M, j])

    speedup = (M * M) / (2.0 * M)

    return {
        "W_2d": W_2d,
        "out_standard": out_standard,
        "out_separable": out_separable,
        "is_equal": np.allclose(out_standard, out_separable),
        "ops_standard": M * M,
        "ops_separable": 2 * M,
        "theoretical_speedup": speedup,
    }


# =============================================================================
# Exercise 10.10: DeepDream Gradient Equivalence
# =============================================================================
def solve_exercise_10_10(a: np.ndarray) -> Dict[str, Any]:
    """
    Exercise 10.10: DeepDream objective E = 0.5 * sum_{i,j,k} a_{ijk}^2.
    Gradient dE / da_{ijk} = a_{ijk}.
    """
    a = np.asarray(a, dtype=np.float64)
    E = 0.5 * np.sum(a ** 2)
    grad_a = a.copy()
    return {
        "objective": float(E),
        "analytical_gradient": grad_a,
        "is_identical_to_activation": np.array_equal(grad_a, a),
    }


# =============================================================================
# Exercise 10.11: Object Detection Class Probabilities
# =============================================================================
def solve_exercise_10_11(logits_single: np.ndarray, logit_obj: float, logits_cond: np.ndarray) -> Dict[str, Any]:
    """
    Exercise 10.11: Compare single softmax over (K+1) vs objectness sigmoid + K-class softmax.
    """
    # (a) Single softmax over (K+1) classes (0 is background)
    exp_single = np.exp(logits_single - np.max(logits_single))
    p_single = exp_single / np.sum(exp_single)

    # (b) Decoupled objectness + conditional
    p_obj = 1.0 / (1.0 + np.exp(-logit_obj))
    exp_cond = np.exp(logits_cond - np.max(logits_cond))
    p_cond = exp_cond / np.sum(exp_cond)
    p_joint = p_obj * p_cond

    return {
        "p_single": p_single,
        "p_obj": float(p_obj),
        "p_cond": p_cond,
        "p_joint": p_joint,
    }


# =============================================================================
# Exercise 10.12: Sliding Window vs Convolutional Acceleration
# =============================================================================
def solve_exercise_10_12() -> Dict[str, Any]:
    """
    Exercise 10.12: Bishop (2024, p. 352) sliding window complexity.
    Input image 8x8, base CNN takes 6x6.
    Conv1: 3x3 filter -> 4x4 (for 6x6 input) or 6x6 (for 8x8 input)
    Conv2: 3x3 filter -> 2x2 (for 6x6 input) or 4x4 (for 8x8 input)
    Conv3: 2x2 filter -> 1x1 (for 6x6 input) or 3x3 (for 8x8 input)
    """
    # Naive: 9 windows of size 6x6
    # Each window:
    # Conv1: 4 * 4 * (3 * 3) = 16 * 9 = 144
    # Conv2: 2 * 2 * (3 * 3) = 4 * 9 = 36
    # Conv3: 1 * 1 * (2 * 2) = 1 * 4 = 4
    # Total per window = 144 + 36 + 4 = 184
    # For 9 windows = 9 * 184 = 1656
    naive_conv1 = 9 * (4 * 4 * 9)
    naive_conv2 = 9 * (2 * 2 * 9)
    naive_conv3 = 9 * (1 * 1 * 4)
    naive_total = naive_conv1 + naive_conv2 + naive_conv3

    # Convolutional: 8x8 evaluated once
    # Conv1: 8x8 with 3x3 filter -> 6x6 output: 6 * 6 * 9 = 324
    # Conv2: 6x6 with 3x3 filter -> 4x4 output: 4 * 4 * 9 = 144
    # Conv3: 4x4 with 2x2 filter -> 3x3 output: 3 * 3 * 4 = 36
    conv_conv1 = 6 * 6 * 9
    conv_conv2 = 4 * 4 * 9
    conv_conv3 = 3 * 3 * 4
    conv_total = conv_conv1 + conv_conv2 + conv_conv3

    speedup = naive_total / conv_total

    return {
        "naive_multiplications": naive_total,
        "conv_multiplications": conv_total,
        "speedup_factor": float(speedup),
    }


# =============================================================================
# Exercise 10.13: Transposed Convolution as Matrix Transpose A^T Duality
# =============================================================================
def solve_exercise_10_13(x: np.ndarray, y: np.ndarray, w: np.ndarray, S: int = 2) -> Dict[str, Any]:
    """
    Exercise 10.13: Duality <A x, y> = <x, A^T y>.
    Downsampling convolution is y_hat = A @ x.
    Transposed convolution is z_hat = A.T @ y.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    M = len(w)
    D = len(x)
    D_prime = (D - M) // S + 1
    assert len(y) == D_prime, f"y length {len(y)} must match D_prime {D_prime}"

    # Build downsampling matrix A of size (D_prime, D)
    A = np.zeros((D_prime, D), dtype=np.float64)
    for i in range(D_prime):
        A[i, i * S : i * S + M] = w

    # Forward: Ax
    Ax = A @ x
    # Adjoint / Transpose: A^T y
    ATy = A.T @ y

    # Inner products: <Ax, y> and <x, A^T y>
    inner_1 = float(np.dot(Ax, y))
    inner_2 = float(np.dot(x, ATy))

    # Transposed convolution direct expansion formula:
    # Hin = D_prime, M = len(w), stride = S, padding = 0
    # Hout = (D_prime - 1) * S + M = D
    direct_trans = np.zeros(D, dtype=np.float64)
    for i in range(D_prime):
        direct_trans[i * S : i * S + M] += y[i] * w

    return {
        "matrix_A": A,
        "matrix_AT": A.T,
        "Ax": Ax,
        "ATy": ATy,
        "direct_trans": direct_trans,
        "inner_product_Ax_y": inner_1,
        "inner_product_x_ATy": inner_2,
        "is_dual": bool(np.isclose(inner_1, inner_2)),
        "matches_direct_transposed_conv": bool(np.allclose(ATy, direct_trans)),
    }
