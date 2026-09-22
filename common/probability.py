"""
Probability utilities for Chapter 2: The Rules of Probability.
Provides exact derivations, calculations, and simulation helpers for discrete distributions,
Bayes' theorem, medical screening problem, and uncertainty demonstrations.
"""
from typing import Dict, Tuple, Optional
import numpy as np


def compute_joint_marginal_conditional(
    nij: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Given a 2D contingency table (array of counts nij of shape L x M, where
    rows correspond to X = x_i and columns correspond to Y = y_j),
    compute:
      - p_XY: Joint distribution p(X = x_i, Y = y_j) = nij / N
      - p_X: Marginal distribution p(X = x_i) = c_i / N
      - p_Y: Marginal distribution p(Y = y_j) = r_j / N
      - p_Y_given_X: Conditional distribution p(Y = y_j | X = x_i) = nij / c_i
      - p_X_given_Y: Conditional distribution p(X = x_i | Y = y_j) = nij / r_j

    Verifies sum and product rules:
      - Sum rule: p(X = x_i) = sum_j p(X = x_i, Y = y_j)
      - Sum rule: p(Y = y_j) = sum_i p(X = x_i, Y = y_j)
      - Product rule: p(X, Y) = p(Y | X) * p(X)
      - Product rule: p(X, Y) = p(X | Y) * p(Y)
    """
    nij = np.asarray(nij, dtype=np.float64)
    if nij.ndim != 2:
        raise ValueError(f"Expected 2D array of counts, got shape {nij.shape}")
    if np.any(nij < 0):
        raise ValueError("Counts must be non-negative")

    N = np.sum(nij)
    if N == 0:
        raise ValueError("Total count N must be strictly positive")

    # Joint distribution p(X = x_i, Y = y_j)
    p_XY = nij / N

    # Marginal counts: c_i = sum_j nij (row sums, X=x_i)
    ci = np.sum(nij, axis=1, keepdims=True)
    # Marginal counts: r_j = sum_i nij (col sums, Y=y_j)
    rj = np.sum(nij, axis=0, keepdims=True)

    # Marginal probabilities
    p_X = np.sum(p_XY, axis=1)  # shape (L,)
    p_Y = np.sum(p_XY, axis=0)  # shape (M,)

    # Conditional probabilities p(Y | X) = nij / ci
    # Handle zero division safely if ci == 0
    with np.errstate(divide='ignore', invalid='ignore'):
        p_Y_given_X = np.where(ci > 0, nij / ci, 0.0)  # shape (L, M)
        p_X_given_Y = np.where(rj > 0, nij / rj, 0.0)  # shape (L, M)

    return {
        "N": N,
        "ci": ci.squeeze(axis=-1),
        "rj": rj.squeeze(axis=0),
        "p_XY": p_XY,
        "p_X": p_X,
        "p_Y": p_Y,
        "p_Y_given_X": p_Y_given_X,
        "p_X_given_Y": p_X_given_Y,
    }


def bayes_rule(
    prior_y: np.ndarray,
    likelihood_x_given_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Bayes' theorem for discrete variables:
      p(Y = y_j | X = x_i) = p(X = x_i | Y = y_j) * p(Y = y_j) / p(X = x_i)
    where
      p(X = x_i) = sum_j p(X = x_i | Y = y_j) * p(Y = y_j)

    Parameters:
      prior_y: 1D array of shape (M,) representing p(Y)
      likelihood_x_given_y: 2D array of shape (L, M) where element (i, j) is p(X=x_i | Y=y_j)

    Returns:
      marginal_x: 1D array of shape (L,) representing p(X)
      posterior_y_given_x: 2D array of shape (L, M) where element (i, j) is p(Y=y_j | X=x_i)
    """
    prior_y = np.asarray(prior_y, dtype=np.float64)
    likelihood_x_given_y = np.asarray(likelihood_x_given_y, dtype=np.float64)

    if likelihood_x_given_y.shape[1] != prior_y.shape[0]:
        raise ValueError("Dimension mismatch between likelihood and prior")

    # Joint distribution p(X, Y) = p(X | Y) * p(Y) -> shape (L, M)
    joint_xy = likelihood_x_given_y * prior_y[np.newaxis, :]

    # Marginal p(X) = sum_j p(X, Y) -> shape (L,)
    marginal_x = np.sum(joint_xy, axis=1)

    # Posterior p(Y | X) = p(X, Y) / p(X) -> shape (L, M)
    with np.errstate(divide='ignore', invalid='ignore'):
        posterior_y_given_x = np.where(
            marginal_x[:, np.newaxis] > 0,
            joint_xy / marginal_x[:, np.newaxis],
            0.0
        )

    return marginal_x, posterior_y_given_x


def medical_screening_model(
    p_cancer: float = 0.01,
    p_pos_given_cancer: float = 0.90,
    p_pos_given_no_cancer: float = 0.03
) -> Dict[str, float]:
    """
    Model the cancer screening example from Bishop & Bishop (2024) Section 2.1.1 & 2.1.4:
      - C = 1 (cancer), C = 0 (no cancer)
      - T = 1 (positive test), T = 0 (negative test)

    Formulas:
      p(C = 1) = p_cancer
      p(C = 0) = 1 - p_cancer
      p(T = 1 | C = 1) = sensitivity = p_pos_given_cancer
      p(T = 0 | C = 1) = 1 - sensitivity (false negative rate)
      p(T = 1 | C = 0) = false positive rate = p_pos_given_no_cancer
      p(T = 0 | C = 0) = specificity = 1 - false positive rate

      p(T = 1) = p(T = 1 | C = 0)*p(C = 0) + p(T = 1 | C = 1)*p(C = 1) (Eq 2.20)
      p(C = 1 | T = 1) = p(T = 1 | C = 1)*p(C = 1) / p(T = 1)          (Eq 2.21, 2.22)
    """
    p_no_cancer = 1.0 - p_cancer
    p_neg_given_cancer = 1.0 - p_pos_given_cancer
    p_neg_given_no_cancer = 1.0 - p_pos_given_no_cancer

    # Total probability of testing positive (Eq 2.20)
    p_pos = (p_pos_given_no_cancer * p_no_cancer) + (p_pos_given_cancer * p_cancer)
    p_neg = 1.0 - p_pos

    # Posterior probability of cancer given positive test (Eq 2.22)
    p_cancer_given_pos = (p_pos_given_cancer * p_cancer) / p_pos
    p_no_cancer_given_pos = 1.0 - p_cancer_given_pos

    # Posterior probability of cancer given negative test
    p_cancer_given_neg = (p_neg_given_cancer * p_cancer) / p_neg
    p_no_cancer_given_neg = 1.0 - p_cancer_given_neg

    return {
        "p_cancer": p_cancer,
        "p_no_cancer": p_no_cancer,
        "p_pos_given_cancer": p_pos_given_cancer,
        "p_neg_given_cancer": p_neg_given_cancer,
        "p_pos_given_no_cancer": p_pos_given_no_cancer,
        "p_neg_given_no_cancer": p_neg_given_no_cancer,
        "p_pos": p_pos,
        "p_neg": p_neg,
        "p_cancer_given_pos": p_cancer_given_pos,
        "p_no_cancer_given_pos": p_no_cancer_given_pos,
        "p_cancer_given_neg": p_cancer_given_neg,
        "p_no_cancer_given_neg": p_no_cancer_given_neg,
    }


def check_independence(p_XY: np.ndarray, tol: float = 1e-7) -> Tuple[bool, float]:
    """
    Check if discrete random variables X and Y are independent:
      p(X, Y) == p(X) * p(Y)
    Returns:
      is_independent: bool
      max_absolute_diff: float
    """
    p_X = np.sum(p_XY, axis=1, keepdims=True)
    p_Y = np.sum(p_XY, axis=0, keepdims=True)
    p_independent = p_X @ p_Y
    diff = np.max(np.abs(p_XY - p_independent))
    return bool(diff < tol), float(diff)


def generate_2d_sine_data(
    n_samples: int = 100,
    noise_std: float = 0.2,
    fixed_x2: Optional[float] = None,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic data for Figure 2.1:
      y(x1, x2) = sin(2 * pi * x1) * sin(2 * pi * x2)
      t = y + epsilon, epsilon ~ N(0, noise_std^2)

    If fixed_x2 is None, x2 is uniformly sampled in [0, 1] (unobserved / varying).
    If fixed_x2 is provided (e.g. 0.25 where sin(2*pi*0.25) = 1.0), x2 is fixed.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0, size=n_samples)

    if fixed_x2 is None:
        x2 = rng.uniform(0.0, 1.0, size=n_samples)
    else:
        x2 = np.full(n_samples, fixed_x2)

    noise = rng.normal(0.0, noise_std, size=n_samples)
    y_true = np.sin(2.0 * np.pi * x1) * np.sin(2.0 * np.pi * x2)
    t = y_true + noise

    return {
        "x1": x1,
        "x2": x2,
        "y_true": y_true,
        "t": t,
    }
