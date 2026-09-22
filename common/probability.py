"""
Probability utilities for Chapter 2: The Rules of Probability.
Provides exact derivations, calculations, and simulation helpers for discrete distributions,
Bayes' theorem, medical screening problem, and uncertainty demonstrations.
"""
import os
from typing import Dict, Tuple, Optional, Callable, Union, List, Any, Sequence
import numpy as np
import scipy.linalg as la
from scipy import special
import matplotlib.pyplot as plt
import pandas as pd


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


# =============================================================================
# Section 2.2: Continuous Probability Densities, Expectations, and Covariances
# =============================================================================

class UniformDistribution:
    """
    Uniform distribution over the finite interval (a, b) (Eq 2.33):
      p(x) = 1 / (b - a)  for x in (a, b), and 0 elsewhere.
    """
    def __init__(self, a: float, b: float):
        if b <= a:
            raise ValueError(f"Upper bound b ({b}) must be strictly greater than lower bound a ({a})")
        self.a = float(a)
        self.b = float(b)

    @property
    def mean(self) -> float:
        return (self.a + self.b) / 2.0

    @property
    def variance(self) -> float:
        return ((self.b - self.a) ** 2) / 12.0

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        val = np.where((x_arr >= self.a) & (x_arr <= self.b), 1.0 / (self.b - self.a), 0.0)
        return val if x_arr.ndim > 0 else float(val)

    def cdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        val = np.clip((x_arr - self.a) / (self.b - self.a), 0.0, 1.0)
        return val if x_arr.ndim > 0 else float(val)

    def sample(self, size: int = 1, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(self.a, self.b, size=size)


class ExponentialDistribution:
    """
    Exponential distribution with rate parameter lambda > 0 (Eq 2.34):
      p(x | lambda) = lambda * exp(-lambda * x)  for x >= 0, and 0 for x < 0.
    """
    def __init__(self, lam: float):
        if lam <= 0:
            raise ValueError(f"Rate parameter lambda ({lam}) must be strictly positive")
        self.lam = float(lam)

    @property
    def mean(self) -> float:
        return 1.0 / self.lam

    @property
    def variance(self) -> float:
        return 1.0 / (self.lam ** 2)

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        val = np.where(x_arr >= 0.0, self.lam * np.exp(-self.lam * x_arr), 0.0)
        return val if x_arr.ndim > 0 else float(val)

    def cdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        val = np.where(x_arr >= 0.0, 1.0 - np.exp(-self.lam * x_arr), 0.0)
        return val if x_arr.ndim > 0 else float(val)

    def sample(self, size: int = 1, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.exponential(scale=1.0 / self.lam, size=size)


class LaplaceDistribution:
    """
    Laplace distribution with location mu and scale gamma > 0 (Eq 2.35):
      p(x | mu, gamma) = (1 / (2 * gamma)) * exp(-|x - mu| / gamma)
    """
    def __init__(self, mu: float = 0.0, gamma: float = 1.0):
        if gamma <= 0:
            raise ValueError(f"Scale parameter gamma ({gamma}) must be strictly positive")
        self.mu = float(mu)
        self.gamma = float(gamma)

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def variance(self) -> float:
        return 2.0 * (self.gamma ** 2)

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        val = (1.0 / (2.0 * self.gamma)) * np.exp(-np.abs(x_arr - self.mu) / self.gamma)
        return val if x_arr.ndim > 0 else float(val)

    def cdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        diff = (x_arr - self.mu) / self.gamma
        val = np.where(x_arr < self.mu, 0.5 * np.exp(diff), 1.0 - 0.5 * np.exp(-diff))
        return val if x_arr.ndim > 0 else float(val)

    def sample(self, size: int = 1, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.laplace(loc=self.mu, scale=self.gamma, size=size)


class GaussianMixture1D:
    """
    1D Gaussian Mixture Distribution:
      p(x) = sum_k w_k * N(x | mu_k, sigma_k^2)
    Useful for bimodal distributions such as Figure 2.6.
    """
    def __init__(
        self,
        weights: Union[List[float], np.ndarray],
        means: Union[List[float], np.ndarray],
        stds: Union[List[float], np.ndarray]
    ):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.means = np.asarray(means, dtype=np.float64)
        self.stds = np.asarray(stds, dtype=np.float64)

        if not (len(self.weights) == len(self.means) == len(self.stds)):
            raise ValueError("weights, means, and stds must have the same length")
        if np.any(self.stds <= 0):
            raise ValueError("All stds must be strictly positive")
        if np.any(self.weights < 0) or np.sum(self.weights) <= 0:
            raise ValueError("Weights must be non-negative and sum to > 0")

        self.weights = self.weights / np.sum(self.weights)

    @property
    def mean(self) -> float:
        return float(np.sum(self.weights * self.means))

    @property
    def variance(self) -> float:
        m = self.mean
        second_moment = np.sum(self.weights * (self.stds ** 2 + self.means ** 2))
        return float(second_moment - m ** 2)

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        # Compute component densities
        densities = np.zeros_like(x_arr)
        for w, m, s in zip(self.weights, self.means, self.stds):
            gauss = (1.0 / (np.sqrt(2.0 * np.pi) * s)) * np.exp(-0.5 * ((x_arr - m) / s) ** 2)
            densities += w * gauss
        return densities if x_arr.ndim > 0 else float(densities)

    def cdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        res = np.zeros_like(x_arr)
        for w, m, s in zip(self.weights, self.means, self.stds):
            res += w * 0.5 * (1.0 + special.erf((x_arr - m) / (s * np.sqrt(2.0))))
        return res if x_arr.ndim > 0 else float(res)

    def sample(self, size: int = 1, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        components = rng.choice(len(self.weights), size=size, p=self.weights)
        return rng.normal(loc=self.means[components], scale=self.stds[components], size=size)


class EmpiricalDistribution:
    """
    Empirical distribution constructed from a finite dataset D = {x_1, ..., x_N} (Eq 2.37):
      p(x | D) = (1 / N) * sum_n delta(x - x_n)
    """
    def __init__(self, data: np.ndarray):
        self.data = np.asarray(data, dtype=np.float64).flatten()
        if len(self.data) == 0:
            raise ValueError("Dataset D cannot be empty")
        self.N = len(self.data)

    @property
    def mean(self) -> float:
        return float(np.mean(self.data))

    @property
    def variance(self) -> float:
        return float(np.var(self.data, ddof=0))

    def expectation(self, f: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> float:
        """
        Evaluate E[f] = (1 / N) * sum_n f(x_n) (Eq 2.40).
        """
        if f is None:
            return float(np.mean(self.data))
        return float(np.mean(f(self.data)))

    def cdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        # Fraction of data points <= x
        if x_arr.ndim == 0:
            return float(np.mean(self.data <= x_arr))
        # Vectorized comparison: shape (len(data), len(x_arr))
        return np.mean(self.data[:, np.newaxis] <= x_arr[np.newaxis, :], axis=0)


def monte_carlo_expectation(
    samples: np.ndarray,
    f: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> float:
    """
    Approximate the expectation E[f] via sample mean (Eq 2.40):
      E[f] ~= (1 / N) * sum_n f(x_n)
    """
    samples = np.asarray(samples, dtype=np.float64)
    if len(samples) == 0:
        raise ValueError("Samples array cannot be empty")
    if f is None:
        return float(np.mean(samples))
    return float(np.mean(f(samples)))


def compute_variance(
    samples: np.ndarray,
    f: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Dict[str, float]:
    """
    Compute the sample variance of f(x) and verify the algebraic identity (Eq 2.44, 2.45):
      var[f] = E[(f(x) - E[f(x)])^2]
             = E[f(x)^2] - (E[f(x)])^2
    """
    samples = np.asarray(samples, dtype=np.float64)
    vals = samples if f is None else f(samples)

    e_f = float(np.mean(vals))
    e_f2 = float(np.mean(vals ** 2))
    var_definition = float(np.mean((vals - e_f) ** 2))
    var_algebraic = e_f2 - (e_f ** 2)

    return {
        "mean": e_f,
        "second_moment": e_f2,
        "var_definition": var_definition,
        "var_algebraic": var_algebraic,
        "var": var_definition,
    }


def compute_covariance(
    x: np.ndarray,
    y: np.ndarray
) -> Dict[str, float]:
    """
    Compute sample covariance between scalar random variables x and y (Eq 2.47):
      cov[x, y] = E[(x - E[x])(y - E[y])]
                = E[xy] - E[x]E[y]
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of samples")
    if len(x) == 0:
        raise ValueError("Samples cannot be empty")

    e_x = float(np.mean(x))
    e_y = float(np.mean(y))
    e_xy = float(np.mean(x * y))

    cov_definition = float(np.mean((x - e_x) * (y - e_y)))
    cov_algebraic = e_xy - (e_x * e_y)

    return {
        "E_x": e_x,
        "E_y": e_y,
        "E_xy": e_xy,
        "E_x_E_y": e_x * e_y,
        "cov_definition": cov_definition,
        "cov_algebraic": cov_algebraic,
        "cov": cov_definition,
    }


def compute_covariance_matrix(
    X: np.ndarray
) -> np.ndarray:
    """
    Compute sample covariance matrix for vector variable x of shape (N, D) (Eq 2.48):
      cov[x] = E[(x - E[x])(x - E[x])^T]
    Returns D x D symmetric positive semi-definite matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (N, D), got {X.shape}")
    N, D = X.shape
    if N == 0:
        raise ValueError("Number of samples N must be > 0")

    mean_vec = np.mean(X, axis=0, keepdims=True)
    diff = X - mean_vec
    cov_matrix = (diff.T @ diff) / N
    return cov_matrix


# =============================================================================
# Section 2.3: The Gaussian Distribution, Maximum Likelihood, and Linear Regression
# =============================================================================

class Gaussian1D:
    """
    Univariate Gaussian (Normal) Distribution (Eq 2.49):
      N(x | mu, sigma^2) = (1 / (2 * pi * sigma^2)^{1/2}) * exp(- 1 / (2 * sigma^2) * (x - mu)^2)

    Governed by parameters:
      mu: mean (and mode)
      sigma^2: variance (sigma > 0 is standard deviation)
      beta: precision = 1 / sigma^2
    """
    def __init__(self, mu: float = 0.0, sigma2: float = 1.0):
        if sigma2 <= 0:
            raise ValueError(f"Variance sigma2 ({sigma2}) must be strictly positive")
        self.mu = float(mu)
        self.sigma2 = float(sigma2)
        self.sigma = float(np.sqrt(sigma2))
        self.beta = 1.0 / self.sigma2

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def variance(self) -> float:
        return self.sigma2

    @property
    def precision(self) -> float:
        return self.beta

    @property
    def mode(self) -> float:
        return self.mu

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate Gaussian probability density function (Eq 2.49)."""
        x_arr = np.asarray(x, dtype=np.float64)
        norm_const = 1.0 / (np.sqrt(2.0 * np.pi * self.sigma2))
        exponent = -0.5 * ((x_arr - self.mu) ** 2) / self.sigma2
        val = norm_const * np.exp(exponent)
        return val if x_arr.ndim > 0 else float(val)

    def log_pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate log Gaussian probability density function."""
        x_arr = np.asarray(x, dtype=np.float64)
        val = -0.5 * np.log(2.0 * np.pi * self.sigma2) - 0.5 * ((x_arr - self.mu) ** 2) / self.sigma2
        return val if x_arr.ndim > 0 else float(val)

    def cdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate cumulative distribution function."""
        x_arr = np.asarray(x, dtype=np.float64)
        val = 0.5 * (1.0 + special.erf((x_arr - self.mu) / (self.sigma * np.sqrt(2.0))))
        return val if x_arr.ndim > 0 else float(val)

    def sample(self, size: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """Draw independent samples from the Gaussian distribution."""
        rng = np.random.default_rng(seed)
        return rng.normal(loc=self.mu, scale=self.sigma, size=size)


def gaussian_maximum_likelihood(data: np.ndarray) -> Dict[str, float]:
    """
    Maximum likelihood parameter estimation for 1D Gaussian (Eq 2.56 - 2.58, 2.63):
      mu_ML = (1 / N) * sum_{n=1}^N x_n                 (Eq 2.57)
      sigma2_ML = (1 / N) * sum_{n=1}^N (x_n - mu_ML)^2 (Eq 2.58)
      sigma2_unbiased = (1 / (N - 1)) * sum_{n=1}^N (x_n - mu_ML)^2 = (N / (N - 1)) * sigma2_ML (Eq 2.63)

    Returns:
      Dict containing mu_ML, sigma2_ML, sigma_ML, sigma2_unbiased, sigma_unbiased, log_likelihood, N
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    N = len(data)
    if N == 0:
        raise ValueError("Data array cannot be empty")

    mu_ml = float(np.mean(data))
    # Note: np.var with ddof=0 gives the ML sample variance (Eq 2.58)
    sigma2_ml = float(np.mean((data - mu_ml) ** 2))
    sigma_ml = float(np.sqrt(sigma2_ml))

    if N > 1:
        sigma2_unbiased = float(np.var(data, ddof=1))
        sigma_unbiased = float(np.sqrt(sigma2_unbiased))
    else:
        sigma2_unbiased = float("nan")
        sigma_unbiased = float("nan")

    # Evaluate log likelihood under ML estimates (Eq 2.56)
    if sigma2_ml > 0:
        log_lik = -0.5 * N - 0.5 * N * np.log(sigma2_ml) - 0.5 * N * np.log(2.0 * np.pi)
    else:
        log_lik = float("inf")

    return {
        "N": N,
        "mu_ML": mu_ml,
        "sigma2_ML": sigma2_ml,
        "sigma_ML": sigma_ml,
        "sigma2_unbiased": sigma2_unbiased,
        "sigma_unbiased": sigma_unbiased,
        "log_likelihood": log_lik,
    }


def simulate_gaussian_mle_bias(
    mu: float = 0.0,
    sigma2: float = 1.0,
    N: int = 2,
    n_trials: int = 10000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Simulate the bias of maximum likelihood estimators for a Gaussian (Section 2.3.3):
      E[mu_ML] = mu                   (Eq 2.59, unbiased)
      E[sigma2_ML] = ((N - 1) / N) * sigma2 (Eq 2.60, biased by factor (N-1)/N)
      E[sigma2_hat] = sigma2          (Eq 2.62, variance measured relative to true mean, unbiased)
      E[sigma2_tilde] = sigma2        (Eq 2.63, Bessel-corrected sample variance, unbiased)

    Returns dictionary with empirical and theoretical expectations.
    """
    if N < 2:
        raise ValueError("Sample size N must be at least 2 to evaluate variance estimators")

    rng = np.random.default_rng(seed)
    sigma = np.sqrt(sigma2)
    # Shape: (n_trials, N)
    samples = rng.normal(loc=mu, scale=sigma, size=(n_trials, N))

    # Estimates for each trial
    mu_ml_trials = np.mean(samples, axis=1)  # shape (n_trials,)
    sigma2_ml_trials = np.mean((samples - mu_ml_trials[:, np.newaxis]) ** 2, axis=1)
    sigma2_hat_trials = np.mean((samples - mu) ** 2, axis=1)  # relative to true mean mu
    sigma2_tilde_trials = np.var(samples, axis=1, ddof=1)  # Bessel's correction

    # Empirical expectations across trials
    e_mu_ml = float(np.mean(mu_ml_trials))
    e_sigma2_ml = float(np.mean(sigma2_ml_trials))
    e_sigma2_hat = float(np.mean(sigma2_hat_trials))
    e_sigma2_tilde = float(np.mean(sigma2_tilde_trials))

    theoretical_e_sigma2_ml = ((N - 1.0) / N) * sigma2

    return {
        "true_mu": float(mu),
        "true_sigma2": float(sigma2),
        "N": N,
        "n_trials": n_trials,
        "E_mu_ML": e_mu_ml,
        "E_sigma2_ML": e_sigma2_ml,
        "theoretical_E_sigma2_ML": theoretical_e_sigma2_ml,
        "E_sigma2_hat": e_sigma2_hat,
        "E_sigma2_tilde": e_sigma2_tilde,
        "bias_sigma2_ML": e_sigma2_ml - sigma2,
        "theoretical_bias_sigma2_ML": -(1.0 / N) * sigma2,
    }


class GaussianLinearRegression:
    """
    Probabilistic Linear / Polynomial Regression with Gaussian noise (Section 2.3.4):
      p(t | x, w, sigma^2) = N(t | y(x; w), sigma^2)     (Eq 2.64)
      where y(x; w) = sum_{j=0}^M w_j * x^j = phi(x)^T w

    Training via maximum likelihood determines:
      w_ML = argmin E(w) = (Phi^T Phi)^{-1} Phi^T t       (Eq 2.67)
      sigma2_ML = (1 / N) * sum_{n=1}^N (y(x_n; w_ML) - t_n)^2 (Eq 2.68)

    Predictive distribution:
      p(t | x, w_ML, sigma2_ML) = N(t | y(x; w_ML), sigma2_ML) (Eq 2.69)
    """
    def __init__(self, degree: int = 3):
        if degree < 0:
            raise ValueError("Polynomial degree must be non-negative")
        self.degree = degree
        self.w: Optional[np.ndarray] = None
        self.sigma2_ml: Optional[float] = None
        self.sigma_ml: Optional[float] = None

    def _design_matrix(self, x: np.ndarray) -> np.ndarray:
        """Construct Vandermonde design matrix Phi for polynomial basis."""
        x_flat = np.asarray(x, dtype=np.float64).flatten()
        powers = np.arange(self.degree + 1)
        # Shape: (N, degree + 1)
        return x_flat[:, np.newaxis] ** powers[np.newaxis, :]

    def fit(self, x: np.ndarray, t: np.ndarray) -> "GaussianLinearRegression":
        """
        Fit polynomial regression model parameters w_ML and sigma2_ML by maximum likelihood.
        """
        x_flat = np.asarray(x, dtype=np.float64).flatten()
        t_flat = np.asarray(t, dtype=np.float64).flatten()
        if len(x_flat) != len(t_flat):
            raise ValueError("x and t must have the same length")
        N = len(x_flat)
        if N <= self.degree:
            raise ValueError(f"Number of samples N ({N}) must be greater than degree ({self.degree})")

        Phi = self._design_matrix(x_flat)
        # Solve normal equations: (Phi^T Phi) w = Phi^T t using lstsq
        self.w, _, _, _ = np.linalg.lstsq(Phi, t_flat, rcond=None)

        # Residuals and ML variance (Eq 2.68)
        y_pred = Phi @ self.w
        residuals = t_flat - y_pred
        self.sigma2_ml = float(np.mean(residuals ** 2))
        self.sigma_ml = float(np.sqrt(self.sigma2_ml))
        return self

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Evaluate predictive distribution for input x (Eq 2.69):
          Returns (y_mean, sigma_ml), where predictive distribution is N(t | y_mean, sigma2_ml).
        """
        if self.w is None or self.sigma_ml is None:
            raise RuntimeError("Model must be fitted before making predictions")
        Phi = self._design_matrix(x)
        y_mean = Phi @ self.w
        return y_mean, self.sigma_ml

    def sum_of_squares_error(self, x: np.ndarray, t: np.ndarray) -> float:
        """Evaluate sum-of-squares error function E(w) (Eq 2.67): E(w) = 0.5 * sum (y - t)^2."""
        if self.w is None:
            raise RuntimeError("Model must be fitted")
        x_flat = np.asarray(x, dtype=np.float64).flatten()
        t_flat = np.asarray(t, dtype=np.float64).flatten()
        Phi = self._design_matrix(x_flat)
        residuals = Phi @ self.w - t_flat
        return float(0.5 * np.sum(residuals ** 2))

    def log_likelihood(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        Evaluate log likelihood of targets t given inputs x under fitted parameters (Eq 2.66):
          ln p(t | x, w_ML, sigma2_ML) = - 1/(2*sigma2) * sum (y - t)^2 - N/2 ln(sigma2) - N/2 ln(2*pi)
        """
        if self.w is None or self.sigma2_ml is None:
            raise RuntimeError("Model must be fitted")
        x_flat = np.asarray(x, dtype=np.float64).flatten()
        t_flat = np.asarray(t, dtype=np.float64).flatten()
        N = len(x_flat)
        Phi = self._design_matrix(x_flat)
        residuals = Phi @ self.w - t_flat
        sse = np.sum(residuals ** 2)
        ll = -0.5 * sse / self.sigma2_ml - 0.5 * N * np.log(self.sigma2_ml) - 0.5 * N * np.log(2.0 * np.pi)
        return float(ll)


# ==============================================================================
# Chapter 2 Section 2.4: Transformation of Densities
# ==============================================================================

def transform_density_1d(
    px_func: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
    g_func: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
    g_prime_func: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
    y: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Compute transformed 1D probability density py(y) according to Bishop (2024) Eq (2.71):
      py(y) = px(g(y)) * |g'(y)|
    where x = g(y).
    """
    y_arr = np.asarray(y, dtype=np.float64)
    x_val = g_func(y_arr)
    px_val = px_func(x_val)
    abs_det = np.abs(g_prime_func(y_arr))
    res = px_val * abs_det
    if np.isscalar(y):
        return float(res)
    return res


class DensityTransformation1DExample:
    """
    Encapsulates the 1D density transformation example from Section 2.4 and Figure 2.12.

    Variables and Mapping:
      - x follows a Gaussian distribution: px(x) = N(x | mu, sigma^2) (default mu=6.0, sigma=1.0)
      - Change of variables: x = g(y) = ln(y) - ln(1 - y) + 5 (Eq 2.74, logit shifted by 5)
      - Inverse mapping: y = g^{-1}(x) = 1 / (1 + exp(-x + 5)) = sigmoid(x - 5) (Eq 2.75)
      - Derivative: g'(y) = 1 / (y * (1 - y))
      - Second derivative: g''(y) = (2*y - 1) / (y^2 * (1 - y)^2)
      - Naive function transformation: px(g(y))
      - True density transformation: py(y) = px(g(y)) * |g'(y)| (Eq 2.71)
      - Derivative of py(y) (Eq 2.73):
          py'(y) = s * px'(g(y)) * (g'(y))^2 + s * px(g(y)) * g''(y)
    """
    def __init__(self, mu: float = 6.0, sigma: float = 1.0, offset: float = 5.0):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.sigma2 = self.sigma ** 2
        self.offset = float(offset)

    def px(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Gaussian density px(x) = N(x | mu, sigma^2)."""
        x_arr = np.asarray(x, dtype=np.float64)
        norm = 1.0 / (np.sqrt(2.0 * np.pi) * self.sigma)
        val = norm * np.exp(-0.5 * ((x_arr - self.mu) / self.sigma) ** 2)
        if np.isscalar(x):
            return float(val)
        return val

    def px_prime(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Derivative of Gaussian density d px(x) / dx = - ((x - mu) / sigma^2) * px(x)."""
        x_arr = np.asarray(x, dtype=np.float64)
        val = - ((x_arr - self.mu) / self.sigma2) * self.px(x_arr)
        if np.isscalar(x):
            return float(val)
        return val

    def g(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Forward mapping x = g(y) = ln(y) - ln(1 - y) + offset (Eq 2.74)."""
        y_arr = np.asarray(y, dtype=np.float64)
        val = np.log(y_arr) - np.log(1.0 - y_arr) + self.offset
        if np.isscalar(y):
            return float(val)
        return val

    def g_inv(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Inverse mapping y = g^{-1}(x) = 1 / (1 + exp(-(x - offset))) (Eq 2.75)."""
        x_arr = np.asarray(x, dtype=np.float64)
        val = 1.0 / (1.0 + np.exp(-(x_arr - self.offset)))
        if np.isscalar(x):
            return float(val)
        return val

    def g_prime(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Derivative dg/dy = 1 / (y * (1 - y))."""
        y_arr = np.asarray(y, dtype=np.float64)
        val = 1.0 / (y_arr * (1.0 - y_arr))
        if np.isscalar(y):
            return float(val)
        return val

    def g_double_prime(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Second derivative d^2g/dy^2 = (2*y - 1) / (y * (1 - y))^2."""
        y_arr = np.asarray(y, dtype=np.float64)
        val = (2.0 * y_arr - 1.0) / ((y_arr * (1.0 - y_arr)) ** 2)
        if np.isscalar(y):
            return float(val)
        return val

    def px_transformed_as_function(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Naive function transformation px(g(y)). Peak occurs at y = g^{-1}(mu)."""
        y_arr = np.asarray(y, dtype=np.float64)
        return self.px(self.g(y_arr))

    def py(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """True transformed probability density py(y) = px(g(y)) * |g'(y)| (Eq 2.71)."""
        y_arr = np.asarray(y, dtype=np.float64)
        return self.px(self.g(y_arr)) * np.abs(self.g_prime(y_arr))

    def py_prime(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Derivative of transformed density d py(y) / dy (Eq 2.73):
          py'(y) = s * px'(g(y)) * (g'(y))^2 + s * px(g(y)) * g''(y)
        Here s = +1 because g'(y) > 0 for y in (0, 1).
        """
        y_arr = np.asarray(y, dtype=np.float64)
        x_val = self.g(y_arr)
        px_val = self.px(x_val)
        px_prime_val = self.px_prime(x_val)
        gp = self.g_prime(y_arr)
        gpp = self.g_double_prime(y_arr)
        val = px_prime_val * (gp ** 2) + px_val * gpp
        if np.isscalar(y):
            return float(val)
        return val

    def mode_x(self) -> float:
        """Mode of Gaussian density px(x), which is mu."""
        return self.mu

    def mode_function_transform(self) -> float:
        """Mode of naive function transform px(g(y)), which is g^{-1}(mu)."""
        return float(self.g_inv(self.mu))

    def mode_py(self) -> float:
        """
        Mode of true density py(y).
        Satisfies Eq (2.73) with py'(y) = 0:
          (g(y) - mu) / sigma^2 = (2y - 1)
        """
        from scipy.optimize import root_scalar
        def f_root(y_val):
            return self.g(y_val) - self.mu - self.sigma2 * (2.0 * y_val - 1.0)
        sol = root_scalar(f_root, bracket=[0.5, 0.999], method='brentq')
        return float(sol.root)

    def sample(self, N: int = 50000, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample N points from px(x) and transform via y = g^{-1}(x).
        Returns (x_samples, y_samples).
        """
        rng = np.random.default_rng(seed)
        x_samples = rng.normal(loc=self.mu, scale=self.sigma, size=N)
        y_samples = self.g_inv(x_samples)
        return x_samples, y_samples


class LinearTransformation1D:
    """
    Demonstrates that mode transformation equivariance hat{x} = g(hat{y}) holds
    for LINEAR transformations x = g(y) = a * y + b, because g''(y) = 0 vanishes (Eq 2.73).
    """
    def __init__(self, a: float = 2.5, b: float = 1.0, mu_x: float = 4.0, sigma_x: float = 1.2):
        if a == 0:
            raise ValueError("Linear transformation slope 'a' cannot be zero.")
        self.a = float(a)
        self.b = float(b)
        self.mu_x = float(mu_x)
        self.sigma_x = float(sigma_x)

    def g(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Forward mapping x = g(y) = a * y + b."""
        return self.a * np.asarray(y, dtype=np.float64) + self.b

    def g_inv(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Inverse mapping y = g^{-1}(x) = (x - b) / a."""
        return (np.asarray(x, dtype=np.float64) - self.b) / self.a

    def g_prime(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """First derivative g'(y) = a."""
        return np.full_like(y, self.a, dtype=np.float64) if hasattr(y, "__len__") else float(self.a)

    def g_double_prime(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Second derivative g''(y) = 0."""
        return np.zeros_like(y, dtype=np.float64) if hasattr(y, "__len__") else 0.0

    def px(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Gaussian density px(x) = N(x | mu_x, sigma_x^2)."""
        x_arr = np.asarray(x, dtype=np.float64)
        norm = 1.0 / (np.sqrt(2.0 * np.pi) * self.sigma_x)
        return norm * np.exp(-0.5 * ((x_arr - self.mu_x) / self.sigma_x) ** 2)

    def py(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Transformed density py(y) = px(g(y)) * |a|."""
        return self.px(self.g(y)) * abs(self.a)

    def mode_x(self) -> float:
        """Mode of px(x), which is mu_x."""
        return self.mu_x

    def mode_y(self) -> float:
        """
        Mode of py(y). For linear transformation, hat{y} = g^{-1}(hat{x}) = (mu_x - b) / a.
        """
        return float(self.g_inv(self.mu_x))


class BivariateTransformation2DExample:
    """
    Encapsulates the 2D density transformation example from Section 2.4.1 and Figure 2.13.

    Transformation Equations (Eq 2.78, 2.79):
      y1 = x1 + tanh(5 * x1)
      y2 = x2 + tanh(5 * x2) + (x1^3) / 3

    Jacobian Matrix (Exercise 2.20):
      J = [ [ 1 + 5*sech^2(5*x1),           0           ],
            [          x1^2,        1 + 5*sech^2(5*x2) ] ]

    Jacobian Determinant:
      det J = (1 + 5*sech^2(5*x1)) * (1 + 5*sech^2(5*x2))

    Base Distribution px(x):
      Standard 2D Gaussian: N(x | 0, sigma^2 * I) (default sigma=0.4)
    """
    def __init__(self, sigma: float = 0.4):
        self.sigma = float(sigma)
        self.sigma2 = self.sigma ** 2
        # Initialize fine 1D spline for ultra-fast vector inversion
        from scipy.interpolate import CubicSpline
        u_fine = np.linspace(-6.0, 6.0, 4000)
        v_fine = u_fine + np.tanh(5.0 * u_fine)
        self._inv_spline = CubicSpline(v_fine, u_fine)

    def forward(self, x1: np.ndarray, x2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate forward transformation y = f(x) (Eq 2.78, 2.79)."""
        x1_arr = np.asarray(x1, dtype=np.float64)
        x2_arr = np.asarray(x2, dtype=np.float64)
        y1 = x1_arr + np.tanh(5.0 * x1_arr)
        y2 = x2_arr + np.tanh(5.0 * x2_arr) + (x1_arr ** 3) / 3.0
        return y1, y2

    def jacobian_matrix(self, x1: float, x2: float) -> np.ndarray:
        """
        Evaluate 2x2 Jacobian matrix J_yx = d(y1, y2) / d(x1, x2) at (x1, x2) (Exercise 2.20).
        """
        x1_f = float(x1)
        x2_f = float(x2)
        sech1_sq = 1.0 / (np.cosh(5.0 * x1_f) ** 2)
        sech2_sq = 1.0 / (np.cosh(5.0 * x2_f) ** 2)
        j11 = 1.0 + 5.0 * sech1_sq
        j12 = 0.0
        j21 = x1_f ** 2
        j22 = 1.0 + 5.0 * sech2_sq
        return np.array([[j11, j12], [j21, j22]], dtype=np.float64)

    def jacobian_det(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Evaluate Jacobian determinant det J_yx = (1 + 5*sech^2(5*x1)) * (1 + 5*sech^2(5*x2)).
        """
        x1_arr = np.asarray(x1, dtype=np.float64)
        x2_arr = np.asarray(x2, dtype=np.float64)
        sech1_sq = 1.0 / (np.cosh(5.0 * x1_arr) ** 2)
        sech2_sq = 1.0 / (np.cosh(5.0 * x2_arr) ** 2)
        det_j = (1.0 + 5.0 * sech1_sq) * (1.0 + 5.0 * sech2_sq)
        return det_j

    def log_jacobian_det(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Evaluate log determinant ln |det J_yx|."""
        x1_arr = np.asarray(x1, dtype=np.float64)
        x2_arr = np.asarray(x2, dtype=np.float64)
        sech1_sq = 1.0 / (np.cosh(5.0 * x1_arr) ** 2)
        sech2_sq = 1.0 / (np.cosh(5.0 * x2_arr) ** 2)
        j11 = 1.0 + 5.0 * sech1_sq
        j22 = 1.0 + 5.0 * sech2_sq
        return np.log(j11) + np.log(j22)

    def inverse_single(self, y1: float, y2: float) -> Tuple[float, float]:
        """
        Compute inverse mapping (x1, x2) = f^{-1}(y1, y2) using exact 1D root finding.
        Because J is lower-triangular:
          y1 = x1 + tanh(5*x1) is strictly monotonic in x1 -> solve for x1.
          y2 - x1^3/3 = x2 + tanh(5*x2) is strictly monotonic in x2 -> solve for x2.
        """
        from scipy.optimize import root_scalar
        y1_f = float(y1)
        y2_f = float(y2)

        def eq1(x):
            return x + np.tanh(5.0 * x) - y1_f
        x1_sol = root_scalar(eq1, bracket=[y1_f - 1.05, y1_f + 1.05], method='brentq').root

        target2 = y2_f - (x1_sol ** 3) / 3.0
        def eq2(x):
            return x + np.tanh(5.0 * x) - target2
        x2_sol = root_scalar(eq2, bracket=[target2 - 1.05, target2 + 1.05], method='brentq').root

        return float(x1_sol), float(x2_sol)

    def inverse(self, y1: np.ndarray, y2: np.ndarray, use_spline: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized evaluation of inverse mapping (x1, x2) = f^{-1}(y1, y2).
        If use_spline is True, evaluates instantaneously using high-precision cubic spline.
        """
        y1_arr = np.asarray(y1, dtype=np.float64)
        y2_arr = np.asarray(y2, dtype=np.float64)
        if use_spline:
            x1 = self._inv_spline(y1_arr)
            target2 = y2_arr - (x1 ** 3) / 3.0
            x2 = self._inv_spline(target2)
            return x1, x2

        shape = y1_arr.shape
        y1_flat = y1_arr.ravel()
        y2_flat = y2_arr.ravel()
        x1_out = np.empty_like(y1_flat)
        x2_out = np.empty_like(y2_flat)
        for i in range(len(y1_flat)):
            x1_out[i], x2_out[i] = self.inverse_single(y1_flat[i], y2_flat[i])
        return x1_out.reshape(shape), x2_out.reshape(shape)

    def px(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Standard 2D isotropic Gaussian density N(x | 0, sigma^2 * I)."""
        x1_arr = np.asarray(x1, dtype=np.float64)
        x2_arr = np.asarray(x2, dtype=np.float64)
        norm = 1.0 / (2.0 * np.pi * self.sigma2)
        return norm * np.exp(-0.5 * (x1_arr ** 2 + x2_arr ** 2) / self.sigma2)

    def py(self, y1: np.ndarray, y2: np.ndarray, use_spline: bool = True) -> np.ndarray:
        """
        Transformed 2D density py(y) = px(f^{-1}(y)) / |det J_yx(f^{-1}(y))| (Eq 2.76).
        """
        x1, x2 = self.inverse(y1, y2, use_spline=use_spline)
        px_val = self.px(x1, x2)
        det_j = self.jacobian_det(x1, x2)
        return px_val / np.abs(det_j)

    def sample(self, N: int = 800, seed: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample N points from px(x) and transform to y-space via forward mapping.
        Returns (x1_samples, x2_samples, y1_samples, y2_samples).
        """
        rng = np.random.default_rng(seed)
        x1_samples = rng.normal(loc=0.0, scale=self.sigma, size=N)
        x2_samples = rng.normal(loc=0.0, scale=self.sigma, size=N)
        y1_samples, y2_samples = self.forward(x1_samples, x2_samples)
        return x1_samples, x2_samples, y1_samples, y2_samples


# ==============================================================================
# Section 2.5: Information Theory Utilities
# ==============================================================================

def discrete_entropy(p: np.ndarray, base: str = 'e') -> float:
    """
    Compute discrete Shannon entropy H[p] = - sum_i p_i log(p_i) (Eq 2.81, 2.86).
    Handles p_i = 0 with lim_{p -> 0} p log p = 0.
    
    Parameters
    ----------
    p : np.ndarray
        1D probability mass array (must sum to ~1 and be non-negative).
    base : str, default='e'
        Logarithm base: 'e' for nats, '2' for bits.
    """
    p = np.asarray(p, dtype=float)
    if not np.all(p >= -1e-9):
        raise ValueError("Probabilities must be non-negative.")
    p = np.clip(p, 0.0, 1.0)
    total = np.sum(p)
    if total <= 0:
        return 0.0
    p = p / total
    
    pos_mask = p > 0.0
    if not np.any(pos_mask):
        return 0.0
    
    if base == '2':
        return float(-np.sum(p[pos_mask] * np.log2(p[pos_mask])))
    elif base == 'e':
        return float(-np.sum(p[pos_mask] * np.log(p[pos_mask])))
    else:
        raise ValueError(f"Unsupported log base '{base}'. Choose 'e' or '2'.")


def differential_entropy_gaussian(sigma2: float) -> float:
    """
    Differential entropy of a 1D Gaussian distribution N(mu, sigma2) (Eq 2.99):
    H[x] = 0.5 * (1 + ln(2 * pi * sigma2))
    """
    if sigma2 <= 0:
        raise ValueError("Variance sigma2 must be strictly positive.")
    return 0.5 * (1.0 + np.log(2.0 * np.pi * sigma2))


def differential_entropy_1d(
    pdf_func: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n_points: int = 4000
) -> float:
    """
    Numerical approximation of 1D differential entropy H[x] = - int p(x) ln p(x) dx (Eq 2.91).
    """
    x = np.linspace(a, b, n_points)
    dx = (b - a) / (n_points - 1)
    p = pdf_func(x)
    p = np.clip(p, 0.0, None)
    pos_mask = p > 1e-15
    integrand = np.zeros_like(p)
    integrand[pos_mask] = -p[pos_mask] * np.log(p[pos_mask])
    return float(np.trapezoid(integrand, x)) if hasattr(np, 'trapezoid') else float(np.trapz(integrand, x))


def kl_divergence_discrete(p: np.ndarray, q: np.ndarray, base: str = 'e') -> float:
    """
    Kullback-Leibler divergence KL(p || q) for discrete distributions (Eq 2.100):
    KL(p || q) = sum_i p_i log(p_i / q_i).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    q = np.clip(q, 0.0, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    pos_mask = p > 0.0
    if np.any((q == 0.0) & pos_mask):
        return float('inf')
    
    ratio = p[pos_mask] / q[pos_mask]
    if base == '2':
        return float(np.sum(p[pos_mask] * np.log2(ratio)))
    elif base == 'e':
        return float(np.sum(p[pos_mask] * np.log(ratio)))
    else:
        raise ValueError(f"Unsupported log base '{base}'. Choose 'e' or '2'.")


def kl_divergence_gaussian_1d(mu1: float, sigma1_sq: float, mu2: float, sigma2_sq: float) -> float:
    """
    Exact analytical KL divergence KL(N(mu1, sigma1_sq) || N(mu2, sigma2_sq)):
    KL(p || q) = 0.5 * (ln(sigma2_sq / sigma1_sq) + (sigma1_sq + (mu1 - mu2)^2) / sigma2_sq - 1)
    """
    if sigma1_sq <= 0 or sigma2_sq <= 0:
        raise ValueError("Variances must be strictly positive.")
    return 0.5 * (
        np.log(sigma2_sq / sigma1_sq)
        + (sigma1_sq + (mu1 - mu2) ** 2) / sigma2_sq
        - 1.0
    )


def conditional_entropy_discrete(p_xy: np.ndarray) -> float:
    """
    Conditional entropy H[Y|X] = - sum_{x,y} p(x,y) ln p(y|x) (Eq 2.107).
    Satisfies H[X,Y] = H[Y|X] + H[X] (Eq 2.108).
    """
    p_xy = np.asarray(p_xy, dtype=float)
    p_xy = p_xy / np.sum(p_xy)
    p_x = np.sum(p_xy, axis=1)  # Marginal p(x)
    
    h_xy = discrete_entropy(p_xy.ravel(), base='e')
    h_x = discrete_entropy(p_x, base='e')
    return float(h_xy - h_x)


def mutual_information_discrete(p_xy: np.ndarray) -> float:
    """
    Mutual information I[X, Y] = KL(p(x,y) || p(x)p(y)) (Eq 2.109).
    I[X, Y] = H[X] - H[X|Y] = H[Y] - H[Y|X] = H[X] + H[Y] - H[X, Y] (Eq 2.110).
    """
    p_xy = np.asarray(p_xy, dtype=float)
    p_xy = p_xy / np.sum(p_xy)
    p_x = np.sum(p_xy, axis=1)  # marginal p(x)
    p_y = np.sum(p_xy, axis=0)  # marginal p(y)
    
    p_prod = np.outer(p_x, p_y)
    return kl_divergence_discrete(p_xy.ravel(), p_prod.ravel(), base='e')


def mutual_information_gaussian(cov_matrix: np.ndarray) -> float:
    """
    Mutual information for a 2D bivariate Gaussian N(mu, Sigma):
    I[X, Y] = -0.5 * ln(1 - rho^2) where rho = Sigma_xy / sqrt(Sigma_xx * Sigma_yy).
    """
    cov = np.asarray(cov_matrix, dtype=float)
    sigma_x = np.sqrt(cov[0, 0])
    sigma_y = np.sqrt(cov[1, 1])
    cov_xy = cov[0, 1]
    rho = cov_xy / (sigma_x * sigma_y)
    rho = np.clip(rho, -0.99999999, 0.99999999)
    return float(-0.5 * np.log(1.0 - rho ** 2))


# ==============================================================================
# Section 2.6: Bayesian Probabilities & Non-Transitive Dice Utilities
# ==============================================================================

class BayesianLinearRegression:
    """
    Bayesian Linear (Polynomial) Regression (Eq 2.111 - 2.118).
    Prior: p(w) = N(w | 0, alpha^{-1} I)
    Likelihood: p(t | x, w) = N(t | w^T phi(x), beta^{-1})
    Posterior: p(w | D) = N(w | m_N, S_N)
    Predictive: p(t | x, D) = N(t | m_N^T phi(x), beta^{-1} + phi(x)^T S_N phi(x))
    """
    def __init__(self, degree: int = 3, alpha: float = 0.005, beta: float = 11.1):
        self.degree = degree
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.m_N: Optional[np.ndarray] = None
        self.S_N: Optional[np.ndarray] = None
        self.S_N_inv: Optional[np.ndarray] = None

    def _design_matrix(self, x: np.ndarray) -> np.ndarray:
        x_flat = np.asarray(x, dtype=float).ravel()
        return np.vstack([x_flat ** i for i in range(self.degree + 1)]).T

    def fit(self, x: np.ndarray, t: np.ndarray) -> "BayesianLinearRegression":
        Phi = self._design_matrix(x)
        t_arr = np.asarray(t, dtype=float).ravel()
        M_dim = self.degree + 1
        
        # S_N^{-1} = alpha * I + beta * Phi^T Phi
        self.S_N_inv = self.alpha * np.eye(M_dim) + self.beta * (Phi.T @ Phi)
        self.S_N = np.linalg.inv(self.S_N_inv)
        
        # m_N = beta * S_N * Phi^T * t
        self.m_N = self.beta * (self.S_N @ Phi.T @ t_arr)
        return self

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute predictive mean and variance (Eq 2.118).
        Returns (mean, variance).
        """
        if self.m_N is None or self.S_N is None:
            raise RuntimeError("Model must be fitted before predict.")
        Phi = self._design_matrix(x)
        mean = Phi @ self.m_N
        # Var[t|x, D] = 1/beta + phi(x)^T S_N phi(x)
        variance = (1.0 / self.beta) + np.sum((Phi @ self.S_N) * Phi, axis=1)
        return mean, variance

    def sample_weights(self, size: int = 5, seed: Optional[int] = None) -> np.ndarray:
        """Sample weight vectors w ~ N(m_N, S_N) from posterior."""
        if self.m_N is None or self.S_N is None:
            raise RuntimeError("Model must be fitted before sampling weights.")
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(mean=self.m_N, cov=self.S_N, size=size)


# Efron Dice definitions from Bishop Figure 2.16
EFRON_DICE = {
    'Yellow': np.array([3, 3, 3, 3, 3, 3]),
    'Blue': np.array([0, 4, 4, 4, 0, 4]),     # four 4s, two 0s
    'Green': np.array([5, 1, 5, 1, 5, 1]),    # three 5s, three 1s
    'Red': np.array([2, 2, 6, 2, 2, 6]),      # four 2s, two 6s
}

def efron_dice_win_probability(die_a: np.ndarray, die_b: np.ndarray) -> float:
    """
    Calculate exact win probability P(die_a > die_b) for two 6-sided dice.
    """
    wins = 0
    total = len(die_a) * len(die_b)
    for v_a in die_a:
        for v_b in die_b:
            if v_a > v_b:
                wins += 1
            elif v_a == v_b:
                wins += 0.5  # Tie
    return wins / total


def alpha_divergence_discrete(p: np.ndarray, q: np.ndarray, alpha: float) -> float:
    """
    Compute alpha-divergence between two discrete probability distributions:
        D_alpha(p || q) = 4 / (1 - alpha^2) * (1 - sum_i p_i^{(1+alpha)/2} * q_i^{(1-alpha)/2})
    Handles the limits alpha -> 1 (KL(p || q)) and alpha -> -1 (KL(q || p)).
    (Bishop Eq 2.129).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if np.isclose(alpha, 1.0, atol=1e-5):
        return kl_divergence_discrete(p, q, base='e')
    elif np.isclose(alpha, -1.0, atol=1e-5):
        return kl_divergence_discrete(q, p, base='e')
    
    a = (1.0 + alpha) / 2.0
    b = (1.0 - alpha) / 2.0
    integral_val = np.sum((p ** a) * (q ** b))
    return float((4.0 / (1.0 - alpha ** 2)) * (1.0 - integral_val))


def bent_coin_bayes(
    p_heads_given_H1: float = 0.4,
    p_heads_given_H2: float = 0.6,
    prior_H1: float = 0.1,
    n_heads: int = 8,
    n_tails: int = 2
) -> Dict[str, float]:
    """
    Solve Exercise 2.40: Bent coin Bayesian inference.
    H1: convex side is heads -> P(heads | H1) = 0.40
    H2: concave side is heads -> P(heads | H2) = 0.60
    Prior P(H1) = 0.10, P(H2) = 0.90.
    Observed: n_heads=8, n_tails=2.
    """
    import math
    n_total = n_heads + n_tails
    comb = math.comb(n_total, n_heads)
    
    prior_H2 = 1.0 - prior_H1
    lik_H1 = comb * (p_heads_given_H1 ** n_heads) * ((1.0 - p_heads_given_H1) ** n_tails)
    lik_H2 = comb * (p_heads_given_H2 ** n_heads) * ((1.0 - p_heads_given_H2) ** n_tails)
    
    evidence = lik_H1 * prior_H1 + lik_H2 * prior_H2
    post_H1 = (lik_H1 * prior_H1) / evidence
    post_H2 = (lik_H2 * prior_H2) / evidence
    
    p_next_heads = p_heads_given_H1 * post_H1 + p_heads_given_H2 * post_H2
    
    return {
        'prior_H1': prior_H1,
        'prior_H2': prior_H2,
        'likelihood_H1': lik_H1,
        'likelihood_H2': lik_H2,
        'evidence': evidence,
        'posterior_H1': post_H1,
        'posterior_H2': post_H2,
        'p_next_heads': p_next_heads
    }


def binary_joint_entropy_analysis(p_xy: np.ndarray) -> Dict[str, float]:
    """
    Analyze all information theoretic quantities for a 2D discrete joint distribution:
    H[X], H[Y], H[X, Y], H[Y|X], H[X|Y], I[X; Y].
    (Exercise 2.36).
    """
    p_xy = np.asarray(p_xy, dtype=np.float64)
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    
    # Entropy helper in nats
    def _ent(prob):
        nz = prob[prob > 0]
        return -float(np.sum(nz * np.log(nz)))
    
    h_x = _ent(p_x)
    h_y = _ent(p_y)
    h_xy = _ent(p_xy.flatten())
    h_y_given_x = h_xy - h_x
    h_x_given_y = h_xy - h_y
    mi = h_x + h_y - h_xy
    
    return {
        'H_X_nats': h_x,
        'H_Y_nats': h_y,
        'H_XY_nats': h_xy,
        'H_Y_given_X_nats': h_y_given_x,
        'H_X_given_Y_nats': h_x_given_y,
        'I_XY_nats': mi,
        'H_X_bits': h_x / np.log(2),
        'H_Y_bits': h_y / np.log(2),
        'H_XY_bits': h_xy / np.log(2),
        'H_Y_given_X_bits': h_y_given_x / np.log(2),
        'H_X_given_Y_bits': h_x_given_y / np.log(2),
        'I_XY_bits': mi / np.log(2),
    }


# ==============================================================================
# Chapter 3: Standard Distributions - 3.1 Discrete Variables
# ==============================================================================

class BernoulliDistribution:
    """
    Bernoulli distribution for a single binary variable x in {0, 1} (Section 3.1.1).
    
    Formula:
      p(x | mu) = mu^x * (1 - mu)^(1 - x)   (Eq 3.2)
      E[x] = mu                               (Eq 3.3)
      var[x] = mu * (1 - mu)                  (Eq 3.4)
    """
    def __init__(self, mu: float):
        if not (0.0 <= mu <= 1.0):
            raise ValueError(f"Parameter mu must be in [0, 1], got {mu}")
        self.mu = float(mu)

    def pmf(self, x: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute probability mass p(x | mu) (Eq 3.2)."""
        x_arr = np.asarray(x)
        # Check support: x must be in {0, 1}
        if not np.all(np.isin(x_arr, [0, 1])):
            raise ValueError("Bernoulli variable x must be 0 or 1")
        # Direct calculation: p(1) = mu, p(0) = 1 - mu
        prob = np.where(x_arr == 1, self.mu, 1.0 - self.mu)
        if np.isscalar(x):
            return float(prob)
        return prob

    def log_pmf(self, x: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute log probability ln p(x | mu) = x ln(mu) + (1-x) ln(1-mu)."""
        x_arr = np.asarray(x)
        if not np.all(np.isin(x_arr, [0, 1])):
            raise ValueError("Bernoulli variable x must be 0 or 1")
        eps = 1e-15
        mu_clamped = np.clip(self.mu, eps, 1.0 - eps)
        log_prob = x_arr * np.log(mu_clamped) + (1.0 - x_arr) * np.log(1.0 - mu_clamped)
        if np.isscalar(x):
            return float(log_prob)
        return log_prob

    @property
    def mean(self) -> float:
        """E[x] = mu (Eq 3.3)."""
        return self.mu

    @property
    def variance(self) -> float:
        """var[x] = mu * (1 - mu) (Eq 3.4)."""
        return self.mu * (1.0 - self.mu)

    def sample(self, size: Union[int, Tuple[int, ...]] = 1, seed: Optional[int] = None) -> np.ndarray:
        """Draw random samples from Bern(x | mu)."""
        rng = np.random.default_rng(seed)
        return rng.binomial(n=1, p=self.mu, size=size)

    @staticmethod
    def log_likelihood(data: np.ndarray, mu: float) -> float:
        """
        Compute log-likelihood ln p(D | mu) (Eq 3.6):
          ln p(D | mu) = sum_{n=1}^N { x_n ln(mu) + (1 - x_n) ln(1 - mu) }
        """
        data = np.asarray(data)
        if not np.all(np.isin(data, [0, 1])):
            raise ValueError("Data elements must be 0 or 1")
        eps = 1e-15
        mu_clamped = np.clip(mu, eps, 1.0 - eps)
        m = np.sum(data == 1)
        N = data.size
        return float(m * np.log(mu_clamped) + (N - m) * np.log(1.0 - mu_clamped))

    @staticmethod
    def fit_mle(data: np.ndarray) -> float:
        """
        Compute Maximum Likelihood Estimator mu_ML = m / N (Eq 3.7, 3.8).
        """
        data = np.asarray(data)
        if data.size == 0:
            raise ValueError("Data cannot be empty")
        if not np.all(np.isin(data, [0, 1])):
            raise ValueError("Data elements must be 0 or 1")
        return float(np.mean(data))


class BinomialDistribution:
    """
    Binomial distribution Bin(m | N, mu) for the number of successes m in N trials (Section 3.1.2).
    
    Formula:
      Bin(m | N, mu) = (N choose m) * mu^m * (1 - mu)^(N - m)   (Eq 3.9)
      (N choose m) = N! / ((N - m)! * m!)                       (Eq 3.10)
      E[m] = N * mu                                             (Eq 3.11)
      var[m] = N * mu * (1 - mu)                                (Eq 3.12)
    """
    def __init__(self, N: int, mu: float):
        if not isinstance(N, (int, np.integer)) or N < 0:
            raise ValueError(f"N must be a non-negative integer, got {N}")
        if not (0.0 <= mu <= 1.0):
            raise ValueError(f"Parameter mu must be in [0, 1], got {mu}")
        self.N = int(N)
        self.mu = float(mu)

    def pmf(self, m: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute Bin(m | N, mu) (Eq 3.9)."""
        m_arr = np.asarray(m)
        is_scalar = np.isscalar(m)
        
        # Valid domain: 0 <= m <= N and integer
        valid_mask = (m_arr >= 0) & (m_arr <= self.N) & (np.floor(m_arr) == m_arr)
        
        result = np.zeros_like(m_arr, dtype=np.float64)
        if np.any(valid_mask):
            m_valid = m_arr[valid_mask].astype(int)
            # Use log-factorial / gammaln for numerical stability
            log_comb = (
                special.gammaln(self.N + 1)
                - special.gammaln(m_valid + 1)
                - special.gammaln(self.N - m_valid + 1)
            )
            # Handle boundary probabilities 0 and 1
            if self.mu == 0.0:
                prob = np.where(m_valid == 0, 1.0, 0.0)
            elif self.mu == 1.0:
                prob = np.where(m_valid == self.N, 1.0, 0.0)
            else:
                log_prob = (
                    log_comb
                    + m_valid * np.log(self.mu)
                    + (self.N - m_valid) * np.log(1.0 - self.mu)
                )
                prob = np.exp(log_prob)
            result[valid_mask] = prob
            
        if is_scalar:
            return float(result)
        return result

    def log_pmf(self, m: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute ln Bin(m | N, mu)."""
        prob = self.pmf(m)
        eps = 1e-300
        return np.log(np.maximum(prob, eps))

    @property
    def mean(self) -> float:
        """E[m] = N * mu (Eq 3.11)."""
        return self.N * self.mu

    @property
    def variance(self) -> float:
        """var[m] = N * mu * (1 - mu) (Eq 3.12)."""
        return self.N * self.mu * (1.0 - self.mu)

    def pmf_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return all possible outcomes m = 0, 1, ..., N and their probabilities.
        Useful for reproducing Figure 3.1.
        """
        m_vals = np.arange(self.N + 1, dtype=int)
        probs = self.pmf(m_vals)
        return m_vals, probs

    def sample(self, size: Union[int, Tuple[int, ...]] = 1, seed: Optional[int] = None) -> np.ndarray:
        """Draw samples from Bin(m | N, mu)."""
        rng = np.random.default_rng(seed)
        return rng.binomial(n=self.N, p=self.mu, size=size)

    @staticmethod
    def fit_mle(m: int, N: int) -> float:
        """Maximum likelihood estimator for binomial: mu_ML = m / N."""
        if N <= 0:
            raise ValueError("N must be strictly positive")
        if not (0 <= m <= N):
            raise ValueError(f"m must be between 0 and N, got m={m}, N={N}")
        return float(m / N)


class MultinomialDistribution:
    """
    Multinomial distribution and 1-of-K categorical scheme (Section 3.1.3).
    
    Formula:
      Categorical: p(x | mu) = prod_{k=1}^K mu_k^{x_k}                     (Eq 3.14)
      E[x | mu] = mu                                                       (Eq 3.16)
      Sufficient statistics: m_k = sum_{n=1}^N x_{nk}                       (Eq 3.18)
      MLE via Lagrange multipliers: mu_k^ML = m_k / N                       (Eq 3.22)
      Multinomial: Mult(m_1, ..., m_K | mu, N) = (N! / prod m_k!) prod mu_k^{m_k} (Eq 3.23)
    """
    def __init__(self, mu: Union[List[float], np.ndarray], N: int = 1):
        self.mu = np.asarray(mu, dtype=np.float64)
        if self.mu.ndim != 1:
            raise ValueError(f"mu must be a 1D probability vector, got shape {self.mu.shape}")
        if np.any(self.mu < 0.0):
            raise ValueError("Probabilities mu_k must be non-negative")
        s = np.sum(self.mu)
        if not np.isclose(s, 1.0, atol=1e-5):
            raise ValueError(f"Probabilities must sum to 1, got sum {s}")
        # Normalize slightly to ensure exact sum of 1.0
        self.mu = self.mu / np.sum(self.mu)
        self.K = len(self.mu)
        if not isinstance(N, (int, np.integer)) or N < 0:
            raise ValueError(f"N must be a non-negative integer, got {N}")
        self.N = int(N)

    def pmf_categorical(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """
        Probability p(x | mu) = prod_{k=1}^K mu_k^{x_k} for 1-of-K vectors (Eq 3.14).
        """
        x = np.asarray(x)
        if x.shape[-1] != self.K:
            raise ValueError(f"Last dimension of x must equal K={self.K}, got {x.shape}")
        # Check one-hot condition
        if not np.all(np.isclose(np.sum(x, axis=-1), 1.0)) or not np.all(np.isin(x, [0, 1])):
            raise ValueError("Input x must be valid 1-of-K (one-hot) vector")
        # For one-hot vector, prod mu_k^{x_k} is simply mu_k for the active k
        active_indices = np.argmax(x, axis=-1)
        prob = self.mu[active_indices]
        if prob.ndim == 0:
            return float(prob)
        return prob

    def pmf(self, m: np.ndarray) -> float:
        """
        Multinomial distribution Mult(m_1, ..., m_K | mu, N) (Eq 3.23, 3.24).
        """
        m = np.asarray(m, dtype=int)
        if m.shape != (self.K,):
            raise ValueError(f"m must have shape ({self.K},), got {m.shape}")
        if np.any(m < 0):
            raise ValueError("Counts m_k must be non-negative")
        if np.sum(m) != self.N:
            raise ValueError(f"Counts m_k must sum to N={self.N}, got sum {np.sum(m)}")
        
        # log Mult = ln(N!) - sum ln(m_k!) + sum m_k ln(mu_k)
        log_coef = special.gammaln(self.N + 1) - np.sum(special.gammaln(m + 1))
        
        eps = 1e-300
        # If m_k == 0, m_k * ln(mu_k) = 0 even if mu_k == 0
        log_term = np.where(m > 0, m * np.log(np.maximum(self.mu, eps)), 0.0)
        log_prob = log_coef + np.sum(log_term)
        return float(np.exp(log_prob))

    @property
    def mean(self) -> np.ndarray:
        """E[m] = N * mu (Eq 3.16 when N=1)."""
        return self.N * self.mu

    @property
    def covariance(self) -> np.ndarray:
        """
        Covariance matrix of counts m:
          Cov[m_i, m_j] = - N * mu_i * mu_j (i != j)
          Var[m_i] = N * mu_i * (1 - mu_i) (i == j)
        """
        cov = - self.N * np.outer(self.mu, self.mu)
        diag = self.N * self.mu * (1.0 - self.mu)
        np.fill_diagonal(cov, diag)
        return cov

    def sample(self, size: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """Draw count vector samples from Mult(m | mu, N)."""
        rng = np.random.default_rng(seed)
        return rng.multinomial(n=self.N, pvals=self.mu, size=size)

    @staticmethod
    def compute_sufficient_statistics(data_one_hot: np.ndarray) -> np.ndarray:
        """
        Compute sufficient statistics m_k = sum_{n=1}^N x_{nk} (Eq 3.18).
        """
        data_one_hot = np.asarray(data_one_hot)
        if data_one_hot.ndim != 2:
            raise ValueError(f"Expected 2D array of one-hot vectors (N, K), got shape {data_one_hot.shape}")
        return np.sum(data_one_hot, axis=0)

    @staticmethod
    def fit_mle(data_or_counts: np.ndarray, is_counts: bool = False) -> np.ndarray:
        """
        Compute Maximum Likelihood Estimator mu_k^ML = m_k / N (Eq 3.22)
        derived via Lagrange multipliers.
        """
        arr = np.asarray(data_or_counts, dtype=np.float64)
        if is_counts:
            counts = arr
            N = np.sum(counts)
            if N <= 0:
                raise ValueError("Total count must be positive")
            return counts / N
        else:
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array of one-hot vectors (N, K), got {arr.shape}")
            counts = np.sum(arr, axis=0)
            N = arr.shape[0]
            if N <= 0:
                raise ValueError("Data points N must be positive")
            return counts / N


def plot_figure_3_1(
    N: int = 10,
    mu: float = 0.25,
    save_paths: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Faithfully reproduce Figure 3.1 from Bishop & Bishop (2024), page 68:
    Histogram plot of the binomial distribution (3.9) as a function of m for N = 10 and mu = 0.25.
    """
    binom = BinomialDistribution(N=N, mu=mu)
    m_vals, probs = binom.pmf_all()

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=300)
        created_fig = True
    else:
        fig = ax.get_figure()

    # Bishop textbook blue style
    bar_color = '#0000FF'
    edge_color = 'black'
    
    bars = ax.bar(
        m_vals,
        probs,
        width=0.72,
        color=bar_color,
        edgecolor=edge_color,
        linewidth=0.8,
        zorder=3
    )

    ax.set_xlim(-0.6, N + 0.6)
    ax.set_ylim(0.0, 0.3)
    ax.set_xticks(np.arange(0, N + 1))
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.set_xlabel(r'$m$', fontsize=12)
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, labelsize=10)

    # Clean styling
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)
        
    ax.grid(False) # Textbook figure has no grid lines

    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.1 saved successfully to: {p}")

    if show and created_fig:
        plt.show()

    return fig, ax


# ==============================================================================
# Chapter 3.2: The Multivariate Gaussian
# ==============================================================================

class MultivariateGaussian:
    """
    Multivariate Gaussian distribution N(x | mu, Sigma) in D dimensions (Section 3.2).
    
    Formulas:
      p(x | mu, Sigma) = 1 / ((2pi)^(D/2) |Sigma|^(1/2)) * exp(-1/2 * (x - mu)^T Sigma^-1 (x - mu))  (Eq 3.26)
      Delta^2 = (x - mu)^T Sigma^-1 (x - mu)                                                           (Eq 3.27)
      E[x] = mu                                                                                        (Eq 3.48)
      cov[x] = E[(x - mu)(x - mu)^T] = Sigma                                                          (Eq 3.54)
    """
    def __init__(
        self,
        mu: Union[List[float], np.ndarray],
        sigma: Union[List[List[float]], np.ndarray],
        allow_singular: bool = False
    ):
        self.mu = np.asarray(mu, dtype=np.float64).flatten()
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.D = len(self.mu)

        if self.sigma.shape != (self.D, self.D):
            raise ValueError(
                f"Dimension mismatch: mu has dimension {self.D}, but sigma has shape {self.sigma.shape}"
            )

        # Check symmetry
        if not np.allclose(self.sigma, self.sigma.T, atol=1e-8):
            raise ValueError("Covariance matrix Sigma must be symmetric")

        # Eigendecomposition Sigma * u_i = lambda_i * u_i (Eq 3.28, 3.30)
        eigvals, eigvecs = la.eigh(self.sigma)
        sort_idx = np.argsort(eigvals)[::-1]
        self.eigenvalues = eigvals[sort_idx]
        self.eigenvectors = eigvecs[:, sort_idx]  # Columns are u_i

        if not allow_singular and np.any(self.eigenvalues <= 0):
            raise ValueError(
                f"Covariance matrix Sigma must be strictly positive definite, got eigenvalues: {self.eigenvalues}"
            )

        # Precision matrix Lambda = Sigma^-1 (Eq 3.31)
        self.precision = la.inv(self.sigma)

        # Log-determinant and normalization constant: ln |Sigma| = sum ln(lambda_i) (Eq 3.38)
        self.log_det = float(np.sum(np.log(np.maximum(self.eigenvalues, 1e-300))))
        self.log_norm_const = -0.5 * (self.D * np.log(2.0 * np.pi) + self.log_det)

        # Cholesky factor for fast sampling Sigma = L @ L.T
        try:
            self._chol_L = la.cholesky(self.sigma, lower=True)
        except la.LinAlgError:
            self._chol_L = None

    @property
    def mean(self) -> np.ndarray:
        """Mean vector mu (Eq 3.48)."""
        return self.mu.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Covariance matrix Sigma (Eq 3.54)."""
        return self.sigma.copy()

    def mahalanobis_distance_squared(self, x: Union[List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute squared Mahalanobis distance Delta^2 = (x - mu)^T Sigma^-1 (x - mu) (Eq 3.27).
        """
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            diff = x_arr - self.mu
            return float(diff.T @ self.precision @ diff)
        else:
            diff = x_arr - self.mu
            return np.sum(diff * (diff @ self.precision), axis=-1)

    def log_pdf(self, x: Union[List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """Compute log probability density ln N(x | mu, Sigma)."""
        dist_sq = self.mahalanobis_distance_squared(x)
        log_prob = self.log_norm_const - 0.5 * dist_sq
        if np.isscalar(dist_sq):
            return float(log_prob)
        return log_prob

    def pdf(self, x: Union[List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """Compute probability density N(x | mu, Sigma) (Eq 3.26)."""
        return np.exp(self.log_pdf(x))

    def transform_to_eigen_basis(self, x: np.ndarray) -> np.ndarray:
        """
        Transform x to coordinate system defined by eigenvectors of Sigma:
          y_i = u_i^T (x - mu), or y = U^T (x - mu) (Eq 3.32)
        """
        x_arr = np.asarray(x, dtype=np.float64)
        diff = x_arr - self.mu
        if diff.ndim == 1:
            return self.eigenvectors.T @ diff
        return diff @ self.eigenvectors

    def transform_from_eigen_basis(self, y: np.ndarray) -> np.ndarray:
        """
        Transform from eigenvector basis back to original coordinates:
          x = mu + U y
        """
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            return self.mu + self.eigenvectors @ y_arr
        return self.mu + y_arr @ self.eigenvectors.T

    def covariance_type(self) -> str:
        """
        Determine if covariance is 'spherical' (isotropic), 'diagonal', or 'full' (Figure 3.4).
        """
        off_diag = self.sigma - np.diag(np.diag(self.sigma))
        if np.allclose(off_diag, 0.0, atol=1e-7):
            diag_vals = np.diag(self.sigma)
            if np.allclose(diag_vals, diag_vals[0], atol=1e-7):
                return 'spherical'
            return 'diagonal'
        return 'full'

    def sample(self, size: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """Draw samples from N(x | mu, Sigma) using Cholesky factor."""
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(size=(size, self.D))
        if self._chol_L is not None:
            samples = self.mu + z @ self._chol_L.T
        else:
            U = self.eigenvectors
            L = U * np.sqrt(np.maximum(self.eigenvalues, 0.0))
            samples = self.mu + z @ L.T
        if size == 1:
            return samples[0]
        return samples

    def condition_on(
        self,
        indices_a: List[int],
        indices_b: List[int],
        x_b: Union[List[float], np.ndarray]
    ) -> "MultivariateGaussian":
        """
        Compute conditional distribution p(x_a | x_b) = N(x_a | mu_{a|b}, Sigma_{a|b}) (Section 3.2.4).
        
        Formulas:
          mu_{a|b} = mu_a + Sigma_{ab} Sigma_{bb}^-1 (x_b - mu_b)       (Eq 3.80)
          Sigma_{a|b} = Sigma_{aa} - Sigma_{ab} Sigma_{bb}^-1 Sigma_{ba} (Eq 3.79)
          Sigma_{a|b} = Lambda_{aa}^-1                                  (Eq 3.73)
        """
        idx_a = np.asarray(indices_a, dtype=int)
        idx_b = np.asarray(indices_b, dtype=int)
        x_b_arr = np.asarray(x_b, dtype=np.float64).flatten()

        if len(x_b_arr) != len(idx_b):
            raise ValueError(f"Length of x_b ({len(x_b_arr)}) must match indices_b ({len(idx_b)})")

        mu_a = self.mu[idx_a]
        mu_b = self.mu[idx_b]

        sigma_aa = self.sigma[np.ix_(idx_a, idx_a)]
        sigma_ab = self.sigma[np.ix_(idx_a, idx_b)]
        sigma_ba = self.sigma[np.ix_(idx_b, idx_a)]
        sigma_bb = self.sigma[np.ix_(idx_b, idx_b)]

        sigma_bb_inv_sigma_ba = la.solve(sigma_bb, sigma_ba)
        mu_a_cond = mu_a + (sigma_ab @ la.solve(sigma_bb, x_b_arr - mu_b))
        sigma_a_cond = sigma_aa - (sigma_ab @ sigma_bb_inv_sigma_ba)
        sigma_a_cond = 0.5 * (sigma_a_cond + sigma_a_cond.T)

        return MultivariateGaussian(mu=mu_a_cond, sigma=sigma_a_cond)

    def marginalize(self, indices_a: List[int]) -> "MultivariateGaussian":
        """
        Compute marginal distribution p(x_a) = N(x_a | mu_a, Sigma_{aa}) (Section 3.2.5, Eq 3.82).
        """
        idx_a = np.asarray(indices_a, dtype=int)
        mu_a = self.mu[idx_a]
        sigma_aa = self.sigma[np.ix_(idx_a, idx_a)]
        return MultivariateGaussian(mu=mu_a, sigma=sigma_aa)

    @staticmethod
    def bayes_linear_gaussian(
        prior: "MultivariateGaussian",
        A: np.ndarray,
        b: np.ndarray,
        L_cov: np.ndarray,
        y_obs: Optional[np.ndarray] = None
    ) -> Tuple["MultivariateGaussian", Optional["MultivariateGaussian"]]:
        """
        Bayes' theorem for linear-Gaussian models (Section 3.2.6):
          Prior:       p(x) = N(x | mu, Lambda^-1)                (Eq 3.83)
          Conditional: p(y | x) = N(y | A x + b, L^-1)             (Eq 3.84)
          Marginal:    p(y) = N(y | A mu + b, L^-1 + A Sigma A^T)  (Eq 3.93, 3.94)
          Posterior:   p(x | y) = N(x | Sigma^* (A^T L (y - b) + Lambda mu), Sigma^*) (Eq 3.97, 3.98)
          where Sigma^* = (Lambda + A^T L A)^-1
        """
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64).flatten()
        L_cov = np.asarray(L_cov, dtype=np.float64)
        
        mu_y = A @ prior.mu + b
        sigma_y = L_cov + A @ prior.sigma @ A.T
        sigma_y = 0.5 * (sigma_y + sigma_y.T)
        marginal_y = MultivariateGaussian(mu=mu_y, sigma=sigma_y)

        posterior_x = None
        if y_obs is not None:
            y_arr = np.asarray(y_obs, dtype=np.float64).flatten()
            L_prec = la.inv(L_cov)
            sigma_star = la.inv(prior.precision + A.T @ L_prec @ A)
            sigma_star = 0.5 * (sigma_star + sigma_star.T)
            mu_star = sigma_star @ (A.T @ L_prec @ (y_arr - b) + prior.precision @ prior.mu)
            posterior_x = MultivariateGaussian(mu=mu_star, sigma=sigma_star)

        return marginal_y, posterior_x

    @staticmethod
    def fit_mle(X: np.ndarray, unbiased_cov: bool = False) -> "MultivariateGaussian":
        """
        Compute Maximum Likelihood Estimator for Multivariate Gaussian (Section 3.2.7):
          mu_ML = 1/N sum_{n=1}^N x_n                                 (Eq 3.106)
          Sigma_ML = 1/N sum_{n=1}^N (x_n - mu_ML)(x_n - mu_ML)^T     (Eq 3.108)
          Sigma_unbiased = 1/(N - 1) sum (x_n - mu_ML)(x_n - mu_ML)^T (Eq 3.109)
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Data matrix X must be 2D of shape (N, D), got {X_arr.shape}")
        N, D = X_arr.shape
        if N <= 0:
            raise ValueError("Data matrix cannot be empty")

        mu_ml = np.mean(X_arr, axis=0)
        diff = X_arr - mu_ml
        denom = (N - 1) if (unbiased_cov and N > 1) else N
        sigma_ml = (diff.T @ diff) / denom
        sigma_ml = 0.5 * (sigma_ml + sigma_ml.T)

        return MultivariateGaussian(mu=mu_ml, sigma=sigma_ml)

    @staticmethod
    def sequential_mean_update(mu_old: np.ndarray, x_new: np.ndarray, N: int) -> np.ndarray:
        """
        Sequential estimation of the mean (Section 3.2.8, Eq 3.110):
          mu^{(N)} = mu^{(N-1)} + 1/N (x_N - mu^{(N-1)})
        """
        if N <= 0:
            raise ValueError("Iteration count N must be strictly positive")
        return mu_old + (1.0 / N) * (x_new - mu_old)


class GaussianMixtureModel:
    """
    Gaussian Mixture Model (GMM) with K components (Section 3.2.9):
      p(x) = sum_{k=1}^K pi_k N(x | mu_k, Sigma_k)  (Eq 3.111)
      sum_{k=1}^K pi_k = 1,  0 <= pi_k <= 1          (Eq 3.112, 3.113)
      gamma_k(x) = pi_k N(x | mu_k, Sigma_k) / sum_j pi_j N(x | mu_j, Sigma_j) (Eq 3.119)
    """
    def __init__(self, pi: Union[List[float], np.ndarray], components: List[MultivariateGaussian]):
        self.pi = np.asarray(pi, dtype=np.float64)
        self.components = components
        self.K = len(self.pi)

        if len(self.components) != self.K:
            raise ValueError(f"Number of mixing weights ({self.K}) must match components ({len(self.components)})")
        if np.any(self.pi < 0.0):
            raise ValueError("Mixing coefficients must be non-negative")
        if not np.isclose(np.sum(self.pi), 1.0, atol=1e-5):
            raise ValueError(f"Mixing coefficients must sum to 1.0, got sum {np.sum(self.pi)}")

        self.pi = self.pi / np.sum(self.pi)
        self.D = self.components[0].D
        for k, comp in enumerate(self.components):
            if comp.D != self.D:
                raise ValueError(f"Component {k} dimension {comp.D} does not match model dimension {self.D}")

    def log_pdf(self, x: Union[List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """Compute log probability density ln p(x) using log-sum-exp for numerical stability."""
        x_arr = np.asarray(x, dtype=np.float64)
        is_single = (x_arr.ndim == 1)
        orig_shape = x_arr.shape

        if is_single:
            x_eval = x_arr.reshape(1, self.D)
        elif x_arr.ndim > 2:
            x_eval = x_arr.reshape(-1, self.D)
        else:
            x_eval = x_arr

        N_eval = x_eval.shape[0]
        comp_log_probs = np.zeros((N_eval, self.K))
        for k in range(self.K):
            comp_log_probs[:, k] = np.log(self.pi[k] + 1e-300) + self.components[k].log_pdf(x_eval)

        max_log = np.max(comp_log_probs, axis=1, keepdims=True)
        log_prob = np.squeeze(max_log + np.log(np.sum(np.exp(comp_log_probs - max_log), axis=1, keepdims=True)))

        if is_single:
            return float(log_prob)
        if len(orig_shape) > 2:
            return log_prob.reshape(orig_shape[:-1])
        return log_prob

    def pdf(self, x: Union[List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """Compute probability density p(x) = sum_k pi_k N(x | mu_k, Sigma_k) (Eq 3.111)."""
        return np.exp(self.log_pdf(x))

    def responsibilities(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Compute responsibilities gamma_k(x) = p(k | x) (Eq 3.119).
        Returns array of shape (N, K) or (K,) for single observation.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        is_single = (x_arr.ndim == 1)
        orig_shape = x_arr.shape

        if is_single:
            x_eval = x_arr.reshape(1, self.D)
        elif x_arr.ndim > 2:
            x_eval = x_arr.reshape(-1, self.D)
        else:
            x_eval = x_arr

        N_eval = x_eval.shape[0]
        weighted_log = np.zeros((N_eval, self.K))
        for k in range(self.K):
            weighted_log[:, k] = np.log(self.pi[k] + 1e-300) + self.components[k].log_pdf(x_eval)

        max_log = np.max(weighted_log, axis=1, keepdims=True)
        weighted_p = np.exp(weighted_log - max_log)
        gamma = weighted_p / np.sum(weighted_p, axis=1, keepdims=True)

        if is_single:
            return gamma[0]
        if len(orig_shape) > 2:
            return gamma.reshape(orig_shape[:-1] + (self.K,))
        return gamma

    def sample(self, size: int = 1, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Draw samples from GMM. Returns (samples, component_assignments)."""
        rng = np.random.default_rng(seed)
        k_indices = rng.choice(self.K, size=size, p=self.pi)
        samples = np.zeros((size, self.D))
        for i, k in enumerate(k_indices):
            samples[i] = self.components[k].sample(size=1, seed=None)
        if size == 1:
            return samples[0], int(k_indices[0])
        return samples, k_indices

    @classmethod
    def fit_em(
        cls,
        X: np.ndarray,
        K: int,
        max_iter: int = 100,
        tol: float = 1e-5,
        reg_cov: float = 1e-6,
        seed: Optional[int] = None
    ) -> "GaussianMixtureModel":
        """
        Fit GMM using Expectation-Maximization (EM) algorithm (Section 3.2.9).
        """
        rng = np.random.default_rng(seed)
        X_arr = np.asarray(X, dtype=np.float64)
        N, D = X_arr.shape

        means = np.zeros((K, D))
        init_idx = rng.choice(N)
        means[0] = X_arr[init_idx]
        for k in range(1, K):
            dists = np.min([np.sum((X_arr - means[j])**2, axis=1) for j in range(k)], axis=0)
            probs = dists / np.sum(dists)
            means[k] = X_arr[rng.choice(N, p=probs)]

        pi = np.full(K, 1.0 / K)
        base_cov = np.cov(X_arr, rowvar=False)
        if base_cov.ndim == 0:
            base_cov = np.array([[float(base_cov)]])
        covariances = np.array([base_cov.copy() + reg_cov * np.eye(D) for _ in range(K)])

        components = [MultivariateGaussian(mu=means[k], sigma=covariances[k]) for k in range(K)]
        model = cls(pi=pi, components=components)

        prev_ll = -np.inf
        for it in range(max_iter):
            gamma = model.responsibilities(X_arr)
            N_k = np.sum(gamma, axis=0)
            pi_new = N_k / N
            means_new = np.zeros((K, D))
            covs_new = np.zeros((K, D, D))

            for k in range(K):
                means_new[k] = np.sum(gamma[:, k:k+1] * X_arr, axis=0) / (N_k[k] + 1e-15)
                diff = X_arr - means_new[k]
                covs_new[k] = (gamma[:, k:k+1] * diff).T @ diff / (N_k[k] + 1e-15) + reg_cov * np.eye(D)
                covs_new[k] = 0.5 * (covs_new[k] + covs_new[k].T)

            components = [MultivariateGaussian(mu=means_new[k], sigma=covs_new[k]) for k in range(K)]
            model = cls(pi=pi_new, components=components)

            current_ll = np.sum(model.log_pdf(X_arr))
            if np.abs(current_ll - prev_ll) < tol:
                break
            prev_ll = current_ll

        return model


def plot_figure_3_2(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, Any]:
    """
    Faithfully reproduce Figure 3.2 from Bishop & Bishop (2024), page 71:
    Histogram plots of the mean of N uniformly distributed numbers for N = 1, 2, 10.
    """
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=300, sharey=True)
    N_values = [1, 2, 10]
    num_samples = 200000

    for ax, N in zip(axes, N_values):
        samples = np.mean(np.random.uniform(0.0, 1.0, size=(num_samples, N)), axis=1)
        ax.hist(
            samples, bins=50, range=(0.0, 1.0), density=True,
            color='#E8BA3A', edgecolor='black', linewidth=0.5
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 3.5)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0, 1, 2, 3])
        ax.text(0.5, 3.1, f"$N = {N}$", ha='center', fontsize=12, fontweight='bold')
        ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.2 saved successfully to: {p}")
    if show:
        plt.show()
    return fig, axes


def plot_figure_3_3(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Faithfully reproduce Figure 3.3 from Bishop & Bishop (2024), page 72:
    Elliptical surface of constant probability density with eigenvectors and eigenvalues.
    """
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=300)
    mu = np.array([2.5, 2.5])
    theta = np.radians(35)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    lambda1, lambda2 = 2.2, 0.6
    Sigma = R @ np.diag([lambda1, lambda2]) @ R.T

    eigvals, eigvecs = la.eigh(Sigma)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    u1, u2 = eigvecs[:, 0], eigvecs[:, 1]
    l1, l2 = eigvals[0], eigvals[1]

    t = np.linspace(0, 2*np.pi, 200)
    circle = np.array([np.cos(t), np.sin(t)])
    ellipse = mu[:, None] + eigvecs @ (np.diag(np.sqrt(eigvals)) @ circle)

    ax.plot(ellipse[0], ellipse[1], color='#E02020', linewidth=2.0, zorder=4)

    axis_len1 = 2.0 * np.sqrt(l1)
    axis_len2 = 2.0 * np.sqrt(l2)
    ax.plot([mu[0] - axis_len1*u1[0], mu[0] + axis_len1*u1[0]],
            [mu[1] - axis_len1*u1[1], mu[1] + axis_len1*u1[1]],
            color='gray', linestyle='--', linewidth=1.0, zorder=2)
    ax.plot([mu[0] - axis_len2*u2[0], mu[0] + axis_len2*u2[0]],
            [mu[1] - axis_len2*u2[1], mu[1] + axis_len2*u2[1]],
            color='gray', linestyle='--', linewidth=1.0, zorder=2)

    ax.annotate('', xy=mu + np.sqrt(l1)*u1, xytext=mu,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=mu + np.sqrt(l2)*u2, xytext=mu,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.scatter([mu[0]], [mu[1]], color='black', s=30, zorder=5)
    ax.text(mu[0] - 0.25, mu[1] - 0.35, r'$\boldsymbol{\mu}$', fontsize=13)

    p1 = mu + 0.65 * np.sqrt(l1)*u1
    ax.text(p1[0] + 0.1, p1[1] - 0.25, r'$\lambda_1^{1/2}\mathbf{u}_1$', fontsize=12)
    p2 = mu + 0.65 * np.sqrt(l2)*u2
    ax.text(p2[0] - 0.55, p2[1] + 0.15, r'$\lambda_2^{1/2}\mathbf{u}_2$', fontsize=12)

    end1 = mu + axis_len1 * u1
    ax.text(end1[0] + 0.1, end1[1], r'$\mathbf{u}_1$', fontsize=12, fontweight='bold')
    end2 = mu + axis_len2 * u2
    ax.text(end2[0], end2[1] + 0.1, r'$\mathbf{u}_2$', fontsize=12, fontweight='bold')

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)

    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.3 saved successfully to: {p}")
    if show:
        plt.show()
    return fig, ax


def plot_figure_3_4(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, Any]:
    """
    Faithfully reproduce Figure 3.4 from Bishop & Bishop (2024), page 76:
    Contours of constant probability density for (a) General, (b) Diagonal, (c) Isotropic covariance.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), dpi=300)
    titles = [
        "(a) General form\n(arbitrary covariance)",
        "(b) Diagonal\n(axis-aligned)",
        "(c) Spherical / Isotropic\n" + r"($\mathbf{\Sigma} = \sigma^2 \mathbf{I}$)"
    ]

    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    cov_general = np.array([[1.5, 0.9], [0.9, 1.0]])
    cov_diag = np.array([[1.8, 0.0], [0.0, 0.6]])
    cov_spherical = np.array([[1.0, 0.0], [0.0, 1.0]])
    covs = [cov_general, cov_diag, cov_spherical]

    for ax, cov, title in zip(axes, covs, titles):
        inv_cov = la.inv(cov)
        quad = np.einsum('...i,ij,...j->...', pos, inv_cov, pos)
        density = np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(la.det(cov)))

        levels = np.linspace(0.02, density.max() * 0.9, 6)
        ax.contour(X, Y, density, levels=levels, colors='#E02020', linewidths=1.5)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel(r'$x_1$', fontsize=11)
        ax.set_ylabel(r'$x_2$', fontsize=11)
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='major', direction='in', top=True, right=True)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.4 saved successfully to: {p}")
    if show:
        plt.show()
    return fig, axes


def plot_figure_3_5(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, Any]:
    """
    Faithfully reproduce Figure 3.5 from Bishop & Bishop (2024), page 82:
    Joint Gaussian contours, marginal distribution p(xa), and conditional distribution p(xa | xb = 0.7).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)
    
    mu = np.array([0.5, 0.5])
    sigma = np.array([[0.04, 0.032], [0.032, 0.04]])
    inv_sigma = la.inv(sigma)

    xa = np.linspace(0.0, 1.0, 200)
    xb = np.linspace(0.0, 1.0, 200)
    XA, XB = np.meshgrid(xa, xb)
    pos = np.dstack((XA - mu[0], XB - mu[1]))
    quad = np.einsum('...i,ij,...j->...', pos, inv_sigma, pos)
    density = np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(la.det(sigma)))

    # (a) Contours of p(xa, xb)
    levels = np.linspace(0.5, density.max() * 0.95, 7)
    ax1.contour(XA, XB, density, levels=levels, colors='#E02020', linewidths=1.4)
    xb_val = 0.7
    ax1.axhline(xb_val, color='gray', linestyle='--', linewidth=1.2)
    ax1.text(0.1, xb_val + 0.03, r'$x_b = 0.7$', fontsize=11)
    ax1.text(0.65, 0.35, r'$p(x_a, x_b)$', fontsize=12, color='#E02020')

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel(r'$x_a$', fontsize=12)
    ax1.set_ylabel(r'$x_b$', fontsize=12)
    ax1.set_title('(a) Joint distribution contours', fontsize=12)
    ax1.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for spine in ax1.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)

    # (b) Marginal and Conditional
    mu_a = mu[0]
    var_a = sigma[0, 0]
    p_xa_marginal = (1.0 / np.sqrt(2 * np.pi * var_a)) * np.exp(-0.5 * (xa - mu_a)**2 / var_a)

    mu_cond = mu_a + (sigma[0, 1] / sigma[1, 1]) * (xb_val - mu[1])
    var_cond = sigma[0, 0] - (sigma[0, 1]**2 / sigma[1, 1])
    p_xa_conditional = (1.0 / np.sqrt(2 * np.pi * var_cond)) * np.exp(-0.5 * (xa - mu_cond)**2 / var_cond)

    ax2.plot(xa, p_xa_marginal, color='#1E56A0', linewidth=2.0, label=r'$p(x_a)$ (marginal)')
    ax2.plot(xa, p_xa_conditional, color='#E02020', linewidth=2.0, label=r'$p(x_a | x_b = 0.7)$ (conditional)')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 7)
    ax2.set_xlabel(r'$x_a$', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('(b) Marginal and conditional distributions', fontsize=12)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax2.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for spine in ax2.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.5 saved successfully to: {p}")
    if show:
        plt.show()
    return fig, (ax1, ax2)


def plot_figure_3_6(
    data_path: str = 'common/data/faithful.csv',
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, Any]:
    """
    Faithfully reproduce Figure 3.6 from Bishop & Bishop (2024), page 86:
    Old Faithful data with (a) Single Gaussian fit (MLE), and (b) Two-component GMM (EM).
    """
    if not os.path.exists(data_path):
        candidates = [
            os.path.join('..', data_path),
            os.path.join(os.path.dirname(__file__), 'data', os.path.basename(data_path)),
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', data_path)),
            os.path.join(os.getcwd(), data_path),
            os.path.join(os.getcwd(), '..', data_path),
        ]
        for c in candidates:
            if os.path.exists(c):
                data_path = c
                break

    df = pd.read_csv(data_path)
    X = df[['duration', 'waiting']].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300, sharex=True, sharey=True)

    x_grid = np.linspace(1.2, 5.8, 200)
    y_grid = np.linspace(35, 100, 200)
    XX, YY = np.meshgrid(x_grid, y_grid)
    grid_pos = np.dstack((XX, YY))

    # (a) Single Gaussian MLE fit
    single_gauss = MultivariateGaussian.fit_mle(X)
    dens_mle = single_gauss.pdf(grid_pos)

    ax1.scatter(X[:, 0], X[:, 1], s=25, facecolors='none', edgecolors='#1E56A0', linewidth=0.9, alpha=0.8)
    levels1 = np.linspace(0.0005, dens_mle.max() * 0.9, 6)
    ax1.contour(XX, YY, dens_mle, levels=levels1, colors='#E02020', linewidths=1.4)
    ax1.set_xlabel('Eruption duration (min)', fontsize=11)
    ax1.set_ylabel('Waiting time to next eruption (min)', fontsize=11)
    ax1.set_title('(a) Single Gaussian fit (MLE)', fontsize=12)

    # (b) 2-Component GMM fit (EM)
    gmm = GaussianMixtureModel.fit_em(X, K=2, max_iter=100, seed=42)
    dens_gmm = gmm.pdf(grid_pos)

    ax2.scatter(X[:, 0], X[:, 1], s=25, facecolors='none', edgecolors='#1E56A0', linewidth=0.9, alpha=0.8)
    levels2 = np.linspace(0.0005, dens_gmm.max() * 0.9, 7)
    ax2.contour(XX, YY, dens_gmm, levels=levels2, colors='#E02020', linewidths=1.4)
    ax2.set_xlabel('Eruption duration (min)', fontsize=11)
    ax2.set_title('(b) Two-component Gaussian mixture (EM)', fontsize=12)

    for ax in (ax1, ax2):
        ax.set_xlim(1.2, 5.8)
        ax.set_ylim(35, 100)
        ax.tick_params(axis='both', which='major', direction='in', top=True, right=True)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.6 saved successfully to: {p}")
    if show:
        plt.show()
    return fig, (ax1, ax2)


def plot_figure_3_7(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Faithfully reproduce Figure 3.7 from Bishop & Bishop (2024), page 87:
    Example of a Gaussian mixture distribution in 1D showing 3 Gaussians in blue and sum in red.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    x = np.linspace(-4, 6, 500)
    means = [-1.5, 0.5, 3.0]
    variances = [0.4, 0.25, 0.7]
    weights = [0.35, 0.40, 0.25]

    mixture = np.zeros_like(x)
    for k, (mu, var, pi) in enumerate(zip(means, variances, weights)):
        comp = pi * (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * (x - mu)**2 / var)
        mixture += comp
        ax.plot(x, comp, color='#1E56A0', linestyle='--', linewidth=1.5,
                label=f'Component {k+1}')

    ax.plot(x, mixture, color='#E02020', linewidth=2.2, label=r'Sum $p(x) = \sum \pi_k \mathcal{N}_k$')

    ax.set_xlim(-4, 6)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$p(x)$', fontsize=12)
    ax.set_title('Gaussian mixture distribution in one dimension', fontsize=12)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.7 saved successfully to: {p}")
    if show:
        plt.show()
    return fig, ax


def plot_figure_3_8(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, Any]:
    """
    Faithfully reproduce Figure 3.8 from Bishop & Bishop (2024), page 88:
    2D mixture of three Gaussians: (a) Component contours, (b) Mixture contours, (c) 3D surface plot.
    """
    fig = plt.figure(figsize=(15, 4.5), dpi=300)

    pi = [0.5, 0.3, 0.2]
    mu1 = np.array([-1.0, -0.8])
    Sigma1 = np.array([[0.6, 0.3], [0.3, 0.5]])

    mu2 = np.array([1.2, 0.2])
    Sigma2 = np.array([[0.4, -0.2], [-0.2, 0.6]])

    mu3 = np.array([-0.5, 1.5])
    Sigma3 = np.array([[0.5, 0.1], [0.1, 0.4]])

    mus = [mu1, mu2, mu3]
    Sigmas = [Sigma1, Sigma2, Sigma3]
    colors = ['#E02020', '#1E56A0', '#2CA02C']

    x = np.linspace(-3.5, 3.5, 200)
    y = np.linspace(-3.0, 3.5, 200)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # (a) Component contours
    ax1 = fig.add_subplot(1, 3, 1)
    for k in range(3):
        diff = pos - mus[k]
        inv = la.inv(Sigmas[k])
        quad = np.einsum('...i,ij,...j->...', diff, inv, diff)
        comp_dens = np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(la.det(Sigmas[k])))
        levels = np.linspace(comp_dens.max() * 0.15, comp_dens.max() * 0.95, 4)
        ax1.contour(X, Y, comp_dens, levels=levels, colors=colors[k], linewidths=1.4)
        ax1.text(mus[k][0], mus[k][1] - 0.9, rf'$\pi_{k+1} = {pi[k]}$',
                 ha='center', fontsize=11, color=colors[k], fontweight='bold')

    ax1.set_xlim(-3.5, 3.5)
    ax1.set_ylim(-3.0, 3.5)
    ax1.set_xlabel(r'$x_1$', fontsize=11)
    ax1.set_ylabel(r'$x_2$', fontsize=11)
    ax1.set_title('(a) Component contours', fontsize=12)
    ax1.set_aspect('equal')
    ax1.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for spine in ax1.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)

    # (b) Marginal density contours
    ax2 = fig.add_subplot(1, 3, 2)
    mixture_dens = np.zeros_like(X)
    for k in range(3):
        diff = pos - mus[k]
        inv = la.inv(Sigmas[k])
        quad = np.einsum('...i,ij,...j->...', diff, inv, diff)
        comp_dens = np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(la.det(Sigmas[k])))
        mixture_dens += pi[k] * comp_dens

    levels_mix = np.linspace(0.015, mixture_dens.max() * 0.95, 8)
    ax2.contour(X, Y, mixture_dens, levels=levels_mix, colors='#E02020', linewidths=1.4)
    ax2.set_xlim(-3.5, 3.5)
    ax2.set_ylim(-3.0, 3.5)
    ax2.set_xlabel(r'$x_1$', fontsize=11)
    ax2.set_ylabel(r'$x_2$', fontsize=11)
    ax2.set_title(r'(b) Marginal density $p(\mathbf{x})$ contours', fontsize=12)
    ax2.set_aspect('equal')
    ax2.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for spine in ax2.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)

    # (c) 3D Surface
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    surf = ax3.plot_surface(X, Y, mixture_dens, cmap='viridis', edgecolor='none', alpha=0.9, antialiased=True)
    ax3.set_xlabel(r'$x_1$', fontsize=10, labelpad=5)
    ax3.set_ylabel(r'$x_2$', fontsize=10, labelpad=5)
    ax3.set_zlabel(r'$p(\mathbf{x})$', fontsize=10, labelpad=5)
    ax3.set_title(r'(c) 3D surface of $p(\mathbf{x})$', fontsize=12)
    ax3.view_init(elev=40, azim=-60)
    ax3.tick_params(labelsize=8)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.8 saved successfully to: {p}")
    if show:
        plt.show()
    return fig, (ax1, ax2, ax3)


# =====================================================================
# Chapter 3 Section 3.3: Periodic Variables (Von Mises Distribution)
# =====================================================================

class VonMisesDistribution:
    """
    Von Mises Distribution (Circular Normal) for periodic variables (Section 3.3.1):
      p(theta | theta_0, m) = 1 / (2*pi*I_0(m)) * exp{m * cos(theta - theta_0)}  (Eq 3.129)
    where:
      theta_0: mean angle (location parameter)
      m: concentration parameter (analogous to inverse variance / precision)
      I_0(m): zeroth-order modified Bessel function of the first kind (Eq 3.130)
    """
    def __init__(self, theta_0: float = 0.0, m: float = 1.0):
        if m < 0:
            raise ValueError(f"Concentration parameter m must be non-negative, got {m}")
        self.theta_0 = float(theta_0) % (2 * np.pi)
        self.m = float(m)

    def pdf(self, theta: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute probability density p(theta | theta_0, m) (Eq 3.129).
        Uses exponentially scaled Bessel function i0e(m) = exp(-m)*i0(m) for numerical stability.
        """
        th = np.asarray(theta, dtype=np.float64)
        is_scalar = (th.ndim == 0)
        
        if self.m == 0.0:
            val = np.full_like(th, 1.0 / (2 * np.pi))
        else:
            # exp(m * cos(th - th0)) / (2*pi * i0(m)) = exp(m * (cos(th - th0) - 1)) / (2*pi * i0e(m))
            cos_diff = np.cos(th - self.theta_0)
            val = np.exp(self.m * (cos_diff - 1.0)) / (2 * np.pi * special.i0e(self.m))
            
        if is_scalar:
            return float(val)
        return val

    def log_pdf(self, theta: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """Compute log probability density ln p(theta | theta_0, m) (Eq 3.131)."""
        th = np.asarray(theta, dtype=np.float64)
        is_scalar = (th.ndim == 0)
        
        if self.m == 0.0:
            val = np.full_like(th, -np.log(2 * np.pi))
        else:
            cos_diff = np.cos(th - self.theta_0)
            val = -np.log(2 * np.pi) - np.log(special.i0e(self.m)) + self.m * (cos_diff - 1.0)
            
        if is_scalar:
            return float(val)
        return val

    def sample(self, size: int = 1, seed: Optional[int] = None) -> Union[float, np.ndarray]:
        """Draw samples from von Mises distribution."""
        rng = np.random.default_rng(seed)
        samples = rng.vonmises(self.theta_0, self.m, size=size) % (2 * np.pi)
        if size == 1:
            return float(samples[0])
        return samples

    @staticmethod
    def circular_mean(thetas: Union[List[float], np.ndarray]) -> float:
        """
        Compute sample circular mean direction theta_bar (Eq 3.119, 3.134):
          theta_bar = atan2(1/N sum sin theta_n, 1/N sum cos theta_n)
        """
        th = np.asarray(thetas, dtype=np.float64)
        s = np.mean(np.sin(th))
        c = np.mean(np.cos(th))
        return float(np.arctan2(s, c) % (2 * np.pi))

    @staticmethod
    def circular_resultant_length(thetas: Union[List[float], np.ndarray]) -> float:
        """
        Compute sample mean resultant vector length r_bar in [0, 1] (Eq 3.118, 3.135):
          r_bar = ||1/N sum x_n|| = sqrt((1/N sum cos theta_n)^2 + (1/N sum sin theta_n)^2)
        """
        th = np.asarray(thetas, dtype=np.float64)
        s = np.mean(np.sin(th))
        c = np.mean(np.cos(th))
        return float(np.sqrt(s**2 + c**2))

    @staticmethod
    def circular_variance(thetas: Union[List[float], np.ndarray]) -> float:
        """Compute sample circular variance V = 1 - r_bar in [0, 1]."""
        return 1.0 - VonMisesDistribution.circular_resultant_length(thetas)

    @staticmethod
    def bessel_ratio_A(m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute A(m) = I_1(m) / I_0(m) (Eq 3.136)."""
        m_arr = np.asarray(m, dtype=np.float64)
        is_scalar = (m_arr.ndim == 0)
        res = np.zeros_like(m_arr)
        mask = (m_arr > 0)
        if np.any(mask):
            # i1e(m) / i0e(m) = i1(m) / i0(m)
            res[mask] = special.i1e(m_arr[mask]) / special.i0e(m_arr[mask])
        if is_scalar:
            return float(res)
        return res

    @classmethod
    def fit_mle(cls, thetas: Union[List[float], np.ndarray]) -> "VonMisesDistribution":
        """
        Compute Maximum Likelihood Estimator for von Mises distribution (Section 3.3.1):
          theta_0_ML = atan2(sum sin theta_n, sum cos theta_n)  (Eq 3.134)
          A(m_ML) = 1/N sum cos(theta_n - theta_0_ML) = r_bar     (Eq 3.135, 3.137)
        """
        th = np.asarray(thetas, dtype=np.float64)
        if len(th) == 0:
            raise ValueError("Data array cannot be empty")
        
        theta_0_ml = cls.circular_mean(th)
        r_bar = cls.circular_resultant_length(th)
        
        if r_bar < 1e-6:
            m_ml = 0.0
        elif r_bar >= 1.0 - 1e-6:
            m_ml = 500.0  # highly concentrated
        else:
            # Mardia & Jupp (2000) initial approximation
            if r_bar < 0.53:
                m_init = 2 * r_bar + r_bar**3 + (5.0 / 6.0) * r_bar**5
            elif r_bar < 0.85:
                m_init = -0.4 + 1.39 * r_bar + 0.43 / (1.0 - r_bar)
            else:
                m_init = 1.0 / (r_bar**3 - 4 * r_bar**2 + 3 * r_bar)
                
            from scipy.optimize import root_scalar
            def obj(m_val):
                return cls.bessel_ratio_A(m_val) - r_bar
            
            bracket_low = max(0.0, m_init * 0.5)
            bracket_high = min(1000.0, m_init * 2.0 + 1.0)
            try:
                sol = root_scalar(obj, bracket=[bracket_low, bracket_high], method='brentq')
                m_ml = float(sol.root)
            except Exception:
                sol = root_scalar(obj, bracket=[0.0, 500.0], method='brentq')
                m_ml = float(sol.root)
                
        return cls(theta_0=theta_0_ml, m=m_ml)


VonMises = VonMisesDistribution


class VonMisesMixture:
    """
    Mixture of K Von Mises distributions (Section 3.3.1):
      p(theta) = sum_{k=1}^K pi_k * p(theta | theta_{0k}, m_k)
    where:
      pi_k: mixing coefficients, sum(pi_k) = 1, pi_k >= 0
      theta_{0k}: mean direction of k-th component
      m_k: concentration parameter of k-th component
    """
    def __init__(
        self,
        weights: Sequence[float],
        theta_0s: Sequence[float],
        ms: Sequence[float]
    ):
        weights_arr = np.asarray(weights, dtype=np.float64)
        if np.any(weights_arr < 0):
            raise ValueError("Mixing weights must be non-negative")
        total = np.sum(weights_arr)
        if total <= 0:
            raise ValueError("Sum of weights must be positive")
        self.weights = weights_arr / total
        self.K = len(self.weights)
        if len(theta_0s) != self.K or len(ms) != self.K:
            raise ValueError("weights, theta_0s, and ms must have the same length")
        self.components = [VonMisesDistribution(theta_0=th, m=m) for th, m in zip(theta_0s, ms)]

    def pdf(self, theta: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """Compute mixture probability density p(theta)."""
        th = np.asarray(theta, dtype=np.float64)
        is_scalar = (th.ndim == 0)
        densities = np.zeros_like(th)
        for pi_k, comp in zip(self.weights, self.components):
            densities += pi_k * comp.pdf(th)
        if is_scalar:
            return float(densities)
        return densities

    def log_pdf(self, theta: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """Compute mixture log probability density ln p(theta) using log-sum-exp."""
        th = np.asarray(theta, dtype=np.float64)
        is_scalar = (th.ndim == 0)
        th_flat = np.atleast_1d(th)
        log_comp = np.zeros((len(th_flat), self.K))
        for k, (pi_k, comp) in enumerate(zip(self.weights, self.components)):
            log_comp[:, k] = np.log(pi_k + 1e-300) + comp.log_pdf(th_flat)
        res = special.logsumexp(log_comp, axis=1)
        if is_scalar:
            return float(res[0])
        return res.reshape(th.shape)

    def responsibilities(self, theta: Union[List[float], np.ndarray]) -> np.ndarray:
        """Compute responsibilities gamma_{nk} for observations theta."""
        th = np.asarray(theta, dtype=np.float64).ravel()
        log_comp = np.zeros((len(th), self.K))
        for k, (pi_k, comp) in enumerate(zip(self.weights, self.components)):
            log_comp[:, k] = np.log(pi_k + 1e-300) + comp.log_pdf(th)
        log_norm = special.logsumexp(log_comp, axis=1, keepdims=True)
        return np.exp(log_comp - log_norm)

    def sample(self, size: int = 1, seed: Optional[int] = None) -> Union[float, np.ndarray]:
        """Sample from the von Mises mixture."""
        rng = np.random.default_rng(seed)
        k_indices = rng.choice(self.K, size=size, p=self.weights)
        samples = np.zeros(size)
        for i, k in enumerate(k_indices):
            samples[i] = self.components[k].sample(size=1, seed=int(rng.integers(0, 1000000)))
        if size == 1:
            return float(samples[0])
        return samples


def plot_figure_3_9(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Faithfully reproduce Figure 3.9 from Bishop & Bishop (2024), page 90:
    Periodic variables on unit circle and sample mean vector.
    """
    from matplotlib.patches import Wedge

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)
    
    # Red unit circle
    theta_grid = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta_grid), np.sin(theta_grid), color='#E02020', linewidth=1.8, zorder=2)

    # Coordinate axes with arrowheads
    ax.annotate('', xy=(1.40, 0), xytext=(-1.25, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2, mutation_scale=12), zorder=1)
    ax.annotate('', xy=(0, 1.40), xytext=(0, -1.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2, mutation_scale=12), zorder=1)
    ax.text(1.45, -0.04, r'$x_1$', fontsize=13, va='top', ha='left')
    ax.text(-0.06, 1.45, r'$x_2$', fontsize=13, va='bottom', ha='right')

    # Data points x_1, x_2, x_3, x_4 on the circle (Bishop page 90)
    thetas = np.array([-0.30, 0.58, 1.30, 2.25])
    labels = [r'$\mathbf{x}_1$', r'$\mathbf{x}_2$', r'$\mathbf{x}_3$', r'$\mathbf{x}_4$']
    offsets = [(0.12, -0.08), (0.12, 0.05), (0.05, 0.12), (-0.12, 0.08)]
    
    for th, lbl, off in zip(thetas, labels, offsets):
        x_i = np.cos(th)
        y_i = np.sin(th)
        ax.scatter([x_i], [y_i], color='#1E56A0', s=45, zorder=5)
        ax.text(x_i + off[0], y_i + off[1], lbl, fontsize=12, ha='center', va='center', color='black')

    # Sample mean vector x_bar
    x_vecs = np.column_stack([np.cos(thetas), np.sin(thetas)])
    x_bar = np.mean(x_vecs, axis=0)
    theta_bar = np.arctan2(x_bar[1], x_bar[0])

    # Shaded wedge for theta_bar
    wedge = Wedge((0, 0), 0.35, 0, np.degrees(theta_bar), facecolor='#30B0B0', alpha=0.85, edgecolor='black', linewidth=1.0, zorder=3)
    ax.add_patch(wedge)
    ax.text(0.42 * np.cos(theta_bar / 2), 0.42 * np.sin(theta_bar / 2), r'$\bar{\theta}$', fontsize=13, va='center', ha='center')

    # Vector x_bar
    ax.annotate('', xy=(x_bar[0], x_bar[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#1E56A0', lw=2.4, mutation_scale=15), zorder=4)
    ax.scatter([x_bar[0]], [x_bar[1]], color='#1E56A0', s=50, zorder=6)
    ax.text(x_bar[0] + 0.08, x_bar[1] + 0.02, r'$\bar{\mathbf{x}}$', fontsize=14, color='black', fontweight='bold')
    ax.text(0.5 * x_bar[0] - 0.07, 0.5 * x_bar[1] + 0.07, r'$\bar{r}$', fontsize=13, color='black')

    ax.set_xlim(-1.35, 1.55)
    ax.set_ylim(-1.35, 1.55)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.9 saved successfully to: {p}")
    if show:
        plt.show()
    return fig, ax


def plot_figure_3_10(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Faithfully reproduce Figure 3.10 from Bishop & Bishop (2024), page 91:
    2D Gaussian conditioned on unit circle yielding the von Mises distribution.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)
    r0 = 1.3
    theta0 = np.radians(50)
    mu1 = r0 * np.cos(theta0)
    mu2 = r0 * np.sin(theta0)
    sigma = 0.55

    x = np.linspace(-1.5, 2.5, 300)
    y = np.linspace(-1.5, 2.5, 300)
    X, Y = np.meshgrid(x, y)

    dist_sq = (X - mu1)**2 + (Y - mu2)**2
    density = np.exp(-0.5 * dist_sq / (sigma**2)) / (2 * np.pi * sigma**2)

    # Gaussian concentric circular contours in blue
    radii = np.array([0.28, 0.52, 0.78, 1.04])
    levels = np.sort(np.exp(-0.5 * (radii / sigma)**2) / (2 * np.pi * sigma**2))
    ax.contour(X, Y, density, levels=levels, colors='#1E56A0', linewidths=1.3, zorder=2)

    # Unit circle in red
    circle_theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(circle_theta), np.sin(circle_theta), color='#E02020', linewidth=1.8, zorder=3)

    # Axes with arrowheads
    ax.annotate('', xy=(2.3, 0), xytext=(-1.3, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2, mutation_scale=12), zorder=1)
    ax.annotate('', xy=(0, 2.3), xytext=(0, -1.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2, mutation_scale=12), zorder=1)
    ax.text(2.35, -0.05, r'$x_1$', fontsize=13, va='top', ha='left')
    ax.text(-0.06, 2.35, r'$x_2$', fontsize=13, va='bottom', ha='right')

    # Annotations matching Bishop page 91
    ax.text(-0.35, -0.95, r'$r = 1$', fontsize=13, color='black')
    ax.text(mu1 + 0.65, mu2 + 0.45, r'$p(\mathbf{x})$', fontsize=13, color='black')

    ax.set_xlim(-1.45, 2.45)
    ax.set_ylim(-1.45, 2.45)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.10 saved successfully to: {p}")
    if show:
        plt.show()
    return fig, ax


def plot_figure_3_11(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, Any]:
    """
    Faithfully reproduce Figure 3.11 from Bishop & Bishop (2024), page 92:
    The von Mises distribution plotted for two different parameter values:
    Left: Cartesian plot. Right: Polar plot.
    m = 5, theta0 = pi/4 (red)
    m = 1, theta0 = 3*pi/4 (blue)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.0), dpi=300)

    m1, th1 = 5.0, np.pi / 4.0
    m2, th2 = 1.0, 3.0 * np.pi / 4.0

    theta = np.linspace(0, 2 * np.pi, 600)
    vm1 = VonMisesDistribution(theta_0=th1, m=m1)
    vm2 = VonMisesDistribution(theta_0=th2, m=m2)
    pdf1 = vm1.pdf(theta)
    pdf2 = vm2.pdf(theta)

    # 1. Cartesian plot (left)
    ax1.plot(theta, pdf1, color='#E02020', linewidth=2.0, label=r'$m=5, \theta_0 = \pi/4$')
    ax1.plot(theta, pdf2, color='#1E56A0', linewidth=2.0, label=r'$m=1, \theta_0 = 3\pi/4$')

    ax1.set_xlim(0, 2 * np.pi)
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks([0, 2 * np.pi])
    ax1.set_xticklabels([r'$0$', r'$2\pi$'], fontsize=12)
    ax1.set_yticks([])
    ax1.legend(loc='upper right', frameon=False, fontsize=11)
    for spine in ax1.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)

    # 2. Polar plot (right) plotted as 2D parametric curve matching Bishop page 92
    x1_pol = pdf1 * np.cos(theta)
    y1_pol = pdf1 * np.sin(theta)
    x2_pol = pdf2 * np.cos(theta)
    y2_pol = pdf2 * np.sin(theta)

    ax2.plot(x1_pol, y1_pol, color='#E02020', linewidth=2.0, label=r'$m=5, \theta_0 = \pi/4$')
    ax2.plot(x2_pol, y2_pol, color='#1E56A0', linewidth=2.0, label=r'$m=1, \theta_0 = 3\pi/4$')

    # Reference rays
    ax2.plot([0, 0.95], [0, 0], color='black', linewidth=1.2)
    ax2.text(0.97, 0.05, r'$0$', fontsize=11)
    ax2.text(0.97, -0.09, r'$2\pi$', fontsize=11)

    r_ray1 = 0.98
    ax2.plot([0, r_ray1 * np.cos(np.pi / 4)], [0, r_ray1 * np.sin(np.pi / 4)], color='black', linewidth=1.2)
    ax2.text(r_ray1 * np.cos(np.pi / 4) + 0.02, r_ray1 * np.sin(np.pi / 4) + 0.02, r'$\pi/4$', fontsize=11)

    r_ray2 = 0.65
    ax2.plot([0, r_ray2 * np.cos(3 * np.pi / 4)], [0, r_ray2 * np.sin(3 * np.pi / 4)], color='black', linewidth=1.2)
    ax2.text(r_ray2 * np.cos(3 * np.pi / 4) - 0.12, r_ray2 * np.sin(3 * np.pi / 4) + 0.04, r'$3\pi/4$', fontsize=11)

    ax2.set_xlim(-0.65, 1.15)
    ax2.set_ylim(-0.45, 1.05)
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.legend(loc='lower right', frameon=False, fontsize=11)
    for spine in ax2.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.11 saved successfully to: {p}")
    if show:
        plt.show()
    return fig, (ax1, ax2)


def plot_figure_3_12(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, Any]:
    """
    Faithfully reproduce Figure 3.12 from Bishop & Bishop (2024), page 93:
    Left: Modified Bessel function I0(m) defined by (3.130).
    Right: Function A(m) = I1(m) / I0(m) defined by (3.136).
    Both curves shown in red matching the textbook.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=300)

    m = np.linspace(0, 10, 300)
    I0 = special.i0(m)
    A = VonMisesDistribution.bessel_ratio_A(m)

    # (a) I0(m)
    ax1.plot(m, I0, color='#E02020', linewidth=2.0)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 3000)
    ax1.set_xticks([0, 5, 10])
    ax1.set_yticks([0, 1000, 2000, 3000])
    ax1.set_xlabel(r'$m$', fontsize=12)
    ax1.set_ylabel(r'$I_0(m)$', fontsize=12)
    ax1.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for spine in ax1.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)

    # (b) A(m) = I1(m) / I0(m)
    ax2.plot(m, A, color='#E02020', linewidth=2.0)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 1.0)
    ax2.set_xticks([0, 5, 10])
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set_xlabel(r'$m$', fontsize=12)
    ax2.set_ylabel(r'$A(m)$', fontsize=12)
    ax2.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for spine in ax2.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.12 saved successfully to: {p}")
    if show:
        plt.show()
    return fig, (ax1, ax2)




# =====================================================================
# Chapter 3, Section 3.4: The Exponential Family
# =====================================================================

class ExponentialFamilyBase:
    """
    Abstract base class for distributions in the Exponential Family:
        p(x | eta) = h(x) * g(eta) * exp(eta^T u(x))
                   = h(x) * exp(eta^T u(x) - A(eta))
    where:
        - eta: natural parameter vector
        - u(x): sufficient statistics vector
        - h(x): base measure
        - g(eta): normalizer coefficient
        - A(eta) = -ln g(eta): log partition function (cumulant generating function)
    """
    def log_partition(self, eta: np.ndarray) -> float:
        """Compute log-partition function A(eta) = -ln g(eta)."""
        raise NotImplementedError

    def grad_log_partition(self, eta: np.ndarray) -> np.ndarray:
        """Gradient nabla_eta A(eta) = E[u(x)]."""
        raise NotImplementedError

    def hessian_log_partition(self, eta: np.ndarray) -> np.ndarray:
        """Hessian nabla^2_eta A(eta) = Cov[u(x)]."""
        raise NotImplementedError

    def sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        """Compute sufficient statistics u(x)."""
        raise NotImplementedError

    def base_measure(self, x: np.ndarray) -> np.ndarray:
        """Compute base measure h(x)."""
        raise NotImplementedError


class BernoulliExponential(ExponentialFamilyBase):
    """
    Bernoulli distribution as an exponential family member (Eq 3.140 - 3.147):
        p(x | mu) = mu^x (1 - mu)^(1 - x)
                  = sigma(-eta) * exp(eta * x)
    where:
        - eta = ln(mu / (1 - mu)) (logit)
        - mu = sigma(eta) = 1 / (1 + exp(-eta)) (logistic sigmoid)
        - u(x) = x
        - h(x) = 1
        - g(eta) = sigma(-eta) = 1 / (1 + exp(eta))
        - A(eta) = ln(1 + exp(eta)) (softplus)
        - A'(eta) = sigma(eta) = mu = E[x]
        - A''(eta) = sigma(eta) * (1 - sigma(eta)) = Var[x]
    """
    def __init__(self, mu: Optional[float] = None, eta: Optional[float] = None):
        if eta is not None:
            self.eta = float(eta)
            self.mu = 1.0 / (1.0 + np.exp(-self.eta))
        elif mu is not None:
            if not (0.0 < mu < 1.0):
                raise ValueError(f"mu must be in (0, 1), got {mu}")
            self.mu = float(mu)
            self.eta = float(np.log(self.mu / (1.0 - self.mu)))
        else:
            self.mu = 0.5
            self.eta = 0.0

    def log_partition(self, eta: Optional[float] = None) -> float:
        if eta is None:
            eta = self.eta
        return float(np.log1p(np.exp(eta)))

    def grad_log_partition(self, eta: Optional[float] = None) -> float:
        if eta is None:
            eta = self.eta
        return float(1.0 / (1.0 + np.exp(-eta)))

    def hessian_log_partition(self, eta: Optional[float] = None) -> float:
        if eta is None:
            eta = self.eta
        p = self.grad_log_partition(eta)
        return float(p * (1.0 - p))

    def sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    def base_measure(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x, dtype=np.float64)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return self.eta * x - self.log_partition(self.eta)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf(x))

    @classmethod
    def fit_mle(cls, X: np.ndarray) -> 'BernoulliExponential':
        """Maximum likelihood estimation using sufficient statistics."""
        X = np.asarray(X, dtype=np.float64)
        mean_u = float(np.mean(X))
        eps = 1e-12
        mean_u = np.clip(mean_u, eps, 1.0 - eps)
        return cls(mu=mean_u)


class GaussianExponential1D(ExponentialFamilyBase):
    """
    Univariate Gaussian distribution as an exponential family member (Eq 3.162 - 3.167):
        p(x | mu, sigma^2) = (2 pi sigma^2)^(-1/2) * exp(-(x - mu)^2 / (2 sigma^2))
    Natural parameters:
        eta_1 = mu / sigma^2
        eta_2 = -1 / (2 sigma^2)   (eta_2 < 0)
    Standard parameters from natural parameters:
        sigma^2 = -1 / (2 eta_2)
        mu = -eta_1 / (2 eta_2)
    Sufficient statistics:
        u(x) = (x, x^2)^T
    Base measure:
        h(x) = (2 pi)^(-1/2)
    Normalizer:
        g(eta) = (-2 eta_2)^(1/2) * exp(eta_1^2 / (4 eta_2))
    Log-partition function:
        A(eta) = -1/2 * ln(-2 eta_2) - eta_1^2 / (4 eta_2)
    Gradient:
        nabla A(eta) = [-eta_1 / (2 eta_2), eta_1^2 / (4 eta_2^2) - 1 / (2 eta_2)]^T
                     = [mu, mu^2 + sigma^2]^T = E[u(x)]
    Hessian:
        nabla^2 A(eta) = [[sigma^2, 2 mu sigma^2], [2 mu sigma^2, 4 mu^2 sigma^2 + 2 sigma^4]]
                       = Cov[u(x)]
    """
    def __init__(self, mu: Optional[float] = None, sigma2: Optional[float] = None,
                 eta: Optional[Union[np.ndarray, List[float]]] = None):
        if eta is not None:
            eta = np.asarray(eta, dtype=np.float64)
            if eta.shape != (2,):
                raise ValueError(f"eta must have shape (2,), got {eta.shape}")
            if eta[1] >= 0:
                raise ValueError(f"eta_2 must be strictly negative, got {eta[1]}")
            self.eta = eta
            self.sigma2 = float(-1.0 / (2.0 * eta[1]))
            self.mu = float(-eta[0] / (2.0 * eta[1]))
        elif mu is not None and sigma2 is not None:
            if sigma2 <= 0:
                raise ValueError(f"sigma2 must be strictly positive, got {sigma2}")
            self.mu = float(mu)
            self.sigma2 = float(sigma2)
            self.eta = np.array([self.mu / self.sigma2, -1.0 / (2.0 * self.sigma2)], dtype=np.float64)
        else:
            self.mu = 0.0
            self.sigma2 = 1.0
            self.eta = np.array([0.0, -0.5], dtype=np.float64)

    def log_partition(self, eta: Optional[np.ndarray] = None) -> float:
        if eta is None:
            eta = self.eta
        if eta[1] >= 0:
            raise ValueError(f"eta_2 must be strictly negative, got {eta[1]}")
        return float(-0.5 * np.log(-2.0 * eta[1]) - (eta[0]**2) / (4.0 * eta[1]))

    def grad_log_partition(self, eta: Optional[np.ndarray] = None) -> np.ndarray:
        if eta is None:
            eta = self.eta
        if eta[1] >= 0:
            raise ValueError(f"eta_2 must be strictly negative, got {eta[1]}")
        e1, e2 = eta[0], eta[1]
        grad1 = -e1 / (2.0 * e2)
        grad2 = (e1**2) / (4.0 * e2**2) - 1.0 / (2.0 * e2)
        return np.array([grad1, grad2], dtype=np.float64)

    def hessian_log_partition(self, eta: Optional[np.ndarray] = None) -> np.ndarray:
        if eta is None:
            eta = self.eta
        if eta[1] >= 0:
            raise ValueError(f"eta_2 must be strictly negative, got {eta[1]}")
        e1, e2 = eta[0], eta[1]
        H11 = -1.0 / (2.0 * e2)
        H12 = e1 / (2.0 * e2**2)
        H22 = -(e1**2) / (2.0 * e2**3) + 1.0 / (2.0 * e2**2)
        return np.array([[H11, H12], [H12, H22]], dtype=np.float64)

    def sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0:
            return np.array([float(x), float(x**2)])
        return np.column_stack([x, x**2])

    def base_measure(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        val = 1.0 / np.sqrt(2.0 * np.pi)
        if x.ndim == 0:
            return val
        return np.full_like(x, val)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        u = self.sufficient_statistics(x)
        h = self.base_measure(x)
        if x.ndim == 0:
            eta_dot_u = np.dot(self.eta, u)
        else:
            eta_dot_u = np.dot(u, self.eta)
        return np.log(h) + eta_dot_u - self.log_partition(self.eta)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf(x))

    @classmethod
    def fit_mle(cls, X: np.ndarray) -> 'GaussianExponential1D':
        X = np.asarray(X, dtype=np.float64).ravel()
        if len(X) < 2:
            raise ValueError("Need at least 2 samples to fit Gaussian")
        mean_u1 = float(np.mean(X))
        mean_u2 = float(np.mean(X**2))
        mu_mle = mean_u1
        sigma2_mle = mean_u2 - mean_u1**2
        if sigma2_mle <= 1e-12:
            sigma2_mle = 1e-12
        return cls(mu=mu_mle, sigma2=sigma2_mle)


class MultinomialExponential(ExponentialFamilyBase):
    """
    Multinomial / Categorical distribution in non-redundant canonical form (Eq 3.155 - 3.161):
    For M categories, we use M - 1 independent natural parameters:
        eta_k = ln(mu_k / mu_M)  for k = 1, ..., M - 1
    Inverse mapping (Softmax):
        mu_k = exp(eta_k) / (1 + sum_{j=1}^{M-1} exp(eta_j))
        mu_M = 1 / (1 + sum_{j=1}^{M-1} exp(eta_j))
    Sufficient statistics:
        u(x) = (x_1, ..., x_{M-1})^T
    Log-partition function:
        A(eta) = ln(1 + sum_{k=1}^{M-1} exp(eta_k))
    Gradient:
        nabla A(eta) = (mu_1, ..., mu_{M-1})^T = E[u(x)]
    Hessian:
        nabla^2 A(eta) = diag(mu_{1:M-1}) - mu_{1:M-1} mu_{1:M-1}^T = Cov[u(x)]
    """
    def __init__(self, mu: Optional[np.ndarray] = None, eta: Optional[np.ndarray] = None, M: Optional[int] = None):
        if eta is not None:
            self.eta = np.asarray(eta, dtype=np.float64)
            self.M = len(self.eta) + 1
            exp_eta = np.exp(self.eta)
            denom = 1.0 + np.sum(exp_eta)
            mu_first = exp_eta / denom
            mu_last = 1.0 / denom
            self.mu = np.append(mu_first, mu_last)
        elif mu is not None:
            self.mu = np.asarray(mu, dtype=np.float64)
            self.M = len(self.mu)
            if self.M < 2:
                raise ValueError(f"Need at least 2 categories, got {self.M}")
            if not np.isclose(np.sum(self.mu), 1.0):
                self.mu = self.mu / np.sum(self.mu)
            self.eta = np.log(self.mu[:-1] / self.mu[-1])
        elif M is not None:
            self.M = int(M)
            self.mu = np.full(self.M, 1.0 / self.M)
            self.eta = np.zeros(self.M - 1)
        else:
            self.M = 3
            self.mu = np.full(3, 1.0 / 3.0)
            self.eta = np.zeros(2)

    def log_partition(self, eta: Optional[np.ndarray] = None) -> float:
        if eta is None:
            eta = self.eta
        return float(np.log1p(np.sum(np.exp(eta))))

    def grad_log_partition(self, eta: Optional[np.ndarray] = None) -> np.ndarray:
        if eta is None:
            eta = self.eta
        exp_eta = np.exp(eta)
        return exp_eta / (1.0 + np.sum(exp_eta))

    def hessian_log_partition(self, eta: Optional[np.ndarray] = None) -> np.ndarray:
        if eta is None:
            eta = self.eta
        grad = self.grad_log_partition(eta)
        return np.diag(grad) - np.outer(grad, grad)

    def sufficient_statistics(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            return x[:-1]
        return x[:, :-1]

    def base_measure(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            return 1.0
        return np.ones(x.shape[0])

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        u = self.sufficient_statistics(x)
        if x.ndim == 1:
            eta_dot_u = np.dot(self.eta, u)
        else:
            eta_dot_u = np.dot(u, self.eta)
        return eta_dot_u - self.log_partition(self.eta)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf(x))

    @classmethod
    def fit_mle(cls, X: np.ndarray) -> 'MultinomialExponential':
        """Fit categorical distribution from one-hot encoded matrix X (N x M)."""
        X = np.asarray(X, dtype=np.float64)
        mean_mu = np.mean(X, axis=0)
        eps = 1e-12
        mean_mu = np.clip(mean_mu, eps, None)
        mean_mu = mean_mu / np.sum(mean_mu)
        return cls(mu=mean_mu)


class VonMisesExponential(ExponentialFamilyBase):
    """
    Von Mises distribution as an exponential family member:
        p(theta | theta_0, m) = 1 / (2 pi I_0(m)) * exp(m cos(theta - theta_0))
                              = 1 / (2 pi I_0(||eta||)) * exp(eta^T u(theta))
    where:
        - eta = [m cos theta_0, m sin theta_0]^T
        - u(theta) = [cos theta, sin theta]^T
        - h(theta) = 1
        - g(eta) = 1 / (2 pi I_0(||eta||))
        - A(eta) = ln(2 pi) + ln(I_0(||eta||))
        - grad A(eta) = A(m) * [cos theta_0, sin theta_0]^T = E[u(theta)]
    """
    def __init__(self, theta_0: Optional[float] = None, m: Optional[float] = None,
                 eta: Optional[Union[np.ndarray, List[float]]] = None):
        if eta is not None:
            self.eta = np.asarray(eta, dtype=np.float64)
            if self.eta.shape != (2,):
                raise ValueError(f"eta must have shape (2,), got {self.eta.shape}")
            self.m = float(np.linalg.norm(self.eta))
            self.theta_0 = float(np.arctan2(self.eta[1], self.eta[0]) % (2.0 * np.pi))
        elif theta_0 is not None and m is not None:
            if m < 0:
                raise ValueError(f"Concentration m must be >= 0, got {m}")
            self.theta_0 = float(theta_0 % (2.0 * np.pi))
            self.m = float(m)
            self.eta = np.array([self.m * np.cos(self.theta_0), self.m * np.sin(self.theta_0)])
        else:
            self.theta_0 = 0.0
            self.m = 1.0
            self.eta = np.array([1.0, 0.0])

    def log_partition(self, eta: Optional[np.ndarray] = None) -> float:
        if eta is None:
            eta = self.eta
        norm_eta = float(np.linalg.norm(eta))
        return float(np.log(2.0 * np.pi) + np.log(special.i0(norm_eta)))

    def grad_log_partition(self, eta: Optional[np.ndarray] = None) -> np.ndarray:
        if eta is None:
            eta = self.eta
        norm_eta = float(np.linalg.norm(eta))
        if norm_eta < 1e-12:
            return np.zeros(2)
        ratio = float(special.i1(norm_eta) / special.i0(norm_eta))
        return ratio * (eta / norm_eta)

    def sufficient_statistics(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 0:
            return np.array([np.cos(theta), np.sin(theta)])
        return np.column_stack([np.cos(theta), np.sin(theta)])

    def base_measure(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 0:
            return 1.0
        return np.ones_like(theta)

    def log_pdf(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        u = self.sufficient_statistics(theta)
        if theta.ndim == 0:
            eta_dot_u = np.dot(self.eta, u)
        else:
            eta_dot_u = np.dot(u, self.eta)
        return eta_dot_u - self.log_partition(self.eta)

    def pdf(self, theta: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf(theta))


def plot_figure_3_13_exp_family_geometry(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, Any]:
    """Figure 3.13: Geometry of the Exponential Family."""
    from common.plot_utils import setup_style
    setup_style()
    eta = np.linspace(-5.0, 5.0, 300)
    A = np.log1p(np.exp(eta))
    mu = 1.0 / (1.0 + np.exp(-eta))
    var = mu * (1.0 - mu)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    
    # 1. Log-partition function A(eta)
    ax1 = axes[0]
    ax1.plot(eta, A, color='#1E56A0', linewidth=2.5, label=r"$A(\eta) = \ln(1 + e^{\eta})$")
    for et0, col in zip([-2.0, 0.0, 2.0], ['#E02020', '#2E8B57', '#9370DB']):
        A0 = np.log1p(np.exp(et0))
        slope0 = 1.0 / (1.0 + np.exp(-et0))
        tangent = A0 + slope0 * (eta - et0)
        ax1.plot(eta, tangent, linestyle='--', color=col, alpha=0.8,
                 label=rf"Tangent at $\eta={et0:+.1f}$")
        ax1.scatter([et0], [A0], color=col, s=40, zorder=5)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-0.5, 5.5)
    ax1.set_xlabel(r"Natural Parameter $\eta$", fontsize=11)
    ax1.set_ylabel(r"Log-partition $A(\eta) = -\ln g(\eta)$", fontsize=11)
    ax1.set_title(r"(a) Strictly Convex $A(\eta)$ & Supporting Tangents", fontsize=11)
    ax1.legend(loc='upper left', fontsize=9, frameon=True)
    ax1.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for s in ax1.spines.values():
        s.set_linewidth(0.8)

    # 2. Gradient mapping nabla A(eta) = E[u]
    ax2 = axes[1]
    ax2.plot(eta, mu, color='#E02020', linewidth=2.5, label=r"$\nabla A(\eta) = \sigma(\eta) = \mathbb{E}[x]$")
    ax2.axhline(0.0, color='gray', linestyle=':', alpha=0.6)
    ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax2.axvline(0.0, color='gray', linestyle=':', alpha=0.6)
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel(r"Natural Parameter $\eta$", fontsize=11)
    ax2.set_ylabel(r"Expectation Parameter $\mu = \mathbb{E}[u(x)]$", fontsize=11)
    ax2.set_title(r"(b) Dual Mapping $\nabla A(\eta): \mathcal{H} \to \mathcal{M}$", fontsize=11)
    ax2.legend(loc='lower right', fontsize=10, frameon=True)
    ax2.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for s in ax2.spines.values():
        s.set_linewidth(0.8)

    # 3. Hessian nabla^2 A(eta) = Var[u]
    ax3 = axes[2]
    ax3.plot(eta, var, color='#2E8B57', linewidth=2.5, label=r"$\nabla^2 A(\eta) = \mu(1-\mu) = \mathrm{Var}[x]$")
    ax3.fill_between(eta, var, color='#2E8B57', alpha=0.2)
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(-0.02, 0.3)
    ax3.set_xlabel(r"Natural Parameter $\eta$", fontsize=11)
    ax3.set_ylabel(r"Curvature / Variance $\nabla^2 A(\eta)$", fontsize=11)
    ax3.set_title(r"(c) Curvature as Variance / Fisher Information", fontsize=11)
    ax3.legend(loc='upper right', fontsize=10, frameon=True)
    ax3.tick_params(axis='both', which='major', direction='in', top=True, right=True)
    for s in ax3.spines.values():
        s.set_linewidth(0.8)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.13 saved successfully to: {p}")
    if show:
        plt.show()
    return fig, axes


def plot_figure_3_14_exp_family_members(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, Any]:
    """Figure 3.14: Major Exponential Family Members and their Canonical Mappings."""
    from common.plot_utils import setup_style
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    
    # (a) Bernoulli
    ax_a = axes[0, 0]
    mu_vals = np.linspace(0.01, 0.99, 200)
    eta_vals = np.log(mu_vals / (1.0 - mu_vals))
    ax_a.plot(mu_vals, eta_vals, color='#1E56A0', linewidth=2.2)
    ax_a.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax_a.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax_a.set_xlabel(r"Mean Parameter $\mu \in (0, 1)$", fontsize=11)
    ax_a.set_ylabel(r"Natural Parameter $\eta = \ln(\mu / (1-\mu))$", fontsize=11)
    ax_a.set_title(r"(a) Bernoulli: Logit Link ($\mathbf{u}(x)=x$)", fontsize=12)
    ax_a.tick_params(direction='in', top=True, right=True)

    # (b) Multinomial (3-state simplex to natural coordinates)
    ax_b = axes[0, 1]
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, np.sqrt(3)/2])
    tri = np.array([v1, v2, v3, v1])
    ax_b.plot(tri[:, 0], tri[:, 1], 'k-', linewidth=1.5)
    
    grid_pts = []
    colors = []
    for p1 in np.linspace(0.05, 0.9, 18):
        for p2 in np.linspace(0.05, 0.9 - p1, 18):
            p3 = 1.0 - p1 - p2
            if p3 > 0.02:
                xy = p1 * v1 + p2 * v2 + p3 * v3
                grid_pts.append(xy)
                colors.append(np.log(p1 / p3))
    grid_pts = np.array(grid_pts)
    sc = ax_b.scatter(grid_pts[:, 0], grid_pts[:, 1], c=colors, cmap='coolwarm', s=25, alpha=0.85)
    plt.colorbar(sc, ax=ax_b, label=r"$\eta_1 = \ln(\mu_1 / \mu_3)$", pad=0.02)
    ax_b.text(v1[0] - 0.06, v1[1] - 0.04, r"$\mu_1=1$", fontsize=10, color='#1E56A0', fontweight='bold')
    ax_b.text(v2[0] + 0.02, v2[1] - 0.04, r"$\mu_2=1$", fontsize=10, color='#E02020', fontweight='bold')
    ax_b.text(v3[0] - 0.05, v3[1] + 0.04, r"$\mu_3=1$", fontsize=10, color='#2E8B57', fontweight='bold')
    ax_b.set_title(r"(b) Multinomial: Probability Simplex & Softmax", fontsize=12)
    ax_b.set_aspect('equal')
    ax_b.axis('off')

    # (c) Univariate Gaussian Natural Parameter space (eta1, eta2) with eta2 < 0
    ax_c = axes[1, 0]
    eta1_grid = np.linspace(-3.0, 3.0, 150)
    eta2_grid = np.linspace(-3.0, -0.1, 150)
    E1, E2 = np.meshgrid(eta1_grid, eta2_grid)
    MU = -E1 / (2.0 * E2)
    cs = ax_c.contour(E1, E2, MU, levels=np.linspace(-2, 2, 9), cmap='coolwarm', linewidths=1.2)
    ax_c.clabel(cs, inline=True, fontsize=8, fmt=r"$\mu=%.1f$")
    ax_c.axhline(0, color='black', linewidth=1.5)
    ax_c.fill_between(eta1_grid, 0, 0.5, color='gray', alpha=0.3)
    ax_c.text(0, 0.15, r"Invalid domain ($\eta_2 \geq 0$)", ha='center', fontsize=10, color='darkred')
    ax_c.set_xlim(-3, 3)
    ax_c.set_ylim(-3, 0.5)
    ax_c.set_xlabel(r"Natural Parameter $\eta_1 = \mu / \sigma^2$", fontsize=11)
    ax_c.set_ylabel(r"Natural Parameter $\eta_2 = -1 / (2\sigma^2)$", fontsize=11)
    ax_c.set_title(r"(c) Gaussian: $\mathbf{u}(x) = (x, x^2)^T$ Domain ($\eta_2 < 0$)", fontsize=12)
    ax_c.tick_params(direction='in', top=True, right=True)

    # (d) Von Mises Natural Parameter space eta = (m cos theta0, m sin theta0)
    ax_d = axes[1, 1]
    radii = [1.0, 2.5, 4.0]
    circle_theta = np.linspace(0, 2*np.pi, 200)
    for r in radii:
        ax_d.plot(r * np.cos(circle_theta), r * np.sin(circle_theta), 'k--', alpha=0.4)
        ax_d.text(r * 0.707 + 0.1, r * 0.707 + 0.1, f"$m={r:.1f}$", fontsize=8, color='gray')
    
    test_params = [(np.pi/4, 4.0, '#E02020', r"$\theta_0=\frac{\pi}{4}, m=4$"),
                   (3*np.pi/4, 2.5, '#1E56A0', r"$\theta_0=\frac{3\pi}{4}, m=2.5$"),
                   (-np.pi/3, 3.2, '#2E8B57', r"$\theta_0=-\frac{\pi}{3}, m=3.2$")]
    for th0, m_val, col, lbl in test_params:
        e1 = m_val * np.cos(th0)
        e2 = m_val * np.sin(th0)
        ax_d.annotate('', xy=(e1, e2), xytext=(0, 0),
                      arrowprops=dict(facecolor=col, edgecolor=col, width=1.8, headwidth=8))
        ax_d.scatter([e1], [e2], color=col, s=40, zorder=5, label=lbl)
    
    ax_d.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax_d.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax_d.set_xlim(-5, 5)
    ax_d.set_ylim(-5, 5)
    ax_d.set_aspect('equal')
    ax_d.set_xlabel(r"Natural Parameter $\eta_1 = m \cos \theta_0$", fontsize=11)
    ax_d.set_ylabel(r"Natural Parameter $\eta_2 = m \sin \theta_0$", fontsize=11)
    ax_d.set_title(r"(d) Von Mises: $\mathbf{u}(\theta) = (\cos\theta, \sin\theta)^T$", fontsize=12)
    ax_d.legend(loc='lower left', fontsize=9, frameon=True)
    ax_d.tick_params(direction='in', top=True, right=True)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.14 saved successfully to: {p}")
    if show:
        plt.show()
    return fig, axes


def plot_figure_3_15_sufficient_statistics_online(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, Any]:
    """Figure 3.15: Sufficient Statistics and Online / Stream Estimation."""
    from common.plot_utils import setup_style
    setup_style()
    np.random.seed(42)
    true_mu = 3.5
    true_sigma2 = 2.0
    N = 500
    X = np.random.normal(true_mu, np.sqrt(true_sigma2), size=N)
    
    sum_x = np.cumsum(X)
    sum_x2 = np.cumsum(X**2)
    n_arr = np.arange(1, N + 1)
    
    mu_mle_seq = sum_x / n_arr
    sigma2_mle_seq = (sum_x2 / n_arr) - (mu_mle_seq**2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=300)
    
    # (a) Online convergence
    ax1.plot(n_arr, mu_mle_seq, color='#1E56A0', linewidth=1.8, label=r"$\mu^{\mathrm{ML}}_N = \frac{1}{N}\sum x_n$")
    ax1.axhline(true_mu, color='#1E56A0', linestyle='--', alpha=0.7, label=r"True $\mu = 3.5$")
    
    ax1.plot(n_arr, sigma2_mle_seq, color='#E02020', linewidth=1.8, label=r"${\sigma^2}^{\mathrm{ML}}_N = \frac{1}{N}\sum x_n^2 - (\mu^{\mathrm{ML}}_N)^2$")
    ax1.axhline(true_sigma2, color='#E02020', linestyle='--', alpha=0.7, label=r"True $\sigma^2 = 2.0$")
    
    ax1.set_xlim(1, N)
    ax1.set_ylim(0, 5.5)
    ax1.set_xlabel(r"Sample Count $N$ (Stream Step)", fontsize=11)
    ax1.set_ylabel("Parameter Estimate", fontsize=11)
    ax1.set_title("(a) Online Stream MLE via Running Sufficient Statistics", fontsize=11)
    ax1.legend(loc='upper right', fontsize=9, frameon=True)
    ax1.tick_params(direction='in', top=True, right=True)
    for s in ax1.spines.values():
        s.set_linewidth(0.8)

    # (b) Memory Footprint: Buffer O(N) vs Sufficient Statistics O(1)
    bytes_per_float = 8
    raw_buffer_bytes = n_arr * bytes_per_float
    suff_stat_bytes = np.full_like(n_arr, 24)
    
    ax2.plot(n_arr, raw_buffer_bytes / 1024.0, color='#E02020', linewidth=2.0,
             label=r"Raw Data Buffering: $\mathcal{O}(N)$ Storage")
    ax2.plot(n_arr, suff_stat_bytes / 1024.0, color='#2E8B57', linewidth=2.2,
             label=r"Sufficient Statistics $\sum \mathbf{u}(x_n)$: $\mathcal{O}(1)$ Storage")
    ax2.fill_between(n_arr, suff_stat_bytes / 1024.0, raw_buffer_bytes / 1024.0,
                     color='#E02020', alpha=0.1)
    ax2.set_xlim(1, N)
    ax2.set_xlabel(r"Sample Count $N$", fontsize=11)
    ax2.set_ylabel("Memory Footprint (KB)", fontsize=11)
    ax2.set_title(r"(b) Memory Efficiency of Sufficient Statistics", fontsize=11)
    ax2.legend(loc='upper left', fontsize=10, frameon=True)
    ax2.tick_params(direction='in', top=True, right=True)
    for s in ax2.spines.values():
        s.set_linewidth(0.8)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.15 saved successfully to: {p}")
    if show:
        plt.show()
    return fig, (ax1, ax2)


# =====================================================================
# 3.5 Nonparametric Methods
# =====================================================================

class HistogramDensity1D:
    """
    1D Histogram Density Estimator (Section 3.5.1, Eq 3.175):
        p_i = n_i / (N * Delta_i)
    where:
        n_i: number of observations in bin i
        N: total number of observations
        Delta_i: width of bin i
    """
    def __init__(
        self,
        bin_width: Optional[float] = None,
        bin_edges: Optional[Union[List[float], np.ndarray]] = None,
        range_bounds: Tuple[float, float] = (0.0, 1.0)
    ):
        if bin_edges is not None:
            self.bin_edges = np.asarray(bin_edges, dtype=np.float64)
            self.bin_widths = np.diff(self.bin_edges)
        elif bin_width is not None:
            low, high = range_bounds
            self.bin_edges = np.arange(low, high + bin_width * 0.5, bin_width)
            self.bin_widths = np.diff(self.bin_edges)
        else:
            raise ValueError("Must specify either bin_width or bin_edges")

        self.n_bins = len(self.bin_widths)
        self.counts = np.zeros(self.n_bins, dtype=int)
        self.density = np.zeros(self.n_bins, dtype=np.float64)
        self.total_samples = 0

    def fit(self, X: Union[List[float], np.ndarray]) -> "HistogramDensity1D":
        """Fit histogram density on 1D observations X."""
        X_arr = np.asarray(X, dtype=np.float64).ravel()
        self.total_samples = len(X_arr)
        if self.total_samples == 0:
            raise ValueError("Cannot fit on empty data")

        self.counts, _ = np.histogram(X_arr, bins=self.bin_edges)
        self.density = self.counts / (self.total_samples * self.bin_widths)
        return self

    def evaluate(self, x: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate piecewise-constant probability density at points x."""
        x_arr = np.asarray(x, dtype=np.float64)
        is_scalar = (x_arr.ndim == 0)
        x_flat = np.atleast_1d(x_arr)
        
        # Identify bin index for each query point
        # np.digitize returns 1-indexed bin
        bin_indices = np.digitize(x_flat, self.bin_edges) - 1
        
        res = np.zeros_like(x_flat, dtype=np.float64)
        valid = (bin_indices >= 0) & (bin_indices < self.n_bins)
        res[valid] = self.density[bin_indices[valid]]

        # Edge case: right boundary point
        right_boundary = (x_flat == self.bin_edges[-1])
        if np.any(right_boundary) and self.n_bins > 0:
            res[right_boundary] = self.density[-1]

        if is_scalar:
            return float(res[0])
        return res

    def sample(self, size: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """Sample from the piecewise constant histogram density model."""
        rng = np.random.RandomState(seed)
        # Select bin according to discrete probabilities p_i * Delta_i = n_i / N
        probs = self.counts / float(self.total_samples)
        chosen_bins = rng.choice(self.n_bins, size=size, p=probs)
        # Uniform within each chosen bin
        lows = self.bin_edges[chosen_bins]
        highs = self.bin_edges[chosen_bins + 1]
        return rng.uniform(lows, highs)

    def pdf(self, x: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """Alias for evaluate(x)."""
        return self.evaluate(x)

    def score_samples(self, x: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """Compute log-density log p(x)."""
        p = np.asarray(self.evaluate(x), dtype=np.float64)
        return np.log(np.maximum(p, 1e-300))


class KernelDensity1D:
    """
    1D Kernel Density Estimator / Parzen Window (Section 3.5.2, Eq 3.183 - 3.184):
        p(x) = (1 / N) sum_{n=1}^N (1 / h) k((x - x_n) / h)
    Supports Gaussian, Boxcar (Tophat), and Epanechnikov kernels.
    """
    def __init__(self, h: float = 0.05, kernel: str = "gaussian"):
        if h <= 0:
            raise ValueError(f"Bandwidth h must be positive, got {h}")
        self.h = float(h)
        self.kernel = kernel.lower()
        if self.kernel not in ["gaussian", "boxcar", "tophat", "epanechnikov"]:
            raise ValueError(f"Unsupported kernel '{kernel}'. Choose from 'gaussian', 'boxcar', 'epanechnikov'.")
        self.X: Optional[np.ndarray] = None
        self.N: int = 0

    def fit(self, X: Union[List[float], np.ndarray]) -> "KernelDensity1D":
        """Store training sample observations."""
        self.X = np.asarray(X, dtype=np.float64).ravel()
        self.N = len(self.X)
        if self.N == 0:
            raise ValueError("Cannot fit KDE on empty data")
        return self

    def evaluate(self, x: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate kernel density estimate at query points x."""
        if self.X is None:
            raise ValueError("KernelDensity1D must be fitted before evaluation")

        x_arr = np.asarray(x, dtype=np.float64)
        is_scalar = (x_arr.ndim == 0)
        x_flat = np.atleast_1d(x_arr)

        # Distance matrix (M, N) normalized by bandwidth h
        u = (x_flat[:, None] - self.X[None, :]) / self.h

        if self.kernel == "gaussian":
            # k(u) = (1 / sqrt(2*pi)) * exp(-0.5 * u^2) (Eq 3.184)
            k_val = np.exp(-0.5 * u**2) / np.sqrt(2.0 * np.pi)
        elif self.kernel in ["boxcar", "tophat"]:
            # k(u) = 1 if |u| <= 0.5, else 0 (Eq 3.181)
            k_val = np.where(np.abs(u) <= 0.5, 1.0, 0.0)
        elif self.kernel == "epanechnikov":
            # k(u) = 0.75 * (1 - u^2) for |u| <= 1, else 0
            k_val = np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u**2), 0.0)

        dens = np.sum(k_val, axis=1) / (self.N * self.h)
        if is_scalar:
            return float(dens[0])
        return dens

    @staticmethod
    def silverman_bandwidth(X: Union[List[float], np.ndarray]) -> float:
        """Silverman's rule-of-thumb bandwidth for Gaussian KDE: h = 1.06 * std * N^(-1/5)."""
        X_arr = np.asarray(X, dtype=np.float64).ravel()
        n = len(X_arr)
        if n < 2:
            return 1.0
        std = np.std(X_arr, ddof=1)
        return float(1.06 * std * (n ** (-0.2)))

    def pdf(self, x: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """Alias for evaluate(x)."""
        return self.evaluate(x)

    def score_samples(self, x: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """Compute log-density log p(x)."""
        p = np.asarray(self.evaluate(x), dtype=np.float64)
        return np.log(np.maximum(p, 1e-300))

    def sample(self, size: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """Sample from the kernel density model."""
        if self.X is None:
            raise ValueError("Model must be fitted before sampling.")
        rng = np.random.default_rng(seed)
        chosen_indices = rng.integers(0, self.N, size=size)
        centers = self.X[chosen_indices]
        if self.kernel == "gaussian":
            noise = rng.normal(0, self.h, size=size)
        elif self.kernel in ["boxcar", "tophat"]:
            noise = rng.uniform(-0.5 * self.h, 0.5 * self.h, size=size)
        else:
            noise = rng.normal(0, self.h, size=size)
        return centers + noise


class KernelDensityND:
    """
    Multidimensional Kernel Density Estimator (Section 3.5.2, Eq 3.184):
        p(x) = (1 / N) sum_{n=1}^N (1 / (2*pi*h^2)^(D/2)) exp(- ||x - x_n||^2 / (2*h^2))
    """
    def __init__(self, h: float = 0.1):
        if h <= 0:
            raise ValueError(f"Bandwidth h must be positive, got {h}")
        self.h = float(h)
        self.X: Optional[np.ndarray] = None
        self.N: int = 0
        self.D: int = 0

    def fit(self, X: np.ndarray) -> "KernelDensityND":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        self.X = X_arr
        self.N, self.D = self.X.shape
        return self

    def evaluate(self, X_query: np.ndarray) -> np.ndarray:
        if self.X is None:
            raise ValueError("Model must be fitted before evaluation")
        Q = np.asarray(X_query, dtype=np.float64)
        if Q.ndim == 1:
            Q = Q[None, :]

        # Squared Euclidean distances: ||x - x_n||^2
        # (M, 1, D) - (1, N, D) -> (M, N)
        diff = Q[:, None, :] - self.X[None, :, :]
        sq_dist = np.sum(diff**2, axis=-1)

        norm_const = 1.0 / ((2.0 * np.pi * (self.h**2)) ** (self.D / 2.0))
        k_vals = norm_const * np.exp(-0.5 * sq_dist / (self.h**2))
        return np.mean(k_vals, axis=1)


class KNNDensityEstimator:
    """
    K-Nearest-Neighbour Density Estimator (Section 3.5.3, Eq 3.180):
        p(x) = K / (N * V(x))
    where V(x) is the volume of a hypersphere of radius r_K(x) (distance to K-th neighbor):
        V_D(r) = (pi^(D/2) / Gamma(D/2 + 1)) * r^D
    """
    def __init__(self, K: int = 5):
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        self.K = int(K)
        self.X: Optional[np.ndarray] = None
        self.N: int = 0
        self.D: int = 0

    def fit(self, X: Union[List[float], np.ndarray]) -> "KNNDensityEstimator":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        self.X = X_arr
        self.N, self.D = self.X.shape
        if self.K > self.N:
            raise ValueError(f"K={self.K} cannot exceed sample size N={self.N}")
        return self

    def _sphere_volume(self, r: np.ndarray) -> np.ndarray:
        """Volume of D-dimensional sphere of radius r."""
        c = (np.pi ** (self.D / 2.0)) / special.gamma(self.D / 2.0 + 1.0)
        return c * (r ** self.D)

    def evaluate(self, x: Union[float, List[float], np.ndarray], eps: float = 1e-9) -> Union[float, np.ndarray]:
        """Evaluate KNN density at query points x."""
        if self.X is None:
            raise ValueError("Model must be fitted before evaluation")

        x_arr = np.asarray(x, dtype=np.float64)
        is_scalar = (x_arr.ndim == 0) or (x_arr.ndim == 1 and self.D == 1 and len(x_arr) == 1)
        if x_arr.ndim == 1 and self.D > 1:
            x_arr = x_arr[None, :]
        elif x_arr.ndim == 1 and self.D == 1:
            x_arr = x_arr[:, None]
        elif x_arr.ndim == 0:
            x_arr = np.array([[x_arr]])

        # Pairwise distance matrix (M, N)
        diff = x_arr[:, None, :] - self.X[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))

        # Distance to K-th nearest neighbor (0-indexed: K - 1)
        sorted_dists = np.sort(dists, axis=1)
        r_K = sorted_dists[:, self.K - 1]

        # Regularize near-zero distances
        r_K_safe = np.maximum(r_K, eps)
        vol = self._sphere_volume(r_K_safe)
        dens = self.K / (self.N * vol)

        if is_scalar:
            return float(dens[0])
        return dens

    def pdf(self, x: Union[float, List[float], np.ndarray], eps: float = 1e-9) -> Union[float, np.ndarray]:
        """Alias for evaluate(x)."""
        return self.evaluate(x, eps=eps)

    def score_samples(self, x: Union[float, List[float], np.ndarray], eps: float = 1e-9) -> Union[float, np.ndarray]:
        """Compute log-density log p(x)."""
        p = np.asarray(self.evaluate(x, eps=eps), dtype=np.float64)
        return np.log(np.maximum(p, 1e-300))


class KNNClassifier:
    """
    K-Nearest-Neighbour Classifier (Section 3.5.3, Eq 3.187 - 3.190):
        p(C_k | x) = K_k / K
    Classifies a query point to the majority class amongst its K nearest neighbours.
    """
    def __init__(self, K: int = 3):
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        self.K = int(K)
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.N: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Store training points and class labels."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=int).ravel()
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if len(X_arr) != len(y_arr):
            raise ValueError("X and y must have same length")

        self.X = X_arr
        self.y = y_arr
        self.classes_ = np.unique(y_arr)
        self.N = len(X_arr)
        if self.K > self.N:
            raise ValueError(f"K={self.K} cannot exceed dataset size N={self.N}")
        return self

    def predict_proba(self, X_query: np.ndarray) -> np.ndarray:
        """
        Compute posterior probabilities p(C_k | x) = K_k / K (Eq 3.190).
        Returns array of shape (len(X_query), n_classes).
        """
        if self.X is None:
            raise ValueError("Model must be fitted before prediction")
        Q = np.asarray(X_query, dtype=np.float64)
        if Q.ndim == 1:
            Q = Q[None, :]

        # Distance matrix (M, N)
        diff = Q[:, None, :] - self.X[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))

        # Find indices of K nearest neighbors
        knn_indices = np.argsort(dists, axis=1)[:, :self.K]

        # Count occurrences of each class in neighbors
        n_queries = len(Q)
        n_classes = len(self.classes_)
        proba = np.zeros((n_queries, n_classes), dtype=np.float64)

        for i in range(n_queries):
            neighbor_labels = self.y[knn_indices[i]]
            for c_idx, cls in enumerate(self.classes_):
                proba[i, c_idx] = np.sum(neighbor_labels == cls) / float(self.K)

        return proba

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        """Predict class label by majority vote (maximum posterior probability)."""
        proba = self.predict_proba(X_query)
        best_indices = np.argmax(proba, axis=1)
        return self.classes_[best_indices]

    def get_k_nearest_indices(self, x_query: np.ndarray) -> np.ndarray:
        """Return indices of K nearest training points to a single query vector."""
        q = np.asarray(x_query, dtype=np.float64).ravel()
        dists = np.sqrt(np.sum((self.X - q)**2, axis=1))
        return np.argsort(dists)[:self.K]


# =====================================================================
# Figure Plotting Functions for Section 3.5
# =====================================================================

def get_mixture_pdf_3_5(x: np.ndarray) -> np.ndarray:
    """Mixture of two Gaussians on [0, 1] used in Figures 3.13, 3.14, 3.15."""
    pi1, mu1, s1 = 0.3, 0.3, 0.09
    pi2, mu2, s2 = 0.7, 0.75, 0.085
    g1 = (1.0 / (np.sqrt(2 * np.pi) * s1)) * np.exp(-0.5 * ((x - mu1) / s1) ** 2)
    g2 = (1.0 / (np.sqrt(2 * np.pi) * s2)) * np.exp(-0.5 * ((x - mu2) / s2) ** 2)
    return pi1 * g1 + pi2 * g2


def get_synthetic_50_points_3_5(seed: int = 26) -> np.ndarray:
    """Generate the exact 50 synthetic data points used in Figures 3.13, 3.14, 3.15."""
    rng = np.random.RandomState(seed)
    n1 = rng.binomial(50, 0.3)
    n2 = 50 - n1
    s1 = rng.normal(0.3, 0.09, size=n1)
    s2 = rng.normal(0.75, 0.085, size=n2)
    return np.sort(np.clip(np.concatenate([s1, s2]), 0.01, 0.99))


def plot_figure_3_13_histogram(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Faithfully reproduce Figure 3.13 from Bishop & Bishop (2024), page 99:
    Histogram density estimates for Delta = 0.04, 0.08, 0.25 on 50 data points.
    """
    data = get_synthetic_50_points_3_5()
    deltas = [0.04, 0.08, 0.25]
    x_grid = np.linspace(0, 1, 500)
    true_pdf = get_mixture_pdf_3_5(x_grid)

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 5.5), dpi=300, sharex=True)

    for ax, delta in zip(axes, deltas):
        hist_model = HistogramDensity1D(bin_width=delta, range_bounds=(0.0, 1.0)).fit(data)
        bins = hist_model.bin_edges
        counts = hist_model.counts
        density = hist_model.density

        for i in range(len(counts)):
            b_left = bins[i]
            h = density[i]
            ax.bar(b_left, h, width=delta, align='edge',
                   facecolor='#4D72B8', edgecolor='black', linewidth=0.8, alpha=0.9, zorder=2)

        ax.plot(x_grid, true_pdf, color='#00C000', linewidth=1.8, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 5.2)
        ax.set_yticks([0, 5])
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(direction='in', top=True, right=True)
        ax.text(0.04, 4.0, rf"$\Delta = {delta}$", fontsize=11, zorder=4)
        for s in ax.spines.values():
            s.set_linewidth(0.8)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.13 saved successfully to: {p}")
    if show:
        plt.show()
    return fig, axes


def plot_figure_3_14_kernel_density(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Faithfully reproduce Figure 3.14 from Bishop & Bishop (2024), page 102:
    Kernel density estimation with Gaussian kernel h = 0.005, 0.07, 0.2.
    """
    data = get_synthetic_50_points_3_5()
    bandwidths = [0.005, 0.07, 0.2]
    x_grid = np.linspace(0, 1, 1000)
    true_pdf = get_mixture_pdf_3_5(x_grid)

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 5.5), dpi=300, sharex=True)

    for ax, h in zip(axes, bandwidths):
        kde = KernelDensity1D(h=h, kernel="gaussian").fit(data)
        dens_est = kde.evaluate(x_grid)

        ax.plot(x_grid, dens_est, color='#0044FF', linewidth=1.5, zorder=2)
        ax.plot(x_grid, true_pdf, color='#00C000', linewidth=1.8, zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 5.2)
        ax.set_yticks([0, 5])
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(direction='in', top=True, right=True)
        ax.text(0.04, 4.0, rf"$h = {h}$", fontsize=11, zorder=4)
        for s in ax.spines.values():
            s.set_linewidth(0.8)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.14 saved successfully to: {p}")
    if show:
        plt.show()
    return fig, axes


def plot_figure_3_15_knn_density(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Faithfully reproduce Figure 3.15 from Bishop & Bishop (2024), page 103:
    K-nearest-neighbour density estimation with K = 1, 5, 30.
    """
    data = get_synthetic_50_points_3_5()
    K_values = [1, 5, 30]
    x_grid = np.linspace(0, 1, 2000)
    true_pdf = get_mixture_pdf_3_5(x_grid)

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 5.5), dpi=300, sharex=True)

    for ax, K in zip(axes, K_values):
        knn = KNNDensityEstimator(K=K).fit(data)
        dens_est = knn.evaluate(x_grid)
        # Cap infinite spikes for visualization
        dens_est_clipped = np.clip(dens_est, 0, 5.15)

        ax.plot(x_grid, dens_est_clipped, color='#0044FF', linewidth=1.5, zorder=2)
        ax.plot(x_grid, true_pdf, color='#00C000', linewidth=1.8, zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 5.2)
        ax.set_yticks([0, 5])
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(direction='in', top=True, right=True)
        ax.text(0.04, 4.0, rf"$K = {K}$", fontsize=11, zorder=4)
        for s in ax.spines.values():
            s.set_linewidth(0.8)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.15 saved successfully to: {p}")
    if show:
        plt.show()
    return fig, axes


def plot_figure_3_16_knn_classification(
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Faithfully reproduce Figure 3.16 from Bishop & Bishop (2024), page 104:
    (a) K-nearest-neighbour classifier (K = 3) with query point.
    (b) Nearest-neighbour (K = 1) decision boundary (Voronoi bisectors).
    """
    red_pts = np.array([
        [0.05, 0.85], [0.10, 0.65], [0.12, 0.35], [0.18, 0.80],
        [0.22, 0.55], [0.20, 0.34], [0.25, 0.70], [0.30, 0.85],
        [0.35, 0.40], [0.40, 0.60]
    ])
    blue_pts = np.array([
        [0.18, 0.26], [0.30, 0.08], [0.36, 0.25], [0.48, 0.06],
        [0.52, 0.25], [0.55, 0.45], [0.60, 0.88], [0.64, 0.15],
        [0.68, 0.70], [0.72, 0.38], [0.78, 0.55], [0.85, 0.15],
        [0.90, 0.65]
    ])
    query_pt = np.array([0.45, 0.63])

    all_pts = np.vstack([red_pts, blue_pts])
    all_labels = np.array([0]*len(red_pts) + [1]*len(blue_pts))

    clf_k3 = KNNClassifier(K=3).fit(all_pts, all_labels)
    knn_idx = clf_k3.get_k_nearest_indices(query_pt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.2), dpi=300)

    # Subplot (a) K=3
    ax1.scatter(red_pts[:, 0], red_pts[:, 1], color='#E02020', s=45, zorder=3, edgecolors='none')
    ax1.scatter(blue_pts[:, 0], blue_pts[:, 1], color='#0044FF', s=45, zorder=3, edgecolors='none')
    for idx in knn_idx:
        nbr = all_pts[idx]
        ax1.plot([query_pt[0], nbr[0]], [query_pt[1], nbr[1]], color='#00C000', linewidth=2.0, zorder=2)
    ax1.scatter(query_pt[0], query_pt[1], marker='D', s=60, facecolor='#00E000',
                edgecolor='black', linewidth=1.5, zorder=4)

    ax1.set_xlim(-0.02, 1.05)
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel(r"$x_1$", fontsize=12, loc='right')
    ax1.set_ylabel(r"$x_2$", fontsize=12, loc='top', rotation=0, labelpad=8)
    ax1.set_title("(a)", fontsize=12, y=-0.15)
    ax1.spines['left'].set_position(('data', 0))
    ax1.spines['bottom'].set_position(('data', 0))
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.plot(1.03, 0, ">k", clip_on=False, markersize=6)
    ax1.plot(0, 1.03, "^k", clip_on=False, markersize=6)

    # Subplot (b) 1-NN Voronoi boundary
    ax2.scatter(red_pts[:, 0], red_pts[:, 1], color='#E02020', s=45, zorder=3, edgecolors='none')
    ax2.scatter(blue_pts[:, 0], blue_pts[:, 1], color='#0044FF', s=45, zorder=3, edgecolors='none')

    gx = np.linspace(-0.02, 1.05, 500)
    gy = np.linspace(-0.02, 1.05, 500)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

    d_red = np.min(np.linalg.norm(grid_pts[:, None, :] - red_pts[None, :, :], axis=2), axis=1)
    d_blue = np.min(np.linalg.norm(grid_pts[:, None, :] - blue_pts[None, :, :], axis=2), axis=1)
    diff_grid = (d_red - d_blue).reshape(GX.shape)

    ax2.contour(GX, GY, diff_grid, levels=[0], colors=['#00C000'], linewidths=[2.2], zorder=2)

    ax2.set_xlim(-0.02, 1.05)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel(r"$x_1$", fontsize=12, loc='right')
    ax2.set_ylabel(r"$x_2$", fontsize=12, loc='top', rotation=0, labelpad=8)
    ax2.set_title("(b)", fontsize=12, y=-0.15)
    ax2.spines['left'].set_position(('data', 0))
    ax2.spines['bottom'].set_position(('data', 0))
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.plot(1.03, 0, ">k", clip_on=False, markersize=6)
    ax2.plot(0, 1.03, "^k", clip_on=False, markersize=6)

    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 3.16 saved successfully to: {p}")
    if show:
        plt.show()
    return fig, (ax1, ax2)


# =====================================================================
# Aliases for Section 3.5 Nonparametric Methods
# =====================================================================
HistogramDensity = HistogramDensity1D
KernelDensityEstimator = KernelDensity1D
KNearestNeighborsDensity = KNNDensityEstimator
KNearestNeighborsClassifier = KNNClassifier

plot_figure_3_13 = plot_figure_3_13_histogram
plot_figure_3_14 = plot_figure_3_14_kernel_density
plot_figure_3_15 = plot_figure_3_15_knn_density
plot_figure_3_16 = plot_figure_3_16_knn_classification
