"""
Probability utilities for Chapter 2: The Rules of Probability.
Provides exact derivations, calculations, and simulation helpers for discrete distributions,
Bayes' theorem, medical screening problem, and uncertainty demonstrations.
"""
from typing import Dict, Tuple, Optional, Callable, Union, List
import numpy as np
from scipy import special


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
