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


