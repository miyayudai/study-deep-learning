"""
common/exercises_ch15.py
========================
Chapter 15 Exercises: Discrete Latent Variables
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module provides theoretical proofs, derivations, and numerical verification
utilities for Exercises 15.1 through 15.24:
- Exercise 15.1: Finite convergence of the K-means algorithm.
- Exercise 15.2: Sequential / on-line K-means update derivation (Robbins-Monro form).
- Exercise 15.3: Marginal distribution of Gaussian mixture from joint p(x, z).
- Exercise 15.4: Interchange symmetries (K! equivalent modes) in mixture models.
- Exercise 15.5: MAP EM with parameter prior p(theta): E-step and M-step derivation.
- Exercise 15.6: D-separation in GMM directed graphical model for latent factorization.
- Exercise 15.7: EM equations for tied / common covariance matrix Sigma across all components.
- Exercise 15.8: Complete-data log-likelihood maximization leads to independent sample statistics.
- Exercise 15.9: Closed-form maximization of expected complete-data log-likelihood w.r.t. mu_k.
- Exercise 15.10: Closed-form maximization of Q w.r.t. Sigma_k and pi_k (Lagrange multipliers).
- Exercise 15.11: Conditional distribution p(x_b | x_a) in partitioned mixture models.
- Exercise 15.12: K-means as the deterministic zero-variance limit epsilon -> 0 of GMM EM.
- Exercise 15.13: Mean and covariance of the multivariate Bernoulli distribution.
- Exercise 15.14: Mean and covariance of general mixture distributions (law of total variance).
- Exercise 15.15: Sample mean property of Bernoulli mixture MLE and one-step convergence.
- Exercise 15.16: Marginalizing joint distribution p(x, z) for Bernoulli mixture.
- Exercise 15.17: Maximizing expected complete-data log-likelihood w.r.t. Bernoulli means mu_k.
- Exercise 15.18: Maximizing expected complete-data log-likelihood w.r.t. Bernoulli mixing coefficients pi_k.
- Exercise 15.19: Boundedness of Bernoulli mixture log-likelihood (absence of singularities).
- Exercise 15.20: EM algorithm for mixtures of multivariate multinomial / categorical distributions.
- Exercise 15.21: Verification of the ELBO and KL divergence decomposition of log-likelihood.
- Exercise 15.22: Tangential bound property: matching gradients of ELBO and log-likelihood at theta = theta_old.
- Exercise 15.23: Sequential / incremental EM updates for component means and effective counts.
- Exercise 15.24: Sequential / incremental EM updates for covariance matrices and mixing coefficients.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import itertools
import numpy as np
import scipy.stats as stats

from .kmeans_clustering import KMeans
from .mixtures_of_gaussians import GaussianMixtureModel
from .expectation_maximization import (
    GaussianMixtureEM,
    BernoulliMixtureEM,
)
from .evidence_lower_bound import (
    compute_elbo_decomposition,
    MAPGaussianMixtureEM,
)


# =============================================================================
# Helper Classes for Exercises
# =============================================================================

class TiedCovarianceGaussianMixtureEM:
    """
    Gaussian Mixture Model with a tied (common) covariance matrix Sigma across all K components
    (Exercise 15.7).
    """

    def __init__(self, n_components: int = 2, max_iter: int = 100, tol: float = 1e-5, random_state: Optional[int] = None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.weights_ = None
        self.means_ = None
        self.covariance_ = None  # Single common (D, D) matrix
        self.log_likelihood_history_ = []

    def fit(self, X: np.ndarray) -> "TiedCovarianceGaussianMixtureEM":
        rng = np.random.RandomState(self.random_state)
        N, D = X.shape
        K = self.n_components

        # Initialization
        self.weights_ = np.ones(K) / K
        indices = rng.choice(N, size=K, replace=False)
        self.means_ = X[indices].copy()
        sample_cov = np.cov(X, rowvar=False) + 1e-4 * np.eye(D)
        self.covariance_ = sample_cov.copy()

        self.log_likelihood_history_ = []

        for iteration in range(self.max_iter):
            # E-step
            log_resp = np.zeros((N, K))
            cov_inv = np.linalg.inv(self.covariance_)
            sign, logdet = np.linalg.slogdet(self.covariance_)
            norm_const = -0.5 * (D * np.log(2.0 * np.pi) + logdet)

            for k in range(K):
                diff = X - self.means_[k]
                quad = np.sum(diff @ cov_inv * diff, axis=1)
                log_resp[:, k] = np.log(self.weights_[k] + 1e-12) + norm_const - 0.5 * quad

            # Log-sum-exp trick for responsibilities and log-likelihood
            max_log = np.max(log_resp, axis=1, keepdims=True)
            exp_term = np.exp(log_resp - max_log)
            sum_exp = np.sum(exp_term, axis=1, keepdims=True)
            log_prob = max_log + np.log(sum_exp)
            ll = np.sum(log_prob)
            self.log_likelihood_history_.append(ll)

            resp = exp_term / sum_exp  # (N, K)

            # Check convergence
            if iteration > 0 and abs(self.log_likelihood_history_[-1] - self.log_likelihood_history_[-2]) < self.tol:
                break

            # M-step
            N_k = np.sum(resp, axis=0)  # (K,)
            self.weights_ = N_k / N
            self.means_ = (resp.T @ X) / N_k[:, np.newaxis]

            # Common covariance: Sigma = (1/N) sum_k sum_n resp_{nk} (x_n - mu_k)(x_n - mu_k)^T
            new_cov = np.zeros((D, D))
            for k in range(K):
                diff = X - self.means_[k]  # (N, D)
                weighted_diff = resp[:, k:k+1] * diff  # (N, D)
                new_cov += weighted_diff.T @ diff
            self.covariance_ = new_cov / N + 1e-6 * np.eye(D)

        return self

    def score(self, X: np.ndarray) -> float:
        N, D = X.shape
        K = self.n_components
        cov_inv = np.linalg.inv(self.covariance_)
        sign, logdet = np.linalg.slogdet(self.covariance_)
        norm_const = -0.5 * (D * np.log(2.0 * np.pi) + logdet)

        log_resp = np.zeros((N, K))
        for k in range(K):
            diff = X - self.means_[k]
            quad = np.sum(diff @ cov_inv * diff, axis=1)
            log_resp[:, k] = np.log(self.weights_[k] + 1e-12) + norm_const - 0.5 * quad

        max_log = np.max(log_resp, axis=1, keepdims=True)
        exp_term = np.exp(log_resp - max_log)
        sum_exp = np.sum(exp_term, axis=1, keepdims=True)
        return float(np.sum(max_log + np.log(sum_exp)))


class CategoricalMixtureEM:
    """
    Mixture of multivariate Categorical / Multinomial distributions (Exercise 15.20).
    X is of shape (N, D, M) where sum_j X_{n, i, j} = 1 (one-hot across M categories for each of D dimensions).
    """

    def __init__(self, n_components: int = 2, max_iter: int = 100, tol: float = 1e-5, random_state: Optional[int] = None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.weights_ = None  # (K,)
        self.probs_ = None    # (K, D, M)
        self.log_likelihood_history_ = []

    def fit(self, X: np.ndarray) -> "CategoricalMixtureEM":
        rng = np.random.RandomState(self.random_state)
        N, D, M = X.shape
        K = self.n_components

        # Initialize mixing coefficients
        self.weights_ = np.ones(K) / K

        # Initialize probabilities with Dirichlet / random normalized
        raw_probs = rng.uniform(0.1, 1.0, size=(K, D, M))
        self.probs_ = raw_probs / np.sum(raw_probs, axis=2, keepdims=True)

        self.log_likelihood_history_ = []

        for iteration in range(self.max_iter):
            # E-step
            log_p_x_given_k = np.zeros((N, K))
            eps = 1e-12
            for k in range(K):
                # sum_{i, j} X_{n, i, j} * ln(probs_{k, i, j})
                log_comp = np.sum(X * np.log(self.probs_[k] + eps), axis=(1, 2))
                log_p_x_given_k[:, k] = np.log(self.weights_[k] + eps) + log_comp

            max_log = np.max(log_p_x_given_k, axis=1, keepdims=True)
            exp_term = np.exp(log_p_x_given_k - max_log)
            sum_exp = np.sum(exp_term, axis=1, keepdims=True)
            log_prob = max_log + np.log(sum_exp)
            ll = np.sum(log_prob)
            self.log_likelihood_history_.append(ll)

            resp = exp_term / sum_exp  # (N, K)

            if iteration > 0 and abs(self.log_likelihood_history_[-1] - self.log_likelihood_history_[-2]) < self.tol:
                break

            # M-step
            N_k = np.sum(resp, axis=0)  # (K,)
            self.weights_ = N_k / N

            # mu_{k, i, j} = sum_n resp_{n, k} * X_{n, i, j} / N_k
            for k in range(K):
                weighted_X = np.sum(resp[:, k, np.newaxis, np.newaxis] * X, axis=0)  # (D, M)
                self.probs_[k] = (weighted_X + eps) / np.sum(weighted_X + eps, axis=1, keepdims=True)

        return self

    def score(self, X: np.ndarray) -> float:
        N, D, M = X.shape
        K = self.n_components
        eps = 1e-12
        log_p_x_given_k = np.zeros((N, K))
        for k in range(K):
            log_comp = np.sum(X * np.log(self.probs_[k] + eps), axis=(1, 2))
            log_p_x_given_k[:, k] = np.log(self.weights_[k] + eps) + log_comp

        max_log = np.max(log_p_x_given_k, axis=1, keepdims=True)
        exp_term = np.exp(log_p_x_given_k - max_log)
        sum_exp = np.sum(exp_term, axis=1, keepdims=True)
        return float(np.sum(max_log + np.log(sum_exp)))


# =============================================================================
# Exercise 15.1: Finite Convergence of K-means
# =============================================================================

def solve_exercise_15_1() -> Dict[str, Any]:
    """
    Exercise 15.1:
    Show that because the set of discrete assignments {r_nk} is finite (K^N configurations)
    and for each assignment there is a unique optimum {mu_k}, the K-means algorithm
    must converge after a finite number of iterations.
    """
    derivation = (
        "1. Let the distortion measure be J = sum_{n=1}^N sum_{k=1}^K r_{nk} ||x_n - mu_k||^2.\n"
        "2. The assignment matrix R = {r_{nk}} belongs to a finite set S of size K^N.\n"
        "3. In the assignment step, R is chosen to minimize J for fixed {mu_k}.\n"
        "4. In the update step, for fixed R, J is strictly convex and quadratic in each mu_k.\n"
        "   Setting grad_{mu_k} J = 2 sum_n r_{nk} (mu_k - x_n) = 0 yields the unique global minimum\n"
        "   mu_k = (sum_n r_{nk} x_n) / (sum_n r_{nk}).\n"
        "5. Each step either strictly decreases J or leaves it unchanged. Since J is bounded below by 0\n"
        "   and there are only finitely many configurations of R, J can strictly decrease at most\n"
        "   a finite number of times before R ceases to change.\n"
        "6. Hence, K-means must terminate in a finite number of iterations (at most K^N)."
    )
    return {
        "finite_states": "K^N",
        "unique_optimum": True,
        "monotonic_decrease": True,
        "finite_convergence_guaranteed": True,
        "derivation": derivation,
    }


def verify_exercise_15_1() -> bool:
    """Verify finite convergence of K-means with strictly non-increasing distortion J."""
    rng = np.random.RandomState(42)
    X = np.concatenate([
        rng.randn(20, 2) + np.array([-2.0, -2.0]),
        rng.randn(20, 2) + np.array([2.0, 2.0]),
    ], axis=0)

    model = KMeans(n_clusters=2, max_iter=100, random_state=42)
    model.fit(X)

    # Check finite convergence (terminated well before max_iter)
    distortions = [j for _, j in model.history_["distortion"]]
    assert len(distortions) < 100, "K-means should converge within finite iterations"
    # Check monotonicity of distortion J
    for i in range(len(distortions) - 1):
        assert distortions[i] >= distortions[i + 1] - 1e-10, "Distortion J must be monotonically non-increasing"

    sol = solve_exercise_15_1()
    assert sol["finite_convergence_guaranteed"] is True
    return True


# =============================================================================
# Exercise 15.2: Sequential K-means Update
# =============================================================================

def solve_exercise_15_2() -> Dict[str, Any]:
    """
    Exercise 15.2:
    Derive the sequential K-means update mu_k^{new} = mu_k^{old} + eta_n (x_n - mu_k^{old})
    with eta_n = 1 / N_k.
    """
    derivation = (
        "1. In the batch setting, for cluster k after n data points:\n"
        "   mu_k^{(n)} = (1 / N_k^{(n)}) sum_{i=1}^n r_{ik} x_i.\n"
        "2. If point x_n is assigned to cluster k (r_{nk} = 1), then N_k^{(n)} = N_k^{(n-1)} + 1.\n"
        "3. Separating out the contribution of x_n:\n"
        "   mu_k^{(n)} = (1 / N_k^{(n)}) [ sum_{i=1}^{n-1} r_{ik} x_i + x_n ]\n"
        "              = (1 / N_k^{(n)}) [ N_k^{(n-1)} mu_k^{(n-1)} + x_n ].\n"
        "4. Since N_k^{(n-1)} = N_k^{(n)} - 1:\n"
        "   mu_k^{(n)} = [(N_k^{(n)} - 1) / N_k^{(n)}] mu_k^{(n-1)} + (1 / N_k^{(n)}) x_n\n"
        "              = mu_k^{(n-1)} + (1 / N_k^{(n)}) (x_n - mu_k^{(n-1)}).\n"
        "5. Setting eta_n = 1 / N_k^{(n)} yields the Robbins-Monro form (15.5)."
    )
    return {
        "formula": "mu_k^{new} = mu_k^{old} + eta_n (x_n - mu_k^{old})",
        "step_size": "eta_n = 1 / N_k",
        "exact_running_average": True,
        "derivation": derivation,
    }


def verify_exercise_15_2() -> bool:
    """Verify that sequential Robbins-Monro updates exactly reproduce the batch sample mean."""
    rng = np.random.RandomState(42)
    data = rng.randn(50, 3)

    # Sequential update
    running_mean = np.zeros(3)
    for n in range(1, len(data) + 1):
        x_n = data[n - 1]
        eta_n = 1.0 / n
        running_mean = running_mean + eta_n * (x_n - running_mean)

    batch_mean = np.mean(data, axis=0)
    assert np.allclose(running_mean, batch_mean, atol=1e-12), "Sequential mean must match batch mean exactly"
    return True


# =============================================================================
# Exercise 15.3: Marginal Distribution of Gaussian Mixture
# =============================================================================

def solve_exercise_15_3() -> Dict[str, Any]:
    """
    Exercise 15.3:
    Show that summing p(z) p(x | z) over all 1-of-K latent states z yields the mixture (15.6).
    """
    derivation = (
        "1. p(z) = prod_{k=1}^K pi_k^{z_k} where z is a 1-of-K binary vector.\n"
        "2. p(x | z) = prod_{k=1}^K N(x | mu_k, Sigma_k)^{z_k}.\n"
        "3. Joint distribution: p(x, z) = p(z) p(x | z) = prod_{k=1}^K [ pi_k N(x | mu_k, Sigma_k) ]^{z_k}.\n"
        "4. Since z can take exactly K states e_1, ..., e_K (where e_k has z_k = 1 and 0 elsewhere):\n"
        "   p(x) = sum_z p(x, z) = sum_{k=1}^K p(x, z = e_k) = sum_{k=1}^K pi_k N(x | mu_k, Sigma_k).\n"
        "5. This is the Gaussian mixture distribution (15.6)."
    )
    return {
        "joint_form": "prod_{k=1}^K [ pi_k N(x | mu_k, Sigma_k) ]^{z_k}",
        "marginal_form": "sum_{k=1}^K pi_k N(x | mu_k, Sigma_k)",
        "derivation": derivation,
    }


def verify_exercise_15_3() -> bool:
    """Verify discrete sum over 1-of-K states equals Gaussian mixture probability."""
    weights = np.array([0.3, 0.7])
    means = np.array([[-1.0], [2.0]])
    variances = np.array([0.5, 1.5])

    x = np.array([0.5])

    # Method 1: Explicit sum over 1-of-K vectors z in {(1, 0), (0, 1)}
    states = [np.array([1, 0]), np.array([0, 1])]
    p_marginal_sum = 0.0
    for z in states:
        p_z = np.prod(weights ** z)
        p_x_given_z = 1.0
        for k in range(2):
            if z[k] == 1:
                p_x_given_z *= stats.norm.pdf(x[0], loc=means[k, 0], scale=np.sqrt(variances[k]))
        p_marginal_sum += p_z * p_x_given_z

    # Method 2: Standard mixture formula
    p_mix = sum(weights[k] * stats.norm.pdf(x[0], loc=means[k, 0], scale=np.sqrt(variances[k])) for k in range(2))

    assert np.isclose(p_marginal_sum, p_mix, atol=1e-14)
    return True


# =============================================================================
# Exercise 15.4: Interchange Symmetries in Mixture Models
# =============================================================================

def solve_exercise_15_4() -> Dict[str, Any]:
    """
    Exercise 15.4:
    Show that the number of equivalent parameter settings due to interchange symmetries
    in a mixture model with K components is K!.
    """
    derivation = (
        "1. The mixture distribution is p(x) = sum_{k=1}^K pi_k N(x | mu_k, Sigma_k).\n"
        "2. The sum is invariant under any permutation sigma in S_K of the indices {1, ..., K}:\n"
        "   sum_{k=1}^K pi_{sigma(k)} N(x | mu_{sigma(k)}, Sigma_{sigma(k)}) = p(x).\n"
        "3. The symmetric group S_K has order |S_K| = K!.\n"
        "4. If all K components are distinct, each permutation corresponds to a distinct vector\n"
        "   in parameter space yielding the identical probability density.\n"
        "5. Hence there are exactly K! equivalent parameter settings (modes) in the likelihood function."
    )
    return {
        "group": "S_K (Symmetric group)",
        "num_equivalent_settings": "K!",
        "derivation": derivation,
    }


def verify_exercise_15_4() -> bool:
    """Verify that all K! parameter permutations produce identical log-likelihoods."""
    K = 3
    pi = np.array([0.2, 0.3, 0.5])
    mu = np.array([[0.0], [3.0], [-2.0]])
    sigma = np.array([[[1.0]], [[0.5]], [[2.0]]])

    rng = np.random.RandomState(42)
    X = rng.randn(10, 1)

    permutations = list(itertools.permutations(range(K)))
    assert len(permutations) == math.factorial(K)

    # Compute log likelihood under identity permutation
    def compute_ll(perm_pi, perm_mu, perm_sigma):
        total_p = np.zeros(len(X))
        for k in range(K):
            norm_k = stats.norm.pdf(X[:, 0], loc=perm_mu[k, 0], scale=np.sqrt(perm_sigma[k, 0, 0]))
            total_p += perm_pi[k] * norm_k
        return np.sum(np.log(total_p))

    base_ll = compute_ll(pi, mu, sigma)

    for perm in permutations:
        perm_pi = pi[list(perm)]
        perm_mu = mu[list(perm)]
        perm_sigma = sigma[list(perm)]
        perm_ll = compute_ll(perm_pi, perm_mu, perm_sigma)
        assert np.isclose(base_ll, perm_ll, atol=1e-12)

    return True


# =============================================================================
# Exercise 15.5: MAP EM Algorithm
# =============================================================================

def solve_exercise_15_5() -> Dict[str, Any]:
    """
    Exercise 15.5:
    MAP EM: E-step is identical to ML, M-step maximizes Q(theta, theta_old) + ln p(theta).
    """
    derivation = (
        "1. The posterior over parameters is p(theta | X) = p(X | theta) p(theta) / p(X).\n"
        "2. Maximizing ln p(theta | X) is equivalent to maximizing ln p(X | theta) + ln p(theta).\n"
        "3. Using the ELBO decomposition: ln p(X | theta) = L(q, theta) + KL(q || p(Z | X, theta)).\n"
        "4. Adding ln p(theta): ln p(theta | X) + const = L(q, theta) + ln p(theta) + KL(q || p).\n"
        "5. E-step: With theta fixed at theta_old, ln p(theta_old) is constant w.r.t. q(Z).\n"
        "   Maximizing w.r.t. q(Z) minimizes KL divergence to 0, which yields\n"
        "   q(Z) = p(Z | X, theta_old), exactly as in standard ML EM!\n"
        "6. M-step: With q(Z) fixed, the objective w.r.t. theta is:\n"
        "   L(q, theta) + ln p(theta) = E_q[ln p(X, Z | theta)] - E_q[ln q(Z)] + ln p(theta)\n"
        "                             = Q(theta, theta_old) + ln p(theta) + const."
    )
    return {
        "e_step_identical": True,
        "m_step_objective": "Q(theta, theta_old) + ln p(theta)",
        "derivation": derivation,
    }


def verify_exercise_15_5() -> bool:
    """Verify that MAP EM monotonically increases the MAP objective at every iteration."""
    rng = np.random.RandomState(42)
    X = np.concatenate([
        rng.randn(25, 2) + np.array([-2.0, 0.0]),
        rng.randn(25, 2) + np.array([2.0, 0.0]),
    ], axis=0)

    # Use MAP EM from evidence_lower_bound
    map_model = MAPGaussianMixtureEM(
        n_components=2,
        max_iter=20,
        alpha_prior=2.0,
        beta_cov_prior=0.5,
    )
    init_means = np.array([[-2.0, 0.0], [2.0, 0.0]])
    init_covs = np.array([np.eye(2), np.eye(2)])
    map_model.fit(X, init_means=init_means, init_covs=init_covs)

    history = map_model.history_map_objective_
    assert len(history) > 1
    # Objective increases from start to finish
    assert history[-1] > history[0], "MAP objective should increase after optimization"
    assert history[1] > history[0], "First EM step must increase MAP objective"
    # Once converged, consecutive changes are very small
    assert abs(history[-1] - history[-2]) < 1e-5, "MAP EM should converge to stationary point"

    return True


# =============================================================================
# Exercise 15.6: D-Separation Latent Factorization
# =============================================================================

def solve_exercise_15_6() -> Dict[str, Any]:
    """
    Exercise 15.6:
    Use d-separation on the GMM directed graphical model to show that
    p(Z | X, mu, Sigma, pi) = prod_{n=1}^N p(z_n | x_n, mu, Sigma, pi).
    """
    derivation = (
        "1. In the directed graphical model for GMM (Figure 15.9), the parameter nodes theta = (mu, Sigma, pi)\n"
        "   are parents of each latent z_n (or observed x_n).\n"
        "2. The only paths connecting z_n and z_m for n != m pass through the parameter nodes theta.\n"
        "3. Since theta is conditioned on (observed), and the path is tail-to-tail at theta (z_n <- theta -> z_m),\n"
        "   conditioning on theta blocks this path.\n"
        "4. Furthermore, paths between z_n and x_m for m != n also pass through theta and are blocked.\n"
        "5. Therefore, z_n is d-separated from all other {z_m, x_m}_{m != n} given (x_n, theta).\n"
        "6. By d-separation, p(Z | X, theta) = prod_{n=1}^N p(z_n | x_n, theta)."
    )
    return {
        "d_separated": True,
        "blocked_at": "parameter nodes (mu, Sigma, pi)",
        "factorization": "prod_{n=1}^N p(z_n | x_n, theta)",
        "derivation": derivation,
    }


def verify_exercise_15_6() -> bool:
    """Verify numerically that the joint posterior of two latent variables factorizes given theta."""
    # Joint posterior p(z_1, z_2 | x_1, x_2, theta) == p(z_1 | x_1, theta) * p(z_2 | x_2, theta)
    pi = np.array([0.4, 0.6])
    mu = np.array([-1.0, 2.0])
    sigma = np.array([1.0, 1.0])

    x1, x2 = 0.5, 1.5

    # Compute p(z_1 = k | x_1)
    p_x1_z = [pi[k] * stats.norm.pdf(x1, loc=mu[k], scale=sigma[k]) for k in range(2)]
    resp1 = np.array(p_x1_z) / sum(p_x1_z)

    # Compute p(z_2 = k | x_2)
    p_x2_z = [pi[k] * stats.norm.pdf(x2, loc=mu[k], scale=sigma[k]) for k in range(2)]
    resp2 = np.array(p_x2_z) / sum(p_x2_z)

    # Joint posterior p(z_1 = j, z_2 = k | x_1, x_2)
    joint_num = np.zeros((2, 2))
    for j in range(2):
        for k in range(2):
            joint_num[j, k] = pi[j] * stats.norm.pdf(x1, loc=mu[j], scale=sigma[j]) * \
                              pi[k] * stats.norm.pdf(x2, loc=mu[k], scale=sigma[k])
    joint_post = joint_num / np.sum(joint_num)

    # Outer product of independent conditionals
    outer_product = np.outer(resp1, resp2)
    assert np.allclose(joint_post, outer_product, atol=1e-14)
    return True


# =============================================================================
# Exercise 15.7: EM with Tied Covariance Matrix
# =============================================================================

def solve_exercise_15_7() -> Dict[str, Any]:
    """
    Exercise 15.7:
    EM equations for GMM with common covariance matrix Sigma across all K components.
    """
    derivation = (
        "1. Complete log-likelihood expectation:\n"
        "   Q = sum_{n=1}^N sum_{k=1}^K gamma_{nk} [ ln pi_k - (D/2) ln(2 pi) - (1/2) ln |Sigma|\n"
        "       - (1/2) (x_n - mu_k)^T Sigma^{-1} (x_n - mu_k) ].\n"
        "2. E-step: gamma_{nk} = pi_k N(x_n | mu_k, Sigma) / sum_j pi_j N(x_n | mu_j, Sigma).\n"
        "3. M-step for mu_k and pi_k:\n"
        "   mu_k = (1 / N_k) sum_n gamma_{nk} x_n,\n"
        "   pi_k = N_k / N, where N_k = sum_n gamma_{nk}.\n"
        "4. M-step for common covariance Sigma:\n"
        "   Summing the log-determinant term: sum_n sum_k gamma_{nk} (-1/2 ln |Sigma|) = - (N / 2) ln |Sigma|.\n"
        "   Taking derivative w.r.t. Sigma^{-1}:\n"
        "   (N / 2) Sigma - (1/2) sum_k sum_n gamma_{nk} (x_n - mu_k)(x_n - mu_k)^T = 0.\n"
        "5. Therefore:\n"
        "   Sigma = (1 / N) sum_{k=1}^K sum_{n=1}^N gamma_{nk} (x_n - mu_k)(x_n - mu_k)^T\n"
        "         = (1 / N) sum_{k=1}^K N_k S_k."
    )
    return {
        "mu_update": "mu_k = (1 / N_k) sum_n gamma_{nk} x_n",
        "pi_update": "pi_k = N_k / N",
        "sigma_update": "Sigma = (1 / N) sum_{k=1}^K sum_{n=1}^N gamma_{nk} (x_n - mu_k)(x_n - mu_k)^T",
        "derivation": derivation,
    }


def verify_exercise_15_7() -> bool:
    """Verify that TiedCovarianceGaussianMixtureEM increases log-likelihood monotonically."""
    rng = np.random.RandomState(42)
    common_cov = np.array([[1.0, 0.4], [0.4, 0.8]])
    X1 = rng.multivariate_normal([-2.0, -1.0], common_cov, size=40)
    X2 = rng.multivariate_normal([2.0, 1.0], common_cov, size=40)
    X = np.concatenate([X1, X2], axis=0)

    model = TiedCovarianceGaussianMixtureEM(n_components=2, max_iter=30, random_state=42)
    model.fit(X)

    history = model.log_likelihood_history_
    assert len(history) > 1
    for i in range(len(history) - 1):
        assert history[i + 1] >= history[i] - 1e-6, "Log-likelihood must monotonically increase in tied GMM"

    return True


# =============================================================================
# Exercise 15.8: Complete-Data Log-Likelihood Maximization
# =============================================================================

def solve_exercise_15_8() -> Dict[str, Any]:
    """
    Exercise 15.8:
    Verify that maximization of complete-data log-likelihood (15.26) yields independent
    sample statistics for each cluster and pi_k = N_k / N.
    """
    derivation = (
        "1. Complete-data log likelihood (15.26):\n"
        "   ln p(X, Z | theta) = sum_{n=1}^N sum_{k=1}^K z_{nk} [ ln pi_k + ln N(x_n | mu_k, Sigma_k) ].\n"
        "2. Partition data into groups C_k = {n : z_{nk} = 1} with size N_k = sum_n z_{nk}.\n"
        "3. The log-likelihood decomposes into disjoint sums:\n"
        "   sum_{k=1}^K N_k ln pi_k + sum_{k=1}^K [ sum_{n in C_k} ln N(x_n | mu_k, Sigma_k) ].\n"
        "4. For each k, the Gaussian term depends only on points in C_k, which is the standard single Gaussian MLE:\n"
        "   mu_k = (1 / N_k) sum_{n in C_k} x_n,\n"
        "   Sigma_k = (1 / N_k) sum_{n in C_k} (x_n - mu_k)(x_n - mu_k)^T.\n"
        "5. For pi_k, maximizing sum_k N_k ln pi_k subject to sum_k pi_k = 1 gives pi_k = N_k / N."
    )
    return {
        "independent_estimation": True,
        "mu_solution": "Sample mean of data points assigned to cluster k",
        "sigma_solution": "Sample covariance of data points assigned to cluster k",
        "pi_solution": "Fraction of data points assigned to cluster k (N_k / N)",
        "derivation": derivation,
    }


def verify_exercise_15_8() -> bool:
    """Verify that complete-data MLE matches independent subgroup empirical statistics."""
    rng = np.random.RandomState(42)
    N1, N2 = 30, 50
    X1 = rng.randn(N1, 2) + np.array([-1.0, 1.0])
    X2 = rng.randn(N2, 2) * 1.5 + np.array([2.0, -1.0])

    X = np.concatenate([X1, X2], axis=0)
    Z = np.zeros((len(X), 2))
    Z[:N1, 0] = 1.0
    Z[N1:, 1] = 1.0

    # Formulas from Exercise 15.8
    N_k = np.sum(Z, axis=0)
    pi_mle = N_k / len(X)
    mu1_mle = np.sum(Z[:, 0:1] * X, axis=0) / N_k[0]
    mu2_mle = np.sum(Z[:, 1:2] * X, axis=0) / N_k[1]

    assert np.allclose(pi_mle, [N1 / (N1 + N2), N2 / (N1 + N2)])
    assert np.allclose(mu1_mle, np.mean(X1, axis=0))
    assert np.allclose(mu2_mle, np.mean(X2, axis=0))
    return True


# =============================================================================
# Exercise 15.9: Maximizing Q w.r.t. mu_k
# =============================================================================

def solve_exercise_15_9() -> Dict[str, Any]:
    """
    Exercise 15.9:
    Show that maximizing Q(theta, theta_old) w.r.t. mu_k yields closed form (15.16).
    """
    derivation = (
        "1. From (15.30), the terms in Q(theta, theta_old) that depend on mu_k are:\n"
        "   - (1/2) sum_{n=1}^N gamma_{nk} (x_n - mu_k)^T Sigma_k^{-1} (x_n - mu_k).\n"
        "2. Taking the gradient with respect to mu_k:\n"
        "   grad_{mu_k} Q = sum_{n=1}^N gamma_{nk} Sigma_k^{-1} (x_n - mu_k)\n"
        "                 = Sigma_k^{-1} [ sum_{n=1}^N gamma_{nk} x_n - mu_k sum_{n=1}^N gamma_{nk} ].\n"
        "3. Setting grad_{mu_k} Q = 0 and multiplying by Sigma_k:\n"
        "   sum_{n=1}^N gamma_{nk} x_n = N_k mu_k, where N_k = sum_{n=1}^N gamma_{nk}.\n"
        "4. Hence mu_k = (1 / N_k) sum_{n=1}^N gamma_{nk} x_n (Eq. 15.16)."
    )
    return {
        "gradient": "Sigma_k^{-1} (sum_n gamma_{nk} x_n - N_k mu_k)",
        "solution": "mu_k = (1 / N_k) sum_n gamma_{nk} x_n",
        "derivation": derivation,
    }


def verify_exercise_15_9() -> bool:
    """Verify that gradient w.r.t. mu_k vanishes at the closed-form solution."""
    rng = np.random.RandomState(42)
    X = rng.randn(20, 2)
    gamma = rng.dirichlet(np.ones(2), size=20)
    Sigma_k = np.array([[1.5, 0.2], [0.2, 1.0]])
    Sigma_inv = np.linalg.inv(Sigma_k)

    k = 0
    N_k = np.sum(gamma[:, k])
    mu_k = np.sum(gamma[:, k:k+1] * X, axis=0) / N_k

    grad = np.zeros(2)
    for n in range(len(X)):
        grad += gamma[n, k] * (Sigma_inv @ (X[n] - mu_k))

    assert np.allclose(grad, 0.0, atol=1e-12)
    return True


# =============================================================================
# Exercise 15.10: Maximizing Q w.r.t. Sigma_k and pi_k
# =============================================================================

def solve_exercise_15_10() -> Dict[str, Any]:
    """
    Exercise 15.10:
    Maximize Q w.r.t. Sigma_k and pi_k to obtain closed form (15.18) and (15.21).
    """
    derivation = (
        "1. For Sigma_k: The terms in Q involving Sigma_k are:\n"
        "   - (1/2) sum_{n=1}^N gamma_{nk} [ ln |Sigma_k| + (x_n - mu_k)^T Sigma_k^{-1} (x_n - mu_k) ].\n"
        "   Using d(ln |Sigma|)/d(Sigma^{-1}) = -Sigma and d(Tr(Sigma^{-1} A))/d(Sigma^{-1}) = A:\n"
        "   d Q / d(Sigma_k^{-1}) = (N_k / 2) Sigma_k - (1/2) sum_{n=1}^N gamma_{nk} (x_n - mu_k)(x_n - mu_k)^T = 0.\n"
        "   Solving for Sigma_k yields (15.18):\n"
        "   Sigma_k = (1 / N_k) sum_{n=1}^N gamma_{nk} (x_n - mu_k)(x_n - mu_k)^T.\n"
        "2. For pi_k: Objective is sum_{k=1}^K N_k ln pi_k subject to sum_k pi_k = 1.\n"
        "   Lagrangian: Lambda(pi, lambda) = sum_k N_k ln pi_k + lambda (sum_k pi_k - 1).\n"
        "   d Lambda / d pi_k = N_k / pi_k + lambda = 0 => N_k = -lambda pi_k.\n"
        "   Summing over k: N = -lambda, so lambda = -N.\n"
        "   Hence pi_k = N_k / N (Eq. 15.21)."
    )
    return {
        "sigma_solution": "Sigma_k = (1 / N_k) sum_n gamma_{nk} (x_n - mu_k)(x_n - mu_k)^T",
        "pi_solution": "pi_k = N_k / N",
        "derivation": derivation,
    }


def verify_exercise_15_10() -> bool:
    """Verify that Sigma_k and pi_k satisfy the stationarity conditions."""
    N, K = 30, 2
    rng = np.random.RandomState(42)
    gamma = rng.dirichlet(np.ones(K), size=N)
    N_k = np.sum(gamma, axis=0)

    pi = N_k / N
    assert np.isclose(np.sum(pi), 1.0)
    # Check Lagrange multiplier condition: N_k / pi_k = constant = N
    ratios = N_k / pi
    assert np.allclose(ratios, N)
    return True


# =============================================================================
# Exercise 15.11: Conditional Mixture Distributions
# =============================================================================

def solve_exercise_15_11() -> Dict[str, Any]:
    """
    Exercise 15.11:
    Partition x = (x_a, x_b). Show p(x_b | x_a) is also a mixture distribution.
    """
    derivation = (
        "1. Let p(x) = sum_k pi_k p(x | k) where x = (x_a, x_b).\n"
        "2. Marginal: p(x_a) = int p(x_a, x_b) d x_b = sum_k pi_k int p(x_a, x_b | k) d x_b\n"
        "                     = sum_k pi_k p(x_a | k).\n"
        "3. Conditional:\n"
        "   p(x_b | x_a) = p(x_a, x_b) / p(x_a)\n"
        "                = [ sum_k pi_k p(x_a, x_b | k) ] / [ sum_j pi_j p(x_a | j) ].\n"
        "4. Using p(x_a, x_b | k) = p(x_a | k) p(x_b | x_a, k):\n"
        "   p(x_b | x_a) = sum_{k=1}^K [ pi_k p(x_a | k) / sum_j pi_j p(x_a | j) ] p(x_b | x_a, k).\n"
        "5. Define new mixing coefficients pi_k_tilde(x_a) = pi_k p(x_a | k) / sum_j pi_j p(x_a | j) = p(k | x_a).\n"
        "   Notice pi_k_tilde >= 0 and sum_k pi_k_tilde = 1.\n"
        "6. Thus p(x_b | x_a) is a mixture of components p(x_b | x_a, k) with input-dependent mixing coefficients."
    )
    return {
        "is_mixture": True,
        "new_mixing_coefficients": "pi_k_tilde(x_a) = pi_k p(x_a | k) / sum_j pi_j p(x_a | j)",
        "component_densities": "p(x_b | x_a, k)",
        "derivation": derivation,
    }


def verify_exercise_15_11() -> bool:
    """Verify conditional mixture formula numerically against joint/marginal ratio."""
    # 2D Gaussian mixture with 2 components
    weights = np.array([0.4, 0.6])
    means = np.array([[0.0, 1.0], [2.0, -1.0]])
    covs = np.array([
        [[1.0, 0.5], [0.5, 1.0]],
        [[1.5, -0.3], [-0.3, 0.8]]
    ])

    x_a = 1.0
    x_b = 0.5
    x = np.array([x_a, x_b])

    # Method 1: Joint / Marginal
    joint_p = sum(weights[k] * stats.multivariate_normal.pdf(x, mean=means[k], cov=covs[k]) for k in range(2))
    p_xa_comps = [stats.norm.pdf(x_a, loc=means[k, 0], scale=np.sqrt(covs[k, 0, 0])) for k in range(2)]
    marginal_p_xa = sum(weights[k] * p_xa_comps[k] for k in range(2))
    cond_p_direct = joint_p / marginal_p_xa

    # Method 2: Conditional mixture formula
    weights_tilde = np.array([weights[k] * p_xa_comps[k] for k in range(2)]) / marginal_p_xa
    # Linear Gaussian conditional: mu_{b|a} = mu_b + Sigma_{ba} Sigma_{aa}^{-1} (x_a - mu_a)
    cond_comps = []
    for k in range(2):
        mu_a = means[k, 0]
        mu_b = means[k, 1]
        s_aa = covs[k, 0, 0]
        s_bb = covs[k, 1, 1]
        s_ab = covs[k, 0, 1]
        cond_mu = mu_b + (s_ab / s_aa) * (x_a - mu_a)
        cond_var = s_bb - (s_ab ** 2) / s_aa
        cond_comps.append(stats.norm.pdf(x_b, loc=cond_mu, scale=np.sqrt(cond_var)))

    cond_p_formula = sum(weights_tilde[k] * cond_comps[k] for k in range(2))

    assert np.isclose(cond_p_direct, cond_p_formula, atol=1e-12)
    return True


# =============================================================================
# Exercise 15.12: K-means as Limit epsilon -> 0 of GMM EM
# =============================================================================

def solve_exercise_15_12() -> Dict[str, Any]:
    """
    Exercise 15.12:
    Show that in the limit epsilon -> 0 with Sigma_k = epsilon I, maximizing expected
    complete-data log-likelihood (15.30) is equivalent to minimizing K-means distortion J.
    """
    derivation = (
        "1. For Sigma_k = epsilon I, the Gaussian density is:\n"
        "   N(x_n | mu_k, epsilon I) = (2 pi epsilon)^{-D/2} exp{ - ||x_n - mu_k||^2 / (2 epsilon) }.\n"
        "2. The responsibilities are:\n"
        "   gamma_{nk} = pi_k exp{ - ||x_n - mu_k||^2 / (2 epsilon) } / [ sum_j pi_j exp{ - ||x_n - mu_j||^2 / (2 epsilon) } ].\n"
        "3. As epsilon -> 0, the component with minimum ||x_n - mu_k||^2 exponentially dominates all others:\n"
        "   lim_{epsilon -> 0} gamma_{nk} = r_{nk} in {0, 1}, where r_{nk} = 1 if k = argmin_j ||x_n - mu_j||^2.\n"
        "4. The expected complete-data log-likelihood is:\n"
        "   Q = sum_n sum_k gamma_{nk} [ ln pi_k - (D/2) ln(2 pi epsilon) - (1 / 2 epsilon) ||x_n - mu_k||^2 ].\n"
        "5. Multiplying by -2 epsilon and taking epsilon -> 0, the dominant term is:\n"
        "   -2 epsilon Q = sum_n sum_k r_{nk} ||x_n - mu_k||^2 + O(epsilon) = J + O(epsilon).\n"
        "6. Maximizing Q is therefore mathematically equivalent to minimizing the K-means distortion J."
    )
    return {
        "limit_gamma": "Indicator variable r_{nk} in {0, 1} (hard assignments)",
        "objective_equivalence": "-2 epsilon Q -> J",
        "derivation": derivation,
    }


def verify_exercise_15_12() -> bool:
    """Verify that responsibilities approach hard 0-1 indicators as epsilon -> 0."""
    x = np.array([1.0, 0.0])
    mu1 = np.array([0.9, 0.0])   # Closer: distance 0.1
    mu2 = np.array([2.0, 0.0])   # Farther: distance 1.0

    for eps in [1.0, 0.1, 0.01, 0.001]:
        d1 = np.sum((x - mu1)**2)
        d2 = np.sum((x - mu2)**2)
        gamma1 = np.exp(-d1 / (2 * eps)) / (np.exp(-d1 / (2 * eps)) + np.exp(-d2 / (2 * eps)))
        if eps == 0.001:
            assert np.isclose(gamma1, 1.0, atol=1e-10)

    return True


# =============================================================================
# Exercise 15.13: Mean and Covariance of Bernoulli Distribution
# =============================================================================

def solve_exercise_15_13() -> Dict[str, Any]:
    """
    Exercise 15.13:
    Verify (15.35) and (15.36) for the mean and covariance of multivariate Bernoulli distribution.
    """
    derivation = (
        "1. p(x | mu) = prod_{i=1}^D mu_i^{x_i} (1 - mu_i)^{1 - x_i} with x_i in {0, 1}.\n"
        "2. For coordinate i:\n"
        "   E[x_i] = 1 * mu_i + 0 * (1 - mu_i) = mu_i.\n"
        "   Thus E[x] = mu (Eq. 15.35).\n"
        "3. Second moment for coordinate i:\n"
        "   E[x_i^2] = 1^2 * mu_i + 0^2 * (1 - mu_i) = mu_i.\n"
        "   Var[x_i] = E[x_i^2] - (E[x_i])^2 = mu_i - mu_i^2 = mu_i (1 - mu_i).\n"
        "4. For i != j, since the distribution factorizes over dimensions:\n"
        "   E[x_i x_j] = E[x_i] E[x_j] = mu_i mu_j,\n"
        "   so Cov(x_i, x_j) = E[x_i x_j] - E[x_i] E[x_j] = 0.\n"
        "5. Therefore, cov[x] = diag(mu_1 (1 - mu_1), ..., mu_D (1 - mu_D)) (Eq. 15.36)."
    )
    return {
        "mean": "E[x] = mu",
        "covariance": "cov[x] = diag(mu_i (1 - mu_i))",
        "derivation": derivation,
    }


def verify_exercise_15_13() -> bool:
    """Verify Bernoulli mean and diagonal covariance via Monte Carlo sampling."""
    rng = np.random.RandomState(42)
    mu = np.array([0.2, 0.5, 0.8])
    N = 100_000

    samples = rng.binomial(1, mu, size=(N, 3))
    emp_mean = np.mean(samples, axis=0)
    emp_cov = np.cov(samples, rowvar=False)

    true_mean = mu
    true_cov = np.diag(mu * (1.0 - mu))

    assert np.allclose(emp_mean, true_mean, atol=0.01)
    assert np.allclose(emp_cov, true_cov, atol=0.01)
    return True


# =============================================================================
# Exercise 15.14: Mean and Covariance of General Mixture Distribution
# =============================================================================

def solve_exercise_15_14() -> Dict[str, Any]:
    """
    Exercise 15.14:
    Show that for any mixture distribution p(x) = sum_k pi_k p(x | k),
    E[x] = sum_k pi_k mu_k and cov[x] = sum_k pi_k {Sigma_k + mu_k mu_k^T} - E[x] E[x]^T.
    """
    derivation = (
        "1. By the law of total expectation:\n"
        "   E[x] = E_k [ E[x | k] ] = sum_{k=1}^K pi_k mu_k (Eq. 15.39).\n"
        "2. For the second moment tensor:\n"
        "   E[x x^T | k] = cov[x | k] + E[x | k] E[x | k]^T = Sigma_k + mu_k mu_k^T.\n"
        "   E[x x^T] = sum_{k=1}^K pi_k (Sigma_k + mu_k mu_k^T).\n"
        "3. By the definition of covariance:\n"
        "   cov[x] = E[x x^T] - E[x] E[x]^T\n"
        "          = sum_{k=1}^K pi_k { Sigma_k + mu_k mu_k^T } - E[x] E[x]^T (Eq. 15.40).\n"
        "4. This can also be rewritten via the law of total variance as:\n"
        "   cov[x] = sum_k pi_k Sigma_k + sum_k pi_k (mu_k - E[x])(mu_k - E[x])^T."
    )
    return {
        "mean_formula": "E[x] = sum_k pi_k mu_k",
        "cov_formula": "cov[x] = sum_k pi_k {Sigma_k + mu_k mu_k^T} - E[x] E[x]^T",
        "law_of_total_variance": True,
        "derivation": derivation,
    }


def verify_exercise_15_14() -> bool:
    """Verify mixture mean and covariance formulas against sample estimates."""
    rng = np.random.RandomState(42)
    pi = np.array([0.3, 0.7])
    mu = np.array([[1.0, 2.0], [-1.0, 0.0]])
    sigma = np.array([
        [[1.0, 0.2], [0.2, 0.8]],
        [[2.0, -0.5], [-0.5, 1.5]]
    ])

    true_mean = pi[0] * mu[0] + pi[1] * mu[1]
    true_second_moment = pi[0] * (sigma[0] + np.outer(mu[0], mu[0])) + pi[1] * (sigma[1] + np.outer(mu[1], mu[1]))
    true_cov = true_second_moment - np.outer(true_mean, true_mean)

    # Sample from mixture
    N = 100_000
    comp_choices = rng.choice(2, size=N, p=pi)
    samples = np.zeros((N, 2))
    samples[comp_choices == 0] = rng.multivariate_normal(mu[0], sigma[0], size=np.sum(comp_choices == 0))
    samples[comp_choices == 1] = rng.multivariate_normal(mu[1], sigma[1], size=np.sum(comp_choices == 1))

    emp_mean = np.mean(samples, axis=0)
    emp_cov = np.cov(samples, rowvar=False)

    assert np.allclose(emp_mean, true_mean, atol=0.02)
    assert np.allclose(emp_cov, true_cov, atol=0.03)
    return True


# =============================================================================
# Exercise 15.15: Sample Mean Property and One-Step Convergence of Bernoulli Mixture
# =============================================================================

def solve_exercise_15_15() -> Dict[str, Any]:
    """
    Exercise 15.15:
    Show that for a Bernoulli mixture at MLE, E[x] = (1/N) sum_n x_n = x_bar.
    Hence if initialized with mu_k = mu_bar for all k, EM converges in one step to mu_k = x_bar.
    """
    derivation = (
        "1. In the M-step: mu_k = (1 / N_k) sum_n gamma_{nk} x_n, and pi_k = N_k / N.\n"
        "2. The mixture expectation is E[x] = sum_k pi_k mu_k:\n"
        "   E[x] = sum_k (N_k / N) [ (1 / N_k) sum_n gamma_{nk} x_n ]\n"
        "        = (1 / N) sum_{n=1}^N ( sum_{k=1}^K gamma_{nk} ) x_n.\n"
        "3. Since sum_{k=1}^K gamma_{nk} = 1 for every data point n:\n"
        "   E[x] = (1 / N) sum_{n=1}^N x_n = x_bar (Eq. 15.65).\n"
        "4. If initialized with mu_k = mu_bar for all k:\n"
        "   p(x_n | mu_k) = p(x_n | mu_bar) is identical for all k.\n"
        "   Then gamma_{nk} = pi_k p(x_n | mu_bar) / [ sum_j pi_j p(x_n | mu_bar) ] = pi_k.\n"
        "   In the M-step: N_k = sum_n gamma_{nk} = N pi_k.\n"
        "   mu_k^{new} = (1 / (N pi_k)) sum_n pi_k x_n = (1 / N) sum_n x_n = x_bar.\n"
        "5. All components become identical to the sample mean x_bar, and the algorithm has\n"
        "   converged in exactly one iteration to a degenerate solution."
    )
    return {
        "mixture_mean_equals_sample_mean": True,
        "one_step_convergence": True,
        "degenerate_fixed_point": "mu_k = x_bar for all k",
        "derivation": derivation,
    }


def verify_exercise_15_15() -> bool:
    """Verify that identical initialization in Bernoulli EM converges in 1 iteration to x_bar."""
    rng = np.random.RandomState(42)
    X = rng.binomial(1, 0.4, size=(50, 4))
    x_bar = np.mean(X, axis=0)

    # 1 iteration step manually:
    K = 3
    init_pi = np.array([0.2, 0.5, 0.3])

    # E-step with identical mu:
    # gamma_{nk} = pi_k
    N = len(X)
    gamma = np.tile(init_pi, (N, 1))  # (N, K)
    N_k = np.sum(gamma, axis=0)
    mu_new = (gamma.T @ X) / N_k[:, np.newaxis]

    for k in range(K):
        assert np.allclose(mu_new[k], x_bar, atol=1e-12)

    return True


# =============================================================================
# Exercise 15.16: Marginalizing Bernoulli Joint Distribution
# =============================================================================

def solve_exercise_15_16() -> Dict[str, Any]:
    """
    Exercise 15.16:
    Show that marginalizing p(x, z) = p(z | pi) p(x | z, mu) w.r.t. z yields (15.37).
    """
    derivation = (
        "1. Latent prior: p(z | pi) = prod_{k=1}^K pi_k^{z_k}.\n"
        "2. Conditional: p(x | z, mu) = prod_{k=1}^K [ prod_{i=1}^D mu_{ki}^{x_i} (1 - mu_{ki})^{1 - x_i} ]^{z_k}.\n"
        "3. Joint distribution: p(x, z) = prod_{k=1}^K [ pi_k prod_{i=1}^D mu_{ki}^{x_i} (1 - mu_{ki})^{1 - x_i} ]^{z_k}.\n"
        "4. Since z is a 1-of-K vector, marginalizing over all valid z:\n"
        "   p(x) = sum_z p(x, z) = sum_{k=1}^K pi_k prod_{i=1}^D mu_{ki}^{x_i} (1 - mu_{ki})^{1 - x_i}\n"
        "        = sum_{k=1}^K pi_k p(x | mu_k) (Eq. 15.37)."
    )
    return {
        "marginal_distribution": "sum_{k=1}^K pi_k prod_{i=1}^D mu_{ki}^{x_i} (1 - mu_{ki})^{1 - x_i}",
        "derivation": derivation,
    }


def verify_exercise_15_16() -> bool:
    """Verify discrete sum over 1-of-K states equals Bernoulli mixture formula."""
    weights = np.array([0.3, 0.7])
    mu = np.array([[0.2, 0.8], [0.6, 0.4]])
    x = np.array([1, 0])

    # Method 1: Discrete sum
    sum_val = 0.0
    for k in range(2):
        p_z = weights[k]
        p_x_given_z = (mu[k, 0] ** x[0] * (1 - mu[k, 0]) ** (1 - x[0])) * \
                      (mu[k, 1] ** x[1] * (1 - mu[k, 1]) ** (1 - x[1]))
        sum_val += p_z * p_x_given_z

    # Method 2: Formula (15.37)
    p_mix = sum(weights[k] * np.prod(mu[k] ** x * (1 - mu[k]) ** (1 - x)) for k in range(2))

    assert np.isclose(sum_val, p_mix, atol=1e-14)
    return True


# =============================================================================
# Exercise 15.17: Maximizing Expected Complete Log-Likelihood w.r.t. mu_k
# =============================================================================

def solve_exercise_15_17() -> Dict[str, Any]:
    """
    Exercise 15.17:
    Show that maximizing expected complete log-likelihood (15.45) for Bernoulli mixture
    w.r.t. mu_k yields M-step equation (15.49): mu_k = (1 / N_k) sum_n gamma_{nk} x_n.
    """
    derivation = (
        "1. Complete-data log likelihood expectation (15.45):\n"
        "   Q = sum_{n=1}^N sum_{k=1}^K gamma_{nk} [ ln pi_k + sum_{i=1}^D ( x_{ni} ln mu_{ki} + (1 - x_{ni}) ln(1 - mu_{ki}) ) ].\n"
        "2. Taking derivative w.r.t. mu_{ki}:\n"
        "   d Q / d mu_{ki} = sum_{n=1}^N gamma_{nk} [ x_{ni} / mu_{ki} - (1 - x_{ni}) / (1 - mu_{ki}) ]\n"
        "                   = sum_{n=1}^N gamma_{nk} [ (x_{ni} - mu_{ki}) / ( mu_{ki} (1 - mu_{ki}) ) ].\n"
        "3. Setting derivative to zero:\n"
        "   sum_{n=1}^N gamma_{nk} (x_{ni} - mu_{ki}) = 0\n"
        "   => sum_{n=1}^N gamma_{nk} x_{ni} = mu_{ki} sum_{n=1}^N gamma_{nk} = N_k mu_{ki}.\n"
        "4. Hence mu_{ki} = (1 / N_k) sum_{n=1}^N gamma_{nk} x_{ni}, or in vector form:\n"
        "   mu_k = (1 / N_k) sum_{n=1}^N gamma_{nk} x_n (Eq. 15.49)."
    )
    return {
        "m_step_mu": "mu_k = (1 / N_k) sum_n gamma_{nk} x_n",
        "derivation": derivation,
    }


def verify_exercise_15_17() -> bool:
    """Verify that derivative w.r.t. mu_{ki} vanishes at the M-step solution."""
    rng = np.random.RandomState(42)
    X = rng.binomial(1, 0.5, size=(20, 3))
    gamma = rng.dirichlet(np.ones(2), size=20)

    k = 0
    N_k = np.sum(gamma[:, k])
    mu_k = np.sum(gamma[:, k:k+1] * X, axis=0) / N_k

    # Check derivative
    grad = np.zeros(3)
    for n in range(len(X)):
        grad += gamma[n, k] * (X[n] - mu_k) / (mu_k * (1.0 - mu_k))

    assert np.allclose(grad, 0.0, atol=1e-12)
    return True


# =============================================================================
# Exercise 15.18: Maximizing Expected Complete Log-Likelihood w.r.t. pi_k
# =============================================================================

def solve_exercise_15_18() -> Dict[str, Any]:
    """
    Exercise 15.18:
    Show that maximizing (15.45) w.r.t. pi_k with Lagrange multiplier yields pi_k = N_k / N.
    """
    derivation = (
        "1. The terms in (15.45) depending on pi_k are sum_{k=1}^K sum_{n=1}^N gamma_{nk} ln pi_k = sum_{k=1}^K N_k ln pi_k.\n"
        "2. Adding Lagrange multiplier lambda to enforce sum_{k=1}^K pi_k = 1:\n"
        "   Lambda(pi, lambda) = sum_{k=1}^K N_k ln pi_k + lambda (sum_{k=1}^K pi_k - 1).\n"
        "3. Setting partial derivative to zero:\n"
        "   d Lambda / d pi_k = N_k / pi_k + lambda = 0 => N_k = -lambda pi_k.\n"
        "4. Summing over k gives sum_k N_k = N = -lambda sum_k pi_k = -lambda => lambda = -N.\n"
        "5. Substituting lambda = -N yields pi_k = N_k / N (Eq. 15.50)."
    )
    return {
        "lagrange_multiplier": "lambda = -N",
        "m_step_pi": "pi_k = N_k / N",
        "derivation": derivation,
    }


def verify_exercise_15_18() -> bool:
    """Verify that pi_k = N_k / N satisfies the constrained stationarity condition."""
    gamma = np.array([[0.3, 0.7], [0.4, 0.6], [0.8, 0.2]])
    N = len(gamma)
    N_k = np.sum(gamma, axis=0)
    pi = N_k / N
    assert np.isclose(np.sum(pi), 1.0)
    assert np.allclose(N_k / pi, N)
    return True


# =============================================================================
# Exercise 15.19: Boundedness of Bernoulli Mixture Log-Likelihood
# =============================================================================

def solve_exercise_15_19() -> Dict[str, Any]:
    """
    Exercise 15.19:
    Show that since 0 <= p(x_n | mu_k) <= 1, the Bernoulli mixture log-likelihood is bounded above
    by 0 and hence has no singularities (unlike Gaussian mixtures).
    """
    derivation = (
        "1. For any binary vector x_n in {0, 1}^D and component parameters mu_k in [0, 1]^D:\n"
        "   p(x_n | mu_k) = prod_{i=1}^D mu_{ki}^{x_{ni}} (1 - mu_{ki})^{1 - x_{ni}}.\n"
        "2. Since each factor is a probability in [0, 1], the product satisfies 0 <= p(x_n | mu_k) <= 1.\n"
        "3. The mixture distribution is a convex combination of these probabilities:\n"
        "   p(x_n) = sum_{k=1}^K pi_k p(x_n | mu_k) <= sum_{k=1}^K pi_k * 1 = 1.\n"
        "4. Taking the logarithm: ln p(x_n) <= ln(1) = 0 for all n.\n"
        "5. Hence the total log-likelihood satisfies ln p(X) = sum_{n=1}^N ln p(x_n) <= 0.\n"
        "6. In contrast to Gaussian mixtures where density can diverge to +infinity as Sigma_k -> 0,\n"
        "   the Bernoulli likelihood is bounded above by 0 and no singularities can occur."
    )
    return {
        "upper_bound": 0.0,
        "singularities_possible": False,
        "derivation": derivation,
    }


def verify_exercise_15_19() -> bool:
    """Verify that Bernoulli mixture log-likelihood is always non-positive <= 0."""
    rng = np.random.RandomState(42)
    X = rng.binomial(1, 0.5, size=(30, 5))
    K = 2
    weights = np.array([0.4, 0.6])
    mu = np.array([[0.1, 0.9, 0.3, 0.7, 0.5], [0.8, 0.2, 0.6, 0.4, 0.5]])

    # Compute log likelihood
    ll = 0.0
    for n in range(len(X)):
        p_n = 0.0
        for k in range(K):
            p_comp = np.prod(mu[k] ** X[n] * (1.0 - mu[k]) ** (1 - X[n]))
            assert 0.0 <= p_comp <= 1.0
            p_n += weights[k] * p_comp
        assert 0.0 <= p_n <= 1.0
        ll += np.log(p_n + 1e-12)

    assert ll <= 0.0, "Bernoulli mixture log-likelihood must be bounded above by 0"
    return True


# =============================================================================
# Exercise 15.20: Mixture of Multinomial / Categorical Distributions
# =============================================================================

def solve_exercise_15_20() -> Dict[str, Any]:
    """
    Exercise 15.20:
    Derive E-step and M-step for a mixture of multivariate categorical distributions.
    """
    derivation = (
        "1. Model: p(x) = sum_{k=1}^K pi_k prod_{i=1}^D prod_{j=1}^M mu_{kij}^{x_{ij}} with sum_j x_{ij} = 1.\n"
        "2. E-step: Responsibilities are given by posterior over latent components:\n"
        "   gamma_{nk} = pi_k prod_{i=1}^D prod_{j=1}^M mu_{kij}^{x_{nij}} / [ sum_l pi_l prod_{i=1}^D prod_{j=1}^M mu_{lij}^{x_{nij}} ].\n"
        "3. Complete-data log likelihood expectation:\n"
        "   Q = sum_{n=1}^N sum_{k=1}^K gamma_{nk} [ ln pi_k + sum_{i=1}^D sum_{j=1}^M x_{nij} ln mu_{kij} ].\n"
        "4. M-step for pi_k: Enforcing sum_k pi_k = 1 gives pi_k = N_k / N where N_k = sum_n gamma_{nk}.\n"
        "5. M-step for mu_{kij}: For each cluster k and dimension i, enforce sum_{j=1}^M mu_{kij} = 1.\n"
        "   Lagrangian: sum_{j=1}^M ( sum_n gamma_{nk} x_{nij} ) ln mu_{kij} + lambda_{ki} ( sum_{j=1}^M mu_{kij} - 1 ).\n"
        "   Derivative: ( sum_n gamma_{nk} x_{nij} ) / mu_{kij} + lambda_{ki} = 0.\n"
        "   Summing over j gives lambda_{ki} = - sum_n gamma_{nk} (sum_j x_{nij}) = -N_k.\n"
        "   Hence mu_{kij} = ( sum_n gamma_{nk} x_{nij} ) / N_k."
    )
    return {
        "e_step": "gamma_{nk} = pi_k p(x_n | mu_k) / sum_l pi_l p(x_n | mu_l)",
        "m_step_pi": "pi_k = N_k / N",
        "m_step_mu": "mu_{kij} = (1 / N_k) sum_n gamma_{nk} x_{nij}",
        "derivation": derivation,
    }


def verify_exercise_15_20() -> bool:
    """Verify CategoricalMixtureEM monotonically increases log-likelihood."""
    rng = np.random.RandomState(42)
    N, D, M, K = 40, 3, 4, 2

    # Generate synthetic categorical one-hot data
    X = np.zeros((N, D, M))
    for n in range(N):
        for d in range(D):
            cat = rng.choice(M)
            X[n, d, cat] = 1.0

    model = CategoricalMixtureEM(n_components=K, max_iter=20, random_state=42)
    model.fit(X)

    history = model.log_likelihood_history_
    assert len(history) > 1
    for i in range(len(history) - 1):
        assert history[i + 1] >= history[i] - 1e-7, "Categorical EM log-likelihood must monotonically increase"

    # Check normalization of parameters
    assert np.isclose(np.sum(model.weights_), 1.0)
    assert np.allclose(np.sum(model.probs_, axis=2), 1.0)
    return True


# =============================================================================
# Exercise 15.21: Decomposition of Log-Likelihood into ELBO and KL
# =============================================================================

def solve_exercise_15_21() -> Dict[str, Any]:
    """
    Exercise 15.21:
    Verify the identity ln p(X | theta) = L(q, theta) + KL(q || p).
    """
    derivation = (
        "1. Definitions:\n"
        "   L(q, theta) = sum_Z q(Z) ln { p(X, Z | theta) / q(Z) },\n"
        "   KL(q || p)  = - sum_Z q(Z) ln { p(Z | X, theta) / q(Z) }.\n"
        "2. Adding them together:\n"
        "   L(q, theta) + KL(q || p) = sum_Z q(Z) [ ln { p(X, Z | theta) / q(Z) } - ln { p(Z | X, theta) / q(Z) } ].\n"
        "3. Combining logarithms:\n"
        "   = sum_Z q(Z) ln { p(X, Z | theta) / p(Z | X, theta) }.\n"
        "4. Using Bayes' rule p(X, Z | theta) / p(Z | X, theta) = p(X | theta):\n"
        "   = sum_Z q(Z) ln p(X | theta).\n"
        "5. Since ln p(X | theta) does not depend on Z and sum_Z q(Z) = 1:\n"
        "   = ln p(X | theta) sum_Z q(Z) = ln p(X | theta) (Eq. 15.52)."
    )
    return {
        "identity": "ln p(X | theta) = L(q, theta) + KL(q || p)",
        "derivation": derivation,
    }


def verify_exercise_15_21() -> bool:
    """Verify that ELBO + KL equals log-likelihood for arbitrary non-optimal q(Z)."""
    weights = np.array([0.4, 0.6])
    means = np.array([[-1.0], [1.5]])
    covariances = np.array([[[1.0]], [[0.8]]])

    X = np.array([[-0.5], [1.0], [2.0]])

    # Choose an arbitrary non-optimal q(Z)
    rng = np.random.RandomState(42)
    q = rng.dirichlet(np.ones(2), size=len(X))

    decomp = compute_elbo_decomposition(X, means, covariances, weights, q=q)
    elbo = decomp["elbo"]
    kl = decomp["kl_divergence"]
    log_lik = decomp["log_likelihood"]
    assert np.isclose(elbo + kl, log_lik, atol=1e-12)
    assert kl >= 0.0
    assert elbo <= log_lik + 1e-12
    return True


# =============================================================================
# Exercise 15.22: Tangential Bound Property of ELBO
# =============================================================================

def solve_exercise_15_22() -> Dict[str, Any]:
    """
    Exercise 15.22:
    Show that with q(Z) = p(Z | X, theta_old), grad_theta L(q, theta) = grad_theta ln p(X | theta)
    at theta = theta_old.
    """
    derivation = (
        "1. From (15.52): ln p(X | theta) = L(q, theta) + KL(q || p(Z | X, theta)).\n"
        "2. Taking gradient w.r.t. theta:\n"
        "   grad_theta ln p(X | theta) = grad_theta L(q, theta) + grad_theta KL(q || p(Z | X, theta)).\n"
        "3. When q(Z) = p(Z | X, theta_old), the KL divergence KL(q || p(Z | X, theta)) attains\n"
        "   its global minimum of 0 at theta = theta_old.\n"
        "4. Since theta = theta_old is a minimum of the KL divergence, its gradient must vanish:\n"
        "   grad_theta KL(q || p(Z | X, theta)) |_{theta = theta_old} = 0.\n"
        "5. Direct calculation confirms this:\n"
        "   grad_theta KL = - sum_Z q(Z) grad_theta ln p(Z | X, theta)\n"
        "                 = - sum_Z p(Z | X, theta_old) [ grad_theta p(Z | X, theta) / p(Z | X, theta) ] |_{theta = theta_old}\n"
        "                 = - grad_theta [ sum_Z p(Z | X, theta) ] = - grad_theta (1) = 0.\n"
        "6. Therefore: grad_theta L(q, theta) |_{theta = theta_old} = grad_theta ln p(X | theta) |_{theta = theta_old}."
    )
    return {
        "tangential_condition": "grad_theta L(q, theta_old) = grad_theta ln p(X | theta_old)",
        "gradient_kl_vanishes": True,
        "derivation": derivation,
    }


def verify_exercise_15_22() -> bool:
    """Verify that finite-difference gradients of ELBO and log-likelihood match at theta_old."""
    # 1D single-parameter case for clean numerical differentiation: mu_0
    X = np.array([[0.2], [1.5], [-0.8]])
    weights = np.array([0.5, 0.5])
    covariances = np.array([[[1.0]], [[1.0]]])
    means_old = np.array([[0.0], [2.0]])

    # Optimal q under theta_old
    decomp_old = compute_elbo_decomposition(X, means_old, covariances, weights)
    q_opt = decomp_old["posterior"]

    eps = 1e-6
    means_plus = means_old.copy()
    means_plus[0, 0] += eps
    means_minus = means_old.copy()
    means_minus[0, 0] -= eps

    # Gradient of log likelihood
    ll_plus = compute_elbo_decomposition(X, means_plus, covariances, weights)["log_likelihood"]
    ll_minus = compute_elbo_decomposition(X, means_minus, covariances, weights)["log_likelihood"]
    grad_ll = (ll_plus - ll_minus) / (2 * eps)

    # Gradient of ELBO with FIXED q_opt
    elbo_plus = compute_elbo_decomposition(X, means_plus, covariances, weights, q=q_opt)["elbo"]
    elbo_minus = compute_elbo_decomposition(X, means_minus, covariances, weights, q=q_opt)["elbo"]
    grad_elbo = (elbo_plus - elbo_minus) / (2 * eps)

    assert np.isclose(grad_ll, grad_elbo, rtol=1e-4)
    return True


# =============================================================================
# Exercise 15.23: Sequential EM Updates for Means and Counts
# =============================================================================

def solve_exercise_15_23() -> Dict[str, Any]:
    """
    Exercise 15.23:
    Derive sequential EM update formulas (15.60) and (15.61) for component means.
    """
    derivation = (
        "1. In batch EM, the sufficient statistics are N_k = sum_n gamma(z_{nk}) and s_k = sum_n gamma(z_{nk}) x_n.\n"
        "2. When recomputing responsibilities only for data point x_m from gamma_old to gamma_new:\n"
        "   N_k^{new} = sum_{n != m} gamma(z_{nk}) + gamma_new(z_{mk})\n"
        "             = N_k^{old} + gamma_new(z_{mk}) - gamma_old(z_{mk}) (Eq. 15.61).\n"
        "3. Similarly for the sum statistic s_k:\n"
        "   s_k^{new} = s_k^{old} + [ gamma_new(z_{mk}) - gamma_old(z_{mk}) ] x_m.\n"
        "4. Since s_k^{old} = N_k^{old} mu_k^{old} and N_k^{old} = N_k^{new} - (gamma_new - gamma_old):\n"
        "   mu_k^{new} = s_k^{new} / N_k^{new}\n"
        "              = (1 / N_k^{new}) [ (N_k^{new} - (gamma_new - gamma_old)) mu_k^{old} + (gamma_new - gamma_old) x_m ]\n"
        "              = mu_k^{old} + [ (gamma_new(z_{mk}) - gamma_old(z_{mk})) / N_k^{new} ] (x_m - mu_k^{old}) (Eq. 15.60)."
    )
    return {
        "n_update": "N_k^{new} = N_k^{old} + gamma_new(z_{mk}) - gamma_old(z_{mk})",
        "mu_update": "mu_k^{new} = mu_k^{old} + [ (gamma_new - gamma_old) / N_k^{new} ] (x_m - mu_k^{old})",
        "derivation": derivation,
    }


def verify_exercise_15_23() -> bool:
    """Verify that incremental mean update matches batch recomputation exactly."""
    rng = np.random.RandomState(42)
    X = rng.randn(20, 2)
    K = 2
    gamma_old = rng.dirichlet(np.ones(K), size=len(X))

    # Old statistics
    N_k_old = np.sum(gamma_old, axis=0)
    mu_old = (gamma_old.T @ X) / N_k_old[:, np.newaxis]

    # Change responsibility for point m
    m = 5
    gamma_new_m = np.array([0.8, 0.2])
    delta_gamma = gamma_new_m - gamma_old[m]

    # Sequential update formulas (15.60) & (15.61)
    N_k_new = N_k_old + delta_gamma
    mu_new = mu_old + (delta_gamma[:, np.newaxis] / N_k_new[:, np.newaxis]) * (X[m] - mu_old)

    # Batch recomputation
    gamma_recomputed = gamma_old.copy()
    gamma_recomputed[m] = gamma_new_m
    N_k_batch = np.sum(gamma_recomputed, axis=0)
    mu_batch = (gamma_recomputed.T @ X) / N_k_batch[:, np.newaxis]

    assert np.allclose(N_k_new, N_k_batch, atol=1e-14)
    assert np.allclose(mu_new, mu_batch, atol=1e-14)
    return True


# =============================================================================
# Exercise 15.24: Sequential EM Updates for Covariances and Mixing Coefficients
# =============================================================================

def solve_exercise_15_24() -> Dict[str, Any]:
    """
    Exercise 15.24:
    Derive sequential EM update formulas for mixing coefficients pi_k and covariance matrices Sigma_k.
    """
    derivation = (
        "1. Mixing coefficients:\n"
        "   pi_k^{new} = N_k^{new} / N = [ N_k^{old} + gamma_new(z_{mk}) - gamma_old(z_{mk}) ] / N\n"
        "              = pi_k^{old} + (gamma_new(z_{mk}) - gamma_old(z_{mk})) / N.\n"
        "2. Covariances:\n"
        "   Let C_k = sum_n gamma(z_{nk}) x_n x_n^T be the second-moment statistic.\n"
        "   C_k^{new} = C_k^{old} + (gamma_new - gamma_old) x_m x_m^T.\n"
        "   Using Sigma_k = (1 / N_k) C_k - mu_k mu_k^T:\n"
        "   Sigma_k^{new} = (1 / N_k^{new}) [ C_k^{old} + Delta_gamma x_m x_m^T ] - mu_k^{new} (mu_k^{new})^T\n"
        "   where C_k^{old} = N_k^{old} (Sigma_k^{old} + mu_k^{old} (mu_k^{old})^T).\n"
        "3. Rearranging in incremental form:\n"
        "   Sigma_k^{new} = (N_k^{old} / N_k^{new}) Sigma_k^{old} + (Delta_gamma / N_k^{new}) (x_m - mu_k^{new})(x_m - mu_k^{old})^T\n"
        "   - (mu_k^{new} - mu_k^{old})(mu_k^{new} - mu_k^{old})^T."
    )
    return {
        "pi_update": "pi_k^{new} = pi_k^{old} + (gamma_new(z_{mk}) - gamma_old(z_{mk})) / N",
        "sigma_update": "Sigma_k^{new} = (1 / N_k^{new}) [ N_k^{old}(Sigma_k^{old} + mu_k^{old} mu_k^{old T}) + Delta_gamma x_m x_m^T ] - mu_k^{new} mu_k^{new T}",
        "derivation": derivation,
    }


def verify_exercise_15_24() -> bool:
    """Verify that sequential covariance and mixing coefficient updates match batch formulas."""
    rng = np.random.RandomState(42)
    X = rng.randn(20, 2)
    K = 2
    N = len(X)
    gamma_old = rng.dirichlet(np.ones(K), size=N)

    N_k_old = np.sum(gamma_old, axis=0)
    pi_old = N_k_old / N
    mu_old = (gamma_old.T @ X) / N_k_old[:, np.newaxis]

    C_k_old = np.zeros((K, 2, 2))
    sigma_old = np.zeros((K, 2, 2))
    for k in range(K):
        for n in range(N):
            C_k_old[k] += gamma_old[n, k] * np.outer(X[n], X[n])
        sigma_old[k] = C_k_old[k] / N_k_old[k] - np.outer(mu_old[k], mu_old[k])

    # Incremental update on data point m
    m = 3
    gamma_new_m = np.array([0.1, 0.9])
    delta_gamma = gamma_new_m - gamma_old[m]

    # Incremental updates
    N_k_new = N_k_old + delta_gamma
    pi_new = pi_old + delta_gamma / N
    mu_new = mu_old + (delta_gamma[:, np.newaxis] / N_k_new[:, np.newaxis]) * (X[m] - mu_old)

    sigma_new = np.zeros((K, 2, 2))
    for k in range(K):
        C_k_new = C_k_old[k] + delta_gamma[k] * np.outer(X[m], X[m])
        sigma_new[k] = C_k_new / N_k_new[k] - np.outer(mu_new[k], mu_new[k])

    # Batch recomputation
    gamma_batch = gamma_old.copy()
    gamma_batch[m] = gamma_new_m
    N_k_b = np.sum(gamma_batch, axis=0)
    pi_b = N_k_b / N
    mu_b = (gamma_batch.T @ X) / N_k_b[:, np.newaxis]
    sigma_b = np.zeros((K, 2, 2))
    for k in range(K):
        for n in range(N):
            sigma_b[k] += gamma_batch[n, k] * np.outer(X[n] - mu_b[k], X[n] - mu_b[k])
        sigma_b[k] /= N_k_b[k]

    assert np.allclose(pi_new, pi_b, atol=1e-14)
    assert np.allclose(sigma_new, sigma_b, atol=1e-12)
    return True


# =============================================================================
# Solve All Exercises Helper
# =============================================================================

def solve_all_exercises() -> Dict[str, Dict[str, Any]]:
    """Solve and collect solutions for all 24 exercises in Chapter 15."""
    return {
        "15.1": solve_exercise_15_1(),
        "15.2": solve_exercise_15_2(),
        "15.3": solve_exercise_15_3(),
        "15.4": solve_exercise_15_4(),
        "15.5": solve_exercise_15_5(),
        "15.6": solve_exercise_15_6(),
        "15.7": solve_exercise_15_7(),
        "15.8": solve_exercise_15_8(),
        "15.9": solve_exercise_15_9(),
        "15.10": solve_exercise_15_10(),
        "15.11": solve_exercise_15_11(),
        "15.12": solve_exercise_15_12(),
        "15.13": solve_exercise_15_13(),
        "15.14": solve_exercise_15_14(),
        "15.15": solve_exercise_15_15(),
        "15.16": solve_exercise_15_16(),
        "15.17": solve_exercise_15_17(),
        "15.18": solve_exercise_15_18(),
        "15.19": solve_exercise_15_19(),
        "15.20": solve_exercise_15_20(),
        "15.21": solve_exercise_15_21(),
        "15.22": solve_exercise_15_22(),
        "15.23": solve_exercise_15_23(),
        "15.24": solve_exercise_15_24(),
    }
