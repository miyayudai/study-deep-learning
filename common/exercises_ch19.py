r"""Chapter 19 Exercises: Theory, Proofs, and Numerical Verifications (Bishop & Bishop, 2024).

This module implements solutions, numerical verifications, and assertions for:
- Exercise 19.1: REINFORCE / Score function estimator, unbiasedness proof, and variance comparison
- Exercise 19.2: Affine transformation of 1D Gaussian random variable and density transformation
- Exercise 19.3: Multivariate affine transformation, covariance factorization, and Cholesky parameterization
- Exercise 19.4: Analytical evaluation of Gaussian KL divergence, non-negativity, and exact gradients
- Exercise 19.5: Alternative decomposition of ELBO into expected joint log-likelihood and posterior entropy
- Exercise 19.6: Unconstrained optimal variational distribution recovering the true posterior and exact EM
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.stats import norm, kstest


# =====================================================================
# Exercise 19.1: Score Function / REINFORCE Estimator & Reparameterization
# =====================================================================

def score_function_gaussian_1d(
    z: np.ndarray,
    mu: float,
    sigma2: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute score function :math:`\nabla_\phi \ln q(z|\phi)` for 1D Gaussian.

    .. math::
        q(z|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(z - \mu)^2}{2\sigma^2}\right) \\
        \nabla_\mu \ln q(z|\mu, \sigma^2) = \frac{z - \mu}{\sigma^2} \\
        \nabla_{\ln \sigma^2} \ln q(z|\mu, \sigma^2) = -\frac{1}{2} + \frac{(z - \mu)^2}{2\sigma^2}
    """
    score_mu = (z - mu) / sigma2
    score_log_sigma2 = -0.5 + 0.5 * ((z - mu) ** 2) / sigma2
    return score_mu, score_log_sigma2


def reinforce_gradient_estimator_1d(
    mu: float,
    sigma2: float,
    G_func: Callable[[np.ndarray], np.ndarray],
    S: int = 10000,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    r"""Compute REINFORCE / score function gradient estimate of :math:`\mathbb{E}_{q}[G(z)]` w.r.t :math:`\mu`.

    .. math::
        \nabla_\mu \mathbb{E}_q[G(z)] = \mathbb{E}_q[G(z) \nabla_\mu \ln q(z)]
        \approx \frac{1}{S} \sum_{s=1}^S G(z^{(s)}) \frac{z^{(s)} - \mu}{\sigma^2}
    """
    rng = np.random.RandomState(random_state)
    sigma = np.sqrt(sigma2)
    z = rng.normal(mu, sigma, size=S)
    G_val = G_func(z)
    score_mu = (z - mu) / sigma2
    grad_samples = G_val * score_mu
    est_mean = float(np.mean(grad_samples))
    est_var = float(np.var(grad_samples))
    return est_mean, est_var, float(np.mean(score_mu))


def reparameterization_gradient_estimator_1d(
    mu: float,
    sigma2: float,
    grad_G_func: Callable[[np.ndarray], np.ndarray],
    S: int = 10000,
    random_state: int = 42,
) -> Tuple[float, float]:
    r"""Compute reparameterization gradient estimate of :math:`\mathbb{E}_q[G(z)]` w.r.t :math:`\mu`.

    .. math::
        z = \mu + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1) \\
        \nabla_\mu \mathbb{E}_q[G(z)] = \mathbb{E}_{\epsilon}[\nabla_z G(\mu + \sigma \epsilon) \cdot 1]
    """
    rng = np.random.RandomState(random_state)
    sigma = np.sqrt(sigma2)
    eps = rng.randn(S)
    z = mu + sigma * eps
    grad_samples = grad_G_func(z)
    est_mean = float(np.mean(grad_samples))
    est_var = float(np.var(grad_samples))
    return est_mean, est_var


def verify_exercise_19_1(
    mu: float = 2.0,
    sigma2: float = 1.5,
    S: int = 50000,
    random_state: int = 42,
) -> Dict[str, float]:
    r"""Exercise 19.1: Verify REINFORCE identity and compare variance to reparameterization trick.

    For test function :math:`G(z) = z^2`, we have:
    .. math::
        \mathbb{E}[z^2] = \mu^2 + \sigma^2 \implies \frac{\partial}{\partial \mu}\mathbb{E}[z^2] = 2\mu
    """
    true_grad_mu = 2.0 * mu

    G_func = lambda z: z ** 2
    grad_G_func = lambda z: 2.0 * z

    rf_mean, rf_var, score_mean = reinforce_gradient_estimator_1d(
        mu=mu, sigma2=sigma2, G_func=G_func, S=S, random_state=random_state
    )

    rp_mean, rp_var = reparameterization_gradient_estimator_1d(
        mu=mu, sigma2=sigma2, grad_G_func=grad_G_func, S=S, random_state=random_state
    )

    diff_rf = abs(rf_mean - true_grad_mu)
    diff_rp = abs(rp_mean - true_grad_mu)
    var_ratio = rf_var / (rp_var + 1e-12)

    return {
        "true_grad_mu": float(true_grad_mu),
        "reinforce_mean": float(rf_mean),
        "reparam_mean": float(rp_mean),
        "score_mean": float(score_mean),
        "diff_rf": float(diff_rf),
        "diff_rp": float(diff_rp),
        "reinforce_var": float(rf_var),
        "reparam_var": float(rp_var),
        "var_ratio": float(var_ratio),
    }


# =====================================================================
# Exercise 19.2: Affine Transformation of 1D Gaussian
# =====================================================================

def affine_transform_1d_density(
    z: np.ndarray,
    mu: float,
    sigma: float,
) -> np.ndarray:
    r"""Compute transformed density :math:`p_z(z) = p_\epsilon(g^{-1}(z)) |dg^{-1}/dz|`.

    .. math::
        \epsilon = g^{-1}(z) = \frac{z - \mu}{\sigma}, \quad \left|\frac{dg^{-1}}{dz}\right| = \frac{1}{\sigma} \\
        p_z(z) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{z - \mu}{\sigma}\right)^2\right) \frac{1}{\sigma}
               = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(z - \mu)^2}{2\sigma^2}\right)
    """
    eps = (z - mu) / sigma
    p_eps = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * eps ** 2)
    jacobian_inv = 1.0 / sigma
    return p_eps * jacobian_inv


def verify_exercise_19_2(
    mu: float = 3.5,
    sigma: float = 1.8,
    N: int = 50000,
    random_state: int = 42,
) -> Dict[str, float]:
    r"""Exercise 19.2: Verify affine transformation z = mu + sigma * eps produces N(mu, sigma^2)."""
    rng = np.random.RandomState(random_state)
    eps = rng.randn(N)
    z = mu + sigma * eps

    sample_mean = float(np.mean(z))
    sample_var = float(np.var(z, ddof=1))

    mean_err = abs(sample_mean - mu)
    var_err = abs(sample_var - sigma ** 2)

    # Check density identity across a grid
    grid = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    p_affine = affine_transform_1d_density(grid, mu, sigma)
    p_scipy = norm.pdf(grid, loc=mu, scale=sigma)
    max_density_diff = float(np.max(np.abs(p_affine - p_scipy)))

    # Kolmogorov-Smirnov test against target Gaussian
    ks_stat, ks_pval = kstest(z, 'norm', args=(mu, sigma))

    return {
        "sample_mean": sample_mean,
        "sample_var": sample_var,
        "mean_err": mean_err,
        "var_err": var_err,
        "max_density_diff": max_density_diff,
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval),
    }


# =====================================================================
# Exercise 19.3: Multivariate Affine Transformation and Cholesky Covariance
# =====================================================================

def multivariate_affine_transform_density(
    z: np.ndarray,
    mu: np.ndarray,
    L: np.ndarray,
) -> np.ndarray:
    r"""Compute multivariate density :math:`p_z(z) = p_\epsilon(L^{-1}(z - \mu)) |\det L|^{-1}`.

    .. math::
        \mathbf{z} = \boldsymbol{\mu} + \mathbf{L} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_D) \\
        \operatorname{cov}[\mathbf{z}] = \mathbf{L} \mathbf{L}^T = \mathbf{\Sigma}
    """
    D = len(mu)
    diff = z - mu
    # Solve L @ eps = diff
    eps = np.linalg.solve(L, diff.T).T  # (N, D)
    log_p_eps = -0.5 * D * np.log(2.0 * np.pi) - 0.5 * np.sum(eps ** 2, axis=-1)
    log_abs_det_L = np.sum(np.log(np.abs(np.diag(L)))) if np.allclose(L, np.tril(L)) else np.linalg.slogdet(L)[1]
    log_p_z = log_p_eps - log_abs_det_L
    return np.exp(log_p_z)


def verify_exercise_19_3(
    D: int = 3,
    N: int = 50000,
    random_state: int = 42,
) -> Dict[str, float]:
    r"""Exercise 19.3: Verify expectation and covariance of multivariate affine transformation."""
    rng = np.random.RandomState(random_state)
    mu = rng.randn(D)

    # Create a random positive-definite covariance matrix and Cholesky factor
    A = rng.randn(D, D)
    Sigma = A @ A.T + 0.5 * np.eye(D)
    L = np.linalg.cholesky(Sigma)

    # Transform standard normal vectors
    eps = rng.randn(N, D)
    z = mu + (L @ eps.T).T

    sample_mean = np.mean(z, axis=0)
    sample_cov = np.cov(z, rowvar=False)

    mean_err = float(np.max(np.abs(sample_mean - mu)))
    cov_err = float(np.max(np.abs(sample_cov - Sigma)))
    chol_recon_err = float(np.max(np.abs(L @ L.T - Sigma)))

    # Verify probability density calculation against analytical multivariate normal
    test_points = rng.randn(20, D)
    p_trans = multivariate_affine_transform_density(test_points, mu, L)

    # Scipy or direct MVN evaluation
    inv_sigma = np.linalg.inv(Sigma)
    det_sigma = np.linalg.det(Sigma)
    diff = test_points - mu
    quad = np.sum(diff @ inv_sigma * diff, axis=-1)
    p_true = (1.0 / np.sqrt(((2.0 * np.pi) ** D) * det_sigma)) * np.exp(-0.5 * quad)

    max_p_diff = float(np.max(np.abs(p_trans - p_true)))

    return {
        "mean_err": mean_err,
        "cov_err": cov_err,
        "chol_recon_err": chol_recon_err,
        "max_p_diff": max_p_diff,
    }


# =====================================================================
# Exercise 19.4: Analytical Evaluation of Gaussian KL Divergence & Gradients
# =====================================================================

def gaussian_kl_analytical(
    mu: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    r"""Compute analytical KL divergence :math:`\operatorname{KL}(q(z) \parallel p(z))` (Eq. 19.15).

    .. math::
        q(\mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}, \operatorname{diag}(\boldsymbol{\sigma}^2)), \quad p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
        \operatorname{KL}(q \parallel p) = -\frac{1}{2} \sum_{j=1}^D \left( 1 + \ln \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)
    """
    return float(-0.5 * np.sum(1.0 + np.log(sigma2) - mu ** 2 - sigma2))


def gaussian_kl_gradients(
    mu: np.ndarray,
    sigma2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute analytical gradients of :math:`\operatorname{KL}(q \parallel p)` w.r.t :math:`\boldsymbol{\mu}` and :math:`\ln \boldsymbol{\sigma}^2`.

    .. math::
        \frac{\partial \operatorname{KL}}{\partial \boldsymbol{\mu}} = \boldsymbol{\mu} \\
        \frac{\partial \operatorname{KL}}{\partial \ln \boldsymbol{\sigma}^2} = \frac{1}{2}(\boldsymbol{\sigma}^2 - 1)
    """
    grad_mu = mu.copy()
    grad_log_sigma2 = 0.5 * (sigma2 - 1.0)
    return grad_mu, grad_log_sigma2


def gaussian_kl_monte_carlo(
    mu: np.ndarray,
    sigma2: np.ndarray,
    S: int = 100000,
    random_state: int = 42,
) -> float:
    r"""Compute Monte Carlo estimate of :math:`\operatorname{KL}(q \parallel p) = \mathbb{E}_q[\ln q(z) - \ln p(z)]`."""
    rng = np.random.RandomState(random_state)
    D = len(mu)
    sigma = np.sqrt(sigma2)
    eps = rng.randn(S, D)
    z = mu + sigma * eps

    log_q = -0.5 * D * np.log(2.0 * np.pi) - 0.5 * np.sum(np.log(sigma2)) - 0.5 * np.sum(((z - mu) ** 2) / sigma2, axis=-1)
    log_p = -0.5 * D * np.log(2.0 * np.pi) - 0.5 * np.sum(z ** 2, axis=-1)
    return float(np.mean(log_q - log_p))


def verify_exercise_19_4(
    D: int = 4,
    random_state: int = 42,
) -> Dict[str, float]:
    r"""Exercise 19.4: Verify analytical Gaussian KL and its gradients."""
    rng = np.random.RandomState(random_state)
    mu = rng.randn(D) * 0.8
    sigma2 = np.exp(rng.randn(D) * 0.5)

    kl_ana = gaussian_kl_analytical(mu, sigma2)
    kl_mc = gaussian_kl_monte_carlo(mu, sigma2, S=100000, random_state=random_state)
    kl_mc_diff = abs(kl_ana - kl_mc)

    # Check non-negativity and minimum at mu=0, sigma2=1
    kl_zero = gaussian_kl_analytical(np.zeros(D), np.ones(D))

    # Analytical gradients
    grad_mu, grad_log_s2 = gaussian_kl_gradients(mu, sigma2)

    # Finite difference gradients
    eps_fd = 1e-6
    num_grad_mu = np.zeros(D)
    num_grad_log_s2 = np.zeros(D)

    for i in range(D):
        mu_pos = mu.copy()
        mu_pos[i] += eps_fd
        mu_neg = mu.copy()
        mu_neg[i] -= eps_fd
        num_grad_mu[i] = (gaussian_kl_analytical(mu_pos, sigma2) - gaussian_kl_analytical(mu_neg, sigma2)) / (2.0 * eps_fd)

        s2_pos = sigma2.copy()
        s2_pos[i] = np.exp(np.log(sigma2[i]) + eps_fd)
        s2_neg = sigma2.copy()
        s2_neg[i] = np.exp(np.log(sigma2[i]) - eps_fd)
        num_grad_log_s2[i] = (gaussian_kl_analytical(mu, s2_pos) - gaussian_kl_analytical(mu, s2_neg)) / (2.0 * eps_fd)

    err_grad_mu = float(np.max(np.abs(grad_mu - num_grad_mu)))
    err_grad_log_s2 = float(np.max(np.abs(grad_log_s2 - num_grad_log_s2)))

    return {
        "kl_ana": kl_ana,
        "kl_mc": kl_mc,
        "kl_mc_diff": kl_mc_diff,
        "kl_zero": kl_zero,
        "err_grad_mu": err_grad_mu,
        "err_grad_log_s2": err_grad_log_s2,
    }


# =====================================================================
# Exercise 19.5: Alternative Form of the ELBO
# =====================================================================

def gaussian_differential_entropy(sigma2: np.ndarray) -> float:
    r"""Compute differential entropy of diagonal Gaussian :math:`\mathcal{H}(q) = \frac{1}{2}\sum_j (1 + \ln(2\pi\sigma_j^2))`."""
    D = len(sigma2)
    return float(0.5 * D * (1.0 + np.log(2.0 * np.pi)) + 0.5 * np.sum(np.log(sigma2)))


def verify_exercise_19_5(
    mu: np.ndarray,
    sigma2: np.ndarray,
    x: np.ndarray,
    W: np.ndarray,
    obs_var: float = 0.5,
) -> Dict[str, float]:
    r"""Exercise 19.5: Verify equivalence of ELBO representations.

    .. math::
        \text{Form A: } \mathcal{L} = \mathbb{E}_q[\ln p(\mathbf{x}|\mathbf{z})] - \operatorname{KL}(q \parallel p) \\
        \text{Form B: } \mathcal{L} = \mathbb{E}_q[\ln p(\mathbf{x}, \mathbf{z})] + \mathcal{H}(q)
    """
    D = len(mu)
    Dx = len(x)

    # 1. KL divergence to prior p(z) = N(0, I)
    kl = gaussian_kl_analytical(mu, sigma2)

    # 2. Entropy H(q)
    entropy_q = gaussian_differential_entropy(sigma2)

    # 3. Expected log likelihood E_q[ln p(x|z)] where p(x|z) = N(W z, obs_var * I)
    # E_q[ ||x - W z||^2 ] = ||x - W mu||^2 + Tr(W diag(sigma2) W^T)
    x_pred = W @ mu
    recon_sq_mean = np.sum((x - x_pred) ** 2)
    trace_term = np.sum((W ** 2) @ sigma2)
    expected_sq_diff = recon_sq_mean + trace_term
    expected_log_p_x_given_z = float(-0.5 * Dx * np.log(2.0 * np.pi * obs_var) - 0.5 * expected_sq_diff / obs_var)

    # Form A:
    elbo_A = expected_log_p_x_given_z - kl

    # Expected log prior E_q[ln p(z)] where p(z) = N(0, I)
    # E_q[ -0.5 D ln(2pi) - 0.5 z^T z ] = -0.5 D ln(2pi) - 0.5 ( ||mu||^2 + sum(sigma2) )
    expected_log_p_z = float(-0.5 * D * np.log(2.0 * np.pi) - 0.5 * np.sum(mu ** 2 + sigma2))

    # Expected complete data log-likelihood E_q[ln p(x, z)] = E_q[ln p(x|z)] + E_q[ln p(z)]
    expected_log_joint = expected_log_p_x_given_z + expected_log_p_z

    # Form B:
    elbo_B = expected_log_joint + entropy_q

    diff = abs(elbo_A - elbo_B)

    return {
        "elbo_A": elbo_A,
        "elbo_B": elbo_B,
        "diff": diff,
        "kl": kl,
        "entropy_q": entropy_q,
        "expected_log_joint": expected_log_joint,
    }


# =====================================================================
# Exercise 19.6: Unconstrained Optimal Variational Distribution
# =====================================================================

def evaluate_mixture_model_elbo(
    x: float,
    pi: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    q: np.ndarray,
) -> Tuple[float, float, float]:
    r"""Evaluate ELBO :math:`\mathcal{L}(q)` and true marginal log-likelihood for 1D GMM with discrete latent component :math:`k \in \{1,\dots,K\}`.

    .. math::
        p(x, k) = \pi_k \mathcal{N}(x | \mu_k, \sigma_k^2), \quad p(x) = \sum_k p(x, k) \\
        p(k|x) = \frac{p(x, k)}{p(x)} \\
        \mathcal{L}(q) = \sum_k q_k \ln \left( \frac{p(x, k)}{q_k} \right) = \ln p(x) - \operatorname{KL}(q \parallel p(k|x))
    """
    K = len(pi)
    # Joint likelihoods p(x, k)
    p_x_k = np.array([pi[k] * norm.pdf(x, loc=mus[k], scale=sigmas[k]) for k in range(K)])
    p_x = float(np.sum(p_x_k))
    log_p_x = float(np.log(p_x))

    # True posterior p(k|x)
    true_posterior = p_x_k / p_x

    # ELBO for variational distribution q
    eps = 1e-15
    q_safe = np.clip(q, eps, 1.0)
    elbo = float(np.sum(q * (np.log(p_x_k + eps) - np.log(q_safe))))

    # KL(q || true_posterior)
    kl = float(np.sum(q * (np.log(q_safe) - np.log(true_posterior + eps))))

    return elbo, log_p_x, kl


def verify_exercise_19_6(
    x: float = 1.2,
    random_state: int = 42,
) -> Dict[str, float]:
    r"""Exercise 19.6: Verify unconstrained optimal variational distribution equals true posterior."""
    rng = np.random.RandomState(random_state)
    pi = np.array([0.4, 0.6])
    mus = np.array([-1.0, 2.0])
    sigmas = np.array([0.8, 1.2])

    # Compute true posterior q*
    p_x_k = np.array([pi[k] * norm.pdf(x, loc=mus[k], scale=sigmas[k]) for k in range(len(pi))])
    q_opt = p_x_k / np.sum(p_x_k)

    # Evaluate at optimal q*
    elbo_opt, log_p_x, kl_opt = evaluate_mixture_model_elbo(x, pi, mus, sigmas, q_opt)
    diff_opt = abs(elbo_opt - log_p_x)

    # Evaluate at suboptimal q
    q_sub = np.array([0.9, 0.1])
    elbo_sub, _, kl_sub = evaluate_mixture_model_elbo(x, pi, mus, sigmas, q_sub)
    gap_sub = log_p_x - elbo_sub
    kl_gap_diff = abs(gap_sub - kl_sub)

    return {
        "log_p_x": log_p_x,
        "elbo_opt": elbo_opt,
        "kl_opt": kl_opt,
        "diff_opt": diff_opt,
        "elbo_sub": elbo_sub,
        "kl_sub": kl_sub,
        "gap_sub": gap_sub,
        "kl_gap_diff": kl_gap_diff,
        "q_opt_0": float(q_opt[0]),
        "q_opt_1": float(q_opt[1]),
    }
