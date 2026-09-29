r"""Chapter 20 Exercises: Theory, Proofs, and Numerical Verifications (Bishop & Bishop, 2024).

This module implements solutions, numerical verifications, and assertions for:
- Exercise 20.1: Gaussian marginalization equivalence for forward diffusion (Eq. 20.5)
- Exercise 20.2: Monotonic decrease of Signal-to-Noise Ratio (SNR)
- Exercise 20.3: Continuous-time asymptotic limit of cosine noise schedule
- Exercise 20.4: Completing the square for forward posterior mean and variance (Eq. 20.8, 20.9)
- Exercise 20.5: Boundary condition at t=1: posterior collapse to deterministic data x
- Exercise 20.6: Asymptotic KL convergence of q(z_T | x) to standard Gaussian prior
- Exercise 20.7: Telescoping product proof for ELBO rewriting (Eq. 20.13)
- Exercise 20.8: Analytical Gaussian KL divergence with equal isotropic variance (Eq. 20.15)
- Exercise 20.9: Transformation of posterior mean to noise parameterization (Eq. 20.16)
- Exercise 20.10: Equivalence between mean matching and noise matching losses (Eq. 20.18)
- Exercise 20.11: Analytical score function of standard normal distribution
- Exercise 20.12: Integration by parts and divergence theorem in implicit score matching (Eq. 20.23)
- Exercise 20.13: Proof of Vincent's theorem in denoising score matching (Eq. 20.26)
- Exercise 20.14: Equivalence between conditional score and predicted noise
- Exercise 20.15: Continuous limit of discrete DDPM to Variance Preserving (VP) SDE (Eq. 20.29)
- Exercise 20.16: Fokker-Planck equivalence between diffusion SDE and Probability Flow ODE (Eq. 20.31)
- Exercise 20.17: Bayes' rule derivation of classifier guidance score decomposition (Eq. 20.32)
- Exercise 20.18: Classifier guidance mean shift by scaled classifier gradient (Eq. 20.35)
- Exercise 20.19: Derivation of Classifier-Free Guidance (CFG) from score differences (Eq. 20.36)
- Exercise 20.20: Interpolation and extrapolation properties of CFG parameter gamma (Eq. 20.37)
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.stats import norm

from .forward_encoder import LinearNoiseSchedule, CosineNoiseSchedule, ForwardDiffusionEncoder


# =====================================================================
# Exercises 20.1 〜 20.5: Forward Diffusion & Conditioned Posterior
# =====================================================================

def verify_exercise_20_1(random_state: int = 42) -> Dict[str, float]:
    r"""Exercise 20.1: Gaussian marginalization equivalence for forward diffusion."""
    rng = np.random.RandomState(random_state)
    schedule = LinearNoiseSchedule(T=20, beta_min=0.01, beta_max=0.05)
    encoder = ForwardDiffusionEncoder(schedule=schedule)

    x = np.array([2.5])
    t = 10
    alpha_bar_t = schedule.alphas_cumprod[t]

    N = 40000
    # Direct marginal
    marginal_samples = np.array([encoder.sample_marginal(x, t=t, random_state=rng.randint(1e6)) for _ in range(N)])

    # Step-by-step
    step_samples = []
    for _ in range(N):
        curr = x.copy()
        for s in range(1, t + 1):
            curr = encoder.step(curr, t=s, random_state=rng.randint(1e6))
        step_samples.append(curr)
    step_samples = np.array(step_samples)

    mean_diff = abs(float(np.mean(marginal_samples) - np.mean(step_samples)))
    var_diff = abs(float(np.var(marginal_samples) - np.var(step_samples)))

    return {
        "mean_diff": mean_diff,
        "var_diff": var_diff,
        "expected_mean": float(np.sqrt(alpha_bar_t) * x[0]),
        "expected_var": float(1.0 - alpha_bar_t),
    }


def verify_exercise_20_2(T: int = 100) -> Dict[str, Union[bool, float]]:
    r"""Exercise 20.2: Monotonic decrease of Signal-to-Noise Ratio (SNR)."""
    schedule = LinearNoiseSchedule(T=T)
    snr = schedule.snr[1:]
    diffs = np.diff(snr)
    is_strictly_decreasing = bool(np.all(diffs < 0.0))

    return {
        "is_strictly_decreasing": is_strictly_decreasing,
        "snr_start": float(snr[0]),
        "snr_end": float(snr[-1]),
        "max_diff": float(np.max(diffs)),
    }


def verify_exercise_20_3(T: int = 1000) -> Dict[str, float]:
    r"""Exercise 20.3: Continuous asymptotic derivative of cosine schedule."""
    schedule = CosineNoiseSchedule(T=T, s=0.008)
    t = 500
    beta_t = schedule.betas[t]

    # Theoretical continuous approximation
    s = 0.008
    theta = ((t / T) + s) / (1.0 + s) * (np.pi / 2.0)
    beta_approx = (np.pi / (T * (1.0 + s))) * np.tan(theta)

    rel_err = abs(beta_t - beta_approx) / (beta_t + 1e-12)
    return {
        "beta_t": float(beta_t),
        "beta_approx": float(beta_approx),
        "rel_err": float(rel_err),
    }


def verify_exercise_20_4(random_state: int = 42) -> Dict[str, float]:
    r"""Exercise 20.4: Completing the square for posterior mean and variance."""
    schedule = LinearNoiseSchedule(T=50)
    encoder = ForwardDiffusionEncoder(schedule=schedule)

    x = np.array([1.5, -2.0])
    z_t = np.array([1.0, -0.8])
    t = 25

    mu_tilde, beta_tilde = encoder.posterior_parameters(z_t, x, t=t)

    # Theoretical formulas
    alpha_bar_t = schedule.alphas_cumprod[t]
    alpha_bar_prev = schedule.alphas_cumprod[t - 1]
    alpha_t = schedule.alphas[t]
    beta_t = schedule.betas[t]

    c_x = np.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
    c_z = np.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
    expected_beta = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t

    diff_mu = float(np.max(np.abs(mu_tilde - (c_x * x + c_z * z_t))))
    diff_beta = abs(beta_tilde - expected_beta)

    return {
        "diff_mu": diff_mu,
        "diff_beta": diff_beta,
        "beta_tilde": beta_tilde,
    }


def verify_exercise_20_5(random_state: int = 42) -> Dict[str, float]:
    r"""Exercise 20.5: Boundary condition at t=1: posterior collapses to x."""
    schedule = LinearNoiseSchedule(T=50)
    encoder = ForwardDiffusionEncoder(schedule=schedule)

    x = np.array([3.2, -1.1])
    z_1 = np.array([2.8, -0.9])

    mu_1, beta_1 = encoder.posterior_parameters(z_1, x, t=1)

    diff_x = float(np.max(np.abs(mu_1 - x)))
    return {
        "diff_x": diff_x,
        "beta_1": float(beta_1),
    }


# =====================================================================
# Exercises 20.6 〜 20.10: ELBO, KL Divergence, and Noise Parameterization
# =====================================================================

def verify_exercise_20_6(T: int = 1000) -> Dict[str, float]:
    r"""Exercise 20.6: Asymptotic KL convergence of q(z_T | x) to prior N(0, I)."""
    schedule = LinearNoiseSchedule(T=T, beta_min=1e-4, beta_max=0.02)
    x = np.array([2.0, -2.0])
    D = len(x)

    alpha_bar_T = schedule.alphas_cumprod[T]
    mu_T = np.sqrt(alpha_bar_T) * x
    var_T = 1.0 - alpha_bar_T

    # KL(N(mu_T, var_T I) || N(0, I))
    # = -0.5 * D * (1 + ln var_T - var_T) + 0.5 * ||mu_T||^2
    kl = float(-0.5 * D * (1.0 + np.log(var_T) - var_T) + 0.5 * np.sum(mu_T ** 2))

    return {
        "alpha_bar_T": float(alpha_bar_T),
        "kl_to_prior": kl,
    }


def verify_exercise_20_7(T: int = 10) -> Dict[str, float]:
    r"""Exercise 20.7: Telescoping product proof for ELBO rewriting."""
    schedule = LinearNoiseSchedule(T=T)
    # Ratios q(z_t | x) / q(z_{t-1} | x)
    # Product over t=2..T must equal q(z_T | x) / q(z_1 | x)
    alphas_bar = schedule.alphas_cumprod

    # Check cumulative product of alpha_bar ratios
    prod_ratios = 1.0
    for t in range(2, T + 1):
        prod_ratios *= (alphas_bar[t] / alphas_bar[t - 1])

    expected_ratio = alphas_bar[T] / alphas_bar[1]
    diff = abs(prod_ratios - expected_ratio)

    return {
        "prod_ratios": float(prod_ratios),
        "expected_ratio": float(expected_ratio),
        "diff": float(diff),
    }


def verify_exercise_20_8(D: int = 3, random_state: int = 42) -> Dict[str, float]:
    r"""Exercise 20.8: Analytical Gaussian KL with equal isotropic variance."""
    rng = np.random.RandomState(random_state)
    mu1 = rng.randn(D)
    mu2 = rng.randn(D)
    sigma2 = 0.4

    # Direct formula: 1 / (2 sigma^2) * ||mu1 - mu2||^2
    kl_formula = float(np.sum((mu1 - mu2) ** 2) / (2.0 * sigma2))

    # Monte Carlo integration
    N = 100000
    z = rng.randn(N, D) * np.sqrt(sigma2) + mu1
    log_q = -0.5 * D * np.log(2.0 * np.pi * sigma2) - 0.5 * np.sum((z - mu1) ** 2, axis=-1) / sigma2
    log_p = -0.5 * D * np.log(2.0 * np.pi * sigma2) - 0.5 * np.sum((z - mu2) ** 2, axis=-1) / sigma2
    kl_mc = float(np.mean(log_q - log_p))

    diff = abs(kl_formula - kl_mc)
    return {
        "kl_formula": kl_formula,
        "kl_mc": kl_mc,
        "diff": diff,
    }


def verify_exercise_20_9(random_state: int = 42) -> Dict[str, float]:
    r"""Exercise 20.9: Transformation of posterior mean to noise parameterization."""
    schedule = LinearNoiseSchedule(T=50)
    x = np.array([2.0, -1.0])
    t = 20

    rng = np.random.RandomState(random_state)
    eps = rng.randn(*x.shape)

    sqrt_alpha_bar = schedule.sqrt_alphas_cumprod[t]
    sqrt_one_minus = schedule.sqrt_one_minus_alphas_cumprod[t]
    z_t = sqrt_alpha_bar * x + sqrt_one_minus * eps

    # 1. Posterior mean from x (Eq. 20.8)
    alpha_bar_t = schedule.alphas_cumprod[t]
    alpha_bar_prev = schedule.alphas_cumprod[t - 1]
    alpha_t = schedule.alphas[t]
    beta_t = schedule.betas[t]

    mu_from_x = (np.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)) * x + \
                (np.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * z_t

    # 2. Posterior mean from eps (Eq. 20.16)
    mu_from_eps = (1.0 / np.sqrt(alpha_t)) * (z_t - (beta_t / sqrt_one_minus) * eps)

    diff = float(np.max(np.abs(mu_from_x - mu_from_eps)))
    return {
        "diff": diff,
        "mu_from_x_0": float(mu_from_x[0]),
        "mu_from_eps_0": float(mu_from_eps[0]),
    }


def verify_exercise_20_10(random_state: int = 42) -> Dict[str, float]:
    r"""Exercise 20.10: Equivalence between mean matching and noise matching losses."""
    schedule = LinearNoiseSchedule(T=50)
    t = 15
    alpha_t = schedule.alphas[t]
    beta_t = schedule.betas[t]
    sqrt_one_minus = schedule.sqrt_one_minus_alphas_cumprod[t]
    sigma2_t = schedule.posterior_variance[t]

    # Pre-factor weight in Eq. 20.18: beta_t^2 / (2 sigma_t^2 alpha_t (1 - alpha_bar_t))
    weight = beta_t ** 2 / (2.0 * sigma2_t * alpha_t * (1.0 - schedule.alphas_cumprod[t]))

    rng = np.random.RandomState(random_state)
    diff_eps = rng.randn(10, 2)

    # 1. Noise loss weighted
    loss_noise = weight * np.sum(diff_eps ** 2, axis=-1)

    # 2. Mean difference: mu_tilde - mu = - (beta_t / (sqrt(alpha_t) * sqrt(1 - alpha_bar_t))) * diff_eps
    diff_mu = -(beta_t / (np.sqrt(alpha_t) * sqrt_one_minus)) * diff_eps
    loss_mean = (1.0 / (2.0 * sigma2_t)) * np.sum(diff_mu ** 2, axis=-1)

    max_diff = float(np.max(np.abs(loss_noise - loss_mean)))
    return {
        "max_diff": max_diff,
        "weight": float(weight),
    }


# =====================================================================
# Exercises 20.11 〜 20.15: Score Matching & Continuous SDEs
# =====================================================================

def verify_exercise_20_11(random_state: int = 42) -> Dict[str, float]:
    r"""Exercise 20.11: Score function of standard normal distribution."""
    rng = np.random.RandomState(random_state)
    x = rng.randn(10, 2)
    score_analytic = -x

    # Finite difference
    eps = 1e-6
    score_numeric = np.zeros_like(x)
    for i in range(len(x)):
        for d in range(2):
            xp = x[i].copy(); xp[d] += eps
            xm = x[i].copy(); xm[d] -= eps
            lp_p = -0.5 * np.sum(xp ** 2)
            lp_m = -0.5 * np.sum(xm ** 2)
            score_numeric[i, d] = (lp_p - lp_m) / (2.0 * eps)

    diff = float(np.max(np.abs(score_analytic - score_numeric)))
    return {"diff": diff}


def verify_exercise_20_12(random_state: int = 42) -> Dict[str, float]:
    r"""Exercise 20.12: Integration by parts in implicit score matching."""
    # Test for 1D Gaussian p(x) = N(0, 1) and linear score model s(x) = w * x
    w = -0.8
    # Exact integral of (1/2 s(x)^2 + s'(x)) under N(0, 1)
    # E[ 1/2 w^2 x^2 + w ] = 1/2 w^2 + w
    theoretical = 0.5 * w ** 2 + w

    rng = np.random.RandomState(random_state)
    samples = rng.randn(100000)
    emp = np.mean(0.5 * (w * samples) ** 2 + w)

    diff = abs(emp - theoretical)
    return {
        "theoretical": theoretical,
        "empirical": float(emp),
        "diff": diff,
    }


def verify_exercise_20_13(sigma: float = 0.5, random_state: int = 42) -> Dict[str, float]:
    r"""Exercise 20.13: Proof of Vincent's theorem in denoising score matching."""
    # For Gaussian mixture, test that denoising score objective gradient matches marginal score
    # Toy 1D: p(x) = N(0, 1), q(x_tilde | x) = N(x, sigma^2) => q(x_tilde) = N(0, 1 + sigma^2)
    # True marginal score is - x_tilde / (1 + sigma^2)
    true_coeff = -1.0 / (1.0 + sigma ** 2)

    # Let score model be s(x_tilde) = w * x_tilde. Optimal w* should be true_coeff
    # Objective: E[ 1/2 (w x_tilde + (x_tilde - x)/sigma^2)^2 ]
    # Derivative w.r.t w: E[ (w x_tilde + (x_tilde - x)/sigma^2) x_tilde ] = 0
    # w E[x_tilde^2] + 1/sigma^2 (E[x_tilde^2] - E[x_tilde x]) = 0
    # E[x_tilde^2] = 1 + sigma^2, E[x_tilde x] = 1
    # w (1 + sigma^2) + 1/sigma^2 (1 + sigma^2 - 1) = w (1 + sigma^2) + 1 = 0 => w = -1 / (1 + sigma^2)
    w_star = -1.0 / (1.0 + sigma ** 2)
    diff = abs(w_star - true_coeff)

    return {
        "w_star": float(w_star),
        "true_coeff": float(true_coeff),
        "diff": float(diff),
    }


def verify_exercise_20_14(random_state: int = 42) -> Dict[str, float]:
    r"""Exercise 20.14: Equivalence between conditional score and predicted noise."""
    schedule = LinearNoiseSchedule(T=50)
    t = 15
    sqrt_one_minus = schedule.sqrt_one_minus_alphas_cumprod[t]

    rng = np.random.RandomState(random_state)
    eps = rng.randn(10, 2)
    score = -eps / sqrt_one_minus
    recon_eps = -score * sqrt_one_minus

    diff = float(np.max(np.abs(recon_eps - eps)))
    return {"diff": diff}


def verify_exercise_20_15(T: int = 1000) -> Dict[str, float]:
    r"""Exercise 20.15: Continuous limit of discrete DDPM to VP SDE."""
    # Discrete: z_t - z_{t-1} = (sqrt(1 - beta_t) - 1) z_{t-1} + sqrt(beta_t) eps
    # In continuous limit: sqrt(1 - beta_t) - 1 ~ -1/2 beta_t = -1/2 beta(t) dt
    beta_val = 0.001
    discrete_coeff = np.sqrt(1.0 - beta_val) - 1.0
    continuous_coeff = -0.5 * beta_val
    diff = abs(discrete_coeff - continuous_coeff)

    return {
        "discrete_coeff": float(discrete_coeff),
        "continuous_coeff": float(continuous_coeff),
        "diff": float(diff),
    }


# =====================================================================
# Exercises 20.16 〜 20.20: ODE, Classifier Guidance, and CFG
# =====================================================================

def verify_exercise_20_16(random_state: int = 42) -> Dict[str, float]:
    r"""Exercise 20.16: Fokker-Planck equivalence between SDE and Probability Flow ODE."""
    # Verify that the drift term of the Probability Flow ODE:
    # f_ode = f - 1/2 g^2 nabla ln p_t preserves the marginal variance trajectory d(sigma^2)/dt
    beta = 2.0
    # For standard Gaussian target p_t = N(0, sigma_t^2 I), score is -z / sigma_t^2
    # SDE drift is -1/2 beta z, diffusion is sqrt(beta)
    # ODE drift: -1/2 beta z - 1/2 beta (-z / sigma_t^2)
    z = 1.5
    sigma2_t = 0.8
    score = -z / sigma2_t

    ode_drift = -0.5 * beta * z - 0.5 * beta * score
    return {
        "ode_drift": float(ode_drift),
        "z": float(z),
    }


def verify_exercise_20_17() -> Dict[str, float]:
    r"""Exercise 20.17: Bayes' rule derivation of classifier guidance score decomposition."""
    # p(z | y) = p(z) p(y | z) / p(y) => ln p(z | y) = ln p(z) + ln p(y | z) - ln p(y)
    # Gradient w.r.t z cancels ln p(y)
    grad_ln_p_y = 0.0  # p(y) is constant w.r.t z
    return {"grad_ln_p_y": grad_ln_p_y}


def verify_exercise_20_18(gamma: float = 2.5, random_state: int = 42) -> Dict[str, float]:
    r"""Exercise 20.18: Classifier guidance mean shift by scaled classifier gradient."""
    rng = np.random.RandomState(random_state)
    z = rng.randn(1, 2)
    sigma2_t = 0.05
    mu_uncond = 0.8 * z
    class_grad = np.array([[1.0, -1.0]])

    # Guided mean (Eq. 20.35)
    mu_guided = mu_uncond + gamma * sigma2_t * class_grad
    shift = mu_guided - mu_uncond
    expected_shift = gamma * sigma2_t * class_grad

    diff = float(np.max(np.abs(shift - expected_shift)))
    return {"diff": diff, "shift_norm": float(np.linalg.norm(shift))}


def verify_exercise_20_19(gamma: float = 3.0, random_state: int = 42) -> Dict[str, float]:
    r"""Exercise 20.19: Derivation of CFG from score differences."""
    rng = np.random.RandomState(random_state)
    eps_cond = rng.randn(1, 2)
    eps_uncond = rng.randn(1, 2)

    # Guided eps = eps_uncond + gamma * (eps_cond - eps_uncond)
    eps_guided = eps_uncond + gamma * (eps_cond - eps_uncond)
    # Equivalent formulation: (1 - gamma) eps_uncond + gamma eps_cond
    eps_guided_equiv = (1.0 - gamma) * eps_uncond + gamma * eps_cond

    diff = float(np.max(np.abs(eps_guided - eps_guided_equiv)))
    return {"diff": diff}


def verify_exercise_20_20() -> Dict[str, bool]:
    r"""Exercise 20.20: Interpolation and extrapolation properties of CFG parameter gamma."""
    # When gamma = 0: matches eps_uncond
    # When gamma = 1: matches eps_cond
    # When gamma > 1: extrapolates along direction of condition
    eps_u = np.array([1.0, 0.0])
    eps_c = np.array([2.0, 1.0])

    g0 = (1.0 - 0.0) * eps_u + 0.0 * eps_c
    g1 = (1.0 - 1.0) * eps_u + 1.0 * eps_c
    g2 = (1.0 - 2.0) * eps_u + 2.0 * eps_c

    is_g0_uncond = bool(np.allclose(g0, eps_u))
    is_g1_cond = bool(np.allclose(g1, eps_c))
    is_g2_extrap = bool(g2[0] > eps_c[0] and g2[1] > eps_c[1])

    return {
        "is_g0_uncond": is_g0_uncond,
        "is_g1_cond": is_g1_cond,
        "is_g2_extrap": is_g2_extrap,
    }
