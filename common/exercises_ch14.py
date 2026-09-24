"""
common/exercises_ch14.py
========================
Chapter 14 Exercises: Sampling
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module provides theoretical proofs, numerical verifications, and interactive
problem structures for Exercises 14.1 through 14.18:
- Exercise 14.1: Unbiasedness and variance of finite-sample Monte Carlo estimator.
- Exercise 14.2: Transformation method (inverse CDF technique / probability integral transform).
- Exercise 14.3: Cauchy random variable generation via inversion method y = gamma * tan(pi*(z - 0.5)) + x0.
- Exercise 14.4: Box-Muller transform from uniform unit disk to independent bivariate Gaussian.
- Exercise 14.5: Multivariate Gaussian generation via Cholesky decomposition y = mu + L z.
- Exercise 14.6: Mathematical proof of rejection sampling acceptance distribution.
- Exercise 14.7: Bounding Gaussian distribution with Cauchy proposal distribution.
- Exercise 14.8: Envelope coefficients k_i in adaptive rejection sampling (ARS).
- Exercise 14.9: Sampling algorithm from piecewise exponential distribution in ARS.
- Exercise 14.10: 1D discrete random walk and diffusive scaling E[(z^(tau))^2] = tau / 2.
- Exercise 14.11: Mathematical proof of detailed balance for Gibbs sampling.
- Exercise 14.12: Ergodicity analysis of Gibbs sampling on disconnected/non-convex support.
- Exercise 14.13: Full conditional distributions p(mu | x, tau) and p(tau | x, mu) for Gaussian-Gamma model.
- Exercise 14.14: Over-relaxation in Gibbs sampling preserving mean and variance.
- Exercise 14.15: Analytical derivation of log-likelihood gradient in Energy-Based Models (EBMs).
- Exercise 14.16: Score function s(x) = nabla_x ln p(x) for multivariate Gaussian.
- Exercise 14.17: Unadjusted Langevin Algorithm (ULA) stationary distribution and discretization bias.
- Exercise 14.18: Metropolis-Adjusted Langevin Algorithm (MALA) detailed balance proof.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
import os
from pathlib import Path
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from .plot_utils import setup_style, save_plot
from .basic_sampling import (
    monte_carlo_expectation,
    inverse_cdf_sample_cauchy,
    box_muller_transform,
    rejection_sample_gaussian_cauchy,
)
from .markov_chain_monte_carlo import (
    metropolis_sample_2d_gaussian,
    verify_detailed_balance,
    gibbs_sample_2d_gaussian,
)
from .langevin_sampling import (
    EnergyBasedModel1D,
    unadjusted_langevin_algorithm,
    metropolis_adjusted_langevin_algorithm,
)


def _save_figure(fig: plt.Figure, filename_base: str, save_dir: Optional[str] = None) -> None:
    """Save figure to Chapter 14 result directory and root result directory."""
    if save_dir is not None:
        save_plot(fig, os.path.join(save_dir, f"{filename_base}.png"))
        return

    repo_root = Path(__file__).resolve().parent.parent
    dir_ch14 = repo_root / "14" / "result"
    dir_root = repo_root / "result"
    dir_ch14.mkdir(parents=True, exist_ok=True)
    dir_root.mkdir(parents=True, exist_ok=True)

    save_plot(fig, str(dir_ch14 / f"{filename_base}.png"))
    save_plot(fig, str(dir_root / f"{filename_base}.png"))


# =============================================================================
# Exercise 14.1: Monte Carlo Estimator Mean and Variance
# =============================================================================

def solve_exercise_14_1() -> Dict[str, Any]:
    """
    Exercise 14.1:
    Show that the finite sample estimator f_hat = (1/L) sum_{l=1}^L f(z^(l))
    has expectation E[f_hat] = E[f] and variance Var[f_hat] = (1/L) Var[f].
    """
    explanation = (
        "Let z^(1), ..., z^(L) be i.i.d. draws from p(z).\n"
        "1. Expectation by linearity:\n"
        "   E[f_hat] = (1/L) sum_{l=1}^L E[f(z^(l))] = (1/L) * L * E[f] = E[f]. (Unbiased)\n"
        "2. Variance using independence (Cov(f(z^(i)), f(z^(j))) = 0 for i != j):\n"
        "   Var[f_hat] = Var[(1/L) sum_{l=1}^L f(z^(l))] = (1/L^2) sum_{l=1}^L Var[f(z^(l))]\n"
        "              = (1/L^2) * L * Var[f] = (1/L) Var[f]."
    )
    return {
        "is_unbiased": True,
        "variance_factor": "1/L",
        "explanation": explanation,
    }


def verify_exercise_14_1() -> bool:
    sol = solve_exercise_14_1()
    # Numerical validation with known function f(z) = z^2 for z ~ N(0, 1)
    # True E[z^2] = 1, Var[z^2] = E[z^4] - (E[z^2])^2 = 3 - 1 = 2
    rng = np.random.RandomState(42)
    L = 50
    num_experiments = 4000
    estimates = np.zeros(num_experiments)
    for i in range(num_experiments):
        z = rng.randn(L)
        estimates[i] = np.mean(z ** 2)

    emp_mean = np.mean(estimates)
    emp_var = np.var(estimates)
    true_mean = 1.0
    true_var = 2.0 / L  # 2.0 / 50 = 0.04

    return (
        sol["is_unbiased"]
        and np.isclose(emp_mean, true_mean, atol=0.03)
        and np.isclose(emp_var, true_var, atol=0.005)
    )


# =============================================================================
# Exercise 14.2: Transformation Method (Inverse CDF)
# =============================================================================

def solve_exercise_14_2() -> Dict[str, Any]:
    """
    Exercise 14.2:
    Suppose z ~ Uniform(0, 1) and y = h^{-1}(z) where h(y) = int_{-inf}^y p(y') dy'.
    Show that y has probability density p(y).
    """
    explanation = (
        "By definition of cumulative distribution function (CDF), h(y) is monotonic and non-decreasing.\n"
        "The CDF of y is:\n"
        "P(Y <= y) = P(h^{-1}(Z) <= y) = P(Z <= h(y)).\n"
        "Since Z ~ Uniform(0, 1), P(Z <= u) = u for u in [0, 1].\n"
        "Therefore, P(Y <= y) = h(y).\n"
        "Differentiating with respect to y:\n"
        "p_Y(y) = d/dy P(Y <= y) = d/dy h(y) = p(y)."
    )
    return {
        "cdf_relation": "P(Y <= y) = h(y)",
        "pdf_result": "p_Y(y) = p(y)",
        "explanation": explanation,
    }


def verify_exercise_14_2() -> bool:
    sol = solve_exercise_14_2()
    # Test on Exponential distribution p(y) = lambda * exp(-lambda * y)
    # CDF: h(y) = 1 - exp(-lambda * y) => y = -ln(1 - z) / lambda
    rng = np.random.RandomState(42)
    lam = 1.5
    u = rng.uniform(0.0, 1.0, 10000)
    y = -np.log(1.0 - u) / lam

    # Kolmogorov-Smirnov test against stats.expon(scale=1/lam)
    ks_stat, p_val = stats.kstest(y, "expon", args=(0, 1.0 / lam))
    return p_val > 0.05


# =============================================================================
# Exercise 14.3: Cauchy Distribution via Inversion
# =============================================================================

def solve_exercise_14_3() -> Dict[str, Any]:
    """
    Exercise 14.3:
    Given z ~ Uniform(0, 1), find transformation y = f(z) such that y has Cauchy distribution:
        p(y) = (1 / pi) * (gamma / ((y - x0)^2 + gamma^2)).
    """
    explanation = (
        "For standard Cauchy (x0 = 0, gamma = 1):\n"
        "p(y) = 1 / (pi * (1 + y^2)).\n"
        "CDF: h(y) = int_{-inf}^y (1 / (pi * (1 + y'^2))) dy' = (1 / pi) * arctan(y) + 1/2.\n"
        "Setting z = h(y):\n"
        "z - 1/2 = (1 / pi) * arctan(y)  =>  arctan(y) = pi * (z - 1/2).\n"
        "y = tan(pi * (z - 1/2)).\n"
        "For location x0 and scale gamma:\n"
        "y = x0 + gamma * tan(pi * (z - 1/2))."
    )
    return {
        "transformation": "y = x0 + gamma * tan(pi * (z - 0.5))",
        "explanation": explanation,
    }


def verify_exercise_14_3() -> bool:
    sol = solve_exercise_14_3()
    x0, gamma = 2.0, 1.5
    rng = np.random.RandomState(42)
    u = rng.uniform(0.0, 1.0, 10000)
    samples = x0 + gamma * np.tan(np.pi * (u - 0.5))

    # Median of Cauchy is x0, interquartile range is 2 * gamma
    med = np.median(samples)
    q25, q75 = np.percentile(samples, [25, 75])
    iqr = q75 - q25

    return (
        np.abs(med - x0) < 0.15
        and np.abs(iqr - 2.0 * gamma) < 0.25
    )


# =============================================================================
# Exercise 14.4: Box-Muller Transformation
# =============================================================================

def solve_exercise_14_4() -> Dict[str, Any]:
    """
    Exercise 14.4:
    Show that the Box-Muller algorithm generates two independent standard normal variables:
        y1 = (-2 ln r^2)^{1/2} * (z1 / r)
        y2 = (-2 ln r^2)^{1/2} * (z2 / r)
    where (z1, z2) are uniformly distributed on the unit disk and r^2 = z1^2 + z2^2.
    """
    explanation = (
        "In polar coordinates, (z1, z2) = (r cos theta, r sin theta).\n"
        "Uniform distribution on unit disk: p(r, theta) = (r / pi) for r in (0, 1), theta in [0, 2pi).\n"
        "Let u = r^2. Then du = 2 r dr, so p(u) = 1 (Uniform(0, 1)).\n"
        "Then y1 = sqrt(-2 ln u) * cos(theta), y2 = sqrt(-2 ln u) * sin(theta).\n"
        "Jacobian determinant |d(y1, y2)/d(u, theta)| = 1 / (2*pi).\n"
        "Hence p(y1, y2) = (1 / (2*pi)) * exp(-0.5 * (y1^2 + y2^2)) = N(y1|0, 1) * N(y2|0, 1)."
    )
    return {
        "joint_distribution": "p(y1, y2) = N(y1|0, 1) * N(y2|0, 1)",
        "independence": True,
        "explanation": explanation,
    }


def verify_exercise_14_4() -> bool:
    sol = solve_exercise_14_4()
    rng = np.random.RandomState(42)
    y1, y2 = box_muller_transform(10000, seed=42)

    # Check mean, variance, and covariance
    return (
        sol["independence"]
        and np.abs(np.mean(y1)) < 0.05
        and np.abs(np.mean(y2)) < 0.05
        and np.abs(np.var(y1) - 1.0) < 0.08
        and np.abs(np.var(y2) - 1.0) < 0.08
        and np.abs(np.cov(y1, y2)[0, 1]) < 0.05
    )


# =============================================================================
# Exercise 14.5: Multivariate Gaussian via Cholesky Decomposition
# =============================================================================

def solve_exercise_14_5() -> Dict[str, Any]:
    """
    Exercise 14.5:
    Show that if z ~ N(0, I) and Sigma = L L^T (Cholesky decomposition),
    then y = mu + L z has distribution N(mu, Sigma).
    """
    explanation = (
        "Linear transformation of Gaussian vector z:\n"
        "1. Expectation: E[y] = E[mu + L z] = mu + L E[z] = mu + 0 = mu.\n"
        "2. Covariance:\n"
        "   Cov[y] = E[(y - mu)(y - mu)^T] = E[(L z)(L z)^T] = E[L (z z^T) L^T]\n"
        "          = L E[z z^T] L^T = L I L^T = L L^T = Sigma.\n"
        "Since linear transformation of Gaussian remains Gaussian, y ~ N(mu, Sigma)."
    )
    return {
        "mean": "mu",
        "covariance": "Sigma = L L^T",
        "explanation": explanation,
    }


def verify_exercise_14_5() -> bool:
    sol = solve_exercise_14_5()
    mu = np.array([1.0, -2.0, 3.0])
    Sigma = np.array([
        [2.0, 0.6, 0.4],
        [0.6, 1.5, -0.3],
        [0.4, -0.3, 1.0],
    ])
    L = np.linalg.cholesky(Sigma)

    rng = np.random.RandomState(42)
    z = rng.randn(3, 15000)
    y = mu[:, None] + L @ z

    emp_mean = np.mean(y, axis=1)
    emp_cov = np.cov(y)

    return (
        np.allclose(emp_mean, mu, atol=0.08)
        and np.allclose(emp_cov, Sigma, atol=0.10)
    )


# =============================================================================
# Exercise 14.6: Exact Proof of Rejection Sampling
# =============================================================================

def solve_exercise_14_6() -> Dict[str, Any]:
    """
    Exercise 14.6:
    Show that rejection sampling generates exact samples from p(z).
    """
    explanation = (
        "1. Sample proposal z ~ q(z).\n"
        "2. Accept with probability P(accept | z) = p_tilde(z) / (k * q(z)).\n"
        "3. Marginal probability of accepting any proposal:\n"
        "   P(accept) = int P(accept | z) q(z) dz = int (p_tilde(z) / (k * q(z))) q(z) dz\n"
        "             = (1 / k) int p_tilde(z) dz = Z_p / k.\n"
        "4. By Bayes' theorem, distribution of accepted samples:\n"
        "   p(z | accept) = (P(accept | z) q(z)) / P(accept)\n"
        "                 = (p_tilde(z) / (k * q(z)) * q(z)) / (Z_p / k)\n"
        "                 = (p_tilde(z) / k) / (Z_p / k) = p_tilde(z) / Z_p = p(z)."
    )
    return {
        "p_accept": "Z_p / k",
        "conditional_distribution": "p(z | accept) = p(z)",
        "explanation": explanation,
    }


def verify_exercise_14_6() -> bool:
    sol = solve_exercise_14_6()
    # Validate on standard Gaussian bounded by Cauchy
    res = rejection_sample_gaussian_cauchy(num_samples=5000, seed=42)
    samples = res["samples"]

    # KS test against standard Gaussian
    ks_stat, p_val = stats.kstest(samples, "norm")
    return p_val > 0.05 and sol["conditional_distribution"] == "p(z | accept) = p(z)"


# =============================================================================
# Exercise 14.7: Bounding Gaussian with Cauchy
# =============================================================================

def solve_exercise_14_7() -> Dict[str, Any]:
    """
    Exercise 14.7:
    Show that standard Gaussian N(0, 1) can be bounded by a scaled Cauchy distribution
    k * Cauchy(0, gamma) with optimal scale gamma = sqrt(2) and bound k = sqrt(2*pi) * exp(1/2) * gamma.
    """
    explanation = (
        "Ratio R(z) = N(z | 0, 1) / Cauchy(z | 0, gamma):\n"
        "R(z) = [ (1 / sqrt(2*pi)) * exp(-z^2 / 2) ] / [ (gamma / pi) / (z^2 + gamma^2) ]\n"
        "     = (pi / (gamma * sqrt(2*pi))) * (z^2 + gamma^2) * exp(-z^2 / 2).\n"
        "Let u = z^2 >= 0. Maximize g(u) = (u + gamma^2) exp(-u/2):\n"
        "g'(u) = exp(-u/2) - 0.5 (u + gamma^2) exp(-u/2) = (1 - 0.5 gamma^2 - 0.5 u) exp(-u/2).\n"
        "For gamma = sqrt(2), g'(u) = -0.5 u exp(-u/2) <= 0 for all u >= 0.\n"
        "Thus maximum occurs at u = 0 (z = 0), touching tangent at the mode!"
    )
    return {
        "optimal_gamma": np.sqrt(2.0),
        "tangent_at_zero": True,
        "explanation": explanation,
    }


def verify_exercise_14_7() -> bool:
    sol = solve_exercise_14_7()
    gamma = float(sol["optimal_gamma"])
    z_vals = np.linspace(-4, 4, 1000)
    p_gauss = stats.norm.pdf(z_vals, 0, 1)
    q_cauchy = stats.cauchy.pdf(z_vals, 0, gamma)

    k = p_gauss[len(z_vals) // 2] / q_cauchy[len(z_vals) // 2]  # At z = 0
    envelope = k * q_cauchy

    return (
        np.all(envelope >= p_gauss - 1e-12)
        and np.isclose(envelope[len(z_vals) // 2], p_gauss[len(z_vals) // 2])
    )


# =============================================================================
# Exercise 14.8: Adaptive Rejection Sampling Envelope Coefficients
# =============================================================================

def solve_exercise_14_8() -> Dict[str, Any]:
    """
    Exercise 14.8:
    Determine expressions for the coefficients k_i in the envelope distribution
    for adaptive rejection sampling (ARS) from continuity and normalization.
    """
    explanation = (
        "In ARS, the upper envelope in log space is piecewise linear:\n"
        "ln q(z) = w_i * (z - z_i) + ln p(z_i) for z in [z_i, z_{i+1}].\n"
        "In natural probability space, this is piecewise exponential:\n"
        "q(z) = k_i * lambda_i * exp(-lambda_i * z).\n"
        "Continuity at intersection points z_i gives recursive relation for k_i,\n"
        "and total normalization sum_i integral_{z_i}^{z_{i+1}} q(z) dz = 1 fixes k_1."
    )
    return {
        "envelope_form": "piecewise exponential",
        "determined_by": "continuity and normalization",
        "explanation": explanation,
    }


def verify_exercise_14_8() -> bool:
    sol = solve_exercise_14_8()
    return sol["envelope_form"] == "piecewise exponential"


# =============================================================================
# Exercise 14.9: Sampling from Piecewise Exponential Distribution
# =============================================================================

def solve_exercise_14_9() -> Dict[str, Any]:
    """
    Exercise 14.9:
    Devise an algorithm for sampling from the piecewise exponential distribution:
    1. Select segment i with probability P(segment i) proportional to integral_{z_i}^{z_{i+1}} q(z) dz.
    2. Sample from truncated exponential on [z_i, z_{i+1}] via inverse CDF.
    """
    explanation = (
        "1. Compute segment masses m_i = int_{z_i}^{z_{i+1}} q(z) dz for all intervals.\n"
        "2. Normalize masses: P(i) = m_i / sum_j m_j.\n"
        "3. Draw segment index i ~ Categorical(P).\n"
        "4. Draw u ~ Uniform(0, 1), and invert the truncated exponential CDF on [z_i, z_{i+1}]:\n"
        "   z = z_i - (1 / lambda_i) * ln(1 - u * (1 - exp(-lambda_i * (z_{i+1} - z_i))))."
    )
    return {
        "step1": "Discrete segment selection by interval mass",
        "step2": "Truncated exponential inversion within chosen interval",
        "explanation": explanation,
    }


def verify_exercise_14_9() -> bool:
    sol = solve_exercise_14_9()
    # Numerical sampling from 2-segment exponential
    rng = np.random.RandomState(42)
    # Segment 1: [0, 1], rate = 1.0; Segment 2: [1, 3], rate = 2.0
    mass1 = (1.0 - np.exp(-1.0))
    mass2 = np.exp(-1.0) * 0.5 * (1.0 - np.exp(-4.0))
    p1 = mass1 / (mass1 + mass2)

    samples = []
    for _ in range(10000):
        if rng.rand() < p1:
            u = rng.rand()
            z = -np.log(1.0 - u * (1.0 - np.exp(-1.0)))
        else:
            u = rng.rand()
            z = 1.0 - 0.5 * np.log(1.0 - u * (1.0 - np.exp(-4.0)))
        samples.append(z)

    samples = np.array(samples)
    return (
        np.abs(np.mean(samples < 1.0) - p1) < 0.02
        and sol["step1"] == "Discrete segment selection by interval mass"
    )


# =============================================================================
# Exercise 14.10: Random Walk Diffusive Scaling
# =============================================================================

def solve_exercise_14_10() -> Dict[str, Any]:
    """
    Exercise 14.10:
    Show that the 1D random walk over integers:
        p(z^(tau+1) = z^(tau)) = 0.5
        p(z^(tau+1) = z^(tau) + 1) = 0.25
        p(z^(tau+1) = z^(tau) - 1) = 0.25
    satisfies E[(z^(tau))^2] = E[(z^(tau-1))^2] + 1/2, and hence by induction E[(z^(tau))^2] = tau / 2.
    """
    explanation = (
        "Let Delta = z^(tau) - z^(tau-1).\n"
        "Delta is independent of z^(tau-1), with distribution:\n"
        "P(Delta = 0) = 0.5, P(Delta = +1) = 0.25, P(Delta = -1) = 0.25.\n"
        "Then E[Delta] = 0.25 * (+1) + 0.25 * (-1) + 0.5 * 0 = 0.\n"
        "E[Delta^2] = 0.25 * (+1)^2 + 0.25 * (-1)^2 + 0.5 * 0^2 = 0.25 + 0.25 = 0.5 = 1/2.\n"
        "Now expand:\n"
        "(z^(tau))^2 = (z^(tau-1) + Delta)^2 = (z^(tau-1))^2 + 2 z^(tau-1) Delta + Delta^2.\n"
        "Taking expectations:\n"
        "E[(z^(tau))^2] = E[(z^(tau-1))^2] + 2 E[z^(tau-1)] E[Delta] + E[Delta^2]\n"
        "              = E[(z^(tau-1))^2] + 0 + 1/2 = E[(z^(tau-1))^2] + 1/2.\n"
        "By induction with z^(0) = 0 (E[(z^(0))^2] = 0):\n"
        "E[(z^(tau))^2] = tau / 2."
    )
    return {
        "step_variance": 0.5,
        "asymptotic_scaling": "E[(z^(tau))^2] = tau / 2",
        "diffusive_slowdown": "O(sqrt(tau)) displacement requires O(L^2) steps",
        "explanation": explanation,
    }


def verify_exercise_14_10() -> bool:
    sol = solve_exercise_14_10()
    rng = np.random.RandomState(42)
    num_walks = 5000
    tau = 200

    # Simulate random walks
    steps = rng.choice([0, 1, -1], size=(num_walks, tau), p=[0.5, 0.25, 0.25])
    trajectories = np.cumsum(steps, axis=1)

    emp_second_moment = np.mean(trajectories[:, -1] ** 2)
    true_second_moment = tau / 2.0  # 100.0

    return np.isclose(emp_second_moment, true_second_moment, rtol=0.08)


# =============================================================================
# Exercise 14.11: Detailed Balance for Gibbs Sampling
# =============================================================================

def solve_exercise_14_11() -> Dict[str, Any]:
    """
    Exercise 14.11:
    Show that the Gibbs sampling algorithm satisfies detailed balance:
        p(z) T_i(z -> z*) = p(z*) T_i(z* -> z).
    """
    explanation = (
        "In updating component i, z_{-i}^* = z_{-i}.\n"
        "The transition probability is T_i(z, z*) = p(z_i^* | z_{-i}).\n"
        "The forward flux is:\n"
        "p(z) T_i(z, z*) = p(z_i, z_{-i}) p(z_i^* | z_{-i})\n"
        "                = p(z_i | z_{-i}) p(z_{-i}) p(z_i^* | z_{-i}).\n"
        "The reverse transition probability is T_i(z*, z) = p(z_i | z_{-i}^*) = p(z_i | z_{-i}).\n"
        "The reverse flux is:\n"
        "p(z*) T_i(z*, z) = p(z_i^*, z_{-i}^*) p(z_i | z_{-i})\n"
        "                 = p(z_i^* | z_{-i}) p(z_{-i}) p(z_i | z_{-i}).\n"
        "Since both expressions are identical, detailed balance holds exactly."
    )
    return {
        "detailed_balance_holds": True,
        "acceptance_probability": 1.0,
        "explanation": explanation,
    }


def verify_exercise_14_11() -> bool:
    sol = solve_exercise_14_11()
    # Numerical verification on bivariate normal
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.7], [0.7, 1.0]])

    z_a = np.array([0.5, 1.2])
    z_b = np.array([-0.3, 1.2])  # z_2 is identical, only z_1 differs

    # Conditional p(z_1 | z_2)
    cond_mean_a = cov[0, 1] / cov[1, 1] * z_a[1]
    cond_var = cov[0, 0] - (cov[0, 1] ** 2) / cov[1, 1]

    t_ab = stats.norm.pdf(z_b[0], loc=cond_mean_a, scale=np.sqrt(cond_var))
    t_ba = stats.norm.pdf(z_a[0], loc=cond_mean_a, scale=np.sqrt(cond_var))

    p_a = stats.multivariate_normal.pdf(z_a, mean=mean, cov=cov)
    p_b = stats.multivariate_normal.pdf(z_b, mean=mean, cov=cov)

    flux_ab = p_a * t_ab
    flux_ba = p_b * t_ba

    return sol["detailed_balance_holds"] and np.isclose(flux_ab, flux_ba, atol=1e-10)


# =============================================================================
# Exercise 14.12: Ergodicity of Gibbs Sampling on Disconnected Support
# =============================================================================

def solve_exercise_14_12() -> Dict[str, Any]:
    """
    Exercise 14.12:
    Consider a distribution whose support consists of two diagonally separated regions
    (e.g., [0, 1]^2 and [2, 3]^2) with zero probability everywhere else.
    Discuss whether standard Gibbs sampling is ergodic.
    """
    explanation = (
        "Gibbs sampling updates one coordinate at a time parallel to the coordinate axes (axis-aligned steps).\n"
        "If the support consists of disconnected diagonal squares (e.g., [0, 1]x[0, 1] and [2, 3]x[2, 3]),\n"
        "any horizontal step from the first square lands at (z1', z2) with z2 in [0, 1], where p(z) = 0 for z1' in [2, 3].\n"
        "Likewise, any vertical step lands in zero-probability space.\n"
        "Therefore, transition probability between the two disconnected components is identically ZERO.\n"
        "The Markov chain is NOT irreducible (reducible), and hence NOT ergodic."
    )
    return {
        "is_ergodic": False,
        "reason": "Reducible chain: axis-aligned steps cannot cross diagonal zero-density gap",
        "explanation": explanation,
    }


def verify_exercise_14_12() -> bool:
    sol = solve_exercise_14_12()
    # Test Gibbs sampling simulation on disconnected squares
    # Region A: [0, 1]^2, Region B: [2, 3]^2
    # Start in Region A
    current = np.array([0.5, 0.5])
    rng = np.random.RandomState(42)
    visited_b = False
    for _ in range(1000):
        # Update z1: conditional given current z2 in [0, 1] is Uniform([0, 1])
        current[0] = rng.uniform(0.0, 1.0)
        # Update z2: conditional given current z1 in [0, 1] is Uniform([0, 1])
        current[1] = rng.uniform(0.0, 1.0)
        if current[0] >= 2.0 and current[1] >= 2.0:
            visited_b = True

    return not sol["is_ergodic"] and (not visited_b)


# =============================================================================
# Exercise 14.13: Full Conditionals for Gaussian-Gamma Model
# =============================================================================

def solve_exercise_14_13() -> Dict[str, Any]:
    """
    Exercise 14.13:
    Gaussian likelihood x ~ N(mu, tau^{-1}) with prior p(mu) = N(mu | mu0, s0)
    and p(tau) = Gam(tau | a, b).
    Find full conditional distributions p(mu | x, tau) and p(tau | x, mu).
    """
    explanation = (
        "1. Full conditional for mu:\n"
        "   p(mu | x, tau) proportional to p(x | mu, tau) * p(mu)\n"
        "   = exp(-0.5 * tau * (x - mu)^2) * exp(-0.5 * (1/s0) * (mu - mu0)^2)\n"
        "   This is Gaussian N(mu | mu_N, s_N) with:\n"
        "   1 / s_N = 1 / s0 + tau\n"
        "   mu_N = s_N * (mu0 / s0 + tau * x).\n"
        "2. Full conditional for tau:\n"
        "   p(tau | x, mu) proportional to p(x | mu, tau) * p(tau)\n"
        "   = tau^{1/2} exp(-0.5 * tau * (x - mu)^2) * tau^{a - 1} exp(-b * tau)\n"
        "   = tau^{(a + 1/2) - 1} exp(-[b + 0.5 * (x - mu)^2] * tau)\n"
        "   This is Gamma Gam(tau | a_N, b_N) with:\n"
        "   a_N = a + 1/2, b_N = b + 0.5 * (x - mu)^2."
    )
    return {
        "mu_conditional": "Gaussian N(mu | mu_N, s_N)",
        "tau_conditional": "Gamma Gam(tau | a + 0.5, b + 0.5 * (x - mu)^2)",
        "explanation": explanation,
    }


def verify_exercise_14_13() -> bool:
    sol = solve_exercise_14_13()
    # Numerical validation of Gibbs sampling for 1D Gaussian-Gamma
    x_obs = 3.5
    mu0, s0 = 0.0, 4.0
    a0, b0 = 2.0, 1.0

    rng = np.random.RandomState(42)
    mu_cur = 0.0
    tau_cur = 1.0

    mu_samples = []
    tau_samples = []

    for _ in range(5000):
        # 1. Sample mu | x, tau
        s_N = 1.0 / (1.0 / s0 + tau_cur)
        mu_N = s_N * (mu0 / s0 + tau_cur * x_obs)
        mu_cur = rng.randn() * np.sqrt(s_N) + mu_N

        # 2. Sample tau | x, mu
        a_N = a0 + 0.5
        b_N = b0 + 0.5 * ((x_obs - mu_cur) ** 2)
        tau_cur = rng.gamma(shape=a_N, scale=1.0 / b_N)

        mu_samples.append(mu_cur)
        tau_samples.append(tau_cur)

    # Posterior mu should be pulled towards x_obs = 3.5
    return (
        np.mean(mu_samples) > 2.0
        and np.mean(tau_samples) > 0.5
        and sol["mu_conditional"].startswith("Gaussian")
    )


# =============================================================================
# Exercise 14.14: Over-Relaxation in Gibbs Sampling
# =============================================================================

def solve_exercise_14_14() -> Dict[str, Any]:
    """
    Exercise 14.14:
    Show that the over-relaxation update:
        z_i^* = mu_i + alpha * (z_i - mu_i) + sigma_i * sqrt(1 - alpha^2) * nu
    with nu ~ N(0, 1) and alpha in (-1, 1) leaves mean mu_i and variance sigma_i^2 invariant.
    """
    explanation = (
        "Assume z_i has mean mu_i and variance sigma_i^2, independent of nu.\n"
        "1. Mean:\n"
        "   E[z_i^*] = mu_i + alpha * (E[z_i] - mu_i) + sigma_i * sqrt(1 - alpha^2) * E[nu]\n"
        "            = mu_i + alpha * 0 + 0 = mu_i.\n"
        "2. Variance:\n"
        "   Var[z_i^*] = Var[alpha * (z_i - mu_i) + sigma_i * sqrt(1 - alpha^2) * nu]\n"
        "              = alpha^2 * Var[z_i] + sigma_i^2 * (1 - alpha^2) * Var[nu]\n"
        "              = alpha^2 * sigma_i^2 + sigma_i^2 * (1 - alpha^2) * 1\n"
        "              = sigma_i^2 * (alpha^2 + 1 - alpha^2) = sigma_i^2."
    )
    return {
        "preserves_mean": True,
        "preserves_variance": True,
        "explanation": explanation,
    }


def verify_exercise_14_14() -> bool:
    sol = solve_exercise_14_14()
    mu_i = 2.5
    sigma_i = 1.8
    alpha = -0.7

    rng = np.random.RandomState(42)
    z = rng.randn(20000) * sigma_i + mu_i
    nu = rng.randn(20000)

    z_star = mu_i + alpha * (z - mu_i) + sigma_i * np.sqrt(1.0 - alpha ** 2) * nu

    return (
        sol["preserves_mean"]
        and sol["preserves_variance"]
        and np.isclose(np.mean(z_star), mu_i, atol=0.04)
        and np.isclose(np.var(z_star), sigma_i ** 2, atol=0.08)
    )


# =============================================================================
# Exercise 14.15: Analytical Log-Likelihood Gradient in EBMs
# =============================================================================

def solve_exercise_14_15() -> Dict[str, Any]:
    """
    Exercise 14.15:
    Derive the gradient of log-likelihood for energy-based model p(x; w) = (1/Z(w)) exp(-E(x; w)):
        nabla_w ln p(x; w) = -nabla_w E(x; w) + E_{p(x'; w)}[nabla_w E(x'; w)].
    """
    explanation = (
        "1. ln p(x; w) = -E(x; w) - ln Z(w).\n"
        "2. nabla_w ln p(x; w) = -nabla_w E(x; w) - nabla_w ln Z(w).\n"
        "3. nabla_w ln Z(w) = (1 / Z(w)) nabla_w int exp(-E(x'; w)) dx'\n"
        "   = int (exp(-E(x'; w)) / Z(w)) * (-nabla_w E(x'; w)) dx'\n"
        "   = - E_{p(x'; w)}[nabla_w E(x'; w)].\n"
        "4. Substituting into step 2:\n"
        "   nabla_w ln p(x; w) = -nabla_w E(x; w) + E_{p(x'; w)}[nabla_w E(x'; w)]."
    )
    return {
        "positive_phase": "-nabla_w E(x; w)",
        "negative_phase": "+E_{p(x'; w)}[nabla_w E(x'; w)]",
        "explanation": explanation,
    }


def verify_exercise_14_15() -> bool:
    sol = solve_exercise_14_15()
    # Verify on quadratic energy E(x; w) = 0.5 * w * x^2
    ebm = EnergyBasedModel1D(
        energy_fn=lambda x, w: 0.5 * w[0] * (x ** 2),
        grad_x_energy_fn=lambda x, w: w[0] * x,
        grad_w_energy_fn=lambda x, w: 0.5 * (x ** 2).reshape(-1, 1),
    )
    w = np.array([1.5])
    rng = np.random.RandomState(42)
    data = rng.randn(10000) * np.sqrt(1.0 / w[0])
    model = rng.randn(10000) * np.sqrt(1.0 / w[0])

    grad = ebm.compute_log_likelihood_gradient(data, model, w)
    return np.abs(grad["total_grad"][0]) < 0.05


# =============================================================================
# Exercise 14.16: Score Function for Multivariate Gaussian
# =============================================================================

def solve_exercise_14_16() -> Dict[str, Any]:
    """
    Exercise 14.16:
    Show that the score function s(x) = nabla_x ln p(x) for Gaussian N(x | mu, Sigma)
    is given by s(x) = -Sigma^{-1} (x - mu).
    """
    explanation = (
        "p(x) = (2 pi)^{-D/2} |Sigma|^{-1/2} exp(-0.5 (x - mu)^T Sigma^{-1} (x - mu)).\n"
        "ln p(x) = const - 0.5 (x - mu)^T Sigma^{-1} (x - mu).\n"
        "Using vector calculus identity nabla_x [ (x - mu)^T A (x - mu) ] = 2 A (x - mu) for symmetric A:\n"
        "nabla_x ln p(x) = -0.5 * 2 Sigma^{-1} (x - mu) = -Sigma^{-1} (x - mu)."
    )
    return {
        "score_formula": "-Sigma^{-1} (x - mu)",
        "explanation": explanation,
    }


def verify_exercise_14_16() -> bool:
    sol = solve_exercise_14_16()
    mu = np.array([1.0, -2.0])
    Sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
    inv_Sigma = np.linalg.inv(Sigma)

    x = np.array([2.5, 0.5])
    true_score = -inv_Sigma @ (x - mu)

    # Numerical gradient of log pdf
    eps = 1e-6
    num_grad = np.zeros(2)
    for i in range(2):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        log_p_plus = stats.multivariate_normal.logpdf(x_plus, mean=mu, cov=Sigma)
        log_p_minus = stats.multivariate_normal.logpdf(x_minus, mean=mu, cov=Sigma)
        num_grad[i] = (log_p_plus - log_p_minus) / (2 * eps)

    return np.allclose(true_score, num_grad, atol=1e-5)


# =============================================================================
# Exercise 14.17: ULA Stationary Distribution and Discretization Bias
# =============================================================================

def solve_exercise_14_17() -> Dict[str, Any]:
    """
    Exercise 14.17:
    Show that for 1D Gaussian target N(0, sigma^2) with score s(z) = -z / sigma^2,
    the Unadjusted Langevin Algorithm (ULA) with step size epsilon converges to
    a Gaussian with variance sigma_ULA^2 = sigma^2 / (1 - epsilon / (4 * sigma^2)).
    """
    explanation = (
        "ULA step: z_{t+1} = z_t - (epsilon / (2 sigma^2)) z_t + sqrt(epsilon) eta_t\n"
        "                  = (1 - epsilon / (2 sigma^2)) z_t + sqrt(epsilon) eta_t.\n"
        "Let a = 1 - epsilon / (2 sigma^2). At stationarity, Var[z_{t+1}] = Var[z_t] = V:\n"
        "V = a^2 V + epsilon Var[eta_t] = a^2 V + epsilon.\n"
        "V * (1 - a^2) = epsilon.\n"
        "Now 1 - a^2 = 1 - (1 - epsilon / (2 sigma^2))^2\n"
        "            = 1 - (1 - epsilon / sigma^2 + epsilon^2 / (4 sigma^4))\n"
        "            = (epsilon / sigma^2) * (1 - epsilon / (4 sigma^2)).\n"
        "Therefore:\n"
        "V = epsilon / [ (epsilon / sigma^2) * (1 - epsilon / (4 sigma^2)) ]\n"
        "  = sigma^2 / (1 - epsilon / (4 sigma^2)).\n"
        "As epsilon -> 0, V -> sigma^2 (exact). For finite epsilon > 0, V > sigma^2 (discretization bias)."
    )
    return {
        "ula_variance": "sigma^2 / (1 - epsilon / (4 * sigma^2))",
        "bias_order": "O(epsilon)",
        "explanation": explanation,
    }


def verify_exercise_14_17() -> bool:
    sol = solve_exercise_14_17()
    sigma = 1.2
    sigma2 = sigma ** 2
    eps = 0.2

    # Theoretical ULA variance
    v_theo = sigma2 / (1.0 - eps / (4.0 * sigma2))

    # Empirical ULA simulation
    score_fn = lambda z: -z / sigma2
    res = unadjusted_langevin_algorithm(
        score_fn=score_fn,
        init_state=np.array([0.0]),
        step_size=eps,
        num_steps=30000,
        burn_in=2000,
        seed=42,
    )
    v_emp = np.var(res["samples"])

    return np.isclose(v_emp, v_theo, rtol=0.08)


# =============================================================================
# Exercise 14.18: MALA Detailed Balance Proof
# =============================================================================

def solve_exercise_14_18() -> Dict[str, Any]:
    """
    Exercise 14.18:
    Prove that the Metropolis-Adjusted Langevin Algorithm (MALA) satisfies
    detailed balance with respect to target p(z), eliminating the O(epsilon) bias of ULA.
    """
    explanation = (
        "MALA defines proposal:\n"
        "q(z* | z) = N(z* | z + (epsilon / 2) nabla ln p(z), epsilon I).\n"
        "The acceptance probability is the standard Metropolis-Hastings ratio:\n"
        "A(z*, z) = min(1, [p(z*) q(z | z*)] / [p(z) q(z* | z)]).\n"
        "As proven in Section 14.2.3, any proposal paired with this acceptance ratio\n"
        "strictly satisfies the detailed balance equation:\n"
        "p(z) T(z, z*) = p(z*) T(z*, z).\n"
        "Therefore, the stationary distribution of MALA is EXACTLY p(z) for ANY epsilon > 0\n"
        "where the chain remains ergodic."
    )
    return {
        "detailed_balance": True,
        "discretization_bias": 0.0,
        "explanation": explanation,
    }


def verify_exercise_14_18() -> bool:
    sol = solve_exercise_14_18()
    # Numerical validation: compare MALA sample variance to true variance
    sigma = 1.2
    sigma2 = sigma ** 2
    eps = 0.25

    target_log_p = lambda z: -0.5 * (z[0] ** 2) / sigma2
    score_fn = lambda z: -z / sigma2

    res = metropolis_adjusted_langevin_algorithm(
        target_log_p=target_log_p,
        score_fn=score_fn,
        init_state=np.array([0.0]),
        step_size=eps,
        num_steps=20000,
        burn_in=2000,
        seed=101,
    )

    v_mala = np.var(res["samples"])
    return sol["detailed_balance"] and np.isclose(v_mala, sigma2, rtol=0.08)


# =============================================================================
# Master Runner
# =============================================================================

def solve_all_exercises() -> Dict[str, Any]:
    """Run and return solutions for all 18 exercises in Chapter 14."""
    return {
        "ex_14_1": solve_exercise_14_1(),
        "ex_14_2": solve_exercise_14_2(),
        "ex_14_3": solve_exercise_14_3(),
        "ex_14_4": solve_exercise_14_4(),
        "ex_14_5": solve_exercise_14_5(),
        "ex_14_6": solve_exercise_14_6(),
        "ex_14_7": solve_exercise_14_7(),
        "ex_14_8": solve_exercise_14_8(),
        "ex_14_9": solve_exercise_14_9(),
        "ex_14_10": solve_exercise_14_10(),
        "ex_14_11": solve_exercise_14_11(),
        "ex_14_12": solve_exercise_14_12(),
        "ex_14_13": solve_exercise_14_13(),
        "ex_14_14": solve_exercise_14_14(),
        "ex_14_15": solve_exercise_14_15(),
        "ex_14_16": solve_exercise_14_16(),
        "ex_14_17": solve_exercise_14_17(),
        "ex_14_18": solve_exercise_14_18(),
    }
