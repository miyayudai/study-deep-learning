"""
common/basic_sampling.py
========================
Chapter 14: Sampling
Section 14.1: Basic Sampling Algorithms
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module provides comprehensive theoretical implementations, mathematical utilities,
and faithful figure reproduction generators for Section 14.1:
- 14.1.1 Expectations: Monte Carlo estimator, convergence, unbiasedness, and variance
- 14.1.2 Standard distributions: Inversion method (Exponential, Cauchy), Box-Muller transform,
  and uniform sampling on unit disk
- 14.1.3 Rejection sampling: General rejection sampler, Gamma distribution sampling with Cauchy proposal,
  and Gaussian bounded by Cauchy
- 14.1.4 Adaptive rejection sampling (ARS): Envelope construction for log-concave distributions
- 14.1.5 Importance sampling: Unnormalized & normalized importance sampling, weights, and Effective Sample Size (ESS)
- 14.1.6 Sampling-Importance-Resampling (SIR): Bootstrap resampling from candidate proposals
- Figures 14.1 - 14.8: Faithful reproductions matching Bishop & Bishop (2024)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
from pathlib import Path
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .plot_utils import setup_style, save_plot


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
# 1. Expectations & Monte Carlo Estimator (Section 14.1.1)
# =============================================================================

def monte_carlo_expectation(
    f: Callable[[np.ndarray], np.ndarray],
    samples: np.ndarray,
) -> Dict[str, float]:
    """
    Monte Carlo estimator of the expectation E[f(z)] = int f(z) p(z) dz (Eq 14.1):
        \\hat{f} = (1 / L) * sum_{l=1}^L f(z^{(l)})

    Parameters
    ----------
    f : Callable
        Function whose expectation is to be computed.
    samples : np.ndarray, shape (L,) or (L, D)
        Independent samples drawn from p(z).

    Returns
    -------
    Dict containing:
        - 'estimate': sample mean \\hat{f}
        - 'variance': empirical variance of f(z)
        - 'std_error': standard error of estimator sigma / sqrt(L)
        - 'ci_95': 95% confidence interval [low, high]
    """
    L = len(samples)
    f_vals = np.asarray(f(samples))
    estimate = float(np.mean(f_vals))
    variance = float(np.var(f_vals, ddof=1)) if L > 1 else 0.0
    std_error = float(np.sqrt(variance / L)) if L > 0 else 0.0
    ci_95 = (estimate - 1.96 * std_error, estimate + 1.96 * std_error)

    return {
        "estimate": estimate,
        "variance": variance,
        "std_error": std_error,
        "ci_95": ci_95,
        "num_samples": L,
    }


def convergence_analysis(
    f: Callable[[np.ndarray], np.ndarray],
    samples: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the running cumulative Monte Carlo estimate and confidence bounds:
        \\hat{f}_l = (1 / l) * sum_{i=1}^l f(z^{(i)})

    Returns
    -------
    running_mean : np.ndarray, shape (L,)
    running_std_err : np.ndarray, shape (L,)
    step_indices : np.ndarray, shape (L,)
    """
    f_vals = np.asarray(f(samples))
    L = len(f_vals)
    step_indices = np.arange(1, L + 1)
    cum_sum = np.cumsum(f_vals)
    running_mean = cum_sum / step_indices

    # Running sample variance
    cum_sum_sq = np.cumsum(f_vals ** 2)
    running_var = (cum_sum_sq - (cum_sum ** 2) / step_indices) / np.maximum(1, step_indices - 1)
    running_std_err = np.sqrt(np.maximum(0.0, running_var) / step_indices)

    return running_mean, running_std_err, step_indices


# =============================================================================
# 2. Standard Distributions & Inversion Method (Section 14.1.2)
# =============================================================================

def inverse_cdf_sample_exponential(
    lam: float,
    num_samples: int = 1000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample from Exponential distribution p(y) = lambda * exp(-lambda * y), y >= 0
    via the inversion method:
        F(y) = 1 - exp(-lambda * y) = u  ==>  y = -(1 / lambda) * ln(1 - u)
    """
    rng = np.random.RandomState(seed)
    u = rng.uniform(0.0, 1.0, size=num_samples)
    return -(1.0 / lam) * np.log(1.0 - u)


def inverse_cdf_sample_cauchy(
    x0: float = 0.0,
    gamma: float = 1.0,
    num_samples: int = 1000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample from Cauchy distribution p(y) = 1 / (pi * gamma * [1 + ((y - x0) / gamma)^2])
    via the inversion method:
        F(y) = 1/2 + (1/pi) * arctan((y - x0) / gamma) = u
        ==>  y = x0 + gamma * tan(pi * (u - 1/2))
    """
    rng = np.random.RandomState(seed)
    u = rng.uniform(0.0, 1.0, size=num_samples)
    return x0 + gamma * np.tan(np.pi * (u - 0.5))


def box_muller_transform(
    num_pairs: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Box-Muller algorithm for generating independent standard Gaussian pairs (z1, z2):
        z1 = sqrt(-2 ln u1) * cos(2 pi u2)
        z2 = sqrt(-2 ln u1) * sin(2 pi u2)
    where u1, u2 ~ Uniform(0, 1) are independent.
    """
    rng = np.random.RandomState(seed)
    u1 = rng.uniform(1e-10, 1.0, size=num_pairs)
    u2 = rng.uniform(0.0, 1.0, size=num_pairs)

    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2

    z1 = r * np.cos(theta)
    z2 = r * np.sin(theta)

    return z1, z2


def rejection_sample_unit_disk(
    num_samples: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Sample uniformly from the 2D unit disk z1^2 + z2^2 <= 1 via rejection sampling
    from the enclosing square [-1, 1] x [-1, 1] (Figure 14.3).

    Returns
    -------
    accepted : np.ndarray, shape (N_accepted, 2)
    acceptance_rate : float (theoretical value pi / 4 = ~0.7854)
    all_proposals : np.ndarray, shape (N_total, 2)
    mask_accepted : np.ndarray, shape (N_total,), dtype=bool
    """
    rng = np.random.RandomState(seed)
    # Estimate total needed proposals with safety margin
    total_needed = int(num_samples / (np.pi / 4.0) * 1.3) + 50
    proposals = rng.uniform(-1.0, 1.0, size=(total_needed, 2))
    r_sq = np.sum(proposals ** 2, axis=1)
    mask = r_sq <= 1.0

    accepted = proposals[mask][:num_samples]
    total_eval = np.where(np.cumsum(mask) == num_samples)[0]
    num_used = total_eval[0] + 1 if len(total_eval) > 0 else len(proposals)

    acc_rate = float(num_samples / num_used)
    return accepted, acc_rate, proposals[:num_used], mask[:num_used]


# =============================================================================
# 3. Rejection Sampling (Section 14.1.3)
# =============================================================================

class RejectionSampler:
    """
    General Rejection Sampling framework:
    Given unnormalized target \\tilde{p}(z) <= k * q(z),
    where q(z) is a proposal density from which we can sample directly.
    """
    def __init__(
        self,
        target_p_tilde: Callable[[np.ndarray], np.ndarray],
        proposal_q: Callable[[np.ndarray], np.ndarray],
        proposal_sample: Callable[[int, np.random.RandomState], np.ndarray],
        k: float,
    ):
        self.target_p_tilde = target_p_tilde
        self.proposal_q = proposal_q
        self.proposal_sample = proposal_sample
        self.k = float(k)

    def sample(
        self,
        num_samples: int,
        seed: Optional[int] = None,
        max_batch: int = 50000,
    ) -> Dict[str, Any]:
        """
        Generate samples from target p(z) = \\tilde{p}(z) / Z_p.

        Returns
        -------
        Dict with:
            - 'samples': accepted samples of shape (num_samples, ...)
            - 'acceptance_rate': float
            - 'total_proposed': int
            - 'proposals_history': list of proposed (z, u)
            - 'accepted_history': boolean mask of accepted
        """
        rng = np.random.RandomState(seed)
        accepted_samples: List[np.ndarray] = []
        all_z: List[np.ndarray] = []
        all_u: List[np.ndarray] = []
        all_acc: List[np.ndarray] = []

        total_proposed = 0

        while len(accepted_samples) < num_samples:
            batch_size = min(max_batch, max(100, int((num_samples - len(accepted_samples)) * self.k * 1.2)))
            z_batch = self.proposal_sample(batch_size, rng)
            total_proposed += batch_size

            # Sample u ~ Uniform(0, k * q(z))
            k_q_vals = self.k * self.proposal_q(z_batch)
            u_batch = rng.uniform(0.0, k_q_vals)
            p_tilde_vals = self.target_p_tilde(z_batch)

            accept_mask = u_batch <= p_tilde_vals

            # Record history for visualization / diagnostics
            all_z.append(z_batch)
            all_u.append(u_batch)
            all_acc.append(accept_mask)

            for z_val in z_batch[accept_mask]:
                accepted_samples.append(z_val)
                if len(accepted_samples) == num_samples:
                    break

        samples_arr = np.array(accepted_samples[:num_samples])
        all_acc_arr = np.concatenate(all_acc)
        acc_rate = float(np.mean(all_acc_arr))

        return {
            "samples": samples_arr,
            "acceptance_rate": acc_rate,
            "total_proposed": total_proposed,
            "all_z": np.concatenate(all_z),
            "all_u": np.concatenate(all_u),
            "all_accepted": all_acc_arr,
        }


def rejection_sample_gamma(
    a: float = 10.0,
    b: float = 1.0,
    num_samples: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Rejection sampling for Gamma distribution Gam(z | a, b) with a > 1 (Figure 14.5).
    Using Cauchy proposal distribution matching the peak at mode (a - 1) / b.
    """
    mode = (a - 1.0) / b
    gamma_width = np.sqrt(2.0 * a - 1.0) / b

    def target_p(z: np.ndarray) -> np.ndarray:
        out = np.zeros_like(z, dtype=float)
        valid = z > 0
        out[valid] = stats.gamma.pdf(z[valid], a=a, scale=1.0 / b)
        return out

    def proposal_q(z: np.ndarray) -> np.ndarray:
        return stats.cauchy.pdf(z, loc=mode, scale=gamma_width)

    def sample_q(n: int, rng: np.random.RandomState) -> np.ndarray:
        return rng.standard_cauchy(size=n) * gamma_width + mode

    # Determine scale factor k so that k * q(z) >= p(z)
    z_grid = np.linspace(0.01, 35.0, 2000)
    ratios = target_p(z_grid) / np.maximum(1e-12, proposal_q(z_grid))
    k = float(np.max(ratios) * 1.02)

    sampler = RejectionSampler(target_p, proposal_q, sample_q, k)
    res = sampler.sample(num_samples, seed=seed)
    res["k"] = k
    res["mode"] = mode
    res["gamma_width"] = gamma_width
    return res


def rejection_sample_gaussian_cauchy(
    num_samples: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Rejection sampling of standard normal N(0, 1) bounded by Cauchy proposal (Figure 14.7):
        p(z) = (1 / sqrt(2 pi)) * exp(-z^2 / 2)
        q(z) = 1 / (pi * gamma * [1 + (z / gamma)^2]) with gamma = sqrt(2)
        k = sqrt(pi) =~ 1.7725
    This guarantees k * q(0) = p(0) and k * q(z) >= p(z) for all z,
    giving theoretical acceptance rate 1 / k = 1 / sqrt(pi) =~ 0.5642.
    """
    gamma = float(np.sqrt(2.0))
    k = float(np.sqrt(np.pi))

    def target_p(z: np.ndarray) -> np.ndarray:
        return stats.norm.pdf(z, loc=0.0, scale=1.0)

    def proposal_q(z: np.ndarray) -> np.ndarray:
        return stats.cauchy.pdf(z, loc=0.0, scale=gamma)

    def sample_q(n: int, rng: np.random.RandomState) -> np.ndarray:
        return rng.standard_cauchy(size=n) * gamma

    sampler = RejectionSampler(target_p, proposal_q, sample_q, k)
    res = sampler.sample(num_samples, seed=seed)
    res["theoretical_acc_rate"] = float(1.0 / np.sqrt(np.pi))
    return res


# =============================================================================
# 4. Adaptive Rejection Sampling (ARS) (Section 14.1.4)
# =============================================================================

class AdaptiveRejectionSampler:
    """
    Adaptive Rejection Sampling (ARS) for log-concave distributions (Figure 14.6):
        d^2 / dz^2 ln \\tilde{p}(z) <= 0
    Constructs piecewise linear upper envelope in log-space from tangent lines,
    proposes from piecewise exponential envelope, and adaptively refines knots upon rejection.
    """
    def __init__(
        self,
        log_p_tilde: Callable[[np.ndarray], np.ndarray],
        d_log_p_tilde: Callable[[np.ndarray], np.ndarray],
        initial_points: List[float],
        domain_bounds: Tuple[float, float] = (-5.0, 5.0),
    ):
        self.log_p_tilde = log_p_tilde
        self.d_log_p_tilde = d_log_p_tilde
        self.support_points = sorted(list(initial_points))
        self.domain_bounds = domain_bounds

    def _compute_envelope_lines(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute tangent slopes and intercepts:
            tangent_i(z) = a_i * z + b_i
        and intersection points z_{i, i+1}^*.
        """
        z_pts = np.array(self.support_points)
        slopes = np.asarray(self.d_log_p_tilde(z_pts))
        log_p = np.asarray(self.log_p_tilde(z_pts))
        intercepts = log_p - slopes * z_pts

        intersections = []
        for i in range(len(z_pts) - 1):
            if np.isclose(slopes[i], slopes[i + 1]):
                intersections.append((z_pts[i] + z_pts[i + 1]) / 2.0)
            else:
                intersections.append((intercepts[i + 1] - intercepts[i]) / (slopes[i] - slopes[i + 1]))

        return slopes, intercepts, np.array(intersections)

    def evaluate_envelope(self, z: np.ndarray) -> np.ndarray:
        """Evaluate log-upper envelope at points z."""
        slopes, intercepts, intersections = self._compute_envelope_lines()
        z_flat = np.atleast_1d(z)
        env = np.zeros_like(z_flat, dtype=float)

        for idx, val in enumerate(z_flat):
            segment_idx = 0
            while segment_idx < len(intersections) and val > intersections[segment_idx]:
                segment_idx += 1
            env[idx] = slopes[segment_idx] * val + intercepts[segment_idx]

        return env.reshape(np.shape(z))

    def sample(self, num_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate samples from the log-concave distribution using piecewise exponential
        envelope sampling and rejection with adaptive refinement.
        """
        rng = np.random.RandomState(seed)
        samples = []

        while len(samples) < num_samples:
            slopes, intercepts, intersections = self._compute_envelope_lines()
            M = len(slopes)
            knots = np.concatenate([[self.domain_bounds[0]], intersections, [self.domain_bounds[1]]])

            # Compute segment integrals
            areas = []
            for i in range(M):
                a, b = slopes[i], intercepts[i]
                x0, x1 = knots[i], knots[i + 1]
                if abs(a) < 1e-8:
                    area = np.exp(b) * (x1 - x0)
                else:
                    max_v = max(a * x0 + b, a * x1 + b)
                    area = np.exp(max_v) * (np.exp(a * x1 + b - max_v) - np.exp(a * x0 + b - max_v)) / a
                areas.append(max(1e-12, float(area)))

            probs = np.array(areas) / np.sum(areas)

            # Draw proposal from chosen segment
            seg_idx = rng.choice(M, p=probs)
            a, b = slopes[seg_idx], intercepts[seg_idx]
            x0, x1 = knots[seg_idx], knots[seg_idx + 1]
            u = rng.uniform(0.0, 1.0)
            if abs(a) < 1e-8:
                z_prop = x0 + u * (x1 - x0)
            else:
                max_v = max(a * x0 + b, a * x1 + b)
                term0 = np.exp(a * x0 + b - max_v)
                term1 = np.exp(a * x1 + b - max_v)
                val = max(1e-12, term0 + u * (term1 - term0))
                z_prop = (np.log(val) + max_v - b) / a

            log_env = a * z_prop + b
            log_p = float(self.log_p_tilde(z_prop))

            # Accept/reject test
            if np.log(rng.uniform(0.0, 1.0)) <= log_p - log_env:
                samples.append(z_prop)
            else:
                # Refine envelope by adding rejected point
                if len(self.support_points) < 40 and not any(np.isclose(z_prop, pt, atol=0.05) for pt in self.support_points):
                    self.support_points.append(float(z_prop))
                    self.support_points.sort()

        return np.array(samples[:num_samples])


# =============================================================================
# 5. Importance Sampling (Section 14.1.5)
# =============================================================================

def compute_effective_sample_size(weights: np.ndarray) -> float:
    """
    Effective Sample Size (ESS) for importance weights:
        ESS = (sum_l w_l)^2 / sum_l w_l^2 = 1 / sum_l W_l^2
    where W_l = w_l / sum_m w_m are the normalized weights.
    """
    sum_w = np.sum(weights)
    if sum_w <= 0.0:
        return 0.0
    W = weights / sum_w
    return float(1.0 / np.sum(W ** 2))


def importance_sampling(
    f: Callable[[np.ndarray], np.ndarray],
    target_p: Callable[[np.ndarray], np.ndarray],
    proposal_q: Callable[[np.ndarray], np.ndarray],
    sample_q: Callable[[int, np.random.RandomState], np.ndarray],
    num_samples: int = 5000,
    is_normalized: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Importance Sampling estimator:
        - If p and q are normalized (is_normalized=True):
            \\hat{f} = (1 / L) * sum_{l=1}^L w(z^{(l)}) * f(z^{(l)}),  w = p / q
        - If p or q are unnormalized (is_normalized=False, ratio estimator):
            \\hat{f} = sum_{l=1}^L W_l * f(z^{(l)}),  W_l = \\tilde{w}_l / sum \\tilde{w}_m

    Returns
    -------
    Dict containing:
        - 'estimate': float
        - 'variance': empirical variance
        - 'ess': effective sample size
        - 'weights': unnormalized weights
        - 'norm_weights': normalized weights
        - 'samples': proposed samples
    """
    rng = np.random.RandomState(seed)
    z_samples = sample_q(num_samples, rng)
    f_vals = np.asarray(f(z_samples))

    p_vals = np.asarray(target_p(z_samples))
    q_vals = np.asarray(proposal_q(z_samples))

    weights = p_vals / np.maximum(1e-12, q_vals)
    sum_w = np.sum(weights)
    norm_weights = weights / sum_w if sum_w > 0 else np.ones(num_samples) / num_samples

    if is_normalized:
        estimate = float(np.mean(weights * f_vals))
    else:
        estimate = float(np.sum(norm_weights * f_vals))

    ess = compute_effective_sample_size(weights)

    return {
        "estimate": estimate,
        "ess": ess,
        "weights": weights,
        "norm_weights": norm_weights,
        "samples": z_samples,
        "num_samples": num_samples,
    }


# =============================================================================
# 6. Sampling-Importance-Resampling (SIR) (Section 14.1.6)
# =============================================================================

def sampling_importance_resampling(
    target_p: Callable[[np.ndarray], np.ndarray],
    proposal_q: Callable[[np.ndarray], np.ndarray],
    sample_q: Callable[[int, np.random.RandomState], np.ndarray],
    num_candidates: int = 5000,
    num_resamples: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Sampling-Importance-Resampling (SIR / weighted bootstrap):
    1. Propose L candidate points z^{(1)}, ..., z^{(L)} ~ q(z).
    2. Compute unnormalized weights w_l = p(z^{(l)}) / q(z^{(l)}).
    3. Compute normalized weights W_l = w_l / sum_m w_m.
    4. Resample M points z^{*(1)}, ..., z^{*(M)} from {z^{(l)}} with replacement using {W_l}.

    As L -> inf, distribution of resampled points approaches p(z).
    """
    rng = np.random.RandomState(seed)
    candidates = sample_q(num_candidates, rng)
    p_vals = np.asarray(target_p(candidates))
    q_vals = np.asarray(proposal_q(candidates))

    weights = p_vals / np.maximum(1e-12, q_vals)
    sum_w = np.sum(weights)
    norm_weights = weights / sum_w

    resampled_indices = rng.choice(num_candidates, size=num_resamples, replace=True, p=norm_weights)
    resamples = candidates[resampled_indices]

    return {
        "resamples": resamples,
        "candidates": candidates,
        "norm_weights": norm_weights,
        "ess": compute_effective_sample_size(weights),
        "num_unique_resamples": len(np.unique(resampled_indices)),
    }


# =============================================================================
# 7. Faithful Reproduction of Chapter 14 Figures (Figures 14.1 - 14.8)
# =============================================================================

def _get_bimodal_density() -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Return bimodal density p(z) and integrand f(z) matching Figures 14.1 and 14.8."""
    def p(z: np.ndarray) -> np.ndarray:
        return (
            0.45 * stats.norm.pdf(z, loc=-1.4, scale=0.45)
            + 0.55 * stats.norm.pdf(z, loc=1.3, scale=0.65)
        )

    # Monotonic smooth curve crossing zero between the modes
    def f(z: np.ndarray) -> np.ndarray:
        return 0.52 * np.tanh(0.7 * z) + 0.05

    return p, f


def generate_figure_14_1(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Faithful reproduction of Figure 14.1:
    Expectation of a function f(z) under probability density p(z).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)

    z = np.linspace(-3.5, 3.5, 1000)
    p_func, f_func = _get_bimodal_density()

    p_vals = p_func(z)
    f_vals = f_func(z)

    # Plot red density curve and blue function curve
    ax.plot(z, p_vals, color="#e53935", linewidth=2.5, label="p(z)")
    ax.plot(z, f_vals, color="#1e88e5", linewidth=2.5, label="f(z)")

    # Baseline axis with arrow
    ax.axhline(0, color="#263238", linewidth=1.5)
    ax.annotate(
        "", xy=(3.6, 0), xytext=(3.4, 0),
        arrowprops=dict(arrowstyle="-|>", color="#263238", lw=1.5, mutation_scale=15),
    )
    ax.text(3.65, -0.04, "$z$", fontsize=14, fontstyle="italic")

    # Labels directly on curves matching textbook style
    ax.text(-2.0, 0.35, "$p(z)$", fontsize=15, fontstyle="italic", ha="center")
    ax.text(3.2, 0.48, "$f(z)$", fontsize=15, fontstyle="italic", ha="center")

    ax.set_xlim(-3.6, 3.8)
    ax.set_ylim(-0.45, 0.55)
    ax.axis("off")
    plt.tight_layout()

    _save_figure(fig, "fig_14_1_expectation", save_dir)
    return fig


def generate_figure_14_2(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Faithful reproduction of Figure 14.2:
    Transformation / Inversion method with probability density p(y) and CDF h(y).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.8), dpi=300)

    y = np.linspace(-3.5, 3.5, 1000)
    p_func, _ = _get_bimodal_density()

    # Normalize p so that total integral is exactly 1
    dy = y[1] - y[0]
    p_vals = p_func(y)
    p_vals = p_vals / np.sum(p_vals * dy)

    # Cumulative distribution function h(y)
    h_vals = np.cumsum(p_vals) * dy

    # Plot red density and blue CDF
    ax.plot(y, p_vals, color="#e53935", linewidth=2.5, label="p(y)")
    ax.plot(y, h_vals, color="#1e88e5", linewidth=2.5, label="h(y)")

    # Dashed line at h = 1 from y-axis to right
    ax.plot([-3.5, 3.5], [1.0, 1.0], color="#263238", linestyle="--", linewidth=1.2)

    # Coordinate axes with arrows
    ax.annotate(
        "", xy=(3.6, 0), xytext=(-3.5, 0),
        arrowprops=dict(arrowstyle="-|>", color="#263238", lw=1.8, mutation_scale=15),
    )
    ax.annotate(
        "", xy=(-3.5, 1.15), xytext=(-3.5, -0.05),
        arrowprops=dict(arrowstyle="-|>", color="#263238", lw=1.8, mutation_scale=15),
    )

    # Ticks and labels
    ax.text(-3.7, 1.0, "1", fontsize=14, ha="right", va="center")
    ax.text(-3.7, 0.0, "0", fontsize=14, ha="right", va="center")
    ax.text(3.65, -0.05, "$y$", fontsize=14, fontstyle="italic")

    # Curve labels
    ax.text(-2.3, 0.45, "$p(y)$", fontsize=15, fontstyle="italic")
    ax.text(2.1, 0.85, "$h(y)$", fontsize=15, fontstyle="italic")

    ax.set_xlim(-4.0, 3.9)
    ax.set_ylim(-0.1, 1.25)
    ax.axis("off")
    plt.tight_layout()

    _save_figure(fig, "fig_14_2_transformation_method", save_dir)
    return fig


def generate_figure_14_3(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Faithful reproduction of Figure 14.3:
    Rejection sampling of uniform points on the 2D unit disk enclosed by [-1, 1]^2.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=300)

    # Enclosing square [-1, 1] x [-1, 1] with blue border
    rect = patches.Rectangle((-1, -1), 2, 2, fill=False, edgecolor="#1e88e5", linewidth=2.5, zorder=2)
    ax.add_patch(rect)

    # Unit circle with light green fill and green boundary
    circle = patches.Circle((0, 0), 1.0, facecolor="#c8e6c9", edgecolor="#43a047", linewidth=2.5, zorder=3)
    ax.add_patch(circle)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")

    # Labels matching Figure 14.3
    ax.text(-1.18, -1.0, "$-1$", fontsize=13, ha="right", va="center")
    ax.text(-1.18, 1.0, "$1$", fontsize=13, ha="right", va="center")
    ax.text(-1.0, -1.18, "$-1$", fontsize=13, ha="center", va="top")
    ax.text(1.0, -1.18, "$1$", fontsize=13, ha="center", va="top")
    ax.text(0.0, -1.22, "$z_1$", fontsize=14, fontstyle="italic", ha="center")
    ax.text(-1.22, 0.0, "$z_2$", fontsize=14, fontstyle="italic", va="center")

    ax.axis("off")
    plt.tight_layout()

    _save_figure(fig, "fig_14_3_unit_disk_sampling", save_dir)
    return fig


def generate_figure_14_4(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Faithful reproduction of Figure 14.4:
    Rejection sampling geometry showing proposal k*q(z), target \\tilde{p}(z),
    and sample (z0, u0).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=300)

    z = np.linspace(-3.5, 3.5, 1000)

    # Target unnormalized distribution \\tilde{p}(z)
    p_tilde = (
        0.18 * stats.norm.pdf(z, loc=-1.1, scale=0.45)
        + 0.32 * stats.norm.pdf(z, loc=0.8, scale=0.60)
    )

    # Proposal distribution k * q(z) (Gaussian completely bounding target)
    k_q = 0.38 * stats.norm.pdf(z, loc=0.1, scale=1.45) / stats.norm.pdf(0.1, loc=0.1, scale=1.45)

    # Fill gray rejection region between \\tilde{p}(z) and k*q(z)
    ax.fill_between(z, p_tilde, k_q, color="#cfd8dc", alpha=0.9, zorder=1)
    ax.fill_between(z, 0, p_tilde, color="#ffffff", zorder=2)

    # Curves
    ax.plot(z, k_q, color="#1e88e5", linewidth=2.2, zorder=3)
    ax.plot(z, p_tilde, color="#e53935", linewidth=2.5, zorder=4)

    # Sample z0 and vertical segment
    z0 = -1.1
    k_q_z0 = float(0.38 * stats.norm.pdf(z0, loc=0.1, scale=1.45) / stats.norm.pdf(0.1, loc=0.1, scale=1.45))
    u0 = 0.06

    ax.plot([z0, z0], [0, k_q_z0], color="#263238", linewidth=1.6, zorder=5)
    ax.scatter([z0], [k_q_z0], color="#263238", s=20, zorder=6)
    ax.scatter([z0], [u0], color="#263238", s=20, zorder=6)

    # Baseline axis
    ax.plot([-3.8, 3.8], [0, 0], color="#263238", linewidth=1.8, zorder=3)
    ax.plot([-4.2, -3.8], [0, 0], color="#263238", linestyle="--", linewidth=1.5)
    ax.plot([3.8, 4.2], [0, 0], color="#263238", linestyle="--", linewidth=1.5)

    # Annotations matching Figure 14.4
    ax.text(z0 - 0.12, k_q_z0 + 0.015, "$kq(z_0)$", fontsize=14, fontstyle="italic", ha="right")
    ax.text(z0 + 0.12, u0, "$u_0$", fontsize=14, fontstyle="italic", ha="left", va="center")
    ax.text(z0, -0.045, "$z_0$", fontsize=14, fontstyle="italic", ha="center")
    ax.text(0.7, 0.40, "$kq(z)$", fontsize=15, fontstyle="italic")
    ax.text(0.8, 0.12, "$\\widetilde{p}(z)$", fontsize=15, fontstyle="italic")
    ax.text(3.9, -0.045, "$z$", fontsize=14, fontstyle="italic")

    ax.set_xlim(-4.3, 4.3)
    ax.set_ylim(-0.07, 0.45)
    ax.axis("off")
    plt.tight_layout()

    _save_figure(fig, "fig_14_4_rejection_sampling", save_dir)
    return fig


def generate_figure_14_5(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Faithful reproduction of Figure 14.5:
    Rejection sampling for a Gamma distribution Gam(z | a=10, b=1) using Cauchy proposal.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)

    z = np.linspace(0.0, 30.0, 1000)
    a, b = 10.0, 1.0

    # Gamma target distribution
    gamma_pdf = stats.gamma.pdf(z, a=a, scale=1.0 / b)

    # Cauchy proposal centered at mode with optimal scale parameter sqrt(2a-1)
    mode = (a - 1.0) / b
    gamma_scale = np.sqrt(2.0 * a - 1.0) / b
    cauchy_pdf = stats.cauchy.pdf(z, loc=mode, scale=gamma_scale)

    # Scale factor k so that peak height matches exactly
    k = float(np.max(gamma_pdf) / np.max(cauchy_pdf))
    proposal_bound = k * cauchy_pdf

    ax.plot(z, proposal_bound, color="#e53935", linewidth=2.2, label="$k q(z)$")
    ax.plot(z, gamma_pdf, color="#00e676", linewidth=2.4, label="$p(z)$")

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 0.15)
    ax.set_xlabel("$z$", fontsize=13, fontstyle="italic")
    ax.set_ylabel("$p(z)$", fontsize=13, fontstyle="italic", rotation=0, labelpad=15)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks([0, 0.05, 0.10, 0.15])

    plt.tight_layout()
    _save_figure(fig, "fig_14_5_rejection_gamma", save_dir)
    return fig


def generate_figure_14_6(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Faithful reproduction of Figure 14.6:
    Adaptive Rejection Sampling (ARS) for log-concave distributions.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)

    z = np.linspace(-2.5, 2.5, 1000)
    # Concave log-density: peak near 0.2
    c = 0.2
    log_p = -0.5 * (z - c) ** 2

    # Three support points: z1 to left, z2 to right of peak, z3 further right
    z_pts = np.array([-1.2, 0.6, 1.5])
    slopes = -(z_pts - c)
    intercepts = -0.5 * (z_pts - c) ** 2 - slopes * z_pts

    # Intersections
    z_int1 = (intercepts[1] - intercepts[0]) / (slopes[0] - slopes[1])
    z_int2 = (intercepts[2] - intercepts[1]) / (slopes[1] - slopes[2])

    # Construct envelope
    env = np.zeros_like(z)
    for i, val in enumerate(z):
        if val <= z_int1:
            env[i] = slopes[0] * val + intercepts[0]
        elif val <= z_int2:
            env[i] = slopes[1] * val + intercepts[1]
        else:
            env[i] = slopes[2] * val + intercepts[2]

    # Baseline y level
    y_base = -3.2

    # Plot concave density in red and piecewise envelope in blue
    ax.plot(z, log_p, color="#e53935", linewidth=2.2, label="$\\ln p(z)$")
    ax.plot(z, env, color="#1e88e5", linewidth=2.2, label="Envelope")

    # Dashed vertical lines from support points down to x-axis
    for pt in z_pts:
        y_val = -0.5 * (pt - c) ** 2
        ax.plot([pt, pt], [y_base, y_val], color="#1e88e5", linestyle="--", linewidth=1.5)

    # Coordinate axes
    ax.annotate(
        "", xy=(2.7, y_base), xytext=(-2.7, y_base),
        arrowprops=dict(arrowstyle="-|>", color="#263238", lw=1.8, mutation_scale=15),
    )
    ax.annotate(
        "", xy=(-2.6, 0.5), xytext=(-2.6, y_base),
        arrowprops=dict(arrowstyle="-|>", color="#263238", lw=1.8, mutation_scale=15),
    )

    # Labels
    ax.text(-2.7, 0.45, "$\\ln p(z)$", fontsize=14, fontstyle="italic", ha="right")
    ax.text(2.7, y_base - 0.2, "$z$", fontsize=14, fontstyle="italic")
    ax.text(z_pts[0], y_base - 0.35, "$z_1$", fontsize=13, fontstyle="italic", ha="center")
    ax.text(z_pts[1], y_base - 0.35, "$z_2$", fontsize=13, fontstyle="italic", ha="center")
    ax.text(z_pts[2], y_base - 0.35, "$z_3$", fontsize=13, fontstyle="italic", ha="center")

    ax.set_xlim(-2.9, 2.9)
    ax.set_ylim(-3.7, 0.6)
    ax.axis("off")
    plt.tight_layout()

    _save_figure(fig, "fig_14_6_adaptive_rejection_sampling", save_dir)
    return fig


def generate_figure_14_7(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Faithful reproduction of Figure 14.7:
    Comparison of Gaussian target distribution and bounding Cauchy distribution.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)

    z = np.linspace(-5.0, 5.0, 1000)
    p_norm = stats.norm.pdf(z, loc=0.0, scale=1.0)

    # Cauchy proposal with scale gamma = sqrt(2) and k = sqrt(pi)
    # This guarantees k*q(0) = p(0) and curvature at 0 matches,
    # so k*q(z) >= p(z) holds strictly for all z.
    gamma = np.sqrt(2.0)
    k = np.sqrt(np.pi)
    k_q_cauchy = k * stats.cauchy.pdf(z, loc=0.0, scale=gamma)

    ax.plot(z, k_q_cauchy, color="#e53935", linewidth=2.2, label="$k q(z)$ (Cauchy)")
    ax.plot(z, p_norm, color="#00e676", linewidth=2.4, label="$p(z)$ (Gaussian)")

    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("$z$", fontsize=13, fontstyle="italic")
    ax.set_ylabel("$p(z)$", fontsize=13, fontstyle="italic", rotation=0, labelpad=15)
    ax.set_xticks([-5, 0, 5])
    ax.set_yticks([0, 0.25, 0.5])

    plt.tight_layout()
    _save_figure(fig, "fig_14_7_gaussian_cauchy_rejection", save_dir)
    return fig


def generate_figure_14_8(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Faithful reproduction of Figure 14.8:
    Importance sampling geometry with target p(z), proposal q(z), and integrand f(z).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)

    z = np.linspace(-3.5, 3.5, 1000)
    p_func, f_func = _get_bimodal_density()

    p_vals = p_func(z)
    f_vals = f_func(z)

    # Proposal q(z): broad Gaussian centered near the middle
    q_vals = stats.norm.pdf(z, loc=-0.2, scale=1.35)

    # Plot red target, green proposal, blue function
    ax.plot(z, p_vals, color="#e53935", linewidth=2.5, label="p(z)")
    ax.plot(z, q_vals, color="#00e676", linewidth=2.5, label="q(z)")
    ax.plot(z, f_vals, color="#1e88e5", linewidth=2.5, label="f(z)")

    # Baseline axis with arrow
    ax.axhline(0, color="#263238", linewidth=1.5)
    ax.annotate(
        "", xy=(3.6, 0), xytext=(3.4, 0),
        arrowprops=dict(arrowstyle="-|>", color="#263238", lw=1.5, mutation_scale=15),
    )
    ax.text(3.65, -0.04, "$z$", fontsize=14, fontstyle="italic")

    # Labels directly on curves matching textbook style
    ax.text(-2.0, 0.35, "$p(z)$", fontsize=15, fontstyle="italic", ha="center")
    ax.text(-0.2, 0.32, "$q(z)$", fontsize=15, fontstyle="italic", ha="center")
    ax.text(3.2, 0.48, "$f(z)$", fontsize=15, fontstyle="italic", ha="center")

    ax.set_xlim(-3.6, 3.8)
    ax.set_ylim(-0.45, 0.55)
    ax.axis("off")
    plt.tight_layout()

    _save_figure(fig, "fig_14_8_importance_sampling", save_dir)
    return fig


def generate_all_section_14_1_figures(save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
    """Generate and save all 8 figures (Figures 14.1 - 14.8) for Section 14.1."""
    return {
        "fig_14_1": generate_figure_14_1(save_dir),
        "fig_14_2": generate_figure_14_2(save_dir),
        "fig_14_3": generate_figure_14_3(save_dir),
        "fig_14_4": generate_figure_14_4(save_dir),
        "fig_14_5": generate_figure_14_5(save_dir),
        "fig_14_6": generate_figure_14_6(save_dir),
        "fig_14_7": generate_figure_14_7(save_dir),
        "fig_14_8": generate_figure_14_8(save_dir),
    }
