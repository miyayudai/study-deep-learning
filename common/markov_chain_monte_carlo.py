"""
common/markov_chain_monte_carlo.py
==================================
Chapter 14: Sampling
Section 14.2: Markov Chain Monte Carlo (MCMC)
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module provides comprehensive theoretical implementations, mathematical utilities,
and faithful figure reproduction generators for Section 14.2:
- 14.2.1 The Metropolis algorithm: Random-walk Metropolis, acceptance ratio, detailed balance
- 14.2.2 Markov chains: Transition kernels, stationary distributions, ergodicity, autocorrelation
- 14.2.3 The Metropolis–Hastings algorithm: Non-symmetric proposals, Hastings ratio, diffusion scaling (L/l)^2
- 14.2.4 Gibbs sampling: Coordinate-wise exact sampling, Markov blanket conditioning
- 14.2.5 Ancestral sampling: Forward sampling in directed graphical models
- Figures 14.9 - 14.12: Faithful reproductions matching Bishop & Bishop (2024)
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
# 1. The Metropolis Algorithm (Section 14.2.1)
# =============================================================================

def metropolis_sampler(
    target_log_p: Callable[[np.ndarray], float],
    proposal_std: float,
    num_samples: int,
    init_state: np.ndarray,
    burn_in: int = 500,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Random-walk Metropolis algorithm with symmetric Gaussian proposal:
        q(z^* | z) = N(z^* | z, proposal_std^2 * I)
        A(z^*, z) = min(1, p(z^*) / p(z)) = min(1, exp(log_p(z^*) - log_p(z)))
    """
    rng = np.random.RandomState(seed)
    current = np.array(init_state, dtype=float)
    current_log_p = float(target_log_p(current))

    dim = len(current)
    total_steps = burn_in + num_samples

    samples = np.zeros((num_samples, dim))
    accepted_count = 0

    all_accepted_steps: List[Tuple[np.ndarray, np.ndarray]] = []
    all_rejected_steps: List[Tuple[np.ndarray, np.ndarray]] = []

    for step in range(total_steps):
        proposal = current + rng.randn(dim) * proposal_std
        prop_log_p = float(target_log_p(proposal))

        log_alpha = prop_log_p - current_log_p
        u = rng.uniform(0.0, 1.0)

        if np.log(u) <= log_alpha:
            if step >= burn_in:
                all_accepted_steps.append((current.copy(), proposal.copy()))
            current = proposal
            current_log_p = prop_log_p
            accepted_count += 1
        else:
            if step >= burn_in:
                all_rejected_steps.append((current.copy(), proposal.copy()))

        if step >= burn_in:
            samples[step - burn_in] = current

    acc_rate = float(accepted_count / total_steps)

    return {
        "samples": samples,
        "acceptance_rate": acc_rate,
        "accepted_steps": all_accepted_steps,
        "rejected_steps": all_rejected_steps,
    }


def metropolis_sample_2d_gaussian(
    mean: np.ndarray,
    cov: np.ndarray,
    num_steps: int = 80,
    proposal_std: float = 0.22,
    init_state: Optional[np.ndarray] = None,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Metropolis sampling on a 2D Gaussian matching Figure 14.9.
    Records every individual accepted step and rejected proposal.
    """
    rng = np.random.RandomState(seed)
    inv_cov = np.linalg.inv(cov)

    def log_p(z: np.ndarray) -> float:
        diff = z - mean
        return -0.5 * float(diff.T @ inv_cov @ diff)

    if init_state is None:
        init_state = mean.copy() + rng.randn(2) * 0.3

    current = np.array(init_state, dtype=float)
    current_log_p = log_p(current)

    trajectory = [current.copy()]
    accepted_steps: List[Tuple[np.ndarray, np.ndarray]] = []
    rejected_proposals: List[Tuple[np.ndarray, np.ndarray]] = []

    for _ in range(num_steps):
        proposal = current + rng.randn(2) * proposal_std
        prop_log_p = log_p(proposal)

        log_alpha = prop_log_p - current_log_p
        u = rng.uniform(0.0, 1.0)

        if np.log(u) <= log_alpha:
            accepted_steps.append((current.copy(), proposal.copy()))
            current = proposal
            current_log_p = prop_log_p
            trajectory.append(current.copy())
        else:
            rejected_proposals.append((current.copy(), proposal.copy()))
            trajectory.append(current.copy())

    acc_rate = float(len(accepted_steps) / num_steps)

    return {
        "trajectory": np.array(trajectory),
        "accepted_steps": accepted_steps,
        "rejected_proposals": rejected_proposals,
        "acceptance_rate": acc_rate,
        "mean": mean,
        "cov": cov,
    }


# =============================================================================
# 2. Markov Chains & Diagnostics (Section 14.2.2)
# =============================================================================

def compute_autocorrelation(samples: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """
    Autocorrelation function of a 1D MCMC chain:
        rho(k) = Cov(z_t, z_{t+k}) / Var(z_t)
    """
    x = samples - np.mean(samples)
    n = len(x)
    var = np.var(samples)
    if var <= 1e-12:
        return np.ones(max_lag + 1)

    autocorr = np.zeros(max_lag + 1)
    autocorr[0] = 1.0
    for k in range(1, max_lag + 1):
        autocorr[k] = np.mean(x[:n - k] * x[k:]) / var

    return autocorr


def compute_integrated_autocorrelation_time(samples: np.ndarray, max_lag: int = 50) -> float:
    """
    Integrated autocorrelation time:
        tau_int = 1 + 2 * sum_{k=1}^K rho(k)
    """
    rho = compute_autocorrelation(samples, max_lag)
    # Sum positive autocorrelations
    pos_rho = []
    for r in rho[1:]:
        if r > 0.05:
            pos_rho.append(r)
        else:
            break
    return float(1.0 + 2.0 * sum(pos_rho))


def compute_effective_sample_size_mcmc(samples: np.ndarray, max_lag: int = 50) -> float:
    """
    Effective sample size for correlated MCMC chain:
        N_eff = N / tau_int
    """
    tau = compute_integrated_autocorrelation_time(samples, max_lag)
    return float(len(samples) / max(1.0, tau))


def verify_detailed_balance(
    target_p: Callable[[float], float],
    transition_p: Callable[[float, float], float],
    z_a: float,
    z_b: float,
) -> Dict[str, Any]:
    """
    Verify detailed balance condition:
        p(z_a) * T(z_a -> z_b) == p(z_b) * T(z_b -> z_a)
    """
    flux_ab = target_p(z_a) * transition_p(z_a, z_b)
    flux_ba = target_p(z_b) * transition_p(z_b, z_a)
    is_balanced = np.isclose(flux_ab, flux_ba, rtol=1e-4, atol=1e-6)

    return {
        "flux_ab": float(flux_ab),
        "flux_ba": float(flux_ba),
        "difference": float(abs(flux_ab - flux_ba)),
        "is_balanced": bool(is_balanced),
    }


# =============================================================================
# 3. The Metropolis–Hastings Algorithm (Section 14.2.3)
# =============================================================================

class MetropolisHastingsSampler:
    """
    General Metropolis-Hastings framework for arbitrary (possibly non-symmetric) proposal:
        A(z^*, z) = min(1, [p(z^*) * q(z | z^*)] / [p(z) * q(z^* | z)])
    """
    def __init__(
        self,
        target_log_p: Callable[[np.ndarray], float],
        proposal_sample: Callable[[np.ndarray, np.random.RandomState], np.ndarray],
        proposal_log_q: Callable[[np.ndarray, np.ndarray], float],
    ):
        self.target_log_p = target_log_p
        self.proposal_sample = proposal_sample
        self.proposal_log_q = proposal_log_q

    def sample(
        self,
        num_samples: int,
        init_state: np.ndarray,
        burn_in: int = 500,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        rng = np.random.RandomState(seed)
        current = np.array(init_state, dtype=float)
        current_log_p = float(self.target_log_p(current))

        dim = len(current)
        total_steps = burn_in + num_samples

        samples = np.zeros((num_samples, dim))
        accepted_count = 0

        for step in range(total_steps):
            prop = self.proposal_sample(current, rng)
            prop_log_p = float(self.target_log_p(prop))

            # Hastings ratio
            log_q_forward = float(self.proposal_log_q(prop, current))
            log_q_reverse = float(self.proposal_log_q(current, prop))

            log_alpha = (prop_log_p + log_q_reverse) - (current_log_p + log_q_forward)
            u = rng.uniform(0.0, 1.0)

            if np.log(u) <= log_alpha:
                current = prop
                current_log_p = prop_log_p
                accepted_count += 1

            if step >= burn_in:
                samples[step - burn_in] = current

        acc_rate = float(accepted_count / total_steps)
        return {
            "samples": samples,
            "acceptance_rate": acc_rate,
        }


# =============================================================================
# 4. Gibbs Sampling (Section 14.2.4)
# =============================================================================

def gibbs_sample_2d_gaussian(
    mean: np.ndarray,
    cov: np.ndarray,
    num_steps: int = 1000,
    init_state: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Gibbs sampling on 2D Gaussian distribution using exact analytical conditionals:
        p(z_1 | z_2) = N(mu_1 + (cov_12 / cov_22) * (z_2 - mu_2), cov_11 - cov_12^2 / cov_22)
        p(z_2 | z_1) = N(mu_2 + (cov_12 / cov_11) * (z_1 - mu_1), cov_22 - cov_12^2 / cov_11)
    """
    rng = np.random.RandomState(seed)
    mu1, mu2 = mean[0], mean[1]
    s11, s22 = cov[0, 0], cov[1, 1]
    s12 = cov[0, 1]

    # Conditional variances
    cond_var1 = s11 - (s12 ** 2) / s22
    cond_std1 = np.sqrt(max(1e-12, cond_var1))

    cond_var2 = s22 - (s12 ** 2) / s11
    cond_std2 = np.sqrt(max(1e-12, cond_var2))

    current = np.array(init_state if init_state is not None else [mu1, mu2], dtype=float)
    samples = np.zeros((num_steps, 2))
    steps_history: List[np.ndarray] = [current.copy()]

    for step in range(num_steps):
        # 1. Update z1 conditioned on current z2
        cond_mu1 = mu1 + (s12 / s22) * (current[1] - mu2)
        current[0] = cond_mu1 + rng.randn() * cond_std1
        steps_history.append(current.copy())

        # 2. Update z2 conditioned on new z1
        cond_mu2 = mu2 + (s12 / s11) * (current[0] - mu1)
        current[1] = cond_mu2 + rng.randn() * cond_std2
        steps_history.append(current.copy())

        samples[step] = current

    return {
        "samples": samples,
        "trajectory": np.array(steps_history),
        "cond_std1": cond_std1,
        "cond_std2": cond_std2,
    }


# =============================================================================
# 5. Ancestral Sampling (Section 14.2.5)
# =============================================================================

class AncestralSampler:
    """
    Ancestral sampling in directed acyclic graphical models (DAGs):
        p(z_1, ..., z_N) = prod_{i=1}^N p(z_i | pa_i)
    Samples nodes in topological sort order.
    """
    def __init__(self, topo_order: List[str], conditionals: Dict[str, Callable[[Dict[str, Any], np.random.RandomState], Any]]):
        self.topo_order = topo_order
        self.conditionals = conditionals

    def sample(self, num_samples: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        rng = np.random.RandomState(seed)
        samples = []
        for _ in range(num_samples):
            state: Dict[str, Any] = {}
            for node in self.topo_order:
                val = self.conditionals[node](state, rng)
                state[node] = val
            samples.append(state)
        return samples


# =============================================================================
# 6. Faithful Reproduction of Figures 14.9 - 14.12
# =============================================================================

def generate_figure_14_9(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Faithful reproduction of Figure 14.9:
    Metropolis algorithm on 2D correlated Gaussian.
    Shows elliptic contour, accepted trajectory steps in green,
    and rejected proposals as red spikes.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=300)

    # Correlated 2D Gaussian matching textbook Figure 14.9
    mean = np.array([1.5, 1.5])
    cov = np.array([[0.30, 0.177], [0.177, 0.213]])

    # Generate trajectory with seed ensuring rich mixture of accepts and rejects across the ellipse
    res = metropolis_sample_2d_gaussian(
        mean=mean,
        cov=cov,
        num_steps=60,
        proposal_std=0.20,
        init_state=np.array([1.5, 0.7]),
        seed=220,
    )

    # Plot 2.0-sigma contour ellipse of the Gaussian
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2.0 * 2.0 * np.sqrt(vals)
    ellipse = patches.Ellipse(
        xy=mean, width=width, height=height, angle=theta,
        edgecolor="#263238", facecolor="none", linewidth=2.0, zorder=2,
    )
    ax.add_patch(ellipse)

    # Plot rejected proposals as red segments from current state
    for cur, prop in res["rejected_proposals"]:
        ax.plot([cur[0], prop[0]], [cur[1], prop[1]], color="#e53935", linewidth=1.8, zorder=3)

    # Plot accepted trajectory steps in green
    for cur, prop in res["accepted_steps"]:
        ax.plot([cur[0], prop[0]], [cur[1], prop[1]], color="#00e676", linewidth=2.0, zorder=4)

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.set_aspect("equal")

    plt.tight_layout()
    _save_figure(fig, "fig_14_9_metropolis_algorithm", save_dir)
    return fig


def generate_figure_14_10(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Faithful reproduction of Figure 14.10:
    Comparison of proposal step size rho with covariance scale sigma_min and sigma_max.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=300)

    # Highly eccentric ellipse
    center = (0.5, 0.5)
    width = 2.4
    height = 0.38
    angle = 32.0

    ellipse = patches.Ellipse(
        xy=center, width=width, height=height, angle=angle,
        edgecolor="#e53935", facecolor="none", linewidth=2.2, zorder=2,
    )
    ax.add_patch(ellipse)

    # Proposal distribution circle of radius rho centered inside
    circle_center = (0.15, 0.22)
    rho = 0.25
    circle = patches.Circle(
        xy=circle_center, radius=rho,
        edgecolor="#1e88e5", facecolor="none", linewidth=2.0, zorder=3,
    )
    ax.add_patch(circle)

    # Center dot and arrow of radius rho
    ax.scatter([circle_center[0]], [circle_center[1]], color="#1e88e5", s=15, zorder=4)
    arrow_end = (circle_center[0] + rho * np.cos(np.pi / 4), circle_center[1] + rho * np.sin(np.pi / 4))
    ax.annotate(
        "", xy=arrow_end, xytext=circle_center,
        arrowprops=dict(arrowstyle="-|>", color="#1e88e5", lw=2.0, mutation_scale=15),
    )
    ax.text(circle_center[0] + 0.18, circle_center[1] + 0.22, "$\\rho$", fontsize=16, fontstyle="italic")

    # Double-headed arrow for sigma_max
    p1 = (-0.5, 0.8)
    p2 = (1.5, 2.05)
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle="<->", color="#263238", lw=1.8, mutation_scale=15),
    )
    ax.text(0.45, 1.55, "$\\sigma_{\\max}$", fontsize=15, fontstyle="italic", ha="center")

    # Double-headed arrow for sigma_min
    q1 = (-0.45, 0.1)
    q2 = (-0.6, 0.35)
    ax.annotate(
        "", xy=q2, xytext=q1,
        arrowprops=dict(arrowstyle="<->", color="#263238", lw=1.8, mutation_scale=15),
    )
    ax.text(-0.65, 0.12, "$\\sigma_{\\min}$", fontsize=15, fontstyle="italic", ha="right")

    ax.set_xlim(-0.8, 1.7)
    ax.set_ylim(-0.1, 2.2)
    ax.axis("off")
    ax.set_aspect("equal")

    plt.tight_layout()
    _save_figure(fig, "fig_14_10_step_size_scaling", save_dir)
    return fig


def generate_figure_14_11(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Faithful reproduction of Figure 14.11:
    Diffusive slow-down in correlated distributions: step length l vs total length L.
    Shows conditional distribution p(z_1 | z_2) and orthogonal Gibbs-style steps.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=300)

    # Coordinate axes z1, z2 with arrows
    ax.annotate(
        "", xy=(3.8, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="#263238", lw=2.0, mutation_scale=15),
    )
    ax.annotate(
        "", xy=(0, 3.8), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="#263238", lw=2.0, mutation_scale=15),
    )
    ax.text(3.7, -0.25, "$z_1$", fontsize=15, fontstyle="italic")
    ax.text(-0.25, 3.7, "$z_2$", fontsize=15, fontstyle="italic")

    # Correlated Gaussian ellipse
    center = (2.0, 2.0)
    ellipse = patches.Ellipse(
        xy=center, width=3.4, height=0.55, angle=45.0,
        edgecolor="#e53935", facecolor="none", linewidth=2.2, zorder=2,
    )
    ax.add_patch(ellipse)

    # Current z2 level
    z2_val = 2.6
    # Mean of z1 given z2=2.6 is approximately 2.6
    z1_cond_mean = 2.6

    # Dashed green line across at z2 = 2.6
    ax.plot([0, 3.4], [z2_val, z2_val], color="#00e676", linestyle="--", linewidth=1.5, zorder=1)
    # Dashed green line down from (z1_cond_mean, z2_val) to x-axis
    ax.plot([z1_cond_mean, z1_cond_mean], [0, z2_val], color="#00e676", linestyle="--", linewidth=1.5, zorder=1)

    # 1D Conditional distribution curve at bottom
    z1_grid = np.linspace(1.8, 3.4, 500)
    cond_pdf = stats.norm.pdf(z1_grid, loc=z1_cond_mean, scale=0.22) * 0.35
    ax.plot(z1_grid, cond_pdf, color="#00e676", linewidth=2.2, zorder=3)

    # Orthogonal blue steps (Gibbs / coordinate steps)
    steps = [
        (1.3, 1.3),
        (1.3, 1.9),
        (1.9, 1.9),
        (1.9, 1.6),
        (1.35, 1.6),
        (2.3, 1.6),
        (2.3, 2.3),
    ]
    xs = [p[0] for p in steps]
    ys = [p[1] for p in steps]
    ax.plot(xs, ys, color="#1e88e5", linewidth=2.2, zorder=4)
    # Arrowhead at end of path
    ax.annotate(
        "", xy=(2.3, 2.35), xytext=(2.3, 2.1),
        arrowprops=dict(arrowstyle="-|>", color="#1e88e5", lw=2.2, mutation_scale=15),
    )

    # Double arrow L at top
    ax.annotate(
        "", xy=(3.2, 3.4), xytext=(0.6, 3.4),
        arrowprops=dict(arrowstyle="<->", color="#263238", lw=1.8, mutation_scale=15),
    )
    ax.text(1.9, 3.52, "$L$", fontsize=15, fontstyle="italic", ha="center")

    # Double arrow l above conditional distribution
    ax.annotate(
        "", xy=(3.0, 1.0), xytext=(2.2, 1.0),
        arrowprops=dict(arrowstyle="<->", color="#263238", lw=1.8, mutation_scale=15),
    )
    ax.text(2.6, 1.15, "$l$", fontsize=15, fontstyle="italic", ha="center")

    ax.set_xlim(-0.3, 4.0)
    ax.set_ylim(-0.3, 4.0)
    ax.axis("off")
    ax.set_aspect("equal")

    plt.tight_layout()
    _save_figure(fig, "fig_14_11_diffusive_scaling", save_dir)
    return fig


def generate_figure_14_12(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Faithful reproduction of Figure 14.12:
    Markov blanket of node z in a directed graphical model.
    Includes parents, children, and co-parents.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.8), dpi=300)

    # Node positions
    center_pos = (0.0, 0.0)
    # Parents (top left, top right)
    parent_l = (-0.9, 1.2)
    parent_r = (0.9, 1.2)
    # Children (bottom left, bottom right)
    child_l = (-0.9, -1.0)
    child_r = (0.9, -1.0)
    # Co-parents (far left, far right)
    coparent_l = (-1.8, 0.1)
    coparent_r = (1.8, 0.1)

    r = 0.32

    # Draw arrows
    def draw_directed_edge(p_start, p_end):
        vec = np.array(p_end) - np.array(p_start)
        dist = np.linalg.norm(vec)
        unit = vec / dist
        start = np.array(p_start) + unit * r
        end = np.array(p_end) - unit * r
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(arrowstyle="-|>", color="#e53935", lw=2.2, mutation_scale=16),
            zorder=2,
        )

    # Parents -> Center
    draw_directed_edge(parent_l, center_pos)
    draw_directed_edge(parent_r, center_pos)
    # Center -> Children
    draw_directed_edge(center_pos, child_l)
    draw_directed_edge(center_pos, child_r)
    # Co-parents -> Children
    draw_directed_edge(coparent_l, child_l)
    draw_directed_edge(coparent_r, child_r)

    # Draw Markov blanket nodes (blue fill, red border)
    mb_nodes = [parent_l, parent_r, child_l, child_r, coparent_l, coparent_r]
    for pos in mb_nodes:
        c = patches.Circle(pos, r, facecolor="#c5cae9", edgecolor="#e53935", linewidth=2.5, zorder=3)
        ax.add_patch(c)

    # Center target node z (white fill, red border)
    c_center = patches.Circle(center_pos, r, facecolor="#ffffff", edgecolor="#e53935", linewidth=2.5, zorder=4)
    ax.add_patch(c_center)
    ax.text(center_pos[0], center_pos[1], "$\\mathbf{z}$", fontsize=15, fontweight="bold", ha="center", va="center", zorder=5)

    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-1.6, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    _save_figure(fig, "fig_14_12_markov_blanket", save_dir)
    return fig


def generate_all_section_14_2_figures(save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
    """Generate and save all 4 figures (Figures 14.9 - 14.12) for Section 14.2."""
    return {
        "fig_14_9": generate_figure_14_9(save_dir),
        "fig_14_10": generate_figure_14_10(save_dir),
        "fig_14_11": generate_figure_14_11(save_dir),
        "fig_14_12": generate_figure_14_12(save_dir),
    }
