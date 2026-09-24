"""
Chapter 14: Sampling
Section 14.3: Langevin Sampling

Provides core mathematical algorithms and figure reproductions for:
- 14.3.1 Energy-based models (EBMs, score functions, partition function intractability)
- 14.3.2 Maximizing the likelihood (Positive & negative phases of EBM training)
- 14.3.3 Langevin dynamics (ULA, MALA, score-based sampling, multimodal limitations, annealed Langevin)
- Figures 14.13 and 14.14 faithful reproductions
"""

import os
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath

from common.plot_utils import setup_style, save_plot


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
# 1. Energy-Based Models (Section 14.3.1 & 14.3.2)
# =============================================================================

class EnergyBasedModel1D:
    """
    1D Energy-Based Model (EBM):
        p(x; w) = (1 / Z(w)) * exp(-E(x; w))
        Z(w) = int exp(-E(x; w)) dx
        s(x) = nabla_x ln p(x; w) = -nabla_x E(x; w)  (Score function)
    """
    def __init__(self, energy_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 grad_x_energy_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 grad_w_energy_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.energy_fn = energy_fn
        self.grad_x_energy_fn = grad_x_energy_fn
        self.grad_w_energy_fn = grad_w_energy_fn

    def compute_energy(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Compute energy E(x; w)."""
        return self.energy_fn(x, w)

    def score(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Score function s(x) = nabla_x ln p(x; w) = -nabla_x E(x; w).
        Independent of the partition function Z(w).
        """
        return -self.grad_x_energy_fn(x, w)

    def compute_partition_function(self, w: np.ndarray, x_min: float = -10.0, x_max: float = 10.0) -> float:
        """Compute normalization constant Z(w) via numerical quadrature."""
        integrand = lambda x_val: np.exp(-float(self.energy_fn(np.array([x_val]), w)[0]))
        z_val, _ = integrate.quad(integrand, x_min, x_max)
        return float(z_val)

    def compute_log_likelihood_gradient(
        self,
        data_samples: np.ndarray,
        model_samples: np.ndarray,
        w: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute log-likelihood gradient (Equation 14.34):
            nabla_w ln p(D; w) = - (1/N_D) sum_{x in D} nabla_w E(x; w)
                                 + (1/N_M) sum_{x_m in M} nabla_w E(x_m; w)
            = (Positive Phase) + (Negative Phase)
        """
        # Positive phase: pulls down energy on observed data
        pos_grad = -np.mean(self.grad_w_energy_fn(data_samples, w), axis=0)

        # Negative phase: pushes up energy on ungrounded model samples
        neg_grad = np.mean(self.grad_w_energy_fn(model_samples, w), axis=0)

        total_grad = pos_grad + neg_grad

        return {
            "total_grad": total_grad,
            "positive_phase": pos_grad,
            "negative_phase": neg_grad,
        }


# =============================================================================
# 2. Langevin Dynamics (Section 14.3.3)
# =============================================================================

def unadjusted_langevin_algorithm(
    score_fn: Callable[[np.ndarray], np.ndarray],
    init_state: np.ndarray,
    step_size: float,
    num_steps: int,
    burn_in: int = 0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Unadjusted Langevin Algorithm (ULA):
        z^{(tau+1)} = z^{(tau)} + (epsilon / 2) * score(z^{(tau)}) + sqrt(epsilon) * eta^{(tau)}
        where eta^{(tau)} ~ N(0, I).
    """
    rng = np.random.RandomState(seed)
    current = np.array(init_state, dtype=float)
    dim = len(current)

    trajectory = np.zeros((num_steps + 1, dim))
    trajectory[0] = current

    for step in range(num_steps):
        score = score_fn(current)
        noise = rng.randn(dim)
        current = current + 0.5 * step_size * score + np.sqrt(step_size) * noise
        trajectory[step + 1] = current

    samples = trajectory[burn_in:]
    return {
        "samples": samples,
        "trajectory": trajectory,
        "step_size": step_size,
    }


def metropolis_adjusted_langevin_algorithm(
    target_log_p: Callable[[np.ndarray], float],
    score_fn: Callable[[np.ndarray], np.ndarray],
    init_state: np.ndarray,
    step_size: float,
    num_steps: int,
    burn_in: int = 0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Metropolis-Adjusted Langevin Algorithm (MALA):
        Proposal:
            q(z^* | z) = N(z^* | z + (epsilon / 2) * nabla ln p(z), epsilon * I)
        Acceptance probability:
            A(z^*, z) = min(1, [p(z^*) * q(z | z^*)] / [p(z) * q(z^* | z)])
    """
    rng = np.random.RandomState(seed)
    current = np.array(init_state, dtype=float)
    dim = len(current)
    current_log_p = float(target_log_p(current))

    trajectory = np.zeros((num_steps + 1, dim))
    trajectory[0] = current

    accepted_steps: List[Tuple[np.ndarray, np.ndarray]] = []
    rejected_proposals: List[Tuple[np.ndarray, np.ndarray]] = []
    accepted_count = 0

    for step in range(num_steps):
        cur_score = score_fn(current)
        mu_forward = current + 0.5 * step_size * cur_score

        # Propose candidate
        noise = rng.randn(dim)
        proposal = mu_forward + np.sqrt(step_size) * noise
        prop_log_p = float(target_log_p(proposal))

        # Backward proposal drift
        prop_score = score_fn(proposal)
        mu_reverse = proposal + 0.5 * step_size * prop_score

        # Transition log-probabilities q(z^* | z) and q(z | z^*)
        log_q_forward = -0.5 * np.sum((proposal - mu_forward) ** 2) / step_size
        log_q_reverse = -0.5 * np.sum((current - mu_reverse) ** 2) / step_size

        log_alpha = (prop_log_p + log_q_reverse) - (current_log_p + log_q_forward)
        u = rng.uniform(0.0, 1.0)

        if np.log(u) <= log_alpha:
            accepted_steps.append((current.copy(), proposal.copy()))
            current = proposal
            current_log_p = prop_log_p
            accepted_count += 1
        else:
            rejected_proposals.append((current.copy(), proposal.copy()))

        trajectory[step + 1] = current

    samples = trajectory[burn_in:]
    acc_rate = float(accepted_count / num_steps) if num_steps > 0 else 0.0

    return {
        "samples": samples,
        "trajectory": trajectory,
        "acceptance_rate": acc_rate,
        "accepted_steps": accepted_steps,
        "rejected_proposals": rejected_proposals,
    }


def annealed_langevin_dynamics(
    score_fns: List[Callable[[np.ndarray], np.ndarray]],
    sigmas: List[float],
    init_state: np.ndarray,
    num_steps_per_level: int,
    step_size_factor: float = 0.00002,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Annealed Langevin Dynamics across geometric noise levels sigma_1 > sigma_2 > ... > sigma_L:
    Allows transitioning across isolated modes where single-scale Langevin is trapped.
    """
    rng = np.random.RandomState(seed)
    current = np.array(init_state, dtype=float)
    dim = len(current)
    sigma_min = sigmas[-1]

    trajectory: List[np.ndarray] = [current.copy()]

    for level, (score_fn, sigma) in enumerate(zip(score_fns, sigmas)):
        # Step size proportional to (sigma / sigma_min)^2
        alpha_i = step_size_factor * (sigma / sigma_min) ** 2
        for _ in range(num_steps_per_level):
            score = score_fn(current)
            noise = rng.randn(dim)
            current = current + 0.5 * alpha_i * score + np.sqrt(alpha_i) * noise
            trajectory.append(current.copy())

    return {
        "final_sample": current,
        "trajectory": np.array(trajectory),
        "sigmas": sigmas,
    }


# =============================================================================
# 3. Faithful Reproduction of Figures 14.13 & 14.14
# =============================================================================

def generate_figure_14_13(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Faithful reproduction of Figure 14.13:
    Energy-based model training via maximum likelihood.
    Shows:
    - Data distribution p_D(x) (bimodal red dashed curve)
    - Model distribution p_M(x) (unimodal blue dashed curve)
    - Energy function E(x, w) (green solid curve)
    - Data points x ~ p_D (red dots) with vertical dashed lines and downward green arrows (positive phase)
    - Model samples x ~ p_M (blue dots) with vertical dashed lines and upward green arrows (negative phase)
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)

    x = np.linspace(0.0, 10.0, 600)

    # 1. Data distribution p_D(x): bimodal mixture of Gaussians
    p_D = 0.35 * np.exp(-0.5 * ((x - 2.0) / 0.85) ** 2) + 0.48 * np.exp(-0.5 * ((x - 4.8) / 0.95) ** 2)

    # 2. Model distribution p_M(x): unimodal Gaussian matching the valley of E(x)
    p_M = 0.70 * np.exp(-0.5 * ((x - 8.5) / 1.0) ** 2)

    # 3. Energy curve E(x, w): smooth quartic curve high on left, shoulder at x=4.8, minimum at x=8.5
    pts_x = [1.5, 2.5, 3.2, 4.8, 7.0, 8.5, 10.0]
    pts_y = [2.6, 2.2, 1.8, 1.4, 0.95, 0.65, 1.2]
    poly = np.poly1d(np.polyfit(pts_x, pts_y, 4))
    E = poly(x)

    ax.plot(x, p_D, color="#e53935", linestyle="--", linewidth=1.8, label="data distribution")
    ax.plot(x, p_M, color="#1e88e5", linestyle="--", linewidth=1.8, label="model distribution")
    ax.plot(x, E, color="#2e7d32", linestyle="-", linewidth=2.2, label="energy function")

    # 4. Red data points (Positive phase: pulls down energy)
    x_data = [2.5, 3.2, 4.8]
    for xd in x_data:
        yd = float(poly(xd))
        ax.plot(xd, 0.01, marker="o", color="#e53935", markersize=6.5, zorder=5)
        ax.plot([xd, xd], [0.01, yd], color="#757575", linestyle="--", linewidth=1.2, zorder=3)
        # Downward green arrow
        ax.annotate(
            "",
            xy=(xd, yd - 0.28),
            xytext=(xd, yd),
            arrowprops=dict(facecolor="#2e7d32", edgecolor="#2e7d32", width=1.5, headwidth=6, headlength=7, shrink=0.0),
            zorder=4,
        )

    # 5. Blue model samples (Negative phase: pushes up energy)
    x_model = [7.0, 8.3, 8.6]
    for xm in x_model:
        ym = float(poly(xm))
        ax.plot(xm, 0.01, marker="o", color="#1e88e5", markersize=6.5, zorder=5)
        ax.plot([xm, xm], [0.01, ym], color="#757575", linestyle="--", linewidth=1.2, zorder=3)
        # Upward green arrow
        ax.annotate(
            "",
            xy=(xm, ym + 0.28),
            xytext=(xm, ym),
            arrowprops=dict(facecolor="#2e7d32", edgecolor="#2e7d32", width=1.5, headwidth=6, headlength=7, shrink=0.0),
            zorder=4,
        )

    # Labels matching textbook exactly
    ax.text(0.5, 0.50, r"$p_{\mathcal{D}}(x)$", fontsize=13, color="#263238")
    ax.text(8.0, 0.95, r"$p_{\mathcal{M}}(x)$", fontsize=13, color="#263238")
    ax.text(5.0, 1.85, r"$E(x, \mathbf{w})$", fontsize=13, color="#263238")
    ax.set_xlabel(r"$x$", fontsize=13)

    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(-0.02, 2.6)
    ax.set_xticks([])
    ax.set_yticks([])

    # Ensure clean outer frame
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    plt.tight_layout()
    _save_figure(fig, "fig_14_13_energy_based_training", save_dir)
    return fig


def generate_figure_14_14(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Faithful reproduction of Figure 14.14:
    Multimodal distribution with two isolated high-density modes.
    Shows the fundamental challenge for Langevin dynamics and score-based sampling,
    where zero score in the interstitial space prevents traversal between modes.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=300)

    # Smooth organic salmon-pink mode 1 (lower left)
    verts1 = [
        (0.24, 0.12),
        (0.35, 0.13), (0.43, 0.18), (0.43, 0.26),
        (0.43, 0.35), (0.35, 0.40), (0.28, 0.39),
        (0.20, 0.38), (0.18, 0.28), (0.18, 0.22),
        (0.18, 0.15), (0.20, 0.12), (0.24, 0.12),
    ]
    codes1 = [MplPath.MOVETO] + [MplPath.CURVE4] * 12
    path1 = MplPath(verts1, codes1)
    patch1 = patches.PathPatch(path1, facecolor="#ff7070", edgecolor="none", zorder=2)
    ax.add_patch(patch1)

    # Smooth organic salmon-pink mode 2 (upper right)
    verts2 = [
        (0.78, 0.63),
        (0.85, 0.65), (0.87, 0.74), (0.86, 0.80),
        (0.85, 0.87), (0.78, 0.90), (0.72, 0.89),
        (0.64, 0.88), (0.61, 0.81), (0.61, 0.77),
        (0.61, 0.70), (0.70, 0.62), (0.78, 0.63),
    ]
    codes2 = [MplPath.MOVETO] + [MplPath.CURVE4] * 12
    path2 = MplPath(verts2, codes2)
    patch2 = patches.PathPatch(path2, facecolor="#ff7070", edgecolor="none", zorder=2)
    ax.add_patch(patch2)

    # Axes with arrowheads
    ax.annotate(
        "",
        xy=(1.03, 0.05),
        xytext=(0.05, 0.05),
        arrowprops=dict(facecolor="black", edgecolor="black", width=1.4, headwidth=7, headlength=9),
        zorder=3,
    )
    ax.annotate(
        "",
        xy=(0.05, 1.03),
        xytext=(0.05, 0.05),
        arrowprops=dict(facecolor="black", edgecolor="black", width=1.4, headwidth=7, headlength=9),
        zorder=3,
    )

    # Axis labels
    ax.text(0.97, -0.01, r"$z_1$", fontsize=15, color="black")
    ax.text(-0.02, 0.96, r"$z_2$", fontsize=15, color="black")

    ax.set_xlim(0.0, 1.08)
    ax.set_ylim(0.0, 1.08)
    ax.axis("off")

    plt.tight_layout()
    _save_figure(fig, "fig_14_14_multimodal_challenge", save_dir)
    return fig
