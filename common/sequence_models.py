"""
common/sequence_models.py
=========================
Section 11.3: Sequence Models
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements:
1. Autoregressive and Markovian Sequence Models (Section 11.3):
   - General Autoregressive Model: p(x_1..x_N) = prod_{n=1}^N p(x_n | x_1..x_{n-1}) (Figure 11.27).
   - Independent sequence model: p(x_1..x_N) = prod_{n=1}^N p(x_n) (Figure 11.28).
   - First-order Markov chain: p(x_n | x_1..x_{n-1}) = p(x_n | x_{n-1}) (Figure 11.29).
   - Second-order Markov chain: p(x_n | x_1..x_{n-1}) = p(x_n | x_{n-1}, x_{n-2}) (Figure 11.30).
2. State-Space Models & Hidden Markov Models (Section 11.3.1, Figure 11.31):
   - Latent Markov chain z_1 -> z_2 -> ... -> z_N with emissions z_n -> x_n.
   - Long-range dependence preservation despite sparse local connectivity.
   - Forward-backward algorithm and Viterbi algorithm for hidden state decoding.
3. High-Resolution Figure Reproductions (Figures 11.27 - 11.31):
   - Figures 11.27 to 11.31 saved to 11/result/ and result/.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .plot_utils import setup_style, save_plot
from .graphical_models import DirectedGraph, _draw_node, _draw_arrow


def _save_figure(fig: plt.Figure, filename_base: str, save_dir: Optional[str] = None) -> None:
    """Save figure to Chapter 11 result directory and root result directory."""
    if save_dir is not None:
        save_plot(fig, os.path.join(save_dir, f"{filename_base}.png"))
        return

    repo_root = Path(__file__).resolve().parent.parent
    dir_ch11 = repo_root / "11" / "result"
    dir_root = repo_root / "result"
    dir_ch11.mkdir(parents=True, exist_ok=True)
    dir_root.mkdir(parents=True, exist_ok=True)

    save_plot(fig, str(dir_ch11 / f"{filename_base}.png"))
    save_plot(fig, str(dir_root / f"{filename_base}.png"))


# =============================================================================
# 1. Markov Chain & Hidden Markov Model Simulation & Algorithms
# =============================================================================

class MarkovChainSimulator:
    """Discrete Markov chain simulation and stationary distribution."""

    def __init__(self, transition_matrix: np.ndarray, initial_dist: np.ndarray) -> None:
        self.T = np.asarray(transition_matrix, dtype=np.float64)  # T[i, j] = p(x_{t+1}=j | x_t=i)
        self.pi0 = np.asarray(initial_dist, dtype=np.float64)
        self.K = len(self.pi0)
        assert self.T.shape == (self.K, self.K)

    def sample_sequence(self, length: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        seq = np.zeros(length, dtype=np.int64)
        seq[0] = rng.choice(self.K, p=self.pi0)
        for t in range(1, length):
            seq[t] = rng.choice(self.K, p=self.T[seq[t - 1]])
        return seq

    def compute_stationary_distribution(self) -> np.ndarray:
        """Find pi such that pi @ T = pi and sum(pi) = 1."""
        eigvals, eigvecs = np.linalg.eig(self.T.T)
        # Find eigenvalue closest to 1.0
        idx = np.argmin(np.abs(eigvals - 1.0))
        pi = np.real(eigvecs[:, idx])
        pi = pi / np.sum(pi)
        return pi


class HiddenMarkovModel:
    """
    Hidden Markov Model (HMM) / Discrete State-Space Model.
    - Latent states z_t in {0, ..., K-1}
    - Observations x_t in {0, ..., V-1} (discrete emissions)
    """

    def __init__(self, transition_matrix: np.ndarray, emission_matrix: np.ndarray,
                 initial_dist: np.ndarray) -> None:
        self.A = np.asarray(transition_matrix, dtype=np.float64)  # A[i, j] = p(z_{t+1}=j | z_t=i)
        self.B = np.asarray(emission_matrix, dtype=np.float64)    # B[j, k] = p(x_t=k | z_t=j)
        self.pi = np.asarray(initial_dist, dtype=np.float64)
        self.K = len(self.pi)

    def forward_algorithm(self, observations: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute marginal likelihood p(x_1..x_N) via forward recursion alpha_t(j).
        alpha_1(j) = pi_j * B[j, x_1]
        alpha_t(j) = [sum_i alpha_{t-1}(i) * A[i, j]] * B[j, x_t]
        """
        T_len = len(observations)
        alpha = np.zeros((T_len, self.K), dtype=np.float64)

        # Initial step
        alpha[0] = self.pi * self.B[:, observations[0]]

        # Recursion
        for t in range(1, T_len):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, observations[t]]

        marginal_likelihood = float(np.sum(alpha[-1]))
        return marginal_likelihood, alpha

    def viterbi_algorithm(self, observations: np.ndarray) -> np.ndarray:
        """
        Find most probable hidden state path z* = argmax_z p(z, x).
        """
        T_len = len(observations)
        viterbi = np.zeros((T_len, self.K), dtype=np.float64)
        backpointer = np.zeros((T_len, self.K), dtype=np.int64)

        viterbi[0] = self.pi * self.B[:, observations[0]]

        for t in range(1, T_len):
            for j in range(self.K):
                prob_trans = viterbi[t - 1] * self.A[:, j]
                best_prev = np.argmax(prob_trans)
                viterbi[t, j] = prob_trans[best_prev] * self.B[j, observations[t]]
                backpointer[t, j] = best_prev

        # Backtracking
        best_path = np.zeros(T_len, dtype=np.int64)
        best_path[-1] = np.argmax(viterbi[-1])
        for t in range(T_len - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        return best_path


# =============================================================================
# 2. High-Resolution Figure Reproductions (Figures 11.27 - 11.31)
# =============================================================================

def generate_figure_11_27(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.27: General Autoregressive Model with 4 nodes.
    x1 connects to x2, x3, x4. x2 connects to x3, x4. x3 connects to x4.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 3.2)
    ax.axis('off')
    ax.set_title("Figure 11.27: General Autoregressive Model (Fully Connected DAG)",
                 fontsize=11.5, fontweight='bold', pad=12)

    positions = [
        (1.0, 1.0),
        (2.8, 1.0),
        (4.6, 1.0),
        (6.4, 1.0),
    ]

    for i, pos in enumerate(positions, 1):
        _draw_node(ax, pos, f"$x_{i}$")

    # Directed edges: all i < j
    # Direct adjacent edges
    for i in range(3):
        _draw_arrow(ax, positions[i], positions[i + 1])

    # Arc edges for i to i+2, i+3
    # x1 -> x3
    ax.annotate("", xy=(positions[2][0] - 0.2, positions[2][1] + 0.3),
                xytext=(positions[0][0] + 0.2, positions[0][1] + 0.3),
                arrowprops=dict(arrowstyle="-|>", color='#2c3e50', lw=1.6, connectionstyle="arc3,rad=-0.4", mutation_scale=14))

    # x2 -> x4
    ax.annotate("", xy=(positions[3][0] - 0.2, positions[3][1] + 0.3),
                xytext=(positions[1][0] + 0.2, positions[1][1] + 0.3),
                arrowprops=dict(arrowstyle="-|>", color='#2c3e50', lw=1.6, connectionstyle="arc3,rad=-0.4", mutation_scale=14))

    # x1 -> x4 (higher arc)
    ax.annotate("", xy=(positions[3][0] - 0.25, positions[3][1] + 0.35),
                xytext=(positions[0][0] + 0.25, positions[0][1] + 0.35),
                arrowprops=dict(arrowstyle="-|>", color='#2980b9', lw=1.6, connectionstyle="arc3,rad=-0.55", mutation_scale=14))

    # Time arrow at bottom
    ax.annotate("", xy=(6.8, 0.2), xytext=(0.6, 0.2),
                arrowprops=dict(arrowstyle="->", color='#7f8c8d', lw=1.5))
    ax.text(3.7, 0.0, "time", ha='center', fontsize=9.0, color='#7f8c8d', style='italic')

    plt.tight_layout()
    _save_figure(fig, "Figure_11_27", save_dir)
    return fig


def generate_figure_11_28(save_dir: Optional[str] = None) -> plt.Figure:
    """Figure 11.28: Independent Sequence Model (no links)."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 2.2)
    ax.axis('off')
    ax.set_title("Figure 11.28: Independent Sequence Model ($p(\\mathbf{x}) = \\prod_n p(x_n)$)",
                 fontsize=11.5, fontweight='bold', pad=12)

    positions = [
        (1.0, 1.1),
        (2.8, 1.1),
        (4.6, 1.1),
        (6.4, 1.1),
    ]

    for i, pos in enumerate(positions, 1):
        _draw_node(ax, pos, f"$x_{i}$")

    # Time arrow at bottom
    ax.annotate("", xy=(6.8, 0.3), xytext=(0.6, 0.3),
                arrowprops=dict(arrowstyle="->", color='#7f8c8d', lw=1.5))
    ax.text(3.7, 0.1, "time", ha='center', fontsize=9.0, color='#7f8c8d', style='italic')

    plt.tight_layout()
    _save_figure(fig, "Figure_11_28", save_dir)
    return fig


def generate_figure_11_29(save_dir: Optional[str] = None) -> plt.Figure:
    """Figure 11.29: First-order Markov chain of observations (x1 -> x2 -> x3 -> x4 -> ... -> xN)."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.set_xlim(-0.5, 8.0)
    ax.set_ylim(-0.5, 2.2)
    ax.axis('off')
    ax.set_title("Figure 11.29: First-Order Markov Chain ($x_n \\perp x_1, \\dots, x_{n-2} \\mid x_{n-1}$)",
                 fontsize=11.5, fontweight='bold', pad=12)

    nodes = [
        ((1.0, 1.1), "$x_1$"),
        ((2.6, 1.1), "$x_2$"),
        ((4.2, 1.1), "$x_3$"),
        ((6.8, 1.1), "$x_N$"),
    ]

    for pos, lbl in nodes:
        _draw_node(ax, pos, lbl)

    _draw_arrow(ax, nodes[0][0], nodes[1][0])
    _draw_arrow(ax, nodes[1][0], nodes[2][0])

    ax.text(5.3, 1.1, "$\\dots$", ha='center', va='center', fontsize=18, fontweight='bold', color='#2c3e50')
    _draw_arrow(ax, (5.7, 1.1), nodes[3][0], radius=0.0)

    # Time arrow
    ax.annotate("", xy=(7.2, 0.3), xytext=(0.6, 0.3),
                arrowprops=dict(arrowstyle="->", color='#7f8c8d', lw=1.5))
    ax.text(3.9, 0.1, "time", ha='center', fontsize=9.0, color='#7f8c8d', style='italic')

    plt.tight_layout()
    _save_figure(fig, "Figure_11_29", save_dir)
    return fig


def generate_figure_11_30(save_dir: Optional[str] = None) -> plt.Figure:
    """Figure 11.30: Second-order Markov chain (x_n depends on x_{n-1} and x_{n-2})."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.set_xlim(-0.5, 8.0)
    ax.set_ylim(-0.5, 2.6)
    ax.axis('off')
    ax.set_title("Figure 11.30: Second-Order Markov Chain ($x_n \\perp x_{1..n-3} \\mid x_{n-1}, x_{n-2}$)",
                 fontsize=11.5, fontweight='bold', pad=12)

    nodes = [
        ((1.0, 0.9), "$x_1$"),
        ((2.6, 0.9), "$x_2$"),
        ((4.2, 0.9), "$x_3$"),
        ((5.8, 0.9), "$x_4$"),
    ]

    for pos, lbl in nodes:
        _draw_node(ax, pos, lbl)

    # First-order links
    _draw_arrow(ax, nodes[0][0], nodes[1][0])
    _draw_arrow(ax, nodes[1][0], nodes[2][0])
    _draw_arrow(ax, nodes[2][0], nodes[3][0])

    # Second-order skip links
    # x1 -> x3
    ax.annotate("", xy=(nodes[2][0][0] - 0.2, nodes[2][0][1] + 0.3),
                xytext=(nodes[0][0][0] + 0.2, nodes[0][0][1] + 0.3),
                arrowprops=dict(arrowstyle="-|>", color='#c0392b', lw=1.6, connectionstyle="arc3,rad=-0.4", mutation_scale=14))

    # x2 -> x4
    ax.annotate("", xy=(nodes[3][0][0] - 0.2, nodes[3][0][1] + 0.3),
                xytext=(nodes[1][0][0] + 0.2, nodes[1][0][1] + 0.3),
                arrowprops=dict(arrowstyle="-|>", color='#c0392b', lw=1.6, connectionstyle="arc3,rad=-0.4", mutation_scale=14))

    # Time arrow
    ax.annotate("", xy=(6.5, 0.2), xytext=(0.6, 0.2),
                arrowprops=dict(arrowstyle="->", color='#7f8c8d', lw=1.5))
    ax.text(3.5, 0.0, "time", ha='center', fontsize=9.0, color='#7f8c8d', style='italic')

    plt.tight_layout()
    _save_figure(fig, "Figure_11_30", save_dir)
    return fig


def generate_figure_11_31(save_dir: Optional[str] = None) -> plt.Figure:
    """Figure 11.31: State-space model (hidden Markov chain z_n with emissions x_n)."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.set_xlim(-0.5, 8.0)
    ax.set_ylim(-0.5, 3.5)
    ax.axis('off')
    ax.set_title("Figure 11.31: State-Space Model / Hidden Markov Model (HMM)",
                 fontsize=11.5, fontweight='bold', pad=12)

    latent_nodes = [
        ((1.0, 2.3), "$z_1$"),
        ((2.8, 2.3), "$z_2$"),
        ((4.6, 2.3), "$z_3$"),
        ((6.8, 2.3), "$z_N$"),
    ]

    obs_nodes = [
        ((1.0, 0.9), "$x_1$"),
        ((2.8, 0.9), "$x_2$"),
        ((4.6, 0.9), "$x_3$"),
        ((6.8, 0.9), "$x_N$"),
    ]

    # Draw latent nodes (unshaded, top row)
    for pos, lbl in latent_nodes:
        _draw_node(ax, pos, lbl, facecolor='#fadbd8')  # light pink for latent

    # Draw observation nodes (shaded, bottom row)
    for pos, lbl in obs_nodes:
        _draw_node(ax, pos, lbl, is_observed=True)

    # Latent transition arrows: z_t -> z_{t+1}
    _draw_arrow(ax, latent_nodes[0][0], latent_nodes[1][0])
    _draw_arrow(ax, latent_nodes[1][0], latent_nodes[2][0])

    ax.text(5.5, 2.3, "$\\dots$", ha='center', va='center', fontsize=18, fontweight='bold', color='#2c3e50')
    _draw_arrow(ax, (5.9, 2.3), latent_nodes[3][0], radius=0.0)

    # Emission arrows: z_t -> x_t
    for i in range(4):
        _draw_arrow(ax, latent_nodes[i][0], obs_nodes[i][0])

    ax.text(5.5, 0.9, "$\\dots$", ha='center', va='center', fontsize=18, fontweight='bold', color='#2c3e50')

    # Side labels
    ax.text(-0.2, 2.3, "Latent $\\mathbf{z}$", va='center', ha='right', fontsize=9.5, fontweight='bold', color='#922b21')
    ax.text(-0.2, 0.9, "Observed $\\mathbf{x}$", va='center', ha='right', fontsize=9.5, fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    _save_figure(fig, "Figure_11_31", save_dir)
    return fig


if __name__ == "__main__":
    for i in range(27, 32):
        func_name = f"generate_figure_11_{i}"
        func = globals()[func_name]
        fig = func()
        plt.close(fig)
    print("All 5 figures for Section 11.3 generated successfully!")
