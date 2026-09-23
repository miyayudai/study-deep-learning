"""
common/graphical_models.py
==========================
Section 11.1: Graphical Models
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements:
1. Directed Acyclic Graph (DAG) Representation & Factorization (Eq 11.6):
   - DirectedGraph class: nodes, edges, parent sets, topological sorting.
   - General DAG joint factorization: p(x) = prod_{k=1}^K p(x_k | pa_k).
2. Discrete Variables & Parameter Counting (Section 11.1.3):
   - Fully connected vs independent vs chain vs shared transition matrices.
   - Parameter growth: exponential 2^M vs linear M+1 via logistic sigmoid (Figure 11.6).
3. Linear-Gaussian Graphical Models (Section 11.1.4, Eqs 11.7 - 11.14):
   - Recursive exact mean and covariance computation:
     E[x_i] = sum_{j in pa_i} w_{ij} E[x_j] + b_i
     cov(x_i, x_j) = sum_{k in pa_j} w_{jk} cov(x_i, x_k) + I_{ij} v_j
   - Analytical mean and covariance for 3-node Gaussian model with missing link (Figure 11.7, Eq 11.14).
4. Plate Notation & Bayesian Classification (Sections 11.1.5 - 11.1.6, Eqs 11.17 - 11.19):
   - Stochastic nodes vs deterministic parameters vs observed shaded nodes.
5. Bayes' Theorem Graphical Inversion (Section 11.1.7, Figure 11.13, Eq 11.21):
   - Prior p(x), likelihood p(y|x), marginal evidence p(y), and posterior p(x|y).
6. High-Resolution Figure Reproductions (Figures 11.1 - 11.13):
   - Faithful textbook graphical models with publication-grade node styles, arrows, and plate boxes.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .plot_utils import setup_style, save_plot


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
# 1. Directed Acyclic Graph (DAG) Representation & Linear-Gaussian Networks
# =============================================================================

class DirectedGraph:
    """Directed Acyclic Graph (DAG) for probabilistic graphical models."""

    def __init__(self, nodes: Optional[List[str]] = None) -> None:
        self.nodes: List[str] = list(nodes) if nodes else []
        self.parents: Dict[str, List[str]] = {n: [] for n in self.nodes}
        self.children: Dict[str, List[str]] = {n: [] for n in self.nodes}

    def add_node(self, node: str) -> None:
        if node not in self.nodes:
            self.nodes.append(node)
            self.parents[node] = []
            self.children[node] = []

    def add_edge(self, parent: str, child: str) -> None:
        self.add_node(parent)
        self.add_node(child)
        if parent not in self.parents[child]:
            self.parents[child].append(parent)
        if child not in self.children[parent]:
            self.children[parent].append(child)

    def topological_sort(self) -> List[str]:
        """Kahn's algorithm for topological sorting."""
        in_degree = {n: len(self.parents[n]) for n in self.nodes}
        queue = [n for n in self.nodes if in_degree[n] == 0]
        sorted_nodes = []

        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)
            for ch in self.children[node]:
                in_degree[ch] -= 1
                if in_degree[ch] == 0:
                    queue.append(ch)

        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("Graph contains a directed cycle; not a valid DAG.")
        return sorted_nodes

    def factorization_formula(self) -> str:
        """Return the joint probability factorization string (Eq 11.6)."""
        factors = []
        for n in self.nodes:
            pa = self.parents[n]
            if not pa:
                factors.append(f"p({n})")
            else:
                factors.append(f"p({n} | {', '.join(pa)})")
        return " * ".join(factors)


class LinearGaussianDAG:
    """
    Linear-Gaussian Graphical Model (Bishop 2024, Section 11.1.4, Eqs 11.7 - 11.14).
    p(x_i | pa_i) = N(x_i | sum_{j in pa_i} w_ij x_j + b_i, v_i)
    """

    def __init__(self, dag: DirectedGraph, weights: Dict[Tuple[str, str], float],
                 biases: Dict[str, float], variances: Dict[str, float]) -> None:
        self.dag = dag
        self.weights = weights      # (child, parent) -> w_ij
        self.biases = biases        # node -> b_i
        self.variances = variances  # node -> v_i
        self.order = self.dag.topological_sort()

    def compute_joint_mean_and_cov(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute joint mean vector E[x] and covariance matrix cov(x_i, x_j)
        using recursion formulas (Eqs 11.8 - 11.9).
        """
        D = len(self.order)
        node_to_idx = {n: i for i, n in enumerate(self.order)}
        means = np.zeros(D, dtype=np.float64)
        cov = np.zeros((D, D), dtype=np.float64)

        # 1. Compute means via forward recursion: E[x_i] = sum_{j in pa_i} w_ij E[x_j] + b_i
        for i, n in enumerate(self.order):
            m_i = self.biases.get(n, 0.0)
            for p in self.dag.parents[n]:
                j = node_to_idx[p]
                w_ij = self.weights.get((n, p), 0.0)
                m_i += w_ij * means[j]
            means[i] = m_i

        # 2. Compute covariance via recursion:
        # cov(x_i, x_j) = sum_{k in pa_j} w_jk cov(x_i, x_k) + I_ij v_j (for j >= i)
        for j, n_j in enumerate(self.order):
            v_j = self.variances.get(n_j, 1.0)
            for i in range(j + 1):
                n_i = self.order[i]
                c_ij = 0.0
                for p in self.dag.parents[n_j]:
                    k = node_to_idx[p]
                    w_jk = self.weights.get((n_j, p), 0.0)
                    c_ij += w_jk * cov[i, k]
                if i == j:
                    c_ij += v_j
                cov[i, j] = c_ij
                cov[j, i] = c_ij

        return means, cov


# =============================================================================
# 2. Discrete Variables & Parameter Counting (Section 11.1.3)
# =============================================================================

def count_discrete_parameters(
    model_type: str,
    num_nodes: int = 2,
    num_states: int = 2,
    num_parents: int = 2,
) -> int:
    """
    Count the number of free parameters for discrete graphical models.
    - 'independent': M independent K-state variables -> M * (K - 1)
    - 'fully_connected_two': 2 fully connected K-state variables -> K^2 - 1
    - 'chain_general': Chain of M K-state variables with separate CPTs -> (K - 1) + (M - 1) * K * (K - 1)
    - 'chain_shared': Chain of M K-state variables with shared transition CPT -> (K - 1) + K * (K - 1) = K^2 - 1
    - 'parent_child_full': M binary parents and 1 binary child -> M + 2^M
    - 'parent_child_logistic': M binary parents and 1 binary child with logistic sigmoid -> M + (M + 1)
    """
    K = num_states
    M = num_nodes

    if model_type == 'independent':
        return M * (K - 1)
    elif model_type == 'fully_connected_two':
        return K ** 2 - 1
    elif model_type == 'chain_general':
        return (K - 1) + (M - 1) * K * (K - 1)
    elif model_type == 'chain_shared':
        return (K - 1) + K * (K - 1)
    elif model_type == 'parent_child_full':
        m_p = num_parents
        return m_p + (2 ** m_p)
    elif model_type == 'parent_child_logistic':
        m_p = num_parents
        return m_p + (m_p + 1)
    raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# 3. Bayes' Theorem Graphical Inversion (Section 11.1.7, Eq 11.21)
# =============================================================================

def compute_bayes_discrete(p_x: np.ndarray, p_y_given_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute marginal p(y) and posterior p(x | y) via Bayes' theorem.
    Args:
        p_x: (K_x,) prior distribution.
        p_y_given_x: (K_y, K_x) conditional likelihood matrix where [j, i] = p(y=j | x=i).
    Returns:
        p_y: (K_y,) marginal distribution over y.
        p_x_given_y: (K_x, K_y) posterior distribution matrix where [i, j] = p(x=i | y=j).
    """
    p_x = np.asarray(p_x, dtype=np.float64)
    p_y_given_x = np.asarray(p_y_given_x, dtype=np.float64)

    # Joint: p(x_i, y_j) = p(y_j | x_i) * p(x_i)
    # Shape: (K_x, K_y)
    joint = p_y_given_x.T * p_x[:, np.newaxis]

    # Marginal: p(y_j) = sum_i p(x_i, y_j)
    p_y = np.sum(joint, axis=0)

    # Posterior: p(x_i | y_j) = p(x_i, y_j) / p(y_j)
    p_x_given_y = joint / p_y[np.newaxis, :]
    return p_y, p_x_given_y


# =============================================================================
# 4. High-Resolution Figure Reproductions (Figures 11.1 - 11.13)
# =============================================================================

def _draw_node(ax: plt.Axes, pos: Tuple[float, float], label: str,
               radius: float = 0.35, is_observed: bool = False,
               is_deterministic: bool = False, facecolor: Optional[str] = None,
               fontsize: float = 11.0) -> None:
    """Helper to draw probabilistic graphical model nodes."""
    x, y = pos
    if is_deterministic:
        # Small solid dot
        ax.plot(x, y, 'o', color='#2c3e50', markersize=7, zorder=5)
        ax.text(x, y + 0.28, label, ha='center', va='bottom', fontsize=fontsize, fontweight='bold')
        return

    fc = facecolor if facecolor else ('#bdc3c7' if is_observed else '#ffffff')
    circle = patches.Circle((x, y), radius, facecolor=fc, edgecolor='#2c3e50', lw=2.0, zorder=4)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, fontweight='bold', zorder=5)


def _draw_arrow(ax: plt.Axes, pos1: Tuple[float, float], pos2: Tuple[float, float],
                radius: float = 0.35, color: str = '#2c3e50', lw: float = 1.8) -> None:
    """Helper to draw directed edge with proper boundary offset."""
    x1, y1 = pos1
    x2, y2 = pos2
    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if dist < 2 * radius:
        return
    # Truncate at circle boundary
    ux, uy = dx / dist, dy / dist
    start_x = x1 + ux * radius
    start_y = y1 + uy * radius
    end_x = x2 - ux * radius
    end_y = y2 - uy * radius
    ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14),
                zorder=3)


def generate_figure_11_1(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.1: A directed graphical model representing joint distribution over a, b, c:
    p(a, b, c) = p(a) p(b | a) p(c | a, b).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 3.5)
    ax.axis('off')
    ax.set_title("Figure 11.1: Directed Graphical Model for $p(a, b, c) = p(a)p(b|a)p(c|a,b)$",
                 fontsize=11.5, fontweight='bold', pad=12)

    pos_a = (1.0, 2.5)
    pos_b = (3.0, 2.5)
    pos_c = (2.0, 0.8)

    _draw_node(ax, pos_a, "$a$")
    _draw_node(ax, pos_b, "$b$")
    _draw_node(ax, pos_c, "$c$")

    _draw_arrow(ax, pos_a, pos_b)
    _draw_arrow(ax, pos_a, pos_c)
    _draw_arrow(ax, pos_b, pos_c)

    plt.tight_layout()
    _save_figure(fig, "Figure_11_1", save_dir)
    return fig


def generate_figure_11_2(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.2: Directed graph describing joint distribution over 7 variables:
    p(x1) p(x2) p(x3) p(x4 | x1, x2, x3) p(x5 | x1, x3) p(x6 | x4) p(x7 | x4, x5).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')
    ax.set_title("Figure 11.2: Directed Graph Describing Joint Distribution over $x_1$ to $x_7$",
                 fontsize=11.5, fontweight='bold', pad=12)

    positions = {
        "x1": (1.0, 3.5),
        "x2": (3.5, 3.5),
        "x3": (6.0, 3.5),
        "x4": (2.2, 2.0),
        "x5": (4.8, 2.0),
        "x6": (1.2, 0.6),
        "x7": (3.5, 0.6),
    }

    for name, pos in positions.items():
        _draw_node(ax, pos, f"${name[0]}_{{{name[1]}}}$")

    # Edges according to Eq 11.5:
    # x4 has parents x1, x2, x3
    _draw_arrow(ax, positions["x1"], positions["x4"])
    _draw_arrow(ax, positions["x2"], positions["x4"])
    _draw_arrow(ax, positions["x3"], positions["x4"])

    # x5 has parents x1, x3
    _draw_arrow(ax, positions["x1"], positions["x5"])
    _draw_arrow(ax, positions["x3"], positions["x5"])

    # x6 has parent x4
    _draw_arrow(ax, positions["x4"], positions["x6"])

    # x7 has parents x4, x5
    _draw_arrow(ax, positions["x4"], positions["x7"])
    _draw_arrow(ax, positions["x5"], positions["x7"])

    plt.tight_layout()
    _save_figure(fig, "Figure_11_2", save_dir)
    return fig


def generate_figure_11_3(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.3: Discrete variables.
    (a) Fully connected: x1 -> x2 (K^2 - 1 parameters).
    (b) Independent: x1 and x2 without edge (2(K - 1) parameters).
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.6))

    for ax in [ax1, ax2]:
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 2.5)
        ax.axis('off')

    # (a) Fully connected
    ax1.set_title("(a) Fully Connected ($K^2 - 1$ params)", fontsize=10.5, fontweight='bold', pad=10)
    p1 = (1.2, 1.2)
    p2 = (3.2, 1.2)
    _draw_node(ax1, p1, "$x_1$")
    _draw_node(ax1, p2, "$x_2$")
    _draw_arrow(ax1, p1, p2)

    # (b) Independent
    ax2.set_title("(b) Independent ($2(K - 1)$ params)", fontsize=10.5, fontweight='bold', pad=10)
    _draw_node(ax2, p1, "$x_1$")
    _draw_node(ax2, p2, "$x_2$")

    plt.suptitle("Figure 11.3: Directed Graphs over Two Discrete Variables", fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure(fig, "Figure_11_3", save_dir)
    return fig


def generate_figure_11_4(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.4: Chain of M discrete nodes:
    x1 -> x2 -> x3 -> ... -> xM-1 -> xM
    General CPTs require (K - 1) + (M - 1) K (K - 1) parameters.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 2.0)
    ax.axis('off')
    ax.set_title("Figure 11.4: Chain of $M$ Discrete Nodes ($O(M)$ Parameter Growth)",
                 fontsize=11.5, fontweight='bold', pad=10)

    nodes = [
        ((1.0, 0.9), "$x_1$"),
        ((2.8, 0.9), "$x_2$"),
        ((4.6, 0.9), "$x_3$"),
        ((7.4, 0.9), "$x_M$"),
    ]

    for pos, lbl in nodes:
        _draw_node(ax, pos, lbl)

    _draw_arrow(ax, nodes[0][0], nodes[1][0])
    _draw_arrow(ax, nodes[1][0], nodes[2][0])

    # Dots between x3 and xM
    ax.text(5.9, 0.9, "$\\dots$", ha='center', va='center', fontsize=18, fontweight='bold', color='#2c3e50')
    _draw_arrow(ax, (6.3, 0.9), nodes[3][0], radius=0.0)

    plt.tight_layout()
    _save_figure(fig, "Figure_11_4", save_dir)
    return fig


def generate_figure_11_5(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.5: Homogeneous Markov Chain (parameter sharing across transitions).
    All conditional distributions p(xi | xi-1) share the same K(K - 1) parameters.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 2.0)
    ax.axis('off')
    ax.set_title("Figure 11.5: Homogeneous Markov Chain with Shared Transition Parameters ($K^2 - 1$ params)",
                 fontsize=11.5, fontweight='bold', pad=10)

    nodes = [
        ((1.0, 0.9), "$x_1$"),
        ((2.8, 0.9), "$x_2$"),
        ((4.6, 0.9), "$x_3$"),
        ((7.4, 0.9), "$x_M$"),
    ]

    for pos, lbl in nodes:
        _draw_node(ax, pos, lbl)

    _draw_arrow(ax, nodes[0][0], nodes[1][0])
    _draw_arrow(ax, nodes[1][0], nodes[2][0])

    # Shared parameter labels over arrows
    ax.text(1.9, 1.35, "shared $\\mathbf{T}$", ha='center', fontsize=8.0, color='#8e44ad', fontweight='bold')
    ax.text(3.7, 1.35, "shared $\\mathbf{T}$", ha='center', fontsize=8.0, color='#8e44ad', fontweight='bold')

    ax.text(5.9, 0.9, "$\\dots$", ha='center', va='center', fontsize=18, fontweight='bold', color='#2c3e50')
    _draw_arrow(ax, (6.3, 0.9), nodes[3][0], radius=0.0)

    plt.tight_layout()
    _save_figure(fig, "Figure_11_5", save_dir)
    return fig


def generate_figure_11_6(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.6: Graph comprising M parents x1, ..., xM and a single child y.
    Unparameterized: 2^M. Parameterized logistic: M + 1.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 3.8)
    ax.axis('off')
    ax.set_title("Figure 11.6: $M$ Parents $x_1, \\dots, x_M$ and Single Child $y$",
                 fontsize=11.5, fontweight='bold', pad=12)

    parents = [
        ((1.0, 2.8), "$x_1$"),
        ((2.5, 2.8), "$x_2$"),
        ((4.0, 2.8), "$x_3$"),
        ((6.2, 2.8), "$x_M$"),
    ]
    child = ((3.6, 0.8), "$y$")

    for pos, lbl in parents:
        _draw_node(ax, pos, lbl)
    _draw_node(ax, child[0], child[1])

    for pos, _ in parents[:3]:
        _draw_arrow(ax, pos, child[0])

    ax.text(5.1, 2.8, "$\\dots$", ha='center', va='center', fontsize=18, fontweight='bold', color='#2c3e50')
    _draw_arrow(ax, parents[3][0], child[0])

    plt.tight_layout()
    _save_figure(fig, "Figure_11_6", save_dir)
    return fig


def generate_figure_11_7(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.7: Directed graph over three Gaussian variables with one missing link:
    x1 -> x2 -> x3 (no link between x1 and x3).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')
    ax.set_title("Figure 11.7: Three Gaussian Variables with Missing Link ($x_1 \\to x_2 \\to x_3$)",
                 fontsize=11.5, fontweight='bold', pad=12)

    p1 = (1.0, 1.2)
    p2 = (3.0, 1.2)
    p3 = (5.0, 1.2)

    _draw_node(ax, p1, "$x_1$")
    _draw_node(ax, p2, "$x_2$")
    _draw_node(ax, p3, "$x_3$")

    _draw_arrow(ax, p1, p2)
    _draw_arrow(ax, p2, p3)

    # Edge weight annotations
    ax.text(2.0, 1.55, "$w_{21}$", ha='center', fontsize=9.5, fontweight='bold', color='#2980b9')
    ax.text(4.0, 1.55, "$w_{32}$", ha='center', fontsize=9.5, fontweight='bold', color='#2980b9')

    plt.tight_layout()
    _save_figure(fig, "Figure_11_7", save_dir)
    return fig


def generate_figure_11_8(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.8: Unrolled binary classifier showing stochastic variables t1, ..., tN and w.
    w points to each tn explicitly.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 3.2)
    ax.axis('off')
    ax.set_title("Figure 11.8: Unrolled Binary Classifier with $N$ Observation Nodes",
                 fontsize=11.5, fontweight='bold', pad=12)

    pos_w = (3.5, 2.4)
    _draw_node(ax, pos_w, "$\\mathbf{w}$")

    targets = [
        ((1.0, 0.8), "$t_1$"),
        ((2.3, 0.8), "$t_2$"),
        ((3.6, 0.8), "$t_3$"),
        ((6.2, 0.8), "$t_N$"),
    ]

    for pos, lbl in targets:
        _draw_node(ax, pos, lbl)
        _draw_arrow(ax, pos_w, pos)

    ax.text(4.9, 0.8, "$\\dots$", ha='center', va='center', fontsize=18, fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    _save_figure(fig, "Figure_11_8", save_dir)
    return fig


def generate_figure_11_9(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.9: Compact plate notation representing Figure 11.8.
    w points to tn surrounded by a plate box labelled N.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 3.8)
    ax.axis('off')
    ax.set_title("Figure 11.9: Compact Plate Notation (Box Labelled $N$)",
                 fontsize=11.5, fontweight='bold', pad=12)

    pos_w = (2.0, 2.8)
    pos_t = (2.0, 1.2)

    _draw_node(ax, pos_w, "$\\mathbf{w}$")
    _draw_node(ax, pos_t, "$t_n$")
    _draw_arrow(ax, pos_w, pos_t)

    # Draw plate box
    rect = patches.Rectangle((1.1, 0.3), 1.8, 1.6, fill=False, edgecolor='#2c3e50', lw=1.8)
    ax.add_patch(rect)
    ax.text(2.7, 0.45, "$N$", fontsize=12, fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    _save_figure(fig, "Figure_11_9", save_dir)
    return fig


def generate_figure_11_10(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.10: Same model as Figure 11.9 with deterministic parameters shown explicitly:
    alpha -> w -> tn <- xn (inside plate).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.set_xlim(-0.5, 5.0)
    ax.set_ylim(-0.5, 3.8)
    ax.axis('off')
    ax.set_title("Figure 11.10: Model with Deterministic Parameters $\\alpha$ and $\\mathbf{x}_n$",
                 fontsize=11.5, fontweight='bold', pad=12)

    pos_alpha = (0.8, 2.8)
    pos_w = (2.2, 2.8)
    pos_t = (2.2, 1.2)
    pos_x = (3.6, 1.2)

    # alpha is deterministic parameter (small solid dot)
    _draw_node(ax, pos_alpha, "$\\alpha$", is_deterministic=True)
    _draw_node(ax, pos_w, "$\\mathbf{w}$")
    _draw_node(ax, pos_t, "$t_n$")
    # xn is deterministic input inside plate (small solid dot)
    _draw_node(ax, pos_x, "$\\mathbf{x}_n$", is_deterministic=True)

    _draw_arrow(ax, pos_alpha, pos_w)
    _draw_arrow(ax, pos_w, pos_t)
    _draw_arrow(ax, pos_x, pos_t)

    # Plate box
    rect = patches.Rectangle((1.3, 0.3), 2.8, 1.6, fill=False, edgecolor='#2c3e50', lw=1.8)
    ax.add_patch(rect)
    ax.text(3.9, 0.45, "$N$", fontsize=12, fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    _save_figure(fig, "Figure_11_10", save_dir)
    return fig


def generate_figure_11_11(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.11: As in Figure 11.10 but with nodes {tn} shaded to indicate observed variables.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.set_xlim(-0.5, 5.0)
    ax.set_ylim(-0.5, 3.8)
    ax.axis('off')
    ax.set_title("Figure 11.11: Observed Training Targets Shaded ($t_n$ Observed)",
                 fontsize=11.5, fontweight='bold', pad=12)

    pos_alpha = (0.8, 2.8)
    pos_w = (2.2, 2.8)
    pos_t = (2.2, 1.2)
    pos_x = (3.6, 1.2)

    _draw_node(ax, pos_alpha, "$\\alpha$", is_deterministic=True)
    _draw_node(ax, pos_w, "$\\mathbf{w}$")
    _draw_node(ax, pos_t, "$t_n$", is_observed=True)  # SHADED
    _draw_node(ax, pos_x, "$\\mathbf{x}_n$", is_deterministic=True)

    _draw_arrow(ax, pos_alpha, pos_w)
    _draw_arrow(ax, pos_w, pos_t)
    _draw_arrow(ax, pos_x, pos_t)

    # Plate box
    rect = patches.Rectangle((1.3, 0.3), 2.8, 1.6, fill=False, edgecolor='#2c3e50', lw=1.8)
    ax.add_patch(rect)
    ax.text(3.9, 0.45, "$N$", fontsize=12, fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    _save_figure(fig, "Figure_11_11", save_dir)
    return fig


def generate_figure_11_12(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.12: The classification predictive model showing new input x_hat and prediction t_hat.
    alpha -> w -> tn (observed, plate N)
             w -> t_hat (unshaded) <- x_hat
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 3.8)
    ax.axis('off')
    ax.set_title("Figure 11.12: Predictive Distribution with Test Input $\\hat{\\mathbf{x}}$ and Target $\\hat{t}$",
                 fontsize=11.5, fontweight='bold', pad=12)

    pos_alpha = (0.8, 2.8)
    pos_w = (2.4, 2.8)
    pos_t = (2.0, 1.2)
    pos_x = (3.2, 1.2)

    # Test prediction nodes
    pos_that = (5.2, 1.2)
    pos_xhat = (6.4, 1.2)

    _draw_node(ax, pos_alpha, "$\\alpha$", is_deterministic=True)
    _draw_node(ax, pos_w, "$\\mathbf{w}$")
    _draw_node(ax, pos_t, "$t_n$", is_observed=True)  # Shaded
    _draw_node(ax, pos_x, "$\\mathbf{x}_n$", is_deterministic=True)

    _draw_node(ax, pos_that, "$\\hat{t}$", is_observed=False)  # Unshaded prediction
    _draw_node(ax, pos_xhat, "$\\hat{\\mathbf{x}}$", is_deterministic=True)

    _draw_arrow(ax, pos_alpha, pos_w)
    _draw_arrow(ax, pos_w, pos_t)
    _draw_arrow(ax, pos_x, pos_t)

    # w and xhat point to that
    _draw_arrow(ax, pos_w, pos_that)
    _draw_arrow(ax, pos_xhat, pos_that)

    # Plate box for training set
    rect = patches.Rectangle((1.2, 0.3), 2.5, 1.6, fill=False, edgecolor='#2c3e50', lw=1.8)
    ax.add_patch(rect)
    ax.text(3.5, 0.45, "$N$", fontsize=12, fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    _save_figure(fig, "Figure_11_12", save_dir)
    return fig


def generate_figure_11_13(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.13: A graphical representation of Bayes' theorem:
    (a) Joint distribution x -> y (prior p(x) and likelihood p(y | x)).
    (b) Observed/conditioned y (shaded).
    (c) Inverted graph y -> x with reversed arrow (posterior p(x | y) and evidence p(y)).
    """
    setup_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.5, 3.8))

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 3.5)
        ax.axis('off')

    pos_x = (1.5, 2.5)
    pos_y = (1.5, 0.9)

    # (a) Prior and likelihood
    ax1.set_title("(a) Prior & Likelihood\n$p(x) p(y|x)$", fontsize=10.0, fontweight='bold', pad=8)
    _draw_node(ax1, pos_x, "$x$")
    _draw_node(ax1, pos_y, "$y$")
    _draw_arrow(ax1, pos_x, pos_y)

    # (b) Observed y
    ax2.set_title("(b) Observed Evidence\n$y = \\hat{y}$ (Conditioned)", fontsize=10.0, fontweight='bold', pad=8)
    _draw_node(ax2, pos_x, "$x$")
    _draw_node(ax2, pos_y, "$y$", is_observed=True)
    _draw_arrow(ax2, pos_x, pos_y)

    # (c) Inverted representation (Bayes' rule)
    ax3.set_title("(c) Posterior Inference\n$p(y) p(x|y)$", fontsize=10.0, fontweight='bold', pad=8, color='#922b21')
    _draw_node(ax3, pos_y, "$y$", is_observed=True)
    _draw_node(ax3, pos_x, "$x$")
    _draw_arrow(ax3, pos_y, pos_x, color='#922b21')

    plt.suptitle("Figure 11.13: Graphical Representation of Bayes' Theorem and Inference",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure(fig, "Figure_11_13", save_dir)
    return fig


if __name__ == "__main__":
    for i in range(1, 14):
        func_name = f"generate_figure_11_{i}"
        func = globals()[func_name]
        fig = func()
        plt.close(fig)
    print("All 13 figures for Section 11.1 generated successfully!")
