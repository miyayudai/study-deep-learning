"""
common/conditional_independence.py
==================================
Section 11.2: Conditional Independence
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements:
1. Three Canonical 3-Node Graphs (Section 11.2.1):
   - Tail-to-tail (Figures 11.14, 11.15): a <- c -> b. Unconditioned: dependent; Conditioned on c: a _|_ b | c.
   - Head-to-tail (Figures 11.16, 11.17): a -> c -> b. Unconditioned: dependent; Conditioned on c: a _|_ b | c.
   - Head-to-head (Figures 11.18, 11.19): a -> c <- b. Unconditioned: a _|_ b | empty; Conditioned on c: a /_|_ b | c.
2. Explaining Away in Car Fuel System (Section 11.2.2, Figure 11.20, Eqs 11.32 - 11.35):
   - B (battery), F (fuel), G (gauge).
   - p(G=0) = 0.315
   - p(F=0 | G=0) = 0.257
   - p(F=0 | G=0, B=0) = 0.111 (dead battery explains away empty gauge).
3. D-Separation Algorithm (Section 11.2.3, Figure 11.21):
   - Path-blocking criteria for general DAGs.
4. Naive Bayes & Generative Models (Sections 11.2.4 - 11.2.5, Figures 11.22, 11.23, 11.24):
   - Class-conditional factorized distributions vs non-factorized marginal mixture.
5. Markov Blanket (Section 11.2.6, Figure 11.25):
   - MB(x_i) = parents + children + co-parents.
6. Graphs as Filters (Section 11.2.7, Figure 11.26):
   - P (all distributions), DF (distributions factorized over G), UI (unfaithful).
7. High-Resolution Figure Reproductions (Figures 11.14 - 11.26).
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
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
# 1. Car Fuel System ("Explaining Away", Section 11.2.2, Eqs 11.32 - 11.35)
# =============================================================================

class CarFuelSystem:
    """
    Bishop (2024, pp. 341-343) Car Fuel Tank, Battery, and Gauge Example.
    B in {0, 1}: Battery (1 = charged, 0 = flat)
    F in {0, 1}: Fuel (1 = full, 0 = empty)
    G in {0, 1}: Gauge (1 = reads full, 0 = reads empty)
    """

    def __init__(self) -> None:
        self.p_B1 = 0.9
        self.p_B0 = 0.1
        self.p_F1 = 0.9
        self.p_F0 = 0.1

        # p(G = 1 | B, F) from Bishop (2024, p. 342)
        self.p_G1_given_BF = {
            (1, 1): 0.8,
            (1, 0): 0.2,
            (0, 1): 0.2,
            (0, 0): 0.1,
        }

    def p_G0_given_BF(self, b: int, f: int) -> float:
        return 1.0 - self.p_G1_given_BF[(b, f)]

    def compute_marginal_p_G0(self) -> float:
        """Eq 11.32: sum_{B, F} p(G=0 | B, F) p(B) p(F) = 0.315"""
        val = 0.0
        for b, p_b in [(1, self.p_B1), (0, self.p_B0)]:
            for f, p_f in [(1, self.p_F1), (0, self.p_F0)]:
                val += self.p_G0_given_BF(b, f) * p_b * p_f
        return float(val)

    def compute_conditional_p_G0_given_F0(self) -> float:
        """Eq 11.33: sum_B p(G=0 | B, F=0) p(B) = 0.81"""
        val = 0.0
        for b, p_b in [(1, self.p_B1), (0, self.p_B0)]:
            val += self.p_G0_given_BF(b, 0) * p_b
        return float(val)

    def compute_posterior_p_F0_given_G0(self) -> float:
        """
        Eq 11.34: p(F=0 | G=0) = p(G=0 | F=0) p(F=0) / p(G=0)
        = 0.81 * 0.1 / 0.315 approx 0.257
        """
        num = self.compute_conditional_p_G0_given_F0() * self.p_F0
        denom = self.compute_marginal_p_G0()
        return float(num / denom)

    def compute_posterior_p_F0_given_G0_B0(self) -> float:
        """
        Eq 11.35: p(F=0 | G=0, B=0) = p(G=0 | B=0, F=0) p(F=0) / sum_F p(G=0 | B=0, F) p(F)
        p(G=0 | B=0, F=0) = 1.0, p(F=0) = 0.1 -> num = 0.1
        denom = 1.0 * 0.1 + 0.9 * 0.9 = 0.1 + 0.81 = 0.91
        0.1 / 0.91 approx 0.111
        """
        p_G0_B0_F0 = self.p_G0_given_BF(0, 0) * self.p_F0
        p_G0_B0_F1 = self.p_G0_given_BF(0, 1) * self.p_F1
        denom = p_G0_B0_F0 + p_G0_B0_F1
        return float(p_G0_B0_F0 / denom)


# =============================================================================
# 2. D-Separation Algorithm (Section 11.2.3, Figure 11.21)
# =============================================================================

def get_descendants(dag: DirectedGraph, node: str) -> Set[str]:
    """Get all descendants of a node in DAG (including the node itself)."""
    desc = {node}
    queue = [node]
    while queue:
        curr = queue.pop(0)
        for ch in dag.children.get(curr, []):
            if ch not in desc:
                desc.add(ch)
                queue.append(ch)
    return desc


def get_markov_blanket(dag: DirectedGraph, node: str) -> Set[str]:
    """
    Section 11.2.6 & Figure 11.25:
    Markov blanket MB(x_i) = parents(x_i) U children(x_i) U co_parents(x_i)
    """
    parents = set(dag.parents.get(node, []))
    children = set(dag.children.get(node, []))
    co_parents = set()
    for ch in children:
        for p in dag.parents.get(ch, []):
            if p != node:
                co_parents.add(p)
    return parents | children | co_parents


def check_d_separation(
    dag: DirectedGraph,
    A: Set[str],
    B: Set[str],
    C: Set[str],
) -> bool:
    """
    D-Separation Criterion (Section 11.2.3):
    Returns True if A is conditionally independent of B given C (A _|_ B | C).
    """
    # Precompute descendants for all nodes
    all_descendants = {n: get_descendants(dag, n) for n in dag.nodes}

    # Find all simple undirected paths between any a in A and b in B
    # Build undirected adjacency list with edge direction tags
    adj: Dict[str, List[Tuple[str, str]]] = {n: [] for n in dag.nodes}
    for n in dag.nodes:
        for ch in dag.children.get(n, []):
            adj[n].append((ch, "forward"))  # n -> ch
            adj[ch].append((n, "backward"))  # ch <- n

    def find_paths(start: str, target: str, visited: Set[str]) -> List[List[Tuple[str, Optional[str]]]]:
        if start == target:
            return [[(start, None)]]
        paths = []
        for nbr, direction in adj[start]:
            if nbr not in visited:
                sub_paths = find_paths(nbr, target, visited | {nbr})
                for sp in sub_paths:
                    paths.append([(start, direction)] + sp)
        return paths

    for a in A:
        for b in B:
            paths = find_paths(a, b, {a})
            for p in paths:
                # Check if path p is blocked by conditioning set C
                is_path_blocked = False
                # A path has nodes: p[0][0], p[1][0], ..., p[len-1][0]
                for i in range(1, len(p) - 1):
                    prev_node = p[i - 1][0]
                    curr_node = p[i][0]
                    next_node = p[i + 1][0]

                    # Determine node configuration at curr_node:
                    # prev_node -> curr_node <- next_node: head-to-head
                    is_in_prev = curr_node in dag.children.get(prev_node, [])
                    is_in_next = curr_node in dag.children.get(next_node, [])

                    is_head_to_head = is_in_prev and is_in_next

                    if is_head_to_head:
                        # Head-to-head node: BLOCKED if NEITHER curr_node nor any of its descendants is in C
                        curr_desc = all_descendants[curr_node]
                        if not (curr_desc & C):
                            is_path_blocked = True
                            break
                    else:
                        # Tail-to-tail or head-to-tail: BLOCKED if curr_node is in C
                        if curr_node in C:
                            is_path_blocked = True
                            break

                if not is_path_blocked:
                    # Found an active (unblocked) path between A and B!
                    return False

    return True


# =============================================================================
# 3. High-Resolution Figure Reproductions (Figures 11.14 - 11.26)
# =============================================================================

def generate_figure_11_14(save_dir: Optional[str] = None) -> plt.Figure:
    """Figure 11.14: Tail-to-tail graph over a, b, c: a <- c -> b."""
    setup_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 3.2)
    ax.axis('off')
    ax.set_title("Figure 11.14: Tail-to-Tail Graph ($a \\not\\perp b \\mid \\emptyset$)",
                 fontsize=11.0, fontweight='bold', pad=10)

    pos_c = (2.0, 2.3)
    pos_a = (0.8, 0.7)
    pos_b = (3.2, 0.7)

    _draw_node(ax, pos_c, "$c$")
    _draw_node(ax, pos_a, "$a$")
    _draw_node(ax, pos_b, "$b$")

    _draw_arrow(ax, pos_c, pos_a)
    _draw_arrow(ax, pos_c, pos_b)

    plt.tight_layout()
    _save_figure(fig, "Figure_11_14", save_dir)
    return fig


def generate_figure_11_15(save_dir: Optional[str] = None) -> plt.Figure:
    """Figure 11.15: Tail-to-tail conditioned on c: a <- [c] -> b ($a _|_ b | c$)."""
    setup_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 3.2)
    ax.axis('off')
    ax.set_title("Figure 11.15: Conditioned Tail-to-Tail ($a \\perp b \\mid c$)",
                 fontsize=11.0, fontweight='bold', pad=10)

    pos_c = (2.0, 2.3)
    pos_a = (0.8, 0.7)
    pos_b = (3.2, 0.7)

    _draw_node(ax, pos_c, "$c$", is_observed=True)  # Shaded
    _draw_node(ax, pos_a, "$a$")
    _draw_node(ax, pos_b, "$b$")

    _draw_arrow(ax, pos_c, pos_a)
    _draw_arrow(ax, pos_c, pos_b)

    plt.tight_layout()
    _save_figure(fig, "Figure_11_15", save_dir)
    return fig


def generate_figure_11_16(save_dir: Optional[str] = None) -> plt.Figure:
    """Figure 11.16: Head-to-tail graph: a -> c -> b ($a not_|_ b | empty)."""
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 2.2)
    ax.axis('off')
    ax.set_title("Figure 11.16: Head-to-Tail Graph ($a \\not\\perp b \\mid \\emptyset$)",
                 fontsize=11.0, fontweight='bold', pad=10)

    pos_a = (0.8, 0.9)
    pos_c = (2.5, 0.9)
    pos_b = (4.2, 0.9)

    _draw_node(ax, pos_a, "$a$")
    _draw_node(ax, pos_c, "$c$")
    _draw_node(ax, pos_b, "$b$")

    _draw_arrow(ax, pos_a, pos_c)
    _draw_arrow(ax, pos_c, pos_b)

    plt.tight_layout()
    _save_figure(fig, "Figure_11_16", save_dir)
    return fig


def generate_figure_11_17(save_dir: Optional[str] = None) -> plt.Figure:
    """Figure 11.17: Head-to-tail conditioned on c: a -> [c] -> b ($a _|_ b | c$)."""
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 2.2)
    ax.axis('off')
    ax.set_title("Figure 11.17: Conditioned Head-to-Tail ($a \\perp b \\mid c$)",
                 fontsize=11.0, fontweight='bold', pad=10)

    pos_a = (0.8, 0.9)
    pos_c = (2.5, 0.9)
    pos_b = (4.2, 0.9)

    _draw_node(ax, pos_a, "$a$")
    _draw_node(ax, pos_c, "$c$", is_observed=True)  # Shaded
    _draw_node(ax, pos_b, "$b$")

    _draw_arrow(ax, pos_a, pos_c)
    _draw_arrow(ax, pos_c, pos_b)

    plt.tight_layout()
    _save_figure(fig, "Figure_11_17", save_dir)
    return fig


def generate_figure_11_18(save_dir: Optional[str] = None) -> plt.Figure:
    """Figure 11.18: Head-to-head graph: a -> c <- b ($a _|_ b | empty)."""
    setup_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 3.2)
    ax.axis('off')
    ax.set_title("Figure 11.18: Head-to-Head Graph ($a \\perp b \\mid \\emptyset$)",
                 fontsize=11.0, fontweight='bold', pad=10)

    pos_a = (0.8, 2.3)
    pos_b = (3.2, 2.3)
    pos_c = (2.0, 0.7)

    _draw_node(ax, pos_a, "$a$")
    _draw_node(ax, pos_b, "$b$")
    _draw_node(ax, pos_c, "$c$")

    _draw_arrow(ax, pos_a, pos_c)
    _draw_arrow(ax, pos_b, pos_c)

    plt.tight_layout()
    _save_figure(fig, "Figure_11_18", save_dir)
    return fig


def generate_figure_11_19(save_dir: Optional[str] = None) -> plt.Figure:
    """Figure 11.19: Head-to-head conditioned on c: a -> [c] <- b ($a not_|_ b | c)."""
    setup_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 3.2)
    ax.axis('off')
    ax.set_title("Figure 11.19: Conditioned Head-to-Head ($a \\not\\perp b \\mid c$)",
                 fontsize=11.0, fontweight='bold', pad=10)

    pos_a = (0.8, 2.3)
    pos_b = (3.2, 2.3)
    pos_c = (2.0, 0.7)

    _draw_node(ax, pos_a, "$a$")
    _draw_node(ax, pos_b, "$b$")
    _draw_node(ax, pos_c, "$c$", is_observed=True)  # Shaded

    _draw_arrow(ax, pos_a, pos_c)
    _draw_arrow(ax, pos_b, pos_c)

    plt.tight_layout()
    _save_figure(fig, "Figure_11_19", save_dir)
    return fig


def generate_figure_11_20(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.20: Explaining away in car fuel system:
    Left: unobserved prior (B _|_ F).
    Middle: G = 0 observed (p(F=0 | G=0) approx 0.257).
    Right: G = 0 and B = 0 observed (p(F=0 | G=0, B=0) approx 0.111).
    """
    setup_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.5, 4.2))

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 3.5)
        ax.axis('off')

    pos_b = (0.7, 2.4)
    pos_f = (2.3, 2.4)
    pos_g = (1.5, 0.8)

    # 1. Left: Prior
    ax1.set_title("(a) Prior: $B \\perp F$\n$p(F=0) = 0.1$", fontsize=9.5, fontweight='bold', pad=8)
    _draw_node(ax1, pos_b, "$B$")
    _draw_node(ax1, pos_f, "$F$")
    _draw_node(ax1, pos_g, "$G$")
    _draw_arrow(ax1, pos_b, pos_g)
    _draw_arrow(ax1, pos_f, pos_g)

    # 2. Middle: Empty Gauge Observed
    ax2.set_title("(b) Gauge Empty $G=0$\n$p(F=0 \\mid G=0) \\approx 0.257$", fontsize=9.5, fontweight='bold', pad=8)
    _draw_node(ax2, pos_b, "$B$")
    _draw_node(ax2, pos_f, "$F$")
    _draw_node(ax2, pos_g, "$G$", is_observed=True)
    _draw_arrow(ax2, pos_b, pos_g)
    _draw_arrow(ax2, pos_f, pos_g)

    # 3. Right: Flat Battery Also Observed
    ax3.set_title("(c) Battery Flat $B=0, G=0$\n$p(F=0 \\mid G=0, B=0) \\approx 0.111$", fontsize=9.5, fontweight='bold', pad=8, color='#922b21')
    _draw_node(ax3, pos_b, "$B$", is_observed=True)
    _draw_node(ax3, pos_f, "$F$")
    _draw_node(ax3, pos_g, "$G$", is_observed=True)
    _draw_arrow(ax3, pos_b, pos_g)
    _draw_arrow(ax3, pos_f, pos_g)

    plt.suptitle("Figure 11.20: Explaining Away Phenomenon in Car Fuel System",
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    _save_figure(fig, "Figure_11_20", save_dir)
    return fig


def generate_figure_11_21(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.21: Illustration of D-separation:
    (a) Path from a to b blocked by conditioned node e.
    (b) Path unblocked when head-to-head node c is conditioned.
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2))

    for ax in [ax1, ax2]:
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-0.5, 4.0)
        ax.axis('off')

    pos = {
        "a": (0.8, 1.8),
        "e": (2.2, 3.0),
        "c": (3.5, 1.8),
        "b": (5.2, 1.8),
    }

    # (a) Path blocked by conditioned node e
    ax1.set_title("(a) Blocked Path ($a \\perp b \\mid e$)\n(Head-to-tail node $e$ is observed)",
                  fontsize=10.0, fontweight='bold', pad=10)
    _draw_node(ax1, pos["a"], "$a$")
    _draw_node(ax1, pos["e"], "$e$", is_observed=True)
    _draw_node(ax1, pos["c"], "$c$")
    _draw_node(ax1, pos["b"], "$b$")
    _draw_arrow(ax1, pos["a"], pos["e"])
    _draw_arrow(ax1, pos["e"], pos["c"])
    _draw_arrow(ax1, pos["c"], pos["b"])

    # (b) Path unblocked when head-to-head node c is conditioned
    # a -> c <- d, c -> b
    pos2 = {
        "a": (0.8, 2.5),
        "d": (0.8, 0.8),
        "c": (3.0, 1.6),
        "b": (5.2, 1.6),
    }
    ax2.set_title("(b) Unblocked Path ($a \\not\\perp d \\mid c$)\n(Head-to-head node $c$ is observed)",
                  fontsize=10.0, fontweight='bold', pad=10)
    _draw_node(ax2, pos2["a"], "$a$")
    _draw_node(ax2, pos2["d"], "$d$")
    _draw_node(ax2, pos2["c"], "$c$", is_observed=True)
    _draw_node(ax2, pos2["b"], "$b$")
    _draw_arrow(ax2, pos2["a"], pos2["c"])
    _draw_arrow(ax2, pos2["d"], pos2["c"])
    _draw_arrow(ax2, pos2["c"], pos2["b"])

    plt.suptitle("Figure 11.21: Illustration of D-Separation Path-Blocking Criteria",
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    _save_figure(fig, "Figure_11_21", save_dir)
    return fig


def generate_figure_11_22(save_dir: Optional[str] = None) -> plt.Figure:
    """Figure 11.22: Graphical representation of the naive Bayes model."""
    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 3.2)
    ax.axis('off')
    ax.set_title("Figure 11.22: Naive Bayes Model for Classification ($C_k \\to x^{(1)}, \\dots, x^{(L)}$)",
                 fontsize=11.5, fontweight='bold', pad=12)

    pos_c = (3.0, 2.4)
    _draw_node(ax, pos_c, "$C_k$")

    features = [
        ((0.8, 0.8), "$x^{(1)}$"),
        ((2.0, 0.8), "$x^{(2)}$"),
        ((3.2, 0.8), "$x^{(3)}$"),
        ((5.2, 0.8), "$x^{(L)}$"),
    ]

    for pos, lbl in features:
        _draw_node(ax, pos, lbl)
        _draw_arrow(ax, pos_c, pos)

    ax.text(4.2, 0.8, "$\\dots$", ha='center', va='center', fontsize=18, fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    _save_figure(fig, "Figure_11_22", save_dir)
    return fig


def generate_figure_11_23(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.23: Illustration of a naive Bayes classifier for a 2D data space:
    (a) Class-conditional distributions p(x | Ck) that factorize.
    (b) Marginal mixture distribution p(x) that does NOT factorize.
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.5))

    # Grid for contours
    x1 = np.linspace(-4, 4, 150)
    x2 = np.linspace(-4, 4, 150)
    X1, X2 = np.meshgrid(x1, x2)

    # Class 1: Gaussian at (-1.2, -1.2), diagonal cov
    mu1 = np.array([-1.2, -1.2])
    cov1 = np.array([[0.8, 0.0], [0.0, 0.8]])
    inv_cov1 = np.linalg.inv(cov1)
    diff1 = np.stack([X1 - mu1[0], X2 - mu1[1]], axis=-1)
    p_c1 = np.exp(-0.5 * np.einsum('ijk,kl,ijl->ij', diff1, inv_cov1, diff1)) / (2 * np.pi * np.sqrt(np.linalg.det(cov1)))

    # Class 2: Gaussian at (1.2, 1.2), diagonal cov
    mu2 = np.array([1.2, 1.2])
    cov2 = np.array([[0.8, 0.0], [0.0, 0.8]])
    inv_cov2 = np.linalg.inv(cov2)
    diff2 = np.stack([X1 - mu2[0], X2 - mu2[1]], axis=-1)
    p_c2 = np.exp(-0.5 * np.einsum('ijk,kl,ijl->ij', diff2, inv_cov2, diff2)) / (2 * np.pi * np.sqrt(np.linalg.det(cov2)))

    # (a) Conditional distributions
    ax1.contour(X1, X2, p_c1, levels=5, colors='#2980b9', linewidths=1.5)
    ax1.contour(X1, X2, p_c2, levels=5, colors='#c0392b', linewidths=1.5)
    ax1.text(-1.2, -2.5, "$p(\\mathbf{x} \\mid C_1)$", color='#2980b9', fontsize=11, fontweight='bold', ha='center')
    ax1.text(1.2, 2.5, "$p(\\mathbf{x} \\mid C_2)$", color='#c0392b', fontsize=11, fontweight='bold', ha='center')
    ax1.set_title("(a) Factorized Class Conditionals\n$p(\\mathbf{x} \\mid C_k) = p(x_1 \\mid C_k) p(x_2 \\mid C_k)$",
                  fontsize=10.0, fontweight='bold', pad=10)
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(0, color='gray', linestyle=':', alpha=0.5)

    # (b) Marginal mixture distribution: p(x) = 0.5 * p_c1 + 0.5 * p_c2
    p_marginal = 0.5 * p_c1 + 0.5 * p_c2
    cs = ax2.contour(X1, X2, p_marginal, levels=7, cmap='viridis', linewidths=1.6)
    ax2.set_title("(b) Non-Factorized Marginal Mixture\n$p(\\mathbf{x}) = \\sum_k p(\\mathbf{x} \\mid C_k) p(C_k)$",
                  fontsize=10.0, fontweight='bold', pad=10)
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(0, color='gray', linestyle=':', alpha=0.5)

    plt.suptitle("Figure 11.23: Naive Bayes Factorized Conditionals vs Non-Factorized Mixture",
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    _save_figure(fig, "Figure_11_23", save_dir)
    return fig


def generate_figure_11_24(save_dir: Optional[str] = None) -> plt.Figure:
    """Figure 11.24: Generative model of object images (class, position, scale -> image)."""
    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 3.5)
    ax.axis('off')
    ax.set_title("Figure 11.24: Generative Causal Process for Image Synthesis",
                 fontsize=11.5, fontweight='bold', pad=12)

    pos_c = (1.0, 2.5)
    pos_p = (3.0, 2.5)
    pos_s = (5.0, 2.5)
    pos_img = (3.0, 0.8)

    _draw_node(ax, pos_c, "class", radius=0.42, fontsize=10.0)
    _draw_node(ax, pos_p, "position", radius=0.45, fontsize=9.5)
    _draw_node(ax, pos_s, "scale", radius=0.42, fontsize=10.0)
    _draw_node(ax, pos_img, "image", radius=0.45, fontsize=10.0, is_observed=True)

    _draw_arrow(ax, pos_c, pos_img, radius=0.45)
    _draw_arrow(ax, pos_p, pos_img, radius=0.45)
    _draw_arrow(ax, pos_s, pos_img, radius=0.45)

    plt.tight_layout()
    _save_figure(fig, "Figure_11_24", save_dir)
    return fig


def generate_figure_11_25(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.25: The Markov blanket of a node x_i:
    parents, children, and co-parents (other parents of children).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')
    ax.set_title("Figure 11.25: The Markov Blanket of Node $x_i$",
                 fontsize=12, fontweight='bold', pad=12)

    pos_xi = (2.8, 2.2)

    # Parents of xi
    pos_p1 = (2.0, 3.7)
    pos_p2 = (3.6, 3.7)

    # Children of xi
    pos_c1 = (2.0, 0.8)
    pos_c2 = (3.6, 0.8)

    # Co-parents (other parents of c1 and c2)
    pos_cp1 = (0.6, 2.2)
    pos_cp2 = (5.0, 2.2)

    # Shaded boundary enclosing Markov blanket
    mb_rect = patches.FancyBboxPatch((0.1, 0.3), 5.4, 3.8, boxstyle="round,pad=0.2",
                                     facecolor='#e8f8f5', edgecolor='#16a085', lw=2.0, linestyle='--', zorder=1)
    ax.add_patch(mb_rect)
    ax.text(5.3, 4.1, "Markov Blanket $\\mathrm{MB}(x_i)$", ha='right', fontsize=9.5, fontweight='bold', color='#16a085')

    # Draw nodes
    _draw_node(ax, pos_xi, "$x_i$", facecolor='#f9e79f')  # Central node in yellow
    _draw_node(ax, pos_p1, "$p_1$")
    _draw_node(ax, pos_p2, "$p_2$")
    _draw_node(ax, pos_c1, "$c_1$")
    _draw_node(ax, pos_c2, "$c_2$")
    _draw_node(ax, pos_cp1, "$k_1$")
    _draw_node(ax, pos_cp2, "$k_2$")

    # Arrows
    _draw_arrow(ax, pos_p1, pos_xi)
    _draw_arrow(ax, pos_p2, pos_xi)
    _draw_arrow(ax, pos_xi, pos_c1)
    _draw_arrow(ax, pos_xi, pos_c2)
    _draw_arrow(ax, pos_cp1, pos_c1)
    _draw_arrow(ax, pos_cp2, pos_c2)

    plt.tight_layout()
    _save_figure(fig, "Figure_11_25", save_dir)
    return fig


def generate_figure_11_26(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 11.26: Graphs as filters.
    P: set of all probability distributions over variables.
    DF: distributions that factorize according to graph G.
    UI: unfaithful distributions (satisfy extra independencies not in graph).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 4.2)
    ax.axis('off')
    ax.set_title("Figure 11.26: Directed Graphical Model as a Distribution Filter",
                 fontsize=12, fontweight='bold', pad=12)

    # Outer set P (all distributions)
    rect_P = patches.Rectangle((0.2, 0.2), 5.8, 3.4, facecolor='#f4f6f7', edgecolor='#7f8c8d', lw=2.0)
    ax.add_patch(rect_P)
    ax.text(0.5, 3.3, "$\\mathcal{P}$ (All Distributions)", fontsize=11, fontweight='bold', color='#2c3e50')

    # Middle set DF (Distributions that factorize over G)
    ellipse_DF = patches.Ellipse((3.2, 1.8), 4.2, 2.2, facecolor='#d4e6f1', edgecolor='#2980b9', lw=2.0)
    ax.add_patch(ellipse_DF)
    ax.text(3.2, 2.5, "$\\mathcal{DF}$ (Distributions Factorizing over $G$)",
            ha='center', fontsize=10, fontweight='bold', color='#1b4f72')

    # Inner set UI (Unfaithful distributions possessing accidental independencies)
    ellipse_UI = patches.Ellipse((3.2, 1.4), 1.8, 0.9, facecolor='#fadbd8', edgecolor='#c0392b', lw=1.8)
    ax.add_patch(ellipse_UI)
    ax.text(3.2, 1.4, "$\\mathcal{UI}$\n(Unfaithful)", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#922b21')

    plt.tight_layout()
    _save_figure(fig, "Figure_11_26", save_dir)
    return fig


if __name__ == "__main__":
    for i in range(14, 27):
        func_name = f"generate_figure_11_{i}"
        func = globals()[func_name]
        fig = func()
        plt.close(fig)
    print("All 13 figures for Section 11.2 generated successfully!")
