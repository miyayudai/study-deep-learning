"""
common/machine_learning_on_graphs.py
====================================
Section 13.1: Machine Learning on Graphs
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements:
1. Graph Properties & Data Structures (Section 13.1.1):
   - Graph representation: nodes V, edges E, node feature matrix X, edge features.
   - Adjacency matrix A, Degree matrix D, Graph Laplacian L = D - A.
   - Symmetric normalized Laplacian L_sym = I - D^(-1/2) A D^(-1/2).
   - Random walk Laplacian L_rw = I - D^(-1) A.
   - Re-normalized adjacency with self-loops: A_tilde = A + I, D_tilde^(-1/2) A_tilde D_tilde^(-1/2).
2. Adjacency Matrix & Permutations (Section 13.1.2, Figure 13.2):
   - Construction of permutation matrices P (Eqs 13.1, 13.3).
   - Node feature permutation: X_tilde = P X (Eq 13.4).
   - Adjacency matrix permutation: A_tilde = P A P^T (Eq 13.5).
3. Permutation Equivariance & Invariance (Section 13.1.3, Eqs 13.6, 13.7):
   - Node-level permutation equivariance: f(P A P^T, P X) = P f(A, X).
   - Graph-level permutation invariance: f(P A P^T, P X) = f(A, X).
   - Verification with synthetic and molecular graphs.
4. Faithful High-Resolution Figure Reproductions (Figures 13.1 & 13.2):
   - Figure 13.1: Three examples of graph-structured data (caffeine molecule, rail network, web hyperlinks).
   - Figure 13.2: 5-node graph and its adjacency matrices under standard and permuted orderings.
   - Saved to 13/result/ and result/.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .plot_utils import setup_style, save_plot


def _save_figure(fig: plt.Figure, filename_base: str, save_dir: Optional[str] = None) -> None:
    """Save figure to Chapter 13 result directory and root result directory."""
    if save_dir is not None:
        save_plot(fig, os.path.join(save_dir, f"{filename_base}.png"))
        return

    repo_root = Path(__file__).resolve().parent.parent
    dir_ch13 = repo_root / "13" / "result"
    dir_root = repo_root / "result"
    dir_ch13.mkdir(parents=True, exist_ok=True)
    dir_root.mkdir(parents=True, exist_ok=True)

    save_plot(fig, str(dir_ch13 / f"{filename_base}.png"))
    save_plot(fig, str(dir_root / f"{filename_base}.png"))


# =============================================================================
# 1. Graph Data Structure & Spectral Matrices (Section 13.1.1, 13.1.2)
# =============================================================================

class Graph:
    """
    Representation of a Graph G = (V, E) with node features and adjacency matrices.
    
    Attributes:
        num_nodes (int): Number of nodes N = |V|.
        edges (List[Tuple[int, int]]): List of directed or undirected edge pairs.
        is_directed (bool): True if edges are directed, False for undirected.
        node_labels (List[str]): Names or labels for each node.
        node_features (np.ndarray): Matrix X in R^{N x D} of observed node variables.
        edge_features (Optional[Dict]): Features associated with edges.
    """

    def __init__(
        self,
        num_nodes: int,
        edges: List[Tuple[int, int]],
        node_features: Optional[np.ndarray] = None,
        edge_features: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
        is_directed: bool = False,
        node_labels: Optional[List[str]] = None,
    ):
        self.num_nodes = num_nodes
        self.edges = edges
        self.is_directed = is_directed
        self.node_labels = node_labels or [str(i) for i in range(num_nodes)]

        if node_features is not None:
            assert node_features.shape[0] == num_nodes, (
                f"Node feature matrix shape {node_features.shape} does not match num_nodes {num_nodes}"
            )
            self.node_features = node_features
        else:
            self.node_features = np.eye(num_nodes)

        self.edge_features = edge_features

        # Build adjacency matrix A
        self.A = self._build_adjacency_matrix()

    def _build_adjacency_matrix(self) -> np.ndarray:
        """Construct N x N adjacency matrix A (Section 13.1.2)."""
        A = np.zeros((self.num_nodes, self.num_nodes), dtype=float)
        for u, v in self.edges:
            A[u, v] = 1.0
            if not self.is_directed:
                A[v, u] = 1.0
        return A

    @property
    def degree_matrix(self) -> np.ndarray:
        """
        Degree matrix D in R^{N x N}, where D_ii = sum_j A_ij.
        """
        degrees = np.sum(self.A, axis=1)
        return np.diag(degrees)

    @property
    def degrees(self) -> np.ndarray:
        """1D array of node degrees."""
        return np.sum(self.A, axis=1)

    @property
    def laplacian(self) -> np.ndarray:
        """Unnormalized graph Laplacian matrix L = D - A."""
        return self.degree_matrix - self.A

    @property
    def normalized_laplacian(self) -> np.ndarray:
        """
        Symmetric normalized graph Laplacian:
        L_sym = D^(-1/2) L D^(-1/2) = I - D^(-1/2) A D^(-1/2).
        """
        d = self.degrees
        d_inv_sqrt = np.zeros_like(d)
        mask = (d > 0)
        d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
        D_inv_sqrt = np.diag(d_inv_sqrt)

        I = np.eye(self.num_nodes)
        A_norm = np.matmul(np.matmul(D_inv_sqrt, self.A), D_inv_sqrt)
        return I - A_norm

    @property
    def random_walk_laplacian(self) -> np.ndarray:
        """Random walk normalized Laplacian: L_rw = D^(-1) L = I - D^(-1) A."""
        d = self.degrees
        d_inv = np.zeros_like(d)
        mask = (d > 0)
        d_inv[mask] = 1.0 / d[mask]
        D_inv = np.diag(d_inv)
        I = np.eye(self.num_nodes)
        return I - np.matmul(D_inv, self.A)

    @property
    def renormalized_adjacency(self) -> np.ndarray:
        """
        Renormalized adjacency matrix with self-loops (Kipf & Welling, 2017):
        A_tilde = A + I_N
        D_tilde_ii = sum_j A_tilde_ij
        A_hat = D_tilde^(-1/2) A_tilde D_tilde^(-1/2)
        """
        A_tilde = self.A + np.eye(self.num_nodes)
        d_tilde = np.sum(A_tilde, axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(d_tilde)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        return np.matmul(np.matmul(D_inv_sqrt, A_tilde), D_inv_sqrt)

    def neighbors(self, node: int) -> List[int]:
        """Return list of neighbor node indices for node."""
        return [int(j) for j in range(self.num_nodes) if self.A[node, j] > 0]

    def permute(self, permutation: Union[List[int], np.ndarray]) -> "Graph":
        """
        Create a new permuted Graph under node reordering pi (Section 13.1.3):
        X_tilde = P X (Eq 13.4)
        A_tilde = P A P^T (Eq 13.5)
        
        Args:
            permutation: List of new indices or permutation mapping.
        """
        P = build_permutation_matrix(permutation)
        new_X = np.matmul(P, self.node_features)
        new_A = np.matmul(np.matmul(P, self.A), P.T)

        new_labels = [self.node_labels[permutation[i]] for i in range(self.num_nodes)]

        # Extract edge list from new adjacency matrix
        new_edges = []
        for i in range(self.num_nodes):
            start_j = 0 if self.is_directed else i + 1
            for j in range(start_j, self.num_nodes):
                if new_A[i, j] > 0:
                    new_edges.append((i, j))

        return Graph(
            num_nodes=self.num_nodes,
            edges=new_edges,
            node_features=new_X,
            is_directed=self.is_directed,
            node_labels=new_labels,
        )


# =============================================================================
# 2. Permutation Matrices & Equivariance / Invariance (Section 13.1.3)
# =============================================================================

def build_permutation_matrix(permutation: Union[List[int], np.ndarray]) -> np.ndarray:
    """
    Construct permutation matrix P in {0, 1}^{N x N} (Eqs 13.1, 13.3).
    
    If permutation is [pi(0), pi(1), ..., pi(N-1)], then:
    P[i, pi(i)] = 1, all other entries 0.
    Pre-multiplication P X maps row i of X to row pi(i).
    """
    N = len(permutation)
    P = np.zeros((N, N), dtype=float)
    for i, target in enumerate(permutation):
        P[i, target] = 1.0
    return P


def check_node_equivariance(
    layer_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    A: np.ndarray,
    X: np.ndarray,
    permutation: Union[List[int], np.ndarray],
) -> Tuple[bool, float]:
    """
    Verify node-level permutation equivariance (Eq 13.7):
        f(P A P^T, P X) = P f(A, X)
    """
    P = build_permutation_matrix(permutation)
    A_perm = np.matmul(np.matmul(P, A), P.T)
    X_perm = np.matmul(P, X)

    out_orig = layer_fn(A, X)
    out_perm = layer_fn(A_perm, X_perm)
    expected_out_perm = np.matmul(P, out_orig)

    max_diff = float(np.max(np.abs(out_perm - expected_out_perm)))
    is_equivariant = (max_diff < 1e-6)
    return is_equivariant, max_diff


def check_graph_invariance(
    graph_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    A: np.ndarray,
    X: np.ndarray,
    permutation: Union[List[int], np.ndarray],
) -> Tuple[bool, float]:
    """
    Verify graph-level permutation invariance (Eq 13.6):
        f(P A P^T, P X) = f(A, X)
    """
    P = build_permutation_matrix(permutation)
    A_perm = np.matmul(np.matmul(P, A), P.T)
    X_perm = np.matmul(P, X)

    out_orig = graph_fn(A, X)
    out_perm = graph_fn(A_perm, X_perm)

    max_diff = float(np.max(np.abs(out_perm - out_orig)))
    is_invariant = (max_diff < 1e-6)
    return is_invariant, max_diff


def simple_gcn_layer(
    A: np.ndarray,
    X: np.ndarray,
    W: Optional[np.ndarray] = None,
    activation: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    A single standard Graph Convolutional layer (Eqs 13.9, 13.16, 13.17):
        H^(l+1) = sigma( D_tilde^(-1/2) (A + I) D_tilde^(-1/2) X W )
    This layer is strictly permutation equivariant!
    """
    N, D = X.shape
    if W is None:
        rng = np.random.default_rng(seed)
        W = rng.normal(0, 1.0 / np.sqrt(D), size=(D, D))

    A_tilde = A + np.eye(N)
    d_tilde = np.sum(A_tilde, axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d_tilde)
    D_inv_sqrt = np.diag(d_inv_sqrt)

    norm_A = np.matmul(np.matmul(D_inv_sqrt, A_tilde), D_inv_sqrt)
    agg = np.matmul(norm_A, X)
    lin = np.matmul(agg, W)

    if activation is not None:
        return activation(lin)
    # Default: ReLU activation
    return np.maximum(0.0, lin)


def simple_graph_pooling(
    A: np.ndarray,
    X: np.ndarray,
    pool_type: str = "mean",
) -> np.ndarray:
    """
    Global graph pooling layer (sum, mean, max):
        g = Pool(H)
    This pooling operation is strictly permutation invariant!
    """
    if pool_type == "sum":
        return np.sum(X, axis=0)
    elif pool_type == "max":
        return np.max(X, axis=0)
    else:  # mean
        return np.mean(X, axis=0)


# =============================================================================
# 3. High-Resolution Figure Reproductions (Figures 13.1 & 13.2)
# =============================================================================

def generate_figure_13_1(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 13.1: Three examples of graph-structured data:
    (a) The caffeine molecule consisting of atoms connected by chemical bonds.
    (b) A rail network consisting of cities connected by railway lines.
    (c) The worldwide web consisting of pages connected by hyperlinks.
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=300)

    # -------------------------------------------------------------------------
    # Panel (a): Caffeine Molecule (C8 H10 N4 O2)
    # -------------------------------------------------------------------------
    ax = axes[0]
    ax.set_xlim(-2.5, 3.2)
    ax.set_ylim(-2.5, 2.8)
    ax.set_aspect("equal")
    ax.axis("off")

    # Skeletal structure coordinates
    # 6-membered ring: N1(-1, 0.6), C2(0, 1.2), N3(1, 0.6), C4(1, -0.6), C5(0, -1.2), C6(-1, -0.6)
    # 5-membered ring fused at C4, C5: N7(1.9, 0.2), C8(2.4, -0.5), N9(1.8, -1.1)
    
    # 6-membered ring vertices
    N1 = np.array([-0.9, 0.55])
    C2 = np.array([0.0, 1.1])
    N3 = np.array([0.9, 0.55])
    C4 = np.array([0.9, -0.55])
    C5 = np.array([0.0, -1.1])
    C6 = np.array([-0.9, -0.55])

    # 5-membered ring vertices
    N7 = np.array([1.95, 0.3])
    C8 = np.array([2.55, -0.4])
    N9 = np.array([1.95, -1.0])

    # Draw 6-ring bonds
    ax.plot([N1[0], C2[0]], [N1[1], C2[1]], color="#0F172A", lw=2.2)
    ax.plot([C2[0], N3[0]], [C2[1], N3[1]], color="#0F172A", lw=2.2)
    ax.plot([N3[0], C4[0]], [N3[1], C4[1]], color="#0F172A", lw=2.2)
    ax.plot([C4[0], C5[0]], [C4[1], C5[1]], color="#0F172A", lw=3.2)  # double bond fused
    ax.plot([C5[0], C6[0]], [C5[1], C6[1]], color="#0F172A", lw=2.2)
    ax.plot([C6[0], N1[0]], [C6[1], N1[1]], color="#0F172A", lw=2.2)

    # 5-ring bonds
    ax.plot([C4[0], N7[0]], [C4[1], N7[1]], color="#0F172A", lw=2.2)
    # double bond between N7 and C8
    ax.plot([N7[0], C8[0]], [N7[1], C8[1]], color="#0F172A", lw=2.8)
    ax.plot([C8[0], N9[0]], [C8[1], N9[1]], color="#0F172A", lw=2.2)
    ax.plot([N9[0], C5[0]], [N9[1], C5[1]], color="#0F172A", lw=2.2)

    # Carbonyl double bonds: =O on C2 and C6
    # O on C2 (top)
    ax.plot([-0.05, -0.05], [1.1, 1.8], color="#0F172A", lw=2)
    ax.plot([0.05, 0.05], [1.1, 1.8], color="#0F172A", lw=2)
    ax.text(0.0, 2.05, "O", ha="center", va="center", fontsize=15, fontweight="bold", color="#DC2626")

    # O on C6 (bottom left)
    O_c6_x, O_c6_y = -1.6, -1.0
    ax.plot([C6[0] - 0.05, O_c6_x - 0.05], [C6[1] - 0.05, O_c6_y - 0.05], color="#0F172A", lw=2)
    ax.plot([C6[0] + 0.05, O_c6_x + 0.05], [C6[1] + 0.05, O_c6_y + 0.05], color="#0F172A", lw=2)
    ax.text(O_c6_x - 0.2, O_c6_y - 0.1, "O", ha="center", va="center", fontsize=15, fontweight="bold", color="#DC2626")

    # Methyl branches on N atoms (-CH3 represented as bonds)
    # N1 methyl (top-left)
    ax.plot([N1[0], N1[0] - 0.8], [N1[1], N1[1] + 0.4], color="#0F172A", lw=2.2)
    # N3 methyl (top-right)
    ax.plot([N3[0], N3[0] + 0.3], [N3[1], N3[1] + 0.8], color="#0F172A", lw=2.2)
    # N9 methyl (bottom)
    ax.plot([N9[0] - 0.5, N9[0] - 0.5], [-1.0, -1.8], color="#0F172A", lw=2.2)

    # Nitrogen labels
    ax.text(N1[0], N1[1], "N", ha="center", va="center", fontsize=14, fontweight="bold", color="#2563EB",
            bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", edgecolor="none"))
    ax.text(N3[0], N3[1], "N", ha="center", va="center", fontsize=14, fontweight="bold", color="#2563EB",
            bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", edgecolor="none"))
    ax.text(N7[0], N7[1], "N", ha="center", va="center", fontsize=14, fontweight="bold", color="#2563EB",
            bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", edgecolor="none"))
    ax.text(N9[0] - 0.1, N9[1] + 0.05, "N", ha="center", va="center", fontsize=14, fontweight="bold", color="#2563EB",
            bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", edgecolor="none"))

    ax.text(0.5, -2.3, "(a)", ha="center", fontsize=12, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel (b): Rail Network
    # -------------------------------------------------------------------------
    ax = axes[1]
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(-0.8, 5.2)
    ax.set_aspect("equal")
    ax.axis("off")

    cities = {
        "Leeds": np.array([2.5, 4.4]),
        "Cambridge": np.array([3.6, 2.7]),
        "London": np.array([3.3, 0.4]),
        "Oxford": np.array([2.0, 1.8]),
        "Bristol": np.array([0.7, 0.4]),
    }

    rail_edges = [
        ("Leeds", "Cambridge"),
        ("Leeds", "Oxford"),
        ("Cambridge", "London"),
        ("Cambridge", "Oxford"),
        ("Oxford", "London"),
        ("Bristol", "Oxford"),
        ("Bristol", "London"),
    ]

    # Draw railway track lines
    for c1, c2 in rail_edges:
        p1, p2 = cities[c1], cities[c2]
        if (c1 == "Bristol" and c2 == "London"):
            # Curved southern arc
            ax.annotate("", xy=p2, xytext=p1,
                        arrowprops=dict(arrowstyle="-", lw=2.4, color="#059669",
                                        connectionstyle="arc3,rad=-0.25"))
        elif (c1 == "Leeds" and c2 == "Cambridge"):
            ax.annotate("", xy=p2, xytext=p1,
                        arrowprops=dict(arrowstyle="-", lw=2.4, color="#059669",
                                        connectionstyle="arc3,rad=0.15"))
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#059669", lw=2.4)

    # Draw city station nodes
    for name, pos in cities.items():
        circle = patches.Circle((pos[0], pos[1]), 0.18, facecolor="#DBEAFE",
                                edgecolor="#1E40AF", lw=2.2, zorder=5)
        ax.add_patch(circle)

        # Label offsets
        if name == "Leeds":
            ax.text(pos[0], pos[1] + 0.35, name, ha="center", fontsize=9.5, fontweight="bold")
        elif name == "Cambridge":
            ax.text(pos[0] + 0.35, pos[1] + 0.15, name, ha="left", fontsize=9.5, fontweight="bold")
        elif name == "London":
            ax.text(pos[0], pos[1] - 0.35, name, ha="center", fontsize=9.5, fontweight="bold")
        elif name == "Oxford":
            ax.text(pos[0], pos[1] - 0.35, name, ha="center", fontsize=9.5, fontweight="bold")
        elif name == "Bristol":
            ax.text(pos[0] - 0.15, pos[1] - 0.35, name, ha="center", fontsize=9.5, fontweight="bold")

    ax.text(2.3, -0.7, "(b)", ha="center", fontsize=12, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel (c): The Worldwide Web (Hyperlinked Web Documents)
    # -------------------------------------------------------------------------
    ax = axes[2]
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.8, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Document rectangles
    docs = [
        {"x": 0.0, "y": 2.2, "w": 0.8, "h": 1.1},   # Top left doc
        {"x": 3.0, "y": 2.6, "w": 0.8, "h": 1.1},   # Top right doc
        {"x": 1.0, "y": 0.2, "w": 0.8, "h": 1.1},   # Bottom left doc
        {"x": 2.8, "y": 0.6, "w": 0.8, "h": 1.1},   # Bottom right doc
    ]

    for d in docs:
        rect = patches.Rectangle((d["x"], d["y"]), d["w"], d["h"],
                                 facecolor="#FFFFFF", edgecolor="#0F172A", lw=1.8, zorder=2)
        ax.add_patch(rect)
        # Add horizontal simulated text lines
        for l_idx in range(5):
            line_y = d["y"] + d["h"] - 0.2 - l_idx * 0.18
            line_w = 0.55 if l_idx % 2 == 0 else 0.4
            color = "#3B82F6" if l_idx == 1 else "#94A3B8"
            ax.plot([d["x"] + 0.1, d["x"] + 0.1 + line_w], [line_y, line_y], color=color, lw=1.5, zorder=3)

    # Red directed hyperlinks
    links = [
        # Doc 0 -> Doc 1
        {"start": (0.8, 3.1), "end": (2.95, 3.5), "rad": -0.2},
        # Doc 1 -> Doc 0
        {"start": (2.95, 3.2), "end": (0.85, 2.6), "rad": -0.15},
        # Doc 1 -> Doc 3
        {"start": (3.2, 2.55), "end": (3.4, 1.75), "rad": 0.1},
        # Doc 2 -> Doc 0
        {"start": (1.1, 1.35), "end": (0.6, 2.15), "rad": -0.2},
        # Doc 2 -> Doc 3
        {"start": (1.85, 0.7), "end": (2.75, 0.9), "rad": -0.1},
        # Doc 3 -> Doc 2
        {"start": (2.75, 0.6), "end": (1.85, 0.4), "rad": -0.15},
    ]

    for lk in links:
        ax.annotate("", xy=lk["end"], xytext=lk["start"],
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#DC2626",
                                    connectionstyle=f"arc3,rad={lk['rad']}"))

    ax.text(2.0, -0.7, "(c)", ha="center", fontsize=12, fontweight="bold")

    fig.suptitle("Figure 13.1: Three Examples of Graph-Structured Data",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    _save_figure(fig, "fig_13_1_graph_data_examples", save_dir)
    return fig


def generate_figure_13_2(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 13.2: An example of an adjacency matrix:
    (a) An example of a graph with five nodes A, B, C, D, E.
    (b) The associated adjacency matrix for node order (A, B, C, D, E).
    (c) The adjacency matrix corresponding to node order (C, E, A, D, B).
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), dpi=300)

    # -------------------------------------------------------------------------
    # Panel (a): 5-Node Graph (A, B, C, D, E)
    # -------------------------------------------------------------------------
    ax = axes[0]
    ax.set_xlim(-0.5, 4.2)
    ax.set_ylim(-0.5, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")

    node_pos = {
        "A": np.array([1.2, 3.2]),
        "B": np.array([0.4, 1.4]),
        "C": np.array([1.8, 1.7]),
        "D": np.array([2.9, 3.2]),
        "E": np.array([3.1, 0.8]),
    }

    graph_edges = [
        ("A", "C"),
        ("B", "C"),
        ("C", "D"),
        ("C", "E"),
        ("D", "E"),
    ]

    for n1, n2 in graph_edges:
        p1, p2 = node_pos[n1], node_pos[n2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#0F172A", lw=2.2, zorder=1)

    for name, pos in node_pos.items():
        circle = patches.Circle((pos[0], pos[1]), 0.32, facecolor="#DBEAFE",
                                edgecolor="#1E40AF", lw=2.2, zorder=5)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], name, ha="center", va="center",
                fontsize=13, fontweight="bold", color="#1E3A8A", zorder=6)

    ax.text(1.8, -0.4, "(a)", ha="center", fontsize=12, fontweight="bold")

    # -------------------------------------------------------------------------
    # Helper to draw colored adjacency grid
    # -------------------------------------------------------------------------
    def draw_adjacency_grid(target_ax, A_mat, labels, panel_title):
        target_ax.set_xlim(-1.2, 5.5)
        target_ax.set_ylim(-1.2, 6.2)
        target_ax.set_aspect("equal")
        target_ax.axis("off")

        # Color: Blue for 1, Cream/Yellow-white for 0
        c_one = "#2563EB"
        c_zero = "#FEF9C3"

        # Column labels at top
        for col_idx, lbl in enumerate(labels):
            target_ax.text(col_idx + 0.5, 5.4, lbl, ha="center", va="center",
                           fontsize=12, style="italic", fontweight="bold", color="#0F172A")

        # Row labels at left
        for row_idx, lbl in enumerate(labels):
            target_ax.text(-0.4, 4.5 - row_idx, lbl, ha="center", va="center",
                           fontsize=12, style="italic", fontweight="bold", color="#0F172A")

        # Draw 5x5 grid cells
        for r in range(5):
            for c in range(5):
                val = A_mat[r, c]
                color = c_one if val > 0 else c_zero
                rect = patches.Rectangle((c, 4 - r), 1.0, 1.0, facecolor=color,
                                         edgecolor="#0F172A", lw=1.4)
                target_ax.add_patch(rect)

        target_ax.text(2.5, -0.4, panel_title, ha="center", fontsize=12, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel (b): Standard Adjacency Matrix (A, B, C, D, E)
    # -------------------------------------------------------------------------
    # Order: A, B, C, D, E
    labels_b = ["A", "B", "C", "D", "E"]
    A_b = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0],
    ])
    draw_adjacency_grid(axes[1], A_b, labels_b, "(b)")

    # -------------------------------------------------------------------------
    # Panel (c): Permuted Adjacency Matrix (C, E, A, D, B)
    # -------------------------------------------------------------------------
    # Permutation: (A, B, C, D, E) -> (C, E, A, D, B)
    # Mapping: row 0 is C (idx 2), row 1 is E (idx 4), row 2 is A (idx 0), row 3 is D (idx 3), row 4 is B (idx 1)
    perm_c = [2, 4, 0, 3, 1]
    labels_c = ["C", "E", "A", "D", "B"]
    P_c = build_permutation_matrix(perm_c)
    A_c = np.matmul(np.matmul(P_c, A_b), P_c.T)
    draw_adjacency_grid(axes[2], A_c, labels_c, "(c)")

    fig.suptitle("Figure 13.2: 5-Node Graph and Adjacency Matrices under Different Node Orderings",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    _save_figure(fig, "fig_13_2_adjacency_matrix", save_dir)
    return fig


if __name__ == "__main__":
    print("Testing Graph class...")
    edges = [(0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]
    g = Graph(num_nodes=5, edges=edges, node_labels=["A", "B", "C", "D", "E"])
    print("Adjacency matrix A:\\n", g.A)
    print("Degree matrix D:\\n", g.degree_matrix)
    print("Laplacian matrix L:\\n", g.laplacian)

    print("Verifying node-level equivariance (Eq 13.7)...")
    perm = [2, 4, 0, 3, 1]
    X_dummy = np.random.default_rng(42).normal(size=(5, 4))
    is_equiv, diff = check_node_equivariance(simple_gcn_layer, g.A, X_dummy, perm)
    print(f"Node equivariance satisfied: {is_equiv} (max diff = {diff})")

    print("Verifying graph-level invariance (Eq 13.6)...")
    is_inv, diff_inv = check_graph_invariance(simple_graph_pooling, g.A, X_dummy, perm)
    print(f"Graph invariance satisfied: {is_inv} (max diff = {diff_inv})")

    print("Generating Figure 13.1...")
    generate_figure_13_1()
    print("Generating Figure 13.2...")
    generate_figure_13_2()
    print("All Chapter 13.1 modules and figures generated successfully!")
