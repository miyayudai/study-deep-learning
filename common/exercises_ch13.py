"""
common/exercises_ch13.py
========================
Chapter 13 Exercises: Graph Neural Networks
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module provides theoretical proofs, numerical verifications, and figure generators
for Exercises 13.1 through 13.10:
- Exercise 13.1: Permutation matrix P transforming Figure 13.2 adjacency matrix.
- Exercise 13.2: Diagonal elements of A^2 equal node degrees: (A^2)_{nn} = sum_m A_{nm}^2 = d_n.
- Exercise 13.3: Reconstruction, edge extraction, and visualization of graph with adjacency matrix (13.42).
- Exercise 13.4: Pre-multiplication X_tilde = P X permutes rows according to pi(·).
- Exercise 13.5: Bilinear transformation A_tilde = P A P^T permutes both rows and columns.
- Exercise 13.6: Matrix formulation of linear message passing: Z = A H, argument A H W_neigh + H W_self + 1 b^T.
- Exercise 13.7: Mathematical induction proving full deep GCN equivariance under node permutations.
- Exercise 13.8: Equivariance of GAT attention aggregation under node permutation.
- Exercise 13.9: Equivalence between fully-connected GAT and standard Transformer encoder.
- Exercise 13.10: Invariance of scalar messages and equivariance of coordinate updates in EGNN under E(3).
"""

from typing import Any, Dict, List, Optional, Tuple
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .plot_utils import setup_style, save_plot
from .machine_learning_on_graphs import Graph, build_permutation_matrix
from .neural_message_passing import relu, MessagePassingLayer, GraphConvolutionalNetwork
from .general_graph_networks import GATLayer, EGNNLayer, check_egnn_equivariance


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
# Exercise 13.1: Permutation Matrix for Figure 13.2
# =============================================================================

def solve_exercise_13_1() -> Dict[str, Any]:
    """
    Exercise 13.1:
    Show that the permutation (A, B, C, D, E) -> (C, E, A, D, B)
    can be expressed in the form A_tilde = P A P^T with permutation matrix P.
    """
    # Alphabetical order: A:0, B:1, C:2, D:3, E:4
    # Edges from Figure 13.2:
    # A-B, A-C, A-D, B-C, B-D, C-E, D-E
    edges_alpha = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (3, 4)]
    A_orig = np.zeros((5, 5), dtype=float)
    for u, v in edges_alpha:
        A_orig[u, v] = 1.0
        A_orig[v, u] = 1.0

    # Permuted ordering: (C, E, A, D, B) -> indices [2, 4, 0, 3, 1]
    # Row i of P has 1 at old index pi(i)
    pi = [2, 4, 0, 3, 1]
    P = build_permutation_matrix(pi)

    # Transformed adjacency matrix
    A_perm = P @ A_orig @ P.T

    # Expected A_perm matching Figure 13.2(c):
    # Rows/cols: C(0), E(1), A(2), D(3), B(4)
    # C is connected to: A(2), B(4), E(1) -> cols 1, 2, 4
    # E is connected to: C(0), D(3) -> cols 0, 3
    # A is connected to: B(4), C(0), D(3) -> cols 0, 3, 4
    # D is connected to: A(2), B(4), E(1) -> cols 1, 2, 4
    # B is connected to: A(2), C(0), D(3) -> cols 0, 2, 3
    expected_A_perm = np.array([
        [0, 1, 1, 0, 1],  # C
        [1, 0, 0, 1, 0],  # E
        [1, 0, 0, 1, 1],  # A
        [0, 1, 1, 0, 1],  # D
        [1, 0, 1, 1, 0],  # B
    ], dtype=float)

    is_match = np.allclose(A_perm, expected_A_perm)

    return {
        "P": P,
        "A_orig": A_orig,
        "A_perm": A_perm,
        "expected_A_perm": expected_A_perm,
        "is_correct": is_match,
        "proof_summary": (
            "Permutation matrix P has P_{ij} = 1 if the i-th node in the new ordering "
            "corresponds to the j-th node in the original ordering. Pre-multiplication by P "
            "permutes rows: (PA)_{in} = A_{pi(i), n}. Post-multiplication by P^T permutes columns: "
            "(PAP^T)_{ij} = A_{pi(i), pi(j)}. Evaluating with pi = [C, E, A, D, B] yields exactly "
            "the matrix in Figure 13.2(c)."
        ),
    }


# =============================================================================
# Exercise 13.2: Diagonal Elements of A^2 and Node Degrees
# =============================================================================

def solve_exercise_13_2(A: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Exercise 13.2:
    Show that the number of edges connected to each node of a graph is given by
    the corresponding diagonal element of the matrix A^2 where A is the adjacency matrix.
    """
    if A is None:
        # 5-node graph from Figure 13.2
        res1 = solve_exercise_13_1()
        A = res1["A_orig"]

    A2 = A @ A
    diag_A2 = np.diag(A2)
    degrees = np.sum(A, axis=1)

    is_equal = np.allclose(diag_A2, degrees)

    return {
        "A": A,
        "A2": A2,
        "diag_A2": diag_A2,
        "degrees": degrees,
        "is_equal": is_equal,
        "proof_summary": (
            "For an unweighted undirected graph with adjacency matrix A, A_{nm} in {0, 1} and A = A^T. "
            "The n-th diagonal element of A^2 is given by (A^2)_{nn} = sum_{m=1}^N A_{nm} A_{mn}. "
            "Since A_{mn} = A_{nm} and A_{nm} in {0, 1}, we have A_{nm} A_{mn} = A_{nm}^2 = A_{nm}. "
            "Therefore, (A^2)_{nn} = sum_{m=1}^N A_{nm} = d_n, which is precisely the degree of node n."
        ),
    }


# =============================================================================
# Exercise 13.3: Graph from Given Adjacency Matrix (13.42)
# =============================================================================

def solve_exercise_13_3(save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Exercise 13.3:
    Draw the graph whose adjacency matrix is given by:
        A = [
            [0, 1, 1, 0, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0]
        ]
    """
    A = np.array([
        [0, 1, 1, 0, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
    ], dtype=float)

    N = A.shape[0]
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > 0:
                edges.append((i, j))

    # Degrees: Node 0: 3, Node 1: 4 (hub), Node 2: 3, Node 3: 2, Node 4: 2
    degrees = np.sum(A, axis=1)

    # Visualization
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=300)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.axis("off")

    # Layout: arrange 5 nodes in a pentagon
    node_names = ["1", "2", "3", "4", "5"]
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, N, endpoint=False)
    coords = {i: (1.7 * np.cos(th), 1.7 * np.sin(th)) for i, th in enumerate(angles)}

    # Draw edges
    for u, v in edges:
        p1, p2 = coords[u], coords[v]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#263238", linewidth=2.0, zorder=2)

    # Draw nodes
    for i in range(N):
        pos = coords[i]
        c = patches.Circle(pos, 0.32, facecolor="#e8eaf6", edgecolor="#1a237e", linewidth=2.2, zorder=3)
        ax.add_patch(c)
        ax.text(pos[0], pos[1], node_names[i], fontsize=14, fontweight="bold", ha="center", va="center", zorder=4)

    ax.set_title("Exercise 13.3: Graph with Adjacency Matrix (13.42)", fontsize=13, pad=15)
    plt.tight_layout()

    _save_figure(fig, "fig_13_ex3_graph", save_dir)

    return {
        "A": A,
        "edges": edges,
        "degrees": degrees,
        "figure": fig,
        "is_correct": len(edges) == 7,
    }


# =============================================================================
# Exercise 13.4 & 13.5: Row & Column Permutations
# =============================================================================

def solve_exercise_13_4() -> Dict[str, Any]:
    """
    Exercise 13.4:
    Show that pre-multiplying X by permutation matrix P creates X_tilde = P X
    whose rows are permuted according to pi(·).
    """
    N = 4
    D = 3
    rng = np.random.RandomState(42)
    X = rng.randn(N, D)
    pi = [2, 0, 3, 1]
    P = build_permutation_matrix(pi)

    X_tilde = P @ X
    expected_X_tilde = X[pi]

    is_equal = np.allclose(X_tilde, expected_X_tilde)

    return {
        "P": P,
        "X": X,
        "X_tilde": X_tilde,
        "is_equal": is_equal,
        "proof_summary": (
            "By definition of the permutation matrix, P_{ij} = delta_{j, pi(i)}. "
            "The (i, k)-th entry of X_tilde = P X is (P X)_{ik} = sum_{j=1}^N P_{ij} X_{jk} "
            "= sum_{j=1}^N delta_{j, pi(i)} X_{jk} = X_{pi(i), k}. "
            "Thus, the i-th row of X_tilde is the pi(i)-th row of X, proving row permutation."
        ),
    }


def solve_exercise_13_5() -> Dict[str, Any]:
    """
    Exercise 13.5:
    Show that transformed adjacency matrix A_tilde = P A P^T permutes both rows
    and columns according to pi(·).
    """
    N = 4
    rng = np.random.RandomState(42)
    A = rng.randint(0, 2, (N, N)).astype(float)
    A = np.triu(A, 1) + np.triu(A, 1).T  # symmetric, zero diagonal
    pi = [3, 1, 0, 2]
    P = build_permutation_matrix(pi)

    A_tilde = P @ A @ P.T
    expected_A_tilde = A[np.ix_(pi, pi)]

    is_equal = np.allclose(A_tilde, expected_A_tilde)

    return {
        "P": P,
        "A": A,
        "A_tilde": A_tilde,
        "is_equal": is_equal,
        "proof_summary": (
            "Using P_{ik} = delta_{k, pi(i)} and (P^T)_{lj} = P_{jl} = delta_{l, pi(j)}, "
            "the (i, j)-th entry of A_tilde is (P A P^T)_{ij} = sum_{k, l} P_{ik} A_{kl} (P^T)_{lj} "
            "= sum_{k, l} delta_{k, pi(i)} A_{kl} delta_{l, pi(j)} = A_{pi(i), pi(j)}. "
            "Hence, the (i, j)-th element of A_tilde equals the (pi(i), pi(j))-th element of A, "
            "confirming that both rows and columns are permuted by pi(·)."
        ),
    }


# =============================================================================
# Exercise 13.6: Matrix Formulation of Message Passing
# =============================================================================

def solve_exercise_13_6() -> Dict[str, Any]:
    """
    Exercise 13.6:
    Show that z_n = sum_{m in N(n)} h_m can be written in matrix form as Z = A H,
    and argument to activation function in (13.16) is A H W_neigh + H W_self + 1_N b^T.
    """
    N = 4
    D_in = 3
    D_out = 2
    rng = np.random.RandomState(42)

    A = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)

    H = rng.randn(N, D_in)
    W_neigh = rng.randn(D_in, D_out)
    W_self = rng.randn(D_in, D_out)
    b = rng.randn(D_out)

    # Node-by-node computation (Eqs 13.10, 13.16)
    Z_node = np.zeros_like(H)
    Arg_node = np.zeros((N, D_out))
    for n in range(N):
        neighs = np.where(A[n] > 0)[0]
        z_n = np.sum(H[neighs], axis=0) if len(neighs) > 0 else np.zeros(D_in)
        Z_node[n] = z_n
        Arg_node[n] = z_n @ W_neigh + H[n] @ W_self + b

    # Matrix formulation
    Z_mat = A @ H
    Arg_mat = A @ H @ W_neigh + H @ W_self + np.ones((N, 1)) @ b.reshape(1, -1)

    is_Z_equal = np.allclose(Z_node, Z_mat)
    is_Arg_equal = np.allclose(Arg_node, Arg_mat)

    return {
        "is_Z_equal": is_Z_equal,
        "is_Arg_equal": is_Arg_equal,
        "proof_summary": (
            "Since row n of H is h_n^T, the n-th row of A H is (A H)_{n, :} = sum_{m=1}^N A_{nm} h_m^T "
            "= sum_{m in N(n)} h_m^T = z_n^T, so Z = A H. "
            "Substituting Z into row n gives z_n^T W_neigh^T + h_n^T W_self^T + b^T. "
            "In full matrix form, this yields A H W_neigh + H W_self + 1_N b^T."
        ),
    }


# =============================================================================
# Exercise 13.7: Proof of Deep GCN Equivariance
# =============================================================================

def solve_exercise_13_7() -> Dict[str, Any]:
    """
    Exercise 13.7:
    Show that a complete deep graph convolutional network defined by (13.18)
    is equivariant under node permutations.
    """
    # Numerical validation on a 3-layer GCN
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    N = 4
    rng = np.random.RandomState(42)
    X = rng.randn(N, 5)
    A = np.zeros((N, N))
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0

    gcn = GraphConvolutionalNetwork(layer_dims=[5, 8, 6, 3], seed=42)

    # Original forward
    H_final_orig = gcn.forward(A, X)

    # Permuted forward
    pi = [3, 0, 2, 1]
    P = build_permutation_matrix(pi)
    A_perm = P @ A @ P.T
    X_perm = P @ X

    H_final_perm = gcn.forward(A_perm, X_perm)
    is_equiv = np.allclose(H_final_perm, P @ H_final_orig, atol=1e-5)

    return {
        "is_equivariant": is_equiv,
        "proof_summary": (
            "We prove by induction on layer l in {1, ..., L}. "
            "Base step (l = 1): P H^{(1)} = F(P X, P A P^T, W^{(1)}) by the single-layer equivariance "
            "property (13.19) and node variable permutation X_tilde = P X (13.4). "
            "Inductive step: Assume P H^{(l)} = F_l(P H^{(l-1)}, P A P^T, W^{(l)}) holds for layer l. "
            "Then for layer l + 1: H_tilde^{(l+1)} = F_{l+1}(H_tilde^{(l)}, A_tilde, W^{(l+1)}) "
            "= F_{l+1}(P H^{(l)}, P A P^T, W^{(l+1)}) = P F_{l+1}(H^{(l)}, A, W^{(l+1)}) = P H^{(l+1)}. "
            "By induction, P H^{(L)} = GCN(P A P^T, P X), proving complete network equivariance."
        ),
    }


# =============================================================================
# Exercise 13.8: GAT Attention Equivariance
# =============================================================================

def solve_exercise_13_8() -> Dict[str, Any]:
    """
    Exercise 13.8:
    Explain why the aggregation function (13.24) with MLP attention weights (13.28)
    is equivariant under node reordering.
    """
    return {
        "proof_summary": (
            "The attention coefficients A_{nm} = softmax_{m in N(n)}(MLP(h_n, h_m)) depend only on "
            "the embedding vectors of the pair (h_n, h_m) and not on their integer indices or global ordering. "
            "When nodes are reordered via permutation P, the neighborhood of the permuted node pi(n) "
            "is precisely {pi(m) : m in N(n)}, and their embeddings are h_{pi(m)}. "
            "Since the shared MLP evaluates the same function on (h_{pi(n)}, h_{pi(m)}), "
            "the attention weight A_{pi(n), pi(m)} equals A_{nm}. "
            "The aggregated message z_{pi(n)} = sum_{m in N(n)} A_{nm} h_{pi(m)} is simply the message "
            "of the original node n reassigned to position pi(n). Hence Z_tilde = P Z, exhibiting equivariance."
        ),
    }


# =============================================================================
# Exercise 13.9: GAT vs Standard Transformer Equivalence
# =============================================================================

def solve_exercise_13_9() -> Dict[str, Any]:
    """
    Exercise 13.9:
    Show that a graph attention network on a fully connected graph
    is equivalent to a standard Transformer encoder architecture.
    """
    return {
        "proof_summary": (
            "In a fully connected graph with self-loops, the neighborhood of every node n is the entire graph: "
            "N(n) = V = {1, ..., N}. "
            "The GAT aggregation equation z_n = sum_{m in V} A_{nm} (W_V h_m) becomes an unconstrained sum "
            "over all tokens in the sequence. "
            "When attention weights A_{nm} are constructed via dot-product / bilinear form "
            "A_{nm} = softmax_m(h_n^T W_Q^T W_K h_m / sqrt(d)), this is identically "
            "the Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V. "
            "Combining multiple attention heads via concatenation and projection yields Multi-Head Attention (MHA). "
            "Together with the feedforward update layer, this exactly recovers the standard Transformer encoder."
        ),
    }


# =============================================================================
# Exercise 13.10: E(3) Symmetries in EGNN
# =============================================================================

def solve_exercise_13_10() -> Dict[str, Any]:
    """
    Exercise 13.10:
    Show that under translations, rotations, and reflections in R^3,
    the messages in (13.38), (13.40), (13.41) are invariant,
    and coordinate embeddings in (13.39) are equivariant.
    """
    # Test on EGNNLayer
    edges = [(0, 1), (1, 2), (2, 0)]
    graph = Graph(num_nodes=3, edges=edges, is_directed=False)
    rng = np.random.RandomState(42)
    H = rng.randn(3, 4)
    R = rng.randn(3, 3)

    egnn = EGNNLayer(node_dim=4, edge_dim=2, coord_scale=0.1, seed=42)
    report = check_egnn_equivariance(egnn, graph, H, R)
    all_passed = all(report.values())

    return {
        "report": report,
        "is_correct": all_passed,
        "proof_summary": (
            "1. Distance Invariance: Under transformation r~ = R r + c with orthogonal R (R^T R = I), "
            "||r~_n - r~_m||^2 = ||R (r_n - r_m)||^2 = (r_n - r_m)^T R^T R (r_n - r_m) = ||r_n - r_m||^2. "
            "Since the edge update (13.38) depends on coordinates only through ||r_n - r_m||^2, "
            "the edge messages e_{nm} are strictly invariant. "
            "2. Message and Feature Invariance: Since e_{nm} is invariant, the aggregated message z_n (13.40) "
            "and node update h_n (13.41) depend only on invariant quantities, hence h_n and z_n are invariant. "
            "3. Coordinate Equivariance: In coordinate update (13.39), r~_n - r~_m = R (r_n - r_m). "
            "Thus r~_n^{(l+1)} = (R r_n^{(l)} + c) + C sum R (r_n - r_m) phi(e_{nm}) "
            "= R [ r_n^{(l)} + C sum (r_n - r_m) phi(e_{nm}) ] + c = R r_n^{(l+1)} + c, "
            "which rigorously establishes coordinate equivariance under all translations, rotations, and reflections."
        ),
    }
