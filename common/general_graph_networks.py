"""
common/general_graph_networks.py
================================
Section 13.3: General Graph Networks
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements:
1. Graph Attention Networks (GAT, Section 13.3.1):
   - Attention-weighted aggregation z_n^{(l)} = sum_{m in N(n)} A_{nm} h_m^{(l)} (Eqs 13.24-13.26).
   - Bilinear attention coefficients A_{nm} = softmax(h_n^T W h_m) (Eq 13.27).
   - MLP attention coefficients A_{nm} = softmax(MLP(h_n, h_m)) (Eq 13.28).
   - Multi-Head Graph Attention (MultiHeadGAT) with head concatenation and projection.
2. Edge Embeddings (Section 13.3.2):
   - Edge update e_{nm}^{(l+1)} = Update_edge(e_{nm}^{(l)}, h_n^{(l)}, h_m^{(l)}) (Eq 13.29).
   - Node aggregation from updated edge representations (Eq 13.30).
   - Node update h_n^{(l+1)} = Update_node(h_n^{(l)}, z_n^{(l+1)}) (Eq 13.31).
3. Graph Embeddings & General MPNN (Algorithm 13.2, Figure 13.5, Section 13.3.3):
   - General message-passing with node, edge, and graph embeddings (Battaglia et al., 2018):
     * Edge update (Eq 13.32)
     * Node aggregation (Eq 13.33)
     * Node update (Eq 13.34)
     * Graph update g^{(l+1)} = Update_graph(g^{(l)}, {h_n^{(l+1)}}, {e_{nm}^{(l+1)}}) (Eq 13.35).
4. Over-smoothing Analysis & Mitigations (Section 13.3.4):
   - Dirichlet energy and mean pairwise node distance metrics.
   - Residual connections h_n^{(l+1)} = Update(...) + h_n^{(l)} (Eq 13.36).
   - Jumping Knowledge (JK-Net) multi-layer concatenation/pooling y_n = f(h_n^{(1)} ++ ... ++ h_n^{(L)}) (Eq 13.37).
5. GNN Regularization (Section 13.3.5):
   - Node dropout (random node masking).
   - DropEdge (random edge masking during training, Rong et al. 2020).
   - Parameter sharing across layers.
6. Geometric Deep Learning (EGNN / E(n)-Equivariance, Satorras et al. 2021, Section 13.3.6):
   - Spatial coordinate embeddings r_n^{(l)} in R^3.
   - Coordinate-aware edge update using squared distance ||r_n - r_m||^2 (Eq 13.38).
   - Equivariant coordinate update r_n^{(l+1)} = r_n^{(l)} + C sum (r_n - r_m) phi(e_{nm}) (Eq 13.39).
   - Node aggregation and update (Eqs 13.40, 13.41).
   - Exact numerical verification of E(3) translation, rotation, and reflection equivariance/invariance.
7. Publication-Quality Figure Reproduction:
   - Figure 13.5: General graph message-passing updates (edge, node, and global graph updates).
   - Saved to 13/result/ and result/.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .plot_utils import setup_style, save_plot
from .machine_learning_on_graphs import Graph, build_permutation_matrix
from .neural_message_passing import relu, d_relu, sigmoid, softmax, MLP


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
# 1. Graph Attention Networks (GAT, Section 13.3.1, Eqs 13.24-13.28)
# =============================================================================

def compute_attention_bilinear(
    h_n: np.ndarray,
    neighbor_embeddings: np.ndarray,
    W_attn: np.ndarray,
) -> np.ndarray:
    """
    Bilinear attention coefficients (Eq 13.27):
        A_{nm} = exp(h_n^T W h_m) / sum_{m'} exp(h_n^T W h_{m'})
    """
    if len(neighbor_embeddings) == 0:
        return np.array([])
    # Compute logits: (h_n @ W) @ h_m
    proj_n = np.dot(h_n, W_attn)
    logits = np.dot(neighbor_embeddings, proj_n)
    return softmax(logits)


def compute_attention_mlp(
    h_n: np.ndarray,
    neighbor_embeddings: np.ndarray,
    mlp_attn: MLP,
) -> np.ndarray:
    """
    MLP-based attention coefficients (Eq 13.28):
        A_{nm} = exp(MLP(h_n, h_m)) / sum_{m'} exp(MLP(h_n, h_{m'}))
    where MLP input is symmetric or concatenated [h_n; h_m].
    """
    if len(neighbor_embeddings) == 0:
        return np.array([])
    logits = []
    for h_m in neighbor_embeddings:
        # Concatenate node pairs
        pair_vec = np.concatenate([h_n, h_m], axis=-1)
        score = mlp_attn.forward(pair_vec)
        logits.append(score[0] if isinstance(score, np.ndarray) else score)
    return softmax(np.array(logits))


class GATLayer:
    """
    Single-Head Graph Attention Layer (Velickovic et al., 2017, Section 13.3.1).
        z_n = sum_{m in N(n)} A_{nm} * (h_m @ W_val)   (Eq 13.24)
        h_n^{(l+1)} = activation(z_n + b)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        attn_type: str = "bilinear",
        activation: Callable[[np.ndarray], np.ndarray] = relu,
        seed: Optional[int] = None,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_type = attn_type
        self.activation = activation

        rng = np.random.RandomState(seed)
        limit = np.sqrt(6.0 / (in_dim + out_dim))

        self.W_val = rng.uniform(-limit, limit, (in_dim, out_dim))
        self.W_attn = rng.uniform(-limit, limit, (in_dim, in_dim))
        self.bias = np.zeros(out_dim)

        if attn_type == "mlp":
            self.mlp_attn = MLP([2 * in_dim, out_dim, 1], activation=relu, seed=seed)
        else:
            self.mlp_attn = None

    def forward_node(
        self,
        h_n: np.ndarray,
        neighbor_indices: List[int],
        H: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute updated embedding and attention coefficients for single node n."""
        if len(neighbor_indices) == 0:
            # Self-connection or zero
            return self.activation(np.dot(h_n, self.W_val) + self.bias), np.array([])

        neigh_embs = H[neighbor_indices]

        if self.attn_type == "bilinear":
            attn_weights = compute_attention_bilinear(h_n, neigh_embs, self.W_attn)
        elif self.attn_type == "mlp":
            attn_weights = compute_attention_mlp(h_n, neigh_embs, self.mlp_attn)
        else:
            raise ValueError(f"Unknown attn_type: {self.attn_type}")

        # Weighted aggregation (Eq 13.24)
        neigh_vals = np.dot(neigh_embs, self.W_val)
        z_n = np.sum(attn_weights[:, None] * neigh_vals, axis=0)
        h_next = self.activation(z_n + self.bias)
        return h_next, attn_weights

    def forward(self, graph: Graph, H: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass across all nodes."""
        if H is None:
            H = graph.node_features.copy()

        N = graph.num_nodes
        H_next = np.zeros((N, self.out_dim))

        for n in range(N):
            neighs = graph.neighbors(n) if hasattr(graph, "neighbors") else list(np.where(graph.A[n] > 0)[0])
            # Include self-loop in attention neighborhood
            if n not in neighs:
                neighs = [n] + neighs
            h_n_next, _ = self.forward_node(H[n], neighs, H)
            H_next[n] = h_n_next

        return H_next


class MultiHeadGATLayer:
    """
    Multi-Head Graph Attention Layer (Section 13.3.1).
    Runs H independent attention heads and concatenates their outputs,
    followed by linear projection W_proj.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim_per_head: int,
        num_heads: int = 4,
        attn_type: str = "bilinear",
        activation: Callable[[np.ndarray], np.ndarray] = relu,
        seed: Optional[int] = None,
    ):
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.out_dim_per_head = out_dim_per_head
        self.total_concat_dim = num_heads * out_dim_per_head
        self.activation = activation

        self.heads: List[GATLayer] = []
        for h in range(num_heads):
            h_seed = seed + h if seed is not None else None
            self.heads.append(
                GATLayer(in_dim, out_dim_per_head, attn_type=attn_type, activation=activation, seed=h_seed)
            )

        rng = np.random.RandomState(seed + 99 if seed is not None else None)
        limit = np.sqrt(6.0 / (self.total_concat_dim + out_dim_per_head))
        self.W_proj = rng.uniform(-limit, limit, (self.total_concat_dim, out_dim_per_head))
        self.bias = np.zeros(out_dim_per_head)

    def forward(self, graph: Graph, H: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass running all attention heads and projecting."""
        head_outputs = [head.forward(graph, H) for head in self.heads]
        # Concatenate along feature dimension
        H_concat = np.concatenate(head_outputs, axis=-1)
        # Linear projection
        return self.activation(np.dot(H_concat, self.W_proj) + self.bias)


# =============================================================================
# 2. Edge Embeddings (Section 13.3.2, Eqs 13.29-13.31)
# =============================================================================

class EdgeNodeMPNNLayer:
    """
    Message-Passing with Edge and Node Hidden Variables (Section 13.3.2):
        e_{nm}^{(l+1)} = Update_edge(e_{nm}^{(l)}, h_n^{(l)}, h_m^{(l)})  (Eq 13.29)
        z_n^{(l+1)} = Aggregate_node({e_{nm}^{(l+1)} : m in N(n)})        (Eq 13.30)
        h_n^{(l+1)} = Update_node(h_n^{(l)}, z_n^{(l+1)})                 (Eq 13.31)
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        activation: Callable[[np.ndarray], np.ndarray] = relu,
        seed: Optional[int] = None,
    ):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.activation = activation

        # MLP for edge update: takes [e_nm; h_n; h_m] -> edge_dim
        self.mlp_edge = MLP([edge_dim + 2 * node_dim, edge_dim, edge_dim], activation=activation, seed=seed)
        # MLP for node update: takes [h_n; z_n] -> node_dim
        self.mlp_node = MLP([node_dim + edge_dim, node_dim, node_dim], activation=activation, seed=seed + 1 if seed is not None else None)

    def forward(
        self,
        graph: Graph,
        H: np.ndarray,
        E_dict: Dict[Tuple[int, int], np.ndarray],
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
        """
        Synchronous update of edge and node representations.
        """
        N = graph.num_nodes
        E_next: Dict[Tuple[int, int], np.ndarray] = {}

        # 1. Update edges (Eq 13.29)
        for u, v in graph.edges:
            e_uv = E_dict.get((u, v), np.zeros(self.edge_dim))
            inp_uv = np.concatenate([e_uv, H[u], H[v]], axis=-1)
            E_next[(u, v)] = self.mlp_edge.forward(inp_uv)

            # In undirected graph, also update reverse edge
            if not graph.is_directed:
                e_vu = E_dict.get((v, u), e_uv)
                inp_vu = np.concatenate([e_vu, H[v], H[u]], axis=-1)
                E_next[(v, u)] = self.mlp_edge.forward(inp_vu)

        # 2. Aggregate edges into node messages (Eq 13.30)
        # 3. Update node representations (Eq 13.31)
        H_next = np.zeros_like(H)
        for n in range(N):
            neighs = graph.neighbors(n) if hasattr(graph, "neighbors") else list(np.where(graph.A[n] > 0)[0])
            if len(neighs) > 0:
                edge_msgs = [E_next.get((n, m), E_next.get((m, n), np.zeros(self.edge_dim))) for m in neighs]
                z_n = np.mean(edge_msgs, axis=0)
            else:
                z_n = np.zeros(self.edge_dim)

            inp_node = np.concatenate([H[n], z_n], axis=-1)
            H_next[n] = self.mlp_node.forward(inp_node)

        return H_next, E_next


# =============================================================================
# 3. Graph Embeddings & General MPNN (Algorithm 13.2, Section 13.3.3, Eqs 13.32-13.35)
# =============================================================================

class GeneralMPNNLayer:
    """
    General Graph Network Layer with Node, Edge, and Graph Embeddings
    (Battaglia et al. 2018, Section 13.3.3, Algorithm 13.2, Figure 13.5).
        e_{nm}^{(l+1)} = Update_edge(e_{nm}^{(l)}, h_n^{(l)}, h_m^{(l)}, g^{(l)})           (Eq 13.32)
        z_n^{(l+1)} = Aggregate_node({e_{nm}^{(l+1)} : m in N(n)})                           (Eq 13.33)
        h_n^{(l+1)} = Update_node(h_n^{(l)}, z_n^{(l+1)}, g^{(l)})                          (Eq 13.34)
        g^{(l+1)} = Update_graph(g^{(l)}, {h_n^{(l+1)} : n in V}, {e_{nm}^{(l+1)} : in E})   (Eq 13.35)
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        graph_dim: int,
        activation: Callable[[np.ndarray], np.ndarray] = relu,
        seed: Optional[int] = None,
    ):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.graph_dim = graph_dim
        self.activation = activation

        s = seed if seed is not None else 0
        # Edge update MLP: [e; h_n; h_m; g] -> edge_dim (Eq 13.32)
        self.mlp_edge = MLP([edge_dim + 2 * node_dim + graph_dim, edge_dim, edge_dim], activation=activation, seed=s)
        # Node update MLP: [h; z; g] -> node_dim (Eq 13.34)
        self.mlp_node = MLP([node_dim + edge_dim + graph_dim, node_dim, node_dim], activation=activation, seed=s + 1)
        # Graph update MLP: [g; pooled_h; pooled_e] -> graph_dim (Eq 13.35)
        self.mlp_graph = MLP([graph_dim + node_dim + edge_dim, graph_dim, graph_dim], activation=activation, seed=s + 2)

    def forward(
        self,
        graph: Graph,
        H: np.ndarray,
        E_dict: Dict[Tuple[int, int], np.ndarray],
        g: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], np.ndarray], np.ndarray]:
        """
        Execute one step of Algorithm 13.2.
        """
        N = graph.num_nodes
        E_next: Dict[Tuple[int, int], np.ndarray] = {}

        # Stage 1: Edge updates (Eq 13.32)
        for u, v in graph.edges:
            e_uv = E_dict.get((u, v), np.zeros(self.edge_dim))
            inp_uv = np.concatenate([e_uv, H[u], H[v], g], axis=-1)
            e_uv_next = self.mlp_edge.forward(inp_uv)
            E_next[(u, v)] = e_uv_next

            if not graph.is_directed:
                e_vu = E_dict.get((v, u), e_uv)
                inp_vu = np.concatenate([e_vu, H[v], H[u], g], axis=-1)
                E_next[(v, u)] = self.mlp_edge.forward(inp_vu)

        # Stage 2: Node aggregation (Eq 13.33) & Node updates (Eq 13.34)
        H_next = np.zeros_like(H)
        for n in range(N):
            neighs = graph.neighbors(n) if hasattr(graph, "neighbors") else list(np.where(graph.A[n] > 0)[0])
            if len(neighs) > 0:
                edge_msgs = [E_next.get((n, m), E_next.get((m, n), np.zeros(self.edge_dim))) for m in neighs]
                z_n = np.mean(edge_msgs, axis=0)
            else:
                z_n = np.zeros(self.edge_dim)

            inp_node = np.concatenate([H[n], z_n, g], axis=-1)
            H_next[n] = self.mlp_node.forward(inp_node)

        # Stage 3: Global graph update (Eq 13.35)
        pooled_h = np.mean(H_next, axis=0)
        all_e = list(E_next.values())
        pooled_e = np.mean(all_e, axis=0) if len(all_e) > 0 else np.zeros(self.edge_dim)

        inp_graph = np.concatenate([g, pooled_h, pooled_e], axis=-1)
        g_next = self.mlp_graph.forward(inp_graph)

        return H_next, E_next, g_next


# =============================================================================
# 4. Over-smoothing Analysis & Mitigations (Section 13.3.4, Eqs 13.36-13.37)
# =============================================================================

def compute_dirichlet_energy(H: np.ndarray, L_sym: np.ndarray) -> float:
    """
    Dirichlet energy of node embeddings:
        E(H) = (1/2) * Tr(H^T L_sym H)
    As GNN depth increases, over-smoothing causes E(H) -> 0.
    """
    return float(0.5 * np.trace(np.dot(H.T, np.dot(L_sym, H))))


def compute_mean_pairwise_distance(H: np.ndarray) -> float:
    """Mean Euclidean distance between all pairs of node embeddings."""
    N = H.shape[0]
    if N <= 1:
        return 0.0
    dists = []
    for i in range(N):
        for j in range(i + 1, N):
            dists.append(np.linalg.norm(H[i] - H[j]))
    return float(np.mean(dists))


def jumping_knowledge_pool(layer_representations: List[np.ndarray], mode: str = "concat") -> np.ndarray:
    """
    Jumping Knowledge Network (JK-Net) pooling (Eq 13.37):
        y_n = f(h_n^{(1)} ++ h_n^{(2)} ++ ... ++ h_n^{(L)})
    or max pooling across layers.
    """
    if mode == "concat":
        # Concatenate across feature dimension
        return np.concatenate(layer_representations, axis=-1)
    elif mode == "max":
        # Element-wise max across layers
        stacked = np.stack(layer_representations, axis=0)
        return np.max(stacked, axis=0)
    elif mode == "mean":
        stacked = np.stack(layer_representations, axis=0)
        return np.mean(stacked, axis=0)
    else:
        raise ValueError(f"Unknown JK mode: {mode}")


# =============================================================================
# 5. Regularization (Section 13.3.5)
# =============================================================================

def apply_drop_edge(A: np.ndarray, p_drop: float = 0.2, seed: Optional[int] = None) -> np.ndarray:
    """
    DropEdge regularization (Rong et al. 2020):
    Randomly drops a fraction p_drop of existing edges during training.
    """
    rng = np.random.RandomState(seed)
    A_dropped = A.copy()
    N = A.shape[0]

    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > 0 and rng.rand() < p_drop:
                A_dropped[i, j] = 0.0
                A_dropped[j, i] = 0.0

    return A_dropped


def apply_node_dropout(X: np.ndarray, p_drop: float = 0.2, seed: Optional[int] = None) -> np.ndarray:
    """
    Node Dropout: Randomly masks out entire node features during training.
    """
    rng = np.random.RandomState(seed)
    N = X.shape[0]
    mask = rng.rand(N) >= p_drop
    X_dropped = X.copy()
    X_dropped[~mask] = 0.0
    return X_dropped


# =============================================================================
# 6. Geometric Deep Learning (EGNN / E(n)-Equivariance, Section 13.3.6)
# =============================================================================

class EGNNLayer:
    """
    E(n) Equivariant Graph Neural Network Layer (Satorras et al., 2021, Eqs 13.38-13.41).
    Preserves equivariance under translation, rotation, and reflection in R^3.
        e_{nm}^{(l+1)} = Update_edge(e_{nm}^{(l)}, h_n^{(l)}, h_m^{(l)}, ||r_n^{(l)} - r_m^{(l)}||^2)  (Eq 13.38)
        r_n^{(l+1)} = r_n^{(l)} + C * sum_{m} (r_n^{(l)} - r_m^{(l)}) * phi(e_{nm}^{(l+1)})              (Eq 13.39)
        z_n^{(l+1)} = Aggregate({e_{nm}^{(l+1)} : m in N(n)})                                            (Eq 13.40)
        h_n^{(l+1)} = Update_node(h_n^{(l)}, z_n^{(l+1)})                                                (Eq 13.41)
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        coord_scale: float = 0.1,
        activation: Callable[[np.ndarray], np.ndarray] = relu,
        seed: Optional[int] = None,
    ):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.coord_scale = coord_scale
        self.activation = activation

        s = seed if seed is not None else 0
        # Edge update taking [e; h_n; h_m; ||r_n - r_m||^2] (Eq 13.38)
        self.mlp_edge = MLP([edge_dim + 2 * node_dim + 1, edge_dim, edge_dim], activation=activation, seed=s)
        # Coordinate weight scalar MLP phi: edge_dim -> 1 (Eq 13.39)
        self.mlp_coord = MLP([edge_dim, 8, 1], activation=activation, seed=s + 1)
        # Node update taking [h_n; z_n] (Eq 13.41)
        self.mlp_node = MLP([node_dim + edge_dim, node_dim, node_dim], activation=activation, seed=s + 2)

    def forward(
        self,
        graph: Graph,
        H: np.ndarray,
        R: np.ndarray,
        E_dict: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
        """
        Forward pass with coordinates R in R^{N x 3} and features H in R^{N x D}.
        """
        N = graph.num_nodes
        if E_dict is None:
            E_dict = {}

        E_next: Dict[Tuple[int, int], np.ndarray] = {}

        # 1. Update edge representations using invariant squared distance (Eq 13.38)
        for u, v in graph.edges:
            sq_dist = np.sum((R[u] - R[v]) ** 2)
            e_uv = E_dict.get((u, v), np.zeros(self.edge_dim))
            inp_uv = np.concatenate([e_uv, H[u], H[v], [sq_dist]], axis=-1)
            e_next_uv = self.mlp_edge.forward(inp_uv)
            E_next[(u, v)] = e_next_uv

            if not graph.is_directed:
                inp_vu = np.concatenate([e_uv, H[v], H[u], [sq_dist]], axis=-1)
                E_next[(v, u)] = self.mlp_edge.forward(inp_vu)

        # 2. Update coordinates equivariantly (Eq 13.39)
        R_next = R.copy()
        for n in range(N):
            neighs = graph.neighbors(n) if hasattr(graph, "neighbors") else list(np.where(graph.A[n] > 0)[0])
            coord_disp = np.zeros(3)
            for m in neighs:
                e_nm = E_next.get((n, m), E_next.get((m, n), np.zeros(self.edge_dim)))
                # Scalar weight phi(e_nm)
                phi_weight = self.mlp_coord.forward(e_nm)[0]
                coord_disp += (R[n] - R[m]) * phi_weight
            R_next[n] = R[n] + self.coord_scale * coord_disp

        # 3. Aggregate edge representations (Eq 13.40) & Update node features (Eq 13.41)
        H_next = np.zeros_like(H)
        for n in range(N):
            neighs = graph.neighbors(n) if hasattr(graph, "neighbors") else list(np.where(graph.A[n] > 0)[0])
            if len(neighs) > 0:
                edge_msgs = [E_next.get((n, m), E_next.get((m, n), np.zeros(self.edge_dim))) for m in neighs]
                z_n = np.mean(edge_msgs, axis=0)
            else:
                z_n = np.zeros(self.edge_dim)
            inp_node = np.concatenate([H[n], z_n], axis=-1)
            H_next[n] = self.mlp_node.forward(inp_node)

        return H_next, R_next, E_next


def check_egnn_equivariance(
    layer: EGNNLayer,
    graph: Graph,
    H: np.ndarray,
    R: np.ndarray,
    rtol: float = 1e-4,
) -> Dict[str, bool]:
    """
    Numerically verify E(3) symmetries:
    - Translation equivariance: R' = R + t => R_out' = R_out + t
    - Rotation equivariance: R' = R @ rot_mat => R_out' = R_out @ rot_mat
    - Reflection equivariance: R' = -R => R_out' = -R_out
    - Feature invariance: H_out' == H_out under all rigid transformations.
    """
    H_base, R_base, _ = layer.forward(graph, H, R)

    # 1. Translation
    t = np.array([1.5, -2.0, 0.7])
    R_trans = R + t
    H_t, R_t, _ = layer.forward(graph, H, R_trans)
    trans_equiv = np.allclose(R_t, R_base + t, rtol=rtol, atol=1e-5)
    trans_inv_feat = np.allclose(H_t, H_base, rtol=rtol, atol=1e-5)

    # 2. 3D Rotation matrix (around z-axis)
    theta = np.pi / 3.0
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])
    R_rot = np.dot(R, rot_mat)
    H_r, R_r, _ = layer.forward(graph, H, R_rot)
    rot_equiv = np.allclose(R_r, np.dot(R_base, rot_mat), rtol=rtol, atol=1e-5)
    rot_inv_feat = np.allclose(H_r, H_base, rtol=rtol, atol=1e-5)

    # 3. Reflection (parity inversion)
    R_ref = -R
    H_ref, R_ref_out, _ = layer.forward(graph, H, R_ref)
    ref_equiv = np.allclose(R_ref_out, -R_base, rtol=rtol, atol=1e-5)
    ref_inv_feat = np.allclose(H_ref, H_base, rtol=rtol, atol=1e-5)

    return {
        "translation_equivariance": bool(trans_equiv),
        "translation_invariance_features": bool(trans_inv_feat),
        "rotation_equivariance": bool(rot_equiv),
        "rotation_invariance_features": bool(rot_inv_feat),
        "reflection_equivariance": bool(ref_equiv),
        "reflection_invariance_features": bool(ref_inv_feat),
    }


# =============================================================================
# 7. Faithful Reproduction of Figure 13.5
# =============================================================================

def generate_figure_13_5(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Faithful reproduction of Figure 13.5:
    General graph message-passing updates defined by (13.32) to (13.35):
    (a) Edge updates (e_nm updated in red)
    (b) Node updates (h_n updated in red)
    (c) Global graph updates (g updated in red)
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.2), dpi=300)

    # Canonical 6-node coordinates
    nodes = {
        0: (-2.2, 0.0),    # leftmost
        1: (-1.0, 1.6),    # topleft
        2: (-1.0, -1.6),   # bottomleft
        3: (0.5, 0.0),     # center n
        4: (2.3, 1.2),     # topright
        5: (2.0, -1.3),    # bottomright m
    }

    edges = [
        (0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (3, 5)
    ]

    r_node = 0.42

    # Color palette matching Bishop (2024)
    c_blue_fill = "#d1c4e9"
    c_blue_edge = "#1a237e"
    c_red_fill = "#ffcdd2"
    c_red_edge = "#d32f2f"
    c_gray_fill = "#b0bec5"
    c_gray_edge = "#37474f"

    for ax in axes:
        ax.set_xlim(-3.2, 3.5)
        ax.set_ylim(-2.8, 3.7)
        ax.axis("off")

    # === Subplot (a): Edge updates (Eq 13.32) ===
    ax_a = axes[0]
    box_a = patches.Rectangle((-0.45, 2.4), 0.9, 0.9, facecolor=c_blue_fill, edgecolor=c_blue_edge, linewidth=2.0)
    ax_a.add_patch(box_a)
    ax_a.text(0.0, 2.85, r"$\mathcal{G}$", fontsize=16, ha="center", va="center")
    ax_a.text(0.85, 2.85, r"$\mathbf{g}$", fontsize=14, color=c_blue_edge, ha="left", va="center")

    for u, v in edges:
        p1, p2 = nodes[u], nodes[v]
        if (u, v) == (3, 5) or (v, u) == (3, 5):
            ax_a.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c_red_edge, linewidth=2.8, zorder=2)
            mid_x = (p1[0] + p2[0]) / 2 + 0.05
            mid_y = (p1[1] + p2[1]) / 2 - 0.28
            ax_a.text(mid_x, mid_y, r"$\mathbf{e}_{nm}$", fontsize=13, color=c_red_edge, ha="center", va="center")
        else:
            ax_a.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", linewidth=1.6, zorder=2)

    for idx, pos in nodes.items():
        if idx == 3:  # n (blue)
            fc, ec = c_blue_fill, c_blue_edge
            ax_a.text(pos[0], pos[1], r"$n$", fontsize=13, fontstyle="italic", ha="center", va="center", zorder=5)
            ax_a.text(pos[0] + 0.45, pos[1] + 0.35, r"$\mathbf{h}_n$", fontsize=13, color=c_blue_edge, zorder=5)
        elif idx == 5:  # m (blue)
            fc, ec = c_blue_fill, c_blue_edge
            ax_a.text(pos[0], pos[1], r"$m$", fontsize=13, fontstyle="italic", ha="center", va="center", zorder=5)
            ax_a.text(pos[0] + 0.05, pos[1] - 0.65, r"$\mathbf{h}_m$", fontsize=13, color=c_blue_edge, zorder=5)
        else:
            fc, ec = c_gray_fill, c_gray_edge
        c = patches.Circle(pos, r_node, facecolor=fc, edgecolor=ec, linewidth=2.0, zorder=4)
        ax_a.add_patch(c)

    ax_a.text(0, -2.5, "(a)", fontsize=13, ha="center")

    # === Subplot (b): Node updates (Eq 13.34) ===
    ax_b = axes[1]
    box_b = patches.Rectangle((-0.45, 2.4), 0.9, 0.9, facecolor=c_blue_fill, edgecolor=c_blue_edge, linewidth=2.0)
    ax_b.add_patch(box_b)
    ax_b.text(0.0, 2.85, r"$\mathcal{G}$", fontsize=16, ha="center", va="center")
    ax_b.text(0.85, 2.85, r"$\mathbf{g}$", fontsize=14, color=c_blue_edge, ha="left", va="center")

    for u, v in edges:
        p1, p2 = nodes[u], nodes[v]
        is_incident_3 = (u == 3 or v == 3)
        if is_incident_3:
            ax_b.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c_blue_edge, linewidth=2.5, zorder=2)
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            if (u, v) == (3, 5) or (v, u) == (3, 5):
                ax_b.text(mid_x - 0.05, mid_y - 0.28, r"$\mathbf{e}_{nm}$", fontsize=12, color=c_blue_edge, ha="center")
            elif (u, v) == (1, 3) or (v, u) == (1, 3):
                ax_b.text(mid_x - 0.25, mid_y - 0.05, r"$\mathbf{e}$", fontsize=12, color=c_blue_edge, ha="center")
            elif (u, v) == (2, 3) or (v, u) == (2, 3):
                ax_b.text(mid_x - 0.25, mid_y - 0.05, r"$\mathbf{e}$", fontsize=12, color=c_blue_edge, ha="center")
            elif (u, v) == (3, 4) or (v, u) == (3, 4):
                ax_b.text(mid_x - 0.05, mid_y + 0.25, r"$\mathbf{e}$", fontsize=12, color=c_blue_edge, ha="center")
        else:
            ax_b.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", linewidth=1.6, zorder=2)

    for idx, pos in nodes.items():
        if idx == 3:  # n (red)
            fc, ec = c_red_fill, c_red_edge
            ax_b.text(pos[0], pos[1], r"$n$", fontsize=13, fontstyle="italic", ha="center", va="center", zorder=5)
            ax_b.text(pos[0] + 0.45, pos[1] + 0.35, r"$\mathbf{h}_n$", fontsize=13, color=c_red_edge, zorder=5)
        elif idx == 0:  # gray
            fc, ec = c_gray_fill, c_gray_edge
        else:  # neighbors (blue)
            fc, ec = c_blue_fill, c_blue_edge
            if idx == 5:
                ax_b.text(pos[0], pos[1], r"$m$", fontsize=13, fontstyle="italic", ha="center", va="center", zorder=5)
        c = patches.Circle(pos, r_node, facecolor=fc, edgecolor=ec, linewidth=2.0, zorder=4)
        ax_b.add_patch(c)

    ax_b.text(0, -2.5, "(b)", fontsize=13, ha="center")

    # === Subplot (c): Global graph updates (Eq 13.35) ===
    ax_c = axes[2]
    box_c = patches.Rectangle((-0.45, 2.4), 0.9, 0.9, facecolor=c_red_fill, edgecolor=c_red_edge, linewidth=2.0)
    ax_c.add_patch(box_c)
    ax_c.text(0.0, 2.85, r"$\mathcal{G}$", fontsize=16, ha="center", va="center")
    ax_c.text(0.85, 2.85, r"$\mathbf{g}$", fontsize=14, color=c_red_edge, ha="left", va="center")

    for u, v in edges:
        p1, p2 = nodes[u], nodes[v]
        ax_c.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c_blue_edge, linewidth=2.2, zorder=2)
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        norm = max(np.sqrt(dx**2 + dy**2), 1e-6)
        perp_x = -dy / norm * 0.22
        perp_y = dx / norm * 0.22
        ax_c.text(mid_x + perp_x, mid_y + perp_y, r"$\mathbf{e}$", fontsize=11, color=c_blue_edge, ha="center", va="center")

    for idx, pos in nodes.items():
        c = patches.Circle(pos, r_node, facecolor=c_blue_fill, edgecolor=c_blue_edge, linewidth=2.0, zorder=4)
        ax_c.add_patch(c)
        h_offsets = {
            0: (-0.6, 0.0),
            1: (0.45, 0.45),
            2: (0.45, -0.45),
            3: (0.45, 0.45),
            4: (0.45, 0.45),
            5: (0.0, -0.65),
        }
        ox, oy = h_offsets[idx]
        ax_c.text(pos[0] + ox, pos[1] + oy, r"$\mathbf{h}$", fontsize=12, color=c_blue_edge, ha="center", va="center")

    ax_c.text(0, -2.5, "(c)", fontsize=13, ha="center")

    plt.tight_layout()
    _save_figure(fig, "fig_13_5_general_graph_updates", save_dir)
    _save_figure(fig, "Figure_13_5", save_dir)
    return fig


if __name__ == "__main__":
    print("Generating Figure 13.5...")
    generate_figure_13_5()
    print("Figure 13.5 generated successfully!")
