"""
common/neural_message_passing.py
================================
Section 13.2: Neural Message-Passing
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements:
1. Convolutional Filters on Graphs (Section 13.2.1):
   - Standard 2D conv filter (Eq 13.8) vs Equivariant graph filter (Eq 13.9).
   - Numerical verification of permutation equivariance/invariance.
2. Message-Passing Neural Networks (Algorithm 13.1, Section 13.2.2):
   - Two-stage MPNN framework: Aggregate (Eq 13.10) and Update (Eq 13.11).
3. Aggregation Operators (Section 13.2.3):
   - Summation (Eq 13.12).
   - Mean / Average (Eq 13.13).
   - Symmetric Normalized (Kipf & Welling 2016, Eq 13.14).
   - Element-wise Max and Min pooling.
   - Deep Sets / Parameterized MLP Aggregation (Zaheer et al. 2017, Eq 13.15).
4. Update Operators (Section 13.2.4):
   - Linear combination with activation (Eq 13.16).
   - Shared weight update (Eq 13.17).
   - Concatenation and GRU update operators.
   - Matrix formulation H^{(l+1)} = sigma(A_tilde_norm H^{(l)} W^{(l)}) (Eqs 13.18, 13.19).
5. Downstream Prediction Heads:
   - Node Classification (Section 13.2.5):
     * Softmax readout (Eq 13.20) and Cross-entropy loss (Eq 13.21).
     * Node partitioning: V_train (labeled), V_trans (transductive), V_induct (inductive).
     * Full gradient descent training for multi-layer GCN.
   - Edge Classification / Link Prediction (Section 13.2.6):
     * Pairwise score p(n, m) = sigma(h_n^T h_m) (Eq 13.22) and bilinear form.
   - Graph Classification (Section 13.2.7):
     * Permutation-invariant global pooling y = f(sum_n h_n) (Eq 13.23).
6. Receptive Field Analysis (Figure 13.4):
   - Hop-by-hop neighborhood expansion and tracking.
7. Publication-Quality Figure Reproductions (Figures 13.3 & 13.4):
   - Figure 13.3: 2D image conv filter vs graph message-passing node.
   - Figure 13.4: Receptive field expansion through layers l=1, 2, 3.
   - Saved to 13/result/ and result/.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

from .plot_utils import setup_style, save_plot
from .machine_learning_on_graphs import Graph, build_permutation_matrix


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
# Helper Activations and Math Functions
# =============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit element-wise activation."""
    return np.maximum(0.0, x)


def d_relu(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (x > 0.0).astype(float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid function."""
    x_clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along specified axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# =============================================================================
# 1. Convolutional Filters on Graphs vs Images (Section 13.2.1)
# =============================================================================

def conv2d_filter_standard(
    patch: np.ndarray,
    weights: np.ndarray,
    bias: float = 0.0,
    activation: Callable[[np.ndarray], np.ndarray] = relu,
) -> float:
    """
    Standard 2D convolutional filter (Eq 13.8):
        z_i^{(l+1)} = f(sum_j w_j z_j^{(l)} + b)
    where sum is taken over all pixels in a patch.
    Note: Sensitivity to pixel ordering causes lack of permutation equivariance.
    """
    assert patch.shape == weights.shape, f"Patch shape {patch.shape} != weights shape {weights.shape}"
    linear_val = np.sum(patch * weights) + bias
    return float(activation(np.array([linear_val]))[0])


def conv2d_filter_equivariant(
    center_val: float,
    neighbor_vals: np.ndarray,
    w_self: float,
    w_neigh: float,
    bias: float = 0.0,
    activation: Callable[[np.ndarray], np.ndarray] = relu,
) -> float:
    """
    Permutation-equivariant filter on graph (Eq 13.9):
        z_i^{(l+1)} = f(w_neigh * sum_{j in N(i)} z_j^{(l)} + w_self * z_i^{(l)} + b)
    where neighbor weights are shared across all neighbors.
    """
    sum_neigh = np.sum(neighbor_vals)
    linear_val = w_neigh * sum_neigh + w_self * center_val + bias
    return float(activation(np.array([linear_val]))[0])


def check_conv_permutation_sensitivity(
    patch_values: np.ndarray,
    weights_std: np.ndarray,
    w_self: float,
    w_neigh: float,
    bias: float = 0.0,
) -> Dict[str, Any]:
    """
    Demonstrate that standard conv is sensitive to neighbor reordering,
    while graph equivariant filter is completely invariant to neighbor reordering.
    """
    # Flatten neighbor indices (excluding center)
    center_val = patch_values[1, 1]
    neighbors = np.delete(patch_values.flatten(), 4)

    # Standard conv result on original
    std_orig = conv2d_filter_standard(patch_values, weights_std, bias)

    # Permute neighbors
    perm = np.random.RandomState(42).permutation(len(neighbors))
    neighbors_perm = neighbors[perm]

    # Reconstruct permuted patch
    patch_perm = np.zeros_like(patch_values)
    flat_perm = np.insert(neighbors_perm, 4, center_val)
    patch_perm = flat_perm.reshape(patch_values.shape)

    std_perm = conv2d_filter_standard(patch_perm, weights_std, bias)

    # Graph equivariant conv results
    graph_orig = conv2d_filter_equivariant(center_val, neighbors, w_self, w_neigh, bias)
    graph_perm = conv2d_filter_equivariant(center_val, neighbors_perm, w_self, w_neigh, bias)

    return {
        "std_orig": std_orig,
        "std_perm": std_perm,
        "std_diff": abs(std_orig - std_perm),
        "graph_orig": graph_orig,
        "graph_perm": graph_perm,
        "graph_diff": abs(graph_orig - graph_perm),
        "is_graph_equivariant": bool(np.isclose(graph_orig, graph_perm)),
    }


# =============================================================================
# 2. Aggregation Operators (Section 13.2.3)
# =============================================================================

def aggregate_sum(neighbor_embeddings: np.ndarray) -> np.ndarray:
    """
    Summation aggregation (Eq 13.12):
        Aggregate({h_m^{(l)} : m in N(n)}) = sum_{m in N(n)} h_m^{(l)}
    """
    if neighbor_embeddings.shape[0] == 0:
        return np.zeros(neighbor_embeddings.shape[1] if neighbor_embeddings.ndim > 1 else 0)
    return np.sum(neighbor_embeddings, axis=0)


def aggregate_mean(neighbor_embeddings: np.ndarray) -> np.ndarray:
    """
    Mean / Average aggregation (Eq 13.13):
        Aggregate({h_m^{(l)} : m in N(n)}) = (1 / |N(n)|) sum_{m in N(n)} h_m^{(l)}
    """
    if neighbor_embeddings.shape[0] == 0:
        return np.zeros(neighbor_embeddings.shape[1] if neighbor_embeddings.ndim > 1 else 0)
    return np.mean(neighbor_embeddings, axis=0)


def aggregate_norm(
    node_idx: int,
    neighbor_indices: List[int],
    all_embeddings: np.ndarray,
    degrees: np.ndarray,
) -> np.ndarray:
    """
    Symmetric normalized aggregation (Kipf & Welling 2016, Eq 13.14):
        Aggregate({h_m^{(l)} : m in N(n)}) = sum_{m in N(n)} h_m^{(l)} / sqrt(|N(n)| * |N(m)|)
    """
    deg_n = degrees[node_idx]
    if deg_n == 0 or len(neighbor_indices) == 0:
        return np.zeros(all_embeddings.shape[1])

    z = np.zeros(all_embeddings.shape[1])
    for m in neighbor_indices:
        deg_m = degrees[m]
        norm_factor = 1.0 / np.sqrt(max(deg_n * deg_m, 1e-12))
        z += norm_factor * all_embeddings[m]
    return z


def aggregate_max(neighbor_embeddings: np.ndarray) -> np.ndarray:
    """Element-wise maximum aggregation."""
    if neighbor_embeddings.shape[0] == 0:
        return np.zeros(neighbor_embeddings.shape[1] if neighbor_embeddings.ndim > 1 else 0)
    return np.max(neighbor_embeddings, axis=0)


def aggregate_min(neighbor_embeddings: np.ndarray) -> np.ndarray:
    """Element-wise minimum aggregation."""
    if neighbor_embeddings.shape[0] == 0:
        return np.zeros(neighbor_embeddings.shape[1] if neighbor_embeddings.ndim > 1 else 0)
    return np.min(neighbor_embeddings, axis=0)


class MLP:
    """
    Simple Multi-Layer Perceptron in NumPy.
    Used for Deep Sets parameterization (Eq 13.15) and readout heads.
    """

    def __init__(
        self,
        layer_dims: List[int],
        activation: Callable[[np.ndarray], np.ndarray] = relu,
        seed: Optional[int] = None,
    ):
        self.layer_dims = layer_dims
        self.activation = activation
        rng = np.random.RandomState(seed)

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for d_in, d_out in zip(layer_dims[:-1], layer_dims[1:]):
            limit = np.sqrt(6.0 / (d_in + d_out))
            W = rng.uniform(-limit, limit, (d_in, d_out))
            b = np.zeros(d_out)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through MLP layers."""
        h = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = np.dot(h, W) + b
            if i < len(self.weights) - 1:
                h = self.activation(h)
        return h


def aggregate_deep_sets(
    neighbor_embeddings: np.ndarray,
    mlp_phi: MLP,
    mlp_theta: MLP,
    pool_op: str = "sum",
) -> np.ndarray:
    """
    Deep Sets / Universal permutation-invariant aggregation (Zaheer et al. 2017, Eq 13.15):
        Aggregate({h_m^{(l)} : m in N(n)}) = MLP_theta(sum_{m in N(n)} MLP_phi(h_m^{(l)}))
    """
    if neighbor_embeddings.shape[0] == 0:
        d_out = mlp_theta.layer_dims[-1]
        return np.zeros(d_out)

    # Transform each neighbor embedding with MLP_phi
    phi_embeddings = np.array([mlp_phi.forward(h) for h in neighbor_embeddings])

    # Invariant pooling (sum, mean, max)
    if pool_op == "sum":
        pooled = np.sum(phi_embeddings, axis=0)
    elif pool_op == "mean":
        pooled = np.mean(phi_embeddings, axis=0)
    elif pool_op == "max":
        pooled = np.max(phi_embeddings, axis=0)
    else:
        raise ValueError(f"Unknown pooling operation: {pool_op}")

    # Transform pooled representation with MLP_theta
    return mlp_theta.forward(pooled)


# =============================================================================
# 3. Update Operators (Section 13.2.4)
# =============================================================================

def update_linear(
    h_self: np.ndarray,
    z_neigh: np.ndarray,
    W_self: np.ndarray,
    W_neigh: np.ndarray,
    bias: np.ndarray,
    activation: Callable[[np.ndarray], np.ndarray] = relu,
) -> np.ndarray:
    """
    Linear update operator (Eq 13.16):
        Update(h_n^{(l)}, z_n^{(l)}) = f(W_self * h_n^{(l)} + W_neigh * z_n^{(l)} + b)
    """
    val = np.dot(h_self, W_self) + np.dot(z_neigh, W_neigh) + bias
    return activation(val)


def update_shared(
    h_self: np.ndarray,
    z_sum: np.ndarray,
    W_shared: np.ndarray,
    bias: np.ndarray,
    activation: Callable[[np.ndarray], np.ndarray] = relu,
) -> np.ndarray:
    """
    Shared weight update operator (Eq 13.17):
        Update(h_n^{(l)}, z_n^{(l)}) = f(W_neigh * sum_{m in N(n) U {n}} h_m^{(l)} + b)
        = f(W_shared * (h_self + z_sum) + b)
    """
    val = np.dot(h_self + z_sum, W_shared) + bias
    return activation(val)


def update_concat(
    h_self: np.ndarray,
    z_neigh: np.ndarray,
    W_concat: np.ndarray,
    bias: np.ndarray,
    activation: Callable[[np.ndarray], np.ndarray] = relu,
) -> np.ndarray:
    """Concatenation update operator: f(W [h_self; z_neigh] + b)."""
    concat_vec = np.concatenate([h_self, z_neigh], axis=-1)
    val = np.dot(concat_vec, W_concat) + bias
    return activation(val)


# =============================================================================
# 4. Message-Passing Layer & Network (Algorithm 13.1, Eqs 13.18, 13.19)
# =============================================================================

class MessagePassingLayer:
    """
    General Message-Passing Neural Network Layer (Algorithm 13.1).
    Synchronously updates node embeddings:
        z_n^{(l)} = Aggregate({h_m^{(l)} : m in N(n)})   (Eq 13.10)
        h_n^{(l+1)} = Update(h_n^{(l)}, z_n^{(l)})        (Eq 13.11)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        aggregate_type: str = "sum",
        update_type: str = "linear",
        activation: Callable[[np.ndarray], np.ndarray] = relu,
        seed: Optional[int] = None,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregate_type = aggregate_type
        self.update_type = update_type
        self.activation = activation

        rng = np.random.RandomState(seed)
        limit = np.sqrt(6.0 / (in_dim + out_dim))

        self.W_self = rng.uniform(-limit, limit, (in_dim, out_dim))
        self.W_neigh = rng.uniform(-limit, limit, (in_dim, out_dim))
        self.W_concat = rng.uniform(-limit, limit, (2 * in_dim, out_dim))
        self.bias = np.zeros(out_dim)

        # Deep Sets MLPs if requested
        if aggregate_type == "deep_sets":
            self.mlp_phi = MLP([in_dim, out_dim, in_dim], activation=activation, seed=seed)
            self.mlp_theta = MLP([in_dim, out_dim, in_dim], activation=activation, seed=seed)
        else:
            self.mlp_phi = None
            self.mlp_theta = None

    def aggregate(
        self,
        node_idx: int,
        neighbors: List[int],
        H: np.ndarray,
        degrees: np.ndarray,
    ) -> np.ndarray:
        """Compute aggregated neighbor message z_n^{(l)} (Eq 13.10)."""
        if len(neighbors) == 0:
            if self.aggregate_type == "deep_sets":
                return np.zeros(self.mlp_theta.layer_dims[-1])
            return np.zeros(H.shape[1])

        neighbor_embs = H[neighbors]

        if self.aggregate_type == "sum":
            return aggregate_sum(neighbor_embs)
        elif self.aggregate_type == "mean":
            return aggregate_mean(neighbor_embs)
        elif self.aggregate_type == "norm":
            return aggregate_norm(node_idx, neighbors, H, degrees)
        elif self.aggregate_type == "max":
            return aggregate_max(neighbor_embs)
        elif self.aggregate_type == "min":
            return aggregate_min(neighbor_embs)
        elif self.aggregate_type == "deep_sets":
            return aggregate_deep_sets(neighbor_embs, self.mlp_phi, self.mlp_theta)
        else:
            raise ValueError(f"Unknown aggregation operator: {self.aggregate_type}")

    def update(self, h_self: np.ndarray, z_neigh: np.ndarray) -> np.ndarray:
        """Compute updated node embedding h_n^{(l+1)} (Eq 13.11)."""
        if self.update_type == "linear":
            return update_linear(h_self, z_neigh, self.W_self, self.W_neigh, self.bias, self.activation)
        elif self.update_type == "shared":
            return update_shared(h_self, z_neigh, self.W_neigh, self.bias, self.activation)
        elif self.update_type == "concat":
            return update_concat(h_self, z_neigh, self.W_concat, self.bias, self.activation)
        else:
            raise ValueError(f"Unknown update operator: {self.update_type}")

    def forward(self, graph: Graph, H: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Execute Algorithm 13.1 for one layer synchronously across all nodes.
        H: (N, in_dim) array of embeddings. Defaults to graph.node_features.
        """
        if H is None:
            H = graph.node_features.copy()

        N = graph.num_nodes
        degrees = np.sum(graph.A, axis=1)
        H_next = np.zeros((N, self.out_dim))

        for n in range(N):
            if hasattr(graph, "neighbors"):
                neighs = graph.neighbors(n)
            else:
                neighs = list(np.where(graph.A[n] > 0)[0])
            z_n = self.aggregate(n, neighs, H, degrees)
            H_next[n] = self.update(H[n], z_n)

        return H_next


class GCNLayer:
    """
    Spectral Graph Convolutional Layer (Kipf & Welling 2016, Eqs 13.18, 13.19):
        H^{(l+1)} = sigma(A_tilde_norm * H^{(l)} * W^{(l)} + b)
    where A_tilde = A + I_N, and A_tilde_norm = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Callable[[np.ndarray], np.ndarray] = relu,
        seed: Optional[int] = None,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation

        rng = np.random.RandomState(seed)
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = rng.uniform(-limit, limit, (in_dim, out_dim))
        self.bias = np.zeros(out_dim)

    def forward(self, A_norm: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Matrix formulation forward pass (Eq 13.18):
            Z = A_norm @ H @ W + bias
            H_next = activation(Z)
        """
        linear = np.dot(A_norm, np.dot(H, self.W)) + self.bias
        return self.activation(linear)


class GraphConvolutionalNetwork:
    """
    Multi-Layer Graph Convolutional Network (GCN).
    Supports node classification, link prediction, and graph-level predictions.
    """

    def __init__(
        self,
        layer_dims: List[int],
        activation: Callable[[np.ndarray], np.ndarray] = relu,
        seed: Optional[int] = None,
    ):
        self.layer_dims = layer_dims
        self.activation = activation
        self.layers: List[GCNLayer] = []

        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            l_seed = seed + i if seed is not None else None
            self.layers.append(GCNLayer(d_in, d_out, activation=activation, seed=l_seed))

    @staticmethod
    def compute_normalized_adjacency(A: np.ndarray) -> np.ndarray:
        """
        Compute renormalized adjacency matrix with self-loops:
            A_tilde = A + I_N
            A_tilde_norm = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}
        """
        N = A.shape[0]
        A_tilde = A + np.eye(N)
        deg_tilde = np.sum(A_tilde, axis=1)
        deg_inv_sqrt = np.power(deg_tilde, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        return D_inv_sqrt @ A_tilde @ D_inv_sqrt

    def forward_all_layers(self, A: np.ndarray, X: np.ndarray) -> List[np.ndarray]:
        """Forward pass through all layers returning list [H^{(0)}, H^{(1)}, ..., H^{(L)}]."""
        A_norm = self.compute_normalized_adjacency(A)
        embeddings = [X]
        H = X
        for layer in self.layers:
            H = layer.forward(A_norm, H)
            embeddings.append(H)
        return embeddings

    def forward(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Forward pass returning final node representations H^{(L)}."""
        return self.forward_all_layers(A, X)[-1]


# =============================================================================
# 5. Node Classification Head (Section 13.2.5)
# =============================================================================

class NodeClassifier:
    """
    Node Classification Model (Section 13.2.5).
    - Softmax Readout Layer (Eq 13.20):
        y_{ni} = exp(w_i^T h_n + b_i) / sum_j exp(w_j^T h_n + b_j)
    - Cross-Entropy Loss (Eq 13.21):
        L = - sum_{n in V_train} sum_i t_{ni} ln(y_{ni})
    - Supports node partitions: V_train, V_trans, V_induct.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        seed: Optional[int] = None,
    ):
        self.dims = [in_dim] + hidden_dims + [num_classes]
        self.num_classes = num_classes
        self.rng = np.random.RandomState(seed)

        # Initialize GCN weights
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for d_in, d_out in zip(self.dims[:-1], self.dims[1:]):
            limit = np.sqrt(6.0 / (d_in + d_out))
            W = self.rng.uniform(-limit, limit, (d_in, d_out))
            b = np.zeros(d_out)
            self.weights.append(W)
            self.biases.append(b)

    def forward(
        self, A_norm: np.ndarray, X: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Forward pass with cache for backpropagation.
        Returns:
            activations: list of H^{(l)}
            linears: list of Z^{(l)}
            probabilities: Y in R^{N x C} (Eq 13.20)
        """
        activations = [X]
        linears = []

        H = X
        num_layers = len(self.weights)

        for l in range(num_layers):
            W = self.weights[l]
            b = self.biases[l]
            Z = np.dot(A_norm, np.dot(H, W)) + b
            linears.append(Z)

            if l == num_layers - 1:
                # Softmax readout (Eq 13.20)
                H = softmax(Z, axis=1)
            else:
                H = relu(Z)
            activations.append(H)

        return activations, linears, activations[-1]

    def compute_loss(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
        train_mask: np.ndarray,
    ) -> float:
        """
        Cross-entropy loss over train nodes V_train (Eq 13.21):
            L = - sum_{n in V_train} sum_i t_{ni} ln(y_{ni})
        """
        eps = 1e-12
        probs_clipped = np.clip(probabilities[train_mask], eps, 1.0)
        t_train = targets[train_mask]
        loss = -np.sum(t_train * np.log(probs_clipped))
        return float(loss)

    def fit(
        self,
        A: np.ndarray,
        X: np.ndarray,
        targets: np.ndarray,
        train_mask: np.ndarray,
        epochs: int = 150,
        lr: float = 0.05,
        weight_decay: float = 1e-4,
    ) -> List[float]:
        """
        Train GCN node classifier using gradient descent.
        """
        A_norm = GraphConvolutionalNetwork.compute_normalized_adjacency(A)
        loss_history = []
        num_layers = len(self.weights)

        for _ in range(epochs):
            activations, linears, probs = self.forward(A_norm, X)
            loss = self.compute_loss(probs, targets, train_mask)
            loss_history.append(loss)

            # Backpropagation
            # Output error delta = (Y - T) on V_train, 0 elsewhere
            delta = np.zeros_like(probs)
            delta[train_mask] = probs[train_mask] - targets[train_mask]

            grad_W_list = []
            grad_b_list = []

            for l in reversed(range(num_layers)):
                H_prev = activations[l]
                grad_W = np.dot(np.dot(A_norm, H_prev).T, delta) + weight_decay * self.weights[l]
                grad_b = np.sum(delta, axis=0)

                grad_W_list.append(grad_W)
                grad_b_list.append(grad_b)

                if l > 0:
                    delta_prev = np.dot(np.dot(A_norm, delta), self.weights[l].T)
                    delta = delta_prev * d_relu(linears[l - 1])

            grad_W_list.reverse()
            grad_b_list.reverse()

            for l in range(num_layers):
                self.weights[l] -= lr * grad_W_list[l]
                self.biases[l] -= lr * grad_b_list[l]

        return loss_history

    def predict(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Predict class labels for all nodes."""
        A_norm = GraphConvolutionalNetwork.compute_normalized_adjacency(A)
        _, _, probs = self.forward(A_norm, X)
        return np.argmax(probs, axis=1)


# =============================================================================
# 6. Edge Classification / Link Prediction (Section 13.2.6)
# =============================================================================

class EdgeClassifier:
    """
    Edge Classification / Link Prediction (Section 13.2.6).
    Predicts probability of an edge between nodes n and m:
        p(n, m) = sigma(h_n^T h_m)   (Eq 13.22)
    or bilinear form:
        p(n, m) = sigma(h_n^T W_edge h_m + b)
    """

    def __init__(
        self,
        embed_dim: int,
        use_bilinear: bool = False,
        seed: Optional[int] = None,
    ):
        self.embed_dim = embed_dim
        self.use_bilinear = use_bilinear
        if use_bilinear:
            rng = np.random.RandomState(seed)
            limit = np.sqrt(6.0 / (2 * embed_dim))
            self.W_edge = rng.uniform(-limit, limit, (embed_dim, embed_dim))
            self.bias = 0.0
        else:
            self.W_edge = np.eye(embed_dim)
            self.bias = 0.0

    def predict_pair(self, h_n: np.ndarray, h_m: np.ndarray) -> float:
        """Predict edge probability p(n, m) between two node embeddings (Eq 13.22)."""
        score = np.dot(h_n, np.dot(self.W_edge, h_m)) + self.bias
        return float(sigmoid(np.array([score]))[0])

    def predict_matrix(self, H: np.ndarray) -> np.ndarray:
        """Predict edge probability matrix for all node pairs."""
        scores = np.dot(H, np.dot(self.W_edge, H.T)) + self.bias
        return sigmoid(scores)

    def binary_cross_entropy_loss(
        self,
        prob_matrix: np.ndarray,
        adj_target: np.ndarray,
        eval_mask: Optional[np.ndarray] = None,
    ) -> float:
        """Compute binary cross-entropy loss over specified edge pairs."""
        eps = 1e-12
        p = np.clip(prob_matrix, eps, 1.0 - eps)
        if eval_mask is None:
            # Exclude diagonal (self-loops)
            eval_mask = ~np.eye(adj_target.shape[0], dtype=bool)

        loss = -np.sum(
            adj_target[eval_mask] * np.log(p[eval_mask])
            + (1.0 - adj_target[eval_mask]) * np.log(1.0 - p[eval_mask])
        )
        return float(loss / np.sum(eval_mask))


# =============================================================================
# 7. Graph Classification & Readout (Section 13.2.7)
# =============================================================================

class GraphClassifier:
    """
    Graph-level Classification and Regression (Section 13.2.7).
    Aggregates node embeddings across the entire graph into a global representation:
        y = f(sum_{n in V} h_n^{(L)})   (Eq 13.23)
    followed by an MLP readout layer. Invariant to node label permutations.
    """

    def __init__(
        self,
        node_embed_dim: int,
        readout_dims: List[int],
        pool_type: str = "sum",
        activation: Callable[[np.ndarray], np.ndarray] = relu,
        seed: Optional[int] = None,
    ):
        self.pool_type = pool_type
        self.readout_mlp = MLP(
            [node_embed_dim] + readout_dims,
            activation=activation,
            seed=seed,
        )

    def pool(self, H: np.ndarray) -> np.ndarray:
        """Global graph pooling (Eq 13.23)."""
        if self.pool_type == "sum":
            return np.sum(H, axis=0)
        elif self.pool_type == "mean":
            return np.mean(H, axis=0)
        elif self.pool_type == "max":
            return np.max(H, axis=0)
        else:
            raise ValueError(f"Unknown global pool type: {self.pool_type}")

    def forward(self, H: np.ndarray) -> np.ndarray:
        """Forward pass from node embeddings H to graph-level prediction."""
        h_graph = self.pool(H)
        return self.readout_mlp.forward(h_graph)


# =============================================================================
# 8. Permutation Equivariance & Invariance Verification
# =============================================================================

def check_mpnn_permutation_equivariance(
    layer: MessagePassingLayer,
    graph: Graph,
    P: Optional[np.ndarray] = None,
    rtol: float = 1e-5,
) -> bool:
    """
    Verify layer-level permutation equivariance (Eq 13.19):
        H_permuted = P @ H_original
    where input is (P A P^T, P X).
    """
    N = graph.num_nodes
    if P is None:
        perm = np.random.RandomState(42).permutation(N)
        P = build_permutation_matrix(perm)

    # Original forward
    H_orig = layer.forward(graph)

    # Permuted graph
    A_perm = P @ graph.A @ P.T
    X_perm = P @ graph.node_features
    # Rebuild edges for permuted graph
    edges_perm = []
    for i in range(N):
        for j in range(N):
            if A_perm[i, j] > 0 and (i < j or graph.is_directed):
                edges_perm.append((i, j))

    graph_perm = Graph(
        num_nodes=N,
        edges=edges_perm,
        node_features=X_perm,
        is_directed=graph.is_directed,
    )
    H_perm = layer.forward(graph_perm)

    # Expected: H_perm == P @ H_orig
    H_expected = P @ H_orig
    return bool(np.allclose(H_perm, H_expected, rtol=rtol, atol=1e-6))


def check_graph_readout_invariance(
    classifier: GraphClassifier,
    H: np.ndarray,
    P: Optional[np.ndarray] = None,
    rtol: float = 1e-5,
) -> bool:
    """
    Verify that global readout pooling is strictly permutation invariant (Eq 13.23):
        y(P @ H) == y(H)
    """
    N = H.shape[0]
    if P is None:
        perm = np.random.RandomState(42).permutation(N)
        P = build_permutation_matrix(perm)

    y_orig = classifier.forward(H)
    y_perm = classifier.forward(P @ H)
    return bool(np.allclose(y_orig, y_perm, rtol=rtol, atol=1e-6))


# =============================================================================
# 9. Receptive Field Analysis (Figure 13.4)
# =============================================================================

def compute_receptive_field(
    A: np.ndarray,
    target_node: int,
    max_hops: int,
) -> List[Set[int]]:
    """
    Compute receptive field (set of reached nodes) for a target node across hops:
        hops[0] = {target_node}
        hops[1] = nodes reachable in <= 1 hop
        ...
        hops[k] = nodes reachable in <= k hops
    """
    N = A.shape[0]
    reached = {target_node}
    receptive_fields = [set(reached)]

    current_frontier = {target_node}
    for _ in range(max_hops):
        next_frontier = set()
        for u in current_frontier:
            neighbors = np.where(A[u] > 0)[0]
            for v in neighbors:
                if v not in reached:
                    next_frontier.add(v)
                    reached.add(v)
        current_frontier = next_frontier
        receptive_fields.append(set(reached))

    return receptive_fields


def create_figure_13_4_graph() -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]], np.ndarray]:
    """
    Canonical 11-node graph and layout from Bishop Figure 13.4.
    Returns:
        raw_nodes: List of (x, y) coordinates
        edges: List of (i, j) node pairs
        A: (11, 11) adjacency matrix
    """
    raw_nodes = [
        (139.3, 99.9),   # 0
        (105.2, 122.1),  # 1
        (68.9, 164.3),   # 2
        (146.1, 181.7),  # 3
        (116.6, 199.4),  # 4
        (39.2, 205.9),   # 5
        (89.3, 232.6),   # 6
        (55.2, 249.0),   # 7
        (132.6, 272.0),  # 8
        (100.8, 292.1),  # 9
        (68.8, 323.9),   # 10 (target node)
    ]

    edges = [
        (0, 1), (1, 2), (1, 4), (1, 6), (2, 5), (2, 7),
        (3, 4), (3, 9), (4, 9), (5, 7), (6, 7), (6, 9),
        (6, 10), (8, 9), (9, 10)
    ]

    N = len(raw_nodes)
    A = np.zeros((N, N), dtype=float)
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0

    return raw_nodes, edges, A


# =============================================================================
# 10. Publication-Quality Figure Reproductions (Figures 13.3 & 13.4)
# =============================================================================

def generate_figure_13_3(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Faithful reproduction of Figure 13.3:
    (a) 2D CNN filter as graph computation on layer l projecting to layer l+1.
    (b) Same computation expressed as a graph with central node i receiving messages.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=300)
    ax1, ax2 = axes

    # --- Subplot (a): 2D Image Convolution Filter as Graph Computation ---
    ax1.set_xlim(-1, 11)
    ax1.set_ylim(-1.5, 9.5)
    ax1.axis("off")

    dx = 0.55
    dy = 0.80
    slope = 0.65

    def get_grid_poly(u: int, v: int, x0: float, y0: float, nu: int = 1, nv: int = 1) -> List[Tuple[float, float]]:
        return [
            (x0 + u * dx, y0 + v * dy + u * dx * slope),
            (x0 + (u + nu) * dx, y0 + v * dy + (u + nu) * dx * slope),
            (x0 + (u + nu) * dx, y0 + (v + nv) * dy + (u + nu) * dx * slope),
            (x0 + u * dx, y0 + (v + nv) * dy + u * dx * slope),
        ]

    # Layer l grid: 4 columns x 8 rows
    x0_l, y0_l = 0.5, 0.4
    for u in range(4):
        for v in range(8):
            pts = get_grid_poly(u, v, x0_l, y0_l)
            poly = patches.Polygon(pts, facecolor="#fff9c4", edgecolor="#b0bec5", linewidth=0.7)
            ax1.add_patch(poly)

    # Highlight 3x3 patch on layer l (u in 0..2, v in 4..6)
    for u in range(3):
        for v in range(4, 7):
            pts = get_grid_poly(u, v, x0_l, y0_l)
            poly = patches.Polygon(pts, facecolor="#7986cb", edgecolor="none")
            ax1.add_patch(poly)

    # Center cell of 3x3 patch (u=1, v=5) is red/salmon
    pts_center = get_grid_poly(1, 5, x0_l, y0_l)
    poly_center = patches.Polygon(pts_center, facecolor="#ef9a9a", edgecolor="none")
    ax1.add_patch(poly_center)
    cx_l = x0_l + 1.5 * dx
    cy_l = y0_l + 5.5 * dy + 1.5 * dx * slope
    ax1.text(cx_l, cy_l, r"$i$", fontsize=13, fontstyle="italic", ha="center", va="center")

    # Outline of 3x3 patch
    pts_patch = get_grid_poly(0, 4, x0_l, y0_l, nu=3, nv=3)
    poly_patch_border = patches.Polygon(pts_patch, facecolor="none", edgecolor="black", linewidth=2.0)
    ax1.add_patch(poly_patch_border)

    # Outer border of layer l grid
    pts_grid_l = get_grid_poly(0, 0, x0_l, y0_l, nu=4, nv=8)
    poly_border_l = patches.Polygon(pts_grid_l, facecolor="none", edgecolor="black", linewidth=1.6)
    ax1.add_patch(poly_border_l)

    # Layer l+1 grid: 3 columns x 6 rows
    x0_r, y0_r = 5.2, 1.6
    for u in range(3):
        for v in range(6):
            pts = get_grid_poly(u, v, x0_r, y0_r)
            poly = patches.Polygon(pts, facecolor="#fff9c4", edgecolor="#b0bec5", linewidth=0.7)
            ax1.add_patch(poly)

    # Output cell in layer l+1 (u=0, v=3)
    pts_target = get_grid_poly(0, 3, x0_r, y0_r)
    poly_target = patches.Polygon(pts_target, facecolor="#ef9a9a", edgecolor="black", linewidth=2.0)
    ax1.add_patch(poly_target)
    cx_r = x0_r + 0.5 * dx
    cy_r = y0_r + 3.5 * dy + 0.5 * dx * slope
    ax1.text(cx_r, cy_r, r"$i$", fontsize=13, fontstyle="italic", ha="center", va="center")

    # Outer border of layer l+1 grid
    pts_grid_r = get_grid_poly(0, 0, x0_r, y0_r, nu=3, nv=6)
    poly_border_r = patches.Polygon(pts_grid_r, facecolor="none", edgecolor="black", linewidth=1.6)
    ax1.add_patch(poly_border_r)

    # 4 connecting projection lines
    for p1, p2 in zip(pts_patch, pts_target):
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", linewidth=0.9, zorder=2)

    # Labels below grids
    ax1.text(x0_l + 2.0 * dx, y0_l - 0.7, r"$l$", fontsize=16, ha="center", va="center")
    ax1.text(x0_r + 1.5 * dx, y0_r - 0.7, r"$l+1$", fontsize=16, ha="center", va="center")
    ax1.text(3.6, -1.3, "(a)", fontsize=13, ha="center")

    # --- Subplot (b): Same Computation Expressed as a Graph ---
    ax2.set_xlim(-3.2, 3.2)
    ax2.set_ylim(-3.2, 3.2)
    ax2.axis("off")

    # Center node i
    center_circle = patches.Circle((0, 0), 0.44, facecolor="#ffcdd2", edgecolor="#d32f2f", linewidth=2.0, zorder=5)
    ax2.add_patch(center_circle)
    ax2.text(0, 0, r"$i$", fontsize=14, fontstyle="italic", ha="center", va="center", zorder=6)

    # 8 neighbor nodes arranged symmetrically in 8 directions
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    R = 2.1
    r_node = 0.42

    for theta in angles:
        nx = R * np.cos(theta)
        ny = R * np.sin(theta)

        # Black edge connecting to center
        ax2.plot([0, nx], [0, ny], color="black", linewidth=1.5, zorder=2)

        # Red incoming message arrow parallel to edge
        perp_x = -np.sin(theta) * 0.14
        perp_y = np.cos(theta) * 0.14

        start_x = nx * 0.75 + perp_x
        start_y = ny * 0.75 + perp_y
        end_x = nx * 0.35 + perp_x
        end_y = ny * 0.35 + perp_y

        arrow = FancyArrowPatch(
            (start_x, start_y),
            (end_x, end_y),
            arrowstyle="->,head_width=3.5,head_length=5",
            color="#e53935",
            linewidth=1.8,
            zorder=3,
        )
        ax2.add_patch(arrow)

        # Neighbor circle
        circle = patches.Circle(
            (nx, ny),
            r_node,
            facecolor="#d1c4e9",
            edgecolor="#1a237e",
            linewidth=2.0,
            zorder=4,
        )
        ax2.add_patch(circle)

    ax2.text(0, -2.8, "(b)", fontsize=13, ha="center")

    plt.tight_layout()
    _save_figure(fig, "fig_13_3_convolutional_filters", save_dir)
    _save_figure(fig, "Figure_13_3", save_dir)
    return fig


def generate_figure_13_4(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Faithful reproduction of Figure 13.4:
    Schematic illustration of information flow through successive layers (l=1, 2, 3)
    showing receptive field expansion from 1 to 3 to 8 nodes.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)
    ax.set_xlim(-1, 23)
    ax.set_ylim(-1, 13)
    ax.axis("off")

    raw_nodes, edges, _ = create_figure_13_4_graph()

    # Red nodes per layer:
    # Layer 3 (right): target node 10 only (1 node)
    # Layer 2 (middle): node 10 and neighbors 6, 9 (3 nodes)
    # Layer 1 (left): nodes reached within 2 hops (8 nodes)
    plane_red_nodes = [
        {1, 3, 4, 6, 7, 8, 9, 10},  # Layer 1 (left)
        {6, 9, 10},                 # Layer 2 (middle)
        {10},                       # Layer 3 (right)
    ]
    plane_x_offsets = [0.0, 7.5, 15.0]

    scale_x = 5.0 / 141.0
    scale_y = 11.0 / 403.0

    for p_idx, (x_off, red_set) in enumerate(zip(plane_x_offsets, plane_red_nodes)):
        # Parallelogram corners of plane
        p_bl = (x_off + 0, 0)
        p_tl = (x_off + 0, (420 - 158) * scale_y)
        p_tr = (x_off + 141 * scale_x, (420 - 17) * scale_y)
        p_br = (x_off + 141 * scale_x, (420 - 279) * scale_y)

        poly = patches.Polygon(
            [p_bl, p_tl, p_tr, p_br],
            facecolor="#fffde7",
            edgecolor="black",
            linewidth=1.8,
            zorder=1,
        )
        ax.add_patch(poly)

        # Plot coordinates for nodes
        node_coords = []
        for rx, ry in raw_nodes:
            nx = x_off + (rx - 21) * scale_x
            ny = (420 - ry) * scale_y
            node_coords.append((nx, ny))

        # Draw graph edges
        for i, j in edges:
            p1 = node_coords[i]
            p2 = node_coords[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", linewidth=1.4, zorder=2)

        # Draw nodes (vertical ellipses)
        ew = 0.42
        eh = 0.95
        for idx, (nx, ny) in enumerate(node_coords):
            is_red = idx in red_set
            fc = "#ffcdd2" if is_red else "#d1c4e9"
            ec = "#e53935" if is_red else "#1a237e"

            ellipse = patches.Ellipse(
                (nx, ny),
                width=ew,
                height=eh,
                angle=0,
                facecolor=fc,
                edgecolor=ec,
                linewidth=2.0,
                zorder=3,
            )
            ax.add_patch(ellipse)

    # Black horizontal arrows between planes indicating information flow
    arrow1 = FancyArrowPatch(
        (5.3, 11.2),
        (7.2, 11.2),
        arrowstyle="->,head_width=4,head_length=6",
        color="black",
        linewidth=2.5,
        zorder=5,
    )
    ax.add_patch(arrow1)

    arrow2 = FancyArrowPatch(
        (12.8, 11.2),
        (14.7, 11.2),
        arrowstyle="->,head_width=4,head_length=6",
        color="black",
        linewidth=2.5,
        zorder=5,
    )
    ax.add_patch(arrow2)

    plt.tight_layout()
    _save_figure(fig, "fig_13_4_receptive_field_expansion", save_dir)
    _save_figure(fig, "Figure_13_4", save_dir)
    return fig


# =============================================================================
# 11. Synthetic Benchmark Graph Generators
# =============================================================================

def create_karate_club_graph() -> Tuple[Graph, np.ndarray]:
    """
    Zachary's Karate Club graph benchmark for node classification.
    34 nodes, 78 edges, 2 classes (Mr. Hi vs Officer).
    """
    edges_raw = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10),
        (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
        (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
        (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
        (3, 7), (3, 12), (3, 13),
        (4, 6), (4, 10),
        (5, 6), (5, 10), (5, 16),
        (6, 16),
        (8, 30), (8, 32), (8, 33),
        (9, 33),
        (13, 33),
        (14, 32), (14, 33),
        (15, 32), (15, 33),
        (18, 32), (18, 33),
        (19, 33),
        (20, 32), (20, 33),
        (22, 32), (22, 33),
        (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
        (24, 25), (24, 27), (24, 31),
        (25, 31),
        (26, 29), (26, 33),
        (27, 33),
        (28, 31), (28, 33),
        (29, 32), (29, 33),
        (30, 32), (30, 33),
        (31, 32), (31, 33),
        (32, 33),
    ]

    labels = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ])

    num_nodes = 34
    # Node features: identity or degree one-hot
    features = np.eye(num_nodes)

    graph = Graph(
        num_nodes=num_nodes,
        edges=edges_raw,
        node_features=features,
        is_directed=False,
    )
    return graph, labels


def create_synthetic_graph_dataset(
    num_graphs: int = 40,
    seed: int = 42,
) -> Tuple[List[Graph], np.ndarray]:
    """
    Generate synthetic graph dataset for graph-level classification (Section 13.2.7).
    Two classes of graphs:
      Class 0: Cycle / ring graphs (sparse, 2-regular)
      Class 1: Star graphs (one central hub, leaves)
    """
    rng = np.random.RandomState(seed)
    graphs = []
    labels = []

    for i in range(num_graphs):
        n_nodes = rng.randint(6, 14)
        c = i % 2
        labels.append(c)

        if c == 0:
            # Cycle graph
            edges = [(j, (j + 1) % n_nodes) for j in range(n_nodes)]
        else:
            # Star graph: node 0 is hub
            edges = [(0, j) for j in range(1, n_nodes)]

        # Node features: 4-dim random embeddings
        features = rng.randn(n_nodes, 4)
        g = Graph(num_nodes=n_nodes, edges=edges, node_features=features, is_directed=False)
        graphs.append(g)

    return graphs, np.array(labels)


# =============================================================================
# CLI Main Entrypoint
# =============================================================================

if __name__ == "__main__":
    print("Generating Figure 13.3...")
    generate_figure_13_3()

    print("Generating Figure 13.4...")
    generate_figure_13_4()

    print("All Chapter 13.2 figures generated successfully!")
