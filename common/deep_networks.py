"""
Chapter 6: Deep Neural Networks
Section 6.3: Deep Networks

Bishop & Bishop (2024), Deep Learning: Foundations and Concepts, pp. 186-194.

Covers:
- Hierarchical representations & linear regions (Section 6.3.1, Eq 6.19)
- Distributed representations vs localist representations (Section 6.3.2)
- Representation learning & embedding spaces (Section 6.3.3)
- Transfer learning & multitask learning (Section 6.3.4, Figure 6.13)
- Contrastive learning: InfoNCE, CLIP, and Supervised Contrastive (Section 6.3.5, Eq 6.20, Eq 6.21, Figure 6.14)
- General network architectures: Feedforward DAG networks (Section 6.3.6, Eq 6.22, Figure 6.15)
- Tensors & multidimensional array operations (Section 6.3.7)
"""

import math
import os
from collections import defaultdict, deque
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from common.plot_utils import save_plot, setup_style


def _get_project_root() -> str:
    """Return the absolute path to the repository root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _save_figure(fig: plt.Figure, filename: str, filepath: Optional[str] = None, save_both: bool = True) -> Tuple[str, Optional[str]]:
    """Save figure to target filepath, and/or both 6/result and root result directory."""
    if filepath:
        save_plot(fig, filepath)
    root = _get_project_root()
    path_ch6 = os.path.join(root, "6", "result", filename)
    path_root = os.path.join(root, "result", filename)
    if save_both or not filepath:
        if not filepath or os.path.abspath(filepath) != os.path.abspath(path_ch6):
            save_plot(fig, path_ch6)
        if not filepath or os.path.abspath(filepath) != os.path.abspath(path_root):
            save_plot(fig, path_root)
        return path_ch6, path_root
    return filepath, None


# ==============================================================================
# 1. Hierarchical Representations & Linear Regions (Section 6.3.1)
# ==============================================================================

def montufar_linear_regions_bound(depth: int, width: int, input_dim: int) -> int:
    """Compute the Montufar et al. (2014) lower bound on the maximum number of linear regions
    computed by a deep ReLU network with given depth and hidden layer width.
    
    A deep ReLU neural network with L hidden layers each of width M on an input of dimension D
    can divide the input space into a number of linear regions that is exponential in depth L:
        Bound = (floor(M / D))^{(L - 1) * D} * sum_{j=0}^D binom(M, j)
    
    In contrast, a shallow network (depth=1) with hidden width M_shallow can only produce
    at most sum_{j=0}^D binom(M_shallow, j) regions (polynomial in width).
    
    Args:
        depth: Number of hidden layers L (L >= 1).
        width: Width M of each hidden layer.
        input_dim: Dimension D of input space (D >= 1).
        
    Returns:
        Integer lower bound on the number of linear regions.
    """
    if depth < 1 or width < 1 or input_dim < 1:
        raise ValueError("depth, width, and input_dim must be positive integers.")
    
    base_sum = 0
    for j in range(min(input_dim, width) + 1):
        base_sum += math.comb(width, j)
        
    if depth == 1:
        return base_sum
        
    factor = width // input_dim
    if factor < 1:
        factor = 1
    multiplier = factor ** ((depth - 1) * input_dim)
    return multiplier * base_sum


def count_1d_linear_regions(
    forward_fn: Callable[[np.ndarray], np.ndarray],
    x_range: Tuple[float, float] = (-2.0, 2.0),
    num_samples: int = 4000,
    slope_tol: float = 1e-3,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Numerically detect linear pieces of a 1D scalar function f: R -> R.
    
    Evaluates discrete curvature (second derivative) to identify transition boundaries
    between affine pieces of piecewise-linear functions (e.g. ReLU networks).
    
    Args:
        forward_fn: Function mapping 1D numpy array x to 1D numpy array y.
        x_range: Evaluation interval (min_x, max_x).
        num_samples: Grid density.
        slope_tol: Relative threshold on second derivative magnitude.
        
    Returns:
        (num_regions, x_grid, y_grid): Region count and evaluated grid.
    """
    x = np.linspace(x_range[0], x_range[1], num_samples)
    y = forward_fn(x)
    dx = x[1] - x[0]
    
    dy = np.gradient(y, dx)
    d2y = np.gradient(dy, dx)
    
    max_d2y = np.max(np.abs(d2y))
    if max_d2y < 1e-12:
        return 1, x, y
        
    threshold = max_d2y * slope_tol
    kinks = np.where(np.abs(d2y) > threshold)[0]
    
    if len(kinks) == 0:
        return 1, x, y
        
    kink_clusters = 1
    for i in range(1, len(kinks)):
        if kinks[i] > kinks[i-1] + 1:
            kink_clusters += 1
            
    num_regions = kink_clusters + 1
    return num_regions, x, y


# ==============================================================================
# 2. Distributed Representations vs Localist Representations (Section 6.3.2)
# ==============================================================================

class DistributedFeatureEncoder:
    """Demonstrates the exponential capacity advantage of distributed representations
    over localist (1-of-K) representations.
    
    With M binary units:
    - Localist representation can represent M mutually exclusive states.
    - Distributed representation can represent 2^M independent feature combinations.
    """
    
    def __init__(self, attribute_names: Sequence[str]):
        """Initialize encoder with list of binary attribute names (e.g. ['glasses', 'hat', 'beard'])."""
        self.attribute_names = list(attribute_names)
        self.M = len(attribute_names)
        self.total_combinations = 2 ** self.M
        
    def encode_distributed(self, active_attributes: Sequence[str]) -> np.ndarray:
        """Encode active attributes into an M-dimensional binary feature vector {0, 1}^M."""
        vec = np.zeros(self.M, dtype=np.float64)
        for attr in active_attributes:
            if attr in self.attribute_names:
                idx = self.attribute_names.index(attr)
                vec[idx] = 1.0
        return vec
        
    def decode_distributed(self, vec: np.ndarray, threshold: float = 0.5) -> List[str]:
        """Decode binary feature vector back to active attributes."""
        vec = np.asarray(vec)
        active = []
        for i, val in enumerate(vec):
            if val >= threshold:
                active.append(self.attribute_names[i])
        return active
        
    def encode_localist(self, active_attributes: Sequence[str]) -> np.ndarray:
        """Encode attribute combination into a 1-of-(2^M) one-hot vector."""
        dist_vec = self.encode_distributed(active_attributes)
        idx = int("".join(str(int(b)) for b in dist_vec), 2)
        localist_vec = np.zeros(self.total_combinations, dtype=np.float64)
        localist_vec[idx] = 1.0
        return localist_vec


# ==============================================================================
# 3. Representation Learning & Deep MLP (Section 6.3.3)
# ==============================================================================

class DeepMLP:
    """Multi-layer Perceptron (Eq 6.19) supporting arbitrary layer dimensions,
    feature extraction at intermediate layers, and forward evaluation.
    
    Eq (6.19):
        z^{(l)} = h^{(l)}(W^{(l)} z^{(l-1)})
    where z^{(0)} = x, z^{(L)} = y.
    """
    
    def __init__(
        self,
        layer_dims: Sequence[int],
        hidden_activation: str = "relu",
        output_activation: str = "linear",
        seed: Optional[int] = None,
    ):
        """Initialize Deep MLP.
        
        Args:
            layer_dims: Dimensions of layers [D_in, M_1, M_2, ..., D_out].
            hidden_activation: 'relu', 'tanh', 'sigmoid', or 'linear'.
            output_activation: 'linear', 'sigmoid', or 'softmax'.
            seed: Optional random seed for reproducible weight initialization.
        """
        self.layer_dims = list(layer_dims)
        self.num_layers = len(layer_dims) - 1
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        
        rng = np.random.RandomState(seed)
        
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        
        for l in range(self.num_layers):
            din = self.layer_dims[l]
            dout = self.layer_dims[l + 1]
            if hidden_activation == "relu":
                std = np.sqrt(2.0 / din)
            else:
                std = np.sqrt(1.0 / din)
            W = rng.normal(0.0, std, size=(dout, din))
            b = np.zeros(dout, dtype=np.float64)
            self.weights.append(W)
            self.biases.append(b)
            
    def _activate(self, a: np.ndarray, act_name: str) -> np.ndarray:
        """Apply element-wise activation function."""
        if act_name == "relu":
            return np.maximum(0.0, a)
        elif act_name == "tanh":
            return np.tanh(a)
        elif act_name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(a, -30.0, 30.0)))
        elif act_name in ("linear", "identity"):
            return a.copy()
        elif act_name == "softmax":
            exp_a = np.exp(a - np.max(a, axis=-1, keepdims=True))
            return exp_a / np.sum(exp_a, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Unknown activation: {act_name}")
            
    def forward(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Forward pass through all layers.
        
        Returns:
            (pre_activations, activations):
            pre_activations[l] = a^{(l+1)}
            activations[l] = z^{(l)}, where activations[0] = x.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
            
        activations = [x_arr]
        pre_activations = []
        
        current_z = x_arr
        for l in range(self.num_layers):
            W = self.weights[l]
            b = self.biases[l]
            a = current_z @ W.T + b
            pre_activations.append(a)
            
            act_fn = self.output_activation if l == self.num_layers - 1 else self.hidden_activation
            current_z = self._activate(a, act_fn)
            activations.append(current_z)
            
        return pre_activations, activations
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate network output y(x)."""
        _, activations = self.forward(x)
        return activations[-1]
        
    def extract_features(self, x: np.ndarray, layer_idx: int = -2) -> np.ndarray:
        """Extract intermediate representation z^{(l)} at layer_idx.
        Default layer_idx=-2 corresponds to the penultimate hidden layer (embedding space).
        """
        _, activations = self.forward(x)
        return activations[layer_idx]


# ==============================================================================
# 4. Transfer Learning & Multitask Learning (Section 6.3.4)
# ==============================================================================

class TransferLearningClassifier:
    """Implements Transfer Learning by taking a pre-trained feature extractor,
    freezing early layers, and training a new task-specific classification head.
    
    Bishop & Bishop (2024), Section 6.3.4, Figure 6.13.
    """
    
    def __init__(self, backbone: DeepMLP, num_target_classes: int, freeze_backbone: bool = True):
        """Initialize transfer learning model.
        
        Args:
            backbone: Pre-trained DeepMLP serving as feature extractor.
            num_target_classes: Number of classes K for target task.
            freeze_backbone: If True, keep backbone weights fixed during training.
        """
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.feature_dim = backbone.layer_dims[-2]
        self.num_classes = num_target_classes
        
        rng = np.random.RandomState(42)
        std = np.sqrt(2.0 / self.feature_dim)
        self.W_head = rng.normal(0.0, std, size=(self.num_classes, self.feature_dim))
        self.b_head = np.zeros(self.num_classes, dtype=np.float64)
        
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through feature extractor and target classification head.
        
        Returns:
            (features, probs): Extracted penultimate embeddings and softmax probabilities.
        """
        features = self.backbone.extract_features(x, layer_idx=-2)
        logits = features @ self.W_head.T + self.b_head
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return features, probs
        
    def fit_head(self, x: np.ndarray, y: np.ndarray, lr: float = 0.05, epochs: int = 200, reg: float = 1e-4) -> List[float]:
        """Train the classification head on frozen extracted features using cross-entropy loss.
        
        Args:
            x: Input features (N, D).
            y: One-hot encoded class labels or 1D integer class indices (N,).
            lr: Learning rate.
            epochs: Training epochs.
            reg: L2 regularization strength.
            
        Returns:
            losses: Loss history per epoch.
        """
        N = len(x)
        features = self.backbone.extract_features(x, layer_idx=-2)
        
        if y.ndim == 1:
            y_one_hot = np.zeros((N, self.num_classes), dtype=np.float64)
            y_one_hot[np.arange(N), y.astype(int)] = 1.0
        else:
            y_one_hot = y.astype(np.float64)
            
        losses = []
        for _ in range(epochs):
            logits = features @ self.W_head.T + self.b_head
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            
            eps = 1e-12
            ce_loss = -np.mean(np.sum(y_one_hot * np.log(probs + eps), axis=1))
            reg_loss = 0.5 * reg * np.sum(self.W_head ** 2)
            total_loss = ce_loss + reg_loss
            losses.append(float(total_loss))
            
            grad_logits = (probs - y_one_hot) / N
            grad_W = grad_logits.T @ features + reg * self.W_head
            grad_b = np.sum(grad_logits, axis=0)
            
            self.W_head -= lr * grad_W
            self.b_head -= lr * grad_b
            
        return losses


# ==============================================================================
# 5. Contrastive Learning: InfoNCE, CLIP, and Supervised Contrastive (Section 6.3.5)
# ==============================================================================

def normalize_embeddings(reps: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize feature representations to lie on the unit hypersphere: ||f_w(x)|| = 1.
    
    Args:
        reps: Array of shape (..., D).
        eps: Small constant for numerical stability.
        
    Returns:
        Normalized representations of same shape.
    """
    norms = np.linalg.norm(reps, axis=-1, keepdims=True)
    return reps / np.maximum(norms, eps)


def info_nce_loss(
    anchor_rep: np.ndarray,
    pos_rep: np.ndarray,
    neg_reps: np.ndarray,
    temperature: float = 1.0,
) -> float:
    """Compute InfoNCE contrastive loss (Eq 6.20).
    
    E(w) = - ln [ exp(f_w(x)^T f_w(x^+) / tau) / ( exp(f_w(x)^T f_w(x^+) / tau) + sum_{n=1}^N exp(f_w(x)^T f_w(x^-_n) / tau) ) ]
    
    Args:
        anchor_rep: Normalized representation of anchor x, shape (D,).
        pos_rep: Normalized representation of positive example x^+, shape (D,).
        neg_reps: Normalized representations of negative examples {x^-_n}, shape (N_neg, D).
        temperature: Temperature scaling parameter tau (default 1.0 per Eq 6.20).
        
    Returns:
        Scalar InfoNCE loss value.
    """
    anchor = normalize_embeddings(np.asarray(anchor_rep, dtype=np.float64).ravel())
    pos = normalize_embeddings(np.asarray(pos_rep, dtype=np.float64).ravel())
    negs = normalize_embeddings(np.asarray(neg_reps, dtype=np.float64))
    if negs.ndim == 1:
        negs = negs.reshape(1, -1)
        
    sim_pos = float(np.dot(anchor, pos)) / temperature
    sim_negs = (negs @ anchor) / temperature
    
    all_sims = np.concatenate([[sim_pos], sim_negs])
    max_sim = np.max(all_sims)
    log_denom = max_sim + np.log(np.sum(np.exp(all_sims - max_sim)))
    
    loss = -(sim_pos - log_denom)
    return float(loss)


def batch_info_nce_loss(
    reps_a: np.ndarray,
    reps_b: np.ndarray,
    temperature: float = 0.1,
) -> float:
    """Vectorized SimCLR / NT-Xent style batch InfoNCE loss for a batch of N pairs.
    Each pair (reps_a[i], reps_b[i]) is a positive pair, while all other pairs are negatives.
    
    Args:
        reps_a: Batch representations, shape (N, D).
        reps_b: Augmented batch representations, shape (N, D).
        temperature: Temperature tau.
        
    Returns:
        Average symmetric InfoNCE loss across the batch.
    """
    N, D = reps_a.shape
    a_norm = normalize_embeddings(reps_a)
    b_norm = normalize_embeddings(reps_b)
    
    sim_matrix = (a_norm @ b_norm.T) / temperature
    pos_sim = np.diag(sim_matrix)
    
    max_row = np.max(sim_matrix, axis=1, keepdims=True)
    log_denom_a = max_row.ravel() + np.log(np.sum(np.exp(sim_matrix - max_row), axis=1))
    loss_a = -np.mean(pos_sim - log_denom_a)
    
    max_col = np.max(sim_matrix, axis=0, keepdims=True)
    log_denom_b = max_col.ravel() + np.log(np.sum(np.exp(sim_matrix - max_col), axis=0))
    loss_b = -np.mean(pos_sim - log_denom_b)
    
    return float(0.5 * (loss_a + loss_b))


def clip_loss(
    image_reps: np.ndarray,
    text_reps: np.ndarray,
    temperature: float = 1.0,
) -> float:
    """Compute symmetric multimodal CLIP contrastive loss (Eq 6.21).
    
    Bishop & Bishop (2024), Eq (6.21):
        E(w) = - 0.5 * ln [ exp(f_w(x^+)^T g_theta(y^+) / tau) / ( exp(f_w(x^+)^T g_theta(y^+) / tau) + sum_{n=1}^N exp(f_w(x^-_n)^T g_theta(y^+) / tau) ) ]
               - 0.5 * ln [ exp(f_w(x^+)^T g_theta(y^+) / tau) / ( exp(f_w(x^+)^T g_theta(y^+) / tau) + sum_{m=1}^M exp(f_w(x^+)^T g_theta(y^-_m) / tau) ) ]
               
    Args:
        image_reps: Normalized image embeddings for batch of size B, shape (B, D).
        text_reps: Normalized text embeddings for batch of size B, shape (B, D).
        temperature: Temperature parameter tau.
        
    Returns:
        Average CLIP loss over the batch.
    """
    img_norm = normalize_embeddings(np.asarray(image_reps, dtype=np.float64))
    txt_norm = normalize_embeddings(np.asarray(text_reps, dtype=np.float64))
    
    sim_matrix = (img_norm @ txt_norm.T) / temperature
    pos_sim = np.diag(sim_matrix)
    
    max_row = np.max(sim_matrix, axis=1, keepdims=True)
    log_denom_i2t = max_row.ravel() + np.log(np.sum(np.exp(sim_matrix - max_row), axis=1))
    loss_i2t = -np.mean(pos_sim - log_denom_i2t)
    
    max_col = np.max(sim_matrix, axis=0, keepdims=True)
    log_denom_t2i = max_col.ravel() + np.log(np.sum(np.exp(sim_matrix - max_col), axis=0))
    loss_t2i = -np.mean(pos_sim - log_denom_t2i)
    
    return float(0.5 * (loss_i2t + loss_t2i))


def supervised_contrastive_loss(
    reps: np.ndarray,
    labels: np.ndarray,
    temperature: float = 0.1,
) -> float:
    """Supervised Contrastive Loss (Khosla et al., 2020; Bishop & Bishop 2024, Section 6.3.5).
    Pulls representations of the same class together while pushing representations of different classes apart.
    
    Args:
        reps: Normalized feature embeddings, shape (N, D).
        labels: Class labels, shape (N,).
        temperature: Temperature parameter tau.
        
    Returns:
        Scalar supervised contrastive loss.
    """
    reps_norm = normalize_embeddings(reps)
    N = len(labels)
    labels = np.asarray(labels)
    
    sim_matrix = (reps_norm @ reps_norm.T) / temperature
    
    logits_mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(logits_mask, False)
    
    pos_mask = (labels[:, None] == labels[None, :]) & logits_mask
    
    max_sim = np.max(sim_matrix * logits_mask - 1e9 * (~logits_mask), axis=1, keepdims=True)
    exp_sim = np.exp(sim_matrix - max_sim) * logits_mask
    log_denom = max_sim.ravel() + np.log(np.sum(exp_sim, axis=1) + 1e-12)
    
    loss = 0.0
    valid_anchors = 0
    for i in range(N):
        num_pos = np.sum(pos_mask[i])
        if num_pos > 0:
            log_prob_pos = sim_matrix[i, pos_mask[i]] - log_denom[i]
            loss -= np.sum(log_prob_pos) / num_pos
            valid_anchors += 1
            
    if valid_anchors == 0:
        return 0.0
    return float(loss / valid_anchors)


# ==============================================================================
# 6. General Feedforward DAG Networks (Section 6.3.6)
# ==============================================================================

class FeedForwardDAG:
    """General Feedforward Neural Network defined by an arbitrary Directed Acyclic Graph (DAG).
    
    Bishop & Bishop (2024), Section 6.3.6, Eq (6.22):
        z_k = h( sum_{j in A(k)} w_{kj} z_j + b_k )
    where A(k) denotes the set of ancestor nodes sending connections to node k.
    
    Supports:
    - Arbitrary skip connections and topologies
    - Cycle detection and topological sorting
    - Exact forward propagation
    - Exact analytical backpropagation and gradient checking
    """
    
    def __init__(self, node_names: Sequence[str], input_nodes: Sequence[str], output_nodes: Sequence[str]):
        """Initialize DAG network topology.
        
        Args:
            node_names: All node identifiers in the graph.
            input_nodes: Subset of node identifiers acting as inputs x.
            output_nodes: Subset of node identifiers acting as outputs y.
        """
        self.nodes = list(node_names)
        self.input_nodes = list(input_nodes)
        self.output_nodes = list(output_nodes)
        
        self.outgoing: Dict[str, List[str]] = defaultdict(list)
        self.ancestors: Dict[str, List[str]] = defaultdict(list)
        
        self.weights: Dict[Tuple[str, str], float] = {}
        self.biases: Dict[str, float] = {k: 0.0 for k in self.nodes if k not in self.input_nodes}
        
        self.activations: Dict[str, str] = {k: "tanh" for k in self.nodes if k not in self.input_nodes}
        for out in self.output_nodes:
            self.activations[out] = "linear"
            
        self._topological_order: Optional[List[str]] = None
        
    def add_edge(self, parent_j: str, child_k: str, weight: float = 1.0) -> None:
        """Add directed edge parent_j -> child_k with weight w_{kj}."""
        if parent_j not in self.nodes or child_k not in self.nodes:
            raise ValueError(f"Both nodes must exist in the network: {parent_j} -> {child_k}")
        if child_k in self.input_nodes:
            raise ValueError(f"Input node '{child_k}' cannot have incoming edges.")
            
        self.outgoing[parent_j].append(child_k)
        self.ancestors[child_k].append(parent_j)
        self.weights[(parent_j, child_k)] = float(weight)
        self._topological_order = None
        
    def set_bias(self, node: str, bias: float) -> None:
        """Set bias b_k for node k."""
        if node in self.input_nodes:
            raise ValueError("Input nodes do not have biases.")
        self.biases[node] = float(bias)
        
    def set_activation(self, node: str, act_name: str) -> None:
        """Set activation function for node k ('tanh', 'relu', 'sigmoid', 'linear')."""
        self.activations[node] = act_name
        
    def topological_sort(self) -> List[str]:
        """Compute topological ordering of nodes using Kahn's algorithm.
        Raises ValueError if a cycle is detected.
        """
        if self._topological_order is not None:
            return self._topological_order
            
        in_degree = {k: len(self.ancestors[k]) for k in self.nodes}
        queue = deque([k for k in self.nodes if in_degree[k] == 0])
        order = []
        
        while queue:
            curr = queue.popleft()
            order.append(curr)
            for child in self.outgoing[curr]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
                    
        if len(order) != len(self.nodes):
            raise ValueError("Cycle detected in network graph! A feedforward network must be acyclic.")
            
        self._topological_order = order
        return order
        
    def _apply_activation(self, a: np.ndarray, act_name: str) -> np.ndarray:
        """Evaluate activation function."""
        if act_name == "tanh":
            return np.tanh(a)
        elif act_name == "relu":
            return np.maximum(0.0, a)
        elif act_name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(a, -30.0, 30.0)))
        elif act_name == "linear":
            return a.copy()
        else:
            raise ValueError(f"Unknown activation: {act_name}")
            
    def _apply_activation_deriv(self, a: np.ndarray, z: np.ndarray, act_name: str) -> np.ndarray:
        """Evaluate activation function derivative h'(a)."""
        if act_name == "tanh":
            return 1.0 - z ** 2
        elif act_name == "relu":
            return (a > 0.0).astype(np.float64)
        elif act_name == "sigmoid":
            return z * (1.0 - z)
        elif act_name == "linear":
            return np.ones_like(a)
        else:
            raise ValueError(f"Unknown activation: {act_name}")
            
    def forward(self, input_dict: Dict[str, Union[float, np.ndarray]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Evaluate forward propagation (Eq 6.22) in topological order.
        
        Args:
            input_dict: Dictionary mapping input node names to values/arrays.
            
        Returns:
            (pre_activations, node_values):
            pre_activations[k] = a_k
            node_values[k] = z_k
        """
        order = self.topological_sort()
        
        first_val = next(iter(input_dict.values()))
        first_arr = np.asarray(first_val, dtype=np.float64)
        batch_shape = first_arr.shape
        
        node_values: Dict[str, np.ndarray] = {}
        pre_activations: Dict[str, np.ndarray] = {}
        
        for inp in self.input_nodes:
            if inp not in input_dict:
                raise ValueError(f"Missing input for node '{inp}'")
            node_values[inp] = np.asarray(input_dict[inp], dtype=np.float64)
            
        for node in order:
            if node in self.input_nodes:
                continue
                
            a_k = np.full(batch_shape, self.biases[node], dtype=np.float64)
            for parent in self.ancestors[node]:
                w_kj = self.weights[(parent, node)]
                z_j = node_values[parent]
                a_k = a_k + w_kj * z_j
                
            pre_activations[node] = a_k
            act_name = self.activations[node]
            node_values[node] = self._apply_activation(a_k, act_name)
            
        return pre_activations, node_values
        
    def predict(self, input_dict: Dict[str, Union[float, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Convenience function returning only output node values."""
        _, node_values = self.forward(input_dict)
        return {out: node_values[out] for out in self.output_nodes}
        
    def backward(
        self,
        input_dict: Dict[str, Union[float, np.ndarray]],
        target_dict: Dict[str, Union[float, np.ndarray]],
    ) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
        """Compute exact analytical gradients for sum-of-squares error:
            E = 0.5 * sum_{k in Outputs} (y_k - t_k)^2
            
        Returns:
            (grad_weights, grad_biases): Gradients with respect to weights and biases.
        """
        pre_acts, node_vals = self.forward(input_dict)
        order = self.topological_sort()
        
        deltas: Dict[str, np.ndarray] = {}
        
        for node in reversed(order):
            if node in self.input_nodes:
                continue
                
            act_name = self.activations[node]
            h_prime = self._apply_activation_deriv(pre_acts[node], node_vals[node], act_name)
            
            if node in self.output_nodes:
                target = np.asarray(target_dict[node], dtype=np.float64)
                dE_dz = node_vals[node] - target
            else:
                dE_dz = np.zeros_like(node_vals[node])
                
            for child in self.outgoing[node]:
                w_child_node = self.weights[(node, child)]
                delta_child = deltas[child]
                dE_dz = dE_dz + w_child_node * delta_child
                
            deltas[node] = dE_dz * h_prime
            
        grad_weights: Dict[Tuple[str, str], float] = {}
        grad_biases: Dict[str, float] = {}
        
        for (parent, child), _ in self.weights.items():
            z_parent = node_vals[parent]
            delta_child = deltas[child]
            grad_weights[(parent, child)] = float(np.sum(delta_child * z_parent))
            
        for node, _ in self.biases.items():
            grad_biases[node] = float(np.sum(deltas[node]))
            
        return grad_weights, grad_biases


def build_fig_6_15_network() -> FeedForwardDAG:
    """Build the exact general feed-forward DAG neural network shown in Figure 6.15.
    
    Nodes:
        Inputs: x1, x2
        Hidden: z1, z2, z3
        Outputs: y1, y2
        
    Directed Connections:
        x1 -> z1
        x1 -> z2 (skip connection)
        x2 -> z1
        x2 -> y2 (skip connection directly from input to output)
        z1 -> z2
        z1 -> z3
        z2 -> y1
        z2 -> y2
        z3 -> y2
    """
    nodes = ["x1", "x2", "z1", "z2", "z3", "y1", "y2"]
    inputs = ["x1", "x2"]
    outputs = ["y1", "y2"]
    
    dag = FeedForwardDAG(node_names=nodes, input_nodes=inputs, output_nodes=outputs)
    
    edges = [
        ("x1", "z1", 0.8),
        ("x1", "z2", 0.5),
        ("x2", "z1", -0.7),
        ("x2", "y2", 0.4),
        ("z1", "z2", 1.2),
        ("z1", "z3", -0.9),
        ("z2", "y1", 1.0),
        ("z2", "y2", -0.6),
        ("z3", "y2", 0.7),
    ]
    for p, c, w in edges:
        dag.add_edge(p, c, weight=w)
        
    for node in ["z1", "z2", "z3", "y1", "y2"]:
        dag.set_bias(node, 0.1)
        
    return dag


# ==============================================================================
# 7. Tensors & Multidimensional Arrays (Section 6.3.7)
# ==============================================================================

def create_image_tensor_dataset(
    num_images: int = 10,
    height: int = 32,
    width: int = 32,
    channels: int = 3,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Construct a 4-dimensional image tensor X of shape (num_images, height, width, channels)
    corresponding to X_{ijkn} in Bishop & Bishop (2024), Section 6.3.7.
    """
    rng = np.random.RandomState(seed)
    return rng.uniform(0.0, 1.0, size=(num_images, height, width, channels))


def tensor_contraction_example(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Demonstrate tensor contraction using Einstein summation (np.einsum).
    
    Maps an input image tensor X of shape (N, H, W, C_in) and kernel tensor W of shape (C_out, C_in)
    to an output tensor of shape (N, H, W, C_out) via 1x1 convolution / linear channel mixing:
        Y_{n, i, j, c_out} = sum_{c_in} X_{n, i, j, c_in} * W_{c_out, c_in}
    """
    return np.einsum("nhwc,oc->nhwo", X, W)


# ==============================================================================
# 8. Publication-Quality Figure Generators (Figures 6.13, 6.14, 6.15)
# ==============================================================================

def _load_asset_images() -> Dict[str, np.ndarray]:
    """Load extracted textbook images from data/ch6_sec3_images.npz or generate fallback."""
    root = _get_project_root()
    npz_path = os.path.join(root, "data", "ch6_sec3_images.npz")
    
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        return {
            "cat_fig13": data["cat_fig13"],
            "lesion_fig13": data["lesion_fig13"],
            "cat_anchor": data["cat_anchor"],
            "cat_augmented": data["cat_augmented"],
            "bike_neg": data["bike_neg"],
            "cat_class2": data["cat_class2"],
        }
        
    rng = np.random.RandomState(42)
    cat = np.full((64, 64, 3), [220, 140, 80], dtype=np.uint8)
    cat[15:30, 20:30] = [30, 180, 50]
    cat[15:30, 34:44] = [30, 180, 50]
    lesion = np.full((64, 64, 3), [240, 210, 210], dtype=np.uint8)
    lesion[20:45, 20:45] = [140, 70, 60]
    bike = np.full((64, 64, 3), [180, 180, 190], dtype=np.uint8)
    return {
        "cat_fig13": cat,
        "lesion_fig13": lesion,
        "cat_anchor": cat,
        "cat_augmented": cat,
        "bike_neg": bike,
        "cat_class2": cat,
    }


def generate_figure_6_13(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """Generate and save Figure 6.13: Schematic illustration of transfer learning.
    
    (a) Network trained on task with abundant data (object classification of natural images).
    (b) Early layers (red) copied from task 1, final layers (blue) retrained on new task (skin lesion).
    """
    setup_style()
    assets = _load_asset_images()
    cat_img = assets["cat_fig13"]
    lesion_img = assets["lesion_fig13"]
    
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 5.8)
    ax.axis("off")
    
    red_color = "#f2786d"
    blue_color = "#a2a9f5"
    edge_color = "#222222"
    
    def draw_network_row(y_center: float, img: np.ndarray, is_transfer: bool = False, label: str = "(a)") -> None:
        img_w, img_h = 1.05, 1.05
        img_x, img_y = 0.45, y_center - img_h / 2
        ax.imshow(img, extent=[img_x, img_x + img_w, img_y, img_y + img_h], aspect="auto", zorder=3)
        rect = patches.Rectangle((img_x, img_y), img_w, img_h, linewidth=1.2, edgecolor=edge_color, facecolor="none", zorder=4)
        ax.add_patch(rect)
        
        curr_x = img_x + img_w
        arrow_len = 0.45
        ax.annotate("", xy=(curr_x + arrow_len, y_center), xytext=(curr_x, y_center),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3, mutation_scale=12))
        curr_x += arrow_len
        
        block_w = 0.46
        gap = 0.46
        heights = [2.05, 2.05, 2.05, 1.45, 1.45, 1.45]
        
        for i in range(6):
            h = heights[i]
            color = blue_color if (is_transfer and i >= 4) else red_color
            box = patches.FancyBboxPatch(
                (curr_x, y_center - h / 2), block_w, h,
                boxstyle="round,pad=0.03,rounding_size=0.15",
                facecolor=color, edgecolor=edge_color, linewidth=1.4, zorder=3
            )
            ax.add_patch(box)
            curr_x += block_w
            if i < 5:
                ax.annotate("", xy=(curr_x + gap, y_center), xytext=(curr_x, y_center),
                            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3, mutation_scale=12))
                curr_x += gap
                
        ax.annotate("", xy=(curr_x + 0.42, y_center), xytext=(curr_x, y_center),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3, mutation_scale=12))
        curr_x += 0.46
        
        ax.text(curr_x, y_center, "}", fontsize=42, fontfamily="serif", ha="center", va="center")
        curr_x += 0.30
        
        if not is_transfer:
            labels = ["tree", "cat", "dog", "$\\vdots$"]
            bar_lens = [0.25, 0.70, 0.25, None]
            ys = [y_center + 0.65, y_center + 0.25, y_center - 0.15, y_center - 0.60]
            for l, bl, y in zip(labels, bar_lens, ys):
                ax.text(curr_x, y, l, fontsize=12.5, va="center", ha="left")
                if bl is not None:
                    bar_rect = patches.Rectangle(
                        (curr_x + 0.80, y - 0.05), bl, 0.12,
                        facecolor="blue", edgecolor="black", linewidth=1.0
                    )
                    ax.add_patch(bar_rect)
        else:
            labels = ["cancer", "normal"]
            bar_lens = [0.15, 0.55]
            ys = [y_center + 0.30, y_center - 0.30]
            for l, bl, y in zip(labels, bar_lens, ys):
                ax.text(curr_x, y, l, fontsize=12.5, va="center", ha="left")
                bar_rect = patches.Rectangle(
                    (curr_x + 0.95, y - 0.06), bl, 0.14,
                    facecolor="blue", edgecolor="black", linewidth=1.0
                )
                ax.add_patch(bar_rect)
                
        ax.text(4.35, y_center - 1.40, label, fontsize=13.5, ha="center", va="center")
        
    draw_network_row(4.3, cat_img, is_transfer=False, label="(a)")
    draw_network_row(1.7, lesion_img, is_transfer=True, label="(b)")
    
    plt.tight_layout()
    _save_figure(fig, "fig_6_13_transfer_learning.png", filepath=filepath, save_both=save_both)
    return fig


def generate_figure_6_14(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """Generate and save Figure 6.14: Three contrastive learning paradigms.
    
    (a) Instance discrimination (anchor and augmented version, mapped to unit hypersphere).
    (b) Supervised contrastive learning (images of same class as positive pair).
    (c) CLIP model (image and text caption as positive pair).
    """
    setup_style()
    assets = _load_asset_images()
    cat_anchor = assets["cat_anchor"]
    cat_augmented = assets["cat_augmented"]
    bike_neg = assets["bike_neg"]
    cat_class2 = assets["cat_class2"]
    
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.2))
    
    for ax in axes:
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-3.1, 2.4)
        ax.set_aspect("equal")
        ax.axis("off")
        
    sphere_color = "#b2bcdc"
    point_color = "#f28b82"
    
    def draw_sphere(ax: plt.Axes) -> None:
        circle = patches.Circle((0, 0.7), 1.35, facecolor=sphere_color, edgecolor="#8f9bbd", linewidth=1.5, zorder=2)
        ax.add_patch(circle)
        inner = patches.Circle((-0.2, 0.9), 1.1, facecolor="#c5cee8", edgecolor="none", alpha=0.5, zorder=2)
        ax.add_patch(inner)
        
    p_anchor = np.array([-0.65, 0.95])
    p_pos = np.array([-0.25, 0.15])
    p_neg = np.array([0.70, 0.65])
    
    for i, ax in enumerate(axes):
        draw_sphere(ax)
        
        diff_pos = p_pos - p_anchor
        dist_pos = np.linalg.norm(diff_pos)
        u_pos = diff_pos / dist_pos
        start_pos = p_anchor + u_pos * 0.13
        end_pos = p_pos - u_pos * 0.13
        ax.annotate("", xy=end_pos, xytext=start_pos,
                    arrowprops=dict(arrowstyle="<->", color="#00c853", lw=2.4, mutation_scale=14), zorder=4)
                    
        diff_neg = p_neg - p_anchor
        dist_neg = np.linalg.norm(diff_neg)
        u_neg = diff_neg / dist_neg
        start_neg = p_anchor + u_neg * 0.13
        end_neg = p_neg - u_neg * 0.13
        ax.annotate("", xy=end_neg, xytext=start_neg,
                    arrowprops=dict(arrowstyle="<->", color="#d50000", lw=2.4, mutation_scale=14), zorder=4)
                    
        for p in [p_anchor, p_pos, p_neg]:
            pt = patches.Circle(p, 0.10, facecolor=point_color, edgecolor="black", linewidth=1.2, zorder=5)
            ax.add_patch(pt)
            
        if i in [0, 1]:
            ax.text(p_anchor[0] - 0.05, p_anchor[1] + 0.17, "$f_w(x)$", fontsize=12, ha="center", va="bottom", zorder=6)
            ax.text(p_pos[0] + 0.20, p_pos[1] - 0.05, "$f_w(x^+)$", fontsize=12, ha="left", va="center", zorder=6)
            ax.text(p_neg[0], p_neg[1] + 0.17, "$f_w(x^-)$", fontsize=12, ha="center", va="bottom", zorder=6)
        else:
            ax.text(p_anchor[0] - 0.05, p_anchor[1] + 0.17, "$f_w(x^+)$", fontsize=12, ha="center", va="bottom", zorder=6)
            ax.text(p_pos[0] + 0.20, p_pos[1] - 0.05, "$g_\\theta(y^+)$", fontsize=12, ha="left", va="center", zorder=6)
            ax.text(p_neg[0], p_neg[1] + 0.17, "$g_\\theta(y^-)$", fontsize=12, ha="center", va="bottom", zorder=6)
            
    img_box = 0.82
    y_img = -2.0
    xs = [-1.45, -0.41, 0.63]
    
    # (a)
    axes[0].imshow(cat_anchor, extent=[xs[0], xs[0] + img_box, y_img, y_img + img_box], zorder=3)
    axes[0].imshow(cat_augmented, extent=[xs[1], xs[1] + img_box, y_img, y_img + img_box], zorder=3)
    axes[0].imshow(bike_neg, extent=[xs[2], xs[2] + img_box, y_img, y_img + img_box], zorder=3)
    for x in xs:
        axes[0].add_patch(patches.Rectangle((x, y_img), img_box, img_box, fill=False, edgecolor="black", lw=1.0, zorder=4))
    axes[0].text(xs[0] + img_box/2, y_img - 0.28, "$x$", fontsize=13, ha="center")
    axes[0].text(xs[1] + img_box/2, y_img - 0.28, "$x^+$", fontsize=13, ha="center")
    axes[0].text(xs[2] + img_box/2, y_img - 0.28, "$x^-$", fontsize=13, ha="center")
    axes[0].text(0, -2.75, "(a)", fontsize=14, ha="center")
    
    # (b)
    axes[1].imshow(cat_anchor, extent=[xs[0], xs[0] + img_box, y_img, y_img + img_box], zorder=3)
    axes[1].imshow(cat_class2, extent=[xs[1], xs[1] + img_box, y_img, y_img + img_box], zorder=3)
    axes[1].imshow(bike_neg, extent=[xs[2], xs[2] + img_box, y_img, y_img + img_box], zorder=3)
    for x in xs:
        axes[1].add_patch(patches.Rectangle((x, y_img), img_box, img_box, fill=False, edgecolor="black", lw=1.0, zorder=4))
    axes[1].text(xs[0] + img_box/2, y_img - 0.28, "$x$", fontsize=13, ha="center")
    axes[1].text(xs[1] + img_box/2, y_img - 0.28, "$x^+$", fontsize=13, ha="center")
    axes[1].text(xs[2] + img_box/2, y_img - 0.28, "$x^-$", fontsize=13, ha="center")
    axes[1].text(0, -2.75, "(b)", fontsize=14, ha="center")
    
    # (c)
    axes[2].imshow(cat_anchor, extent=[xs[0], xs[0] + img_box, y_img, y_img + img_box], zorder=3)
    axes[2].add_patch(patches.Rectangle((xs[0], y_img), img_box, img_box, fill=False, edgecolor="black", lw=1.0, zorder=4))
    axes[2].text(xs[1] + img_box/2, y_img + img_box/2, "‘a ginger\ncat sat\non a wall’", fontsize=10.5, ha="center", va="center", style="italic")
    axes[2].text(xs[2] + img_box/2, y_img + img_box/2, "‘a bike\nleaning\non a\nbrick wall’", fontsize=10.5, ha="center", va="center", style="italic")
    axes[2].text(xs[0] + img_box/2, y_img - 0.28, "$x^+$", fontsize=13, ha="center")
    axes[2].text(xs[1] + img_box/2, y_img - 0.28, "$y^+$", fontsize=13, ha="center")
    axes[2].text(xs[2] + img_box/2, y_img - 0.28, "$y^-$", fontsize=13, ha="center")
    axes[2].text(0, -2.75, "(c)", fontsize=14, ha="center")
    
    for ax in axes:
        ax.annotate("", xy=p_anchor + np.array([-0.06, -0.06]), xytext=(xs[0] + img_box/2, y_img + img_box),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, connectionstyle="arc3,rad=-0.20", mutation_scale=12), zorder=3)
        ax.annotate("", xy=p_pos + np.array([-0.04, -0.08]), xytext=(xs[1] + img_box/2, y_img + img_box),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, connectionstyle="arc3,rad=0.15", mutation_scale=12), zorder=3)
        ax.annotate("", xy=p_neg + np.array([0.06, -0.06]), xytext=(xs[2] + img_box/2, y_img + img_box),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, connectionstyle="arc3,rad=-0.25", mutation_scale=12), zorder=3)
                    
    plt.tight_layout()
    _save_figure(fig, "fig_6_14_contrastive_learning.png", filepath=filepath, save_both=save_both)
    return fig


def generate_figure_6_15(filepath: Optional[str] = None, save_both: bool = True) -> plt.Figure:
    """Generate and save Figure 6.15: General feed-forward DAG neural network topology.
    
    Demonstrates Eq (6.22) with arbitrary directed connections, including skip connections
    from inputs to intermediate layers and directly to outputs.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.set_xlim(-0.2, 7.3)
    ax.set_ylim(-0.9, 3.2)
    ax.set_aspect("equal")
    ax.axis("off")
    
    coords = {
        "x1": np.array([0.8, 2.2]),
        "x2": np.array([0.8, 0.8]),
        "z1": np.array([2.7, 1.5]),
        "z2": np.array([4.6, 2.2]),
        "z3": np.array([4.6, 0.8]),
        "y1": np.array([6.5, 2.2]),
        "y2": np.array([6.5, 0.8]),
    }
    
    r = 0.32
    node_fill = "#dce3f9"
    node_edge = "#1a237e"
    
    def draw_arrow(p_from: np.ndarray, p_to: np.ndarray, rad: float = 0.0) -> None:
        diff = p_to - p_from
        dist = np.linalg.norm(diff)
        u = diff / dist
        if rad == 0.0:
            start = p_from + u * r
            end = p_to - u * r
            ax.annotate("", xy=end, xytext=start,
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=14), zorder=2)
        else:
            start = p_from + np.array([r * 0.7, -r * 0.7])
            end = p_to + np.array([-r * 0.7, -r * 0.7])
            ax.annotate("", xy=end, xytext=start,
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, connectionstyle=f"arc3,rad={rad}", mutation_scale=14), zorder=2)
                        
    draw_arrow(coords["x1"], coords["z1"])
    draw_arrow(coords["x1"], coords["z2"])
    draw_arrow(coords["x2"], coords["z1"])
    draw_arrow(coords["z1"], coords["z2"])
    draw_arrow(coords["z1"], coords["z3"])
    draw_arrow(coords["z2"], coords["y1"])
    draw_arrow(coords["z2"], coords["y2"])
    draw_arrow(coords["z3"], coords["y2"])
    draw_arrow(coords["x2"], coords["y2"], rad=0.35)
    
    labels = {
        "x1": "$x_1$", "x2": "$x_2$",
        "z1": "$z_1$", "z2": "$z_2$", "z3": "$z_3$",
        "y1": "$y_1$", "y2": "$y_2$"
    }
    
    for name, p in coords.items():
        circle = patches.Circle(p, r, facecolor=node_fill, edgecolor=node_edge, linewidth=2.0, zorder=4)
        ax.add_patch(circle)
        ax.text(p[0], p[1], labels[name], fontsize=13, ha="center", va="center", color=node_edge, zorder=5)
        
    ax.text(coords["x1"][0], coords["x1"][1] + 0.55, "inputs", fontsize=13, ha="center", va="bottom")
    ax.text(coords["y1"][0], coords["y1"][1] + 0.55, "outputs", fontsize=13, ha="center", va="bottom")
    
    plt.tight_layout()
    _save_figure(fig, "fig_6_15_general_dag_network.png", filepath=filepath, save_both=save_both)
    return fig


def generate_all_figures() -> None:
    """Generate and save all figures for Section 6.3."""
    print("Generating Figure 6.13...")
    generate_figure_6_13()
    print("Generating Figure 6.14...")
    generate_figure_6_14()
    print("Generating Figure 6.15...")
    generate_figure_6_15()
    print("All Section 6.3 figures generated successfully.")


if __name__ == "__main__":
    generate_all_figures()
