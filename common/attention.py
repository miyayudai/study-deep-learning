"""
common/attention.py
===================
Section 12.1: Attention
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements:
1. Core Attention Mechanisms (Sections 12.1.1 - 12.1.7):
   - Attention Weights & Softmax normalization (Eqs 12.2 - 12.5).
   - Raw dot-product self-attention (Eq 12.6).
   - Learnable Query, Key, Value linear transformations (Eqs 12.10 - 12.12).
   - Scaled Dot-Product Self-Attention (Eq 12.14, Algorithm 12.1).
   - Multi-Head Attention (Eqs 12.15 - 12.19, Algorithm 12.2).
   - Layer Normalization (Ba et al., 2016) and Position-wise Feed-Forward MLP.
   - Complete Transformer Layer with Post-LN & Pre-LN architectures (Eqs 12.20 - 12.23, Algorithm 12.3).
2. Positional Encodings (Section 12.1.9):
   - Sinusoidal positional encoding (Eq 12.25) with relative translation invariance.
3. High-Resolution Figure Reproductions (Figures 12.1 - 12.10):
   - Figure 12.1: Contextual attention schematic for 'bank'.
   - Figure 12.2: Attention weights heatmap / bipartite connection graph.
   - Figure 12.3: Data matrix X (N tokens x D features).
   - Figure 12.4: Matrix multiplication Q K^T determining attention logits.
   - Figure 12.5: Matrix multiplication Y = A V computing output tokens.
   - Figure 12.6: Scaled dot-product self-attention block diagram.
   - Figure 12.7: Multi-head attention network architecture.
   - Figure 12.8: Information flow in multi-head attention layer.
   - Figure 12.9: Complete Transformer layer architecture (Add & Norm, MHA, MLP).
   - Figure 12.10: Sinusoidal positional encoding waveforms and 2D heatmap.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .plot_utils import setup_style, save_plot


def _save_figure(fig: plt.Figure, filename_base: str, save_dir: Optional[str] = None) -> None:
    """Save figure to Chapter 12 result directory and root result directory."""
    if save_dir is not None:
        save_plot(fig, os.path.join(save_dir, f"{filename_base}.png"))
        return

    repo_root = Path(__file__).resolve().parent.parent
    dir_ch12 = repo_root / "12" / "result"
    dir_root = repo_root / "result"
    dir_ch12.mkdir(parents=True, exist_ok=True)
    dir_root.mkdir(parents=True, exist_ok=True)

    save_plot(fig, str(dir_ch12 / f"{filename_base}.png"))
    save_plot(fig, str(dir_root / f"{filename_base}.png"))


# =============================================================================
# 1. Mathematical Algorithms & Components
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along specified axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class ScaledDotProductAttention:
    """
    Scaled Dot-Product Self-Attention (Section 12.1.5, Algorithm 12.1, Eq 12.14).
    Attention(Q, K, V) = Softmax(Q K^T / sqrt(D_k)) V
    """

    def __init__(self, d_k: int) -> None:
        self.d_k = d_k
        self.scale = 1.0 / np.sqrt(d_k)

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Q: (N, D_k)
        K: (M, D_k)
        V: (M, D_v)
        mask: Optional (N, M) boolean mask where True means masked out (-inf)
        Returns:
            Y: (N, D_v)
            A: (N, M) attention weights matrix
        """
        # Q K^T: (N, M)
        scores = (Q @ K.T) * self.scale
        if mask is not None:
            scores = np.where(mask, -1e9, scores)

        A = softmax(scores, axis=-1)  # Softmax along keys dimension
        Y = A @ V                     # (N, D_v)
        return Y, A


class MultiHeadAttention:
    """
    Multi-Head Attention (Section 12.1.6, Algorithm 12.2, Eqs 12.15 - 12.19).
    H_h = Attention(X W_Q^(h), X W_K^(h), X W_V^(h))
    Y = Concat[H_1, ..., H_H] W_O
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k if d_k is not None else d_model // num_heads
        self.d_v = d_v if d_v is not None else d_model // num_heads

        rng = np.random.default_rng(seed)
        # Separate projection matrices for each head
        self.W_Q = rng.normal(0, 0.02, size=(num_heads, d_model, self.d_k))
        self.W_K = rng.normal(0, 0.02, size=(num_heads, d_model, self.d_k))
        self.W_V = rng.normal(0, 0.02, size=(num_heads, d_model, self.d_v))
        # Final output projection
        self.W_O = rng.normal(0, 0.02, size=(num_heads * self.d_v, d_model))

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(
        self,
        X: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        X: (N, d_model)
        Returns:
            Y: (N, d_model)
            all_A: (num_heads, N, N) attention weight matrices
        """
        N = X.shape[0]
        head_outputs = []
        head_attentions = []

        for h in range(self.num_heads):
            Q_h = X @ self.W_Q[h]  # (N, d_k)
            K_h = X @ self.W_K[h]  # (N, d_k)
            V_h = X @ self.W_V[h]  # (N, d_v)

            H_h, A_h = self.attention.forward(Q_h, K_h, V_h, mask=mask)
            head_outputs.append(H_h)
            head_attentions.append(A_h)

        # Concatenate across heads: (N, num_heads * d_v)
        H_concat = np.concatenate(head_outputs, axis=-1)
        # Linear projection back to d_model: (N, d_model)
        Y = H_concat @ self.W_O
        all_A = np.stack(head_attentions, axis=0)
        return Y, all_A


class LayerNorm:
    """
    Layer Normalization (Ba, Kiros, & Hinton, 2016; Section 12.1.7).
    LN(x) = (x - mu) / sqrt(var + eps) * gamma + beta
    Normalizes across the feature dimension (D) independently for each token row n.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model, dtype=np.float64)
        self.beta = np.zeros(d_model, dtype=np.float64)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # X: (N, D)
        mu = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        X_norm = (X - mu) / np.sqrt(var + self.eps)
        return X_norm * self.gamma + self.beta


class TransformerMLP:
    """
    Position-wise Feed-Forward Network / MLP (Section 12.1.7).
    MLP(z) = ReLU(z W_1 + b_1) W_2 + b_2
    Applied identically to each token row n.
    """

    def __init__(self, d_model: int, d_ff: int, seed: Optional[int] = None) -> None:
        self.d_model = d_model
        self.d_ff = d_ff
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.02, size=(d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.normal(0, 0.02, size=(d_ff, d_model))
        self.b2 = np.zeros(d_model)

    def forward(self, Z: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0, Z @ self.W1 + self.b1)  # ReLU
        return hidden @ self.W2 + self.b2


class TransformerLayer:
    """
    One full Transformer Layer (Section 12.1.7, Figure 12.9, Algorithm 12.3).
    Supports Post-LN (standard textbook presentation) and Pre-LN.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        pre_norm: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.d_model = d_model
        self.pre_norm = pre_norm
        d_ff = d_ff if d_ff is not None else 4 * d_model

        self.mha = MultiHeadAttention(d_model, num_heads, seed=seed)
        self.ln1 = LayerNorm(d_model)
        self.mlp = TransformerMLP(d_model, d_ff, seed=seed)
        self.ln2 = LayerNorm(d_model)

    def forward(
        self,
        X: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            X_out: (N, d_model)
            A: (num_heads, N, N) attention weights
        """
        if not self.pre_norm:
            # Post-LN (Algorithm 12.3, Eqs 12.20, 12.22)
            Y, A = self.mha.forward(X, mask=mask)
            Z = self.ln1.forward(Y + X)
            mlp_out = self.mlp.forward(Z)
            X_out = self.ln2.forward(mlp_out + Z)
        else:
            # Pre-LN (Eqs 12.21, 12.23)
            X_norm1 = self.ln1.forward(X)
            Y, A = self.mha.forward(X_norm1, mask=mask)
            Z = Y + X
            Z_norm2 = self.ln2.forward(Z)
            mlp_out = self.mlp.forward(Z_norm2)
            X_out = mlp_out + Z

        return X_out, A


def sinusoidal_positional_encoding(
    seq_len: int,
    d_model: int,
    wavelength_base: float = 10000.0,
) -> np.ndarray:
    """
    Sinusoidal Positional Encoding (Section 12.1.9, Eq 12.25).
    r_{n, 2k}   = sin(n / L^{2k / D})
    r_{n, 2k+1} = cos(n / L^{2k / D})
    where n = 0, ..., seq_len - 1, and 2k, 2k+1 are dimension indices.
    """
    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    positions = np.arange(seq_len)[:, np.newaxis]  # (N, 1)
    dim_indices = np.arange(0, d_model, 2)         # (D/2,)

    # div_term = L^{2k / D}
    div_term = wavelength_base ** (dim_indices / d_model)

    pe[:, 0::2] = np.sin(positions / div_term)
    if d_model % 2 == 1:
        pe[:, 1::2] = np.cos(positions / div_term[:-1])
    else:
        pe[:, 1::2] = np.cos(positions / div_term)

    return pe


# =============================================================================
# 2. High-Resolution Figure Reproductions (Figures 12.1 - 12.10)
# =============================================================================

def generate_figure_12_1(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.1: Schematic illustration of attention in which the interpretation
    of the word 'bank' is influenced by 'river' and 'swam'.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')
    ax.set_title("Figure 12.1: Attention Context Mechanism ('bank' influenced by 'river' and 'swam')",
                 fontsize=11.5, fontweight='bold', pad=12)

    sentence = ["I", "swam", "across", "the", "river", "to", "get", "to", "the", "other", "bank"]
    N = len(sentence)

    # Word positions
    x_coords = np.linspace(0.2, 10.8, N)
    y_words = 0.5
    y_target = 1.8

    # Draw bottom words
    for i, word in enumerate(sentence):
        is_key = word in ["swam", "river"]
        is_bank = (word == "bank")
        color = '#c0392b' if is_key else ('#2980b9' if is_bank else '#2c3e50')
        fontweight = 'bold' if (is_key or is_bank) else 'normal'
        ax.text(x_coords[i], y_words, word, ha='center', va='center',
                fontsize=11.0, color=color, fontweight=fontweight)

    # Draw target output representation at top above 'bank'
    bank_idx = N - 1
    target_pos = (x_coords[bank_idx], y_target)
    ax.text(target_pos[0], target_pos[1], "bank\n(riverbank interpretation)",
            ha='center', va='center', fontsize=10.0, color='#2980b9', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ebf5fb', edgecolor='#2980b9', lw=1.5))

    # Attention connections to 'bank' representation
    # Weights for each word to bank
    weights = [0.02, 0.40, 0.03, 0.01, 0.45, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03]
    for i, w in enumerate(weights):
        start = (x_coords[i], y_words + 0.2)
        end = (target_pos[0] - 0.2 + (i - bank_idx) * 0.03, target_pos[1] - 0.35)
        lw = 0.8 + 6.0 * w
        alpha = 0.2 + 0.8 * (w / max(weights))
        edgecolor = '#c0392b' if sentence[i] in ["swam", "river"] else '#7f8c8d'
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=edgecolor, lw=lw, alpha=alpha,
                                    mutation_scale=10 + 10 * w))

    plt.tight_layout()
    _save_figure(fig, "Figure_12_1", save_dir)
    return fig


def generate_figure_12_2(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.2: Example of learned attention weights (heatmap and token-to-token connections).
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 5.0), gridspec_kw={'width_ratios': [1.1, 1.0]})

    tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"]
    N = len(tokens)

    # Synthetic realistic attention matrix focused on coreference resolution ('it' -> 'animal')
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 0.4  # Self attention
        if i > 0:
            A[i, i - 1] += 0.25  # Local context
        if i < N - 1:
            A[i, i + 1] += 0.15
    # Coreference: 'it' (idx 7) attends heavily to 'animal' (idx 1)
    A[7] = [0.02, 0.65, 0.03, 0.05, 0.02, 0.08, 0.05, 0.08, 0.01, 0.00, 0.01]
    # Normalize rows
    A = A / np.sum(A, axis=-1, keepdims=True)

    # Heatmap
    im = ax1.imshow(A, cmap='Blues', vmin=0.0, vmax=0.7)
    ax1.set_xticks(range(N))
    ax1.set_yticks(range(N))
    ax1.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9.0)
    ax1.set_yticklabels(tokens, fontsize=9.0)
    ax1.set_title("(a) Attention Weight Matrix $A_{nm}$", fontsize=11.0, fontweight='bold')
    ax1.set_xlabel("Key tokens $m$", fontsize=10.0)
    ax1.set_ylabel("Query tokens $n$", fontsize=10.0)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Bipartite connection graph focusing on 'it'
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(-0.5, N + 0.5)
    ax2.axis('off')
    ax2.set_title("(b) Attention Flow for query 'it'", fontsize=11.0, fontweight='bold')

    y_pos = np.arange(N)[::-1]
    for i in range(N):
        ax2.text(0.2, y_pos[i], tokens[i], ha='right', va='center', fontsize=9.5,
                 fontweight='bold' if i == 7 else 'normal',
                 color='#c0392b' if i == 7 else '#2c3e50')
        ax2.text(2.8, y_pos[i], tokens[i], ha='left', va='center', fontsize=9.5,
                 fontweight='bold' if i == 1 else 'normal',
                 color='#27ae60' if i == 1 else '#2c3e50')

    # Draw lines from query 'it' (y_pos[7]) to all keys
    it_y = y_pos[7]
    for m in range(N):
        w = A[7, m]
        key_y = y_pos[m]
        color = '#e74c3c' if m == 1 else '#3498db'
        ax2.plot([0.3, 2.7], [it_y, key_y], color=color, lw=0.5 + 8.0 * w, alpha=0.2 + 0.8 * (w / 0.65))

    ax2.text(0.2, N, "Query ($n$)", ha='right', fontsize=10.0, fontweight='bold', color='#7f8c8d')
    ax2.text(2.8, N, "Key ($m$)", ha='left', fontsize=10.0, fontweight='bold', color='#7f8c8d')

    plt.tight_layout()
    _save_figure(fig, "Figure_12_2", save_dir)
    return fig


def generate_figure_12_3(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.3: Data matrix X of dimension N x D, in which row n represents x_n^T.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.set_xlim(-1.0, 5.0)
    ax.set_ylim(-1.0, 4.0)
    ax.axis('off')
    ax.set_title("Figure 12.3: Structure of the Data Matrix $X$ ($N \\times D$)",
                 fontsize=11.5, fontweight='bold', pad=12)

    # Draw matrix rectangle
    rect = patches.Rectangle((0.5, 0.5), 3.0, 2.5, linewidth=2.0,
                             edgecolor='#2c3e50', facecolor='#f8f9fa')
    ax.add_patch(rect)

    # Highlight row n
    row_n = patches.Rectangle((0.5, 1.6), 3.0, 0.35, linewidth=1.5,
                              edgecolor='#2980b9', facecolor='#d4e6f1')
    ax.add_patch(row_n)
    ax.text(2.0, 1.775, "$\\mathbf{x}_n^T$", ha='center', va='center',
            fontsize=11.0, fontweight='bold', color='#1b4f72')

    # Label Matrix X
    ax.text(2.0, 0.9, "$X$", ha='center', va='center',
            fontsize=22.0, fontweight='bold', color='#7f8c8d', alpha=0.5)

    # Dimension arrows
    # Width D (features)
    ax.annotate("", xy=(3.5, 3.2), xytext=(0.5, 3.2),
                arrowprops=dict(arrowstyle="<->", color='#2c3e50', lw=1.5))
    ax.text(2.0, 3.4, "$D$ (features)", ha='center', fontsize=10.0, fontweight='bold')

    # Height N (tokens)
    ax.annotate("", xy=(-0.2, 0.5), xytext=(-0.2, 3.0),
                arrowprops=dict(arrowstyle="<->", color='#2c3e50', lw=1.5))
    ax.text(-0.4, 1.75, "$N$ (tokens)", va='center', ha='right', fontsize=10.0,
            fontweight='bold', rotation=90)

    plt.tight_layout()
    _save_figure(fig, "Figure_12_3", save_dir)
    return fig


def generate_figure_12_4(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.4: Evaluation of matrix Q K^T determining attention coefficients.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')
    ax.set_title("Figure 12.4: Matrix Product $Q K^T$ ($N \\times N$ Attention Logits)",
                 fontsize=11.5, fontweight='bold', pad=12)

    # Q: (N, D_k)
    q_rect = patches.Rectangle((0.5, 0.5), 1.6, 2.8, linewidth=1.8, edgecolor='#2980b9', facecolor='#ebf5fb')
    ax.add_patch(q_rect)
    ax.text(1.3, 1.9, "$Q$\n$(N \\times D_k)$", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#1b4f72')

    ax.text(2.6, 1.9, "$\\times$", ha='center', va='center', fontsize=18, fontweight='bold', color='#2c3e50')

    # K^T: (D_k, N)
    kt_rect = patches.Rectangle((3.2, 1.3), 2.8, 1.6, linewidth=1.8, edgecolor='#27ae60', facecolor='#eafaf1')
    ax.add_patch(kt_rect)
    ax.text(4.6, 2.1, "$K^T$\n$(D_k \\times N)$", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#145a32')

    ax.text(6.5, 1.9, "$=$", ha='center', va='center', fontsize=20, fontweight='bold', color='#2c3e50')

    # Q K^T: (N, N)
    prod_rect = patches.Rectangle((7.1, 0.5), 2.0, 2.8, linewidth=2.0, edgecolor='#8e44ad', facecolor='#f4ecf7')
    ax.add_patch(prod_rect)
    ax.text(8.1, 1.9, "$Q K^T$\n$(N \\times N)$", ha='center', va='center', fontsize=11.0, fontweight='bold', color='#512e5f')

    plt.tight_layout()
    _save_figure(fig, "Figure_12_4", save_dir)
    return fig


def generate_figure_12_5(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.5: Evaluation of matrix product Y = A V.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')
    ax.set_title("Figure 12.5: Matrix Product $Y = A V$ ($N \\times D_v$ Output Tokens)",
                 fontsize=11.5, fontweight='bold', pad=12)

    # A: (N, N)
    a_rect = patches.Rectangle((0.5, 0.5), 2.0, 2.8, linewidth=1.8, edgecolor='#8e44ad', facecolor='#f4ecf7')
    ax.add_patch(a_rect)
    ax.text(1.5, 1.9, "$A$\n$(N \\times N)$", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#512e5f')

    ax.text(3.0, 1.9, "$\\times$", ha='center', va='center', fontsize=18, fontweight='bold', color='#2c3e50')

    # V: (N, D_v)
    v_rect = patches.Rectangle((3.6, 0.5), 1.6, 2.8, linewidth=1.8, edgecolor='#e67e22', facecolor='#fef5e7')
    ax.add_patch(v_rect)
    ax.text(4.4, 1.9, "$V$\n$(N \\times D_v)$", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#7e5109')

    ax.text(5.8, 1.9, "$=$", ha='center', va='center', fontsize=20, fontweight='bold', color='#2c3e50')

    # Y: (N, D_v)
    y_rect = patches.Rectangle((6.5, 0.5), 1.6, 2.8, linewidth=2.0, edgecolor='#c0392b', facecolor='#fadbd8')
    ax.add_patch(y_rect)
    ax.text(7.3, 1.9, "$Y$\n$(N \\times D_v)$", ha='center', va='center', fontsize=11.0, fontweight='bold', color='#78281f')

    plt.tight_layout()
    _save_figure(fig, "Figure_12_5", save_dir)
    return fig


def generate_figure_12_6(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.6: Structure of scaled dot-product self-attention layer (Algorithm 12.1).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 7.5)
    ax.axis('off')
    ax.set_title("Figure 12.6: Scaled Dot-Product Self-Attention Layer",
                 fontsize=11.5, fontweight='bold', pad=12)

    def draw_box(x, y, w, h, text, color='#34495e', bgcolor='#f2f4f4', fontsize=9.5):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                                      linewidth=1.5, edgecolor=color, facecolor=bgcolor)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=color)

    # Input X
    draw_box(2.0, 0.2, 1.0, 0.5, "$X$", color='#2c3e50', bgcolor='#eaecee', fontsize=11.0)

    # Linear projections W_Q, W_K, W_V
    draw_box(0.5, 1.4, 0.9, 0.5, "$W_Q$", color='#2980b9', bgcolor='#ebf5fb')
    draw_box(2.0, 1.4, 0.9, 0.5, "$W_K$", color='#27ae60', bgcolor='#eafaf1')
    draw_box(3.5, 1.4, 0.9, 0.5, "$W_V$", color='#e67e22', bgcolor='#fef5e7')

    # Connecting arrows from X
    ax.annotate("", xy=(0.95, 1.4), xytext=(2.3, 0.7), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))
    ax.annotate("", xy=(2.45, 1.4), xytext=(2.5, 0.7), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))
    ax.annotate("", xy=(3.95, 1.4), xytext=(2.7, 0.7), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    # MatMul (Q, K^T)
    draw_box(1.0, 2.5, 1.8, 0.5, "MatMul ($Q K^T$)", color='#8e44ad', bgcolor='#f4ecf7')
    ax.annotate("", xy=(1.5, 2.5), xytext=(0.95, 1.9), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))
    ax.annotate("", xy=(2.3, 2.5), xytext=(2.45, 1.9), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    # Scale 1/sqrt(D_k)
    draw_box(1.0, 3.4, 1.8, 0.5, "Scale ($1 / \\sqrt{D_k}$)", color='#8e44ad', bgcolor='#f4ecf7')
    ax.annotate("", xy=(1.9, 3.4), xytext=(1.9, 3.0), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    # Mask (optional) / Softmax
    draw_box(1.0, 4.3, 1.8, 0.5, "Softmax ($A$)", color='#8e44ad', bgcolor='#f4ecf7')
    ax.annotate("", xy=(1.9, 4.3), xytext=(1.9, 3.9), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    # MatMul with V: (A @ V)
    draw_box(2.0, 5.3, 1.8, 0.5, "MatMul ($A V$)", color='#c0392b', bgcolor='#fadbd8')
    ax.annotate("", xy=(2.4, 5.3), xytext=(1.9, 4.8), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))
    # V bypass up to MatMul
    ax.plot([3.95, 3.95, 3.4], [1.9, 5.55, 5.55], lw=1.3, color='#2c3e50')
    ax.annotate("", xy=(3.4, 5.55), xytext=(3.6, 5.55), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    # Output Y
    draw_box(2.4, 6.4, 1.0, 0.5, "$Y$", color='#2c3e50', bgcolor='#eaecee', fontsize=11.0)
    ax.annotate("", xy=(2.9, 6.4), xytext=(2.9, 5.8), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    plt.tight_layout()
    _save_figure(fig, "Figure_12_6", save_dir)
    return fig


def generate_figure_12_7(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.7: Multi-head attention architecture: H_1..H_H concatenated and multiplied by W_O.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 3.5)
    ax.axis('off')
    ax.set_title("Figure 12.7: Multi-Head Attention Head Aggregation & Projection",
                 fontsize=11.5, fontweight='bold', pad=12)

    # Concat block: H1, H2, ..., HH
    ax.text(0.3, 1.5, "Concat", ha='center', va='center', fontsize=10.0, fontweight='bold', color='#7f8c8d')

    # Draw heads side-by-side
    heads = ["$H_1$", "$H_2$", "$\\dots$", "$H_H$"]
    for i, h_lbl in enumerate(heads):
        x = 1.0 + i * 0.9
        rect = patches.Rectangle((x, 0.5), 0.7, 2.0, linewidth=1.5,
                                 edgecolor='#2980b9', facecolor='#ebf5fb')
        ax.add_patch(rect)
        ax.text(x + 0.35, 1.5, h_lbl, ha='center', va='center', fontsize=10.0, fontweight='bold')

    ax.text(2.6, 0.1, "$N \\times H D_v$", ha='center', fontsize=9.5, color='#7f8c8d')

    ax.text(4.9, 1.5, "$\\times$", ha='center', va='center', fontsize=18, fontweight='bold')

    # W_O matrix
    w_o_rect = patches.Rectangle((5.4, 0.5), 1.5, 2.0, linewidth=1.8,
                                 edgecolor='#27ae60', facecolor='#eafaf1')
    ax.add_patch(w_o_rect)
    ax.text(6.15, 1.5, "$W^{(o)}$\n$(H D_v \\times D)$", ha='center', va='center',
            fontsize=10.0, fontweight='bold', color='#145a32')

    ax.text(7.3, 1.5, "$=$", ha='center', va='center', fontsize=20, fontweight='bold')

    # Y matrix
    y_rect = patches.Rectangle((7.8, 0.5), 1.2, 2.0, linewidth=2.0,
                               edgecolor='#c0392b', facecolor='#fadbd8')
    ax.add_patch(y_rect)
    ax.text(8.4, 1.5, "$Y$\n$(N \\times D)$", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color='#78281f')

    plt.tight_layout()
    _save_figure(fig, "Figure_12_7", save_dir)
    return fig


def generate_figure_12_8(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.8: Information flow in a multi-head attention layer.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')
    ax.set_title("Figure 12.8: Information Flow in Multi-Head Attention Layer",
                 fontsize=11.5, fontweight='bold', pad=12)

    # Input X
    rect_x = patches.FancyBboxPatch((3.0, 0.2), 1.5, 0.45, boxstyle="round,pad=0.15",
                                    linewidth=1.5, edgecolor='#2c3e50', facecolor='#eaecee')
    ax.add_patch(rect_x)
    ax.text(3.75, 0.425, "$X$", ha='center', va='center', fontsize=11.0, fontweight='bold')

    # Parallel Heads
    head_x = [1.0, 2.7, 4.8, 6.2]
    head_labels = ["Head 1", "Head 2", "$\\dots$", "Head $H$"]
    for x, lbl in zip(head_x, head_labels):
        rect = patches.FancyBboxPatch((x - 0.55, 1.4), 1.1, 0.7, boxstyle="round,pad=0.15",
                                      linewidth=1.5, edgecolor='#2980b9', facecolor='#ebf5fb')
        ax.add_patch(rect)
        ax.text(x, 1.75, lbl, ha='center', va='center', fontsize=9.0, fontweight='bold')
        # Arrow from X
        ax.annotate("", xy=(x, 1.4), xytext=(3.75, 0.65),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color='#2c3e50'))

    # Concat
    rect_cat = patches.FancyBboxPatch((1.5, 2.7), 4.5, 0.45, boxstyle="round,pad=0.15",
                                      linewidth=1.5, edgecolor='#8e44ad', facecolor='#f4ecf7')
    ax.add_patch(rect_cat)
    ax.text(3.75, 2.925, "Concat", ha='center', va='center', fontsize=10.0, fontweight='bold')

    for x in head_x:
        ax.annotate("", xy=(x, 2.7), xytext=(x, 2.1),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color='#2c3e50'))

    # Linear W_O
    rect_lin = patches.FancyBboxPatch((2.75, 3.4), 2.0, 0.45, boxstyle="round,pad=0.15",
                                      linewidth=1.5, edgecolor='#27ae60', facecolor='#eafaf1')
    ax.add_patch(rect_lin)
    ax.text(3.75, 3.625, "Linear ($W^{(o)}$)", ha='center', va='center', fontsize=9.5, fontweight='bold')
    ax.annotate("", xy=(3.75, 3.4), xytext=(3.75, 3.15),
                arrowprops=dict(arrowstyle="->", lw=1.2, color='#2c3e50'))

    # Output Y
    ax.text(3.75, 4.25, "$Y$", ha='center', va='center', fontsize=11.5, fontweight='bold', color='#c0392b')
    ax.annotate("", xy=(3.75, 4.1), xytext=(3.75, 3.85),
                arrowprops=dict(arrowstyle="->", lw=1.4, color='#c0392b'))

    plt.tight_layout()
    _save_figure(fig, "Figure_12_8", save_dir)
    return fig


def generate_figure_12_9(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.9: Transformer Layer Architecture (Add & Norm, MHA, MLP).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.0, 7.5))
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 8.5)
    ax.axis('off')
    ax.set_title("Figure 12.9: Complete Transformer Layer Architecture",
                 fontsize=11.5, fontweight='bold', pad=12)

    def draw_box(x, y, w, h, text, color='#2c3e50', bgcolor='#f8f9fa'):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                                      linewidth=1.6, edgecolor=color, facecolor=bgcolor)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=color)

    # Input X
    draw_box(2.0, 0.2, 1.2, 0.45, "$X$", color='#2c3e50', bgcolor='#eaecee')

    # Multi-head attention
    draw_box(1.5, 1.5, 2.2, 0.6, "Multi-Head\nSelf-Attention", color='#2980b9', bgcolor='#ebf5fb')
    ax.annotate("", xy=(2.6, 1.5), xytext=(2.6, 0.65), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    # Residual skip around MHA
    ax.plot([1.2, 1.2, 2.6], [0.425, 2.7, 2.7], lw=1.3, color='#7f8c8d')
    ax.annotate("", xy=(2.6, 2.7), xytext=(2.4, 2.7), arrowprops=dict(arrowstyle="->", lw=1.3, color='#7f8c8d'))

    # Add & Norm 1
    draw_box(1.7, 2.7, 1.8, 0.5, "Add & Norm", color='#8e44ad', bgcolor='#f4ecf7')
    ax.annotate("", xy=(2.6, 2.7), xytext=(2.6, 2.1), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    # Intermediate state Z
    ax.text(2.6, 3.65, "$Z$", ha='center', va='center', fontsize=11.0, fontweight='bold', color='#1b4f72')
    ax.annotate("", xy=(2.6, 4.0), xytext=(2.6, 3.2), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    # Feed-Forward / MLP
    draw_box(1.7, 4.0, 1.8, 0.6, "Feed Forward\n(MLP)", color='#27ae60', bgcolor='#eafaf1')

    # Residual skip around MLP
    ax.plot([1.2, 1.2, 2.6], [3.65, 5.2, 5.2], lw=1.3, color='#7f8c8d')
    ax.annotate("", xy=(2.6, 5.2), xytext=(2.4, 5.2), arrowprops=dict(arrowstyle="->", lw=1.3, color='#7f8c8d'))

    # Add & Norm 2
    draw_box(1.7, 5.2, 1.8, 0.5, "Add & Norm", color='#8e44ad', bgcolor='#f4ecf7')
    ax.annotate("", xy=(2.6, 5.2), xytext=(2.6, 4.6), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    # Output \tilde{X}
    draw_box(2.0, 6.7, 1.2, 0.45, "$\\widetilde{X}$", color='#c0392b', bgcolor='#fadbd8')
    ax.annotate("", xy=(2.6, 6.7), xytext=(2.6, 5.7), arrowprops=dict(arrowstyle="->", lw=1.4, color='#c0392b'))

    plt.tight_layout()
    _save_figure(fig, "Figure_12_9", save_dir)
    return fig


def generate_figure_12_10(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.10: Positional Encodings:
    (a) Sinusoidal waveforms of increasing wavelength.
    (b) 2D heatmap of position-encoding vectors for D=100, L=30, N=200.
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 5.0), gridspec_kw={'width_ratios': [1.1, 1.0]})

    # (a) Waveforms
    positions = np.linspace(0, 100, 500)
    wavelengths = [10.0, 20.0, 40.0, 80.0]
    colors = ['#c0392b', '#e67e22', '#27ae60', '#2980b9']

    for i, (wl, col) in enumerate(zip(wavelengths, colors)):
        ax1.plot(np.sin(2 * np.pi * positions / wl), positions, color=col,
                 lw=1.5, label=f"$r_{{{2*i+1}}}$ (wl={wl})")

    # Mark two sample positions n and m
    n_pos, m_pos = 35.0, 75.0
    ax1.axhline(n_pos, color='#7f8c8d', linestyle='--', lw=1.2)
    ax1.axhline(m_pos, color='#7f8c8d', linestyle='--', lw=1.2)
    ax1.text(-1.1, n_pos, "$n$", va='center', ha='right', fontsize=11.0, fontweight='bold')
    ax1.text(-1.1, m_pos, "$m$", va='center', ha='right', fontsize=11.0, fontweight='bold')

    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(0, 100)
    ax1.invert_yaxis()  # Top to bottom for sequence position
    ax1.set_xlabel("Embedding dimension component value", fontsize=10.0)
    ax1.set_ylabel("Sequence position", fontsize=10.0)
    ax1.set_title("(a) Sinusoidal Positional Encoding Functions", fontsize=11.0, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8.5)

    # (b) Heatmap for D=100, L=30, N=200 per Bishop text
    N_tokens = 200
    D_features = 100
    L_base = 30.0

    pe = sinusoidal_positional_encoding(N_tokens, D_features, wavelength_base=L_base)
    im = ax2.imshow(pe, cmap='viridis', aspect='auto', extent=[0, D_features, N_tokens, 0])
    ax2.set_xlabel("Embedding dimension ($i$)", fontsize=10.0)
    ax2.set_ylabel("Sequence position ($n$)", fontsize=10.0)
    ax2.set_title(f"(b) Position Vectors Heatmap ($D={D_features}, L={int(L_base)}, N={N_tokens}$)",
                  fontsize=11.0, fontweight='bold')
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    _save_figure(fig, "Figure_12_10", save_dir)
    return fig


if __name__ == "__main__":
    for i in range(1, 11):
        func_name = f"generate_figure_12_{i}"
        func = globals()[func_name]
        fig = func()
        plt.close(fig)
    print("All 10 figures for Section 12.1 generated successfully!")
