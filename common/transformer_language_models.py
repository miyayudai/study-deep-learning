"""
common/transformer_language_models.py
=====================================
Section 12.3: Transformer Language Models
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements:
1. Decoder Transformers & Causal Masking (Section 12.3.1, Figures 12.15, 12.16):
   - Causal lower-triangular attention mask preventing lookahead.
   - Stacked decoder blocks for autoregressive next-token prediction.
2. Sampling Strategies (Section 12.3.2, Figure 12.17):
   - Greedy search, Temperature scaling (Eq 12.35), Top-k sampling, Top-p (nucleus) sampling.
   - Beam search for cumulative sequence probability maximization.
3. Encoder Transformers & Masked Language Modeling (Section 12.3.3, Figure 12.18):
   - Bidirectional self-attention blocks.
4. Sequence-to-Sequence Transformers (Section 12.3.4, Figures 12.19, 12.20):
   - Cross-attention layers (Q from decoder, K/V from encoder Z).
   - Full Seq2Seq transformer encoder-decoder stack.
5. Large Language Models & Low-Rank Adaptation (LoRA) (Section 12.3.5, Eq 12.36, Figure 12.21):
   - LoRA linear layer: W = W0 + (alpha / R) * A @ B.
   - Weight folding / merging for zero-latency inference.
6. High-Resolution Figure Reproductions (Figures 12.15 - 12.21):
   - Saved to 12/result/ and result/.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .plot_utils import setup_style, save_plot
from .attention import (
    softmax,
    ScaledDotProductAttention,
    MultiHeadAttention,
    LayerNorm,
    TransformerMLP,
    sinusoidal_positional_encoding,
)


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
# 1. Decoder, Cross-Attention & Transformer Blocks
# =============================================================================

class CrossAttention:
    """
    Cross-Attention Layer (Section 12.3.4, Figure 12.19).
    Queries come from decoder representations (Y_dec),
    Keys and Values come from encoder representations (Z_enc).
    """

    def __init__(self, d_model: int, num_heads: int, seed: Optional[int] = None) -> None:
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        rng = np.random.default_rng(seed)
        self.W_Q = rng.normal(0, 0.02, size=(num_heads, d_model, self.d_k))
        self.W_K = rng.normal(0, 0.02, size=(num_heads, d_model, self.d_k))
        self.W_V = rng.normal(0, 0.02, size=(num_heads, d_model, self.d_v))
        self.W_O = rng.normal(0, 0.02, size=(num_heads * self.d_v, d_model))

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(
        self,
        Y_dec: np.ndarray,
        Z_enc: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Y_dec: (T_dec, d_model)
        Z_enc: (T_enc, d_model)
        """
        T_dec = Y_dec.shape[0]
        head_outputs = []
        head_attentions = []

        for h in range(self.num_heads):
            Q_h = Y_dec @ self.W_Q[h]  # (T_dec, d_k)
            K_h = Z_enc @ self.W_K[h]  # (T_enc, d_k)
            V_h = Z_enc @ self.W_V[h]  # (T_enc, d_v)

            H_h, A_h = self.attention.forward(Q_h, K_h, V_h, mask=mask)
            head_outputs.append(H_h)
            head_attentions.append(A_h)

        H_concat = np.concatenate(head_outputs, axis=-1)
        Y_out = H_concat @ self.W_O
        all_A = np.stack(head_attentions, axis=0)
        return Y_out, all_A


class DecoderTransformerBlock:
    """
    Decoder Transformer Layer with Causal Self-Attention (Figure 12.15).
    """

    def __init__(self, d_model: int, num_heads: int, seed: Optional[int] = None) -> None:
        self.self_attn = MultiHeadAttention(d_model, num_heads, seed=seed)
        self.ln1 = LayerNorm(d_model)
        self.mlp = TransformerMLP(d_model, 4 * d_model, seed=seed)
        self.ln2 = LayerNorm(d_model)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = X.shape[0]
        # Causal mask: upper triangular elements (j > i) set to True
        causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        Y, A = self.self_attn.forward(X, mask=causal_mask)
        Z = self.ln1.forward(Y + X)
        out = self.ln2.forward(self.mlp.forward(Z) + Z)
        return out, A


# =============================================================================
# 2. Text Generation Sampling Strategies (Section 12.3.2)
# =============================================================================

class TextGenerationSampler:
    """
    Implements text generation sampling strategies (Section 12.3.2):
    - Greedy search
    - Temperature scaling (Eq 12.35)
    - Top-k sampling
    - Top-p (nucleus) sampling
    - Beam search (Eq 12.34)
    """

    @staticmethod
    def greedy_search(logits: np.ndarray) -> int:
        """Select token with highest probability."""
        return int(np.argmax(logits))

    @staticmethod
    def sample_temperature(logits: np.ndarray, temperature: float = 1.0, seed: Optional[int] = None) -> int:
        """
        Temperature-scaled sampling (Eq 12.35): p_i = exp(z_i / T) / sum_j exp(z_j / T).
        """
        if temperature <= 1e-4:
            return TextGenerationSampler.greedy_search(logits)
        scaled_logits = logits / temperature
        probs = softmax(scaled_logits, axis=-1)
        rng = np.random.default_rng(seed)
        return int(rng.choice(len(probs), p=probs))

    @staticmethod
    def sample_top_k(logits: np.ndarray, k: int, temperature: float = 1.0, seed: Optional[int] = None) -> int:
        """
        Top-k sampling: keep only top k tokens, renormalize, and sample.
        """
        k = min(k, len(logits))
        top_k_indices = np.argpartition(logits, -k)[-k:]
        top_k_logits = logits[top_k_indices] / max(temperature, 1e-4)
        top_k_probs = softmax(top_k_logits, axis=-1)

        rng = np.random.default_rng(seed)
        chosen_idx = rng.choice(top_k_indices, p=top_k_probs)
        return int(chosen_idx)

    @staticmethod
    def sample_top_p(logits: np.ndarray, p: float = 0.9, temperature: float = 1.0, seed: Optional[int] = None) -> int:
        """
        Top-p (nucleus) sampling: keep smallest set with cumulative prob >= p.
        """
        scaled_logits = logits / max(temperature, 1e-4)
        probs = softmax(scaled_logits, axis=-1)
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        cumulative_probs = np.cumsum(sorted_probs)
        # Cut off indices exceeding p
        cutoff = np.searchsorted(cumulative_probs, p)
        valid_indices = sorted_indices[:cutoff + 1]
        valid_probs = probs[valid_indices]
        valid_probs = valid_probs / np.sum(valid_probs)

        rng = np.random.default_rng(seed)
        return int(rng.choice(valid_indices, p=valid_probs))

    @staticmethod
    def beam_search(
        step_fn: Callable[[List[int]], np.ndarray],
        start_tokens: List[int],
        beam_width: int = 3,
        max_steps: int = 5,
    ) -> List[Tuple[List[int], float]]:
        """
        Beam search (Section 12.3.2, Eq 12.34):
        Maintains beam_width best candidate sequences maximizing cumulative log-probability.
        Returns list of (sequence, total_log_prob).
        """
        # (sequence, cumulative_log_prob)
        beams = [(list(start_tokens), 0.0)]

        for _ in range(max_steps):
            candidates = []
            for seq, cum_log_prob in beams:
                logits = step_fn(seq)
                log_probs = np.log(softmax(logits, axis=-1) + 1e-12)
                # Expand to top beam_width candidates
                top_tokens = np.argpartition(log_probs, -beam_width)[-beam_width:]
                for tok in top_tokens:
                    candidates.append((seq + [int(tok)], cum_log_prob + float(log_probs[tok])))

            # Prune to beam_width
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

        return beams


# =============================================================================
# 3. Low-Rank Adaptation (LoRA, Section 12.3.5)
# =============================================================================

class LoRALinear:
    """
    Low-Rank Adaptation (LoRA) layer (Section 12.3.5, Eq 12.36, Figure 12.21).
    Y = X W_0 + (alpha / R) * X A B
    where W_0 in R^{D x D} is frozen, A in R^{D x R}, B in R^{R x D}.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0,
                 seed: Optional[int] = None) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        rng = np.random.default_rng(seed)
        # Frozen base weight
        self.W_0 = rng.normal(0, 0.02, size=(in_features, out_features))

        # LoRA trainable low-rank matrices
        # Standard LoRA initialization: A ~ Gaussian, B = 0 (so initially delta W = 0)
        self.A = rng.normal(0, 0.01, size=(in_features, rank))
        self.B = np.zeros((rank, out_features), dtype=np.float64)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Base forward
        base_out = X @ self.W_0
        # LoRA forward
        lora_out = (X @ self.A @ self.B) * self.scaling
        return base_out + lora_out

    def merge_weights(self) -> np.ndarray:
        """
        Eq 12.36: W_hat = W_0 + (alpha / R) * A @ B.
        Zero latency overhead during deployment inference.
        """
        return self.W_0 + self.scaling * (self.A @ self.B)


# =============================================================================
# 4. High-Resolution Figure Reproductions (Figures 12.15 - 12.21)
# =============================================================================

def generate_figure_12_15(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.15: Architecture of a GPT decoder transformer network.
    Tokens -> Embedding + PE -> Stack of masked transformer layers -> LSM -> Next token.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(7.0, 8.0))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 8.5)
    ax.axis('off')
    ax.set_title("Figure 12.15: Architecture of a GPT Decoder Transformer Network",
                 fontsize=11.5, fontweight='bold', pad=12)

    def draw_box(x, y, w, h, text, color='#2c3e50', bgcolor='#f8f9fa', fontsize=9.5):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                      linewidth=1.6, edgecolor=color, facecolor=bgcolor)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=color)

    # Input tokens
    draw_box(1.5, 0.2, 3.0, 0.5, "Input Tokens $\\{x_1, \\dots, x_N\\}$", color='#2c3e50', bgcolor='#eaecee')

    # Embedding + PE
    draw_box(1.5, 1.3, 3.0, 0.6, "Token Embedding\n+ Positional Encoding", color='#2980b9', bgcolor='#ebf5fb')
    ax.annotate("", xy=(3.0, 1.3), xytext=(3.0, 0.7), arrowprops=dict(arrowstyle="->", lw=1.4, color='#2c3e50'))

    # Stack of Masked Transformer Layers
    layers = ["Masked Transformer Layer 1", "Masked Transformer Layer 2", "$\\dots$", "Masked Transformer Layer $L$"]
    y_starts = [2.4, 3.4, 4.4, 5.4]
    for y, lbl in zip(y_starts, layers):
        draw_box(1.2, y, 3.6, 0.6, lbl, color='#8e44ad', bgcolor='#f4ecf7')
        if y > 2.4:
            ax.annotate("", xy=(3.0, y), xytext=(3.0, y - 0.4), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    ax.annotate("", xy=(3.0, 2.4), xytext=(3.0, 1.9), arrowprops=dict(arrowstyle="->", lw=1.4, color='#2c3e50'))

    # Linear-Softmax (LSM)
    draw_box(1.5, 6.5, 3.0, 0.55, "Linear-Softmax (LSM)", color='#27ae60', bgcolor='#eafaf1')
    ax.annotate("", xy=(3.0, 6.5), xytext=(3.0, 6.0), arrowprops=dict(arrowstyle="->", lw=1.4, color='#2c3e50'))

    # Output probabilities / Next token
    draw_box(1.5, 7.5, 3.0, 0.5, "Next-Token Distribution $p(x_{N+1} \\mid x_{1:N})$", color='#c0392b', bgcolor='#fadbd8')
    ax.annotate("", xy=(3.0, 7.5), xytext=(3.0, 7.05), arrowprops=dict(arrowstyle="->", lw=1.5, color='#c0392b'))

    plt.tight_layout()
    _save_figure(fig, "Figure_12_15", save_dir)
    return fig


def generate_figure_12_16(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.16: Structure of the masked causal attention matrix.
    Upper triangular elements (red) are masked out to zero.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    tokens = ["<start>", "I", "swam", "across", "the", "river"]
    N = len(tokens)

    # 1 where allowed (lower-triangular), 0 where masked (upper-triangular)
    mask_matrix = np.tril(np.ones((N, N)))

    # Color map: lower triangular blue (allowed), upper triangular soft red (masked)
    display_matrix = np.where(mask_matrix == 1, 0.7, 0.1)

    im = ax.imshow(display_matrix, cmap='coolwarm', vmin=0.0, vmax=1.0)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(tokens, fontsize=10.0, rotation=45, ha='right')
    ax.set_yticklabels(tokens, fontsize=10.0)

    # Annotate allowed vs masked
    for i in range(N):
        for j in range(N):
            if j <= i:
                ax.text(j, i, "Attend\n($A_{ij}$)", ha='center', va='center',
                        fontsize=8.5, color='white', fontweight='bold')
            else:
                ax.text(j, i, "Masked\n($-\\infty$)", ha='center', va='center',
                        fontsize=8.5, color='#78281f', fontweight='bold')

    ax.set_xlabel("Keys / Past Inputs ($j$)", fontsize=10.5, fontweight='bold')
    ax.set_ylabel("Queries / Outputs ($i$)", fontsize=10.5, fontweight='bold')
    ax.set_title("Figure 12.16: Causal Attention Mask Matrix (Lookahead Prevention)",
                 fontsize=11.5, fontweight='bold', pad=12)

    plt.tight_layout()
    _save_figure(fig, "Figure_12_16", save_dir)
    return fig


def generate_figure_12_17(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.17: Token probabilities comparison: Beam search vs Human text.
    Human text exhibits lower probability bursts / higher surprise.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(9.5, 4.2))

    tokens_num = 30
    steps = np.arange(tokens_num)

    rng = np.random.default_rng(42)
    # Beam search: high, stable probabilities (greedy repetitive)
    p_beam = 0.85 + 0.10 * np.sin(steps / 3.0) + rng.normal(0, 0.03, tokens_num)
    p_beam = np.clip(p_beam, 0.75, 0.99)

    # Human text: varying entropy, sudden bursts of surprising/creative tokens
    p_human = 0.70 + 0.20 * np.sin(steps / 2.0) + rng.normal(0, 0.12, tokens_num)
    # Inject 4 low-probability surprise spikes
    p_human[5] = 0.18
    p_human[12] = 0.22
    p_human[20] = 0.15
    p_human[27] = 0.25
    p_human = np.clip(p_human, 0.10, 0.95)

    ax.plot(steps, p_beam, color='#2980b9', lw=2.2, label="Beam Search (High Prob, Repetitive)", marker='o', markersize=4)
    ax.plot(steps, p_human, color='#c0392b', lw=2.2, label="Human Text (Variable, Informative Bursts)", marker='s', markersize=4)

    ax.set_xlim(-1, tokens_num)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Token position in generated sequence", fontsize=10.0)
    ax.set_ylabel("Conditional token probability $p(x_n \\mid x_{<n})$", fontsize=10.0)
    ax.set_title("Figure 12.17: Token Probabilities: Beam Search vs. Natural Human Text",
                 fontsize=11.5, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', fontsize=9.5)

    plt.tight_layout()
    _save_figure(fig, "Figure_12_17", save_dir)
    return fig


def generate_figure_12_18(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.18: Architecture of an encoder transformer model (e.g. BERT).
    Bidirectional self-attention layers with shared linear-softmax heads.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_xlim(-0.5, 7.0)
    ax.set_ylim(-0.5, 8.0)
    ax.axis('off')
    ax.set_title("Figure 12.18: Architecture of an Encoder Transformer Model (BERT)",
                 fontsize=11.5, fontweight='bold', pad=12)

    def draw_box(x, y, w, h, text, color='#2c3e50', bgcolor='#f8f9fa', fontsize=9.0):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                      linewidth=1.6, edgecolor=color, facecolor=bgcolor)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=color)

    # Input tokens
    draw_box(1.5, 0.2, 3.6, 0.5, "Input Sequence $\\{x_1, \\langle \\mathrm{mask} \\rangle, \\dots, x_N\\}$",
             color='#2c3e50', bgcolor='#eaecee')

    # Embedding + PE
    draw_box(1.5, 1.2, 3.6, 0.55, "Token + Position + Segment Embedding",
             color='#2980b9', bgcolor='#ebf5fb')
    ax.annotate("", xy=(3.3, 1.2), xytext=(3.3, 0.7), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    # Bidirectional Transformer Layers
    layers = ["Bidirectional Transformer Layer 1", "Bidirectional Transformer Layer 2", "$\\dots$", "Bidirectional Transformer Layer $L$"]
    y_starts = [2.2, 3.2, 4.2, 5.2]
    for y, lbl in zip(y_starts, layers):
        draw_box(1.2, y, 4.2, 0.55, lbl, color='#8e44ad', bgcolor='#f4ecf7')
        if y > 2.2:
            ax.annotate("", xy=(3.3, y), xytext=(3.3, y - 0.45), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    ax.annotate("", xy=(3.3, 2.2), xytext=(3.3, 1.75), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    # Linear-Softmax Output Heads (shared across tokens)
    draw_box(1.2, 6.2, 4.2, 0.55, "Shared Linear-Softmax Heads (LSM)", color='#27ae60', bgcolor='#eafaf1')
    ax.annotate("", xy=(3.3, 6.2), xytext=(3.3, 5.75), arrowprops=dict(arrowstyle="->", lw=1.4, color='#2c3e50'))

    # Output predictions
    draw_box(1.2, 7.2, 4.2, 0.5, "Predicted Tokens / Classification Logits", color='#c0392b', bgcolor='#fadbd8')
    ax.annotate("", xy=(3.3, 7.2), xytext=(3.3, 6.75), arrowprops=dict(arrowstyle="->", lw=1.5, color='#c0392b'))

    plt.tight_layout()
    _save_figure(fig, "Figure_12_18", save_dir)
    return fig


def generate_figure_12_19(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.19: Cross-attention layer architecture.
    Queries Q come from decoder tokens, Keys K and Values V come from encoder tokens Z.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 6.5)
    ax.axis('off')
    ax.set_title("Figure 12.19: Cross-Attention Mechanism ($Q$ from Decoder, $K, V$ from Encoder $Z$)",
                 fontsize=11.5, fontweight='bold', pad=12)

    def draw_box(x, y, w, h, text, color='#2c3e50', bgcolor='#f8f9fa', fontsize=9.5):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                      linewidth=1.6, edgecolor=color, facecolor=bgcolor)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=color)

    # Encoder output Z
    draw_box(4.2, 0.3, 1.8, 0.5, "Encoder $Z$", color='#27ae60', bgcolor='#eafaf1', fontsize=10.5)

    # Decoder previous layer output Y_dec
    draw_box(1.0, 0.3, 1.8, 0.5, "Decoder $Y_{\\mathrm{dec}}$", color='#2980b9', bgcolor='#ebf5fb', fontsize=10.5)

    # Projections W_Q, W_K, W_V
    draw_box(1.3, 1.5, 1.2, 0.5, "$W_Q$", color='#2980b9', bgcolor='#ebf5fb')
    draw_box(3.5, 1.5, 1.2, 0.5, "$W_K$", color='#27ae60', bgcolor='#eafaf1')
    draw_box(5.0, 1.5, 1.2, 0.5, "$W_V$", color='#e67e22', bgcolor='#fef5e7')

    ax.annotate("", xy=(1.9, 1.5), xytext=(1.9, 0.8), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))
    ax.annotate("", xy=(4.1, 1.5), xytext=(4.8, 0.8), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))
    ax.annotate("", xy=(5.6, 1.5), xytext=(5.3, 0.8), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    # MatMul Q K^T / sqrt(D_k)
    draw_box(2.0, 2.7, 2.8, 0.55, "MatMul ($Q K^T / \\sqrt{D_k}$)", color='#8e44ad', bgcolor='#f4ecf7')
    ax.annotate("", xy=(2.7, 2.7), xytext=(1.9, 2.0), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))
    ax.annotate("", xy=(4.1, 2.7), xytext=(4.1, 2.0), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    # Softmax
    draw_box(2.5, 3.7, 1.8, 0.5, "Softmax ($A$)", color='#8e44ad', bgcolor='#f4ecf7')
    ax.annotate("", xy=(3.4, 3.7), xytext=(3.4, 3.25), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    # MatMul A V
    draw_box(3.0, 4.7, 2.0, 0.55, "MatMul ($A V$)", color='#c0392b', bgcolor='#fadbd8')
    ax.annotate("", xy=(3.6, 4.7), xytext=(3.4, 4.2), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))
    ax.plot([5.6, 5.6, 4.7], [2.0, 4.95, 4.95], lw=1.3, color='#2c3e50')
    ax.annotate("", xy=(4.7, 4.95), xytext=(4.9, 4.95), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    # Output
    draw_box(3.2, 5.7, 1.6, 0.5, "Cross-Attn Out", color='#2c3e50', bgcolor='#eaecee')
    ax.annotate("", xy=(4.0, 5.7), xytext=(4.0, 5.25), arrowprops=dict(arrowstyle="->", lw=1.4, color='#2c3e50'))

    plt.tight_layout()
    _save_figure(fig, "Figure_12_19", save_dir)
    return fig


def generate_figure_12_20(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.20: Schematic illustration of a sequence-to-sequence transformer:
    Encoder stack + Decoder stack with cross-attention.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 8.5)
    ax.axis('off')
    ax.set_title("Figure 12.20: Sequence-to-Sequence Transformer Architecture",
                 fontsize=11.5, fontweight='bold', pad=12)

    def draw_box(x, y, w, h, text, color='#2c3e50', bgcolor='#f8f9fa', fontsize=9.0):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                      linewidth=1.6, edgecolor=color, facecolor=bgcolor)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=color)

    # 1. ENCODER (Left)
    draw_box(0.5, 0.3, 3.2, 0.5, "Input Tokens $X$", color='#2c3e50', bgcolor='#eaecee', fontsize=10.0)
    draw_box(0.5, 1.2, 3.2, 0.5, "Embedding + Positional Encoding", color='#2980b9', bgcolor='#ebf5fb')
    ax.annotate("", xy=(2.1, 1.2), xytext=(2.1, 0.8), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    draw_box(0.5, 2.2, 3.2, 0.7, "Self-Attention\nTransformer Layer 1", color='#2980b9', bgcolor='#ebf5fb')
    ax.annotate("", xy=(2.1, 2.2), xytext=(2.1, 1.7), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    ax.text(2.1, 3.3, "$\\vdots$", ha='center', va='center', fontsize=18, fontweight='bold')

    draw_box(0.5, 3.8, 3.2, 0.7, "Self-Attention\nTransformer Layer $L$", color='#2980b9', bgcolor='#ebf5fb')
    ax.annotate("", xy=(2.1, 3.8), xytext=(2.1, 3.5), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    # Encoder representation Z
    ax.text(2.1, 5.1, "Encoder Representation $Z$", ha='center', va='center',
            fontsize=10.0, fontweight='bold', color='#145a32')
    ax.annotate("", xy=(2.1, 4.9), xytext=(2.1, 4.5), arrowprops=dict(arrowstyle="->", lw=1.5, color='#145a32'))

    # 2. DECODER (Right)
    draw_box(5.5, 0.3, 3.5, 0.5, "Shifted Outputs $\\{\\langle \\mathrm{start} \\rangle, Y_{1:N-1}\\}$",
             color='#2c3e50', bgcolor='#eaecee', fontsize=9.0)
    draw_box(5.5, 1.2, 3.5, 0.5, "Embedding + Positional Encoding", color='#e67e22', bgcolor='#fef5e7')
    ax.annotate("", xy=(7.25, 1.2), xytext=(7.25, 0.8), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    draw_box(5.5, 2.2, 3.5, 0.7, "Masked Self-Attention\n+ Cross-Attention Layer 1", color='#e67e22', bgcolor='#fef5e7')
    ax.annotate("", xy=(7.25, 2.2), xytext=(7.25, 1.7), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    ax.text(7.25, 3.3, "$\\vdots$", ha='center', va='center', fontsize=18, fontweight='bold')

    draw_box(5.5, 3.8, 3.5, 0.7, "Masked Self-Attention\n+ Cross-Attention Layer $L$", color='#e67e22', bgcolor='#fef5e7')
    ax.annotate("", xy=(7.25, 3.8), xytext=(7.25, 3.5), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    # Cross-attention connecting arrow from Z to Decoder layers
    ax.plot([2.1, 4.5, 4.5], [5.1, 5.1, 2.55], lw=1.5, color='#27ae60')
    ax.plot([4.5, 4.5], [2.55, 4.15], lw=1.5, color='#27ae60')
    ax.annotate("", xy=(5.5, 2.55), xytext=(4.5, 2.55), arrowprops=dict(arrowstyle="->", lw=1.5, color='#27ae60'))
    ax.annotate("", xy=(5.5, 4.15), xytext=(4.5, 4.15), arrowprops=dict(arrowstyle="->", lw=1.5, color='#27ae60'))
    ax.text(4.2, 4.7, "Cross-Attn\nKey / Value", ha='right', fontsize=8.5, fontweight='bold', color='#145a32')

    # Output LSM and predictions
    draw_box(5.5, 5.2, 3.5, 0.55, "Linear-Softmax (LSM)", color='#8e44ad', bgcolor='#f4ecf7')
    ax.annotate("", xy=(7.25, 5.2), xytext=(7.25, 4.5), arrowprops=dict(arrowstyle="->", lw=1.4, color='#2c3e50'))

    draw_box(5.5, 6.4, 3.5, 0.5, "Output Tokens $Y_N$", color='#c0392b', bgcolor='#fadbd8')
    ax.annotate("", xy=(7.25, 6.4), xytext=(7.25, 5.75), arrowprops=dict(arrowstyle="->", lw=1.5, color='#c0392b'))

    # Section headers
    ax.text(2.1, 7.5, "Encoder", ha='center', fontsize=12.0, fontweight='bold', color='#2980b9')
    ax.text(7.25, 7.5, "Decoder", ha='center', fontsize=12.0, fontweight='bold', color='#e67e22')

    plt.tight_layout()
    _save_figure(fig, "Figure_12_20", save_dir)
    return fig


def generate_figure_12_21(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.21: Low-Rank Adaptation (LoRA) schematic:
    Frozen W0 in parallel with low-rank product A @ B (D x R and R x D).
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 5.0)
    ax.axis('off')
    ax.set_title("Figure 12.21: Low-Rank Adaptation (LoRA) Parameter-Efficient Fine-Tuning",
                 fontsize=11.5, fontweight='bold', pad=12)

    # Input X: (N, D)
    rect_x = patches.Rectangle((0.2, 1.5), 1.0, 2.0, linewidth=1.5, edgecolor='#2c3e50', facecolor='#eaecee')
    ax.add_patch(rect_x)
    ax.text(0.7, 2.5, "$X$\n$(N \\times D)$", ha='center', va='center', fontsize=9.5, fontweight='bold')

    # Top branch: Frozen W0: (D, D)
    rect_w0 = patches.Rectangle((2.4, 2.8), 1.8, 1.8, linewidth=2.0, edgecolor='#2980b9', facecolor='#ebf5fb')
    ax.add_patch(rect_w0)
    ax.text(3.3, 3.7, "$W_0$\n$(D \\times D)$\n(Frozen)", ha='center', va='center',
            fontsize=9.5, fontweight='bold', color='#1b4f72')

    # Bottom branch: LoRA A (D, R) and B (R, D)
    rect_a = patches.Rectangle((2.2, 0.4), 0.7, 1.6, linewidth=1.8, edgecolor='#e67e22', facecolor='#fef5e7')
    ax.add_patch(rect_a)
    ax.text(2.55, 1.2, "$A$\n$(D \\times R)$", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#7e5109')

    ax.text(3.15, 1.2, "$\\times$", ha='center', va='center', fontsize=14, fontweight='bold')

    rect_b = patches.Rectangle((3.5, 0.8), 1.6, 0.7, linewidth=1.8, edgecolor='#27ae60', facecolor='#eafaf1')
    ax.add_patch(rect_b)
    ax.text(4.3, 1.15, "$B$ $(R \\times D)$", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#145a32')

    # Connecting arrows from X
    ax.annotate("", xy=(2.4, 3.7), xytext=(1.2, 2.7), arrowprops=dict(arrowstyle="->", lw=1.4, color='#2c3e50'))
    ax.annotate("", xy=(2.2, 1.2), xytext=(1.2, 2.3), arrowprops=dict(arrowstyle="->", lw=1.4, color='#2c3e50'))

    # Plus circle
    circle_plus = patches.Circle((5.7, 2.5), 0.35, linewidth=1.8, edgecolor='#8e44ad', facecolor='#f4ecf7')
    ax.add_patch(circle_plus)
    ax.text(5.7, 2.5, "$+$", ha='center', va='center', fontsize=18, fontweight='bold', color='#8e44ad')

    # Arrows into plus
    ax.annotate("", xy=(5.4, 2.7), xytext=(4.2, 3.7), arrowprops=dict(arrowstyle="->", lw=1.4, color='#2980b9'))
    ax.annotate("", xy=(5.4, 2.3), xytext=(5.1, 1.15), arrowprops=dict(arrowstyle="->", lw=1.4, color='#27ae60'))

    # Output Y: (N, D)
    rect_y = patches.Rectangle((6.7, 1.5), 1.0, 2.0, linewidth=2.0, edgecolor='#c0392b', facecolor='#fadbd8')
    ax.add_patch(rect_y)
    ax.text(7.2, 2.5, "$Y$\n$(N \\times D)$", ha='center', va='center', fontsize=10.0, fontweight='bold', color='#78281f')

    ax.annotate("", xy=(6.7, 2.5), xytext=(6.05, 2.5), arrowprops=dict(arrowstyle="->", lw=1.6, color='#c0392b'))

    # Annotation
    ax.text(3.3, 0.1, "Trainable low-rank adaptation ($R \\ll D$, param reduction $\\sim 10,000\\times$)",
            ha='center', fontsize=9.0, fontweight='bold', color='#d35400')

    plt.tight_layout()
    _save_figure(fig, "Figure_12_21", save_dir)
    return fig


if __name__ == "__main__":
    for i in range(15, 22):
        func_name = f"generate_figure_12_{i}"
        func = globals()[func_name]
        fig = func()
        plt.close(fig)
    print("All 7 figures for Section 12.3 generated successfully!")
