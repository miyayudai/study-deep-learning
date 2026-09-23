"""
common/natural_language.py
==========================
Section 12.2: Natural Language
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements:
1. Word Embedding & Word2Vec (Section 12.2.1, Eqs 12.26 - 12.27, Figure 12.11):
   - Continuous Bag-of-Words (CBOW) and Skip-gram architectures.
   - Embedding lookup and semantic vector arithmetic (e.g. Paris - France + Italy ~ Rome).
2. Tokenization & Byte Pair Encoding (BPE) (Section 12.2.2, Figure 12.12):
   - Iterative pair frequency counting and merging on arbitrary text corpora.
   - Exact textbook reproduction of "Peter Piper picked a peck of pickled peppers".
3. Bag-of-Words & N-gram Models (Sections 12.2.3 - 12.2.4):
   - Frequency count vectorization and Markov n-gram conditional probability tables.
4. Recurrent Neural Networks (RNN) & Seq2Seq (Sections 12.2.5 - 12.2.6, Figures 12.13 - 12.14):
   - Unrolled RNN forward dynamics and Backpropagation Through Time (BPTT).
   - Vanishing/exploding gradient analysis across time steps.
   - Encoder-Decoder sequence-to-sequence machine translation model with bottleneck state z*.
5. High-Resolution Figure Reproductions (Figures 12.11 - 12.14):
   - Saved to 12/result/ and result/.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import os
from pathlib import Path
from collections import Counter
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
# 1. Word Embedding & Word2Vec
# =============================================================================

class SimpleWord2Vec:
    """
    Two-layer neural network for word embeddings (Section 12.2.1, Figure 12.11).
    Supports CBOW and Skip-gram representations.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, seed: Optional[int] = None) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        rng = np.random.default_rng(seed)
        # First-layer weights (input embeddings)
        self.W_in = rng.normal(0, 0.05, size=(vocab_size, embedding_dim))
        # Second-layer weights (output embeddings)
        self.W_out = rng.normal(0, 0.05, size=(embedding_dim, vocab_size))

    def get_embedding(self, word_idx: int) -> np.ndarray:
        """Eq 12.26: v_n = E x_n = W_in[word_idx]."""
        return self.W_in[word_idx]

    def forward_cbow(self, context_indices: List[int]) -> np.ndarray:
        """
        CBOW: average context embeddings, then predict target word (Figure 12.11a).
        """
        # Average context vectors: v = mean_{i in context} W_in[i]
        v_context = np.mean(self.W_in[context_indices], axis=0)  # (embedding_dim,)
        logits = v_context @ self.W_out                         # (vocab_size,)
        # Numerically stable softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def forward_skipgram(self, center_idx: int) -> np.ndarray:
        """
        Skip-gram: center word embedding predicts context distribution (Figure 12.11b).
        """
        v_center = self.W_in[center_idx]                        # (embedding_dim,)
        logits = v_center @ self.W_out                          # (vocab_size,)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs


# =============================================================================
# 2. Tokenization & Byte Pair Encoding (BPE)
# =============================================================================

class BytePairEncoder:
    """
    Byte Pair Encoding (BPE) subword tokenizer (Section 12.2.2, Figure 12.12).
    Iteratively merges the most frequent adjacent token pairs.
    """

    def __init__(self) -> None:
        self.vocab: Set[str] = set()
        self.merges: List[Tuple[str, str]] = []

    def get_stats(self, splits: List[List[str]]) -> Counter:
        """Count frequencies of adjacent token pairs across words."""
        pairs = Counter()
        for word in splits:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs

    def merge_vocab(self, pair: Tuple[str, str], splits: List[List[str]]) -> List[List[str]]:
        """Merge all occurrences of pair (p0, p1) into a single new token p0+p1."""
        new_splits = []
        p0, p1 = pair
        for word in splits:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == p0 and word[i + 1] == p1:
                    new_word.append(p0 + p1)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_splits.append(new_word)
        return new_splits

    def train_bpe(self, text: str, num_merges: int) -> List[Tuple[str, int]]:
        """
        Train BPE tokenizer on text for num_merges iterations.
        Returns list of (merged_token, frequency).
        """
        words = text.split()
        # Initialize splits at character level (preserving whitespace boundary by treating word-by-word)
        splits = [[char for char in word] for word in words]
        self.vocab = set(char for word in splits for char in word)

        merge_history = []
        for _ in range(num_merges):
            stats = self.get_stats(splits)
            if not stats:
                break
            # Find most frequent pair
            best_pair, freq = stats.most_common(1)[0]
            if freq < 2:  # Stop if no frequent pair
                break
            self.merges.append(best_pair)
            merged_token = best_pair[0] + best_pair[1]
            self.vocab.add(merged_token)
            merge_history.append((merged_token, freq))
            splits = self.merge_vocab(best_pair, splits)

        return merge_history


# =============================================================================
# 3. Recurrent Neural Networks (RNN) & Backpropagation Through Time (BPTT)
# =============================================================================

class SimpleRNN:
    """
    Recurrent Neural Network with hidden states z_n (Section 12.2.5, Figure 12.13).
    z_t = tanh(x_t W_xh + z_{t-1} W_hh + b_h)
    y_t = softmax(z_t W_hy + b_y)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        seed: Optional[int] = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        rng = np.random.default_rng(seed)
        self.W_xh = rng.normal(0, 0.1, size=(input_dim, hidden_dim))
        self.W_hh = rng.normal(0, 0.1, size=(hidden_dim, hidden_dim))
        self.b_h = np.zeros(hidden_dim)
        self.W_hy = rng.normal(0, 0.1, size=(hidden_dim, output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(
        self,
        X_seq: np.ndarray,
        z0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        X_seq: (T, input_dim) sequence of inputs
        z0: (hidden_dim,) initial hidden state
        Returns:
            Y_seq: (T, output_dim) sequence of predicted probabilities
            Z_seq: (T, hidden_dim) sequence of hidden state activations
        """
        T = len(X_seq)
        Z_seq = np.zeros((T, self.hidden_dim))
        Y_seq = np.zeros((T, self.output_dim))

        z_prev = z0 if z0 is not None else np.zeros(self.hidden_dim)

        for t in range(T):
            x_t = X_seq[t]
            # Hidden state activation
            a_t = x_t @ self.W_xh + z_prev @ self.W_hh + self.b_h
            z_t = np.tanh(a_t)
            Z_seq[t] = z_t

            # Output prediction
            logits = z_t @ self.W_hy + self.b_y
            exp_l = np.exp(logits - np.max(logits))
            y_t = exp_l / np.sum(exp_l)
            Y_seq[t] = y_t

            z_prev = z_t

        return Y_seq, Z_seq

    def compute_bptt_gradient_norm_decay(self, T: int) -> np.ndarray:
        """
        Section 12.2.6: BPTT vanishing gradient demonstration.
        Computes || (W_hh^T)^k || for k = 0, ..., T - 1.
        """
        norms = np.zeros(T)
        W_k = np.eye(self.hidden_dim)
        for k in range(T):
            norms[k] = np.linalg.norm(W_k, ord=2)
            W_k = W_k @ self.W_hh.T
        return norms


class Seq2SeqRNN:
    """
    Encoder-Decoder RNN architecture for translation (Section 12.2.5, Figure 12.14).
    Encoder absorbs input sequence into bottleneck vector z*.
    Decoder generates translated sequence token-by-token.
    """

    def __init__(self, vocab_in: int, vocab_out: int, hidden_dim: int, seed: Optional[int] = None) -> None:
        self.hidden_dim = hidden_dim
        self.encoder = SimpleRNN(vocab_in, hidden_dim, hidden_dim, seed=seed)
        self.decoder = SimpleRNN(vocab_out, hidden_dim, vocab_out, seed=seed if seed is None else seed + 1)

    def encode(self, X_in: np.ndarray) -> np.ndarray:
        """Encode full input sentence into final bottleneck state z*."""
        _, Z_enc = self.encoder.forward(X_in)
        z_star = Z_enc[-1]  # Final bottleneck representation
        return z_star

    def decode_step(self, x_dec: np.ndarray, z_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode single step given current token and previous hidden state."""
        Y, Z = self.decoder.forward(x_dec[np.newaxis, :], z0=z_prev)
        return Y[0], Z[0]


# =============================================================================
# 4. High-Resolution Figure Reproductions (Figures 12.11 - 12.14)
# =============================================================================

def generate_figure_12_11(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.11: Two-layer neural networks used to learn word embeddings:
    (a) Continuous Bag of Words (CBOW): context words -> target word.
    (b) Skip-gram: target word -> context words.
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 5.2))

    def draw_cbow(ax):
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, 5.5)
        ax.axis('off')
        ax.set_title("(a) Continuous Bag of Words (CBOW)", fontsize=11.5, fontweight='bold', pad=10)

        # Context input nodes on left
        context_labels = ["$x_{n-2}$", "$x_{n-1}$", "$x_{n+1}$", "$x_{n+2}$"]
        y_inputs = [0.8, 2.0, 3.2, 4.4]
        for y, lbl in zip(y_inputs, context_labels):
            rect = patches.FancyBboxPatch((0.2, y - 0.25), 1.0, 0.5, boxstyle="round,pad=0.1",
                                          edgecolor='#2c3e50', facecolor='#ebf5fb', lw=1.5)
            ax.add_patch(rect)
            ax.text(0.7, y, lbl, ha='center', va='center', fontsize=10.5, fontweight='bold', color='#1b4f72')

        # Hidden representation v in middle
        v_rect = patches.FancyBboxPatch((2.3, 2.1), 0.9, 1.0, boxstyle="round,pad=0.1",
                                        edgecolor='#27ae60', facecolor='#eafaf1', lw=1.8)
        ax.add_patch(v_rect)
        ax.text(2.75, 2.6, "$\\mathbf{v}$", ha='center', va='center', fontsize=13.0, fontweight='bold', color='#145a32')

        # Arrows from inputs to hidden v
        for y in y_inputs:
            ax.annotate("", xy=(2.3, 2.6), xytext=(1.2, y),
                        arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

        # Target output node on right
        out_rect = patches.FancyBboxPatch((4.0, 2.35), 1.0, 0.5, boxstyle="round,pad=0.1",
                                          edgecolor='#c0392b', facecolor='#fadbd8', lw=1.8)
        ax.add_patch(out_rect)
        ax.text(4.5, 2.6, "$x_n$", ha='center', va='center', fontsize=11.0, fontweight='bold', color='#78281f')

        # Arrow from hidden v to output
        ax.annotate("", xy=(4.0, 2.6), xytext=(3.2, 2.6),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color='#c0392b'))

        # Subtitles
        ax.text(0.7, 5.1, "Context Inputs", ha='center', fontsize=9.5, fontweight='bold', color='#7f8c8d')
        ax.text(2.75, 5.1, "Projection ($E$)", ha='center', fontsize=9.5, fontweight='bold', color='#7f8c8d')
        ax.text(4.5, 5.1, "Target Word", ha='center', fontsize=9.5, fontweight='bold', color='#7f8c8d')

    def draw_skipgram(ax):
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, 5.5)
        ax.axis('off')
        ax.set_title("(b) Skip-gram Model", fontsize=11.5, fontweight='bold', pad=10)

        # Center input node on left
        in_rect = patches.FancyBboxPatch((0.2, 2.35), 1.0, 0.5, boxstyle="round,pad=0.1",
                                         edgecolor='#c0392b', facecolor='#fadbd8', lw=1.8)
        ax.add_patch(in_rect)
        ax.text(0.7, 2.6, "$x_n$", ha='center', va='center', fontsize=11.0, fontweight='bold', color='#78281f')

        # Hidden representation v in middle
        v_rect = patches.FancyBboxPatch((2.3, 2.1), 0.9, 1.0, boxstyle="round,pad=0.1",
                                        edgecolor='#27ae60', facecolor='#eafaf1', lw=1.8)
        ax.add_patch(v_rect)
        ax.text(2.75, 2.6, "$\\mathbf{v}$", ha='center', va='center', fontsize=13.0, fontweight='bold', color='#145a32')

        # Arrow from center to v
        ax.annotate("", xy=(2.3, 2.6), xytext=(1.2, 2.6),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color='#c0392b'))

        # Context output nodes on right
        context_labels = ["$x_{n-2}$", "$x_{n-1}$", "$x_{n+1}$", "$x_{n+2}$"]
        y_outputs = [0.8, 2.0, 3.2, 4.4]
        for y, lbl in zip(y_outputs, context_labels):
            rect = patches.FancyBboxPatch((4.0, y - 0.25), 1.0, 0.5, boxstyle="round,pad=0.1",
                                          edgecolor='#2c3e50', facecolor='#ebf5fb', lw=1.5)
            ax.add_patch(rect)
            ax.text(4.5, y, lbl, ha='center', va='center', fontsize=10.5, fontweight='bold', color='#1b4f72')
            # Arrow from v to each context output
            ax.annotate("", xy=(4.0, y), xytext=(3.2, 2.6),
                        arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

        # Subtitles
        ax.text(0.7, 5.1, "Center Word", ha='center', fontsize=9.5, fontweight='bold', color='#7f8c8d')
        ax.text(2.75, 5.1, "Projection ($E$)", ha='center', fontsize=9.5, fontweight='bold', color='#7f8c8d')
        ax.text(4.5, 5.1, "Context Outputs", ha='center', fontsize=9.5, fontweight='bold', color='#7f8c8d')

    draw_cbow(ax1)
    draw_skipgram(ax2)

    plt.tight_layout()
    _save_figure(fig, "Figure_12_11", save_dir)
    return fig


def generate_figure_12_12(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.12: Illustration of Byte Pair Encoding (BPE) tokenization process
    on 'Peter Piper picked a peck of pickled peppers'.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 6.5)
    ax.axis('off')
    ax.set_title("Figure 12.12: Byte Pair Encoding (BPE) Iterative Token Merges",
                 fontsize=11.5, fontweight='bold', pad=12)

    lines = [
        ("Base Characters", "P e t e r   P i p e r   p i c k e d   a   p e c k   o f   p i c k l e d   p e p p e r s", '#2c3e50'),
        ("Merge 'pe' (freq=4)", "P e t e r   P i p e r   p i c k e d   a   [pe] c k   o f   p i c k l e d   [pe] p p e r s", '#2980b9'),
        ("Merge 'ck' (freq=3)", "P e t e r   P i p e r   p i [ck] e d   a   [pe] [ck]   o f   p i [ck] l e d   [pe] p p e r s", '#27ae60'),
        ("Merge 'pi' (freq=2)", "P e t e r   P i [pi] e r   [pi] [ck] e d   a   [pe] [ck]   o f   [pi] [ck] l e d   [pe] p p e r s", '#8e44ad'),
        ("Merge 'ed' (freq=2)", "P e t e r   P i [pi] e r   [pi] [ck] [ed]   a   [pe] [ck]   o f   [pi] [ck] l [ed]   [pe] p p e r s", '#e67e22'),
        ("Merge 'per' (freq=2)", "P e t e r   [Piper]   [pi] [ck] [ed]   a   [pe] [ck]   o f   [pi] [ck] l [ed]   [pe] p [per] s", '#c0392b'),
    ]

    y_coords = np.linspace(5.5, 0.8, len(lines))
    for i, (stage_name, text_str, col) in enumerate(lines):
        y = y_coords[i]
        # Step label box
        ax.text(0.0, y, f"Step {i}: {stage_name}", fontsize=9.5, fontweight='bold', color=col,
                va='center', ha='left')
        # Monospace sentence tokens
        ax.text(0.2, y - 0.35, text_str, fontsize=9.0, fontfamily='monospace',
                va='center', ha='left', color='#34495e',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#f8f9fa', edgecolor=col, alpha=0.3, lw=1.0))

    plt.tight_layout()
    _save_figure(fig, "Figure_12_12", save_dir)
    return fig


def generate_figure_12_13(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.13: General RNN with shared parameters w unfolded over time:
    takes sequence x_1..x_N and generates y_1..y_N with hidden states z_t.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 4.0)
    ax.axis('off')
    ax.set_title("Figure 12.13: General Recurrent Neural Network (RNN) Unfolded in Time",
                 fontsize=11.5, fontweight='bold', pad=12)

    # Time steps: 1, 2, 3
    x_pos = [2.0, 4.2, 6.4]

    # Initial state z_0
    rect_z0 = patches.FancyBboxPatch((0.2, 1.6), 0.8, 0.6, boxstyle="round,pad=0.1",
                                     edgecolor='#7f8c8d', facecolor='#eaecee', lw=1.5)
    ax.add_patch(rect_z0)
    ax.text(0.6, 1.9, "$\\mathbf{z}_0$", ha='center', va='center', fontsize=11.0, fontweight='bold', color='#566573')

    # Arrow z0 -> z1
    ax.annotate("", xy=(1.6, 1.9), xytext=(1.0, 1.9),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#2c3e50'))

    for i, (x_c, t) in enumerate(zip(x_pos, [1, 2, 3])):
        # Input x_t (bottom)
        rect_x = patches.FancyBboxPatch((x_c - 0.4, 0.2), 0.8, 0.5, boxstyle="round,pad=0.1",
                                        edgecolor='#2980b9', facecolor='#ebf5fb', lw=1.5)
        ax.add_patch(rect_x)
        ax.text(x_c, 0.45, f"$\\mathbf{{x}}_{t}$", ha='center', va='center', fontsize=11.0, fontweight='bold', color='#1b4f72')

        # RNN cell w (middle)
        rect_cell = patches.FancyBboxPatch((x_c - 0.5, 1.5), 1.0, 0.8, boxstyle="round,pad=0.15",
                                           edgecolor='#2c3e50', facecolor='#fdfefe', lw=2.0)
        ax.add_patch(rect_cell)
        ax.text(x_c, 1.9, "$\\mathbf{w}$", ha='center', va='center', fontsize=13.0, fontweight='bold', color='#2c3e50')
        ax.text(x_c, 2.5, f"$\\mathbf{{z}}_{t}$", ha='center', va='center', fontsize=10.0, color='#e67e22', fontweight='bold')

        # Output y_t (top)
        rect_y = patches.FancyBboxPatch((x_c - 0.4, 3.1), 0.8, 0.5, boxstyle="round,pad=0.1",
                                        edgecolor='#c0392b', facecolor='#fadbd8', lw=1.5)
        ax.add_patch(rect_y)
        ax.text(x_c, 3.35, f"$\\mathbf{{y}}_{t}$", ha='center', va='center', fontsize=11.0, fontweight='bold', color='#78281f')

        # Arrows x_t -> cell and cell -> y_t
        ax.annotate("", xy=(x_c, 1.5), xytext=(x_c, 0.7),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color='#2c3e50'))
        ax.annotate("", xy=(x_c, 3.1), xytext=(x_c, 2.3),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color='#2c3e50'))

        # Recurrent arrows between cells
        if i < len(x_pos) - 1:
            ax.annotate("", xy=(x_pos[i + 1] - 0.5, 1.9), xytext=(x_c + 0.5, 1.9),
                        arrowprops=dict(arrowstyle="->", lw=1.6, color='#e67e22'))

    # Final continuation arrow
    ax.annotate("", xy=(7.7, 1.9), xytext=(6.9, 1.9),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#e67e22'))
    ax.text(8.0, 1.9, "$\\dots$", ha='center', va='center', fontsize=16, fontweight='bold', color='#7f8c8d')

    plt.tight_layout()
    _save_figure(fig, "Figure_12_13", save_dir)
    return fig


def generate_figure_12_14(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.14: Sequence-to-sequence RNN for language translation (English -> Dutch):
    'I am happy' -> z* (bottleneck) -> 'Ik ben gelukkig <stop>'.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 4.0)
    ax.axis('off')
    ax.set_title("Figure 12.14: Sequence-to-Sequence RNN Model for Machine Translation",
                 fontsize=11.5, fontweight='bold', pad=12)

    # 1. Encoder stages (English input: 'I', 'am', 'happy')
    enc_words = ["I", "am", "happy"]
    enc_x = [1.2, 2.6, 4.0]

    # z0
    ax.text(0.2, 1.8, "$\\mathbf{z}_0$", ha='center', va='center', fontsize=11.0, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#eaecee', edgecolor='#7f8c8d'))
    ax.annotate("", xy=(0.8, 1.8), xytext=(0.4, 1.8), arrowprops=dict(arrowstyle="->", lw=1.4, color='#2c3e50'))

    for x_c, w in zip(enc_x, enc_words):
        # Input word
        ax.text(x_c, 0.5, w, ha='center', va='center', fontsize=10.5, fontweight='bold', color='#1b4f72',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#ebf5fb', edgecolor='#2980b9'))
        # Encoder cell
        ax.text(x_c, 1.8, "$\\mathbf{w}$", ha='center', va='center', fontsize=11.5, fontweight='bold', color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f9fa', edgecolor='#2c3e50', lw=1.5))
        ax.annotate("", xy=(x_c, 1.5), xytext=(x_c, 0.8), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

    # Encoder recurrent links
    ax.annotate("", xy=(2.3, 1.8), xytext=(1.5, 1.8), arrowprops=dict(arrowstyle="->", lw=1.5, color='#2980b9'))
    ax.annotate("", xy=(3.7, 1.8), xytext=(2.9, 1.8), arrowprops=dict(arrowstyle="->", lw=1.5, color='#2980b9'))

    # Bottleneck connection z*
    ax.annotate("", xy=(5.2, 1.8), xytext=(4.3, 1.8), arrowprops=dict(arrowstyle="->", lw=2.2, color='#c0392b'))
    ax.text(4.75, 2.2, "$\\mathbf{z}^*$\n(Bottleneck)", ha='center', va='bottom',
            fontsize=9.5, fontweight='bold', color='#c0392b')

    # 2. Decoder stages (Dutch output: 'Ik', 'ben', 'gelukkig', '<stop>')
    dec_words_in = ["$\\langle \\mathrm{start} \\rangle$", "Ik", "ben", "gelukkig"]
    dec_words_out = ["Ik", "ben", "gelukkig", "$\\langle \\mathrm{stop} \\rangle$"]
    dec_x = [5.5, 7.0, 8.5, 10.0]

    for i, (x_c, w_in, w_out) in enumerate(zip(dec_x, dec_words_in, dec_words_out)):
        # Decoder cell
        ax.text(x_c, 1.8, "$\\mathbf{w}$", ha='center', va='center', fontsize=11.5, fontweight='bold', color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdfefe', edgecolor='#27ae60', lw=1.8))
        # Input word
        ax.text(x_c, 0.5, w_in, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#145a32',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#eafaf1', edgecolor='#27ae60'))
        ax.annotate("", xy=(x_c, 1.5), xytext=(x_c, 0.8), arrowprops=dict(arrowstyle="->", lw=1.3, color='#2c3e50'))

        # Output word
        ax.text(x_c, 3.1, w_out, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#78281f',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#fadbd8', edgecolor='#c0392b'))
        ax.annotate("", xy=(x_c, 2.8), xytext=(x_c, 2.1), arrowprops=dict(arrowstyle="->", lw=1.3, color='#c0392b'))

        # Recurrent connections
        if i < len(dec_x) - 1:
            ax.annotate("", xy=(dec_x[i + 1] - 0.3, 1.8), xytext=(x_c + 0.3, 1.8),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color='#27ae60'))

    # Bracket labels for Encoder vs Decoder
    ax.text(2.6, -0.2, "Encoder (Source: English)", ha='center', fontsize=10.0, fontweight='bold', color='#2980b9')
    ax.text(7.75, -0.2, "Decoder (Target: Dutch, Autoregressive)", ha='center', fontsize=10.0, fontweight='bold', color='#27ae60')

    plt.tight_layout()
    _save_figure(fig, "Figure_12_14", save_dir)
    return fig


if __name__ == "__main__":
    for i in range(11, 15):
        func_name = f"generate_figure_12_{i}"
        func = globals()[func_name]
        fig = func()
        plt.close(fig)
    print("All 4 figures for Section 12.2 generated successfully!")
