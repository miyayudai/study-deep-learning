"""
common/multimodal_transformers.py
=================================
Section 12.4: Multimodal Transformers
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module implements:
1. Vision Transformers (ViT) (Section 12.4.1, Figure 12.22):
   - Image patch partitioning (P x P) and flattening into 1D tokens.
   - Linear projection of flattened patches to latent dimension D.
   - Learnable <class> token and learned 1D positional encodings.
   - Transformer Encoder stack with Pre-LN / Multi-Head Attention / MLP.
   - Classification head (Linear + Softmax) for class prediction vector c.
2. Generative Image Transformers (Section 12.4.2, Figures 12.23, 12.24):
   - Autoregressive image decomposition in raster scan order (Eq 12.37).
   - Vector Quantization (VQ) codebook clustering (Eq 12.38) and Straight-Through Estimator (STE).
   - Discrete token generation and image patch decoding.
3. Audio Transformers (AST) & Mel Spectrograms (Section 12.4.3, Figure 12.25):
   - Mel-scale frequency filterbank calculation and Short-Time Fourier Transform (STFT).
   - Mel spectrogram extraction and whale song acoustic synthesis.
   - Patch-based tokenization of 2D time-frequency spectrograms.
4. Text-to-Speech (Vall-E) (Section 12.4.4, Figure 12.26):
   - Conditioning on text tokens and acoustic prompt tokens.
   - Discrete neural codec representation and generative speech modeling.
5. Vision and Language Transformers (CM3 / CM3Leon) (Section 12.4.5, Figure 12.27):
   - Unified joint vocabulary across text and image tokens.
   - Multimodal task handling: text-to-image, captioning, inpainting, and instruction editing.
6. Faithful High-Resolution Figure Reproductions (Figures 12.22 - 12.27):
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
    TransformerLayer,
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
# 1. Vision Transformer (ViT) (Section 12.4.1, Figure 12.22)
# =============================================================================

class VisionTransformer:
    """
    Vision Transformer (ViT) for image classification (Dosovitskiy et al., 2020).
    
    Transforms an image x in R^{H x W x C} into a sequence of N non-overlapping
    patches of size P x P, flattens them, linearly projects each to dimension D,
    prepends a learnable <class> token, adds 1D learnable position embeddings,
    and processes the sequence through L standard Transformer Encoder layers.
    The final output corresponding to the <class> token is transformed via a
    linear layer and softmax (LSM) to output class probabilities c.
    
    Equation & Dimension Reference:
    - Number of patches: N = (H / P) * (W / P)
    - Patch dimension: P^2 * C
    - Latent dimension: D (d_model)
    - Sequence length: N + 1 (patches + <class> token)
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (32, 32),
        patch_size: int = 8,
        in_channels: int = 3,
        num_classes: int = 10,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        mlp_dim: int = 128,
        seed: int = 42,
    ):
        self.H, self.W = img_size
        self.P = patch_size
        self.C = in_channels
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim

        assert self.H % self.P == 0 and self.W % self.P == 0, (
            f"Image dimensions ({self.H}, {self.W}) must be divisible by patch size {self.P}"
        )
        self.grid_h = self.H // self.P
        self.grid_w = self.W // self.P
        self.num_patches = self.grid_h * self.grid_w
        self.patch_dim = self.P * self.P * self.C

        rng = np.random.default_rng(seed)

        # Patch projection W_proj: (patch_dim, d_model), b_proj: (d_model,)
        scale = np.sqrt(2.0 / (self.patch_dim + d_model))
        self.W_proj = rng.normal(0, scale, size=(self.patch_dim, d_model))
        self.b_proj = np.zeros(d_model)

        # Learnable <class> token: (1, d_model)
        self.cls_token = rng.normal(0, 0.02, size=(1, d_model))

        # Learnable 1D position embeddings: (num_patches + 1, d_model)
        self.pos_embed = rng.normal(0, 0.02, size=(self.num_patches + 1, d_model))

        # Transformer encoder layers (Pre-LN)
        self.layers: List[TransformerLayer] = []
        for l_idx in range(num_layers):
            layer_seed = seed + 100 + l_idx
            self.layers.append(
                TransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=mlp_dim,
                    pre_norm=True,
                    seed=layer_seed,
                )
            )

        # Classification Head (LayerNorm + Linear-Softmax)
        self.ln_head = LayerNorm(d_model)
        head_scale = np.sqrt(2.0 / (d_model + num_classes))
        self.W_head = rng.normal(0, head_scale, size=(d_model, num_classes))
        self.b_head = np.zeros(num_classes)

    def patchify(self, x: np.ndarray) -> np.ndarray:
        """
        Partition image(s) into non-overlapping patches and flatten them.
        
        Args:
            x: Input array of shape (H, W, C) or (B, H, W, C).
            
        Returns:
            patches: Array of shape (B, N, P^2 * C) where N = (H/P)*(W/P).
        """
        is_single = (x.ndim == 3)
        if is_single:
            x = x[np.newaxis, ...]  # (1, H, W, C)

        B, H, W, C = x.shape
        assert H == self.H and W == self.W and C == self.C, (
            f"Expected image shape (B, {self.H}, {self.W}, {self.C}), got {x.shape}"
        )

        # Reshape to (B, grid_h, P, grid_w, P, C)
        patches = x.reshape(B, self.grid_h, self.P, self.grid_w, self.P, C)
        # Transpose to (B, grid_h, grid_w, P, P, C)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        # Flatten patches into (B, N, P * P * C)
        patches = patches.reshape(B, self.num_patches, self.patch_dim)

        return patches

    def unpatchify(self, patches: np.ndarray) -> np.ndarray:
        """
        Reconstruct images from patches.
        
        Args:
            patches: Array of shape (B, N, P^2 * C) or (N, P^2 * C).
            
        Returns:
            x: Reconstructed images of shape (B, H, W, C).
        """
        is_single = (patches.ndim == 2)
        if is_single:
            patches = patches[np.newaxis, ...]

        B, N, _ = patches.shape
        assert N == self.num_patches, f"Expected {self.num_patches} patches, got {N}"

        # Reshape to (B, grid_h, grid_w, P, P, C)
        x = patches.reshape(B, self.grid_h, self.grid_w, self.P, self.P, self.C)
        # Transpose to (B, grid_h, P, grid_w, P, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        # Reshape to (B, H, W, C)
        x = x.reshape(B, self.H, self.W, self.C)

        return x[0] if is_single else x

    def forward(
        self,
        x: np.ndarray,
        return_attention_maps: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[np.ndarray]]]:
        """
        Forward pass of Vision Transformer.
        
        Args:
            x: Image array of shape (H, W, C) or (B, H, W, C).
            return_attention_maps: If True, return attention weight matrices for all layers.
            
        Returns:
            logits: Array of shape (B, num_classes)
            probs: Class probabilities of shape (B, num_classes)
            (optional) attention_maps: List of attention matrices from each layer.
        """
        patches = self.patchify(x)  # (B, N, patch_dim)
        B = patches.shape[0]

        # Linear projection to d_model: (B, N, d_model)
        projected = np.matmul(patches, self.W_proj) + self.b_proj

        # Prepend <class> token: (B, 1, d_model)
        cls_tokens = np.tile(self.cls_token, (B, 1, 1))
        # Concatenate: (B, N + 1, d_model)
        z = np.concatenate([cls_tokens, projected], axis=1)

        # Add learned 1D positional encodings
        z = z + self.pos_embed[np.newaxis, ...]

        attention_maps = []
        for layer in self.layers:
            # We process each batch element through TransformerLayer
            layer_outputs = []
            layer_attns = []
            for b in range(B):
                out_b, attn_b = layer.forward(z[b])
                layer_outputs.append(out_b)
                layer_attns.append(attn_b)
            z = np.stack(layer_outputs, axis=0)
            if return_attention_maps:
                attention_maps.append(np.stack(layer_attns, axis=0))

        # Classification from final <class> token output z[:, 0, :]
        cls_out = z[:, 0, :]  # (B, d_model)
        cls_norm = self.ln_head.forward(cls_out)
        logits = np.matmul(cls_norm, self.W_head) + self.b_head
        probs = softmax(logits, axis=-1)

        if return_attention_maps:
            return logits, probs, attention_maps
        return logits, probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return argmax class prediction."""
        _, probs = self.forward(x)
        return np.argmax(probs, axis=-1)


# =============================================================================
# 2. Vector Quantization (VQ) (Section 12.4.2, Eq 12.38)
# =============================================================================

class VectorQuantizer:
    """
    Vector Quantization (VQ) module for discrete representations (Eq 12.38).
    
    Given a set of data vectors x_n in R^D and a codebook C = {c_1, ..., c_K} in R^D,
    each vector x_n is mapped to its nearest codebook vector:
        x_n -> argmin_{c_k in C} ||x_n - c_k||_2^2
        
    Provides:
    - Nearest codebook lookup (Eq 12.38).
    - Straight-Through Gradient Estimation (STE) approximation for backprop.
    - VQ loss and commitment loss calculation:
        L_vq = ||x - sg[z_q]||^2 + beta * ||sg[x] - z_q||^2
    - K-means Lloyd clustering algorithm to fit codebook vectors directly from data.
    """

    def __init__(
        self,
        num_embeddings: int = 16,
        embedding_dim: int = 8,
        commitment_cost: float = 0.25,
        seed: int = 42,
    ):
        self.num_embeddings = num_embeddings  # K
        self.embedding_dim = embedding_dim    # D
        self.commitment_cost = commitment_cost  # beta
        rng = np.random.default_rng(seed)
        # Initialize codebook vectors c_k
        self.codebook = rng.normal(0, 1.0, size=(num_embeddings, embedding_dim))
        # Normalize codebook vectors to unit sphere for stable distance computation
        self.codebook = self.codebook / (np.linalg.norm(self.codebook, axis=-1, keepdims=True) + 1e-8)

    def quantize(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize continuous input vectors to nearest codebook vectors.
        
        Args:
            x: Input array of shape (N, D) or (B, N, D).
            
        Returns:
            indices: Codebook index array of shape (N,) or (B, N).
            z_q: Quantized vectors of shape identical to x.
        """
        orig_shape = x.shape
        D = orig_shape[-1]
        assert D == self.embedding_dim, f"Expected feature dimension {self.embedding_dim}, got {D}"

        x_flat = x.reshape(-1, D)  # (M, D)
        # Compute squared Euclidean distances ||x_m - c_k||^2
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x @ c^T
        x_sq = np.sum(x_flat ** 2, axis=1, keepdims=True)  # (M, 1)
        c_sq = np.sum(self.codebook ** 2, axis=1, keepdims=True).T  # (1, K)
        distances = x_sq + c_sq - 2.0 * np.matmul(x_flat, self.codebook.T)  # (M, K)

        indices = np.argmin(distances, axis=1)  # (M,)
        z_q = self.codebook[indices]  # (M, D)

        return indices.reshape(orig_shape[:-1]), z_q.reshape(orig_shape)

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """
        Map discrete token indices back to D-dimensional codebook vectors.
        
        Args:
            indices: Array of indices in {0, ..., K-1}.
            
        Returns:
            z_q: Array of shape (*indices.shape, embedding_dim).
        """
        return self.codebook[indices]

    def compute_loss(self, x: np.ndarray, z_q: np.ndarray) -> Dict[str, float]:
        """
        Compute VQ and commitment losses:
        L_codebook = mean(||sg[x] - z_q||^2)
        L_commitment = mean(||x - sg[z_q]||^2)
        L_total = L_codebook + beta * L_commitment
        """
        vq_loss = float(np.mean((x - z_q) ** 2))
        total_loss = (1.0 + self.commitment_cost) * vq_loss
        return {
            "vq_loss": vq_loss,
            "commitment_loss": vq_loss,
            "total_loss": total_loss,
        }

    def straight_through_backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Straight-Through Estimator (STE) gradient approximation (Bengio et al., 2013).
        Copies the incoming gradient directly through the non-differentiable argmin:
            grad_input = grad_output.
        """
        return grad_output.copy()

    def fit_kmeans(self, data: np.ndarray, max_iters: int = 20) -> None:
        """
        Fit codebook vectors to training data using K-means clustering (Lloyd's algorithm).
        """
        data_flat = data.reshape(-1, self.embedding_dim)
        M = data_flat.shape[0]
        K = self.num_embeddings

        # Randomly choose K initial centers from data
        rng = np.random.default_rng(42)
        init_idx = rng.choice(M, size=min(K, M), replace=False)
        self.codebook[:len(init_idx)] = data_flat[init_idx]

        for _ in range(max_iters):
            indices, _ = self.quantize(data_flat)
            new_codebook = np.zeros_like(self.codebook)
            for k in range(K):
                cluster_pts = data_flat[indices == k]
                if len(cluster_pts) > 0:
                    new_codebook[k] = np.mean(cluster_pts, axis=0)
                else:
                    new_codebook[k] = self.codebook[k]
            shift = np.linalg.norm(new_codebook - self.codebook)
            self.codebook = new_codebook
            if shift < 1e-4:
                break


# =============================================================================
# 3. Autoregressive Image Generator (Section 12.4.2, Figures 12.23, 12.24)
# =============================================================================

class AutoregressiveImageGenerator:
    """
    Autoregressive Generative Model for Images (Section 12.4.2, Eq 12.37).
    
    Decomposes the joint distribution over raster-scan ordered image pixels or tokens:
        p(x_1, ..., x_N) = prod_{n=1}^N p(x_n | x_1, ..., x_{n-1})
        
    Generates images sequentially step-by-step from marginal p(x_1) to full image.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (4, 4),
        num_tokens: int = 8,
        d_model: int = 32,
        seed: int = 42,
    ):
        self.grid_h, self.grid_w = grid_size
        self.N = self.grid_h * self.grid_w
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.rng = np.random.default_rng(seed)

        # Transition matrix / logits mimicking learned conditional distribution
        # In a real model, this is parameterized by a causal transformer
        self.token_embeddings = self.rng.normal(0, 1.0, size=(num_tokens, d_model))

    def raster_scan_coordinates(self) -> List[Tuple[int, int]]:
        """Return list of (row, col) coordinates in raster scan order."""
        coords = []
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                coords.append((r, c))
        return coords

    def sample_step_by_step(
        self,
        temperature: float = 1.0,
    ) -> List[np.ndarray]:
        """
        Sample an image autoregressively in raster scan order.
        
        Returns:
            frames: List of N grids of shape (grid_h, grid_w) showing intermediate states.
        """
        tokens = np.full((self.grid_h, self.grid_w), fill_value=-1, dtype=int)
        frames = []
        coords = self.raster_scan_coordinates()

        # Prior distribution for first token p(x_1)
        prior_logits = np.ones(self.num_tokens) / self.num_tokens
        
        prev_token = 0
        for step, (r, c) in enumerate(coords):
            if step == 0:
                probs = softmax(prior_logits / temperature)
            else:
                # Condition on previous token with spatial smoothness bias
                cond_logits = np.zeros(self.num_tokens)
                for k in range(self.num_tokens):
                    # Higher probability for similar or smooth transitions
                    cond_logits[k] = -0.5 * (k - prev_token) ** 2
                probs = softmax(cond_logits / temperature)

            sampled_token = self.rng.choice(self.num_tokens, p=probs)
            tokens[r, c] = sampled_token
            prev_token = sampled_token
            frames.append(tokens.copy())

        return frames


# =============================================================================
# 4. Audio Mel Spectrogram Processor (Section 12.4.3, Figure 12.25)
# =============================================================================

class AudioMelSpectrogramProcessor:
    """
    Processor for Audio Data: Mel Spectrogram and Patch Tokenization (AST).
    
    Transforms 1D continuous audio pressure waveforms s(t) into 2D Mel spectrograms,
    following the standard subjective perceptual Mel frequency scale:
        m = 2595 * log10(1 + f / 700)
        f = 700 * (10^{m / 2595} - 1)
        
    Provides:
    - Mel scale conversion and triangular filterbank construction.
    - Short-Time Fourier Transform (STFT) power spectrum computation.
    - Whale song acoustic vocalization synthesis (Figure 12.25).
    - Mel spectrogram patchification into AST tokens for transformer classification.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 64,
        f_min: float = 50.0,
        f_max: float = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        self.mel_filterbank = self._create_mel_filterbank()

    @staticmethod
    def hz_to_mel(f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert frequency in Hertz to Mel scale."""
        return 2595.0 * np.log10(1.0 + f / 700.0)

    @staticmethod
    def mel_to_hz(m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert Mel value to frequency in Hertz."""
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    def _create_mel_filterbank(self) -> np.ndarray:
        """Create triangular Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)."""
        num_freq_bins = self.n_fft // 2 + 1
        fft_freqs = np.linspace(0, self.sample_rate / 2.0, num_freq_bins)

        mel_min = self.hz_to_mel(self.f_min)
        mel_max = self.hz_to_mel(self.f_max)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self.mel_to_hz(mel_points)

        filterbank = np.zeros((self.n_mels, num_freq_bins))
        for m in range(self.n_mels):
            f_m_minus = hz_points[m]
            f_m = hz_points[m + 1]
            f_m_plus = hz_points[m + 2]

            for k, freq in enumerate(fft_freqs):
                if f_m_minus <= freq <= f_m:
                    filterbank[m, k] = (freq - f_m_minus) / (f_m - f_m_minus + 1e-8)
                elif f_m < freq <= f_m_plus:
                    filterbank[m, k] = (f_m_plus - freq) / (f_m_plus - f_m + 1e-8)

        return filterbank

    def compute_spectrogram(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute log-energy Mel spectrogram via Short-Time Fourier Transform (STFT).
        
        Args:
            signal: 1D array of audio samples.
            
        Returns:
            log_mel_spec: Array of shape (n_mels, num_frames).
        """
        # Hanning window
        window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(self.n_fft) / (self.n_fft - 1)))
        num_samples = len(signal)
        num_frames = max(1, (num_samples - self.n_fft) // self.hop_length + 1)

        power_spec = np.zeros((self.n_fft // 2 + 1, num_frames))
        for t in range(num_frames):
            start = t * self.hop_length
            frame = signal[start : start + self.n_fft]
            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))
            windowed = frame * window
            fft_res = np.fft.rfft(windowed, n=self.n_fft)
            power_spec[:, t] = np.abs(fft_res) ** 2

        # Apply Mel filterbank
        mel_spec = np.matmul(self.mel_filterbank, power_spec)  # (n_mels, num_frames)
        # Log compression: log(Mel + eps)
        log_mel_spec = np.log(np.maximum(mel_spec, 1e-6))
        return log_mel_spec

    def synthesize_whale_song(self, duration_sec: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesize realistic humpback whale song audio signal with characteristic
        low-frequency resonant moans, upsweeping whistles, and harmonic overtones.
        
        Returns:
            t: Time vector in seconds.
            audio: Synthetic waveform.
        """
        num_samples = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        audio = np.zeros(num_samples)

        # 1. Low frequency resonant moan (200 Hz to 450 Hz with harmonics)
        envelope1 = np.exp(-((t - 1.0) / 0.6) ** 2)
        f1 = 220.0 + 80.0 * np.sin(2.0 * np.pi * 0.5 * t)
        phase1 = 2.0 * np.pi * np.cumsum(f1) / self.sample_rate
        audio += envelope1 * (np.sin(phase1) + 0.6 * np.sin(2 * phase1) + 0.3 * np.sin(3 * phase1))

        # 2. Ascending whistle upsweep (400 Hz to 1800 Hz)
        envelope2 = np.exp(-((t - 2.5) / 0.5) ** 2)
        f2 = 400.0 + 1200.0 * np.clip((t - 2.0) / 1.0, 0.0, 1.0) ** 1.5
        phase2 = 2.0 * np.pi * np.cumsum(f2) / self.sample_rate
        audio += envelope2 * (0.8 * np.sin(phase2) + 0.4 * np.sin(2 * phase2))

        # 3. Rhythmic pulsed clicks / grunts around t=3.3
        envelope3 = np.exp(-((t - 3.4) / 0.3) ** 2)
        f3 = 180.0
        audio += envelope3 * 0.5 * np.sin(2.0 * np.pi * f3 * t) * (1.0 + 0.5 * np.sin(2.0 * np.pi * 20.0 * t))

        # Add gentle ambient oceanic background noise
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.015, size=num_samples)
        audio = audio + noise
        audio = audio / np.max(np.abs(audio) + 1e-8)

        return t, audio

    def patchify_spectrogram(
        self,
        mel_spec: np.ndarray,
        patch_size: Tuple[int, int] = (16, 16),
    ) -> np.ndarray:
        """
        Split 2D Mel spectrogram into flattened patch tokens for Audio Spectrogram Transformer (AST).
        
        Args:
            mel_spec: 2D array of shape (n_mels, num_frames).
            patch_size: Tuple (P_freq, P_time).
            
        Returns:
            patches: 2D array of shape (num_patches, P_freq * P_time).
        """
        P_f, P_t = patch_size
        F, T = mel_spec.shape
        grid_f = F // P_f
        grid_t = T // P_t
        cropped = mel_spec[:grid_f * P_f, :grid_t * P_t]

        patches = cropped.reshape(grid_f, P_f, grid_t, P_t)
        patches = patches.transpose(0, 2, 1, 3)  # (grid_f, grid_t, P_f, P_t)
        patches = patches.reshape(grid_f * grid_t, P_f * P_t)
        return patches

    # Alias for patchify_spectrogram
    spectrogram_to_patches = patchify_spectrogram


# =============================================================================
# 5. Multimodal Tasks & Unified Vocabulary (Section 12.4.5, Figure 12.27)
# =============================================================================

class MultimodalTokenManager:
    """
    Unified Token Manager for Vision and Language Transformers (CM3 / CM3Leon style).
    
    Creates a joint vocabulary combining natural language text tokens and discrete
    image codebook tokens:
        Vocabulary V_total = V_text union V_image union V_special
    """

    def __init__(
        self,
        text_vocab_size: int = 1000,
        image_codebook_size: int = 512,
    ):
        self.text_vocab_size = text_vocab_size
        self.image_codebook_size = image_codebook_size

        # Special tokens
        self.BOS = 0
        self.EOS = 1
        self.BOI = 2  # Begin Image
        self.EOI = 3  # End Image
        self.MASK = 4
        self.num_special = 5

        self.text_offset = self.num_special
        self.image_offset = self.num_special + text_vocab_size
        self.total_vocab_size = self.num_special + text_vocab_size + image_codebook_size

    def encode_text_tokens(self, text_token_ids: List[int]) -> List[int]:
        """Shift text token IDs by text offset."""
        return [self.text_offset + t for t in text_token_ids]

    def encode_image_tokens(self, codebook_indices: List[int]) -> List[int]:
        """Wrap image codebook tokens between BOI and EOI markers."""
        return [self.BOI] + [self.image_offset + idx for idx in codebook_indices] + [self.EOI]

    def format_sequence(
        self,
        text_ids: Optional[List[int]] = None,
        image_ids: Optional[List[int]] = None,
        task: str = "text_to_image",
    ) -> List[int]:
        """Format joint sequence according to multimodal task."""
        seq = [self.BOS]
        if task == "text_to_image":
            # [BOS] + text tokens + [BOI] + image tokens + [EOI] + [EOS]
            if text_ids is not None:
                seq.extend(self.encode_text_tokens(text_ids))
            if image_ids is not None:
                seq.extend(self.encode_image_tokens(image_ids))
        elif task == "image_to_text":
            # [BOS] + [BOI] + image tokens + [EOI] + text tokens + [EOS]
            if image_ids is not None:
                seq.extend(self.encode_image_tokens(image_ids))
            if text_ids is not None:
                seq.extend(self.encode_text_tokens(text_ids))
        elif task == "inpainting":
            # [BOS] + masked image tokens + completed image tokens + [EOS]
            if image_ids is not None:
                seq.extend(self.encode_image_tokens(image_ids))
        seq.append(self.EOS)
        return seq


# =============================================================================
# 6. High-Resolution Figure Reproductions (Figures 12.22 to 12.27)
# =============================================================================

def generate_figure_12_22(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.22: Vision Transformer (ViT) Architecture for Classification.
    
    Illustrates image patch extraction, flattening, linear embedding projection,
    learned 1D positional encodings, learnable <class> token, transformer encoder,
    and linear-softmax (LSM) classification output vector c.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    # Colors
    c_img = "#E2E8F0"
    c_patch = "#94A3B8"
    c_flat = "#CBD5E1"
    c_emb = "#93C5FD"
    c_pos = "#FDE68A"
    c_cls = "#FCA5A5"
    c_trans = "#C7D2FE"
    c_lsm = "#86EFAC"
    c_out = "#1E293B"

    # 1. Input Image (grid of 3x3 patches) at bottom right
    img_x, img_y = 6.2, 0.4
    patch_w, patch_h = 0.55, 0.55
    ax.text(img_x + 0.85, img_y - 0.25, "Input image $x$", ha="center", fontsize=11, fontweight="bold", color=c_out)
    for r in range(3):
        for c in range(3):
            px = img_x + c * (patch_w + 0.05)
            py = img_y + (2 - r) * (patch_h + 0.05)
            rect = patches.Rectangle((px, py), patch_w, patch_h, facecolor=c_img, edgecolor=c_patch, lw=1.5)
            ax.add_patch(rect)
            ax.text(px + patch_w/2, py + patch_h/2, f"$x_{r*3+c+1}$", ha="center", va="center", fontsize=8, color="#334155")

    # Arrow from image to flattened tokens
    ax.annotate("", xy=(5.5, 2.3), xytext=(img_x + 0.85, img_y + 1.8),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#64748B", connectionstyle="arc3,rad=-0.15"))

    # 2. Patch Flatten & Embedding boxes across x: 1.0 to 9.5
    # Token positions: cls at 1.2, then patches 1, 2, ..., N at 2.6, 4.0, 5.4, 6.8, 8.2, 9.6
    pos_x = [1.2, 2.7, 4.2, 5.7, 7.2, 8.7]
    labels = ["$\\langle \\mathrm{class} \\rangle$", "patch 1", "patch 2", "patch 3", "...", "patch $N$"]

    # Flatten row
    for i in range(1, len(pos_x)):
        x_c = pos_x[i]
        lbl = "..." if i == 4 else "flatten"
        rect = patches.FancyBboxPatch((x_c - 0.55, 2.0), 1.1, 0.45, boxstyle="round,pad=0.05",
                                      facecolor=c_flat, edgecolor="#64748B", lw=1.2)
        ax.add_patch(rect)
        ax.text(x_c, 2.22, lbl, ha="center", va="center", fontsize=9, fontweight="bold")
        # Arrow from flatten to embedding
        ax.annotate("", xy=(x_c, 2.75), xytext=(x_c, 2.45),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#64748B"))

    # <class> token at pos_x[0]
    rect_cls = patches.FancyBboxPatch((pos_x[0] - 0.65, 2.0), 1.3, 0.45, boxstyle="round,pad=0.05",
                                     facecolor=c_cls, edgecolor="#DC2626", lw=1.5)
    ax.add_patch(rect_cls)
    ax.text(pos_x[0], 2.22, "$\\langle \\mathrm{class} \\rangle$", ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.annotate("", xy=(pos_x[0], 2.75), xytext=(pos_x[0], 2.45),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#DC2626"))

    # Embedding row
    for i, x_c in enumerate(pos_x):
        lbl = "..." if i == 4 else "embedding"
        col = c_cls if i == 0 else c_emb
        edge = "#DC2626" if i == 0 else "#2563EB"
        rect = patches.FancyBboxPatch((x_c - 0.6, 2.75), 1.2, 0.45, boxstyle="round,pad=0.05",
                                      facecolor=col, edgecolor=edge, lw=1.2)
        ax.add_patch(rect)
        ax.text(x_c, 2.97, lbl, ha="center", va="center", fontsize=9, fontweight="bold")

        # Plus circle for positional encoding
        circle = patches.Circle((x_c, 3.6), 0.18, facecolor=c_pos, edgecolor="#D97706", lw=1.2)
        ax.add_patch(circle)
        ax.text(x_c, 3.6, "+", ha="center", va="center", fontsize=11, fontweight="bold", color="#B45309")

        # Arrow from embedding to +
        ax.annotate("", xy=(x_c, 3.42), xytext=(x_c, 3.20),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#64748B"))

        # Arrow from + to Transformer Encoder
        ax.annotate("", xy=(x_c, 4.3), xytext=(x_c, 3.78),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#64748B"))

    # Learned positional encoding side annotation
    ax.annotate("learned\npositional\nencoding", xy=(0.8, 3.6), xytext=(0.1, 3.6),
                va="center", ha="left", fontsize=9.5, fontweight="bold", color="#B45309",
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#D97706"))

    # 3. Transformer Encoder big block
    enc_x, enc_y = 0.5, 4.3
    enc_w, enc_h = 9.8, 1.8
    rect_enc = patches.FancyBboxPatch((enc_x, enc_y), enc_w, enc_h, boxstyle="round,pad=0.1",
                                      facecolor=c_trans, edgecolor="#4F46E5", lw=2)
    ax.add_patch(rect_enc)
    ax.text(enc_x + enc_w/2, enc_y + enc_h/2, "transformer encoder", ha="center", va="center",
            fontsize=16, fontweight="bold", color="#312E81")

    # 4. Outputs at top
    # <class> token output routed through LSM
    cls_top_x = pos_x[0]
    ax.annotate("", xy=(cls_top_x, 6.7), xytext=(cls_top_x, enc_y + enc_h),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#16A34A"))

    rect_lsm = patches.FancyBboxPatch((cls_top_x - 0.6, 6.7), 1.2, 0.5, boxstyle="round,pad=0.05",
                                      facecolor=c_lsm, edgecolor="#16A34A", lw=1.5)
    ax.add_patch(rect_lsm)
    ax.text(cls_top_x, 6.95, "LSM", ha="center", va="center", fontsize=11, fontweight="bold", color="#14532D")

    # Output class vector c
    ax.annotate("", xy=(cls_top_x, 7.8), xytext=(cls_top_x, 7.2),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#14532D"))
    ax.text(cls_top_x, 8.0, "$\\mathbf{c}$", ha="center", va="center", fontsize=15, fontweight="bold", color="#14532D")

    # Other outputs (optional, unused for classification)
    for i in range(1, len(pos_x)):
        x_c = pos_x[i]
        ax.annotate("", xy=(x_c, 6.5), xytext=(x_c, enc_y + enc_h),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#94A3B8", linestyle=":"))

    ax.set_title("Figure 12.22: Vision Transformer (ViT) Architecture for Image Classification",
                 fontsize=13, fontweight="bold", pad=12)

    _save_figure(fig, "fig_12_22_vit_architecture", save_dir)
    return fig


def generate_figure_12_23(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.23: Raster scan ordering of pixels in a 2D image.
    
    Shows a 4x4 grid of image pixels labeled x_1 through x_16 with sequential
    raster scan path arrows clearly illustrating the 1D serialization of 2D data.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=300)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")

    colors = [
        "#EFF6FF", "#DBEAFE", "#BFDBFE", "#93C5FD",
        "#60A5FA", "#3B82F6", "#2563EB", "#1D4ED8",
        "#1E40AF", "#1E3A8A", "#172554", "#0F172A",
        "#312E81", "#3730A3", "#4338CA", "#4F46E5",
    ]

    # Draw 4x4 grid
    for r in range(4):
        for c in range(4):
            idx = r * 4 + c + 1
            x_pos = c
            y_pos = 3 - r
            rect = patches.Rectangle((x_pos, y_pos), 0.95, 0.95, facecolor="#F8FAFC",
                                     edgecolor="#64748B", lw=1.5)
            ax.add_patch(rect)
            ax.text(x_pos + 0.475, y_pos + 0.475, f"$x_{{{idx}}}$", ha="center", va="center",
                    fontsize=13, fontweight="bold", color="#0F172A")

    # Draw raster scan arrows
    for r in range(4):
        y_pos = 3 - r + 0.475
        # Horizontal arrows in row
        for c in range(3):
            ax.annotate("", xy=(c + 1.0, y_pos), xytext=(c + 0.75, y_pos),
                        arrowprops=dict(arrowstyle="->", lw=1.8, color="#DC2626"))
        # Diagonal / wrap-around arrow from end of row to start of next row
        if r < 3:
            start_x, start_y = 3.95, y_pos
            end_x, end_y = 0.0, y_pos - 1.0
            ax.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                        arrowprops=dict(arrowstyle="->", lw=1.4, color="#DC2626",
                                        linestyle="--", connectionstyle="arc3,rad=-0.4"))

    ax.set_title("Figure 12.23: Raster Scan Ordering of Pixels in a 2D Image",
                 fontsize=12, fontweight="bold", pad=15)

    _save_figure(fig, "fig_12_23_raster_scan", save_dir)
    return fig


def generate_figure_12_24(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.24: Sampling an Image from an Autoregressive Model.
    
    Illustrates sequential patch-by-patch sampling:
    - Step 1: sampling x_1 ~ p(x_1)
    - Step 2: sampling x_2 ~ p(x_2 | x_1)
    - Step 3: intermediate step x_k ~ p(x_k | x_{<k})
    - Step 4: fully generated complete image.
    """
    setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0), dpi=300)

    # 4x4 sample target pattern (a simple high-contrast recognizable icon: smiley / heart)
    target = np.array([
        [0.2, 0.8, 0.8, 0.2],
        [0.8, 0.2, 0.2, 0.8],
        [0.8, 0.8, 0.8, 0.8],
        [0.8, 0.2, 0.2, 0.8],
    ])

    steps = [1, 2, 7, 16]
    subtitles = [
        "Step 1: $x_1 \\sim p(x_1)$",
        "Step 2: $x_2 \\sim p(x_2 \\mid x_1)$",
        "Step 7: $x_7 \\sim p(x_7 \\mid x_{1:6})$",
        "Step 16: Complete Image",
    ]

    for ax_idx, (k, subtitle) in enumerate(zip(steps, subtitles)):
        ax = axes[ax_idx]
        ax.set_xlim(-0.2, 4.2)
        ax.set_ylim(-0.2, 4.2)
        ax.set_aspect("equal")
        ax.axis("off")

        for r in range(4):
            for c in range(4):
                idx = r * 4 + c + 1
                x_pos = c
                y_pos = 3 - r

                if idx < k:
                    # Already generated pixel
                    val = target[r, c]
                    color = plt.cm.Blues(0.3 + 0.6 * val)
                    rect = patches.Rectangle((x_pos, y_pos), 0.95, 0.95, facecolor=color,
                                             edgecolor="#1E3A8A", lw=1.2)
                    ax.add_patch(rect)
                    ax.text(x_pos + 0.475, y_pos + 0.475, f"$x_{{{idx}}}$", ha="center", va="center",
                            fontsize=9, color="#0F172A", fontweight="bold")
                elif idx == k and k < 16:
                    # Current pixel being sampled
                    rect = patches.Rectangle((x_pos, y_pos), 0.95, 0.95, facecolor="#FDE047",
                                             edgecolor="#CA8A04", lw=2.5)
                    ax.add_patch(rect)
                    ax.text(x_pos + 0.475, y_pos + 0.475, f"$x_{{{idx}}}$", ha="center", va="center",
                            fontsize=10, color="#854D0E", fontweight="bold")
                elif idx == 16 and k == 16:
                    val = target[r, c]
                    color = plt.cm.Blues(0.3 + 0.6 * val)
                    rect = patches.Rectangle((x_pos, y_pos), 0.95, 0.95, facecolor=color,
                                             edgecolor="#1E3A8A", lw=1.2)
                    ax.add_patch(rect)
                    ax.text(x_pos + 0.475, y_pos + 0.475, f"$x_{{{idx}}}$", ha="center", va="center",
                            fontsize=9, color="#0F172A", fontweight="bold")
                else:
                    # Masked / future pixel (not yet generated)
                    rect = patches.Rectangle((x_pos, y_pos), 0.95, 0.95, facecolor="#F1F5F9",
                                             edgecolor="#CBD5E1", lw=1.0, hatch="//")
                    ax.add_patch(rect)

        ax.set_title(subtitle, fontsize=10.5, fontweight="bold", pad=8)

    fig.suptitle("Figure 12.24: Sampling an Image from an Autoregressive Model in Raster Scan Order",
                 fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout()

    _save_figure(fig, "fig_12_24_autoregressive_image_sampling", save_dir)
    return fig


def generate_figure_12_25(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.25: Example Mel Spectrogram of a Humpback Whale Song.
    
    Generates high-resolution time-frequency representation showing characteristic
    whale vocalisations (harmonic frequency contours, upsweeping moans, pulsating calls).
    """
    setup_style()
    processor = AudioMelSpectrogramProcessor(sample_rate=16000, n_fft=512, hop_length=160, n_mels=80)
    t, audio = processor.synthesize_whale_song(duration_sec=4.0)
    mel_spec = processor.compute_spectrogram(audio)

    fig, ax = plt.subplots(figsize=(10, 5.0), dpi=300)

    # Time in seconds along x-axis, Mel frequency bands along y-axis
    time_extent = [0, 4.0, processor.f_min, processor.f_max]
    im = ax.imshow(
        mel_spec,
        origin="lower",
        aspect="auto",
        extent=time_extent,
        cmap="magma",
        interpolation="bilinear",
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Log Mel Energy (dB)", fontsize=10, fontweight="bold")

    ax.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency (Hz, Mel-scale spacing)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 12.25: Example Mel Spectrogram of a Humpback Whale Song",
                 fontsize=12.5, fontweight="bold", pad=12)

    # Annotations highlighting features of whale song
    ax.annotate("Harmonic moan", xy=(1.0, 450), xytext=(0.4, 2500),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#FDE047"),
                fontsize=9.5, fontweight="bold", color="#FDE047",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1E293B", edgecolor="#FDE047", alpha=0.85))

    ax.annotate("Ascending upsweep", xy=(2.6, 1400), xytext=(2.2, 4500),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#FDE047"),
                fontsize=9.5, fontweight="bold", color="#FDE047",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1E293B", edgecolor="#FDE047", alpha=0.85))

    ax.annotate("Pulsed grunt", xy=(3.4, 300), xytext=(3.1, 2000),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#FDE047"),
                fontsize=9.5, fontweight="bold", color="#FDE047",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1E293B", edgecolor="#FDE047", alpha=0.85))

    plt.tight_layout()

    _save_figure(fig, "fig_12_25_mel_spectrogram", save_dir)
    return fig


def generate_figure_12_26(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.26: Architecture of Vall-E Text-to-Speech Model.
    
    Shows high-level architecture: text prompt tokens + acoustic prompt tokens
    -> Transformer -> audio decoder -> synthesized speech waveform.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6.0), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    c_box = "#EFF6FF"
    c_edge = "#3B82F6"
    c_trans = "#C7D2FE"
    c_speech = "#86EFAC"

    # 1. Inputs at bottom
    # Left: Text prompt tokens
    rect_text = patches.FancyBboxPatch((1.0, 1.0), 3.0, 1.0, boxstyle="round,pad=0.1",
                                       facecolor=c_box, edgecolor=c_edge, lw=1.8)
    ax.add_patch(rect_text)
    ax.text(2.5, 1.5, "text prompt\ntokens", ha="center", va="center", fontsize=12, fontweight="bold", color="#1E3A8A")

    # Right: Acoustic prompt -> discrete tokenizer -> acoustic prompt tokens
    rect_prompt = patches.FancyBboxPatch((6.0, 0.2), 3.0, 0.8, boxstyle="round,pad=0.08",
                                         facecolor="#F1F5F9", edgecolor="#64748B", lw=1.4)
    ax.add_patch(rect_prompt)
    ax.text(7.5, 0.6, "acoustic prompt", ha="center", va="center", fontsize=11, fontweight="bold", color="#334155")

    ax.annotate("", xy=(7.5, 1.4), xytext=(7.5, 1.0),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#64748B"))

    rect_tok = patches.FancyBboxPatch((6.2, 1.4), 2.6, 0.65, boxstyle="round,pad=0.05",
                                      facecolor="#FEF08A", edgecolor="#CA8A04", lw=1.4)
    ax.add_patch(rect_tok)
    ax.text(7.5, 1.72, "discrete\ntokenizer", ha="center", va="center", fontsize=10, fontweight="bold", color="#854D0E")

    ax.annotate("", xy=(7.5, 2.5), xytext=(7.5, 2.05),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#64748B"))

    # 2. Main Transformer block in middle
    ax.annotate("", xy=(2.5, 3.0), xytext=(2.5, 2.0),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#1D4ED8"))

    rect_trans = patches.FancyBboxPatch((1.5, 3.0), 7.0, 1.6, boxstyle="round,pad=0.15",
                                        facecolor=c_trans, edgecolor="#4F46E5", lw=2.2)
    ax.add_patch(rect_trans)
    ax.text(5.0, 3.8, "transformer", ha="center", va="center", fontsize=17, fontweight="bold", color="#312E81")

    # 3. Audio Decoder & Synthesized Speech at top
    ax.annotate("", xy=(5.0, 5.2), xytext=(5.0, 4.6),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#4F46E5"))

    rect_dec = patches.FancyBboxPatch((3.5, 5.2), 3.0, 0.8, boxstyle="round,pad=0.08",
                                      facecolor="#FDE68A", edgecolor="#D97706", lw=1.8)
    ax.add_patch(rect_dec)
    ax.text(5.0, 5.6, "audio decoder", ha="center", va="center", fontsize=12, fontweight="bold", color="#92400E")

    ax.annotate("", xy=(5.0, 6.5), xytext=(5.0, 6.0),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#16A34A"))

    rect_speech = patches.FancyBboxPatch((3.2, 6.5), 3.6, 0.7, boxstyle="round,pad=0.08",
                                        facecolor=c_speech, edgecolor="#16A34A", lw=2.0)
    ax.add_patch(rect_speech)
    ax.text(5.0, 6.85, "synthesized speech", ha="center", va="center", fontsize=12.5, fontweight="bold", color="#14532D")

    ax.set_title("Figure 12.26: Architecture of the Vall-E Text-to-Speech Model",
                 fontsize=13, fontweight="bold", pad=12)

    _save_figure(fig, "fig_12_26_valle_architecture", save_dir)
    return fig


def generate_figure_12_27(save_dir: Optional[str] = None) -> plt.Figure:
    """
    Figure 12.27: CM3Leon Multimodal Model Tasks.
    
    Illustrates 4 diverse joint text-image capabilities:
    (a) Text-to-Image Generation
    (b) Image-to-Text Captioning
    (c) Image Inpainting (Completion)
    (d) Instruction-Guided Image Editing
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=300)

    # 1. Text-to-Image
    ax = axes[0, 0]
    ax.axis("off")
    ax.text(0.5, 0.95, "(a) Text-to-Image Generation", ha="center", va="top", fontsize=11.5, fontweight="bold")
    ax.text(0.5, 0.85, 'Prompt: "A cute retriever puppy wearing blue sunglasses"', ha="center", va="top",
            fontsize=9.5, style="italic", color="#1E3A8A", bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF", edgecolor="#93C5FD"))
    # Dummy generated puppy illustration
    rect = patches.FancyBboxPatch((0.2, 0.05), 0.6, 0.65, boxstyle="round,pad=0.05",
                                  facecolor="#FEF3C7", edgecolor="#D97706", lw=1.5)
    ax.add_patch(rect)
    # Draw simple sunglasses & puppy face
    ax.add_patch(patches.Circle((0.5, 0.42), 0.22, facecolor="#FBBF24", edgecolor="#B45309", lw=1.2))
    ax.add_patch(patches.Rectangle((0.38, 0.43), 0.1, 0.06, facecolor="#1E3A8A"))
    ax.add_patch(patches.Rectangle((0.52, 0.43), 0.1, 0.06, facecolor="#1E3A8A"))
    ax.plot([0.48, 0.52], [0.46, 0.46], color="#1E3A8A", lw=2)
    ax.text(0.5, 0.28, "Generated Image", ha="center", fontsize=9, fontweight="bold", color="#78350F")

    # 2. Image-to-Text Captioning
    ax = axes[0, 1]
    ax.axis("off")
    ax.text(0.5, 0.95, "(b) Image-to-Text Captioning", ha="center", va="top", fontsize=11.5, fontweight="bold")
    # Sunflower image
    rect = patches.FancyBboxPatch((0.2, 0.35), 0.6, 0.45, boxstyle="round,pad=0.05",
                                  facecolor="#DCFCE7", edgecolor="#16A34A", lw=1.5)
    ax.add_patch(rect)
    ax.add_patch(patches.Circle((0.5, 0.57), 0.12, facecolor="#FACC15", edgecolor="#EAB308", lw=1.2))
    ax.add_patch(patches.Circle((0.5, 0.57), 0.06, facecolor="#713F12"))
    ax.text(0.5, 0.40, "Input Image", ha="center", fontsize=9, fontweight="bold", color="#14532D")
    # Generated Caption
    ax.text(0.5, 0.18, 'Caption: "A vibrant yellow sunflower blooming\\nin a lush green summer meadow."',
            ha="center", va="center", fontsize=9.5, style="italic", color="#14532D",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FDF4", edgecolor="#86EFAC"))

    # 3. Image Inpainting
    ax = axes[1, 0]
    ax.axis("off")
    ax.text(0.5, 0.95, "(c) Image Inpainting (Completion)", ha="center", va="top", fontsize=11.5, fontweight="bold")
    # Left: Masked Image
    rect_l = patches.Rectangle((0.08, 0.2), 0.38, 0.55, facecolor="#E0E7FF", edgecolor="#4F46E5", lw=1.5)
    ax.add_patch(rect_l)
    mask = patches.Rectangle((0.17, 0.35), 0.2, 0.25, facecolor="#94A3B8", hatch="//", edgecolor="#475569")
    ax.add_patch(mask)
    ax.text(0.27, 0.12, "Masked Input", ha="center", fontsize=9, fontweight="bold", color="#312E81")

    # Arrow
    ax.annotate("", xy=(0.54, 0.47), xytext=(0.47, 0.47),
                arrowprops=dict(arrowstyle="->", lw=2, color="#4F46E5"))

    # Right: Inpainted Image
    rect_r = patches.Rectangle((0.55, 0.2), 0.38, 0.55, facecolor="#E0E7FF", edgecolor="#4F46E5", lw=1.5)
    ax.add_patch(rect_r)
    infilled = patches.Circle((0.74, 0.47), 0.10, facecolor="#EC4899", edgecolor="#BE185D", lw=1.5)
    ax.add_patch(infilled)
    ax.text(0.74, 0.12, "Completed Output", ha="center", fontsize=9, fontweight="bold", color="#312E81")

    # 4. Instruction-Guided Image Editing
    ax = axes[1, 1]
    ax.axis("off")
    ax.text(0.5, 0.95, "(d) Instruction-Guided Image Editing", ha="center", va="top", fontsize=11.5, fontweight="bold")
    ax.text(0.5, 0.85, 'Instruction: "Change the season to autumn"', ha="center", va="top",
            fontsize=9.5, style="italic", color="#9A3412",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEDD5", edgecolor="#FDBA74"))

    # Left: Summer tree (green)
    rect_s = patches.Rectangle((0.08, 0.18), 0.38, 0.52, facecolor="#F8FAFC", edgecolor="#64748B", lw=1.2)
    ax.add_patch(rect_s)
    ax.add_patch(patches.Rectangle((0.25, 0.22), 0.04, 0.18, facecolor="#78350F"))
    ax.add_patch(patches.Circle((0.27, 0.48), 0.12, facecolor="#22C55E", edgecolor="#15803D"))
    ax.text(0.27, 0.10, "Original (Summer)", ha="center", fontsize=9, color="#334155")

    # Arrow
    ax.annotate("", xy=(0.54, 0.44), xytext=(0.47, 0.44),
                arrowprops=dict(arrowstyle="->", lw=2, color="#EA580C"))

    # Right: Autumn tree (orange/golden)
    rect_a = patches.Rectangle((0.55, 0.18), 0.38, 0.52, facecolor="#FFF7ED", edgecolor="#EA580C", lw=1.5)
    ax.add_patch(rect_a)
    ax.add_patch(patches.Rectangle((0.72, 0.22), 0.04, 0.18, facecolor="#78350F"))
    ax.add_patch(patches.Circle((0.74, 0.48), 0.12, facecolor="#F97316", edgecolor="#C2410C"))
    ax.text(0.74, 0.10, "Edited (Autumn)", ha="center", fontsize=9, fontweight="bold", color="#9A3412")

    fig.suptitle("Figure 12.27: CM3Leon Multimodal Model Performing Joint Vision and Language Tasks",
                 fontsize=12.5, fontweight="bold", y=0.99)
    plt.tight_layout()

    _save_figure(fig, "fig_12_27_cm3leon_tasks", save_dir)
    return fig


if __name__ == "__main__":
    print("Generating Figure 12.22...")
    generate_figure_12_22()
    print("Generating Figure 12.23...")
    generate_figure_12_23()
    print("Generating Figure 12.24...")
    generate_figure_12_24()
    print("Generating Figure 12.25...")
    generate_figure_12_25()
    print("Generating Figure 12.26...")
    generate_figure_12_26()
    print("Generating Figure 12.27...")
    generate_figure_12_27()
    print("All Chapter 12.4 figures generated successfully!")
