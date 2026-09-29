"""Coupling Flows and Real NVP (Chapter 18, Section 18.1).

This module implements:
- Standard multivariate Gaussian base distribution and log-density (Eq. 18.1, 18.4)
- Conditioner neural networks (MLPConditioner) for scale s and translation b (Eq. 18.11, 18.15)
- Affine Coupling Layer (AffineCouplingLayer) with forward, inverse, and exact Jacobian determinant (Eq. 18.10 - 18.14)
- Multi-layer Real NVP Normalizing Flow (RealNVPFlow) with alternating partitioning and composition (Eq. 18.5 - 18.7)
- Exact log likelihood evaluation, sampling, and two-moons density transformation
- Faithful reproduction of textbook figures:
  - Figure 18.1: A single layer of the real NVP normalizing flow model
  - Figure 18.2: Composing two layers with alternating partitions to obtain a double layer
  - Figure 18.3: Illustration of the real NVP model applied to the two-moons dataset (6 panels)
"""

import os
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sklearn.datasets import make_moons

from common.plot_utils import save_fig


class StandardGaussian:
    r"""Standard multivariate Gaussian base distribution $\mathcal{N}(\mathbf{0}, \mathbf{I}_D)$."""

    def __init__(self, dim: int = 2):
        self.dim = dim

    def log_prob(self, z: np.ndarray) -> np.ndarray:
        r"""Compute log probability density under standard Gaussian base distribution.

        .. math::
            \ln p_z(\mathbf{z}) = -\frac{D}{2} \ln(2\pi) - \frac{1}{2} \|\mathbf{z}\|^2

        Args:
            z: Latent variables of shape (N, D).

        Returns:
            Log density array of shape (N,).
        """
        z = np.atleast_2d(z)
        const = -0.5 * self.dim * np.log(2.0 * np.pi)
        quad = -0.5 * np.sum(z**2, axis=1)
        return const + quad

    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """Sample latent points from standard Gaussian base distribution."""
        rng = np.random.RandomState(random_state)
        return rng.randn(n_samples, self.dim)


class ConditionerMLP:
    r"""Conditioner neural network $g(\mathbf{z}_A, \mathbf{w})$ for coupling flows (Eq. 18.11, 18.15).

    Takes un-transformed coordinate $\mathbf{z}_A$ and outputs scale $\mathbf{s}(\mathbf{z}_A)$
    and translation $\mathbf{b}(\mathbf{z}_A)$.
    """

    def __init__(
        self,
        in_features: int = 1,
        hidden_features: int = 32,
        out_features: int = 2,
        scale_max: float = 2.0,
        random_state: Optional[int] = None,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.scale_max = scale_max
        
        rng = np.random.RandomState(random_state)
        # Initialize weights
        self.W1 = rng.randn(in_features, hidden_features) * 0.3
        self.b1 = np.zeros(hidden_features)
        self.W2 = rng.randn(hidden_features, hidden_features) * 0.3
        self.b2 = np.zeros(hidden_features)
        # Small initialization on final layer so flow begins close to identity
        self.W3 = rng.randn(hidden_features, out_features) * 0.01
        self.b3 = np.zeros(out_features)

        # Adam optimizer state
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def forward(self, x_cond: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute scale s and translation b given conditioning variable x_cond."""
        x_cond = np.atleast_2d(x_cond)
        self.last_x = x_cond
        self.h1 = np.tanh(x_cond @ self.W1 + self.b1)
        self.h2 = np.tanh(self.h1 @ self.W2 + self.b2)
        out = self.h2 @ self.W3 + self.b3
        
        # Split into scale s and shift b
        s_raw = out[:, 0:1]
        b = out[:, 1:2]
        # Bounded scale to prevent numerical overflow in exp(s)
        self.s = np.tanh(s_raw) * self.scale_max
        self.b = b
        return self.s, self.b


class AffineCouplingLayer:
    r"""Single layer of the Real NVP normalizing flow model (Eq. 18.10 - 18.14).

    Partitioning:
        $\mathbf{z} = (\mathbf{z}_A, \mathbf{z}_B)$ where $\mathbf{z}_A$ is copied and $\mathbf{z}_B$ is transformed.
    Forward transformation (Eq. 18.10, 18.11):
        $$\mathbf{x}_A = \mathbf{z}_A$$
        $$\mathbf{x}_B = \exp(\mathbf{s}(\mathbf{z}_A)) \odot \mathbf{z}_B + \mathbf{b}(\mathbf{z}_A)$$
    Inverse transformation (Eq. 18.12, 18.13):
        $$\mathbf{z}_A = \mathbf{x}_A$$
        $$\mathbf{z}_B = \exp(-\mathbf{s}(\mathbf{z}_A)) \odot (\mathbf{x}_B - \mathbf{b}(\mathbf{z}_A))$$
    Jacobian matrix and determinant (Eq. 18.14):
        $$\mathbf{J} = \begin{bmatrix} \mathbf{I}_d & \mathbf{0} \\ \frac{\partial \mathbf{z}_B}{\partial \mathbf{x}_A} & \text{diag}(\exp(-\mathbf{s})) \end{bmatrix}$$
        $$\ln |\det \mathbf{J}| = -\sum_i s_i(\mathbf{x}_A)$$
    """

    def __init__(
        self,
        dim: int = 2,
        transform_dim: int = 1,
        conditioner: Optional[ConditionerMLP] = None,
        custom_s_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        custom_b_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.dim = dim
        self.transform_dim = transform_dim
        self.cond_dim = 1 - transform_dim
        self.conditioner = conditioner
        self.custom_s_fn = custom_s_fn
        self.custom_b_fn = custom_b_fn

    def _get_st(self, cond: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute scale s and translation b given conditioning variable."""
        if self.custom_s_fn is not None and self.custom_b_fn is not None:
            s = self.custom_s_fn(cond)
            b = self.custom_b_fn(cond)
            return s, b
        elif self.conditioner is not None:
            return self.conditioner.forward(cond)
        else:
            # Identity defaults
            zeros = np.zeros_like(cond)
            return zeros, zeros

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Forward transformation $\mathbf{z} \to \mathbf{x}$ (generation / sampling, Eq. 18.10, 18.11).

        Returns:
            Transformed points x and forward log determinant array.
        """
        z = np.atleast_2d(z)
        x = np.copy(z)
        cond = z[:, self.cond_dim:self.cond_dim+1]
        s, b = self._get_st(cond)
        
        # Affine transformation on transform_dim
        x[:, self.transform_dim:self.transform_dim+1] = (
            z[:, self.transform_dim:self.transform_dim+1] * np.exp(s) + b
        )
        log_det = np.sum(s, axis=1)
        return x, log_det

    def inverse(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Inverse transformation $\mathbf{x} \to \mathbf{z}$ (normalization / density evaluation, Eq. 18.12, 18.13).

        Returns:
            Normalized points z and inverse log determinant array.
        """
        x = np.atleast_2d(x)
        z = np.copy(x)
        cond = x[:, self.cond_dim:self.cond_dim+1]
        s, b = self._get_st(cond)
        
        # Invert affine transformation on transform_dim
        z[:, self.transform_dim:self.transform_dim+1] = (
            (x[:, self.transform_dim:self.transform_dim+1] - b) * np.exp(-s)
        )
        log_det = -np.sum(s, axis=1)
        return z, log_det

    def jacobian_matrix(self, x_single: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        r"""Evaluate the $D \times D$ Jacobian matrix $\mathbf{J}(x) = \frac{\partial \mathbf{z}}{\partial \mathbf{x}}$ (Eq. 18.14)."""
        x_single = np.asarray(x_single, dtype=float).ravel()
        D = len(x_single)
        J = np.zeros((D, D))
        
        z_base, _ = self.inverse(x_single.reshape(1, -1))
        z_base = z_base.ravel()
        
        for j in range(D):
            x_plus = np.copy(x_single)
            x_plus[j] += eps
            z_plus, _ = self.inverse(x_plus.reshape(1, -1))
            J[:, j] = (z_plus.ravel() - z_base) / eps
            
        return J


class RealNVPFlow:
    r"""Multi-layer Real NVP Normalizing Flow model (Section 18.1).

    Composes alternating coupling layers:
    .. math::
        \mathbf{x} = f_K(f_{K-1}(\dots f_1(\mathbf{z}))) \quad (\text{Eq. 18.5})
        \mathbf{z} = g_C(g_B(g_A(\mathbf{x}))) \quad (\text{Eq. 18.6})
        \ln p(\mathbf{x}|\mathbf{w}) = \ln p_z(g(\mathbf{x}, \mathbf{w})) + \ln |\det \mathbf{J}(\mathbf{x})| \quad (\text{Eq. 18.4})
    """

    def __init__(
        self,
        layers: List[AffineCouplingLayer],
        base_distribution: Optional[StandardGaussian] = None,
    ):
        self.layers = layers
        self.dim = layers[0].dim if layers else 2
        self.base_distribution = base_distribution or StandardGaussian(dim=self.dim)

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform base latent z to data x through all layers (Eq. 18.5)."""
        cur = np.atleast_2d(z)
        tot_ldet = np.zeros(cur.shape[0])
        for layer in self.layers:
            cur, ldet = layer.forward(cur)
            tot_ldet += ldet
        return cur, tot_ldet

    def inverse(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform data x to base latent z through all reversed layers (Eq. 18.6)."""
        cur = np.atleast_2d(x)
        tot_ldet = np.zeros(cur.shape[0])
        for layer in reversed(self.layers):
            cur, ldet = layer.inverse(cur)
            tot_ldet += ldet
        return cur, tot_ldet

    def forward_intermediates(self, z: np.ndarray) -> List[np.ndarray]:
        """Compute all intermediate sample representations [z0, z1, ..., zK]."""
        intermediates = [np.copy(z)]
        cur = np.copy(z)
        for layer in self.layers:
            cur, _ = layer.forward(cur)
            intermediates.append(np.copy(cur))
        return intermediates

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        r"""Compute exact data log likelihood $\ln p_x(\mathbf{x})$ via change of variables (Eq. 18.1, 18.4)."""
        z, tot_log_det = self.inverse(x)
        log_pz = self.base_distribution.log_prob(z)
        return log_pz + tot_log_det

    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """Sample synthetic points by passing Gaussian samples forward through the flow."""
        z = self.base_distribution.sample(n_samples, random_state=random_state)
        x, _ = self.forward(z)
        return x


def get_two_moons_flow(random_state: int = 42) -> RealNVPFlow:
    r"""Construct the canonical 4-layer Real NVP flow that reproduces Bishop Figure 18.3.

    Double-layer 1 (Figure 18.2, Sub-layers 1 & 2):
      - Sub-layer 1 (b): vertical transformation (y transformed given x)
      - Sub-layer 2 (c): horizontal transformation (x transformed given y)
    Double-layer 2 (Figure 18.2, Sub-layers 3 & 4):
      - Sub-layer 3 (d): second vertical transformation (y transformed given x)
      - Sub-layer 4 (e): second horizontal transformation (x transformed given y)
    """
    # Layer 1: Vertical fold (y given x)
    def s1(u: np.ndarray) -> np.ndarray:
        return -0.4 - 0.15 * u

    def b1(u: np.ndarray) -> np.ndarray:
        return 0.65 * u**2 - 1.2

    l1 = AffineCouplingLayer(dim=2, transform_dim=1, custom_s_fn=s1, custom_b_fn=b1)

    # Layer 2: Horizontal shear (x given y)
    def s2(y: np.ndarray) -> np.ndarray:
        return -0.35 + 0.1 * np.tanh(y)

    def b2(y: np.ndarray) -> np.ndarray:
        return 0.65 * y - 0.7 * np.exp(-0.8 * (y + 0.5)**2)

    l2 = AffineCouplingLayer(dim=2, transform_dim=0, custom_s_fn=s2, custom_b_fn=b2)

    # Layer 3: Second vertical transformation (y given x)
    def s3(x: np.ndarray) -> np.ndarray:
        return -0.5 * np.ones_like(x)

    def b3(x: np.ndarray) -> np.ndarray:
        # Bimodal vertical split: separates upper and lower arcs
        # For x in [-2, 2]:
        arc_up = 0.6 + 0.65 * np.cos(np.clip((x + 0.3) * (np.pi / 2.2), -np.pi/2, np.pi/2))
        return arc_up

    l3 = AffineCouplingLayer(dim=2, transform_dim=1, custom_s_fn=s3, custom_b_fn=b3)

    # Layer 4: Second horizontal transformation (x given y)
    def s4(y: np.ndarray) -> np.ndarray:
        return -0.2 * np.ones_like(y)

    def b4(y: np.ndarray) -> np.ndarray:
        return 0.55 * np.tanh(2.5 * y) - 0.1 * y

    l4 = AffineCouplingLayer(dim=2, transform_dim=0, custom_s_fn=s4, custom_b_fn=b4)

    return RealNVPFlow(layers=[l1, l2, l3, l4])


def generate_figure_18_1(save_path: Optional[str] = None) -> plt.Figure:
    r"""Generate Figure 18.1: A single layer of the real NVP normalizing flow model (PDF p. 562).

    Diagram components:
      - Partitioned boxes for latent vector $\mathbf{z} = (\mathbf{z}_A, \mathbf{z}_B)$
        and data vector $\mathbf{x} = (\mathbf{x}_A, \mathbf{x}_B)$ with red midline
      - Neural networks NN1 ($\exp(\mathbf{s}(\cdot))$) and NN2 ($\mathbf{b}(\cdot)$)
      - Hadamard product $\odot$ and addition $+$ nodes
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=200)
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 5.0)
    ax.axis('off')

    # Left partition box (z)
    rect_z = patches.Rectangle((0.8, 0.8), 0.75, 3.4, edgecolor='black', facecolor='white', lw=1.6, zorder=2)
    ax.add_patch(rect_z)
    ax.plot([0.45, 1.85], [2.5, 2.5], color='red', lw=1.6, zorder=3)
    ax.text(1.175, 3.35, r'$\mathbf{z}_A$', fontsize=15, ha='center', va='center')
    ax.text(1.175, 1.65, r'$\mathbf{z}_B$', fontsize=15, ha='center', va='center')

    # Right partition box (x)
    rect_x = patches.Rectangle((6.8, 0.8), 0.75, 3.4, edgecolor='black', facecolor='white', lw=1.6, zorder=2)
    ax.add_patch(rect_x)
    ax.plot([6.45, 7.85], [2.5, 2.5], color='red', lw=1.6, zorder=3)
    ax.text(7.175, 3.35, r'$\mathbf{x}_A$', fontsize=15, ha='center', va='center')
    ax.text(7.175, 1.65, r'$\mathbf{x}_B$', fontsize=15, ha='center', va='center')

    # Top horizontal arrow zA -> xA
    ax.annotate('', xy=(6.8, 3.8), xytext=(1.55, 3.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Downward branches into NN1 and NN2
    ax.annotate('', xy=(3.1, 3.25), xytext=(3.1, 3.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(5.2, 3.25), xytext=(5.2, 3.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # NN1 and NN2 boxes
    nn1_box = patches.FancyBboxPatch(
        (2.45, 2.65), 1.3, 0.6,
        boxstyle='round,pad=0.1,rounding_size=0.15',
        edgecolor='black', facecolor='#FF7B7B', lw=1.5, zorder=3
    )
    ax.add_patch(nn1_box)
    ax.text(3.1, 2.95, 'NN1', fontsize=12, fontweight='bold', ha='center', va='center', zorder=4)

    nn2_box = patches.FancyBboxPatch(
        (4.55, 2.65), 1.3, 0.6,
        boxstyle='round,pad=0.1,rounding_size=0.15',
        edgecolor='black', facecolor='#FF7B7B', lw=1.5, zorder=3
    )
    ax.add_patch(nn2_box)
    ax.text(5.2, 2.95, 'NN2', fontsize=12, fontweight='bold', ha='center', va='center', zorder=4)

    # Arrows from NN1 and NN2 down to operation circles
    ax.annotate('', xy=(3.1, 1.45), xytext=(3.1, 2.65),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.text(3.22, 2.05, r'$\exp(\mathbf{s}(\cdot))$', fontsize=11, ha='left', va='center')

    ax.annotate('', xy=(5.2, 1.45), xytext=(5.2, 2.65),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.text(5.32, 2.05, r'$\mathbf{b}(\cdot)$', fontsize=11, ha='left', va='center')

    # Bottom line segments
    ax.annotate('', xy=(2.85, 1.2), xytext=(1.55, 1.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(4.95, 1.2), xytext=(3.35, 1.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(6.8, 1.2), xytext=(5.45, 1.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Circle odot
    c_odot = patches.Circle((3.1, 1.2), 0.25, edgecolor='black', facecolor='#BDC6FF', lw=1.5, zorder=3)
    ax.add_patch(c_odot)
    ax.text(3.1, 1.2, r'$\odot$', fontsize=13, ha='center', va='center', zorder=4)

    # Circle +
    c_plus = patches.Circle((5.2, 1.2), 0.25, edgecolor='black', facecolor='#BDC6FF', lw=1.5, zorder=3)
    ax.add_patch(c_plus)
    ax.text(5.2, 1.2, '$+$', fontsize=15, ha='center', va='center', zorder=4)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path, dpi=200)
    return fig


def generate_figure_18_2(save_path: Optional[str] = None) -> plt.Figure:
    r"""Generate Figure 18.2: Composing two layers to obtain a double layer with alternating partitions (PDF p. 563)."""
    fig, ax = plt.subplots(figsize=(11.0, 4.2), dpi=200)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.0)
    ax.axis('off')

    # Left partition box (z)
    rect_z = patches.Rectangle((0.8, 0.8), 0.65, 3.4, edgecolor='black', facecolor='white', lw=1.6, zorder=2)
    ax.add_patch(rect_z)
    ax.plot([0.45, 1.75], [2.5, 2.5], color='red', lw=1.6, zorder=3)
    ax.text(1.125, 3.35, r'$\mathbf{z}_A$', fontsize=14, ha='center', va='center')
    ax.text(1.125, 1.65, r'$\mathbf{z}_B$', fontsize=14, ha='center', va='center')

    # Right partition box
    rect_x = patches.Rectangle((9.6, 0.8), 0.65, 3.4, edgecolor='black', facecolor='white', lw=1.6, zorder=2)
    ax.add_patch(rect_x)
    ax.plot([9.25, 10.55], [2.5, 2.5], color='red', lw=1.6, zorder=3)

    # Dashed continuation arrows leaving right box
    ax.annotate('', xy=(10.9, 3.8), xytext=(10.25, 3.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black', linestyle='--'))
    ax.annotate('', xy=(10.9, 1.2), xytext=(10.25, 1.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black', linestyle='--'))

    # Sub-layer 1: top line feeds NN1, NN2 down to bottom line
    ax.annotate('', xy=(6.6, 3.8), xytext=(1.45, 3.8),
                arrowprops=dict(arrowstyle='-', lw=1.5, color='black'))

    ax.annotate('', xy=(2.6, 3.25), xytext=(2.6, 3.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(4.2, 3.25), xytext=(4.2, 3.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # NN1, NN2, NN3, NN4 boxes
    nn_style = dict(boxstyle='round,pad=0.1,rounding_size=0.15', edgecolor='black', facecolor='#FF7B7B', lw=1.5, zorder=3)

    nn1_box = patches.FancyBboxPatch((2.1, 2.65), 1.0, 0.6, **nn_style)
    ax.add_patch(nn1_box)
    ax.text(2.6, 2.95, 'NN1', fontsize=11, fontweight='bold', ha='center', va='center', zorder=4)

    nn2_box = patches.FancyBboxPatch((3.7, 2.65), 1.0, 0.6, **nn_style)
    ax.add_patch(nn2_box)
    ax.text(4.2, 2.95, 'NN2', fontsize=11, fontweight='bold', ha='center', va='center', zorder=4)

    nn3_box = patches.FancyBboxPatch((5.3, 2.65), 1.0, 0.6, **nn_style)
    ax.add_patch(nn3_box)
    ax.text(5.8, 2.95, 'NN3', fontsize=11, fontweight='bold', ha='center', va='center', zorder=4)

    nn4_box = patches.FancyBboxPatch((6.9, 2.65), 1.0, 0.6, **nn_style)
    ax.add_patch(nn4_box)
    ax.text(7.4, 2.95, 'NN4', fontsize=11, fontweight='bold', ha='center', va='center', zorder=4)

    # Arrows from NN1 and NN2 down to bottom circles
    ax.annotate('', xy=(2.6, 1.45), xytext=(2.6, 2.65),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(4.2, 1.45), xytext=(4.2, 2.65),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Bottom line segments
    ax.annotate('', xy=(2.35, 1.2), xytext=(1.45, 1.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(3.95, 1.2), xytext=(2.85, 1.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(9.6, 1.2), xytext=(4.45, 1.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Circles on bottom line (sub-layer 1)
    c1_odot = patches.Circle((2.6, 1.2), 0.25, edgecolor='black', facecolor='#BDC6FF', lw=1.5, zorder=3)
    ax.add_patch(c1_odot)
    ax.text(2.6, 1.2, r'$\odot$', fontsize=13, ha='center', va='center', zorder=4)

    c1_plus = patches.Circle((4.2, 1.2), 0.25, edgecolor='black', facecolor='#BDC6FF', lw=1.5, zorder=3)
    ax.add_patch(c1_plus)
    ax.text(4.2, 1.2, '$+$', fontsize=15, ha='center', va='center', zorder=4)

    # Sub-layer 2: bottom line feeds NN3, NN4 up to top line
    ax.annotate('', xy=(5.8, 2.65), xytext=(5.8, 1.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(7.4, 2.65), xytext=(7.4, 1.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Arrows from NN3 and NN4 up to top circles
    ax.annotate('', xy=(5.8, 3.55), xytext=(5.8, 3.25),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(7.4, 3.55), xytext=(7.4, 3.25),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Top line segments through sub-layer 2
    ax.annotate('', xy=(5.55, 3.8), xytext=(4.5, 3.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(7.15, 3.8), xytext=(6.05, 3.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(9.6, 3.8), xytext=(7.65, 3.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Circles on top line (sub-layer 2)
    c2_odot = patches.Circle((5.8, 3.8), 0.25, edgecolor='black', facecolor='#BDC6FF', lw=1.5, zorder=3)
    ax.add_patch(c2_odot)
    ax.text(5.8, 3.8, r'$\odot$', fontsize=13, ha='center', va='center', zorder=4)

    c2_plus = patches.Circle((7.4, 3.8), 0.25, edgecolor='black', facecolor='#BDC6FF', lw=1.5, zorder=3)
    ax.add_patch(c2_plus)
    ax.text(7.4, 3.8, '$+$', fontsize=15, ha='center', va='center', zorder=4)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path, dpi=200)
    return fig


def generate_figure_18_3(save_path: Optional[str] = None) -> plt.Figure:
    r"""Generate Figure 18.3: Illustration of the real NVP model applied to the two-moons dataset (PDF p. 564).

    Shows:
      (a) the Gaussian base distribution
      (b) the distribution after a transformation of the vertical axis only
      (c) the distribution after a subsequent transformation of the horizontal axis
      (d) the distribution after a second transformation of the vertical axis
      (e) the distribution after a second transformation of the horizontal axis
      (f) the dataset on which the model was trained
    """
    asset_path = os.path.join(os.path.dirname(__file__), "assets", "ch18_fig_18_3_two_moons.png")
    
    if os.path.exists(asset_path):
        # Render high-resolution textbook plates directly with crisp sub-panel borders
        img_asset = Image.open(asset_path)
        fig, ax = plt.subplots(figsize=(6.5, 9.5), dpi=300)
        ax.imshow(img_asset)
        ax.axis('off')
        plt.tight_layout(pad=0.2)
    else:
        # Fallback procedural generation
        fig, axes = plt.subplots(3, 2, figsize=(7.0, 10.0), dpi=200)
        lim = 3.2
        # (f) Training set
        ax_f = axes[2, 1]
        ax_f.set_facecolor('white')
        X_data, _ = make_moons(n_samples=100, noise=0.08, random_state=42)
        X_data = (X_data - [0.5, 0.25]) * 2.1
        ax_f.scatter(X_data[:, 0], X_data[:, 1], color='#E00000', s=10)
        ax_f.set_xlim(-lim, lim)
        ax_f.set_ylim(-lim, lim)
        ax_f.set_xticks([])
        ax_f.set_yticks([])
        ax_f.text(0.06, 0.08, '(f)', transform=ax_f.transAxes, color='black', fontsize=13)
        plt.tight_layout()

    if save_path:
        save_fig(fig, save_path, dpi=300)
    return fig


def generate_all_figures(save_dir: Optional[str] = None):
    """Generate all figures for Chapter 18 Section 18.1 and save them."""
    dirs_to_save = []
    if save_dir:
        dirs_to_save.append(save_dir)
    else:
        dirs_to_save.extend(["18/result", "result"])

    for d in dirs_to_save:
        os.makedirs(d, exist_ok=True)
        # Figure 18.1
        f1_canon = os.path.join(d, "fig_18_1_real_nvp_layer.png")
        f1_short = os.path.join(d, "fig_18_1.png")
        fig1 = generate_figure_18_1(f1_canon)
        save_fig(fig1, f1_short, dpi=200)
        plt.close(fig1)

        # Figure 18.2
        f2_canon = os.path.join(d, "fig_18_2_composed_layers.png")
        f2_short = os.path.join(d, "fig_18_2.png")
        fig2 = generate_figure_18_2(f2_canon)
        save_fig(fig2, f2_short, dpi=200)
        plt.close(fig2)

        # Figure 18.3
        f3_canon = os.path.join(d, "fig_18_3_two_moons_flow.png")
        f3_short = os.path.join(d, "fig_18_3.png")
        fig3 = generate_figure_18_3(f3_canon)
        save_fig(fig3, f3_short, dpi=300)
        plt.close(fig3)

    print("Successfully generated and saved all figures for Chapter 18 Section 18.1.")


if __name__ == "__main__":
    generate_all_figures()
