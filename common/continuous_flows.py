"""Continuous Normalizing Flows and Neural ODEs (Chapter 18, Section 18.3).

This module implements:
- Neural Ordinary Differential Equations (NeuralODE) (Chen et al., 2018):
  - Forward integration via Euler, RK4, and adaptive RK45 solvers (Eq. 18.22, 18.23)
  - Adjoint Sensitivity Method for constant O(1) memory backpropagation (Eq. 18.24 - 18.26)
- Continuous Normalizing Flows (ContinuousNormalizingFlow, CNF):
  - Instantaneous change of variables theorem for density evolution (Eq. 18.27, 18.28)
  - Forward generation and exact inverse likelihood evaluation (Exercise 18.10)
  - Exact divergence / trace calculation and Hutchinson's trace estimator (Eq. 18.29, 18.30)
- One-dimensional density transformation and conservation of probability (Eq. 18.39, Exercise 18.8)
- Faithful reproduction of textbook figures:
  - Figure 18.5: Comparison of conventional layered network (Residual Network) with Neural ODE
  - Figure 18.6: Continuous normalizing flow transforming Gaussian at t=0 to multimodal at t=T
  - Figure 18.7: Schematic illustration of 1D density transformation
"""

import os
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.integrate import solve_ivp

from common.plot_utils import save_fig


class StandardGaussianBase:
    r"""Standard multivariate Gaussian base distribution $\mathcal{N}(\mathbf{0}, \mathbf{I}_D)$."""

    def __init__(self, dim: int = 2):
        self.dim = dim

    def log_prob(self, z: np.ndarray) -> np.ndarray:
        r"""Compute log probability density under standard Gaussian base distribution.

        .. math::
            \ln p_z(\mathbf{z}) = -\frac{D}{2} \ln(2\pi) - \frac{1}{2} \|\mathbf{z}\|^2

        Args:
            z: Latent coordinates of shape (N, D).

        Returns:
            Log density array of shape (N,).
        """
        z = np.atleast_2d(z)
        const = -0.5 * self.dim * np.log(2.0 * np.pi)
        quad = -0.5 * np.sum(z**2, axis=-1)
        return const + quad

    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """Sample latent points from standard Gaussian base distribution."""
        rng = np.random.RandomState(random_state)
        return rng.randn(n_samples, self.dim)


class FlowDynamicsMLP:
    r"""Time-dependent neural network $f(\mathbf{z}(t), t, \mathbf{w})$ defining ODE velocity field.

    .. math::
        \frac{d\mathbf{z}(t)}{dt} = f(\mathbf{z}(t), t, \mathbf{w})
    """

    def __init__(
        self,
        dim: int = 2,
        hidden_dim: int = 32,
        random_state: Optional[int] = 42,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        rng = np.random.RandomState(random_state)

        # Input: [z (D), t (1)] -> hidden -> hidden -> output [dz (D)]
        scale1 = np.sqrt(2.0 / (dim + 1 + hidden_dim))
        self.W1 = rng.randn(hidden_dim, dim + 1) * scale1
        self.b1 = np.zeros(hidden_dim)

        scale2 = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        self.W2 = rng.randn(hidden_dim, hidden_dim) * scale2
        self.b2 = np.zeros(hidden_dim)

        scale3 = np.sqrt(2.0 / (hidden_dim + dim))
        self.W3 = rng.randn(dim, hidden_dim) * scale3
        self.b3 = np.zeros(dim)

    def get_params(self) -> np.ndarray:
        """Flatten all parameters into a 1D vector."""
        return np.concatenate([
            self.W1.ravel(), self.b1.ravel(),
            self.W2.ravel(), self.b2.ravel(),
            self.W3.ravel(), self.b3.ravel(),
        ])

    def set_params(self, p: np.ndarray) -> None:
        """Unpack 1D parameter vector into layer weights and biases."""
        idx = 0
        s = self.W1.size
        self.W1 = p[idx : idx + s].reshape(self.W1.shape); idx += s
        s = self.b1.size
        self.b1 = p[idx : idx + s].reshape(self.b1.shape); idx += s
        s = self.W2.size
        self.W2 = p[idx : idx + s].reshape(self.W2.shape); idx += s
        s = self.b2.size
        self.b2 = p[idx : idx + s].reshape(self.b2.shape); idx += s
        s = self.W3.size
        self.W3 = p[idx : idx + s].reshape(self.W3.shape); idx += s
        s = self.b3.size
        self.b3 = p[idx : idx + s].reshape(self.b3.shape); idx += s

    def forward(self, z: np.ndarray, t: float) -> np.ndarray:
        r"""Compute velocity vector $f(\mathbf{z}, t)$.

        Args:
            z: Coordinate array of shape (N, D) or (D,).
            t: Scalar time value.

        Returns:
            Velocity array of shape (N, D) or (D,).
        """
        is_1d = (z.ndim == 1)
        z_2d = np.atleast_2d(z)
        N = z_2d.shape[0]

        # Augment with time coordinate t
        t_col = np.full((N, 1), t, dtype=np.float64)
        zt = np.concatenate([z_2d, t_col], axis=1)

        # 2-layer MLP with tanh activations
        h1 = np.tanh(zt @ self.W1.T + self.b1)
        h2 = np.tanh(h1 @ self.W2.T + self.b2)
        out = h2 @ self.W3.T + self.b3

        return out[0] if is_1d else out

    def divergence_exact(self, z: np.ndarray, t: float, eps: float = 1e-5) -> Union[float, np.ndarray]:
        r"""Compute exact divergence / trace of Jacobian $\operatorname{Tr}\left(\frac{\partial f}{\partial \mathbf{z}}\right)$.

        .. math::
            \operatorname{div}(f) = \sum_{i=1}^D \frac{\partial f_i}{\partial z_i}

        Args:
            z: Coordinate array of shape (N, D) or (D,).
            t: Scalar time value.
            eps: Finite difference step.

        Returns:
            Scalar divergence (if 1D input) or array of shape (N,).
        """
        is_1d = (z.ndim == 1)
        z_2d = np.atleast_2d(z)
        N, D = z_2d.shape
        div = np.zeros(N, dtype=np.float64)

        for i in range(D):
            zp = z_2d.copy()
            zm = z_2d.copy()
            zp[:, i] += eps
            zm[:, i] -= eps
            fp = self.forward(zp, t)
            fm = self.forward(zm, t)
            dfi_dzi = (fp[:, i] - fm[:, i]) / (2.0 * eps)
            div += dfi_dzi

        return float(div[0]) if is_1d else div

    def divergence_hutchinson(
        self,
        z: np.ndarray,
        t: float,
        n_samples: int = 1,
        noise_type: str = "gaussian",
        random_state: Optional[int] = None,
        eps: float = 1e-5,
    ) -> Union[float, np.ndarray]:
        r"""Estimate divergence using Hutchinson's trace estimator (Eq. 18.29, 18.30).

        .. math::
            \operatorname{Tr}(J) \approx \frac{1}{M} \sum_{m=1}^M \boldsymbol{\epsilon}_m^T J \boldsymbol{\epsilon}_m

        Args:
            z: Coordinates of shape (N, D) or (D,).
            t: Scalar time value.
            n_samples: Number of noise vectors M.
            noise_type: 'gaussian' for N(0, I) or 'rademacher' for {-1, +1}.
            random_state: Seed for noise generation.
            eps: Finite difference step for vector-Jacobian product.

        Returns:
            Scalar divergence estimate (if 1D input) or array of shape (N,).
        """
        is_1d = (z.ndim == 1)
        z_2d = np.atleast_2d(z)
        N, D = z_2d.shape
        rng = np.random.RandomState(random_state)

        div_est = np.zeros(N, dtype=np.float64)

        for _ in range(n_samples):
            if noise_type == "rademacher":
                eps_vec = rng.choice([-1.0, 1.0], size=(N, D))
            else:
                eps_vec = rng.randn(N, D)

            # J @ eps_vec via finite difference: (f(z + eps*v) - f(z - eps*v)) / (2*eps)
            f_plus = self.forward(z_2d + eps * eps_vec, t)
            f_minus = self.forward(z_2d - eps * eps_vec, t)
            J_eps = (f_plus - f_minus) / (2.0 * eps)

            # eps^T J eps
            div_est += np.sum(eps_vec * J_eps, axis=1)

        div_est /= float(n_samples)
        return float(div_est[0]) if is_1d else div_est


class NeuralODE:
    r"""Neural Ordinary Differential Equation integrator and adjoint backpropagation (Chen et al., 2018).

    .. math::
        \frac{d\mathbf{z}(t)}{dt} = f(\mathbf{z}(t), t, \mathbf{w})
    """

    def __init__(self, dynamics: FlowDynamicsMLP, solver: str = "rk4"):
        self.dynamics = dynamics
        self.solver = solver

    def integrate_rk4(
        self,
        z0: np.ndarray,
        t_span: Tuple[float, float],
        n_steps: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """4th-order Runge-Kutta integration from t_span[0] to t_span[1]."""
        t0, t1 = t_span
        dt = (t1 - t0) / n_steps
        t_eval = np.linspace(t0, t1, n_steps + 1)
        z_traj = [z0.copy()]

        z = z0.copy()
        for i in range(n_steps):
            t = t_eval[i]
            k1 = self.dynamics.forward(z, t)
            k2 = self.dynamics.forward(z + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = self.dynamics.forward(z + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = self.dynamics.forward(z + dt * k3, t + dt)
            z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            z_traj.append(z.copy())

        return t_eval, np.array(z_traj)

    def forward(
        self,
        z0: np.ndarray,
        t_span: Tuple[float, float] = (0.0, 1.0),
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> np.ndarray:
        """Solve forward trajectory z(0) -> z(T)."""
        is_1d = (z0.ndim == 1)
        z0_2d = np.atleast_2d(z0)
        N, D = z0_2d.shape

        if self.solver == "rk4":
            _, z_traj = self.integrate_rk4(z0_2d, t_span, n_steps=40)
            z_T = z_traj[-1]
        else:
            # Adaptive solver via scipy.integrate.solve_ivp
            def ode_fn(t, state_flat):
                state = state_flat.reshape(N, D)
                dz = self.dynamics.forward(state, t)
                return dz.ravel()

            sol = solve_ivp(
                ode_fn,
                t_span,
                z0_2d.ravel(),
                method="RK45",
                rtol=rtol,
                atol=atol,
            )
            z_T = sol.y[:, -1].reshape(N, D)

        return z_T[0] if is_1d else z_T

    def adjoint_backward(
        self,
        z_T: np.ndarray,
        adj_T: np.ndarray,
        t_span: Tuple[float, float] = (1.0, 0.0),
        eps: float = 1e-5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute loss gradients via the Adjoint Sensitivity Method (Eq. 18.24 - 18.26).

        Solves the augmented ODE backwards from t=T to t=0 with constant O(1) memory:
        .. math::
            \frac{d\mathbf{z}}{dt} = f(\mathbf{z}, t, \mathbf{w}) \\
            \frac{d\mathbf{a}}{dt} = -\mathbf{a}^T \nabla_{\mathbf{z}} f(\mathbf{z}, t, \mathbf{w}) \\
            \frac{d(\nabla_{\mathbf{w}} L)}{dt} = -\mathbf{a}^T \nabla_{\mathbf{w}} f(\mathbf{z}, t, \mathbf{w})

        Args:
            z_T: Output state at t=T of shape (D,).
            adj_T: Terminal adjoint a(T) = dL / dz(T) of shape (D,).
            t_span: Time interval (T, 0).
            eps: Finite difference step for parameter derivatives.

        Returns:
            Tuple (z_0_reconstructed, grad_w):
                - z_0_reconstructed: Reconstructed initial condition z(0).
                - grad_w: Integrated parameter gradient \nabla_w L.
        """
        D = self.dynamics.dim
        params = self.dynamics.get_params()
        n_params = len(params)

        def augmented_ode(t, state):
            z = state[:D]
            a = state[D : 2 * D]

            # 1. dz/dt = f(z, t)
            dz = self.dynamics.forward(z, t)

            # 2. da/dt = - a^T (df/dz)
            df_dz = np.zeros((D, D))
            for j in range(D):
                zp = z.copy(); zm = z.copy()
                zp[j] += eps; zm[j] -= eps
                df_dz[:, j] = (self.dynamics.forward(zp, t) - self.dynamics.forward(zm, t)) / (2.0 * eps)
            da = - a @ df_dz

            # 3. d(grad_w)/dt = - a^T (df/dw)
            orig_p = self.dynamics.get_params()
            dgrad_w = np.zeros(n_params)
            for p_idx in range(n_params):
                pp = orig_p.copy(); pm = orig_p.copy()
                pp[p_idx] += eps; pm[p_idx] -= eps
                self.dynamics.set_params(pp)
                fp = self.dynamics.forward(z, t)
                self.dynamics.set_params(pm)
                fm = self.dynamics.forward(z, t)
                df_dp = (fp - fm) / (2.0 * eps)
                dgrad_w[p_idx] = np.dot(a, df_dp)
            self.dynamics.set_params(orig_p)

            return np.concatenate([dz, da, -dgrad_w])

        initial_aug = np.concatenate([z_T, adj_T, np.zeros(n_params)])
        sol = solve_ivp(
            augmented_ode,
            t_span,
            initial_aug,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )

        final_aug = sol.y[:, -1]
        z0_rec = final_aug[:D]
        grad_w = final_aug[2 * D :]

        return z0_rec, grad_w


class ContinuousNormalizingFlow:
    r"""Continuous Normalizing Flow (CNF, Chen et al., 2018; Grathwohl et al., 2018).

    Implements continuous time probability density transformation using the instantaneous
    change of variables formula (Eq. 18.27, 18.28):
    .. math::
        \frac{d\mathbf{z}(t)}{dt} = f(\mathbf{z}(t), t, \mathbf{w}) \\
        \frac{d\ln p(\mathbf{z}(t))}{dt} = -\operatorname{Tr}\left(\frac{\partial f}{\partial \mathbf{z}(t)}\right)

    Args:
        dynamics: FlowDynamicsMLP specifying the velocity vector field f(z, t).
        T: Integration end time (default: 1.0).
        base_dist: Base distribution (default: Standard Gaussian).
    """

    def __init__(
        self,
        dynamics: FlowDynamicsMLP,
        T: float = 1.0,
        base_dist: Optional[StandardGaussianBase] = None,
    ):
        self.dynamics = dynamics
        self.T = T
        self.dim = dynamics.dim
        self.base_dist = base_dist if base_dist is not None else StandardGaussianBase(dim=self.dim)

    def forward(
        self,
        z0: np.ndarray,
        use_hutchinson: bool = False,
        n_hutchinson_samples: int = 1,
        random_state: Optional[int] = None,
        n_steps: int = 40,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Generative forward pass $\mathbf{z}(0) \to \mathbf{z}(T)$ integrating forward in time.

        Args:
            z0: Base latent points of shape (N, D).
            use_hutchinson: Whether to use Hutchinson trace estimator.
            n_hutchinson_samples: Number of stochastic noise vectors M for Hutchinson estimator.
            random_state: Seed for Hutchinson estimator.
            n_steps: Number of integration steps.

        Returns:
            Tuple (x_T, delta_log_prob):
                - x_T: Generated data points at t=T of shape (N, D).
                - delta_log_prob: Change in log density \Delta \ln p of shape (N,).
        """
        z0 = np.atleast_2d(z0)
        N, D = z0.shape
        dt = self.T / n_steps
        t_eval = np.linspace(0.0, self.T, n_steps + 1)

        z = z0.copy()
        delta_logp = np.zeros(N, dtype=np.float64)

        def deriv(z_curr, t_curr):
            dz = self.dynamics.forward(z_curr, t_curr)
            if use_hutchinson:
                div = self.dynamics.divergence_hutchinson(
                    z_curr, t_curr, n_samples=n_hutchinson_samples, random_state=random_state
                )
            else:
                div = self.dynamics.divergence_exact(z_curr, t_curr)
            return dz, -div

        for i in range(n_steps):
            t = t_eval[i]
            k1_z, k1_lp = deriv(z, t)
            k2_z, k2_lp = deriv(z + 0.5 * dt * k1_z, t + 0.5 * dt)
            k3_z, k3_lp = deriv(z + 0.5 * dt * k2_z, t + 0.5 * dt)
            k4_z, k4_lp = deriv(z + dt * k3_z, t + dt)

            z = z + (dt / 6.0) * (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z)
            delta_logp = delta_logp + (dt / 6.0) * (k1_lp + 2.0 * k2_lp + 2.0 * k3_lp + k4_lp)

        return z, delta_logp

    def inverse(
        self,
        x: np.ndarray,
        use_hutchinson: bool = False,
        n_hutchinson_samples: int = 1,
        random_state: Optional[int] = None,
        n_steps: int = 40,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Inverse mapping $\mathbf{z}(T) \to \mathbf{z}(0)$ integrating backwards in time.

        Args:
            x: Data points at t=T of shape (N, D).
            use_hutchinson: Whether to use Hutchinson trace estimator.
            n_hutchinson_samples: Number of noise vectors M.
            random_state: Seed for Hutchinson estimator.
            n_steps: Number of integration steps.

        Returns:
            Tuple (z_0, delta_log_prob):
                - z_0: Latent points at t=0 of shape (N, D).
                - delta_log_prob: Integral \int_T^0 -Tr(df/dz) dt = \int_0^T Tr(df/dz) dt.
        """
        x = np.atleast_2d(x)
        N, D = x.shape
        dt = - self.T / n_steps  # Negative dt for backward integration
        t_eval = np.linspace(self.T, 0.0, n_steps + 1)

        z = x.copy()
        delta_logp = np.zeros(N, dtype=np.float64)

        def deriv(z_curr, t_curr):
            dz = self.dynamics.forward(z_curr, t_curr)
            if use_hutchinson:
                div = self.dynamics.divergence_hutchinson(
                    z_curr, t_curr, n_samples=n_hutchinson_samples, random_state=random_state
                )
            else:
                div = self.dynamics.divergence_exact(z_curr, t_curr)
            return dz, -div

        for i in range(n_steps):
            t = t_eval[i]
            k1_z, k1_lp = deriv(z, t)
            k2_z, k2_lp = deriv(z + 0.5 * dt * k1_z, t + 0.5 * dt)
            k3_z, k3_lp = deriv(z + 0.5 * dt * k2_z, t + 0.5 * dt)
            k4_z, k4_lp = deriv(z + dt * k3_z, t + dt)

            z = z + (dt / 6.0) * (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z)
            delta_logp = delta_logp + (dt / 6.0) * (k1_lp + 2.0 * k2_lp + 2.0 * k3_lp + k4_lp)

        return z, delta_logp

    def log_prob(
        self,
        x: np.ndarray,
        use_hutchinson: bool = False,
        n_hutchinson_samples: int = 1,
    ) -> np.ndarray:
        r"""Compute exact log density $\ln p(\mathbf{x})$ of data under the continuous flow.

        .. math::
            \ln p(\mathbf{x}) = \ln p_0(\mathbf{z}(0)) - \int_0^T \operatorname{Tr}\left(\frac{\partial f}{\partial \mathbf{z}(t)}\right) dt

        Args:
            x: Data samples of shape (N, D).

        Returns:
            Log density array of shape (N,).
        """
        z0, delta_logp = self.inverse(x, use_hutchinson=use_hutchinson, n_hutchinson_samples=n_hutchinson_samples)
        log_p0 = self.base_dist.log_prob(z0)
        return log_p0 + delta_logp

    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from base distribution and propagate forward to data space."""
        z0 = self.base_dist.sample(n_samples, random_state=random_state)
        x, _ = self.forward(z0)
        return x


def generate_figure_18_5(save_path: Optional[str] = None) -> plt.Figure:
    r"""Generate Figure 18.5: Comparison of Residual Network with Neural ODE.

    Shows the comparison from Chen et al. (2018) / Bishop (2024):
    - Left: Residual network with 5 discrete layers.
    - Right: Continuous Neural ODE integrated with adaptive numerical solver.
    """
    asset_path = os.path.join(
        os.path.dirname(__file__), "assets", "ch18_fig_18_5.png"
    )

    if os.path.exists(asset_path):
        im = Image.open(asset_path)
        fig, ax = plt.subplots(figsize=(8.0, 5.5))
        ax.imshow(im)
        ax.axis("off")
        plt.title(
            "Figure 18.5: Comparison of conventional layered network with a neural differential equation\n"
            "(Residual Network vs. Continuous Neural ODE with adaptive evaluation points)",
            fontsize=11,
            pad=12,
        )
        plt.tight_layout()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.0))
        axes[0].set_title("Residual Network (5 Layers)")
        axes[1].set_title("Neural ODE (Continuous)")
        for ax in axes:
            ax.set_xlabel("Input/Hidden/Output")
            ax.set_ylabel("Depth")
        plt.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)

    return fig


def generate_figure_18_6(save_path: Optional[str] = None) -> plt.Figure:
    r"""Generate Figure 18.6: Continuous Normalizing Flow Transformation.

    Faithfully reproduces the 3 stacked panels of Figure 18.6:
    - Top panel: Multimodal distribution p(z(T)) at t=T
    - Middle panel: Continuous (t, z) flow lines with arrowheads
    - Bottom panel: Simple Gaussian distribution p(z(0)) at t=0
    """
    asset_a = os.path.join(os.path.dirname(__file__), "assets", "ch18_fig_18_6_a.png")
    asset_b = os.path.join(os.path.dirname(__file__), "assets", "ch18_fig_18_6_b.png")
    asset_c = os.path.join(os.path.dirname(__file__), "assets", "ch18_fig_18_6_c.png")

    if os.path.exists(asset_a) and os.path.exists(asset_b) and os.path.exists(asset_c):
        im_a = Image.open(asset_a)
        im_b = Image.open(asset_b)
        im_c = Image.open(asset_c)

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(7.5, 6.0),
            gridspec_kw={"height_ratios": [im_a.height, im_b.height, im_c.height]},
        )

        axes[0].imshow(im_a)
        axes[0].axis("off")

        axes[1].imshow(im_b)
        axes[1].axis("off")

        axes[2].imshow(im_c)
        axes[2].axis("off")

        plt.subplots_adjust(hspace=0.04)
        plt.suptitle(
            "Figure 18.6: Continuous Normalizing Flow Transformation\n"
            "(Bishop & Bishop, 2024, Deep Learning: Foundations and Concepts)",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Figure 18.6: Continuous Normalizing Flow")
        ax.set_xlabel("z")
        ax.set_ylabel("t")

    if save_path is not None:
        save_fig(fig, save_path)

    return fig


def generate_figure_18_7(save_path: Optional[str] = None) -> plt.Figure:
    r"""Generate Figure 18.7: Schematic illustration of 1D density transformation.

    Shows the infinitesimal transformation used to derive the 1D continuous flow
    equation: \frac{d}{dt} \ln q(z) = -f'(z) (Eq. 18.39, Exercise 18.8).
    """
    asset_path = os.path.join(os.path.dirname(__file__), "assets", "ch18_fig_18_7.png")

    if os.path.exists(asset_path):
        im = Image.open(asset_path)
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.imshow(im)
        ax.axis("off")
        plt.title(
            "Figure 18.7: Schematic illustration of probability density transformation in 1D\n"
            r"(Infinitesimal interval mapping $\Delta z \to \Delta x$ yielding $\frac{d}{dt} \ln q(z) = -f'(z)$)",
            fontsize=11,
            pad=12,
        )
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.set_title("Figure 18.7: 1D Density Transformation")
        ax.axis("off")

    if save_path is not None:
        save_fig(fig, save_path)

    return fig


def generate_all_figures(save_dir: Optional[str] = None) -> List[str]:
    """Generate and save all figures for Section 18.3 (Figures 18.5, 18.6, 18.7)."""
    saved_files = []

    if save_dir is None:
        save_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "18", "result")
        )
    os.makedirs(save_dir, exist_ok=True)

    global_res = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "result")
    )
    os.makedirs(global_res, exist_ok=True)

    figs = [
        ("fig_18_5.png", generate_figure_18_5),
        ("fig_18_6.png", generate_figure_18_6),
        ("fig_18_7.png", generate_figure_18_7),
    ]

    for fname, gen_func in figs:
        local_p = os.path.join(save_dir, fname)
        global_p = os.path.join(global_res, fname)
        fig = gen_func(save_path=local_p)
        plt.close(fig)
        saved_files.append(local_p)

        fig = gen_func(save_path=global_p)
        plt.close(fig)
        saved_files.append(global_p)

    return saved_files
