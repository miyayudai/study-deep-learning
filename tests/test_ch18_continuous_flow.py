"""Tests for Chapter 18 Section 18.3: Continuous Flows and Neural ODEs."""

import os
import numpy as np
import pytest

from common.continuous_flows import (
    StandardGaussianBase,
    FlowDynamicsMLP,
    NeuralODE,
    ContinuousNormalizingFlow,
    generate_figure_18_5,
    generate_figure_18_6,
    generate_figure_18_7,
    generate_all_figures,
)


def test_standard_gaussian_base():
    """Test StandardGaussianBase log-density and sampling."""
    dim = 2
    base = StandardGaussianBase(dim=dim)
    z = np.zeros((1, dim))
    expected_logp = -0.5 * dim * np.log(2.0 * np.pi)
    np.testing.assert_allclose(base.log_prob(z), [expected_logp], rtol=1e-6)

    samples = base.sample(50, random_state=42)
    assert samples.shape == (50, dim)
    assert np.all(np.isfinite(samples))


def test_flow_dynamics_forward():
    """Test FlowDynamicsMLP forward evaluation and parameter packing/unpacking."""
    dim = 2
    dynamics = FlowDynamicsMLP(dim=dim, hidden_dim=16, random_state=42)

    # 1D single point input
    z1 = np.array([0.5, -0.3])
    dz1 = dynamics.forward(z1, t=0.5)
    assert dz1.shape == (dim,)
    assert np.all(np.isfinite(dz1))

    # 2D batch input
    z_batch = np.random.randn(10, dim)
    dz_batch = dynamics.forward(z_batch, t=0.5)
    assert dz_batch.shape == (10, dim)

    # Params get/set consistency
    p = dynamics.get_params()
    dynamics.set_params(p + 0.1)
    p_new = dynamics.get_params()
    np.testing.assert_allclose(p_new, p + 0.1)


def test_neural_ode_analytical_solution():
    """Test NeuralODE forward integration on a linear system with known solution."""
    dim = 2
    dynamics = FlowDynamicsMLP(dim=dim, hidden_dim=16, random_state=42)
    # Zero all weights except we test custom dynamics
    node = NeuralODE(dynamics=dynamics, solver="rk4")

    # Linear decay dz/dt = -0.5 * z -> z(t) = z(0) * exp(-0.5 * t)
    z0 = np.array([[1.0, 2.0]])
    t_span = (0.0, 1.0)

    # Custom wrapper for linear decay
    class LinearDecayDynamics:
        def __init__(self, rate=0.5):
            self.rate = rate
        def forward(self, z, t):
            return -self.rate * z

    node_decay = NeuralODE(dynamics=LinearDecayDynamics(rate=0.5), solver="rk4")
    t_eval, traj = node_decay.integrate_rk4(z0, t_span, n_steps=50)

    expected_zT = z0 * np.exp(-0.5 * 1.0)
    np.testing.assert_allclose(traj[-1], expected_zT, rtol=1e-5, atol=1e-5)


def test_adjoint_sensitivity_method():
    """Test Adjoint Sensitivity Method gradients vs finite differences (Eq. 18.24 - 18.26)."""
    dim = 2
    dynamics = FlowDynamicsMLP(dim=dim, hidden_dim=8, random_state=42)
    node = NeuralODE(dynamics=dynamics, solver="rk45")

    z0 = np.array([0.4, -0.2])
    T = 0.5
    z_T = node.forward(z0, t_span=(0.0, T))

    y_target = np.array([1.0, -0.5])
    # Loss L = 0.5 * ||z(T) - y_target||^2
    # Adjoint terminal condition a(T) = z(T) - y_target
    adj_T = z_T - y_target

    z0_rec, grad_w_adj = node.adjoint_backward(z_T, adj_T, t_span=(T, 0.0))

    # 1. State reconstruction backward check: z(0) should be recovered
    np.testing.assert_allclose(z0_rec, z0, atol=1e-3)

    # 2. Gradient check vs finite differences on a subset of parameters
    orig_params = dynamics.get_params()
    eps = 1e-5
    for p_idx in [0, 5, 10, len(orig_params) - 1]:
        p_plus = orig_params.copy(); p_plus[p_idx] += eps
        p_minus = orig_params.copy(); p_minus[p_idx] -= eps

        dynamics.set_params(p_plus)
        zT_p = node.forward(z0, t_span=(0.0, T))
        loss_p = 0.5 * np.sum((zT_p - y_target) ** 2)

        dynamics.set_params(p_minus)
        zT_m = node.forward(z0, t_span=(0.0, T))
        loss_m = 0.5 * np.sum((zT_m - y_target) ** 2)

        dynamics.set_params(orig_params)
        grad_num = (loss_p - loss_m) / (2.0 * eps)

        np.testing.assert_allclose(grad_w_adj[p_idx], grad_num, rtol=1e-2, atol=1e-3)


def test_instantaneous_change_of_variables():
    """Test instantaneous change of variables formula (Eq. 18.28)."""
    dim = 2
    # Define a known velocity field f(z) = [2*z0, -z1]
    # Divergence Tr(df/dz) = 2 - 1 = 1 (constant!)
    class KnownDivergenceDynamics:
        def __init__(self):
            self.dim = 2
        def forward(self, z, t):
            z_2d = np.atleast_2d(z)
            dz = np.column_stack([2.0 * z_2d[:, 0], -1.0 * z_2d[:, 1]])
            return dz[0] if z.ndim == 1 else dz
        def divergence_exact(self, z, t):
            z_2d = np.atleast_2d(z)
            return np.full(z_2d.shape[0], 1.0)
        def divergence_hutchinson(self, z, t, n_samples=1, **kwargs):
            return self.divergence_exact(z, t)

    dyn = KnownDivergenceDynamics()
    cnf = ContinuousNormalizingFlow(dynamics=dyn, T=1.0)

    z0 = np.array([[1.0, 1.0]])
    x_T, delta_logp = cnf.forward(z0, n_steps=50)

    # Analytical z(1): z0 * e^2 = 7.389, z1 * e^-1 = 0.3678
    np.testing.assert_allclose(x_T[0, 0], np.exp(2.0), rtol=1e-3)
    np.testing.assert_allclose(x_T[0, 1], np.exp(-1.0), rtol=1e-3)

    # Change in log density: \Delta \ln p = \int_0^1 -Tr(df/dz) dt = -1.0 * 1.0 = -1.0
    np.testing.assert_allclose(delta_logp[0], -1.0, rtol=1e-3)


def test_cnf_invertibility_and_symmetry():
    """Test CNF forward/inverse invertibility and symmetric computational cost (Exercise 18.10)."""
    dim = 2
    dynamics = FlowDynamicsMLP(dim=dim, hidden_dim=16, random_state=42)
    cnf = ContinuousNormalizingFlow(dynamics=dynamics, T=0.5)

    z0 = np.random.RandomState(42).randn(10, dim)
    x_T, delta_fwd = cnf.forward(z0, n_steps=30)
    z_rec, delta_inv = cnf.inverse(x_T, n_steps=30)

    # Invertibility check
    recon_err = np.max(np.abs(z0 - z_rec))
    assert recon_err < 1e-4, f"Reconstruction error {recon_err} exceeds threshold"

    # Symmetric log density change: delta_fwd + delta_inv should be ~ 0
    np.testing.assert_allclose(delta_fwd + delta_inv, 0.0, atol=1e-4)


def test_hutchinson_trace_estimator():
    """Test Hutchinson trace estimator unbiasedness (Eq. 18.29, 18.30, Exercise 18.11)."""
    dim = 3
    dynamics = FlowDynamicsMLP(dim=dim, hidden_dim=16, random_state=42)
    z_pt = np.array([0.5, -0.2, 0.8])
    t = 0.5

    exact_div = dynamics.divergence_exact(z_pt, t)

    # Gaussian estimator with large M
    est_div_gauss = dynamics.divergence_hutchinson(
        z_pt, t, n_samples=3000, noise_type="gaussian", random_state=42
    )
    np.testing.assert_allclose(est_div_gauss, exact_div, rtol=0.2, atol=0.1)

    # Rademacher estimator with large M
    est_div_rade = dynamics.divergence_hutchinson(
        z_pt, t, n_samples=3000, noise_type="rademacher", random_state=42
    )
    np.testing.assert_allclose(est_div_rade, exact_div, rtol=0.2, atol=0.1)


def test_figure_generation_ch18_sec3(tmp_path):
    """Test Figure 18.5, 18.6, 18.7 generation and file saving."""
    out_5 = str(tmp_path / "fig_18_5.png")
    fig5 = generate_figure_18_5(save_path=out_5)
    assert os.path.exists(out_5)

    out_6 = str(tmp_path / "fig_18_6.png")
    fig6 = generate_figure_18_6(save_path=out_6)
    assert os.path.exists(out_6)

    out_7 = str(tmp_path / "fig_18_7.png")
    fig7 = generate_figure_18_7(save_path=out_7)
    assert os.path.exists(out_7)

    all_files = generate_all_figures(save_dir=str(tmp_path))
    assert len(all_files) >= 3
    for p in all_files:
        assert os.path.exists(p)
