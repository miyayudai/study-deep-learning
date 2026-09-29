"""Unit tests for Chapter 18 Section 18.1 Coupling Flows and Real NVP."""

import os
import numpy as np
import pytest
from scipy.integrate import dblquad

from common.coupling_flows import (
    StandardGaussian,
    ConditionerMLP,
    AffineCouplingLayer,
    RealNVPFlow,
    get_two_moons_flow,
    generate_figure_18_1,
    generate_figure_18_2,
    generate_figure_18_3,
    generate_all_figures,
)


class TestStandardGaussian:
    """Tests for StandardGaussian base distribution."""

    def test_log_prob_and_sampling(self):
        dist = StandardGaussian(dim=2)
        samples = dist.sample(1000, random_state=42)
        assert samples.shape == (1000, 2)
        assert np.allclose(np.mean(samples, axis=0), [0.0, 0.0], atol=0.15)
        assert np.allclose(np.cov(samples.T), np.eye(2), atol=0.2)

        # Check analytical value at origin
        lp_zero = dist.log_prob(np.zeros((1, 2)))[0]
        expected_zero = -np.log(2.0 * np.pi)
        assert np.isclose(lp_zero, expected_zero)

        # Check value at [1.0, 1.0]
        lp_pt = dist.log_prob(np.array([[1.0, 1.0]]))[0]
        expected_pt = -np.log(2.0 * np.pi) - 1.0
        assert np.isclose(lp_pt, expected_pt)


class TestConditionerMLP:
    """Tests for ConditionerMLP."""

    def test_mlp_forward_and_bounds(self):
        mlp = ConditionerMLP(in_features=1, hidden_features=16, out_features=2, scale_max=1.8, random_state=42)
        x = np.array([[-1.5], [0.0], [2.5]])
        s, b = mlp.forward(x)

        assert s.shape == (3, 1)
        assert b.shape == (3, 1)
        assert np.all(np.abs(s) <= 1.8)


class TestAffineCouplingLayer:
    """Tests for AffineCouplingLayer."""

    def test_invertibility_random_inputs(self):
        mlp = ConditionerMLP(in_features=1, hidden_features=24, out_features=2, random_state=0)
        layer = AffineCouplingLayer(dim=2, transform_dim=1, conditioner=mlp)

        rng = np.random.RandomState(42)
        z = rng.randn(100, 2) * 2.0

        # Forward z -> x (Eq. 18.10, 18.11)
        x, fwd_ldet = layer.forward(z)
        assert x.shape == z.shape
        # Untransformed coordinate must match exactly (xA = zA)
        assert np.allclose(x[:, 0], z[:, 0], atol=1e-12)

        # Inverse x -> z (Eq. 18.12, 18.13)
        z_rec, inv_ldet = layer.inverse(x)
        assert np.allclose(z_rec, z, atol=1e-12)
        assert np.allclose(fwd_ldet, -inv_ldet, atol=1e-12)

    def test_jacobian_block_structure_and_determinant(self):
        """Verify Eq. 18.14: block triangular Jacobian matrix and determinant."""
        mlp = ConditionerMLP(in_features=1, hidden_features=24, out_features=2, random_state=7)
        layer = AffineCouplingLayer(dim=2, transform_dim=1, conditioner=mlp)

        x_pt = np.array([0.75, -1.2])
        J = layer.jacobian_matrix(x_pt, eps=1e-6)

        # Top-left block: d(zA)/d(xA) = 1.0 (Id)
        assert np.isclose(J[0, 0], 1.0, atol=1e-5)
        # Top-right block: d(zA)/d(xB) = 0.0
        assert np.isclose(J[0, 1], 0.0, atol=1e-5)

        # Bottom-right block: diag(exp(-s))
        s, _ = layer._get_st(x_pt[0:1])
        expected_diag = np.exp(-s[0, 0])
        assert np.isclose(J[1, 1], expected_diag, atol=1e-4)

        # Determinant of triangular matrix is product of diagonal elements
        det_J = np.linalg.det(J)
        assert np.isclose(det_J, expected_diag, atol=1e-4)

        # Log determinant from layer.inverse
        _, inv_ldet = layer.inverse(x_pt.reshape(1, -1))
        assert np.isclose(np.log(np.abs(det_J)), inv_ldet[0], atol=1e-4)


class TestRealNVPFlow:
    """Tests for multi-layer RealNVPFlow composition and likelihood."""

    def test_flow_composition_and_invertibility(self):
        mlp1 = ConditionerMLP(in_features=1, hidden_features=16, random_state=1)
        mlp2 = ConditionerMLP(in_features=1, hidden_features=16, random_state=2)
        l1 = AffineCouplingLayer(dim=2, transform_dim=1, conditioner=mlp1)
        l2 = AffineCouplingLayer(dim=2, transform_dim=0, conditioner=mlp2)

        flow = RealNVPFlow(layers=[l1, l2])

        rng = np.random.RandomState(99)
        z = rng.randn(50, 2)
        x, fwd_ldet = flow.forward(z)
        z_rec, inv_ldet = flow.inverse(x)

        assert np.allclose(z_rec, z, atol=1e-11)
        assert np.allclose(fwd_ldet, -inv_ldet, atol=1e-11)

    def test_change_of_variables_density_integration(self):
        """Verify that data distribution px(x) integrates to 1.0 (Eq. 18.1)."""
        mlp1 = ConditionerMLP(in_features=1, hidden_features=8, random_state=10)
        l1 = AffineCouplingLayer(dim=2, transform_dim=1, conditioner=mlp1)
        flow = RealNVPFlow(layers=[l1])

        # Evaluate on fine 2D grid and integrate numerically
        lim = 4.5
        grid_1d = np.linspace(-lim, lim, 160)
        dx = grid_1d[1] - grid_1d[0]
        GX, GY = np.meshgrid(grid_1d, grid_1d)
        pts = np.column_stack([GX.ravel(), GY.ravel()])

        log_px = flow.log_prob(pts)
        px = np.exp(log_px).reshape(GX.shape)

        integral = np.sum(px) * (dx**2)
        assert np.isclose(integral, 1.0, atol=0.03)

    def test_two_moons_flow_progression(self):
        flow = get_two_moons_flow()
        assert len(flow.layers) == 4
        assert [l.transform_dim for l in flow.layers] == [1, 0, 1, 0]

        z = np.array([[0.0, 0.0], [1.0, -1.0]])
        intermediates = flow.forward_intermediates(z)
        assert len(intermediates) == 5

        # Check invertibility of entire flow
        x_end = intermediates[-1]
        z_rec, _ = flow.inverse(x_end)
        assert np.allclose(z_rec, z, atol=1e-11)


class TestFigureGeneration:
    """Tests for Chapter 18 Section 18.1 figure generation."""

    def test_figures_exist_and_non_empty(self, tmp_path):
        # Generate figures in temp dir
        generate_all_figures(save_dir=str(tmp_path))

        expected_files = [
            "fig_18_1_real_nvp_layer.png",
            "fig_18_1.png",
            "fig_18_2_composed_layers.png",
            "fig_18_2.png",
            "fig_18_3_two_moons_flow.png",
            "fig_18_3.png",
        ]

        for fname in expected_files:
            fpath = tmp_path / fname
            assert fpath.exists(), f"Missing expected figure file: {fname}"
            assert fpath.stat().st_size > 5000, f"Figure file too small: {fname}"
