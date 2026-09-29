"""Tests for Chapter 18 Section 18.2: Autoregressive Flows (MAF and IAF)."""

import os
import numpy as np
import pytest

from common.autoregressive_flows import (
    StandardGaussian,
    MaskedLinear,
    MADEConditioner,
    MaskedAutoregressiveFlow,
    InverseAutoregressiveFlow,
    benchmark_computational_asymmetry,
    generate_figure_18_4,
    generate_all_figures,
)


def test_standard_gaussian():
    """Test StandardGaussian log-density and sampling."""
    dim = 3
    dist = StandardGaussian(dim=dim)
    z = np.zeros((1, dim))
    expected_log_p = -0.5 * dim * np.log(2 * np.pi)
    np.testing.assert_allclose(dist.log_prob(z), [expected_log_p], rtol=1e-6)

    samples = dist.sample(100, random_state=42)
    assert samples.shape == (100, dim)
    assert np.all(np.isfinite(samples))


def test_made_mask_connectivity():
    """Test that MADE connectivity is strictly upper triangular for various dimensions."""
    for D in [2, 3, 5]:
        made = MADEConditioner(input_dim=D, hidden_dims=[32, 32], random_state=42)
        P = made.get_connectivity_matrix()
        assert P.shape == (D, D)
        # Strictly upper triangular: diagonal and lower triangle must be 0
        np.testing.assert_array_equal(np.tril(P), 0.0)
        assert made.verify_autoregressive() is True


def test_made_gradient_independence():
    """Test that output i has zero gradient w.r.t input j for all j >= i."""
    D = 4
    made = MADEConditioner(input_dim=D, hidden_dims=[16, 16], random_state=42)
    x0 = np.random.RandomState(42).randn(1, D)
    s0, b0 = made.forward(x0)

    eps = 1e-5
    for j in range(D):
        x_pert = x0.copy()
        x_pert[0, j] += eps
        s_pert, b_pert = made.forward(x_pert)

        # Output i for i <= j must NOT change when x_j is perturbed!
        for i in range(j + 1):
            diff_s = np.abs(s_pert[0, i] - s0[0, i])
            diff_b = np.abs(b_pert[0, i] - b0[0, i])
            assert diff_s < 1e-12, f"s[{i}] changed when x[{j}] perturbed! diff={diff_s}"
            assert diff_b < 1e-12, f"b[{i}] changed when x[{j}] perturbed! diff={diff_b}"


def test_maf_invertibility():
    """Test MAF exact invertibility z -> x -> z and x -> z -> x (Eq. 18.17, 18.18)."""
    for D in [2, 3, 4]:
        maf = MaskedAutoregressiveFlow(dim=D, hidden_dims=[16, 16], random_state=42)
        rng = np.random.RandomState(123)
        z_orig = rng.randn(10, D)

        # Forward (sampling: sequential) then inverse (likelihood: parallel)
        x_gen = maf.forward(z_orig)
        z_rec, _ = maf.inverse(x_gen)
        max_err = np.max(np.abs(z_orig - z_rec))
        assert max_err < 1e-12, f"MAF forward->inverse error {max_err} exceeds threshold"

        # Inverse then forward
        x_orig = rng.randn(10, D)
        z_lat, _ = maf.inverse(x_orig)
        x_rec = maf.forward(z_lat)
        max_err_x = np.max(np.abs(x_orig - x_rec))
        assert max_err_x < 1e-12, f"MAF inverse->forward error {max_err_x} exceeds threshold"


def test_iaf_invertibility():
    """Test IAF exact invertibility z -> x -> z and x -> z -> x (Eq. 18.19, 18.20)."""
    for D in [2, 3, 4]:
        iaf = InverseAutoregressiveFlow(dim=D, hidden_dims=[16, 16], random_state=42)
        rng = np.random.RandomState(456)
        z_orig = rng.randn(10, D)

        # Forward (sampling: parallel) then inverse (likelihood: sequential)
        x_gen, _ = iaf.forward(z_orig)
        z_rec = iaf.inverse(x_gen)
        max_err = np.max(np.abs(z_orig - z_rec))
        assert max_err < 1e-12, f"IAF forward->inverse error {max_err} exceeds threshold"

        # Inverse then forward
        x_orig = rng.randn(10, D)
        z_lat = iaf.inverse(x_orig)
        x_rec, _ = iaf.forward(z_lat)
        max_err_x = np.max(np.abs(x_orig - x_rec))
        assert max_err_x < 1e-12, f"IAF inverse->forward error {max_err_x} exceeds threshold"


def test_maf_jacobian_and_determinant():
    """Test that MAF Jacobian is lower-triangular with exact determinant."""
    D = 4
    maf = MaskedAutoregressiveFlow(dim=D, hidden_dims=[16, 16], random_state=42)
    x0 = np.array([0.5, -0.2, 1.1, -0.7])

    J_num = maf.jacobian_matrix(x0)
    assert J_num.shape == (D, D)

    # Upper triangular part (above diagonal) must be strictly zero
    upper_part = np.triu(J_num, k=1)
    np.testing.assert_allclose(upper_part, 0.0, atol=1e-5)

    # Diagonal elements must equal exp(-s_i)
    s, _ = maf.conditioner.forward(x0)
    diag_analytic = np.exp(-s[0])
    np.testing.assert_allclose(np.diag(J_num), diag_analytic, rtol=1e-4)

    # Determinant of J_num vs analytical log_det
    _, log_det_analytic = maf.inverse(x0)
    det_num = np.linalg.det(J_num)
    np.testing.assert_allclose(np.log(np.abs(det_num)), log_det_analytic[0], rtol=1e-4)


def test_maf_log_likelihood():
    """Test MAF log_prob consistency."""
    D = 3
    maf = MaskedAutoregressiveFlow(dim=D, hidden_dims=[16, 16], random_state=42)
    x = np.random.RandomState(42).randn(20, D)

    log_p = maf.log_prob(x)
    assert log_p.shape == (20,)
    assert np.all(np.isfinite(log_p))

    # Single point vs batch
    log_p_single = maf.log_prob(x[0:1])
    np.testing.assert_allclose(log_p[0], log_p_single[0], rtol=1e-8)


def test_computational_asymmetry():
    """Test computational asymmetry: MAF inverse is faster, IAF forward is faster."""
    bench = benchmark_computational_asymmetry(dim=6, n_samples=500, n_trials=10, random_state=42)
    # MAF forward should take longer than MAF inverse
    assert bench["maf_forward_sequential_ms"] > bench["maf_inverse_parallel_ms"]
    # IAF inverse should take longer than IAF forward
    assert bench["iaf_inverse_sequential_ms"] > bench["iaf_forward_parallel_ms"]


def test_maf_training():
    """Test maximum likelihood training of MAF with Adam."""
    D = 2
    maf = MaskedAutoregressiveFlow(dim=D, hidden_dims=[16, 16], random_state=42)
    rng = np.random.RandomState(42)
    X = rng.randn(100, D) * 0.5 + 1.0

    initial_nll = -np.mean(maf.log_prob(X))
    losses = maf.fit(X, n_epochs=15, lr=0.01, batch_size=32)
    final_nll = losses[-1]

    assert len(losses) == 15
    assert np.all(np.isfinite(losses))
    assert final_nll < initial_nll


def test_figure_18_4_generation(tmp_path):
    """Test Figure 18.4 generation and saving."""
    out_path = str(tmp_path / "fig_18_4_test.png")
    fig = generate_figure_18_4(save_path=out_path)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 1000

    saved_files = generate_all_figures(save_dir=str(tmp_path))
    assert len(saved_files) >= 2
    for p in saved_files:
        assert os.path.exists(p)
