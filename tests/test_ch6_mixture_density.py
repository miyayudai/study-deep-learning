"""
Tests for Chapter 6 Section 6.5: Mixture Density Networks
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts, pp. 198-204.
"""

import os
import pytest
import numpy as np

from common.mixture_density import (
    forward_kinematics,
    inverse_kinematics,
    generate_forward_data,
    generate_inverse_data,
    StandardMLPRegressor,
    MixtureDensityNetwork,
    generate_figure_6_16,
    generate_figure_6_17,
    generate_figure_6_18,
    generate_figure_6_19,
)


class TestRobotKinematics:
    """Test Section 6.5.1 Robot kinematics forward and inverse mapping."""

    def test_forward_kinematics_straight(self):
        # Arm extended along x-axis: theta1 = 0, theta2 = 0
        x1, x2 = forward_kinematics(0.0, 0.0, L1=1.0, L2=0.65)
        assert x1 == pytest.approx(1.65)
        assert x2 == pytest.approx(0.0)

    def test_forward_kinematics_right_angle(self):
        # theta1 = 90 deg, theta2 = 0 deg: pointing straight up
        x1, x2 = forward_kinematics(np.pi / 2, 0.0, L1=1.0, L2=0.65)
        assert x1 == pytest.approx(0.0, abs=1e-7)
        assert x2 == pytest.approx(1.65)

    def test_inverse_kinematics_both_solutions(self):
        # Pick reachable angles and compute forward target
        t1, t2 = np.radians(115), np.radians(-75)
        L1, L2 = 1.0, 0.65
        x_target, y_target = forward_kinematics(t1, t2, L1=L1, L2=L2)

        (sol_up, sol_down) = inverse_kinematics(x_target, y_target, L1=L1, L2=L2)

        # Both configurations must reach the target position
        x_up, y_up = forward_kinematics(sol_up[0], sol_up[1], L1=L1, L2=L2)
        assert x_up == pytest.approx(x_target, abs=1e-5)
        assert y_up == pytest.approx(y_target, abs=1e-5)

        x_down, y_down = forward_kinematics(sol_down[0], sol_down[1], L1=L1, L2=L2)
        assert x_down == pytest.approx(x_target, abs=1e-5)
        assert y_down == pytest.approx(y_target, abs=1e-5)

        # Elbow up and elbow down must have different elbow x-positions
        elbow_up_x = L1 * np.cos(sol_up[0])
        elbow_down_x = L1 * np.cos(sol_down[0])
        assert elbow_up_x < elbow_down_x


class TestToyDatasets:
    """Test Section 6.5.1 forward and inverse toy problems."""

    def test_forward_data_shape_and_seed(self):
        x1, t1 = generate_forward_data(n_samples=100, seed=12)
        x2, t2 = generate_forward_data(n_samples=100, seed=12)
        assert x1.shape == (100,)
        assert t1.shape == (100,)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(t1, t2)
        assert np.all(x1 >= 0.0) and np.all(x1 <= 1.0)

    def test_inverse_data_inversion(self):
        x_fwd, t_fwd = generate_forward_data(n_samples=50, seed=42)
        x_inv, t_inv = generate_inverse_data(n_samples=50, seed=42)
        np.testing.assert_array_equal(x_inv, t_fwd)
        np.testing.assert_array_equal(t_inv, x_fwd)


class TestStandardMLPRegressor:
    """Test 2-layer MLP least-squares model."""

    def test_mlp_forward_and_shapes(self):
        mlp = StandardMLPRegressor(n_in=1, n_hidden=6, n_out=1, seed=42)
        X = np.linspace(0, 1, 10)[:, None]
        preds, z = mlp.forward(X)
        assert preds.shape == (10, 1)
        assert z.shape == (10, 6)

    def test_mlp_numerical_gradient(self):
        mlp = StandardMLPRegressor(n_in=1, n_hidden=4, n_out=1, seed=123)
        X = np.linspace(0, 1, 5)[:, None]
        y = np.sin(X)
        p0 = mlp.pack()
        loss, grad = mlp.loss_and_grad(p0, X, y)

        eps = 1e-6
        for i in range(len(p0)):
            p_plus = p0.copy()
            p_plus[i] += eps
            l_plus, _ = mlp.loss_and_grad(p_plus, X, y)
            p_minus = p0.copy()
            p_minus[i] -= eps
            l_minus, _ = mlp.loss_and_grad(p_minus, X, y)
            num_g = (l_plus - l_minus) / (2 * eps)
            assert grad[i] == pytest.approx(num_g, rel=1e-4, abs=1e-5)

    def test_mlp_fit_reduces_loss(self):
        x, t = generate_forward_data(n_samples=100, seed=42)
        mlp = StandardMLPRegressor(n_in=1, n_hidden=6, n_out=1, seed=42)
        p0 = mlp.pack()
        l_init, _ = mlp.loss_and_grad(p0, x[:, None], t[:, None])
        mlp.fit(x, t, maxiter=200)
        p_opt = mlp.pack()
        l_final, _ = mlp.loss_and_grad(p_opt, x[:, None], t[:, None])
        assert l_final < l_init * 0.1


class TestMixtureDensityNetwork:
    """Test Sections 6.5.2 - 6.5.4 Mixture Density Network formulation and backprop."""

    def test_mdn_architecture_shapes(self):
        # 1D input, 1D target, K=3 components -> (1 + 2) * 3 = 9 outputs
        mdn = MixtureDensityNetwork(n_in=1, n_hidden=5, n_components=3, target_dim=1)
        assert mdn.n_out == 9
        X = np.linspace(0, 1, 10)[:, None]
        pi, sigma, mu, _ = mdn.forward(X)
        assert pi.shape == (10, 3)
        assert sigma.shape == (10, 3)
        assert mu.shape == (10, 3)

        # 2D input, 2D target, K=4 components -> (2 + 2) * 4 = 16 outputs
        mdn2 = MixtureDensityNetwork(n_in=2, n_hidden=8, n_components=4, target_dim=2)
        assert mdn2.n_out == 16
        X2 = np.ones((7, 2))
        pi2, sigma2, mu2, _ = mdn2.forward(X2)
        assert pi2.shape == (7, 4)
        assert sigma2.shape == (7, 4)
        assert mu2.shape == (7, 4, 2)

    def test_mdn_constraints_probability_and_variance(self):
        mdn = MixtureDensityNetwork(n_in=1, n_hidden=5, n_components=3, seed=42)
        X = np.random.uniform(0, 1, (20, 1))
        pi, sigma, _, _ = mdn.forward(X)

        # Eq 6.39: sum_k pi_k(x) = 1 and 0 <= pi_k(x) <= 1
        np.testing.assert_allclose(np.sum(pi, axis=1), 1.0, rtol=1e-6)
        assert np.all(pi >= 0.0)
        assert np.all(pi <= 1.0)

        # Eq 6.41: sigma_k(x) > 0
        assert np.all(sigma > 0.0)

    def test_mdn_exact_backprop_gradients_1d(self):
        """Verify analytical gradients (Eq 6.45 - 6.47) match finite differences."""
        mdn = MixtureDensityNetwork(n_in=1, n_hidden=4, n_components=3, target_dim=1, seed=77)
        X = np.array([[0.2], [0.5], [0.8]])
        T = np.array([[0.1], [0.6], [0.9]])

        p0 = mdn.pack()
        loss, grad = mdn.loss_and_grad(p0, X, T)

        eps = 1e-6
        for i in range(len(p0)):
            p_plus = p0.copy()
            p_plus[i] += eps
            l_plus, _ = mdn.loss_and_grad(p_plus, X, T)
            p_minus = p0.copy()
            p_minus[i] -= eps
            l_minus, _ = mdn.loss_and_grad(p_minus, X, T)
            num_g = (l_plus - l_minus) / (2 * eps)
            assert grad[i] == pytest.approx(num_g, rel=1e-4, abs=1e-5)

    def test_mdn_exact_backprop_gradients_multidim(self):
        """Verify analytical gradients for multi-dimensional target (L=2)."""
        mdn = MixtureDensityNetwork(n_in=2, n_hidden=4, n_components=2, target_dim=2, seed=88)
        X = np.random.normal(0, 1, (3, 2))
        T = np.random.normal(0, 1, (3, 2))

        p0 = mdn.pack()
        loss, grad = mdn.loss_and_grad(p0, X, T)

        eps = 1e-6
        for i in range(len(p0)):
            p_plus = p0.copy()
            p_plus[i] += eps
            l_plus, _ = mdn.loss_and_grad(p_plus, X, T)
            p_minus = p0.copy()
            p_minus[i] -= eps
            l_minus, _ = mdn.loss_and_grad(p_minus, X, T)
            num_g = (l_plus - l_minus) / (2 * eps)
            assert grad[i] == pytest.approx(num_g, rel=1e-4, abs=1e-5)

    def test_mdn_fit_inverse_problem(self):
        """Verify MDN trains properly on the inverse toy problem."""
        x_inv, t_inv = generate_inverse_data(n_samples=150, seed=42)
        mdn = MixtureDensityNetwork(n_in=1, n_hidden=5, n_components=3, seed=9)
        p0 = mdn.pack()
        l_init, _ = mdn.loss_and_grad(p0, x_inv[:, None], t_inv[:, None])
        mdn.fit(x_inv, t_inv, maxiter=500)
        p_opt = mdn.pack()
        l_final, _ = mdn.loss_and_grad(p_opt, x_inv[:, None], t_inv[:, None])

        assert l_final < l_init - 100.0

    def test_conditional_mean_and_variance(self):
        mdn = MixtureDensityNetwork(n_in=1, n_hidden=5, n_components=3, seed=9)
        X = np.array([[0.3], [0.7]])
        mean = mdn.predict_mean(X)
        var = mdn.predict_variance(X)
        assert mean.shape == (2,)
        assert var.shape == (2,)
        assert np.all(var > 0.0)

    def test_mode_approximation(self):
        mdn = MixtureDensityNetwork(n_in=1, n_hidden=5, n_components=3, seed=9)
        X = np.array([[0.2], [0.8]])
        mode = mdn.predict_mode_approx(X)
        assert mode.shape == (2,)

    def test_density_integration(self):
        """Density p(t|x) must integrate to approx 1 over target space t."""
        mdn = MixtureDensityNetwork(n_in=1, n_hidden=5, n_components=3, seed=9)
        x_query = np.array([[0.5]])
        t_grid = np.linspace(-3.0, 4.0, 1000)[:, None]
        x_rep = np.full_like(t_grid, 0.5)

        dens = mdn.predict_density(x_rep, t_grid)
        dt = float(t_grid[1, 0] - t_grid[0, 0])
        integral = float(np.sum(dens) * dt)
        assert integral == pytest.approx(1.0, rel=1e-2)

    def test_sampling(self):
        mdn = MixtureDensityNetwork(n_in=1, n_hidden=5, n_components=3, seed=9)
        X = np.array([[0.2], [0.6]])
        samples = mdn.sample(X, n_samples=50, seed=42)
        assert samples.shape == (2, 50)


class TestFigureGenerators:
    """Test reproduction of Figures 6.16 - 6.19."""

    def test_figure_6_16(self, tmp_path):
        out = tmp_path / "fig_6_16.png"
        fig = generate_figure_6_16(filepath=str(out), save_both=False)
        assert fig is not None
        assert os.path.exists(out)
        assert os.path.getsize(out) > 1000

    def test_figure_6_17(self, tmp_path):
        out = tmp_path / "fig_6_17.png"
        fig = generate_figure_6_17(filepath=str(out), save_both=False)
        assert fig is not None
        assert os.path.exists(out)
        assert os.path.getsize(out) > 1000

    def test_figure_6_18(self, tmp_path):
        out = tmp_path / "fig_6_18.png"
        fig = generate_figure_6_18(filepath=str(out), save_both=False)
        assert fig is not None
        assert os.path.exists(out)
        assert os.path.getsize(out) > 1000

    def test_figure_6_19(self, tmp_path):
        out = tmp_path / "fig_6_19.png"
        fig = generate_figure_6_19(filepath=str(out), save_both=False)
        assert fig is not None
        assert os.path.exists(out)
        assert os.path.getsize(out) > 1000
