"""Tests for Chapter 20 Section 20.3: Score Matching and Continuous Diffusion SDEs."""

import os
import pytest
import numpy as np
from common.score_matching import (
    GaussianMixture2D,
    implicit_score_matching_loss,
    denoising_score_matching_loss,
    AnnealedLangevinDynamics,
    VPSDE,
    plot_score_vector_field_and_mixture,
)


class TestGaussianMixtureScore:
    """Test suite for GaussianMixture2D and exact analytical score computation."""

    def test_score_matches_numerical_gradient(self):
        weights = [0.6, 0.4]
        means = [[-1.5, 0.0], [1.5, 0.0]]
        covs = [
            [[0.5, 0.1], [0.1, 0.5]],
            [[0.4, -0.05], [-0.05, 0.4]],
        ]
        gmm = GaussianMixture2D(weights=weights, means=means, covs=covs)

        pts = np.array([
            [-1.2, 0.3],
            [1.0, -0.2],
            [0.1, 0.5],
        ])

        ana_scores = gmm.score(pts)

        # Finite difference gradient of ln p(x)
        eps = 1e-6
        for i in range(len(pts)):
            pt = pts[i]
            for dim in range(2):
                p_pos = pt.copy()
                p_pos[dim] += eps
                p_neg = pt.copy()
                p_neg[dim] -= eps

                log_p_pos = np.log(gmm.pdf(p_pos[None, :])[0])
                log_p_neg = np.log(gmm.pdf(p_neg[None, :])[0])
                num_score = (log_p_pos - log_p_neg) / (2.0 * eps)

                assert ana_scores[i, dim] == pytest.approx(num_score, abs=1e-4)


class TestDenoisingScoreMatching:
    """Test suite for denoising score matching loss."""

    def test_denoising_score_loss_properties(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        sigma = 0.5

        # Dummy model: true score for isotropic Gaussian is -x / (1 + sigma^2)
        def dummy_score_model(x_noisy, s):
            return -x_noisy / (1.0 + s ** 2)

        loss, x_noisy = denoising_score_matching_loss(dummy_score_model, X, sigma=sigma, random_state=42)
        assert loss > 0.0
        assert x_noisy.shape == (100, 2)


class TestAnnealedLangevinDynamics:
    """Test suite for Annealed Langevin dynamics."""

    def test_langevin_sampling_moves_towards_modes(self):
        weights = [1.0]
        means = [[2.0, -1.0]]
        covs = [[[0.3, 0.0], [0.0, 0.3]]]
        gmm = GaussianMixture2D(weights=weights, means=means, covs=covs)

        def score_fn(x, sigma):
            # For smoothed distribution, scale score
            return gmm.score(x)

        sigmas = np.array([2.0, 1.0, 0.5, 0.1])
        sampler = AnnealedLangevinDynamics(score_fn=score_fn, sigmas=sigmas, n_steps_each=50, step_lr=5e-4)

        init_pts = np.array([[0.0, 1.0]])
        final_pts, traj = sampler.sample(init_pts, return_trajectory=True, random_state=42)

        # Final points should have moved much closer to mean [2.0, -1.0]
        dist_init = np.linalg.norm(init_pts[0] - np.array([2.0, -1.0]))
        dist_final = np.linalg.norm(final_pts[0] - np.array([2.0, -1.0]))
        assert dist_final < dist_init


class TestContinuousSDEAndODE:
    """Test suite for continuous VP SDE and Probability Flow ODE."""

    def test_vpsde_properties(self):
        sde = VPSDE(beta_min=0.1, beta_max=20.0)

        z = np.array([[1.0, 2.0]])
        t = 0.5
        dt = 0.01

        score = -z  # standard normal score
        z_rev = sde.reverse_sde_step(z, score, t=t, dt=dt, noise=np.zeros_like(z))
        z_ode = sde.probability_flow_ode_step(z, score, t=t, dt=dt)

        assert z_rev.shape == (1, 2)
        assert z_ode.shape == (1, 2)
        assert not np.any(np.isnan(z_rev))
        assert not np.any(np.isnan(z_ode))

    def test_visualization_generates(self):
        weights = [0.5, 0.5]
        means = [[-1.0, 0.0], [1.0, 0.0]]
        covs = [np.eye(2) * 0.4, np.eye(2) * 0.4]
        gmm = GaussianMixture2D(weights, means, covs)
        fig = plot_score_vector_field_and_mixture(gmm, grid_size=15)
        assert fig is not None
