"""Tests for Chapter 19 Exercises (Exercises 19.1 〜 19.6)."""

import pytest
import numpy as np
from common.exercises_ch19 import (
    score_function_gaussian_1d,
    reinforce_gradient_estimator_1d,
    reparameterization_gradient_estimator_1d,
    verify_exercise_19_1,
    affine_transform_1d_density,
    verify_exercise_19_2,
    multivariate_affine_transform_density,
    verify_exercise_19_3,
    gaussian_kl_analytical,
    gaussian_kl_gradients,
    gaussian_kl_monte_carlo,
    verify_exercise_19_4,
    gaussian_differential_entropy,
    verify_exercise_19_5,
    evaluate_mixture_model_elbo,
    verify_exercise_19_6,
)


class TestChapter19Exercises:
    """Test suite covering Exercises 19.1 through 19.6."""

    def test_exercise_19_1_score_and_reinforce(self):
        """Exercise 19.1: Test score function expectation zero, unbiasedness, and variance comparison."""
        mu = 2.0
        sigma2 = 1.5
        res = verify_exercise_19_1(mu=mu, sigma2=sigma2, S=60000, random_state=42)

        # 1. Unbiasedness: both estimators converge to true gradient 2*mu = 4.0
        assert res["true_grad_mu"] == pytest.approx(4.0, abs=1e-6)
        assert res["diff_rf"] < 0.15  # REINFORCE has larger variance but converges
        assert res["diff_rp"] < 0.05  # Reparameterization has lower variance

        # 2. Score function expectation is zero
        assert abs(res["score_mean"]) < 0.05

        # 3. Variance of REINFORCE is strictly larger than reparameterization trick
        assert res["reinforce_var"] > res["reparam_var"]
        assert res["var_ratio"] > 1.2

    def test_exercise_19_2_affine_transformation(self):
        """Exercise 19.2: Test 1D affine transformation of standard normal."""
        mu = 3.0
        sigma = 1.5
        res = verify_exercise_19_2(mu=mu, sigma=sigma, N=50000, random_state=42)

        # 1. Sample statistics match target parameters
        assert res["mean_err"] < 0.05
        assert res["var_err"] < 0.1

        # 2. Transformed density matches standard Gaussian density transformed by Jacobian
        assert res["max_density_diff"] < 1e-12

        # 3. KS test does not reject normality
        assert res["ks_pval"] > 0.01

    def test_exercise_19_3_multivariate_affine(self):
        """Exercise 19.3: Test multivariate affine transformation and covariance factorization."""
        res = verify_exercise_19_3(D=3, N=50000, random_state=42)

        # 1. Mean and covariance match within Monte Carlo tolerance
        assert res["mean_err"] < 0.05
        assert res["cov_err"] < 0.1

        # 2. Cholesky factorization identity L @ L.T = Sigma
        assert res["chol_recon_err"] < 1e-12

        # 3. Multivariate change-of-variables density matches analytical MVN PDF
        assert res["max_p_diff"] < 1e-12

    def test_exercise_19_4_gaussian_kl_and_gradients(self):
        """Exercise 19.4: Test analytical Gaussian KL and analytical gradients."""
        res = verify_exercise_19_4(D=4, random_state=42)

        # 1. Analytical KL matches Monte Carlo integration
        assert res["kl_mc_diff"] < 0.05

        # 2. Minimum at mu=0, sigma2=1 where KL is exactly 0
        assert res["kl_zero"] == pytest.approx(0.0, abs=1e-12)

        # 3. Non-negativity
        assert res["kl_ana"] >= 0.0

        # 4. Analytical gradients match finite differences
        assert res["err_grad_mu"] < 1e-4
        assert res["err_grad_log_s2"] < 1e-4

    def test_exercise_19_5_elbo_equivalence(self):
        """Exercise 19.5: Test equivalence of Form A and Form B of ELBO."""
        rng = np.random.RandomState(42)
        D = 3
        Dx = 5
        mu = rng.randn(D)
        sigma2 = np.exp(rng.randn(D) * 0.5)
        x = rng.randn(Dx)
        W = rng.randn(Dx, D)

        res = verify_exercise_19_5(mu, sigma2, x, W, obs_var=0.5)

        # Both forms must yield identical values
        assert res["diff"] < 1e-11
        assert res["elbo_A"] == pytest.approx(res["elbo_B"], abs=1e-10)

    def test_exercise_19_6_unconstrained_optimal_variational(self):
        """Exercise 19.6: Test unconstrained optimal distribution recovers true posterior."""
        res = verify_exercise_19_6(x=1.2, random_state=42)

        # 1. At optimal q*, ELBO equals exact marginal log-likelihood ln p(x)
        assert res["diff_opt"] < 1e-12
        assert res["kl_opt"] == pytest.approx(0.0, abs=1e-12)

        # 2. At suboptimal q, ELBO is strictly less than ln p(x)
        assert res["gap_sub"] > 0.01

        # 3. The gap ln p(x) - ELBO(q) exactly equals KL(q || p(z|x))
        assert res["kl_gap_diff"] < 1e-12
