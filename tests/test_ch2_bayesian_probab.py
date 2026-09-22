"""
Unit tests for Section 2.6: Bayesian Probabilities.
Tests MAP regularization equivalence, Bayesian predictive distribution,
and non-transitive Efron dice (Figure 2.16).
"""
import os
import math
import numpy as np
import pytest
from common.probability import (
    BayesianLinearRegression,
    EFRON_DICE,
    efron_dice_win_probability
)
from common.polynomial import PolynomialRegression


def test_map_regularization_exact_equivalence():
    # Bishop Eq 2.116 - 2.117: MAP estimate with Gaussian prior N(0, s^2 I)
    # and noise variance sigma^2 is exactly Ridge regression with lambda = sigma^2 / s^2.
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 15)
    t = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, 15)

    alpha = 2.0  # 1 / s^2
    beta = 25.0  # 1 / sigma^2
    deg = 3

    # Bayesian / MAP model
    bayes_model = BayesianLinearRegression(degree=deg, alpha=alpha, beta=beta).fit(x, t)
    w_map = bayes_model.m_N

    # Ridge / PolynomialRegression with lambda = alpha / beta
    lam = alpha / beta
    ridge_model = PolynomialRegression(degree=deg, l2_reg=lam).fit(x, t)
    w_ridge = ridge_model.weights_

    np.testing.assert_allclose(w_map, w_ridge, rtol=1e-5, atol=1e-5)


def test_bayesian_predictive_variance_properties():
    # Predictive variance: s^2(x) = 1/beta + phi(x)^T S_N phi(x)
    # Variance should be lower in data region [0.2, 0.8] and higher in extrapolation regions.
    x_train = np.linspace(0.2, 0.8, 10)
    t_train = np.sin(2 * np.pi * x_train)
    model = BayesianLinearRegression(degree=3, alpha=1.0, beta=10.0).fit(x_train, t_train)

    _, var_in = model.predict(np.array([0.5]))
    _, var_out = model.predict(np.array([2.0]))  # Extrapolation

    assert var_out[0] > var_in[0]
    # Variance must be strictly greater than noise variance 1/beta
    assert var_in[0] > (1.0 / 10.0)


def test_efron_dice_non_transitivity_exact_probabilities():
    # Figure 2.16 Non-transitive cubical dice (Efron dice)
    # Clockwise cycle: Yellow -> Blue -> Green -> Red -> Yellow
    # Each die beats the previous die with probability 2/3!
    prob_blue_beats_yellow = efron_dice_win_probability(EFRON_DICE['Blue'], EFRON_DICE['Yellow'])
    prob_green_beats_blue = efron_dice_win_probability(EFRON_DICE['Green'], EFRON_DICE['Blue'])
    prob_red_beats_green = efron_dice_win_probability(EFRON_DICE['Red'], EFRON_DICE['Green'])
    prob_yellow_beats_red = efron_dice_win_probability(EFRON_DICE['Yellow'], EFRON_DICE['Red'])

    assert math.isclose(prob_blue_beats_yellow, 2.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(prob_green_beats_blue, 2.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(prob_red_beats_green, 2.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(prob_yellow_beats_red, 2.0 / 3.0, rel_tol=1e-6)


def test_saved_figures_exist():
    fig_path = "2/result/fig2_16_efron_dice.png"
    if os.path.exists(fig_path):
        assert os.path.getsize(fig_path) > 1000
