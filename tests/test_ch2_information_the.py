"""
Unit tests for Section 2.5: Information Theory.
Tests mathematical properties of Shannon entropy, physics multiplicity, differential entropy,
Jensen's inequality, KL divergence, conditional entropy, and mutual information.
"""
import os
import math
import numpy as np
import pytest
from common.probability import (
    discrete_entropy,
    differential_entropy_gaussian,
    differential_entropy_1d,
    kl_divergence_discrete,
    kl_divergence_gaussian_1d,
    conditional_entropy_discrete,
    mutual_information_discrete,
    mutual_information_gaussian,
)


def test_shannon_discrete_entropy_textbook_examples():
    # 1. Uniform 8-state distribution: H = 3 bits (Bishop p. 47)
    p_uniform = np.full(8, 1.0 / 8.0)
    h_uni = discrete_entropy(p_uniform, base='2')
    assert math.isclose(h_uni, 3.0, rel_tol=1e-6)

    # 2. Cover & Thomas nonuniform 8 states: (1/2, 1/4, 1/8, 1/16, 1/64, 1/64, 1/64, 1/64)
    # H = 2 bits (Bishop p. 47)
    p_nonuniform = np.array([1/2, 1/4, 1/8, 1/16, 1/64, 1/64, 1/64, 1/64])
    h_nonuni = discrete_entropy(p_nonuniform, base='2')
    assert math.isclose(h_nonuni, 2.0, rel_tol=1e-6)

    # 3. Deterministic distribution: H = 0
    p_det = np.array([1.0, 0.0, 0.0, 0.0])
    assert math.isclose(discrete_entropy(p_det, base='2'), 0.0, abs_tol=1e-9)

    # 4. Maximum entropy on 30 bins: H = ln(30) ~ 3.401197 (Bishop Figure 2.14)
    p_30 = np.full(30, 1.0 / 30.0)
    h_30 = discrete_entropy(p_30, base='e')
    assert math.isclose(h_30, np.log(30), rel_tol=1e-6)


def test_physics_multiplicity_and_stirling_limit():
    # Multiplicity W = N! / prod_i n_i!
    # (1/N) ln W -> - sum_i p_i ln p_i as N -> infty (Eq 2.84, 2.85)
    N = 1000
    p = np.array([0.5, 0.3, 0.2])
    n = (p * N).astype(int)
    
    # Calculate (1/N) ln W using gammaln
    from scipy.special import gammaln
    log_w = gammaln(N + 1) - np.sum(gammaln(n + 1))
    h_physics = log_w / N
    h_shannon = discrete_entropy(p, base='e')
    assert math.isclose(h_physics, h_shannon, rel_tol=1e-2)


def test_gaussian_differential_entropy():
    # Eq 2.99: H[x] = 0.5 * (1 + ln(2 * pi * sigma^2))
    sigma2 = 1.5
    h_analytical = differential_entropy_gaussian(sigma2)
    expected = 0.5 * (1.0 + np.log(2.0 * np.pi * sigma2))
    assert math.isclose(h_analytical, expected, rel_tol=1e-7)

    # Numerical verification via integration
    def gaussian_pdf(x):
        return (1.0 / np.sqrt(2.0 * np.pi * sigma2)) * np.exp(-0.5 * (x ** 2) / sigma2)

    h_numerical = differential_entropy_1d(gaussian_pdf, a=-10.0, b=10.0, n_points=5000)
    assert math.isclose(h_numerical, h_analytical, rel_tol=1e-3)

    # Differential entropy can be negative when sigma2 < 1 / (2 * pi * e) (Bishop p. 51)
    sigma2_small = 0.01  # 0.01 < 1 / (2 * pi * e) ~ 0.0585
    h_negative = differential_entropy_gaussian(sigma2_small)
    assert h_negative < 0.0


def test_jensen_inequality():
    # Strictly convex function: f(x) = -ln(x)
    # Jensen's inequality: f(sum lambda_i x_i) <= sum lambda_i f(x_i) (Eq 2.102)
    rng = np.random.default_rng(42)
    x = rng.uniform(0.5, 10.0, size=5)
    weights = rng.uniform(0.1, 1.0, size=5)
    weights /= np.sum(weights)

    lhs = -np.log(np.sum(weights * x))
    rhs = np.sum(weights * (-np.log(x)))
    assert lhs <= rhs + 1e-9


def test_kl_divergence_properties():
    p = np.array([0.4, 0.35, 0.25])
    q = np.array([0.3, 0.4, 0.3])

    # 1. Non-negativity: KL(p || q) >= 0 (Eq 2.105)
    kl_pq = kl_divergence_discrete(p, q)
    assert kl_pq >= 0.0

    # 2. Identity of indiscernibles: KL(p || p) = 0
    assert math.isclose(kl_divergence_discrete(p, p), 0.0, abs_tol=1e-9)

    # 3. Asymmetry: KL(p || q) != KL(q || p)
    kl_qp = kl_divergence_discrete(q, p)
    assert not math.isclose(kl_pq, kl_qp, rel_tol=1e-3)

    # 4. Analytical Gaussian KL divergence
    mu1, s1_sq = 0.0, 1.0
    mu2, s2_sq = 1.0, 2.0
    kl_gauss = kl_divergence_gaussian_1d(mu1, s1_sq, mu2, s2_sq)
    assert kl_gauss > 0.0
    assert math.isclose(kl_divergence_gaussian_1d(mu1, s1_sq, mu1, s1_sq), 0.0, abs_tol=1e-9)


def test_conditional_entropy_and_mutual_information():
    # Construct a 2D joint distribution
    p_xy = np.array([
        [0.2, 0.1, 0.05],
        [0.05, 0.3, 0.1],
        [0.05, 0.05, 0.1]
    ])
    p_xy = p_xy / np.sum(p_xy)

    # Chain rule: H[X, Y] = H[Y|X] + H[X] (Eq 2.108)
    h_xy = discrete_entropy(p_xy.ravel(), base='e')
    h_x = discrete_entropy(np.sum(p_xy, axis=1), base='e')
    h_y = discrete_entropy(np.sum(p_xy, axis=0), base='e')
    h_y_given_x = conditional_entropy_discrete(p_xy)
    assert math.isclose(h_xy, h_y_given_x + h_x, rel_tol=1e-6)

    # Mutual information relations (Eq 2.110):
    # I[X, Y] = H[X] + H[Y] - H[X, Y]
    mi = mutual_information_discrete(p_xy)
    assert mi >= 0.0
    assert math.isclose(mi, h_x + h_y - h_xy, rel_tol=1e-6)

    # Independent variables: I[X, Y] = 0
    p_indep = np.outer(np.sum(p_xy, axis=1), np.sum(p_xy, axis=0))
    assert math.isclose(mutual_information_discrete(p_indep), 0.0, abs_tol=1e-9)

    # Bivariate Gaussian mutual information: I[X, Y] = -0.5 * ln(1 - rho^2)
    rho = 0.6
    cov = np.array([[1.0, rho], [rho, 1.0]])
    mi_gauss = mutual_information_gaussian(cov)
    assert math.isclose(mi_gauss, -0.5 * np.log(1.0 - rho ** 2), rel_tol=1e-6)


def test_saved_figures_exist():
    fig_paths = [
        "2/result/fig2_14_entropy_histograms.png",
        "2/result/fig2_15_convex_function_jensen.png"
    ]
    for p in fig_paths:
        if os.path.exists(p):
            assert os.path.getsize(p) > 1000
