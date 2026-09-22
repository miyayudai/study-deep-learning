"""
Unit tests for Chapter 2 Exercises: Exercises 2.1 through 2.41.
Covers all analytical and numerical exercises in Bishop & Bishop (2024) Chapter 2:
- Rules of probability (screening, Efron dice, convolution)
- Probability densities (Uniform, Exponential, Laplace, Dirac delta, moments, conditional variance)
- The Gaussian distribution (normalization, moments, MLE, MLE bias)
- Transformation of densities (Jacobian, monotonic transformations)
- Information theory (entropy additivity, max entropy, KL divergence, alpha-divergence,
  subadditivity, linear transformation, conditional entropy, Jensen's inequality / AM-GM,
  empirical KL to NLL, joint entropy & Venn diagram, mutual information)
- Bayesian probabilities (uncorrelated vs independent, bent coin Bayes' theorem, MAP regularized error)
"""
import math
import numpy as np
import pytest
from scipy import integrate

from common.probability import (
    UniformDistribution,
    ExponentialDistribution,
    LaplaceDistribution,
    Gaussian1D,
    gaussian_maximum_likelihood,
    simulate_gaussian_mle_bias,
    discrete_entropy,
    kl_divergence_discrete,
    kl_divergence_gaussian_1d,
    conditional_entropy_discrete,
    mutual_information_discrete,
    alpha_divergence_discrete,
    bent_coin_bayes,
    binary_joint_entropy_analysis,
    EFRON_DICE,
    efron_dice_win_probability,
    BayesianLinearRegression,
    medical_screening_model
)
from common.polynomial import PolynomialRegression


# ============================================================================
# Section 2.1 Exercises (2.1 - 2.3)
# ============================================================================

def test_ex2_1_medical_screening_low_prevalence():
    """
    Exercise 2.1: Medical screening with lower prior prevalence p(C=1) = 0.001.
    p(T=1|C=1) = 0.90, p(T=1|C=0) = 0.03.
    """
    res = medical_screening_model(
        p_cancer=0.001,
        p_pos_given_cancer=0.90,
        p_pos_given_no_cancer=0.03
    )
    expected_p_pos = 0.90 * 0.001 + 0.03 * 0.999  # 0.03087
    expected_post = (0.90 * 0.001) / expected_p_pos  # 90 / 3087 ~= 0.0291545

    assert res["p_pos"] == pytest.approx(expected_p_pos, rel=1e-7)
    assert res["p_cancer_given_pos"] == pytest.approx(expected_post, rel=1e-7)
    assert res["p_cancer_given_pos"] < 0.03  # Even with positive test, cancer prob is ~2.9%


def test_ex2_2_efron_dice_non_transitivity():
    """
    Exercise 2.2: Figure 2.16 non-transitive cubical dice (Efron dice).
    Each die has 2/3 probability of rolling a higher number than the previous die.
    Cycle: Yellow -> Blue -> Green -> Red -> Yellow.
    """
    p_by = efron_dice_win_probability(EFRON_DICE['Blue'], EFRON_DICE['Yellow'])
    p_gb = efron_dice_win_probability(EFRON_DICE['Green'], EFRON_DICE['Blue'])
    p_rg = efron_dice_win_probability(EFRON_DICE['Red'], EFRON_DICE['Green'])
    p_yr = efron_dice_win_probability(EFRON_DICE['Yellow'], EFRON_DICE['Red'])

    assert math.isclose(p_by, 2.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(p_gb, 2.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(p_rg, 2.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(p_yr, 2.0 / 3.0, rel_tol=1e-6)


def test_ex2_3_convolution_of_independent_variables():
    """
    Exercise 2.3: Convolution p(y) = int p_u(u) p_v(y - u) du for y = u + v.
    Test with two standard normal distributions: convolution must be N(0, 2).
    """
    mu1, var1 = 0.5, 1.2
    mu2, var2 = -0.3, 0.8
    expected_mu = mu1 + mu2
    expected_var = var1 + var2

    def p_u(u):
        return (1.0 / np.sqrt(2 * np.pi * var1)) * np.exp(-((u - mu1) ** 2) / (2 * var1))

    def p_v(v):
        return (1.0 / np.sqrt(2 * np.pi * var2)) * np.exp(-((v - mu2) ** 2) / (2 * var2))

    y_test_points = np.array([-1.0, 0.0, 0.2, 1.0, 2.5])
    for y in y_test_points:
        conv_val, _ = integrate.quad(lambda u: p_u(u) * p_v(y - u), -15, 15)
        expected_val = (1.0 / np.sqrt(2 * np.pi * expected_var)) * np.exp(-((y - expected_mu) ** 2) / (2 * expected_var))
        assert conv_val == pytest.approx(expected_val, rel=1e-4)


# ============================================================================
# Section 2.2 Exercises (2.4 - 2.11)
# ============================================================================

def test_ex2_4_uniform_moments():
    """
    Exercise 2.4: Uniform distribution U(x | a, b) normalization, mean, variance.
    E[x] = (a+b)/2, var[x] = (b-a)^2 / 12.
    """
    a, b = 2.0, 8.0
    dist = UniformDistribution(a=a, b=b)
    assert dist.mean == pytest.approx((a + b) / 2.0)
    assert dist.variance == pytest.approx(((b - a) ** 2) / 12.0)

    # Numerical integration check
    norm, _ = integrate.quad(lambda x: dist.pdf(np.array([x]))[0], a, b)
    assert norm == pytest.approx(1.0)


def test_ex2_5_exponential_and_laplace_normalization():
    """
    Exercise 2.5: Normalization of Exponential(lambda) and Laplace(mu, b).
    """
    exp_dist = ExponentialDistribution(lam=2.5)
    norm_exp, _ = integrate.quad(lambda x: exp_dist.pdf(np.array([x]))[0], 0, 50)
    assert norm_exp == pytest.approx(1.0, rel=1e-6)

    lap_dist = LaplaceDistribution(mu=1.5, gamma=0.8)
    norm_lap, _ = integrate.quad(lambda x: lap_dist.pdf(np.array([x]))[0], -20, 20)
    assert norm_lap == pytest.approx(1.0, rel=1e-6)


def test_ex2_6_ex2_7_empirical_density_and_mc_expectation():
    """
    Exercise 2.6 & 2.7: Dirac delta empirical density normalization & MC expectation.
    """
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=3.0, scale=1.5, size=50000)
    
    # E[x] ~= 3.0, E[x^2] ~= 3^2 + 1.5^2 = 11.25
    mc_mean = np.mean(samples)
    mc_sq = np.mean(samples ** 2)
    
    assert mc_mean == pytest.approx(3.0, abs=0.05)
    assert mc_sq == pytest.approx(11.25, abs=0.1)


def test_ex2_8_variance_identity():
    """
    Exercise 2.8: var[f(x)] = E[f(x)^2] - (E[f(x)])^2.
    """
    rng = np.random.default_rng(123)
    x = rng.uniform(0, 10, 100000)
    f_x = np.sin(x) + 0.5 * x
    
    var_direct = np.var(f_x)
    var_identity = np.mean(f_x ** 2) - (np.mean(f_x) ** 2)
    assert var_direct == pytest.approx(var_identity, rel=1e-7)


def test_ex2_9_independent_variables_zero_covariance():
    """
    Exercise 2.9: If x and y are independent, cov[x, y] = 0.
    """
    rng = np.random.default_rng(456)
    x = rng.normal(2.0, 1.0, 100000)
    y = rng.exponential(3.0, 100000)
    
    cov_xy = np.mean((x - np.mean(x)) * (y - np.mean(y)))
    assert abs(cov_xy) < 0.02  # Covariance effectively 0


def test_ex2_10_sum_of_independent_variables_mean_and_variance():
    """
    Exercise 2.10: E[x + z] = E[x] + E[z], var[x + z] = var[x] + var[z].
    """
    rng = np.random.default_rng(789)
    x = rng.normal(1.5, 2.0, 100000)  # var = 4.0
    z = rng.gamma(shape=2.0, scale=1.5, size=100000)  # mean = 3.0, var = 4.5
    
    sum_xz = x + z
    assert np.mean(sum_xz) == pytest.approx(np.mean(x) + np.mean(z), rel=1e-4)
    assert np.var(sum_xz) == pytest.approx(np.var(x) + np.var(z), rel=0.03)


def test_ex2_11_law_of_total_expectation_and_variance():
    """
    Exercise 2.11: Law of total expectation E[x] = E_y[E_x[x|y]]
    and law of total variance var[x] = E_y[var_x[x|y]] + var_y[E_x[x|y]].
    """
    rng = np.random.default_rng(999)
    y = rng.binomial(n=1, p=0.4, size=200000)
    x = np.where(y == 0, rng.normal(1.0, 1.0, 200000), rng.normal(5.0, 2.0, 200000))

    assert np.mean(x) == pytest.approx(2.6, abs=0.03)
    assert np.var(x) == pytest.approx(6.04, abs=0.06)


# ============================================================================
# Section 2.3 Exercises (2.12 - 2.18)
# ============================================================================

def test_ex2_12_gaussian_normalization_polar():
    """
    Exercise 2.12: Prove normalization of univariate Gaussian via polar coordinates.
    I = sqrt(2 * pi * sigma^2).
    """
    sigma = 2.5
    integral_val, _ = integrate.quad(lambda x: np.exp(- (x ** 2) / (2 * (sigma ** 2))), -np.inf, np.inf)
    expected = np.sqrt(2 * np.pi * (sigma ** 2))
    assert integral_val == pytest.approx(expected, rel=1e-7)


def test_ex2_13_gaussian_moments():
    """
    Exercise 2.13: E[x] = mu, E[(x-mu)^2] = sigma^2, E[x^2] = mu^2 + sigma^2.
    """
    mu, sigma2 = 3.0, 2.25
    g = Gaussian1D(mu=mu, sigma2=sigma2)
    
    # Check pdf expectation integrals
    mean_val, _ = integrate.quad(lambda x: x * g.pdf(x), -15, 15)
    var_val, _ = integrate.quad(lambda x: ((x - mu) ** 2) * g.pdf(x), -15, 15)
    second_moment, _ = integrate.quad(lambda x: (x ** 2) * g.pdf(x), -15, 15)
    
    assert mean_val == pytest.approx(mu, rel=1e-6)
    assert var_val == pytest.approx(sigma2, rel=1e-6)
    assert second_moment == pytest.approx(mu ** 2 + sigma2, rel=1e-6)


def test_ex2_14_gaussian_mode():
    """
    Exercise 2.14: Show mode of Gaussian is given by mu.
    """
    mu, sigma2 = -1.2, 0.49
    g = Gaussian1D(mu=mu, sigma2=sigma2)
    x_grid = np.linspace(-5, 3, 10001)
    max_idx = np.argmax(g.pdf(x_grid))
    assert x_grid[max_idx] == pytest.approx(mu, abs=1e-3)


def test_ex2_15_gaussian_mle_derivation():
    """
    Exercise 2.15: Verify formulas for mu_ML and sigma_ML^2.
    """
    data = np.array([1.2, 2.5, 3.1, 2.8, 1.9])
    res = gaussian_maximum_likelihood(data)
    
    assert res['mu_ML'] == pytest.approx(np.mean(data))
    assert res['sigma2_ML'] == pytest.approx(np.var(data))  # ddof=0 for MLE


def test_ex2_16_mle_bias_exact_formula():
    """
    Exercise 2.16: E[sigma_ML^2] = (N-1)/N * sigma^2.
    """
    mu_true, sigma2_true = 0.0, 4.0
    N = 4
    bias_sim = simulate_gaussian_mle_bias(mu=mu_true, sigma2=sigma2_true, N=N, n_trials=20000, seed=42)
    expected_ratio = (N - 1) / N  # 3/4 = 0.75
    
    assert bias_sim['E_sigma2_ML'] / sigma2_true == pytest.approx(expected_ratio, rel=0.03)


def test_ex2_17_unbiased_estimator_known_mean():
    """
    Exercise 2.17: E[1/N * sum (x_n - mu)^2] = sigma^2 (unbiased with known mean).
    """
    rng = np.random.default_rng(101)
    mu_true, sigma2_true = 2.0, 5.0
    N = 5
    samples = rng.normal(mu_true, np.sqrt(sigma2_true), size=(20000, N))
    unbiased_var_estimates = np.mean((samples - mu_true) ** 2, axis=1)
    
    assert np.mean(unbiased_var_estimates) == pytest.approx(sigma2_true, rel=0.02)


def test_ex2_18_linear_regression_mle_noise_variance():
    """
    Exercise 2.18: sigma_ML^2 = 1/N * sum (y(x_n; w_ML) - t_n)^2.
    """
    rng = np.random.default_rng(202)
    x = np.linspace(0, 1, 20)
    w_true = np.array([1.0, 2.5])
    y_clean = w_true[0] + w_true[1] * x
    t = y_clean + rng.normal(0, 0.3, 20)
    
    model = PolynomialRegression(degree=1, l2_reg=0.0).fit(x, t)
    y_pred = model.predict(x)
    sigma2_ml = np.mean((y_pred - t) ** 2)
    
    assert sigma2_ml == pytest.approx(0.09, abs=0.06)


# ============================================================================
# Section 2.4 Exercises (2.19 - 2.20)
# ============================================================================

def test_ex2_19_monotonic_density_transformation():
    """
    Exercise 2.19: p(y) = q(x) / |f'(x)|.
    Let q(x) = U(0, 1) and y = f(x) = x^2 (f'(x) = 2x = 2*sqrt(y)).
    Then p(y) = 1 / (2*sqrt(y)) for 0 < y < 1.
    """
    # Verify integral of p(y) is 1
    norm, _ = integrate.quad(lambda y: 1.0 / (2.0 * np.sqrt(y)), 0, 1)
    assert norm == pytest.approx(1.0)


def test_ex2_20_jacobian_elements():
    """
    Exercise 2.20: Jacobian matrix for:
    y1 = x1 + tanh(5*x1)
    y2 = x2 + tanh(5*x2) + x1^3 / 3
    J_11 = 1 + 5*sech^2(5*x1), J_12 = 0
    J_21 = x1^2, J_22 = 1 + 5*sech^2(5*x2)
    det J = (1 + 5*sech^2(5*x1)) * (1 + 5*sech^2(5*x2))
    """
    def sech(val):
        return 1.0 / np.cosh(val)

    x1, x2 = 0.4, -0.6
    j11 = 1.0 + 5.0 * (sech(5.0 * x1) ** 2)
    j12 = 0.0
    j21 = x1 ** 2
    j22 = 1.0 + 5.0 * (sech(5.0 * x2) ** 2)
    det_j_analytical = j11 * j22

    # Numerical derivative check
    eps = 1e-6
    y1 = lambda a, b: a + np.tanh(5.0 * a)
    y2 = lambda a, b: b + np.tanh(5.0 * b) + (a ** 3) / 3.0
    
    num_j11 = (y1(x1 + eps, x2) - y1(x1 - eps, x2)) / (2 * eps)
    num_j12 = (y1(x1, x2 + eps) - y1(x1, x2 - eps)) / (2 * eps)
    num_j21 = (y2(x1 + eps, x2) - y2(x1 - eps, x2)) / (2 * eps)
    num_j22 = (y2(x1, x2 + eps) - y2(x1, x2 - eps)) / (2 * eps)
    
    assert j11 == pytest.approx(num_j11, rel=1e-5)
    assert j12 == pytest.approx(num_j12, abs=1e-5)
    assert j21 == pytest.approx(num_j21, rel=1e-5)
    assert j22 == pytest.approx(num_j22, rel=1e-5)
    assert det_j_analytical == pytest.approx(num_j11 * num_j22 - num_j12 * num_j21, rel=1e-5)


# ============================================================================
# Section 2.5 Exercises (2.21 - 2.38)
# ============================================================================

def test_ex2_21_entropy_additivity_log_form():
    """
    Exercise 2.21: Show h(p) proportional to -ln(p) from additivity h(p1 * p2) = h(p1) + h(p2).
    """
    c = 1.0
    h = lambda p: -c * np.log2(p)
    p1, p2 = 0.25, 0.5
    assert h(p1 * p2) == pytest.approx(h(p1) + h(p2))


def test_ex2_22_ex2_23_maximum_entropy_discrete():
    """
    Exercise 2.22 & 2.23: Discrete entropy H[x] <= ln(M) with equality for uniform.
    """
    M = 5
    p_uniform = np.ones(M) / M
    p_non_uniform = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
    
    h_max = discrete_entropy(p_uniform, base='e')
    h_sub = discrete_entropy(p_non_uniform, base='e')
    
    assert h_max == pytest.approx(np.log(M))
    assert h_sub < h_max


def test_ex2_24_ex2_25_continuous_gaussian_entropy():
    """
    Exercise 2.24 & 2.25: Differential entropy of univariate Gaussian is 1/2 * (1 + ln(2*pi*sigma^2)).
    """
    sigma2 = 3.5
    expected_entropy = 0.5 * (1.0 + np.log(2 * np.pi * sigma2))
    
    g = Gaussian1D(mu=0.0, sigma2=sigma2)
    num_ent, _ = integrate.quad(lambda x: -g.pdf(x) * np.log(g.pdf(x)), -20, 20)
    assert num_ent == pytest.approx(expected_entropy, rel=1e-5)


def test_ex2_26_kl_minimization_matches_moments():
    """
    Exercise 2.26: Minimizing KL(p || q) with respect to Gaussian q(x|mu, sigma^2)
    yields mu = E_p[x] and sigma^2 = var_p[x].
    """
    # Let p(x) be a mixture of two Gaussians: 0.5*N(-1, 0.5^2) + 0.5*N(2, 1.0^2)
    # E_p[x] = 0.5*(-1) + 0.5*(2) = 0.5
    # E_p[x^2] = 0.5*(1 + 0.25) + 0.5*(4 + 1.0) = 0.625 + 2.5 = 3.125
    # var_p[x] = 3.125 - 0.25 = 2.875
    def p(x):
        return 0.5 * Gaussian1D(-1.0, 0.25).pdf(x) + 0.5 * Gaussian1D(2.0, 1.0).pdf(x)

    def cross_entropy(params):
        mu_q, s2_q = params
        q = Gaussian1D(mu_q, s2_q)
        val, _ = integrate.quad(lambda x: -p(x) * np.log(q.pdf(x) + 1e-12), -10, 10)
        return val

    # Optimize numerically
    from scipy.optimize import minimize
    res = minimize(cross_entropy, [0.0, 2.0], bounds=[(-5, 5), (0.1, 10)])
    opt_mu, opt_var = res.x
    
    assert opt_mu == pytest.approx(0.5, abs=0.01)
    assert opt_var == pytest.approx(2.875, abs=0.05)


def test_ex2_27_kl_between_two_gaussians():
    """
    Exercise 2.27: KL(N(mu, sigma^2) || N(m, s^2)) = 1/2 [ ln(s^2/sigma^2) + (sigma^2 + (mu-m)^2)/s^2 - 1 ].
    """
    mu, sigma2 = 1.0, 2.0
    m, s2 = 0.0, 3.0
    
    kl_formula = kl_divergence_gaussian_1d(mu, sigma2, m, s2)
    
    # Numerical integration
    p = Gaussian1D(mu, sigma2)
    q = Gaussian1D(m, s2)
    kl_num, _ = integrate.quad(lambda x: p.pdf(x) * np.log(p.pdf(x) / q.pdf(x)), -15, 15)
    
    assert kl_formula == pytest.approx(kl_num, rel=1e-5)


def test_ex2_28_alpha_divergence_limits():
    """
    Exercise 2.28: Alpha divergence limits:
    lim_{alpha -> 1} D_alpha(p || q) = KL(p || q)
    lim_{alpha -> -1} D_alpha(p || q) = KL(q || p)
    """
    p = np.array([0.2, 0.5, 0.3])
    q = np.array([0.3, 0.4, 0.3])
    
    kl_pq = kl_divergence_discrete(p, q, base='e')
    kl_qp = kl_divergence_discrete(q, p, base='e')
    
    d_alpha_pos1 = alpha_divergence_discrete(p, q, alpha=0.9999)
    d_alpha_neg1 = alpha_divergence_discrete(p, q, alpha=-0.9999)
    
    assert d_alpha_pos1 == pytest.approx(kl_pq, rel=1e-3)
    assert d_alpha_neg1 == pytest.approx(kl_qp, rel=1e-3)


def test_ex2_29_entropy_subadditivity():
    """
    Exercise 2.29: H[x, y] <= H[x] + H[y] with equality iff independent.
    """
    # Correlated discrete distribution
    p_xy_corr = np.array([[0.4, 0.1], [0.1, 0.4]])
    h_xy = discrete_entropy(p_xy_corr.flatten(), base='e')
    h_x = discrete_entropy(np.sum(p_xy_corr, axis=1), base='e')
    h_y = discrete_entropy(np.sum(p_xy_corr, axis=0), base='e')
    assert h_xy < (h_x + h_y)

    # Independent distribution
    p_xy_ind = np.outer([0.6, 0.4], [0.7, 0.3])
    h_xy_ind = discrete_entropy(p_xy_ind.flatten(), base='e')
    h_x_ind = discrete_entropy(np.sum(p_xy_ind, axis=1), base='e')
    h_y_ind = discrete_entropy(np.sum(p_xy_ind, axis=0), base='e')
    assert h_xy_ind == pytest.approx(h_x_ind + h_y_ind)


def test_ex2_30_linear_transformation_entropy():
    """
    Exercise 2.30: y = A x => H[y] = H[x] + ln|det A|.
    """
    A = np.array([[2.0, 1.0], [0.0, 3.0]])  # det A = 6.0
    det_A = np.abs(np.linalg.det(A))
    
    # 2D Gaussian entropy: H = 1/2 * ln((2*pi*e)^D * |Sigma|)
    cov_x = np.eye(2)
    cov_y = A @ cov_x @ A.T  # det cov_y = (det A)^2 * det cov_x = 36
    
    h_x = 0.5 * np.log(((2 * np.pi * np.e) ** 2) * np.linalg.det(cov_x))
    h_y = 0.5 * np.log(((2 * np.pi * np.e) ** 2) * np.linalg.det(cov_y))
    
    assert h_y == pytest.approx(h_x + np.log(det_A), rel=1e-7)


def test_ex2_31_zero_conditional_entropy_deterministic():
    """
    Exercise 2.31: H[y|x] = 0 iff y is a deterministic function of x.
    """
    # Deterministic mapping: y = f(x)
    p_xy_det = np.array([
        [1/3, 0.0],
        [0.0, 1/3],
        [1/3, 0.0]
    ])
    h_y_given_x = conditional_entropy_discrete(p_xy_det)
    assert h_y_given_x == pytest.approx(0.0, abs=1e-10)


def test_ex2_32_ex2_33_strictly_convex_and_jensens():
    """
    Exercise 2.32 & 2.33: f''(x) > 0 implies strict convexity and Jensen's inequality.
    """
    f = lambda x: x ** 2
    weights = np.array([0.2, 0.3, 0.5])
    points = np.array([1.0, 4.0, 7.0])
    
    f_of_mean = f(np.sum(weights * points))
    mean_of_f = np.sum(weights * f(points))
    
    assert f_of_mean < mean_of_f


def test_ex2_34_kl_empirical_equals_nll():
    """
    Exercise 2.34: KL(p_emp || q_theta) = -1/N sum ln q(x_n | theta) + const.
    """
    data = np.array([1.0, 2.0, 3.0])
    q_dist = Gaussian1D(mu=2.0, sigma2=1.0)
    nll = -np.sum(np.log(q_dist.pdf(data)))
    # NLL is strictly minimized when mu = mean(data)
    assert nll == pytest.approx(3 * 0.5 * np.log(2 * np.pi) + 0.5 * (1 + 0 + 1), rel=1e-5)


def test_ex2_35_ex2_36_binary_joint_distribution_and_venn():
    """
    Exercise 2.35 & 2.36: Binary joint distribution from Bishop 2.36:
    p(x=0, y=0)=1/3, p(x=0, y=1)=1/3, p(x=1, y=0)=0, p(x=1, y=1)=1/3.
    """
    p_xy = np.array([
        [1.0 / 3.0, 1.0 / 3.0],
        [0.0,       1.0 / 3.0]
    ])
    res = binary_joint_entropy_analysis(p_xy)
    
    expected_H_X = np.log(3) - (2.0 / 3.0) * np.log(2)
    expected_H_Y = expected_H_X
    expected_H_XY = np.log(3)
    expected_cond = (2.0 / 3.0) * np.log(2)
    expected_MI = np.log(3) - (4.0 / 3.0) * np.log(2)
    
    assert res['H_X_nats'] == pytest.approx(expected_H_X, rel=1e-7)
    assert res['H_Y_nats'] == pytest.approx(expected_H_Y, rel=1e-7)
    assert res['H_XY_nats'] == pytest.approx(expected_H_XY, rel=1e-7)
    assert res['H_Y_given_X_nats'] == pytest.approx(expected_cond, rel=1e-7)
    assert res['H_X_given_Y_nats'] == pytest.approx(expected_cond, rel=1e-7)
    assert res['I_XY_nats'] == pytest.approx(expected_MI, rel=1e-7)
    
    # Check additive relation Eq (2.108): H[X, Y] = H[Y|X] + H[X]
    assert res['H_XY_nats'] == pytest.approx(res['H_Y_given_X_nats'] + res['H_X_nats'], rel=1e-7)
    # Check mutual information relation: I[X; Y] = H[X] + H[Y] - H[X, Y]
    assert res['I_XY_nats'] == pytest.approx(res['H_X_nats'] + res['H_Y_nats'] - res['H_XY_nats'], rel=1e-7)


def test_ex2_37_am_gm_inequality():
    """
    Exercise 2.37: Arithmetic mean >= Geometric mean via Jensen's inequality with ln(x).
    """
    vals = np.array([2.0, 5.0, 8.0, 15.0])
    am = np.mean(vals)
    gm = np.prod(vals) ** (1.0 / len(vals))
    assert am >= gm


def test_ex2_38_mutual_information_relations():
    """
    Exercise 2.38: I(x; y) = H[x] - H[x|y] = H[y] - H[y|x].
    """
    p_xy = np.array([[0.3, 0.2], [0.1, 0.4]])
    h_x = discrete_entropy(np.sum(p_xy, axis=1), base='e')
    h_y = discrete_entropy(np.sum(p_xy, axis=0), base='e')
    h_xy = discrete_entropy(p_xy.flatten(), base='e')
    
    h_x_given_y = h_xy - h_y
    h_y_given_x = h_xy - h_x
    mi = mutual_information_discrete(p_xy)
    
    assert mi == pytest.approx(h_x - h_x_given_y, rel=1e-7)
    assert mi == pytest.approx(h_y - h_y_given_x, rel=1e-7)


# ============================================================================
# Section 2.6 Exercises (2.39 - 2.41)
# ============================================================================

def test_ex2_39_zero_correlation_not_independent():
    """
    Exercise 2.39: y1 symmetrically distributed around 0, y2 = y1^2.
    cov(y1, y2) = 0, but y1 and y2 are clearly dependent!
    """
    # Let y1 in {-1, 0, 1} with prob 1/3 each
    y1_vals = np.array([-1, 0, 1])
    probs = np.array([1/3, 1/3, 1/3])
    y2_vals = y1_vals ** 2  # {1, 0, 1}
    
    e_y1 = np.sum(probs * y1_vals)  # 0
    e_y2 = np.sum(probs * y2_vals)  # 2/3
    e_y1_y2 = np.sum(probs * (y1_vals * y2_vals))  # (-1 + 0 + 1)/3 = 0
    cov = e_y1_y2 - e_y1 * e_y2  # 0
    
    assert e_y1 == pytest.approx(0.0)
    assert cov == pytest.approx(0.0)
    # But y2 is completely determined by y1, so p(y2|y1) != p(y2)
    assert (y2_vals[0] != e_y2)  # Dependent!


def test_ex2_40_bent_coin_bayesian_update():
    """
    Exercise 2.40: Bent coin Bayesian inference.
    Prior: P(convex is heads) = 0.10 => P(H1) = 0.10, P(H2) = 0.90.
    Under H1: P(heads) = 0.40. Under H2: P(heads) = 0.60.
    Observed 8 heads, 2 tails out of 10 flips.
    """
    res = bent_coin_bayes(
        p_heads_given_H1=0.40,
        p_heads_given_H2=0.60,
        prior_H1=0.10,
        n_heads=8,
        n_tails=2
    )
    
    assert res['posterior_H2'] == pytest.approx(0.99034, abs=1e-4)  # ~99.03%
    assert res['p_next_heads'] == pytest.approx(0.59807, abs=1e-4)  # ~59.81%


def test_ex2_41_map_regularization_exact_match():
    """
    Exercise 2.41: MAP estimate with Gaussian prior N(0, s^2 I) and noise var sigma^2
    is mathematically identical to Ridge regression with lambda = sigma^2 / s^2.
    """
    rng = np.random.default_rng(333)
    x = rng.uniform(0, 1, 12)
    t = 2.0 * x + 1.0 + rng.normal(0, 0.1, 12)
    
    alpha = 4.0  # 1 / s^2
    beta = 100.0  # 1 / sigma^2
    lam = alpha / beta  # lambda = sigma^2 / s^2
    
    bayes_model = BayesianLinearRegression(degree=1, alpha=alpha, beta=beta).fit(x, t)
    ridge_model = PolynomialRegression(degree=1, l2_reg=lam).fit(x, t)
    
    np.testing.assert_allclose(bayes_model.m_N, ridge_model.weights_, rtol=1e-5, atol=1e-5)
