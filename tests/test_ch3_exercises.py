"""
Unit tests for Chapter 3 Exercises: Exercises 3.1 through 3.38.
Covers all theoretical derivations and numerical verifications in
Bishop & Bishop (2024) Chapter 3: Standard Distributions.

Section 3.1: Discrete Variables (Ex 3.1 - 3.4)
Section 3.2: The Multivariate Gaussian (Ex 3.5 - 3.29)
Section 3.3: Periodic Variables (Ex 3.30 - 3.34)
Section 3.4: The Exponential Family (Ex 3.35 - 3.36)
Section 3.5: Nonparametric Methods (Ex 3.37 - 3.38)
"""
import math
import numpy as np
import pytest
from scipy import special, stats, integrate

from common.probability import (
    symmetric_bernoulli_pmf,
    symmetric_bernoulli_moments,
    pascal_triangle_identity,
    multivariate_gaussian_entropy,
    multivariate_gaussian_kl,
    woodbury_matrix_identity,
    partitioned_matrix_inverse,
    mahalanobis_hyperellipsoid_volume,
    gaussian_three_block_marginal_conditional,
    von_mises_mle_estimation,
    histogram_density_lagrange_mle,
    HistogramDensity1D,
    KernelDensity1D,
    KNNDensityEstimator,
    KNNClassifier
)


# ============================================================================
# Section 3.1: Discrete Variables (Exercises 3.1 - 3.4)
# ============================================================================

def test_ex3_1_bernoulli_properties():
    mu_values = [0.05, 0.2, 0.5, 0.75, 0.95]
    for mu in mu_values:
        p0, p1 = 1.0 - mu, mu
        assert np.isclose(p0 + p1, 1.0)
        assert np.isclose(p1, mu)
        assert np.isclose(mu - mu**2, mu * (1.0 - mu))
        h_expected = -mu * np.log(mu) - (1.0 - mu) * np.log(1.0 - mu)
        h_calc = stats.bernoulli(mu).entropy()
        assert np.isclose(h_calc, h_expected)


def test_ex3_2_symmetric_bernoulli():
    for mu in [-0.8, -0.3, 0.0, 0.4, 0.9]:
        p_minus = symmetric_bernoulli_pmf(-1, mu)
        p_plus = symmetric_bernoulli_pmf(1, mu)
        assert np.isclose(p_minus + p_plus, 1.0)
        
        mean, var, ent = symmetric_bernoulli_moments(mu)
        assert np.isclose(mean, mu)
        assert np.isclose(var, 1.0 - mu**2)
        if mu == 0.0:
            assert np.isclose(ent, np.log(2.0))


def test_ex3_3_binomial_pascal_identity_and_normalization():
    for N in [2, 5, 10, 20]:
        for m in range(1, N + 1):
            left1, left2, right = pascal_triangle_identity(N, m)
            assert left1 + left2 == right

        for mu in [0.1, 0.35, 0.7]:
            terms = [math.comb(N, m) * (mu**m) * ((1.0 - mu)**(N - m)) for m in range(N + 1)]
            assert np.isclose(sum(terms), 1.0)


def test_ex3_4_binomial_moments_via_derivatives():
    for N in [5, 12, 30]:
        for mu in [0.2, 0.5, 0.85]:
            dist = stats.binom(N, mu)
            assert np.isclose(dist.mean(), N * mu)
            assert np.isclose(dist.var(), N * mu * (1.0 - mu))


# ============================================================================
# Section 3.2: The Multivariate Gaussian (Exercises 3.5 - 3.29)
# ============================================================================

def test_ex3_5_multivariate_gaussian_mode():
    mu = np.array([1.5, -2.0, 3.0])
    Sigma = np.array([[2.0, 0.5, 0.1], [0.5, 1.5, -0.3], [0.1, -0.3, 1.0]])
    inv_Sigma = np.linalg.inv(Sigma)
    
    def quad_form(x):
        d = x - mu
        return d.T @ inv_Sigma @ d
        
    assert quad_form(mu) == 0.0
    for _ in range(20):
        delta = np.random.randn(3) * 0.5
        x_pert = mu + delta
        assert quad_form(x_pert) > 0.0


def test_ex3_6_linear_transformation_gaussian():
    np.random.seed(42)
    mu = np.array([1.0, 2.0])
    Sigma = np.array([[2.0, 0.8], [0.8, 1.5]])
    A = np.array([[1.5, -0.5], [0.2, 1.2], [-1.0, 0.5]])
    b = np.array([0.5, -1.0, 2.0])
    
    expected_mu_y = A @ mu + b
    expected_Sigma_y = A @ Sigma @ A.T
    
    X = np.random.multivariate_normal(mu, Sigma, size=100000)
    Y = X @ A.T + b
    assert np.allclose(np.mean(Y, axis=0), expected_mu_y, atol=0.03)
    assert np.allclose(np.cov(Y, rowvar=False), expected_Sigma_y, atol=0.05)


def test_ex3_7_gaussian_kl_divergence():
    mu_q = np.array([0.5, -1.0])
    Sigma_q = np.array([[1.2, 0.3], [0.3, 0.8]])
    mu_p = np.array([-0.2, 0.8])
    Sigma_p = np.array([[2.0, -0.4], [-0.4, 1.5]])
    
    kl_analytic = multivariate_gaussian_kl(mu_q, Sigma_q, mu_p, Sigma_p)
    assert kl_analytic > 0.0
    assert np.isclose(multivariate_gaussian_kl(mu_q, Sigma_q, mu_q, Sigma_q), 0.0)
    
    np.random.seed(42)
    samples = np.random.multivariate_normal(mu_q, Sigma_q, size=100000)
    log_q = stats.multivariate_normal(mu_q, Sigma_q).logpdf(samples)
    log_p = stats.multivariate_normal(mu_p, Sigma_p).logpdf(samples)
    kl_mc = np.mean(log_q - log_p)
    assert np.isclose(kl_analytic, kl_mc, rtol=0.03)


def test_ex3_8_maximum_entropy_variational():
    mu = np.array([0.0])
    var = 2.0
    gauss_ent = 0.5 * np.log(2.0 * np.pi * np.e * var)
    
    b = np.sqrt(var / 2.0)
    laplace_ent = np.log(2.0 * b * np.e)
    assert gauss_ent > laplace_ent
    
    w = np.sqrt(3.0 * var)
    uniform_ent = np.log(2.0 * w)
    assert gauss_ent > uniform_ent


def test_ex3_9_multivariate_gaussian_entropy():
    Sigma_2d = np.array([[2.0, 0.6], [0.6, 1.5]])
    ent_2d = multivariate_gaussian_entropy(Sigma_2d)
    expected_2d = stats.multivariate_normal(cov=Sigma_2d).entropy()
    assert np.isclose(ent_2d, expected_2d)
    
    Sigma_3d = np.diag([1.0, 2.0, 3.0])
    ent_3d = multivariate_gaussian_entropy(Sigma_3d)
    expected_3d = stats.multivariate_normal(cov=Sigma_3d).entropy()
    assert np.isclose(ent_3d, expected_3d)


def test_ex3_10_gaussian_convolution_and_entropy():
    mu1, tau1 = 1.0, 2.0
    mu2, tau2 = -0.5, 4.0
    var_sum = 1.0 / tau1 + 1.0 / tau2
    
    expected_ent = 0.5 * np.log(2.0 * np.pi * np.e * var_sum)
    analytic_ent = stats.norm(scale=np.sqrt(var_sum)).entropy()
    assert np.isclose(expected_ent, analytic_ent)


def test_ex3_11_precision_matrix_symmetry():
    np.random.seed(42)
    M = np.random.randn(4, 4)
    A_anti = 0.5 * (M - M.T)
    
    for _ in range(10):
        v = np.random.randn(4)
        quad = float(v.T @ A_anti @ v)
        assert np.isclose(quad, 0.0, atol=1e-12)


def test_ex3_12_eigenvalues_and_orthogonal_eigenvectors():
    Sigma = np.array([[3.0, 1.0, -0.5], [1.0, 2.0, 0.4], [-0.5, 0.4, 1.5]])
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    assert np.all(np.isreal(eigenvalues))
    assert np.allclose(eigenvectors.T @ eigenvectors, np.eye(3))


def test_ex3_13_spectral_expansion_and_inverse():
    Sigma = np.array([[4.0, 1.2], [1.2, 2.0]])
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    
    recon_Sigma = sum(eigenvalues[i] * np.outer(eigenvectors[:, i], eigenvectors[:, i]) for i in range(2))
    assert np.allclose(Sigma, recon_Sigma)
    
    inv_Sigma = np.linalg.inv(Sigma)
    recon_inv = sum((1.0 / eigenvalues[i]) * np.outer(eigenvectors[:, i], eigenvectors[:, i]) for i in range(2))
    assert np.allclose(inv_Sigma, recon_inv)


def test_ex3_14_positive_definiteness_iff_positive_eigenvalues():
    Sigma_pos = np.array([[2.0, 0.5], [0.5, 1.0]])
    vals_pos = np.linalg.eigvalsh(Sigma_pos)
    assert np.all(vals_pos > 0)
    
    Sigma_indef = np.array([[1.0, 2.0], [2.0, 1.0]])
    vals_indef = np.linalg.eigvalsh(Sigma_indef)
    assert np.any(vals_indef < 0)


def test_ex3_15_symmetric_matrix_degrees_of_freedom():
    for D in [1, 2, 3, 5, 10]:
        expected = D * (D + 1) // 2
        diag_elements = D
        off_diag_elements = D * (D - 1) // 2
        assert diag_elements + off_diag_elements == expected


def test_ex3_16_inverse_of_symmetric_matrix_is_symmetric():
    Sigma = np.array([[3.0, 0.8, -0.4], [0.8, 2.5, 0.2], [-0.4, 0.2, 1.8]])
    inv_Sigma = np.linalg.inv(Sigma)
    assert np.allclose(inv_Sigma, inv_Sigma.T)


def test_ex3_17_mahalanobis_hyperellipsoid_volume():
    Sigma = np.array([[4.0, 0.0], [0.0, 9.0]])
    Delta = 1.0
    vol = mahalanobis_hyperellipsoid_volume(Sigma, Delta)
    assert np.isclose(vol, 6.0 * np.pi)


def test_ex3_18_partitioned_matrix_inverse():
    A = np.array([[3.0, 0.5], [0.5, 2.0]])
    B = np.array([[0.2], [-0.1]])
    C = np.array([[0.2, -0.1]])
    D = np.array([[1.5]])
    
    M = np.block([[A, B], [C, D]])
    M_inv_direct = np.linalg.inv(M)
    M_inv_schur = partitioned_matrix_inverse(A, B, C, D)
    assert np.allclose(M_inv_direct, M_inv_schur)


def test_ex3_19_three_block_marginal_conditional():
    mu = np.array([1.0, 2.0, -1.0])
    Sigma = np.array([
        [2.0, 0.5, 0.3],
        [0.5, 1.5, -0.2],
        [0.3, -0.2, 1.0]
    ])
    x_b = np.array([2.5])
    mu_cond, Sigma_cond = gaussian_three_block_marginal_conditional(mu, Sigma, 1, 1, 1, x_b)
    
    Sigma_ab = Sigma[:2, :2]
    mu_ab = mu[:2]
    mu_expected = mu_ab[0] + (Sigma_ab[0, 1] / Sigma_ab[1, 1]) * (x_b[0] - mu_ab[1])
    Sigma_expected = Sigma_ab[0, 0] - (Sigma_ab[0, 1]**2) / Sigma_ab[1, 1]
    
    assert np.isclose(mu_cond[0], mu_expected)
    assert np.isclose(Sigma_cond[0, 0], Sigma_expected)


def test_ex3_20_woodbury_identity():
    np.random.seed(42)
    A = np.eye(3) * 2.0 + np.random.randn(3, 3) * 0.1
    A = A.T @ A
    B = np.random.randn(3, 2)
    C = np.eye(2) + np.random.randn(2, 2) * 0.1
    C = C.T @ C
    D = np.random.randn(2, 3)
    
    lhs, rhs = woodbury_matrix_identity(A, B, C, D)
    assert np.allclose(lhs, rhs)


def test_ex3_21_sum_independent_vectors():
    mu_x = np.array([1.0, -0.5])
    Sigma_x = np.array([[1.0, 0.2], [0.2, 1.5]])
    mu_z = np.array([-1.0, 2.0])
    Sigma_z = np.array([[2.0, -0.3], [-0.3, 0.8]])
    
    expected_mu_y = mu_x + mu_z
    expected_Sigma_y = Sigma_x + Sigma_z
    
    np.random.seed(42)
    X = np.random.multivariate_normal(mu_x, Sigma_x, size=100000)
    Z = np.random.multivariate_normal(mu_z, Sigma_z, size=100000)
    Y = X + Z
    
    assert np.allclose(np.mean(Y, axis=0), expected_mu_y, atol=0.03)
    assert np.allclose(np.cov(Y, rowvar=False), expected_Sigma_y, atol=0.05)


def test_ex3_22_marginal_and_conditional_from_joint():
    mu_x = np.array([1.0])
    Sigma_x = np.array([[0.5]])
    A = np.array([[2.0]])
    b = np.array([1.0])
    Sigma_y_given_x = np.array([[0.3]])
    
    mu_z = np.array([1.0, 3.0])
    Sigma_z = np.array([[0.5, 1.0], [1.0, 0.3 + 2.0]])
    
    assert np.isclose(mu_z[0], mu_x[0])
    assert np.isclose(Sigma_z[0, 0], Sigma_x[0, 0])


def test_ex3_23_precision_inversion_to_covariance():
    Lambda = np.array([[2.5, -1.0], [-1.0, 1.8]])
    cov = np.linalg.inv(Lambda)
    assert np.allclose(cov @ Lambda, np.eye(2))


def test_ex3_24_joint_mean_verification():
    mu = np.array([2.0, -1.0])
    A = np.array([[1.0, 0.5], [-0.5, 2.0]])
    b = np.array([0.5, 1.0])
    
    expected_joint_mean = np.concatenate([mu, A @ mu + b])
    assert np.allclose(expected_joint_mean, [2.0, -1.0, 2.0, -2.0])


def test_ex3_25_linear_gaussian_convolution():
    mu_x, var_x = 1.0, 0.8
    mu_z, var_z = -0.5, 1.2
    
    expected_mu_y = mu_x + mu_z
    expected_var_y = var_x + var_z
    assert expected_mu_y == 0.5
    assert expected_var_y == 2.0


def test_ex3_26_completing_the_square_marginal():
    mu_x = np.array([1.0, 0.5])
    Sigma_x = np.array([[1.0, 0.3], [0.3, 0.8]])
    A = np.array([[1.2, -0.4], [0.5, 1.0]])
    b = np.array([0.2, -0.3])
    Sigma_y_x = np.array([[0.6, 0.1], [0.1, 0.5]])
    
    expected_mu_y = A @ mu_x + b
    expected_Sigma_y = Sigma_y_x + A @ Sigma_x @ A.T
    assert np.all(np.linalg.eigvalsh(expected_Sigma_y) > 0)


def test_ex3_27_completing_the_square_conditional():
    mu_x = np.array([1.0, 0.5])
    Sigma_x = np.array([[1.0, 0.3], [0.3, 0.8]])
    A = np.array([[1.2, -0.4], [0.5, 1.0]])
    b = np.array([0.2, -0.3])
    Sigma_y_x = np.array([[0.6, 0.1], [0.1, 0.5]])
    
    inv_Sigma_x = np.linalg.inv(Sigma_x)
    inv_Sigma_y_x = np.linalg.inv(Sigma_y_x)
    
    Sigma_x_y = np.linalg.inv(inv_Sigma_x + A.T @ inv_Sigma_y_x @ A)
    assert np.all(np.linalg.eigvalsh(Sigma_x_y) > 0)


def test_ex3_28_maximum_likelihood_covariance_derivation():
    np.random.seed(42)
    X = np.random.randn(50, 3)
    mu_ml = np.mean(X, axis=0)
    diff = X - mu_ml
    Sigma_ml = (diff.T @ diff) / len(X)
    
    inv_Sigma_ml = np.linalg.inv(Sigma_ml)
    grad = -0.5 * len(X) * inv_Sigma_ml + 0.5 * inv_Sigma_ml @ (diff.T @ diff) @ inv_Sigma_ml
    assert np.allclose(grad, 0.0, atol=1e-10)


def test_ex3_29_sample_covariance_unbiasedness():
    np.random.seed(42)
    mu = np.array([1.0, -1.0])
    Sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
    N = 10
    
    num_trials = 3000
    S_sum = np.zeros((2, 2))
    for _ in range(num_trials):
        samples = np.random.multivariate_normal(mu, Sigma, size=N)
        S = np.cov(samples, rowvar=False, ddof=1)
        S_sum += S
    S_mean = S_sum / num_trials
    assert np.allclose(S_mean, Sigma, atol=0.09)


# ============================================================================
# Section 3.3: Periodic Variables (Exercises 3.30 - 3.34)
# ============================================================================

def test_ex3_30_euler_trigonometric_identities():
    for A in [0.3, 1.2, -2.5]:
        assert np.isclose(np.cos(A)**2 + np.sin(A)**2, 1.0)
        for B in [0.7, -1.1, 2.0]:
            cos_diff = np.cos(A) * np.cos(B) + np.sin(A) * np.sin(B)
            assert np.isclose(np.cos(A - B), cos_diff)
            sin_diff = np.sin(A) * np.cos(B) - np.cos(A) * np.sin(B)
            assert np.isclose(np.sin(A - B), sin_diff)


def test_ex3_31_von_mises_asymptotic_gaussian():
    theta_0 = 1.0
    m = 50.0
    sigma = 1.0 / np.sqrt(m)
    
    thetas = np.linspace(theta_0 - 3*sigma, theta_0 + 3*sigma, 100)
    vm_pdf = np.exp(m * np.cos(thetas - theta_0)) / (2.0 * np.pi * special.i0(m))
    gauss_pdf = stats.norm.pdf(thetas, theta_0, sigma)
    assert np.allclose(vm_pdf, gauss_pdf, rtol=0.15)


def test_ex3_32_von_mises_mean_direction_mle():
    thetas = np.array([0.2, 0.4, 0.5, 0.8])
    s = np.sum(np.sin(thetas))
    c = np.sum(np.cos(thetas))
    theta_0_mle = np.arctan2(s, c)
    residuals = np.sum(np.sin(thetas - theta_0_mle))
    assert np.isclose(residuals, 0.0, atol=1e-12)


def test_ex3_33_von_mises_extrema():
    theta_0 = 1.5
    m = 2.5
    
    theta_grid = np.linspace(0, 2*np.pi, 1000)
    pdf_vals = np.exp(m * np.cos(theta_grid - theta_0)) / (2.0 * np.pi * special.i0(m))
    
    max_theta = theta_grid[np.argmax(pdf_vals)]
    assert np.isclose(max_theta, theta_0, atol=0.01)
    
    min_theta = theta_grid[np.argmin(pdf_vals)]
    expected_min = (theta_0 + np.pi) % (2.0 * np.pi)
    assert np.isclose(min_theta, expected_min, atol=0.01)


def test_ex3_34_von_mises_concentration_mle():
    np.random.seed(42)
    true_theta_0, true_m = 1.5, 3.0
    samples = stats.vonmises(kappa=true_m, loc=true_theta_0).rvs(size=1000)
    
    theta_0_est, m_est, r = von_mises_mle_estimation(samples)
    assert np.isclose(theta_0_est, true_theta_0, atol=0.1)
    assert np.isclose(m_est, true_m, atol=0.3)
    assert np.isclose(special.i1(m_est) / special.i0(m_est), r, atol=1e-4)


# ============================================================================
# Section 3.4: The Exponential Family (Exercises 3.35 - 3.36)
# ============================================================================

def test_ex3_35_multivariate_gaussian_exponential_family():
    mu = np.array([1.0, -2.0])
    Sigma = np.array([[2.0, 0.4], [0.4, 1.5]])
    inv_Sigma = np.linalg.inv(Sigma)
    
    eta1 = inv_Sigma @ mu
    eta2 = -0.5 * inv_Sigma
    
    x = np.array([0.5, 1.0])
    quad_manual = -0.5 * (x - mu).T @ inv_Sigma @ (x - mu)
    
    term1 = eta1 @ x
    term2 = np.trace(eta2 @ np.outer(x, x))
    log_partition = 0.5 * mu.T @ inv_Sigma @ mu + 0.5 * np.log(np.linalg.det(Sigma)) + np.log(2*np.pi)
    
    log_prob_manual = -0.5 * np.log(np.linalg.det(Sigma)) - np.log(2*np.pi) + quad_manual
    log_prob_exp = term1 + term2 - log_partition
    assert np.isclose(log_prob_manual, log_prob_exp)


def test_ex3_36_exponential_family_second_derivative_covariance():
    for eta in [-2.0, 0.0, 1.5]:
        mu = 1.0 / (1.0 + np.exp(-eta))
        var_x = mu * (1.0 - mu)
        
        eps = 1e-5
        A_plus = np.log(1.0 + np.exp(eta + eps))
        A_center = np.log(1.0 + np.exp(eta))
        A_minus = np.log(1.0 + np.exp(eta - eps))
        d2A = (A_plus - 2*A_center + A_minus) / (eps**2)
        assert np.isclose(d2A, var_x, atol=1e-5)


# ============================================================================
# Section 3.5: Nonparametric Methods (Exercises 3.37 - 3.38)
# ============================================================================

def test_ex3_37_histogram_mle_lagrange():
    counts = np.array([10, 25, 15])
    widths = np.array([0.2, 0.5, 0.3])
    
    h_mle = histogram_density_lagrange_mle(counts, widths)
    assert np.isclose(np.sum(h_mle * widths), 1.0)
    expected = counts / (np.sum(counts) * widths)
    assert np.allclose(h_mle, expected)


def test_ex3_38_knn_density_integral_divergence():
    data = np.array([0.0])
    knn = KNNDensityEstimator(K=1).fit(data)
    
    Rs = [10.0, 100.0, 1000.0]
    integrals = []
    for R in Rs:
        val, _ = integrate.quad(lambda x: 2.0 * float(knn.evaluate(x)), 0.1, R)
        integrals.append(val)
        
    diff1 = integrals[1] - integrals[0]
    diff2 = integrals[2] - integrals[1]
    assert np.isclose(diff1, np.log(10.0), rtol=1e-3)
    assert np.isclose(diff2, np.log(10.0), rtol=1e-3)
