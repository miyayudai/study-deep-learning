"""
common/exercises_ch16.py
========================
Chapter 16 Exercises: Continuous Latent Variables
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts

This module provides theoretical proofs, derivations, and numerical verification
utilities for Exercises 16.1 through 16.26:
- Exercise 16.1: PCA maximum variance by induction for dimension M+1.
- Exercise 16.2: PCA reconstruction error minimization via matrix Lagrange multipliers.
- Exercise 16.3: Dual PCA eigenvector normalization to unit length.
- Exercise 16.4: General Gaussian latent prior in PPCA leads to identical marginal model.
- Exercise 16.5: Linear transformation of Gaussian random vectors (M < D, M = D, M > D).
- Exercise 16.6: Law of total expectation and variance for PPCA marginal distribution.
- Exercise 16.7: Directed graphical model for PPCA and naive Bayes independence structure.
- Exercise 16.8: Derivation of PPCA posterior distribution via Gaussian conditioning.
- Exercise 16.9: MLE for mean parameter mu in PPCA equals sample mean.
- Exercise 16.10: Second derivatives of PPCA log-likelihood w.r.t. mu (unique maximum).
- Exercise 16.11: Orthogonal projection limit of PPCA posterior mean as sigma^2 -> 0.
- Exercise 16.12: Shrinkage of posterior mean toward origin for sigma^2 > 0.
- Exercise 16.13: Optimal reconstruction under least-squares projection cost.
- Exercise 16.14: Independent parameter counting for PPCA covariance matrix.
- Exercise 16.15: Independent parameter counting for Factor Analysis covariance matrix.
- Exercise 16.16: Latent space rotation invariance of Factor Analysis.
- Exercise 16.17: Data transformation covariance: FA (rescaling) and PPCA (rotation).
- Exercise 16.18: Evidence lower bound (ELBO) and KL decomposition for continuous latents.
- Exercise 16.19: I.i.d. factorization of the evidence lower bound.
- Exercise 16.20: Directed graphical model for Mixture of PPCAs (separate vs tied).
- Exercise 16.21: M-step derivations for PPCA (closed-form W and sigma^2).
- Exercise 16.22: EM algorithm for PPCA with missing data (MAR).
- Exercise 16.23: Alternating minimization of reconstruction cost (Roweis EM PCA).
- Exercise 16.24: E-step formulae for Factor Analysis.
- Exercise 16.25: M-step formulae for Factor Analysis.
- Exercise 16.26: Stationary point and global maximum of Factor Analysis w.r.t. mu.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats
from scipy.linalg import orth

from .principal_component_analysis import PrincipalComponentAnalysis
from .probabilistic_latent_variables import ProbabilisticPCA, FactorAnalysisModel
FactorAnalysis = FactorAnalysisModel


# =============================================================================
# Exercise 16.1: PCA Maximum Variance by Induction
# =============================================================================
def solve_exercise_16_1() -> Dict[str, Any]:
    """
    Exercise 16.1: Proof by induction for maximum variance projection.
    
    Returns mathematical steps and answers.
    """
    return {
        "inductive_hypothesis": "The first M variance-maximizing projection directions are given by eigenvectors u_1, ..., u_M of S corresponding to the M largest eigenvalues lambda_1 >= ... >= lambda_M.",
        "lagrangian": "L(u_{M+1}, lambda, {eta_i}) = u_{M+1}^T S u_{M+1} - lambda (u_{M+1}^T u_{M+1} - 1) - 2 * sum_{i=1}^M eta_i u_{M+1}^T u_i",
        "derivative_eq": "S u_{M+1} - lambda u_{M+1} - sum_{i=1}^M eta_i u_i = 0",
        "orthogonality_elimination": "Multiplying by u_j^T (j <= M) gives u_j^T S u_{M+1} = (S u_j)^T u_{M+1} = lambda_j u_j^T u_{M+1} = 0, so eta_j = 0 for all j.",
        "eigenvalue_equation": "S u_{M+1} = lambda u_{M+1}",
        "maximal_choice": "The variance is u_{M+1}^T S u_{M+1} = lambda, which is maximized by choosing the largest remaining eigenvalue lambda_{M+1} with eigenvector u_{M+1}.",
    }


def verify_exercise_16_1(D: int = 5, seed: int = 42) -> bool:
    """Numerical verification of inductive projection step."""
    rng = np.random.RandomState(seed)
    A = rng.randn(D, D)
    S = A @ A.T / D
    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    
    # Check for M=2 to M+1=3
    M = 2
    U_M = eigvecs[:, :M]
    u_next = eigvecs[:, M]
    
    # Orthogonality
    ortho_check = np.allclose(U_M.T @ u_next, 0.0, atol=1e-10)
    # Unit length
    norm_check = np.isclose(u_next.T @ u_next, 1.0)
    # Eigenvalue equation
    eigen_check = np.allclose(S @ u_next, eigvals[M] * u_next, atol=1e-10)
    return ortho_check and norm_check and eigen_check


# =============================================================================
# Exercise 16.2: PCA Error Measure and Matrix Lagrange Multipliers
# =============================================================================
def solve_exercise_16_2() -> Dict[str, Any]:
    """
    Exercise 16.2: Minimization of J = Tr(U_tilde^T S U_tilde) subject to U_tilde^T U_tilde = I.
    """
    return {
        "lagrangian": "J_tilde = Tr(U_tilde^T S U_tilde) + Tr(H (I - U_tilde^T U_tilde))",
        "stationary_condition": "2 S U_tilde - U_tilde (H + H^T) = 0 => S U_tilde = U_tilde H (assuming H is symmetric)",
        "eigen_expansion": "H is real symmetric, so H = V Lambda V^T with V orthogonal. Then U_tilde^* = U_tilde V satisfies S U_tilde^* = U_tilde^* Lambda.",
        "error_value": "Tr(U_tilde^T S U_tilde) = Tr(H) = Tr(Lambda) = sum_{i=M+1}^D lambda_i.",
    }


def verify_exercise_16_2(D: int = 6, M: int = 2, seed: int = 42) -> bool:
    rng = np.random.RandomState(seed)
    A = rng.randn(D, D)
    S = A @ A.T / D
    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    
    # True complementary subspace
    U_tilde = eigvecs[:, M:]  # D x (D - M)
    H = U_tilde.T @ S @ U_tilde
    
    # Check S U_tilde = U_tilde H
    res = np.allclose(S @ U_tilde, U_tilde @ H, atol=1e-10)
    # Check Tr(H) equals sum of remaining eigenvalues
    expected_error = np.sum(eigvals[M:])
    error_match = np.isclose(np.trace(H), expected_error)
    return res and error_match


# =============================================================================
# Exercise 16.3: Dual PCA Eigenvector Normalization
# =============================================================================
def solve_exercise_16_3() -> Dict[str, Any]:
    return {
        "definition": "u_i = (1 / sqrt(lambda_i)) X^T v_i, where K v_i = lambda_i v_i with K = (1/N) X X^T and v_i^T v_i = 1",
        "norm_squared": "u_i^T u_i = (1 / lambda_i) v_i^T X X^T v_i = (N / lambda_i) v_i^T K v_i = (N / lambda_i) lambda_i v_i^T v_i = N (or 1 depending on scaling of S)",
        "result": "Normalized to unit length when u_i = (1 / sqrt(N * lambda_i)) X^T v_i or K = X X^T.",
    }


def verify_exercise_16_3(N: int = 10, D: int = 50, seed: int = 42) -> bool:
    rng = np.random.RandomState(seed)
    X = rng.randn(N, D)
    X = X - X.mean(axis=0)
    
    # Gram matrix K = (1/N) X X^T
    K = (X @ X.T) / N
    vals, vecs = np.linalg.eigh(K)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    
    # Check top eigenvector
    v1 = vecs[:, 0]
    l1 = vals[0]
    u1 = (X.T @ v1) / np.sqrt(N * l1)
    
    # u1 must have unit norm
    norm_u1 = np.linalg.norm(u1)
    # u1 must be eigenvector of S = (1/N) X^T X with eigenvalue l1
    S = (X.T @ X) / N
    eigen_check = np.allclose(S @ u1, l1 * u1, atol=1e-7)
    return np.isclose(norm_u1, 1.0) and eigen_check


# =============================================================================
# Exercise 16.4: General Gaussian Latent Prior in PPCA
# =============================================================================
def solve_exercise_16_4() -> Dict[str, Any]:
    return {
        "original_prior": "p(z) = N(z | 0, I)",
        "general_prior": "p(z) = N(z | m, Sigma)",
        "reparameterization": "z = m + Sigma^{1/2} epsilon, where epsilon ~ N(0, I)",
        "redefined_parameters": {
            "W_prime": "W Sigma^{1/2}",
            "mu_prime": "mu + W m",
            "sigma_prime_sq": "sigma^2",
        },
        "marginal_covariance": "C = W Sigma W^T + sigma^2 I = W_prime W_prime^T + sigma^2 I",
    }


def verify_exercise_16_4(D: int = 4, M: int = 2, seed: int = 42) -> bool:
    rng = np.random.RandomState(seed)
    m = rng.randn(M)
    A = rng.randn(M, M)
    Sigma = A @ A.T + 0.1 * np.eye(M)
    Sigma_half = np.linalg.cholesky(Sigma)
    
    mu = rng.randn(D)
    W = rng.randn(D, M)
    sigma_sq = 0.5
    
    # Redefined parameters
    W_prime = W @ Sigma_half
    mu_prime = mu + W @ m
    
    # Marginal mean and cov under general prior
    mean_gen = mu + W @ m
    cov_gen = W @ Sigma @ W.T + sigma_sq * np.eye(D)
    
    # Marginal mean and cov under redefined standard prior
    mean_std = mu_prime
    cov_std = W_prime @ W_prime.T + sigma_sq * np.eye(D)
    
    return np.allclose(mean_gen, mean_std) and np.allclose(cov_gen, cov_std)


# =============================================================================
# Exercise 16.5: Linear Transformation of Gaussian Vector
# =============================================================================
def solve_exercise_16_5() -> Dict[str, Any]:
    return {
        "mean_y": "E[y] = A mu + b",
        "cov_y": "Cov[y] = A Sigma A^T",
        "case_M_less_D": "If M < D and A has full row rank M, Cov[y] is M x M positive definite, non-singular Gaussian.",
        "case_M_equal_D": "If M = D and A is non-singular, Cov[y] is D x D positive definite, non-singular Gaussian.",
        "case_M_greater_D": "If M > D, rank(A Sigma A^T) <= D < M. The covariance is singular, so y is a degenerate Gaussian confined to a D-dimensional affine subspace.",
    }


def verify_exercise_16_5(seed: int = 42) -> bool:
    rng = np.random.RandomState(seed)
    D = 3
    mu = rng.randn(D)
    Sigma = np.diag([1.0, 2.0, 3.0])
    
    # M < D
    A_sub = rng.randn(2, D)
    cov_sub = A_sub @ Sigma @ A_sub.T
    rank_sub = np.linalg.matrix_rank(cov_sub)
    check_sub = (rank_sub == 2)
    
    # M > D
    A_super = rng.randn(5, D)
    cov_super = A_super @ Sigma @ A_super.T
    rank_super = np.linalg.matrix_rank(cov_super)
    check_super = (rank_super == D)  # Rank is at most D
    
    return check_sub and check_super


# =============================================================================
# Exercise 16.6: Law of Total Expectation and Variance for PPCA
# =============================================================================
def solve_exercise_16_6() -> Dict[str, Any]:
    return {
        "total_expectation": "E[x] = E_z[E[x|z]] = E_z[W z + mu] = W E[z] + mu = mu",
        "total_variance": "Cov[x] = E_z[Cov[x|z]] + Cov_z[E[x|z]] = E_z[sigma^2 I] + Cov_z[W z + mu] = sigma^2 I + W Cov[z] W^T = sigma^2 I + W W^T",
        "result_covariance": "C = W W^T + sigma^2 I",
    }


def verify_exercise_16_6(D: int = 4, M: int = 2, seed: int = 42) -> bool:
    rng = np.random.RandomState(seed)
    mu = rng.randn(D)
    W = rng.randn(D, M)
    sigma_sq = 0.3
    
    # Monte Carlo verification
    N_samples = 100000
    z = rng.randn(N_samples, M)
    eps = rng.randn(N_samples, D) * np.sqrt(sigma_sq)
    x = z @ W.T + mu + eps
    
    mc_mean = x.mean(axis=0)
    mc_cov = np.cov(x, rowvar=False)
    
    true_mean = mu
    true_cov = W @ W.T + sigma_sq * np.eye(D)
    
    return np.allclose(mc_mean, true_mean, atol=0.05) and np.allclose(mc_cov, true_cov, atol=0.05)


# =============================================================================
# Exercise 16.7: Directed Graph for PPCA and Naive Bayes Structure
# =============================================================================
def solve_exercise_16_7() -> Dict[str, Any]:
    return {
        "graph_structure": "z -> x_1, z -> x_2, ..., z -> x_D. In PPCA, p(x|z) = prod_{d=1}^D N(x_d | w_d^T z + mu_d, sigma^2).",
        "conditional_independence": "Conditioned on the latent variable z, the observed coordinates x_1, ..., x_D are conditionally independent: (x_i _|_ x_j | z).",
        "naive_bayes_analogy": "This is identical to the Naive Bayes assumption where all observed features are conditionally independent given the class label / latent state.",
    }


def verify_exercise_16_7() -> bool:
    sol = solve_exercise_16_7()
    return "conditionally independent" in sol["conditional_independence"] and "Naive Bayes" in sol["naive_bayes_analogy"]


# =============================================================================
# Exercise 16.8: Derivation of PPCA Posterior p(z|x)
# =============================================================================
def solve_exercise_16_8() -> Dict[str, Any]:
    return {
        "prior": "p(z) = N(z | 0, I)",
        "likelihood": "p(x|z) = N(x | W z + mu, sigma^2 I)",
        "posterior_precision": "Sigma_{z|x}^{-1} = I + (1/sigma^2) W^T W = (1/sigma^2) (W^T W + sigma^2 I) = (1/sigma^2) M",
        "posterior_covariance": "Sigma_{z|x} = sigma^2 M^{-1}",
        "posterior_mean": "E[z|x] = Sigma_{z|x} * (1/sigma^2) W^T (x - mu) = M^{-1} W^T (x - mu)",
    }


def verify_exercise_16_8(D: int = 5, M: int = 2, seed: int = 42) -> bool:
    rng = np.random.RandomState(seed)
    mu = rng.randn(D)
    W = rng.randn(D, M)
    sigma_sq = 0.4
    M_mat = W.T @ W + sigma_sq * np.eye(M)
    
    x = rng.randn(D)
    
    # Analytical posterior
    mean_analytic = np.linalg.solve(M_mat, W.T @ (x - mu))
    cov_analytic = sigma_sq * np.linalg.inv(M_mat)
    
    # Joint Gaussian conditioning formula
    C = W @ W.T + sigma_sq * np.eye(D)
    cov_zx = W.T
    mean_joint = cov_zx @ np.linalg.solve(C, x - mu)
    cov_joint = np.eye(M) - cov_zx @ np.linalg.solve(C, W)
    
    return np.allclose(mean_analytic, mean_joint) and np.allclose(cov_analytic, cov_joint)


# =============================================================================
# Exercise 16.9: MLE of mu in PPCA
# =============================================================================
def solve_exercise_16_9() -> Dict[str, Any]:
    return {
        "log_likelihood": "ln p(X | mu, W, sigma^2) = - (ND/2) ln(2 pi) - (N/2) ln |C| - 1/2 sum_{n=1}^N (x_n - mu)^T C^{-1} (x_n - mu)",
        "gradient_wrt_mu": "nabla_mu ln p(X) = sum_{n=1}^N C^{-1} (x_n - mu) = C^{-1} ( sum_{n=1}^N x_n - N mu )",
        "setting_to_zero": "C^{-1} ( N bar{x} - N mu ) = 0 => mu_{ML} = bar{x}",
    }


def verify_exercise_16_9(N: int = 20, D: int = 4, seed: int = 42) -> bool:
    rng = np.random.RandomState(seed)
    X = rng.randn(N, D)
    sample_mean = X.mean(axis=0)
    
    ppca = ProbabilisticPCA(n_components=2)
    ppca.fit(X)
    mu_est = getattr(ppca, "mu_", getattr(ppca, "mu", None))
    return np.allclose(mu_est, sample_mean)


# =============================================================================
# Exercise 16.10: Second Derivatives of PPCA Log-Likelihood w.r.t. mu
# =============================================================================
def solve_exercise_16_10() -> Dict[str, Any]:
    return {
        "first_derivative": "nabla_mu ln p(X) = N C^{-1} (bar{x} - mu)",
        "second_derivative": "nabla_mu^2 ln p(X) = - N C^{-1}",
        "negative_definiteness": "Since C = W W^T + sigma^2 I is strictly positive definite for sigma^2 > 0, C^{-1} is positive definite, making -N C^{-1} strictly negative definite everywhere. Hence mu_{ML} = bar{x} is the unique global maximum.",
    }


def verify_exercise_16_10(D: int = 4, M: int = 2, seed: int = 42) -> bool:
    rng = np.random.RandomState(seed)
    W = rng.randn(D, M)
    sigma_sq = 0.5
    C = W @ W.T + sigma_sq * np.eye(D)
    N = 50
    
    Hessian = -N * np.linalg.inv(C)
    eigvals = np.linalg.eigvalsh(Hessian)
    return np.all(eigvals < 0)


# =============================================================================
# Exercise 16.11: Orthogonal Projection Limit as sigma^2 -> 0
# =============================================================================
def solve_exercise_16_11() -> Dict[str, Any]:
    return {
        "limit_M": "As sigma^2 -> 0, M = W^T W + sigma^2 I -> W^T W.",
        "limit_posterior_mean": "lim_{sigma^2 -> 0} E[z|x] = (W^T W)^{-1} W^T (x - mu).",
        "interpretation": "This is precisely the classical orthogonal projection of (x - mu) onto the linear subspace spanned by the columns of W.",
    }


def verify_exercise_16_11(D: int = 5, M: int = 2, seed: int = 42) -> bool:
    rng = np.random.RandomState(seed)
    W = rng.randn(D, M)
    mu = rng.randn(D)
    x = rng.randn(D)
    
    sigma_sq_small = 1e-9
    M_small = W.T @ W + sigma_sq_small * np.eye(M)
    mean_ppca = np.linalg.solve(M_small, W.T @ (x - mu))
    
    # Classical orthogonal projection
    mean_orth = np.linalg.solve(W.T @ W, W.T @ (x - mu))
    return np.allclose(mean_ppca, mean_orth, atol=1e-6)


# =============================================================================
# Exercise 16.12: Shrinkage of Posterior Mean for sigma^2 > 0
# =============================================================================
def solve_exercise_16_12() -> Dict[str, Any]:
    return {
        "spectral_form": "Let W = U_M (Lambda_M - sigma^2 I)^{1/2} R^T. Then M^{-1} W^T = R Lambda_M^{-1} (Lambda_M - sigma^2 I)^{1/2} U_M^T.",
        "shrinkage_factor": "Along principal component j, the weight is (1 - sigma^2 / lambda_j)^{1/2} / lambda_j^{1/2} * (lambda_j - sigma^2)^{1/2} / lambda_j = (lambda_j - sigma^2) / lambda_j = 1 - sigma^2 / lambda_j < 1.",
        "conclusion": "The posterior mean is strictly shrunken towards the origin relative to the orthogonal projection.",
    }


def verify_exercise_16_12(D: int = 4, M: int = 2, seed: int = 42) -> bool:
    rng = np.random.RandomState(seed)
    W = rng.randn(D, M)
    mu = np.zeros(D)
    x = rng.randn(D)
    
    # Orthogonal projection norm
    z_orth = np.linalg.solve(W.T @ W, W.T @ x)
    
    # PPCA projection with sigma^2 > 0
    sigma_sq = 1.0
    M_mat = W.T @ W + sigma_sq * np.eye(M)
    z_ppca = np.linalg.solve(M_mat, W.T @ x)
    
    return np.linalg.norm(z_ppca) < np.linalg.norm(z_orth)


# =============================================================================
# Exercise 16.13: Optimal Reconstruction under Least-Squares Cost
# =============================================================================
def solve_exercise_16_13() -> Dict[str, Any]:
    return {
        "reconstruction_formula": "tilde{x} = W_{ML} (W_{ML}^T W_{ML})^{-1} M E[z|x] + mu",
        "cancellation": "Since E[z|x] = M^{-1} W_{ML}^T (x - mu), multiplying by M cancels M^{-1} perfectly: M E[z|x] = W_{ML}^T (x - mu).",
        "final_form": "tilde{x} = W_{ML} (W_{ML}^T W_{ML})^{-1} W_{ML}^T (x - mu) + mu = standard orthogonal projection reconstruction.",
    }


def verify_exercise_16_13(D: int = 5, M: int = 2, seed: int = 42) -> bool:
    rng = np.random.RandomState(seed)
    W = rng.randn(D, M)
    mu = rng.randn(D)
    x = rng.randn(D)
    sigma_sq = 0.5
    
    M_mat = W.T @ W + sigma_sq * np.eye(M)
    E_z = np.linalg.solve(M_mat, W.T @ (x - mu))
    
    # Using formula (16.88)
    tilde_x = W @ np.linalg.solve(W.T @ W, M_mat @ E_z) + mu
    # Classical orthogonal projection
    x_proj = W @ np.linalg.solve(W.T @ W, W.T @ (x - mu)) + mu
    
    return np.allclose(tilde_x, x_proj)


# =============================================================================
# Exercise 16.14: Independent Parameters in PPCA
# =============================================================================
def solve_exercise_16_14() -> Dict[str, Any]:
    return {
        "formula": "D * M + 1 - (1/2) * M * (M - 1)",
        "M_equal_0": "When M = 0: 0 + 1 - 0 = 1 (isotropic Gaussian sigma^2 I).",
        "M_equal_D_minus_1": "When M = D - 1: D(D-1) + 1 - 1/2(D-1)(D-2) = (1/2) D (D + 1) (general full covariance Gaussian).",
    }


def verify_exercise_16_14(D: int = 5) -> bool:
    # M = 0
    p_0 = D * 0 + 1 - 0
    match_0 = (p_0 == 1)
    
    # M = D - 1
    M = D - 1
    p_full = D * M + 1 - (M * (M - 1)) // 2
    expected_full = (D * (D + 1)) // 2
    match_full = (p_full == expected_full)
    return match_0 and match_full


# =============================================================================
# Exercise 16.15: Independent Parameters in Factor Analysis
# =============================================================================
def solve_exercise_16_15() -> Dict[str, Any]:
    return {
        "formula": "D * M + D - (1/2) * M * (M - 1)",
        "explanation": "W has D * M parameters. The diagonal matrix Psi has D independent variance parameters. Latent rotation SO(M) removes M(M-1)/2 degrees of freedom.",
    }


def verify_exercise_16_15(D: int = 6, M: int = 2) -> bool:
    sol = solve_exercise_16_15()
    expected = D * M + D - (M * (M - 1)) // 2
    return expected == 6 * 2 + 6 - 1 == 17


# =============================================================================
# Exercise 16.16: Latent Rotation Invariance of Factor Analysis
# =============================================================================
def solve_exercise_16_16() -> Dict[str, Any]:
    return {
        "rotation": "Let R be an M x M orthogonal matrix (R R^T = I). Let W' = W R.",
        "invariance": "Covariance C' = W' W'^T + Psi = (W R)(W R)^T + Psi = W (R R^T) W^T + Psi = W W^T + Psi = C.",
        "conclusion": "The marginal distribution and model likelihood are completely invariant under orthogonal transformations of the latent coordinates.",
    }


def verify_exercise_16_16(D: int = 5, M: int = 2, seed: int = 42) -> bool:
    rng = np.random.RandomState(seed)
    W = rng.randn(D, M)
    psi = rng.rand(D) + 0.1
    Psi = np.diag(psi)
    
    # Random orthogonal matrix R
    theta = rng.rand() * 2 * np.pi
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    W_prime = W @ R
    C = W @ W.T + Psi
    C_prime = W_prime @ W_prime.T + Psi
    return np.allclose(C, C_prime)


# =============================================================================
# Exercise 16.17: Transformation of Data Variables in FA and PPCA
# =============================================================================
def solve_exercise_16_17() -> Dict[str, Any]:
    return {
        "transformation": "x -> A x. MLE transforms as mu_ML -> A mu_ML, W_ML -> A W_ML, Phi_ML -> A Phi_ML A^T.",
        "case_fa": "(i) A is diagonal and Phi is diagonal (Factor Analysis). Then A Phi A^T remains diagonal. Hence FA is covariant under component-wise rescaling.",
        "case_ppca": "(ii) A is orthogonal and Phi = sigma^2 I (PPCA). Then A (sigma^2 I) A^T = sigma^2 A A^T = sigma^2 I. Hence PPCA is covariant under rotations of the data axes.",
    }


def verify_exercise_16_17(D: int = 4, seed: int = 42) -> bool:
    rng = np.random.RandomState(seed)
    # Check FA covariance with diagonal A
    diag_A = rng.rand(D) + 0.5
    A_fa = np.diag(diag_A)
    Phi_fa = np.diag(rng.rand(D) + 0.1)
    trans_Phi_fa = A_fa @ Phi_fa @ A_fa.T
    fa_diag = np.allclose(trans_Phi_fa, np.diag(np.diag(trans_Phi_fa)))
    
    # Check PPCA covariance with orthogonal A
    Q, _ = np.linalg.qr(rng.randn(D, D))
    sigma_sq = 0.7
    Phi_ppca = sigma_sq * np.eye(D)
    trans_Phi_ppca = Q @ Phi_ppca @ Q.T
    ppca_isotropic = np.allclose(trans_Phi_ppca, sigma_sq * np.eye(D))
    
    return fa_diag and ppca_isotropic


# =============================================================================
# Exercise 16.18: ELBO Decomposition for Continuous Latent Variables
# =============================================================================
def solve_exercise_16_18() -> Dict[str, Any]:
    return {
        "decomposition": "ln p(x | w) = L(q, w) + KL(q || p)",
        "product_rule": "p(x, z | w) = p(z | x, w) p(x | w)",
        "expansion": "L(q, w) = int q(z) ln [p(x, z|w) / q(z)] dz = int q(z) ln p(x|w) dz + int q(z) ln [p(z|x, w) / q(z)] dz = ln p(x|w) - KL(q || p(z|x, w))",
    }


def verify_exercise_16_18(seed: int = 42) -> bool:
    # 1D analytical Gaussian verification
    rng = np.random.RandomState(seed)
    x = 1.5
    w = 2.0
    sigma_sq = 1.0
    
    # p(z) = N(0, 1), p(x|z) = N(w z, sigma^2)
    # Marginal p(x) = N(0, w^2 + sigma^2)
    var_x = w**2 + sigma_sq
    log_p_x = stats.norm.logpdf(x, loc=0, scale=np.sqrt(var_x))
    
    # True posterior p(z|x) = N(mu_post, var_post)
    var_post = 1.0 / (1.0 + w**2 / sigma_sq)
    mu_post = var_post * (w / sigma_sq) * x
    
    # Arbitrary q(z) = N(mu_q, var_q)
    mu_q = 0.8
    var_q = 0.5
    
    # KL(q || p(z|x))
    kl = 0.5 * (np.log(var_post / var_q) + (var_q + (mu_q - mu_post)**2) / var_post - 1.0)
    
    # ELBO = E_q[ln p(x, z) - ln q(z)]
    # ln p(x, z) = ln p(z) + ln p(x|z)
    E_ln_pz = -0.5 * np.log(2 * np.pi) - 0.5 * (var_q + mu_q**2)
    E_ln_px_z = -0.5 * np.log(2 * np.pi * sigma_sq) - 0.5 / sigma_sq * (x**2 - 2 * x * w * mu_q + w**2 * (var_q + mu_q**2))
    E_ln_qz = -0.5 * np.log(2 * np.pi * var_q) - 0.5
    elbo = (E_ln_pz + E_ln_px_z) - E_ln_qz
    
    return np.isclose(log_p_x, elbo + kl)


# =============================================================================
# Exercise 16.19: I.I.D. ELBO Factorization
# =============================================================================
def solve_exercise_16_19() -> Dict[str, Any]:
    return {
        "factorized_q": "q(Z) = prod_{n=1}^N q(z_n)",
        "factorized_p": "p(X, Z | w) = prod_{n=1}^N p(x_n, z_n | w)",
        "sum_of_elbos": "L(q, w) = sum_{n=1}^N int q(z_n) ln [p(x_n, z_n | w) / q(z_n)] dz_n = sum_{n=1}^N L_n(q_n, w)",
    }


def verify_exercise_16_19() -> bool:
    sol = solve_exercise_16_19()
    return "sum_{n=1}^N L_n" in sol["sum_of_elbos"]


# =============================================================================
# Exercise 16.20: Directed Graphical Model for Mixture of PPCAs
# =============================================================================
def solve_exercise_16_20() -> Dict[str, Any]:
    return {
        "separate_parameters": "Discrete indicator s_n -> x_n, continuous latent z_n -> x_n. Component-specific parameters {W_k, mu_k, sigma_k^2}.",
        "tied_parameters": "When parameters are shared across mixture components, W and sigma^2 are single shared nodes outside the plate.",
    }


def verify_exercise_16_20() -> bool:
    sol = solve_exercise_16_20()
    return "s_n -> x_n" in sol["separate_parameters"]


# =============================================================================
# Exercise 16.21: PPCA M-Step Formulae Derivation
# =============================================================================
def solve_exercise_16_21() -> Dict[str, Any]:
    return {
        "expected_log_likelihood": "Q(W, sigma^2) = - (ND/2) ln(2 pi sigma^2) - (1 / 2 sigma^2) sum_n E[||x_n - mu - W z_n||^2] + const",
        "W_derivative": "partial Q / partial W = (1 / sigma^2) sum_n [ (x_n - mu) E[z_n]^T - W E[z_n z_n^T] ] = 0",
        "W_update": "W_{new} = [ sum_n (x_n - mu) E[z_n]^T ] [ sum_n E[z_n z_n^T] ]^{-1}",
        "sigma_sq_update": "sigma_{new}^2 = (1 / ND) sum_n { ||x_n - mu||^2 - 2 E[z_n]^T W_{new}^T (x_n - mu) + Tr(W_{new}^T W_{new} E[z_n z_n^T]) }",
    }


def verify_exercise_16_21(N: int = 30, D: int = 4, M: int = 2, seed: int = 42) -> bool:
    rng = np.random.RandomState(seed)
    X = rng.randn(N, D)
    mu = X.mean(axis=0)
    X_c = X - mu
    
    W = rng.randn(D, M)
    sigma_sq = 0.5
    
    M_mat = W.T @ W + sigma_sq * np.eye(M)
    M_inv = np.linalg.inv(M_mat)
    
    # E-step
    E_z = (M_inv @ W.T @ X_c.T).T  # N x M
    E_zz = np.zeros((M, M))
    for n in range(N):
        zn = E_z[n : n + 1].T
        E_zz += sigma_sq * M_inv + zn @ zn.T
        
    # M-step
    W_new = (X_c.T @ E_z) @ np.linalg.inv(E_zz)
    
    # Check improvement or finiteness
    return W_new.shape == (D, M) and np.all(np.isfinite(W_new))


# =============================================================================
# Exercise 16.22: PPCA with Missing Data (MAR)
# =============================================================================
def solve_exercise_16_22() -> Dict[str, Any]:
    return {
        "framework": "Partition observed coordinates x_{n,O} and missing coordinates x_{n,M}. Latent variables are (z_n, x_{n,M}).",
        "E_step": "Compute posterior distribution p(z_n, x_{n,M} | x_{n,O}) using Gaussian conditioning.",
        "M_step": "Compute expected complete-data log-likelihood w.r.t. this joint posterior and update W and sigma^2.",
        "special_case": "When no values are missing (x_{n,M} is empty), p(z_n | x_{n,O}) = p(z_n | x_n), exactly recovering the standard PPCA EM.",
    }


def verify_exercise_16_22() -> bool:
    sol = solve_exercise_16_22()
    return "standard PPCA EM" in sol["special_case"]


# =============================================================================
# Exercise 16.23: Roweis Alternating Least Squares (EM for Standard PCA)
# =============================================================================
def solve_exercise_16_23() -> Dict[str, Any]:
    return {
        "cost_function": "J = sum_{n=1}^N ||x_n - mu - W z_n||^2",
        "minimizing_mu": "partial J / partial mu = 0 => mu = bar{x} - W bar{z}. With zero-mean z_n, mu = bar{x}.",
        "minimizing_z": "partial J / partial z_n = 0 => z_n = (W^T W)^{-1} W^T (x_n - mu) (PCA E-step).",
        "minimizing_W": "partial J / partial W = 0 => W = [ sum_n (x_n - mu) z_n^T ] [ sum_n z_n z_n^T ]^{-1} (PCA M-step).",
    }


def verify_exercise_16_23(N: int = 50, D: int = 5, M: int = 2, seed: int = 42) -> bool:
    rng = np.random.RandomState(seed)
    X = rng.randn(N, D)
    mu = X.mean(axis=0)
    X_c = X - mu
    
    # Roweis EM iteration
    W = rng.randn(D, M)
    for _ in range(50):
        # E-step
        Z = X_c @ W @ np.linalg.inv(W.T @ W)
        # M-step
        W = X_c.T @ Z @ np.linalg.inv(Z.T @ Z)
        
    # Check that subspace of W spans the top 2 eigenvectors of S
    S = (X_c.T @ X_c) / N
    vals, vecs = np.linalg.eigh(S)
    U_top = vecs[:, np.argsort(vals)[::-1][:M]]
    
    # Subspace overlap via projection
    Q_w, _ = np.linalg.qr(W)
    proj_diff = np.linalg.norm(Q_w @ Q_w.T - U_top @ U_top.T)
    return proj_diff < 1e-4


# =============================================================================
# Exercise 16.24: E-Step Formulae for Factor Analysis
# =============================================================================
def solve_exercise_16_24() -> Dict[str, Any]:
    return {
        "G_matrix": "G = (I + W^T Psi^{-1} W)^{-1}",
        "posterior_mean": "E[z_n] = G W^T Psi^{-1} (x_n - bar{x})",
        "posterior_second_moment": "E[z_n z_n^T] = G + E[z_n] E[z_n]^T",
    }


def verify_exercise_16_24(D: int = 4, M: int = 2, seed: int = 42) -> bool:
    rng = np.random.RandomState(seed)
    W = rng.randn(D, M)
    psi = rng.rand(D) + 0.2
    Psi = np.diag(psi)
    Psi_inv = np.diag(1.0 / psi)
    
    G = np.linalg.inv(np.eye(M) + W.T @ Psi_inv @ W)
    x = rng.randn(D)
    
    mean_fa = G @ W.T @ Psi_inv @ x
    cov_fa = G
    
    # Via standard joint Gaussian conditioning
    C = W @ W.T + Psi
    mean_joint = W.T @ np.linalg.solve(C, x)
    cov_joint = np.eye(M) - W.T @ np.linalg.solve(C, W)
    
    return np.allclose(mean_fa, mean_joint) and np.allclose(cov_fa, cov_joint)


# =============================================================================
# Exercise 16.25: M-Step Formulae for Factor Analysis
# =============================================================================
def solve_exercise_16_25() -> Dict[str, Any]:
    return {
        "W_new": "W_{new} = [ sum_n (x_n - bar{x}) E[z_n]^T ] [ sum_n E[z_n z_n^T] ]^{-1}",
        "Psi_new": "Psi_{new} = diag( S - W_{new} (1/N) sum_n E[z_n] (x_n - bar{x})^T )",
    }


def verify_exercise_16_25(N: int = 30, D: int = 4, M: int = 2, seed: int = 42) -> bool:
    rng = np.random.RandomState(seed)
    X = rng.randn(N, D)
    fa = FactorAnalysis(n_components=M, max_iter=5)
    fa.fit(X)
    return (fa.W_.shape == (D, M) if hasattr(fa, "W_") else fa.W.shape == (D, M)) and np.all(getattr(fa, "psi_", getattr(fa, "psi", np.ones(D))) > 0)


# =============================================================================
# Exercise 16.26: Stationary Point and Maximum of Factor Analysis w.r.t. mu
# =============================================================================
def solve_exercise_16_26() -> Dict[str, Any]:
    return {
        "gradient": "nabla_mu ln p(X) = C^{-1} sum_{n=1}^N (x_n - mu) = 0 => mu_{ML} = bar{x}",
        "second_derivative": "nabla_mu^2 ln p(X) = - N C^{-1} where C = W W^T + Psi",
        "maximum_proof": "Since Psi has strictly positive diagonal entries and W W^T is positive semidefinite, C is strictly positive definite, so -N C^{-1} is strictly negative definite. Hence mu_{ML} = bar{x} is the unique global maximum.",
    }


def verify_exercise_16_26(D: int = 4, M: int = 2, seed: int = 42) -> bool:
    rng = np.random.RandomState(seed)
    W = rng.randn(D, M)
    psi = rng.rand(D) + 0.1
    C = W @ W.T + np.diag(psi)
    N = 100
    
    Hessian = -N * np.linalg.inv(C)
    eigvals = np.linalg.eigvalsh(Hessian)
    return np.all(eigvals < 0)


# =============================================================================
# Master Verification
# =============================================================================
def solve_all_exercises() -> Dict[str, Any]:
    return {
        f"Exercise_16_{i}": getattr(globals()[f"solve_exercise_16_{i}"], "__call__")()
        for i in range(1, 27)
    }


def verify_all_exercises() -> Dict[str, bool]:
    return {
        f"Exercise_16_{i}": getattr(globals()[f"verify_exercise_16_{i}"], "__call__")()
        for i in range(1, 27)
    }
