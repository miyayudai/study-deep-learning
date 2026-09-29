"""
Unit tests for Appendix A: Linear Algebra (付録A: 線形代数)
Validates all equations (A.1) - (A.50) from Bishop & Bishop (2024).
"""

import os
from pathlib import Path
import numpy as np
import pytest

from common.linear_algebra import (
    verify_matrix_transpose_product,
    verify_matrix_inverse_product,
    low_rank_update_identity,
    push_through_identity,
    woodbury_inversion,
    verify_trace_cyclic,
    weinstein_aronszajn_determinant,
    rank1_determinant_lemma,
    verify_matrix_inverse_derivative,
    verify_log_det_scalar_derivative,
    verify_matrix_derivatives_identities,
    SymmetricMatrixSpectralAnalysis,
    generate_all_appendix_a_figures,
)


class TestMatrixIdentities:
    """Tests for Section A.1: Matrix Identities (Eqs. A.1 - A.7)."""

    def test_transpose_and_inverse_identities(self):
        np.random.seed(42)
        A = np.random.randn(4, 4)
        B = np.random.randn(4, 4)
        
        # Eq. (A.1): (AB)^T = B^T A^T
        assert verify_matrix_transpose_product(A, B)
        
        # Eq. (A.3), (A.4): (AB)^{-1} = B^{-1} A^{-1}, (A^T)^{-1} = (A^{-1})^T
        assert verify_matrix_inverse_product(A, B)

    def test_low_rank_update_identity(self):
        # Eq. (A.5): (P^{-1} + B^T R^{-1} B)^{-1} B^T R^{-1} = P B^T (B P B^T + R)^{-1}
        np.random.seed(101)
        N, M = 8, 3
        P_raw = np.random.randn(N, N)
        P = P_raw @ P_raw.T + np.eye(N)  # N x N positive definite
        R_raw = np.random.randn(M, M)
        R = R_raw @ R_raw.T + np.eye(M)  # M x M positive definite
        B = np.random.randn(M, N)
        
        res = low_rank_update_identity(P, B, R)
        assert res["is_equivalent"]
        assert res["max_diff"] < 1e-10

    def test_push_through_identity(self):
        # Eq. (A.6): (I + AB)^{-1} A = A (I + BA)^{-1}
        np.random.seed(202)
        N, M = 5, 3
        A = np.random.randn(N, M)
        B = np.random.randn(M, N)
        
        res = push_through_identity(A, B)
        assert res["is_equivalent"]
        assert res["max_diff"] < 1e-10

    def test_woodbury_identity(self):
        # Eq. (A.7): (A + B D^{-1} C)^{-1} = A^{-1} - A^{-1} B (D + C A^{-1} B)^{-1} C A^{-1}
        np.random.seed(303)
        N, M = 6, 2
        A_raw = np.random.randn(N, N)
        A = A_raw @ A_raw.T + np.eye(N)
        D_raw = np.random.randn(M, M)
        D = D_raw @ D_raw.T + np.eye(M)
        B = np.random.randn(N, M)
        C = np.random.randn(M, N)
        
        res = woodbury_inversion(A, B, D, C)
        assert res["is_equivalent"]
        assert res["max_diff"] < 1e-10


class TestTracesAndDeterminants:
    """Tests for Section A.2: Traces and Determinants (Eqs. A.8 - A.15)."""

    def test_cyclic_trace(self):
        # Eqs. (A.8), (A.9): Tr(AB) = Tr(BA), Tr(ABC) = Tr(CAB) = Tr(BCA)
        np.random.seed(404)
        A = np.random.randn(4, 5)
        B = np.random.randn(5, 4)
        assert verify_trace_cyclic([A, B])["is_cyclic_invariant"]
        
        A3 = np.random.randn(4, 5)
        B3 = np.random.randn(5, 6)
        C3 = np.random.randn(6, 4)
        assert verify_trace_cyclic([A3, B3, C3])["is_cyclic_invariant"]

    def test_determinant_multiplication_and_inverse(self):
        # Eqs. (A.12), (A.13): |AB| = |A||B|, |A^{-1}| = 1 / |A|
        np.random.seed(505)
        A = np.random.randn(4, 4)
        B = np.random.randn(4, 4)
        
        det_A = np.linalg.det(A)
        det_B = np.linalg.det(B)
        det_AB = np.linalg.det(A @ B)
        assert np.isclose(det_AB, det_A * det_B)
        
        det_invA = np.linalg.det(np.linalg.inv(A))
        assert np.isclose(det_invA, 1.0 / det_A)

    def test_weinstein_aronszajn_identity(self):
        # Eq. (A.14): |I_N + A B^T| = |I_M + A^T B|
        np.random.seed(606)
        N, M = 7, 3
        A = np.random.randn(N, M)
        B = np.random.randn(N, M)
        res = weinstein_aronszajn_determinant(A, B)
        assert res["is_equivalent"]
        assert res["diff"] < 1e-10

    def test_rank1_determinant_lemma(self):
        # Eq. (A.15): |I_N + a b^T| = 1 + a^T b
        np.random.seed(707)
        a = np.random.randn(8)
        b = np.random.randn(8)
        res = rank1_determinant_lemma(a, b)
        assert res["is_equivalent"]
        assert res["diff"] < 1e-10


class TestMatrixDerivatives:
    """Tests for Section A.3: Matrix Derivatives (Eqs. A.16 - A.28)."""

    def test_linear_form_derivative(self):
        # Eq. (A.19): d/dx (x^T a) = a
        np.random.seed(808)
        a = np.random.randn(5)
        x = np.random.randn(5)
        eps = 1e-6
        num_grad = np.zeros(5)
        for i in range(5):
            x_p = x.copy(); x_p[i] += eps
            x_m = x.copy(); x_m[i] -= eps
            num_grad[i] = (np.dot(x_p, a) - np.dot(x_m, a)) / (2.0 * eps)
        assert np.allclose(num_grad, a, atol=1e-5)

    def test_matrix_inverse_derivative(self):
        # Eq. (A.21): d(A^{-1})/dx = - A^{-1} (dA/dx) A^{-1}
        def A_func(x):
            return np.array([
                [np.cos(x) + 2.0, x, 0.5],
                [x, np.sin(x) + 2.0, -0.3],
                [0.5, -0.3, 1.5 + x**2]
            ])
            
        def dA_dx_func(x):
            return np.array([
                [-np.sin(x), 1.0, 0.0],
                [1.0, np.cos(x), 0.0],
                [0.0, 0.0, 2.0 * x]
            ])
            
        res = verify_matrix_inverse_derivative(A_func, dA_dx_func, x0=0.7)
        assert res["is_close"]
        assert res["max_diff"] < 1e-5

    def test_log_det_scalar_derivative(self):
        # Eq. (A.22): d/dx ln |A| = Tr( A^{-1} dA/dx )
        def A_func(x):
            return np.array([
                [2.0 + x, 0.4, 0.1],
                [0.4, 3.0 + np.exp(0.5 * x), 0.2],
                [0.1, 0.2, 1.5 + np.cos(x)]
            ])
            
        def dA_dx_func(x):
            return np.array([
                [1.0, 0.0, 0.0],
                [0.0, 0.5 * np.exp(0.5 * x), 0.0],
                [0.0, 0.0, -np.sin(x)]
            ])
            
        res = verify_log_det_scalar_derivative(A_func, dA_dx_func, x0=0.5)
        assert res["is_close"]
        assert res["diff"] < 1e-5

    def test_matrix_derivatives_identities(self):
        # Eqs. (A.24) - (A.28)
        np.random.seed(909)
        A = np.random.randn(3, 3) + 2.0 * np.eye(3)
        B = np.random.randn(3, 3)
        res = verify_matrix_derivatives_identities(A, B)
        assert res["all_passed"]


class TestEigenvectorsAndSpectralTheory:
    """Tests for Section A.4: Eigenvectors and Spectral Theory (Eqs. A.29 - A.50)."""

    def test_spectral_decomposition_properties(self):
        # Create symmetric matrix
        np.random.seed(1111)
        M = 4
        X = np.random.randn(M, M)
        A = X @ X.T + 0.5 * np.eye(M)  # symmetric positive definite
        
        spectral = SymmetricMatrixSpectralAnalysis(A)
        
        # Orthonormality (Eq. A.33, A.37)
        ortho_res = spectral.verify_orthonormality()
        assert ortho_res["is_orthogonal"]
        
        # Diagonalization (Eq. A.42, A.43)
        diag_res = spectral.verify_diagonalization()
        assert diag_res["is_diagonalized"]
        
        # Dyadic expansion (Eq. A.45, A.46)
        A_recon, inv_A_recon = spectral.dyadic_expansion()
        assert np.allclose(A_recon, A, atol=1e-10)
        assert np.allclose(inv_A_recon, np.linalg.inv(A), atol=1e-10)
        
        # Trace and determinant identities (Eq. A.47, A.48)
        id_res = spectral.verify_determinant_and_trace_identities()
        assert id_res["is_valid"]
        
        # Definiteness and condition number (Eq. A.50)
        assert spectral.definiteness() == "positive_definite"
        cn = spectral.condition_number()
        assert cn >= 1.0

    def test_counterexample_matrix_positive_elements_indefinite(self):
        # Textbook Eq. (A.49):
        # Matrix [[1, 2], [3, 4]] has positive elements, but eigenvalues ~ 5.37 and -0.37!
        A_book = np.array([[1.0, 2.0], [3.0, 4.0]])
        eigs = np.linalg.eigvals(A_book)
        eigs_sorted = np.sort(eigs)[::-1]
        
        assert np.isclose(eigs_sorted[0], 5.37228132, atol=1e-2)
        assert np.isclose(eigs_sorted[1], -0.37228132, atol=1e-2)
        
        # Symmetrized quadratic form has one positive and one negative eigenvalue
        A_sym = 0.5 * (A_book + A_book.T)
        sym_analysis = SymmetricMatrixSpectralAnalysis(A_sym)
        assert sym_analysis.definiteness() == "indefinite"


class TestAppendixAFigures:
    """Test generation of all 4 Appendix A figures."""

    def test_figures_generation(self):
        saved = generate_all_appendix_a_figures(output_dir="appendix/result")
        assert len(saved) == 8  # 4 figures x 2 directories (appendix/result and result/)
        for path in saved:
            assert os.path.exists(path), f"Figure missing: {path}"
            assert os.path.getsize(path) > 1000, f"Figure empty: {path}"
