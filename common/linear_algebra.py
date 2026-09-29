"""
Bishop & Bishop (2024) Appendix A: Linear Algebra (付録A: 線形代数)
Comprehensive implementation of matrix identities, traces, determinants,
matrix derivatives, and spectral/eigenvector decomposition.

Equations:
- (A.1) - (A.7): Matrix identities (transpose, inverse, Woodbury, low-rank)
- (A.8) - (A.15): Traces and determinants (cyclic property, determinant product, Weinstein-Aronszajn)
- (A.16) - (A.28): Matrix derivatives (scalar, vector, matrix, trace, log-determinant)
- (A.29) - (A.50): Eigenvectors, spectral theorem, positive definiteness, condition number
"""

from typing import Tuple, List, Dict, Any, Optional
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from common.plot_utils import setup_style, save_plot


# ==============================================================================
# A.1 Matrix Identities (行列の恒等式)
# ==============================================================================

def verify_matrix_transpose_product(A: np.ndarray, B: np.ndarray) -> bool:
    """
    Verify (AB)^T = B^T A^T (Eq. A.1).
    """
    lhs = (A @ B).T
    rhs = B.T @ A.T
    return bool(np.allclose(lhs, rhs))


def verify_matrix_inverse_product(A: np.ndarray, B: np.ndarray) -> bool:
    """
    Verify (AB)^{-1} = B^{-1} A^{-1} (Eq. A.3) and (A^T)^{-1} = (A^{-1})^T (Eq. A.4).
    """
    inv_AB = np.linalg.inv(A @ B)
    rhs_prod = np.linalg.inv(B) @ np.linalg.inv(A)
    
    inv_AT = np.linalg.inv(A.T)
    rhs_trans = (np.linalg.inv(A)).T
    
    return bool(np.allclose(inv_AB, rhs_prod) and np.allclose(inv_AT, rhs_trans))


def low_rank_update_identity(P: np.ndarray, B: np.ndarray, R: np.ndarray) -> Dict[str, Any]:
    """
    Verify Equation (A.5):
    (P^{-1} + B^T R^{-1} B)^{-1} B^T R^{-1} = P B^T (B P B^T + R)^{-1}
    where P is N x N, R is M x M, B is M x N.
    When M << N, the RHS is computationally much cheaper than LHS.
    """
    inv_P = np.linalg.inv(P)
    inv_R = np.linalg.inv(R)
    
    # LHS: (P^{-1} + B^T R^{-1} B)^{-1} B^T R^{-1}
    lhs_bracket = np.linalg.inv(inv_P + B.T @ inv_R @ B)
    lhs = lhs_bracket @ B.T @ inv_R
    
    # RHS: P B^T (B P B^T + R)^{-1}
    rhs_bracket = np.linalg.inv(B @ P @ B.T + R)
    rhs = P @ B.T @ rhs_bracket
    
    diff = float(np.max(np.abs(lhs - rhs)))
    return {
        "lhs": lhs,
        "rhs": rhs,
        "max_diff": diff,
        "is_equivalent": diff < 1e-10
    }


def push_through_identity(A: np.ndarray, B: np.ndarray) -> Dict[str, Any]:
    """
    Verify Equation (A.6):
    (I + AB)^{-1} A = A (I + BA)^{-1}
    where A is N x M and B is M x N.
    """
    N, M = A.shape
    I_N = np.eye(N)
    I_M = np.eye(M)
    
    lhs = np.linalg.inv(I_N + A @ B) @ A
    rhs = A @ np.linalg.inv(I_M + B @ A)
    
    diff = float(np.max(np.abs(lhs - rhs)))
    return {
        "lhs": lhs,
        "rhs": rhs,
        "max_diff": diff,
        "is_equivalent": diff < 1e-10
    }


def woodbury_inversion(A: np.ndarray, B: np.ndarray, D: np.ndarray, C: np.ndarray) -> Dict[str, Any]:
    """
    Woodbury Matrix Identity (Eq. A.7):
    (A + B D^{-1} C)^{-1} = A^{-1} - A^{-1} B (D + C A^{-1} B)^{-1} C A^{-1}
    where A is N x N, B is N x M, D is M x M, C is M x N.
    """
    inv_A = np.linalg.inv(A)
    inv_D = np.linalg.inv(D)
    
    # Direct inverse of perturbed matrix (LHS)
    lhs = np.linalg.inv(A + B @ inv_D @ C)
    
    # Woodbury expansion (RHS)
    middle_inv = np.linalg.inv(D + C @ inv_A @ B)
    rhs = inv_A - inv_A @ B @ middle_inv @ C @ inv_A
    
    diff = float(np.max(np.abs(lhs - rhs)))
    return {
        "lhs": lhs,
        "rhs": rhs,
        "max_diff": diff,
        "is_equivalent": diff < 1e-10
    }


# ==============================================================================
# A.2 Traces and Determinants (トレースと行列式)
# ==============================================================================

def verify_trace_cyclic(matrices: List[np.ndarray]) -> Dict[str, Any]:
    """
    Verify cyclic property of trace (Eq. A.8, A.9):
    Tr(AB) = Tr(BA)
    Tr(ABC) = Tr(CAB) = Tr(BCA)
    """
    k = len(matrices)
    traces = []
    
    # Evaluate cyclic permutations
    for shift in range(k):
        perm_mats = matrices[shift:] + matrices[:shift]
        prod = perm_mats[0]
        for m in perm_mats[1:]:
            prod = prod @ m
        traces.append(float(np.trace(prod)))
        
    diffs = [abs(t - traces[0]) for t in traces]
    return {
        "traces": traces,
        "max_diff": float(max(diffs)),
        "is_cyclic_invariant": max(diffs) < 1e-10
    }


def weinstein_aronszajn_determinant(A: np.ndarray, B: np.ndarray) -> Dict[str, Any]:
    """
    Verify Weinstein-Aronszajn determinant identity (Eq. A.14):
    |I_N + A B^T| = |I_M + A^T B|
    where A and B are N x M matrices.
    """
    N, M = A.shape
    I_N = np.eye(N)
    I_M = np.eye(M)
    
    det_lhs = float(np.linalg.det(I_N + A @ B.T))
    det_rhs = float(np.linalg.det(I_M + A.T @ B))
    
    diff = abs(det_lhs - det_rhs)
    return {
        "det_lhs": det_lhs,
        "det_rhs": det_rhs,
        "diff": diff,
        "is_equivalent": diff < 1e-10
    }


def rank1_determinant_lemma(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """
    Matrix determinant lemma for rank-1 update (Eq. A.15):
    |I_N + a b^T| = 1 + a^T b
    where a and b are N-dimensional column vectors.
    """
    N = len(a)
    I_N = np.eye(N)
    a_col = a.reshape(-1, 1)
    b_col = b.reshape(-1, 1)
    
    lhs = float(np.linalg.det(I_N + a_col @ b_col.T))
    rhs = float(1.0 + np.dot(a, b))
    
    diff = abs(lhs - rhs)
    return {
        "det_lhs": lhs,
        "scalar_rhs": rhs,
        "diff": diff,
        "is_equivalent": diff < 1e-10
    }


# ==============================================================================
# A.3 Matrix Derivatives (行列の微分)
# ==============================================================================

def verify_matrix_inverse_derivative(A_func, dA_dx_func, x0: float, eps: float = 1e-6) -> Dict[str, Any]:
    """
    Verify Equation (A.21):
    d(A^{-1}) / dx = - A^{-1} (dA / dx) A^{-1}
    using numerical finite differences.
    """
    A0 = A_func(x0)
    inv_A0 = np.linalg.inv(A0)
    dA_dx0 = dA_dx_func(x0)
    
    # Analytical derivative (Eq. A.21)
    analytical = - inv_A0 @ dA_dx0 @ inv_A0
    
    # Numerical finite differences
    inv_A_plus = np.linalg.inv(A_func(x0 + eps))
    inv_A_minus = np.linalg.inv(A_func(x0 - eps))
    numerical = (inv_A_plus - inv_A_minus) / (2.0 * eps)
    
    max_diff = float(np.max(np.abs(analytical - numerical)))
    return {
        "analytical": analytical,
        "numerical": numerical,
        "max_diff": max_diff,
        "is_close": max_diff < 1e-5
    }


def verify_log_det_scalar_derivative(A_func, dA_dx_func, x0: float, eps: float = 1e-6) -> Dict[str, Any]:
    """
    Verify Equation (A.22):
    d/dx ln |A(x)| = Tr( A(x)^{-1} dA/dx )
    """
    A0 = A_func(x0)
    inv_A0 = np.linalg.inv(A0)
    dA_dx0 = dA_dx_func(x0)
    
    # Analytical derivative (Eq. A.22)
    analytical = float(np.trace(inv_A0 @ dA_dx0))
    
    # Numerical finite difference
    log_det_plus = float(np.log(np.linalg.det(A_func(x0 + eps))))
    log_det_minus = float(np.log(np.linalg.det(A_func(x0 - eps))))
    numerical = (log_det_plus - log_det_minus) / (2.0 * eps)
    
    diff = abs(analytical - numerical)
    return {
        "analytical": analytical,
        "numerical": numerical,
        "diff": diff,
        "is_close": diff < 1e-5
    }


def verify_matrix_derivatives_identities(A: np.ndarray, B: np.ndarray, eps: float = 1e-6) -> Dict[str, Any]:
    """
    Verify matrix gradient formulas:
    - (A.24) d/dA Tr(AB) = B^T
    - (A.25) d/dA Tr(A^T B) = B
    - (A.26) d/dA Tr(A) = I
    - (A.27) d/dA Tr(A B A^T) = A (B + B^T)
    - (A.28) d/dA ln |A| = (A^{-1})^T
    """
    N, M = A.shape
    
    # 1. Tr(AB)
    grad_num_AB = np.zeros_like(A)
    for i in range(N):
        for j in range(M):
            A_p = A.copy(); A_p[i, j] += eps
            A_m = A.copy(); A_m[i, j] -= eps
            grad_num_AB[i, j] = (np.trace(A_p @ B) - np.trace(A_m @ B)) / (2.0 * eps)
    grad_ana_AB = B.T
    diff_AB = float(np.max(np.abs(grad_num_AB - grad_ana_AB)))
    
    # 2. Tr(A^T B)
    grad_num_ATB = np.zeros_like(A)
    for i in range(N):
        for j in range(M):
            A_p = A.copy(); A_p[i, j] += eps
            A_m = A.copy(); A_m[i, j] -= eps
            grad_num_ATB[i, j] = (np.trace(A_p.T @ B) - np.trace(A_m.T @ B)) / (2.0 * eps)
    grad_ana_ATB = B
    diff_ATB = float(np.max(np.abs(grad_num_ATB - grad_ana_ATB)))
    
    # 3. Tr(A) (when N == M)
    diff_trA = 0.0
    if N == M:
        grad_num_trA = np.zeros_like(A)
        for i in range(N):
            for j in range(N):
                A_p = A.copy(); A_p[i, j] += eps
                A_m = A.copy(); A_m[i, j] -= eps
                grad_num_trA[i, j] = (np.trace(A_p) - np.trace(A_m)) / (2.0 * eps)
        grad_ana_trA = np.eye(N)
        diff_trA = float(np.max(np.abs(grad_num_trA - grad_ana_trA)))
        
    # 4. Tr(A B A^T)
    grad_num_ABAT = np.zeros_like(A)
    for i in range(N):
        for j in range(M):
            A_p = A.copy(); A_p[i, j] += eps
            A_m = A.copy(); A_m[i, j] -= eps
            val_p = np.trace(A_p @ B @ A_p.T)
            val_m = np.trace(A_m @ B @ A_m.T)
            grad_num_ABAT[i, j] = (val_p - val_m) / (2.0 * eps)
    grad_ana_ABAT = A @ (B + B.T)
    diff_ABAT = float(np.max(np.abs(grad_num_ABAT - grad_ana_ABAT)))
    
    # 5. ln |A| (when N == M)
    diff_lndet = 0.0
    if N == M:
        grad_num_lndet = np.zeros_like(A)
        for i in range(N):
            for j in range(N):
                A_p = A.copy(); A_p[i, j] += eps
                A_m = A.copy(); A_m[i, j] -= eps
                val_p = np.log(np.linalg.det(A_p))
                val_m = np.log(np.linalg.det(A_m))
                grad_num_lndet[i, j] = (val_p - val_m) / (2.0 * eps)
        grad_ana_lndet = np.linalg.inv(A).T
        diff_lndet = float(np.max(np.abs(grad_num_lndet - grad_ana_lndet)))
        
    return {
        "diff_TrAB": diff_AB,
        "diff_TrATB": diff_ATB,
        "diff_TrA": diff_trA,
        "diff_TrABAT": diff_ABAT,
        "diff_lndet": diff_lndet,
        "all_passed": max(diff_AB, diff_ATB, diff_trA, diff_ABAT, diff_lndet) < 1e-4
    }


# ==============================================================================
# A.4 Eigenvectors and Spectral Theory (固有ベクトルとスペクトル理論)
# ==============================================================================

class SymmetricMatrixSpectralAnalysis:
    """
    Implements spectral decomposition and properties of real symmetric matrices
    from Section A.4 (Eqs. A.29 - A.50).
    """
    def __init__(self, A: np.ndarray):
        assert A.ndim == 2 and A.shape[0] == A.shape[1], "A must be a square matrix"
        diff_sym = np.max(np.abs(A - A.T))
        if diff_sym > 1e-10:
            raise ValueError("A must be symmetric (A^T = A)")
            
        self.A = A
        self.dim = A.shape[0]
        
        # Compute eigenvalues and orthonormal eigenvectors
        # np.linalg.eigh returns eigenvalues in ascending order and orthonormal eigenvectors as columns
        eigvals, eigvecs = np.linalg.eigh(A)
        self.eigenvalues = eigvals
        self.eigenvectors = eigvecs  # U matrix where columns are u_i
        
    def verify_orthonormality(self) -> Dict[str, Any]:
        """
        Verify U^T U = I and U U^T = I (Eq. A.33, A.37).
        """
        U = self.eigenvectors
        I = np.eye(self.dim)
        
        ut_u_diff = float(np.max(np.abs(U.T @ U - I)))
        u_ut_diff = float(np.max(np.abs(U @ U.T - I)))
        det_U = float(np.linalg.det(U))
        
        return {
            "ut_u_diff": ut_u_diff,
            "u_ut_diff": u_ut_diff,
            "det_U": det_U,
            "is_orthogonal": (ut_u_diff < 1e-10) and (u_ut_diff < 1e-10) and np.isclose(abs(det_U), 1.0)
        }
        
    def verify_diagonalization(self) -> Dict[str, Any]:
        """
        Verify U^T A U = Lambda (Eq. A.42) and A = U Lambda U^T (Eq. A.43).
        """
        U = self.eigenvectors
        Lambda = np.diag(self.eigenvalues)
        
        # U^T A U = Lambda
        ut_a_u = U.T @ self.A @ U
        diag_diff = float(np.max(np.abs(ut_a_u - Lambda)))
        
        # A = U Lambda U^T
        reconstructed_A = U @ Lambda @ U.T
        recon_diff = float(np.max(np.abs(reconstructed_A - self.A)))
        
        return {
            "diag_diff": diag_diff,
            "recon_diff": recon_diff,
            "is_diagonalized": (diag_diff < 1e-10) and (recon_diff < 1e-10)
        }
        
    def dyadic_expansion(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Dyadic expansion:
        A = sum_{i=1}^M lambda_i u_i u_i^T (Eq. A.45)
        A^{-1} = sum_{i=1}^M (1 / lambda_i) u_i u_i^T (Eq. A.46)
        """
        A_recon = np.zeros_like(self.A)
        for i in range(self.dim):
            u_i = self.eigenvectors[:, i:i+1]
            A_recon += self.eigenvalues[i] * (u_i @ u_i.T)
            
        inv_A_recon = None
        if not np.any(np.isclose(self.eigenvalues, 0.0)):
            inv_A_recon = np.zeros_like(self.A)
            for i in range(self.dim):
                u_i = self.eigenvectors[:, i:i+1]
                inv_A_recon += (1.0 / self.eigenvalues[i]) * (u_i @ u_i.T)
                
        return A_recon, inv_A_recon

    def verify_determinant_and_trace_identities(self) -> Dict[str, Any]:
        """
        Verify:
        |A| = prod_{i=1}^M lambda_i (Eq. A.47)
        Tr(A) = sum_{i=1}^M lambda_i (Eq. A.48)
        """
        det_actual = float(np.linalg.det(self.A))
        det_from_eig = float(np.prod(self.eigenvalues))
        det_diff = abs(det_actual - det_from_eig)
        
        trace_actual = float(np.trace(self.A))
        trace_from_eig = float(np.sum(self.eigenvalues))
        trace_diff = abs(trace_actual - trace_from_eig)
        
        return {
            "det_actual": det_actual,
            "det_from_eig": det_from_eig,
            "det_diff": det_diff,
            "trace_actual": trace_actual,
            "trace_from_eig": trace_from_eig,
            "trace_diff": trace_diff,
            "is_valid": (det_diff < 1e-8) and (trace_diff < 1e-10)
        }

    def definiteness(self) -> str:
        """
        Check positive definiteness or semi-definiteness:
        - A > 0 (positive definite) iff all lambda_i > 0
        - A >= 0 (positive semidefinite) iff all lambda_i >= 0
        - Indefinite if there are both positive and negative eigenvalues.
        """
        min_eig = np.min(self.eigenvalues)
        max_eig = np.max(self.eigenvalues)
        
        if min_eig > 1e-12:
            return "positive_definite"
        elif min_eig >= -1e-12:
            return "positive_semidefinite"
        elif max_eig < -1e-12:
            return "negative_definite"
        else:
            return "indefinite"

    def condition_number(self) -> float:
        """
        Condition number (Eq. A.50):
        CN = (lambda_max / lambda_min)^{1/2} or ratio of extreme singular values / eigenvalues.
        For symmetric positive definite matrix, CN = (lambda_max / lambda_min)^{1/2} as defined in Bishop (A.50).
        """
        pos_eigs = np.abs(self.eigenvalues)
        lam_max = float(np.max(pos_eigs))
        lam_min = float(np.min(pos_eigs))
        if lam_min < 1e-15:
            return float('inf')
        return float(np.sqrt(lam_max / lam_min))


# ==============================================================================
# Visualization / Figure Reproduction Functions
# ==============================================================================

def generate_figure_a_1_orthogonal_rotation(save_path: Optional[str] = None) -> plt.Figure:
    """
    Figure A.1: Geometric interpretation of orthogonal transformation U.
    Multiplication by U represents a rigid rotation of coordinate axes (Eq. A.39),
    preserving lengths ||Ux|| = ||x|| (Eq. A.40) and inner products / angles (Eq. A.41).
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    # 2D Rotation matrix U (orthogonal: U^T U = I, det(U) = 1)
    theta = np.pi / 6  # 30 degrees
    U = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    # Unit circle and sample vectors
    t = np.linspace(0, 2 * np.pi, 200)
    circle_x = np.cos(t)
    circle_y = np.sin(t)
    circle = np.vstack([circle_x, circle_y])
    
    v1 = np.array([1.2, 0.5])
    v2 = np.array([-0.6, 1.4])
    
    # Transformed coordinates
    circle_rot = U @ circle
    v1_rot = U @ v1
    v2_rot = U @ v2
    
    # Subplot 1: Original frame
    ax1 = axes[0]
    ax1.plot(circle[0], circle[1], 'k--', alpha=0.5, label='Unit circle')
    ax1.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='#1E56A0', label=r'$\mathbf{x}$')
    ax1.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='#E02020', label=r'$\mathbf{y}$')
    ax1.axhline(0, color='gray', lw=0.8, ls=':')
    ax1.axvline(0, color='gray', lw=0.8, ls=':')
    ax1.set_xlim(-2.0, 2.0)
    ax1.set_ylim(-2.0, 2.0)
    ax1.set_aspect('equal')
    ax1.set_title(r'Original Coordinates $\mathbf{x}, \mathbf{y}$' + '\n' +
                  rf'$||\mathbf{{x}}|| = {np.linalg.norm(v1):.2f}, ||\mathbf{{y}}|| = {np.linalg.norm(v2):.2f}, \mathbf{{x}}^T \mathbf{{y}} = {np.dot(v1, v2):.2f}$')
    ax1.legend(loc='upper left')
    
    # Subplot 2: Transformed frame
    ax2 = axes[1]
    ax2.plot(circle_rot[0], circle_rot[1], 'k--', alpha=0.5, label='Transformed circle')
    ax2.quiver(0, 0, v1_rot[0], v1_rot[1], angles='xy', scale_units='xy', scale=1, color='#1E56A0', label=r'$\tilde{\mathbf{x}} = \mathbf{U}\mathbf{x}$')
    ax2.quiver(0, 0, v2_rot[0], v2_rot[1], angles='xy', scale_units='xy', scale=1, color='#E02020', label=r'$\tilde{\mathbf{y}} = \mathbf{U}\mathbf{y}$')
    ax2.axhline(0, color='gray', lw=0.8, ls=':')
    ax2.axvline(0, color='gray', lw=0.8, ls=':')
    ax2.set_xlim(-2.0, 2.0)
    ax2.set_ylim(-2.0, 2.0)
    ax2.set_aspect('equal')
    ax2.set_title(r'Rigidly Rotated Coordinates $\mathbf{U}\mathbf{x}, \mathbf{U}\mathbf{y}$' + '\n' +
                  rf'$||\tilde{{\mathbf{{x}}}}|| = {np.linalg.norm(v1_rot):.2f}, ||\tilde{{\mathbf{{y}}}}|| = {np.linalg.norm(v2_rot):.2f}, \tilde{{\mathbf{{x}}}}^T \tilde{{\mathbf{{y}}}} = {np.dot(v1_rot, v2_rot):.2f}$')
    ax2.legend(loc='upper left')
    
    fig.suptitle(r'Appendix A.4: Orthogonal Matrix $\mathbf{U}^T\mathbf{U} = \mathbf{I}$ as Rigid Rotation (Eqs. A.39–A.41)', fontsize=13)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    return fig


def generate_figure_a_2_spectral_decomposition(save_path: Optional[str] = None) -> plt.Figure:
    """
    Figure A.2: Geometric illustration of Spectral Decomposition A = U Lambda U^T (Eq. A.43, A.45).
    A symmetric matrix stretches space along its orthogonal eigenvectors u_1, u_2 by lambda_1, lambda_2.
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    
    # Construct symmetric matrix A = U Lambda U^T
    theta = np.pi / 4  # 45 degrees
    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    lambdas = np.array([2.5, 0.8])
    A = U @ np.diag(lambdas) @ U.T
    
    # 1. Unit circle
    t = np.linspace(0, 2 * np.pi, 250)
    X = np.vstack([np.cos(t), np.sin(t)])  # (2, 250)
    
    # Step 1: Unit circle and eigenbasis
    ax1 = axes[0]
    ax1.plot(X[0], X[1], 'k-', lw=1.5, label='Unit circle')
    ax1.quiver(0, 0, U[0, 0], U[1, 0], angles='xy', scale_units='xy', scale=1, color='#1E56A0', label=r'$\mathbf{u}_1$')
    ax1.quiver(0, 0, U[0, 1], U[1, 1], angles='xy', scale_units='xy', scale=1, color='#E02020', label=r'$\mathbf{u}_2$')
    ax1.set_xlim(-3, 3); ax1.set_ylim(-3, 3); ax1.set_aspect('equal')
    ax1.set_title(r'(1) Canonical Circle & Eigenbasis $\mathbf{U}$')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Step 2: Aligned stretching along principal axes (Lambda U^T X)
    X_rotated = U.T @ X
    X_scaled = np.diag(lambdas) @ X_rotated
    ax2 = axes[1]
    ax2.plot(X_scaled[0], X_scaled[1], color='#2E7D32', lw=1.5, label=r'Stretched $\mathbf{\Lambda}\mathbf{U}^T\mathbf{x}$')
    ax2.quiver(0, 0, lambdas[0], 0, angles='xy', scale_units='xy', scale=1, color='#1E56A0', label=r'$\lambda_1 \mathbf{e}_1$')
    ax2.quiver(0, 0, 0, lambdas[1], angles='xy', scale_units='xy', scale=1, color='#E02020', label=r'$\lambda_2 \mathbf{e}_2$')
    ax2.set_xlim(-3, 3); ax2.set_ylim(-3, 3); ax2.set_aspect('equal')
    ax2.set_title(r'(2) Principal Axes Scaling by $\mathbf{\Lambda}$')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Step 3: Reconstructed Ellipse (A X = U Lambda U^T X)
    AX = A @ X
    ax3 = axes[2]
    ax3.plot(AX[0], AX[1], color='#6A1B9A', lw=2.0, label=r'Ellipse $\mathbf{A}\mathbf{x}$')
    ax3.quiver(0, 0, lambdas[0] * U[0, 0], lambdas[0] * U[1, 0], angles='xy', scale_units='xy', scale=1, color='#1E56A0', label=r'$\lambda_1 \mathbf{u}_1$')
    ax3.quiver(0, 0, lambdas[1] * U[0, 1], lambdas[1] * U[1, 1], angles='xy', scale_units='xy', scale=1, color='#E02020', label=r'$\lambda_2 \mathbf{u}_2$')
    ax3.set_xlim(-3, 3); ax3.set_ylim(-3, 3); ax3.set_aspect('equal')
    ax3.set_title(r'(3) Action of $\mathbf{A} = \mathbf{U}\mathbf{\Lambda}\mathbf{U}^T$ on Space')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    fig.suptitle(r'Appendix A.4: Spectral Decomposition $\mathbf{A} = \sum_{i=1}^M \lambda_i \mathbf{u}_i \mathbf{u}_i^T$ (Eqs. A.43, A.45)', fontsize=13)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    return fig


def generate_figure_a_3_quadratic_forms(save_path: Optional[str] = None) -> plt.Figure:
    """
    Figure A.3: Quadratic forms w^T A w for Positive Definite vs Indefinite matrices.
    Illustrates the textbook counterexample (A.49):
    Matrix [1 2; 3 4] (symmetrized to [1 2.5; 2.5 4]) has positive elements but is INDEFINITE,
    with eigenvalues ~ 5.37 and -0.37 (saddle shape).
    """
    setup_style()
    fig = plt.figure(figsize=(12, 5.5))
    
    w1 = np.linspace(-2.5, 2.5, 80)
    w2 = np.linspace(-2.5, 2.5, 80)
    W1, W2 = np.meshgrid(w1, w2)
    grid_coords = np.stack([W1.ravel(), W2.ravel()], axis=0)  # (2, 6400)
    
    # 1. Positive Definite Matrix: A_pd = [2 0.8; 0.8 1.5], eigs > 0
    A_pd = np.array([[2.0, 0.8], [0.8, 1.5]])
    eigs_pd = np.linalg.eigvalsh(A_pd)
    Z_pd = np.sum(grid_coords * (A_pd @ grid_coords), axis=0).reshape(W1.shape)
    
    # 2. Indefinite Matrix (Textbook A.49 symmetrized): A_indef = [1 2.5; 2.5 4]
    # Note: w^T A w = w^T (A + A^T)/2 w
    A_indef = np.array([[1.0, 2.5], [2.5, 4.0]])
    eigs_indef = np.linalg.eigvalsh(A_indef)
    Z_indef = np.sum(grid_coords * (A_indef @ grid_coords), axis=0).reshape(W1.shape)
    
    # Subplot 1: Positive Definite Paraboloid
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(W1, W2, Z_pd, cmap='viridis', alpha=0.85, edgecolor='none')
    ax1.set_title(rf"Positive Definite ($\mathbf{{A}} \succ 0$)" + "\n" +
                  rf"$\lambda_1={eigs_pd[1]:.2f}, \lambda_2={eigs_pd[0]:.2f} > 0$ (Strict Minimum)")
    ax1.set_xlabel(r'$w_1$')
    ax1.set_ylabel(r'$w_2$')
    ax1.set_zlabel(r'$\mathbf{w}^T \mathbf{A} \mathbf{w}$')
    
    # Subplot 2: Indefinite Saddle Paraboloid (Eq. A.49)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(W1, W2, Z_indef, cmap='coolwarm', alpha=0.85, edgecolor='none')
    ax2.set_title(rf"Indefinite Counterexample (Eq. A.49)" + "\n" +
                  rf"$\lambda_1={eigs_indef[1]:.2f} > 0, \lambda_2={eigs_indef[0]:.2f} < 0$ (Saddle Point)")
    ax2.set_xlabel(r'$w_1$')
    ax2.set_ylabel(r'$w_2$')
    ax2.set_zlabel(r'$\mathbf{w}^T \mathbf{A} \mathbf{w}$')
    
    fig.suptitle(r'Appendix A.4: Positive Definiteness $\mathbf{w}^T\mathbf{A}\mathbf{w} > 0$ vs Indefinite Counterexample (Eqs. A.49, A.50)', fontsize=13)
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    return fig


def generate_figure_a_4_woodbury_speedup(save_path: Optional[str] = None) -> plt.Figure:
    """
    Figure A.4: Benchmark of Woodbury Matrix Identity (Eq. A.7).
    Compares runtime of direct inversion O(N^3) versus Woodbury inversion O(M^3 + N M^2)
    when performing a rank-M update (M = 10) on an N x N diagonal matrix as N scales from 100 to 2000.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    dimensions = [100, 200, 500, 800, 1200, 1600, 2000]
    M = 10
    
    direct_times = []
    woodbury_times = []
    
    np.random.seed(42)
    for N in dimensions:
        # A is diagonal N x N
        A = np.diag(np.random.uniform(1.0, 3.0, size=N))
        B = np.random.randn(N, M)
        D = np.eye(M) + 0.1 * np.random.randn(M, M)
        D = D @ D.T  # positive definite M x M
        C = B.T
        
        # Direct inversion time
        t0 = time.perf_counter()
        inv_naive = np.linalg.inv(A + B @ np.linalg.inv(D) @ C)
        t_direct = time.perf_counter() - t0
        direct_times.append(t_direct * 1000)  # ms
        
        # Woodbury inversion time (using fast diagonal inverse A^{-1})
        t0 = time.perf_counter()
        inv_A_diag = 1.0 / np.diag(A)
        inv_A_B = inv_A_diag[:, None] * B
        mid = np.linalg.inv(D + C @ inv_A_B)
        inv_woodbury = np.diag(inv_A_diag) - inv_A_B @ mid @ (C * inv_A_diag[None, :])
        t_woodbury = time.perf_counter() - t0
        woodbury_times.append(t_woodbury * 1000)  # ms
        
    ax.plot(dimensions, direct_times, 'o-', color='#E02020', lw=2, label=r'Direct Inversion $\mathcal{O}(N^3)$')
    ax.plot(dimensions, woodbury_times, 's-', color='#1E56A0', lw=2, label=rf'Woodbury Identity $\mathcal{{O}}(M^3 + NM^2)$ ($M={M}$)')
    
    ax.set_xlabel(r'Matrix Dimension $N$ ($M=10$ fixed)', fontsize=12)
    ax.set_ylabel('Execution Time [ms]', fontsize=12)
    ax.set_title(r'Woodbury Identity Computational Scaling (Eq. A.7)', fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=11)
    
    plt.tight_layout()
    if save_path:
        save_plot(fig, save_path)
    return fig


def generate_all_appendix_a_figures(output_dir: str = "appendix/result") -> List[str]:
    """
    Generate and save all figures for Appendix A.
    Saves in both output_dir and root result/.
    """
    repo_root = Path.cwd().parent if Path.cwd().name == "appendix" else Path.cwd()
    out_path = Path(output_dir) if Path(output_dir).is_absolute() else (repo_root / output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    root_result = repo_root / "result"
    root_result.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # 1. Figure A.1: Orthogonal rotation
    p1 = out_path / "figA_1_orthogonal_rotation.png"
    p1_root = root_result / "figA_1_orthogonal_rotation.png"
    fig1 = generate_figure_a_1_orthogonal_rotation(str(p1))
    save_plot(fig1, str(p1_root))
    plt.close(fig1)
    saved_files.extend([str(p1), str(p1_root)])
    
    # 2. Figure A.2: Spectral decomposition
    p2 = out_path / "figA_2_spectral_decomposition_ellipse.png"
    p2_root = root_result / "figA_2_spectral_decomposition_ellipse.png"
    fig2 = generate_figure_a_2_spectral_decomposition(str(p2))
    save_plot(fig2, str(p2_root))
    plt.close(fig2)
    saved_files.extend([str(p2), str(p2_root)])
    
    # 3. Figure A.3: Quadratic forms
    p3 = out_path / "figA_3_quadratic_forms.png"
    p3_root = root_result / "figA_3_quadratic_forms.png"
    fig3 = generate_figure_a_3_quadratic_forms(str(p3))
    save_plot(fig3, str(p3_root))
    plt.close(fig3)
    saved_files.extend([str(p3), str(p3_root)])
    
    # 4. Figure A.4: Woodbury speedup benchmark
    p4 = out_path / "figA_4_woodbury_speedup.png"
    p4_root = root_result / "figA_4_woodbury_speedup.png"
    fig4 = generate_figure_a_4_woodbury_speedup(str(p4))
    save_plot(fig4, str(p4_root))
    plt.close(fig4)
    saved_files.extend([str(p4), str(p4_root)])
    
    return saved_files
