"""
Unit tests for Appendix C: Lagrange Multipliers (付録C: ラグランジュの未定乗数法)
Validates all equations (C.1) - (C.12), KKT conditions, and figures from Bishop & Bishop (2024).
"""

import os
from pathlib import Path
import numpy as np
import pytest

from common.lagrange_multipliers import (
    solve_textbook_example_c5,
    verify_normal_vector_property,
    verify_kkt_conditions,
    ConstrainedLagrangianOptimizer,
    generate_all_appendix_c_figures,
)


class TestAppendixCLagrangeMultipliers:
    """Tests for Appendix C: Lagrange Multipliers."""

    def test_textbook_quadratic_example_c5(self):
        # Eqs. (C.5) - (C.8): Maximize f = 1 - x1^2 - x2^2 s.t. x1 + x2 - 1 = 0
        res = solve_textbook_example_c5()
        
        # Check optimal stationary point (1/2, 1/2) and multiplier lambda = 1
        assert np.allclose(res["x_star"], np.array([0.5, 0.5]))
        assert np.isclose(res["lambda_star"], 1.0)
        assert np.isclose(res["f_star"], 0.5)
        
        # Check stationarity condition grad f + lambda * grad g = 0 (Eq. C.3)
        assert res["is_optimal"]
        assert np.allclose(res["stationarity_residual"], np.array([0.0, 0.0]), atol=1e-10)

    def test_normal_vector_orthogonality(self):
        # Eq. (C.2): epsilon^T grad g = 0 for tangent vector epsilon
        # Constraint surface: circle g(x1, x2) = x1^2 + x2^2 - 1 = 0
        def g_circle(x):
            return float(x[0]**2 + x[1]**2 - 1.0)
            
        def grad_g_circle(x):
            return np.array([2.0 * x[0], 2.0 * x[1]])
            
        # Point on circle at 45 degrees: (1/sqrt(2), 1/sqrt(2))
        x_pt = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])
        # Tangent vector: (-1/sqrt(2), 1/sqrt(2))
        tangent = np.array([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])
        
        res = verify_normal_vector_property(g_circle, grad_g_circle, x_pt, tangent)
        assert res["is_orthogonal"]
        assert abs(res["inner_prod"]) < 1e-10

    def test_kkt_conditions_active_and_inactive(self):
        # Eqs. (C.9) - (C.11): Karush-Kuhn-Tucker conditions
        
        # Case 1: Active boundary constraint (x_A on boundary g=0)
        # Maximize f(x) = -(x - 2)^2 s.t. g(x) = 1 - x >= 0 (so x <= 1)
        # Optimal point is x = 1 (on boundary).
        # grad f = -2*(1 - 2) = +2. grad g = -1.
        # grad f + lambda * grad g = 2 + lambda * (-1) = 0 => lambda = 2 > 0.
        res_active = verify_kkt_conditions(
            x=np.array([1.0]),
            lambda_val=2.0,
            grad_f=np.array([2.0]),
            g_val=0.0,
            grad_g=np.array([-1.0]),
            is_maximization=True
        )
        assert res_active["all_satisfied"]
        assert res_active["complementary_slackness"]
        
        # Case 2: Inactive interior constraint (x_B in interior g > 0)
        # Maximize f(x) = -x^2 s.t. g(x) = 1 - x >= 0 (so x <= 1)
        # Optimal point is unconstrained maximum x = 0 (in interior g(0) = 1 > 0).
        # grad f = 0, lambda = 0.
        res_inactive = verify_kkt_conditions(
            x=np.array([0.0]),
            lambda_val=0.0,
            grad_f=np.array([0.0]),
            g_val=1.0,
            grad_g=np.array([-1.0]),
            is_maximization=True
        )
        assert res_inactive["all_satisfied"]
        assert res_inactive["complementary_slackness"]

    def test_constrained_lagrangian_optimizer(self):
        # Minimize f(x1, x2) = (x1 - 2)^2 + (x2 - 2)^2
        # subject to equality constraint x1 + x2 - 2 = 0
        def f(x):
            return float((x[0] - 2.0)**2 + (x[1] - 2.0)**2)
            
        def grad_f(x):
            return np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 2.0)])
            
        def g(x):
            return float(x[0] + x[1] - 2.0)
            
        def grad_g(x):
            return np.array([1.0, 1.0])
            
        optimizer = ConstrainedLagrangianOptimizer(
            f_func=f,
            grad_f_func=grad_f,
            eq_constraints=[(g, grad_g)]
        )
        
        res = optimizer.solve(x_init=np.array([0.0, 0.0]), n_iters=300, lr=0.05, penalty_rho=20.0)
        
        # Analytical optimum is x1 = 1, x2 = 1 (on line x1 + x2 = 2)
        assert np.allclose(res["x_opt"], np.array([1.0, 1.0]), atol=0.05)
        assert abs(g(res["x_opt"])) < 0.05

    def test_appendix_c_figures_generation(self):
        saved = generate_all_appendix_c_figures(output_dir="appendix/result")
        assert len(saved) == 8  # 4 figures x 2 dirs (appendix/result and result/)
        for path in saved:
            assert os.path.exists(path), f"Missing: {path}"
            assert os.path.getsize(path) > 1000, f"Empty: {path}"
