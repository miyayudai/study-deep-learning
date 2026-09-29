"""
Unit tests for Appendix B: Calculus of Variations (付録B: 変分法)
Validates all concepts, equations (B.1) - (B.10), Euler-Lagrange equations,
and variational optimization from Bishop & Bishop (2024).
"""

import os
from pathlib import Path
import numpy as np
import pytest

from common.calculus_of_variations import (
    functional_derivative_finite_difference,
    solve_euler_lagrange_example_b10,
    evaluate_example_functional_b9,
    optimize_functional_discrete,
    compute_curve_length,
    generate_perturbed_curves,
    compare_maximum_entropy_distributions,
    generate_all_appendix_b_figures,
)


class TestAppendixBCalculusOfVariations:
    """Tests for Appendix B Calculus of Variations."""

    def test_functional_derivative_finite_difference(self):
        # Eq. (B.3): F[y + eps*eta] = F[y] + eps * int (delta F / delta y) eta dx + O(eps^2)
        x = np.linspace(0, 1, 100)
        dx = x[1] - x[0]
        y_base = np.sin(np.pi * x)
        eta = np.sin(2 * np.pi * x)  # vanishes at boundaries x=0, 1
        
        # Test with quadratic functional F[y] = int y(x)^2 dx
        # Analytical delta F / delta y(x) = 2 * y(x)
        # Gateaux directional derivative = int (2 * y(x)) * eta(x) dx
        from common.calculus_of_variations import trapz_compat
        def F_quad(y):
            return trapz_compat(y**2, dx=dx)
            
        res = functional_derivative_finite_difference(F_quad, y_base, eta, dx=dx, eps=1e-5)
        num_deriv = res["numerical_directional_derivative"]
        ana_deriv = trapz_compat(2.0 * y_base * eta, dx=dx)
        
        assert np.isclose(num_deriv, ana_deriv, atol=1e-4)

    def test_euler_lagrange_analytical_solution_b10(self):
        # Eq. (B.9) - (B.10): G = y^2 + (y')^2 => y - y'' = 0
        x = np.linspace(0.0, 1.0, 100)
        y0, y1 = 1.0, 2.5
        y_exact, (C1, C2) = solve_euler_lagrange_example_b10(x, y0, y1)
        
        # Check boundary conditions
        assert np.isclose(y_exact[0], y0)
        assert np.isclose(y_exact[-1], y1)
        
        # Check differential equation y - y'' = 0
        dx = x[1] - x[0]
        d2y_dx2 = np.gradient(np.gradient(y_exact, dx), dx)
        # Check interior points (away from numerical boundary artifacts)
        residual = y_exact[5:-5] - d2y_dx2[5:-5]
        assert np.allclose(residual, 0.0, atol=1e-2)

    def test_discrete_functional_optimization_convergence(self):
        # Numerical gradient descent on F[y] converges to Euler-Lagrange solution
        x = np.linspace(0.0, 1.0, 40)
        y0, y1 = 1.0, 2.0
        y_exact, _ = solve_euler_lagrange_example_b10(x, y0, y1)
        
        y_opt, loss_hist = optimize_functional_discrete(x, y0, y1, n_steps=2000)
        
        # Functional loss strictly decreases
        assert loss_hist[-1] < loss_hist[0]
        
        # Optimized curve is very close to exact analytical solution
        max_diff = np.max(np.abs(y_opt - y_exact))
        assert max_diff < 0.05

    def test_shortest_path_geodesic_property(self):
        # The straight line minimizes curve length L[y] = int sqrt(1 + (y')^2) dx
        x = np.linspace(0.0, 2.0, 200)
        y_straight = 0.5 * x + 1.0
        l_straight = compute_curve_length(x, y_straight)
        
        # Perturbed paths with amplitude != 0 must be strictly longer
        perturbed_curves = generate_perturbed_curves(x, y_straight, [-1.0, -0.5, 0.5, 1.0])
        for amp, y_p, l_p in perturbed_curves:
            assert l_p > l_straight
            assert l_p - l_straight > 0.01

    def test_maximum_entropy_gaussian_property(self):
        # Under fixed variance, Gaussian has maximal differential entropy
        res = compare_maximum_entropy_distributions(variance=2.0)
        assert res["is_gaussian_maximum"]
        assert res["margin_over_second"] > 0.02
        
        entropies = res["entropies"]
        assert entropies["Gaussian"] > entropies["Laplace"]
        assert entropies["Gaussian"] > entropies["Uniform"]
        assert entropies["Gaussian"] > entropies["Triangular"]

    def test_appendix_b_figures_generation(self):
        saved = generate_all_appendix_b_figures(output_dir="appendix/result")
        assert len(saved) == 8  # 4 figures x 2 dirs (appendix/result and result/)
        for path in saved:
            assert os.path.exists(path), f"Missing: {path}"
            assert os.path.getsize(path) > 1000, f"Empty: {path}"
