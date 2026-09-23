"""Tests for Chapter 8 Section 8.2: Automatic Differentiation.

Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
Section 8.2: Automatic Differentiation
- 8.2.1 Forward-mode automatic differentiation (Dual numbers, tangent variables, JVP, Jacobian)
- 8.2.2 Reverse-mode automatic differentiation (Computational graph, adjoint variables, backprop)
- Figures 8.4 and 8.5 reproduction
"""

import os
import numpy as np
import pytest

from common.automatic_differentiation import (
    DualNumber,
    Node,
    d_exp,
    d_log,
    d_sin,
    d_cos,
    d_tanh,
    forward_mode_derivative,
    forward_mode_jvp,
    forward_mode_jacobian,
    example_function_8_49,
    example_function_8_65,
    evaluate_trace_forward_mode,
    evaluate_trace_reverse_mode,
    generate_figure_8_4,
    generate_figure_8_5,
    plot_figure_8_4,
    plot_figure_8_5,
    generate_all_section_8_2_figures,
)


class TestDualNumber:
    def test_basic_arithmetic(self):
        x = DualNumber(4.0, 1.0)
        y = x**2 + 3 * x - 5
        assert np.isclose(y.real, 23.0)
        assert np.isclose(y.dual, 11.0)

    def test_multiplication_product_rule(self):
        x = DualNumber(2.0, 1.0)
        u = 2 * x + 1
        v = 3 * x - 2
        p = u * v
        assert np.isclose(p.real, 20.0)
        assert np.isclose(p.dual, 23.0)

    def test_division_quotient_rule(self):
        x = DualNumber(1.0, 1.0)
        u = 2 * x + 1
        v = x**2 + 1
        q = u / v
        assert np.isclose(q.real, 1.5)
        assert np.isclose(q.dual, -0.5)

    def test_reverse_operators(self):
        x = DualNumber(3.0, 1.0)
        sub = 10.0 - x
        assert np.isclose(sub.real, 7.0)
        assert np.isclose(sub.dual, -1.0)

        div = 12.0 / x
        assert np.isclose(div.real, 4.0)
        assert np.isclose(div.dual, -4.0 / 3.0)

    def test_transcendental_functions(self):
        x_val = 1.5
        x = DualNumber(x_val, 1.0)

        e = x.exp()
        assert np.isclose(e.real, np.exp(x_val))
        assert np.isclose(e.dual, np.exp(x_val))

        l = x.log()
        assert np.isclose(l.real, np.log(x_val))
        assert np.isclose(l.dual, 1.0 / x_val)

        s = x.sin()
        assert np.isclose(s.real, np.sin(x_val))
        assert np.isclose(s.dual, np.cos(x_val))

        c = x.cos()
        assert np.isclose(c.real, np.cos(x_val))
        assert np.isclose(c.dual, -np.sin(x_val))

        t = x.tanh()
        assert np.isclose(t.real, np.tanh(x_val))
        assert np.isclose(t.dual, 1.0 - np.tanh(x_val) ** 2)

        sig = x.sigmoid()
        expected_sig = 1.0 / (1.0 + np.exp(-x_val))
        assert np.isclose(sig.real, expected_sig)
        assert np.isclose(sig.dual, expected_sig * (1.0 - expected_sig))

    def test_free_math_functions(self):
        x = DualNumber(0.5, 1.0)
        assert np.isclose(d_exp(x).real, np.exp(0.5))
        assert np.isclose(d_log(x).real, np.log(0.5))
        assert np.isclose(d_sin(x).real, np.sin(0.5))
        assert np.isclose(d_cos(x).real, np.cos(0.5))
        assert np.isclose(d_tanh(x).real, np.tanh(0.5))

        assert np.isclose(d_exp(0.5), np.exp(0.5))
        assert np.isclose(d_log(0.5), np.log(0.5))


class TestForwardModeEvaluation:
    def test_forward_mode_derivative_scalar(self):
        f = lambda x: x**3 - 2 * x + 5
        val, grad = forward_mode_derivative(f, 2.0)
        assert np.isclose(val, 9.0)
        assert np.isclose(grad, 10.0)

    def test_forward_mode_jvp_function_8_49(self):
        x = np.array([2.0, 3.0])
        x1, x2 = x[0], x[1]
        prod = x1 * x2
        analytical_df_dx1 = x2 * (1.0 + np.exp(prod))
        analytical_df_dx2 = x1 * (1.0 + np.exp(prod)) - np.cos(x2)

        def func_wrapper(duals):
            return example_function_8_49(duals[0], duals[1])

        y, jvp1 = forward_mode_jvp(func_wrapper, x, np.array([1.0, 0.0]))
        assert np.isclose(jvp1[0], analytical_df_dx1)

        _, jvp2 = forward_mode_jvp(func_wrapper, x, np.array([0.0, 1.0]))
        assert np.isclose(jvp2[0], analytical_df_dx2)

        v = np.array([0.7, -0.4])
        expected_dir_deriv = analytical_df_dx1 * v[0] + analytical_df_dx2 * v[1]
        _, jvp_v = forward_mode_jvp(func_wrapper, x, v)
        assert np.isclose(jvp_v[0], expected_dir_deriv)

    def test_forward_mode_trace_step_by_step(self):
        x1, x2 = 1.5, 0.8
        trace = evaluate_trace_forward_mode(x1, x2)

        assert np.isclose(trace["v1"][0], x1)
        assert np.isclose(trace["v1"][1], 1.0)
        assert np.isclose(trace["v2"][0], x2)
        assert np.isclose(trace["v2"][1], 0.0)

        v3_expected = x1 * x2
        v3_dot_expected = x2
        assert np.isclose(trace["v3"][0], v3_expected)
        assert np.isclose(trace["v3"][1], v3_dot_expected)

        v4_expected = np.sin(x2)
        v4_dot_expected = 0.0
        assert np.isclose(trace["v4"][0], v4_expected)
        assert np.isclose(trace["v4"][1], v4_dot_expected)

        v5_expected = np.exp(v3_expected)
        v5_dot_expected = v3_dot_expected * np.exp(v3_expected)
        assert np.isclose(trace["v5"][0], v5_expected)
        assert np.isclose(trace["v5"][1], v5_dot_expected)

        v6_expected = v3_expected - v4_expected
        v6_dot_expected = v3_dot_expected - v4_dot_expected
        assert np.isclose(trace["v6"][0], v6_expected)
        assert np.isclose(trace["v6"][1], v6_dot_expected)

        v7_expected = v5_expected + v6_expected
        v7_dot_expected = v5_dot_expected + v6_dot_expected
        assert np.isclose(trace["v7"][0], v7_expected)
        assert np.isclose(trace["v7"][1], v7_dot_expected)

    def test_forward_mode_jacobian_multi_output(self):
        x = np.array([1.2, 0.7])

        def func_multi(duals):
            f1, f2 = example_function_8_65(duals[0], duals[1])
            return [f1, f2]

        J_forward = forward_mode_jacobian(func_multi, x)
        assert J_forward.shape == (2, 2)

        eps = 1e-7
        J_num = np.zeros((2, 2))
        for i in range(2):
            dx = np.zeros(2)
            dx[i] = eps
            x_plus = x + dx
            x_minus = x - dx
            f_plus = np.array(example_function_8_65(x_plus[0], x_plus[1]))
            f_minus = np.array(example_function_8_65(x_minus[0], x_minus[1]))
            J_num[:, i] = (f_plus - f_minus) / (2.0 * eps)

        assert np.allclose(J_forward, J_num, rtol=1e-5, atol=1e-5)


class TestReverseModeNode:
    def test_node_basic_ops(self):
        a = Node(2.0, label="a")
        b = Node(5.0, label="b")
        y = (a + b) * (b - 3) / a

        assert np.isclose(y.value, 7.0)
        y.backward()

        eps = 1e-7
        y_a_plus = ((a.value + eps + b.value) * (b.value - 3)) / (a.value + eps)
        y_a_minus = ((a.value - eps + b.value) * (b.value - 3)) / (a.value - eps)
        grad_a_num = (y_a_plus - y_a_minus) / (2 * eps)

        y_b_plus = ((a.value + b.value + eps) * (b.value + eps - 3)) / a.value
        y_b_minus = ((a.value + b.value - eps) * (b.value - eps - 3)) / a.value
        grad_b_num = (y_b_plus - y_b_minus) / (2 * eps)

        assert np.isclose(a.grad, grad_a_num, rtol=1e-5)
        assert np.isclose(b.grad, grad_b_num, rtol=1e-5)

    def test_node_elementary_functions(self):
        x_val = 0.8
        x = Node(x_val, label="x")
        y = x.sin().exp() + (x.cos() ** 2) + x.tanh()

        expected_val = (
            np.exp(np.sin(x_val)) + (np.cos(x_val) ** 2) + np.tanh(x_val)
        )
        assert np.isclose(y.value, expected_val)

        y.backward()

        expected_grad = (
            np.exp(np.sin(x_val)) * np.cos(x_val)
            - 2 * np.cos(x_val) * np.sin(x_val)
            + (1.0 - np.tanh(x_val) ** 2)
        )
        assert np.isclose(x.grad, expected_grad, rtol=1e-6)

    def test_node_example_function_8_49(self):
        x1_val, x2_val = 1.3, 0.9
        x1 = Node(x1_val, label="x1")
        x2 = Node(x2_val, label="x2")

        out = example_function_8_49(x1, x2)
        out.backward()

        prod = x1_val * x2_val
        expected_df_dx1 = x2_val * (1.0 + np.exp(prod))
        expected_df_dx2 = x1_val * (1.0 + np.exp(prod)) - np.cos(x2_val)

        assert np.isclose(x1.grad, expected_df_dx1, rtol=1e-6)
        assert np.isclose(x2.grad, expected_df_dx2, rtol=1e-6)

    def test_evaluate_trace_reverse_mode_matches_node(self):
        x1_val, x2_val = 1.4, 0.6
        trace = evaluate_trace_reverse_mode(x1_val, x2_val)

        x1 = Node(x1_val, label="x1")
        x2 = Node(x2_val, label="x2")
        out = example_function_8_49(x1, x2)
        out.backward()

        assert np.isclose(trace["v1"][0], x1.value)
        assert np.isclose(trace["v2"][0], x2.value)
        assert np.isclose(trace["v1"][1], x1.grad)
        assert np.isclose(trace["v2"][1], x2.grad)
        assert np.isclose(trace["v7"][1], 1.0)


class TestFigureGeneration:
    def test_generate_figure_8_4(self, tmp_path):
        out_path = os.path.join(tmp_path, "fig8_4.png")
        fig = generate_figure_8_4(filepath=out_path, save_both=False)
        assert fig is not None
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000

    def test_generate_figure_8_5(self, tmp_path):
        out_path = os.path.join(tmp_path, "fig8_5.png")
        fig = generate_figure_8_5(filepath=out_path, save_both=False)
        assert fig is not None
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000

    def test_figure_aliases(self, tmp_path):
        out_path4 = os.path.join(tmp_path, "f84.png")
        out_path5 = os.path.join(tmp_path, "f85.png")
        f4 = plot_figure_8_4(filepath=out_path4, save_both=False)
        f5 = plot_figure_8_5(filepath=out_path5, save_both=False)
        assert os.path.exists(out_path4)
        assert os.path.exists(out_path5)

    def test_generate_all_section_8_2_figures(self, tmp_path):
        dirs = [str(tmp_path / "res1"), str(tmp_path / "res2")]
        res = generate_all_section_8_2_figures(result_dirs=dirs, save_both=False)
        assert "fig_8_4" in res
        assert "fig_8_5" in res
        for d in dirs:
            assert os.path.exists(os.path.join(d, "fig_8_4_evaluation_trace.png"))
            assert os.path.exists(os.path.join(d, "fig_8_5_multi_output_trace.png"))
