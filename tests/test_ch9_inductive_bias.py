"""Tests for Section 9.1: Inductive Bias.

Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
Chapter 9: Regularization
"""

import os
import pytest
import numpy as np
from PIL import Image

from common.inductive_bias import (
    fit_polynomial_regression,
    compute_smoothness_norm,
    simulate_no_free_lunch,
    create_cyclic_group_c4,
    create_dihedral_group_d4,
    verify_noise_gradient_regularization_equivalence,
    translate_image,
    segment_silhouette,
    verify_equivariance_property,
    generate_figure_9_1,
    generate_figure_9_2,
)


def test_inverse_problem_polynomial_fit():
    """Test polynomial fitting under ill-posed inverse problem setting."""
    rng = np.random.RandomState(42)
    x = np.linspace(-1, 1, 10)
    t = np.sin(np.pi * x) + rng.randn(10) * 0.2

    # Overfitted polynomial (degree 9, lambda=0)
    w_unreg, fn_unreg = fit_polynomial_regression(x, t, degree=9, reg_lambda=0.0)

    # Regularized polynomial (degree 9, lambda=0.1)
    w_reg, fn_reg = fit_polynomial_regression(x, t, degree=9, reg_lambda=0.1)

    x_dense = np.linspace(-1, 1, 200)
    smooth_unreg = compute_smoothness_norm(fn_unreg, x_dense)
    smooth_reg = compute_smoothness_norm(fn_reg, x_dense)

    # Regularization encourages smoother functions (smaller derivative norm)
    assert smooth_reg < smooth_unreg
    assert np.linalg.norm(w_reg) < np.linalg.norm(w_unreg)


def test_no_free_lunch_theorem():
    """Test No Free Lunch theorem on discrete binary classification."""
    res = simulate_no_free_lunch(num_inputs=6, num_train=3)
    assert res["nfl_verified"] is True
    assert abs(res["avg_error_A"] - 0.5) < 1e-10
    assert abs(res["avg_error_B"] - 0.5) < 1e-10
    assert abs(res["avg_error_C"] - 0.5) < 1e-10


def test_group_axioms_c4():
    """Verify group axioms for cyclic group C_4 (rotations of a square)."""
    c4 = create_cyclic_group_c4()
    axioms = c4.verify_axioms()
    assert axioms["closure"] is True
    assert axioms["associativity"] is True
    assert axioms["identity"] is True
    assert axioms["inverse"] is True
    assert axioms["is_valid_group"] is True


def test_group_axioms_d4():
    """Verify group axioms for dihedral group D_4 (symmetries of a square)."""
    d4 = create_dihedral_group_d4()
    axioms = d4.verify_axioms()
    assert axioms["closure"] is True
    assert axioms["associativity"] is True
    assert axioms["identity"] is True
    assert axioms["inverse"] is True
    assert axioms["is_valid_group"] is True


def test_noise_gradient_regularization_equivalence():
    """Test Bishop (1995c) theorem connecting noise augmentation to input gradient penalty."""
    # Nonlinear function y(x) = sin(x1) * cos(x2)
    model_fn = lambda x: np.sin(x[0]) * np.cos(x[1])
    grad_fn = lambda x: np.array([np.cos(x[0]) * np.cos(x[1]), -np.sin(x[0]) * np.sin(x[1])])

    x0 = np.array([0.5, 0.8])
    t0 = model_fn(x0)  # evaluated near minimum (zero residual)

    res = verify_noise_gradient_regularization_equivalence(
        model_fn, grad_fn, x0, t0, noise_sigma=0.01, num_samples=30000
    )
    assert res["verified"] is True
    assert res["rel_diff"] < 0.05


def test_equivariance_property():
    """Test equivariance property S(T(I)) = T(S(I))."""
    # Create synthetic test image with centered circle
    w, h = 100, 100
    im = Image.new("RGB", (w, h), (50, 50, 50))
    arr = np.array(im)
    y, x = np.ogrid[:h, :w]
    mask = (x - 50) ** 2 + (y - 50) ** 2 <= 20 ** 2
    arr[mask] = [200, 100, 30]  # Cat fur-like color
    im = Image.fromarray(arr)

    op_T = lambda img: translate_image(img, 10, 5)
    op_S = lambda img: segment_silhouette(img, fg_color=(255, 120, 0))

    res = verify_equivariance_property(im, op_T, op_S)
    # The two paths should commute inside the valid boundary
    assert res["is_equivariant"] is True
    assert res["max_diff"] == 0


def test_generate_figure_9_1(tmp_path):
    """Test generation and saving of Figure 9.1."""
    fig = generate_figure_9_1(result_dirs=[str(tmp_path)], save_both=True)
    assert fig is not None

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    assert os.path.exists(os.path.join(root, "9", "result", "fig_9_1_data_augmentation.png"))
    assert os.path.exists(os.path.join(root, "result", "fig_9_1_data_augmentation.png"))
    assert os.path.exists(os.path.join(root, "9", "result", "Figure_9_1.png"))
    assert os.path.exists(os.path.join(root, "result", "Figure_9_1.png"))


def test_generate_figure_9_2(tmp_path):
    """Test generation and saving of Figure 9.2."""
    fig = generate_figure_9_2(result_dirs=[str(tmp_path)], save_both=True)
    assert fig is not None

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    assert os.path.exists(os.path.join(root, "9", "result", "fig_9_2_equivariance.png"))
    assert os.path.exists(os.path.join(root, "result", "fig_9_2_equivariance.png"))
    assert os.path.exists(os.path.join(root, "9", "result", "Figure_9_2.png"))
    assert os.path.exists(os.path.join(root, "result", "Figure_9_2.png"))
