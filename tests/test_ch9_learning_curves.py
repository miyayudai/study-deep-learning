"""
Tests for Chapter 9 Section 9.3 Learning Curves (Bishop & Bishop 2024).
"""

from pathlib import Path
import numpy as np
import pytest

from common.learning_curves import (
    EarlyStoppingAnalysis,
    GradientFlowShrinkage,
    DoubleDescentModel,
    generate_figure_9_7,
    generate_figure_9_8,
    generate_figure_9_9,
    generate_figure_9_10,
    generate_figure_9_11,
)


def test_early_stopping_analysis():
    """Verify learning curve generation and early stopping checkpoint detection."""
    analyzer = EarlyStoppingAnalysis(n_steps=51)
    steps, train_err, val_err = analyzer.get_curves()

    assert len(steps) == 51
    assert len(train_err) == 51
    assert len(val_err) == 51

    # Train error monotonically decreases
    assert np.all(np.diff(train_err) <= 1e-6)

    # Validation error has a minimum before the end
    assert 20 <= analyzer.best_step <= 35
    assert analyzer.best_step < 50
    assert analyzer.best_val_error < analyzer.final_val_error
    assert analyzer.error_reduction_from_early_stopping() > 0.0


def test_gradient_flow_and_weight_decay_equivalence():
    """Verify continuous gradient flow solution and equivalence to weight decay (Eq 9.8 discussion)."""
    eta1, eta2 = 0.5, 4.0
    H = np.diag([eta1, eta2])
    w_star = np.array([3.0, 2.0])

    flow = GradientFlowShrinkage(H=H, w_star=w_star)
    assert len(flow.eigenvalues) == 2

    # At t=0, w(0) must be 0
    w_0 = flow.gradient_flow_point(0.0)
    assert np.allclose(w_0, np.zeros(2))

    # At large t, w(t) -> w_star
    w_inf = flow.gradient_flow_point(50.0)
    assert np.allclose(w_inf, w_star, atol=1e-5)

    # Trajectory computation
    t, traj = flow.trajectory(t_max=10.0, n_points=100)
    assert len(t) == 100
    assert traj.shape == (100, 2)

    # Equivalence: for t = 1/lambda, check qualitative agreement
    lam = 1.0
    t_equiv = 1.0 / lam
    w_flow = flow.gradient_flow_point(t_equiv)
    w_decay = flow.weight_decay_point(lam)

    # Both must shrink w1 far more than w2
    assert (w_flow[0] / w_star[0]) < (w_flow[1] / w_star[1])
    assert (w_decay[0] / w_star[0]) < (w_decay[1] / w_star[1])


def test_double_descent_model():
    """Verify model-wise double descent properties (Nakkiran et al. 2019)."""
    model = DoubleDescentModel(max_width=64, interp_width=11)
    widths, train_err, test_err = model.get_curves()

    assert len(widths) == 64
    assert len(train_err) == 64
    assert len(test_err) == 64

    # Train error is zero at and after interpolation threshold
    assert train_err[model.interp_width - 1] == 0.0
    assert np.all(train_err[model.interp_width:] == 0.0)

    # Test error peaks at interpolation threshold
    peak_idx = model.interp_width - 1
    assert test_err[peak_idx] > test_err[peak_idx - 5]
    assert test_err[peak_idx] > test_err[-1]  # Overparameterized regime is better than peak!


def test_generate_all_figures():
    """Verify generation and saving of Figures 9.7, 9.8, 9.9, 9.10, and 9.11."""
    repo_root = Path(__file__).resolve().parent.parent
    dir_ch9 = repo_root / "9" / "result"
    dir_root = repo_root / "result"

    fig9_7 = generate_figure_9_7()
    fig9_8 = generate_figure_9_8()
    fig9_9 = generate_figure_9_9()
    fig9_10 = generate_figure_9_10()
    fig9_11 = generate_figure_9_11()

    assert fig9_7 is not None
    assert fig9_8 is not None
    assert fig9_9 is not None
    assert fig9_10 is not None
    assert fig9_11 is not None

    expected_files = [
        "fig_9_7_early_stopping_curves.png",
        "Figure_9_7.png",
        "fig_9_8_early_stopping_quadratic.png",
        "Figure_9_8.png",
        "fig_9_9_double_descent_resnet18.png",
        "Figure_9_9.png",
        "fig_9_10_epoch_double_descent.png",
        "Figure_9_10.png",
        "fig_9_11_sample_wise_non_monotonicity.png",
        "Figure_9_11.png",
    ]

    for fname in expected_files:
        p_ch9 = dir_ch9 / fname
        p_root = dir_root / fname
        assert p_ch9.exists(), f"Missing {p_ch9}"
        assert p_ch9.stat().st_size > 1000, f"File {p_ch9} is too small"
        assert p_root.exists(), f"Missing {p_root}"
        assert p_root.stat().st_size > 1000, f"File {p_root} is too small"
