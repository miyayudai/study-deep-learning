"""Tests for Chapter 7 Section 7.3: Convergence.

Tests cover:
- Convergence in Quadratic Error Surfaces (Eq 7.24 - 7.30)
- Section 7.3.1: Momentum, effective learning rate eta / (1 - mu), and Algorithm 7.3
- Section 7.3.2: Learning rate schedules (Linear, Power-law, Exponential)
- Section 7.3.3: Adaptive optimizers (AdaGrad, RMSProp, Adam Algorithm 7.4)
- Figures 7.3, 7.4, 7.5, 7.6 generation
"""

import os
import numpy as np
import pytest

from common.convergence import (
    ConvergenceTrajectory,
    ExponentialLRScheduler,
    LinearLRScheduler,
    PowerLawLRScheduler,
    adagrad_optimizer,
    adam_optimizer,
    condition_number,
    convergence_rate_dominant_axis,
    effective_momentum_learning_rate,
    eigen_distance_evolution,
    generate_figure_7_3,
    generate_figure_7_4,
    generate_figure_7_5,
    generate_figure_7_6,
    gradient_descent_momentum,
    max_stable_learning_rate,
    rmsprop_optimizer,
    sgd_momentum_algorithm_7_3,
)


def test_convergence_theoretical_analysis():
    """Verify max stable learning rate, condition number, and decay rate (Eq 7.29, 7.30)."""
    eigenvalues = [0.2, 4.0]
    
    # 1. Max stable learning rate: 2 / lambda_max = 2 / 4.0 = 0.5
    eta_max = max_stable_learning_rate(eigenvalues)
    assert np.isclose(eta_max, 0.5)
    
    # 2. Condition number: lambda_max / lambda_min = 4.0 / 0.2 = 20.0
    kappa = condition_number(eigenvalues)
    assert np.isclose(kappa, 20.0)
    
    # 3. Dominant axis convergence factor: 1 - 2 * lambda_min / lambda_max = 1 - 0.1 = 0.9 (Eq 7.30)
    rate = convergence_rate_dominant_axis(0.2, 4.0)
    assert np.isclose(rate, 0.9)
    
    # 4. Decoupled distance evolution (Eq 7.29)
    dist = eigen_distance_evolution(lambda_val=2.0, lr=0.1, steps=3, alpha_0=10.0)
    # (1 - 0.2) = 0.8 -> [10.0, 8.0, 6.4, 5.12]
    expected_dist = np.array([10.0, 8.0, 6.4, 5.12])
    assert np.allclose(dist, expected_dist)
    
    with pytest.raises(ValueError):
        max_stable_learning_rate([-1.0, 2.0])
    with pytest.raises(ValueError):
        condition_number([0.0, 2.0])


def test_effective_momentum_learning_rate():
    """Verify effective learning rate in low curvature: eta / (1 - mu) (Eq 7.33)."""
    lr = 0.01
    mu = 0.9
    eff_lr = effective_momentum_learning_rate(lr, mu)
    assert np.isclose(eff_lr, 0.1)
    
    with pytest.raises(ValueError):
        effective_momentum_learning_rate(0.01, 1.0)
    with pytest.raises(ValueError):
        effective_momentum_learning_rate(0.01, -0.1)


def test_gradient_descent_momentum_valley():
    """Verify that momentum accelerates progress along an elongated quadratic valley."""
    # Diagonal Hessian: lambda_1 = 0.05, lambda_2 = 2.0 (condition number = 40)
    A = np.diag([0.05, 2.0])
    w_star = np.array([0.0, 0.0])
    w_init = np.array([10.0, 1.0])
    
    def grad_fn(w: np.ndarray) -> np.ndarray:
        return A @ w
        
    def error_fn(w: np.ndarray) -> float:
        return 0.5 * float(w.T @ A @ w)
        
    # Maximum stable learning rate without momentum is 2 / 2.0 = 1.0; use safe lr = 0.4
    lr = 0.4
    
    # 1. Vanilla GD (momentum = 0.0)
    traj_vanilla = gradient_descent_momentum(
        w_init, grad_fn, error_fn=error_fn, lr=lr, momentum=0.0, max_steps=50
    )
    # 2. GD with Momentum (mu = 0.75)
    traj_momentum = gradient_descent_momentum(
        w_init, grad_fn, error_fn=error_fn, lr=lr, momentum=0.75, max_steps=50
    )
    
    # Momentum should achieve lower final error along the slow axis
    assert traj_momentum.errors[-1] < traj_vanilla.errors[-1]
    assert abs(traj_momentum.weights[-1, 0]) < abs(traj_vanilla.weights[-1, 0])


def test_nesterov_momentum():
    """Verify Nesterov accelerated gradient on 2D quadratic optimization."""
    A = np.diag([0.2, 1.5])
    w_init = np.array([5.0, -3.0])
    
    def grad_fn(w: np.ndarray) -> np.ndarray:
        return A @ w
        
    traj_nag = gradient_descent_momentum(
        w_init, grad_fn, lr=0.3, momentum=0.8, nesterov=True, max_steps=60
    )
    assert np.allclose(traj_nag.weights[-1], [0.0, 0.0], atol=1e-2)


def test_sgd_momentum_algorithm_7_3():
    """Verify SGD with Momentum (Algorithm 7.3) on synthetic regression dataset."""
    rng = np.random.default_rng(42)
    N = 100
    B = 10
    w_true = np.array([1.5, -2.0])
    X = rng.normal(size=(N, 2))
    y = X @ w_true + rng.normal(scale=0.05, size=N)
    
    def error_fn(w: np.ndarray) -> float:
        return 0.5 * float(np.mean((X @ w - y) ** 2))
        
    def grad_minibatch_fn(w: np.ndarray, batch_idx: np.ndarray) -> np.ndarray:
        X_b = X[batch_idx]
        y_b = y[batch_idx]
        return (X_b.T @ (X_b @ w - y_b)) / len(batch_idx)
        
    w_init = np.array([0.0, 0.0])
    traj = sgd_momentum_algorithm_7_3(
        w_init=w_init,
        grad_minibatch_fn=grad_minibatch_fn,
        n_samples=N,
        batch_size=B,
        error_fn=error_fn,
        lr=0.05,
        momentum=0.85,
        max_epochs=15,
        shuffle=True,
        random_state=42,
    )
    
    assert traj.errors[-1] < traj.errors[0]
    assert np.allclose(traj.weights[-1], w_true, atol=0.15)


def test_learning_rate_schedulers():
    """Verify Linear, Power-law, and Exponential learning rate decay formulas (Eq 7.36 - 7.38)."""
    # 1. Linear: eta(0) = 0.1, eta(10) = 0.01, K = 10
    sched_lin = LinearLRScheduler(eta_0=0.1, eta_K=0.01, K=10)
    assert np.isclose(sched_lin(0), 0.1)
    assert np.isclose(sched_lin(5), 0.055)
    assert np.isclose(sched_lin(10), 0.01)
    assert np.isclose(sched_lin(20), 0.01)  # clamped at eta_K
    
    # 2. Power-law: eta(0) = 0.1, s = 10, c = 1.0
    sched_pow = PowerLawLRScheduler(eta_0=0.1, s=10.0, c=1.0)
    assert np.isclose(sched_pow(0), 0.1)
    assert np.isclose(sched_pow(10), 0.05)  # 0.1 / (1 + 1)^1 = 0.05
    
    # 3. Exponential: eta(0) = 0.1, s = 10, c = 0.5
    sched_exp = ExponentialLRScheduler(eta_0=0.1, s=10.0, c=0.5)
    assert np.isclose(sched_exp(0), 0.1)
    assert np.isclose(sched_exp(10), 0.05)  # 0.1 * (0.5)^1 = 0.05


def test_adagrad_optimizer():
    """Verify AdaGrad optimization (Eq 7.39, 7.40)."""
    A = np.diag([2.0, 0.5])
    b = np.array([1.0, -1.0])
    w_star = np.array([0.5, -2.0])
    
    def grad_fn(w: np.ndarray) -> np.ndarray:
        return A @ w - b
        
    w_init = np.array([0.0, 0.0])
    traj = adagrad_optimizer(w_init, grad_fn, lr=1.0, max_steps=100)
    assert np.allclose(traj.weights[-1], w_star, atol=0.1)


def test_rmsprop_optimizer():
    """Verify RMSProp optimization (Eq 7.41, 7.42)."""
    A = np.diag([3.0, 0.5])
    b = np.array([3.0, 1.0])
    w_star = np.array([1.0, 2.0])
    
    def grad_fn(w: np.ndarray) -> np.ndarray:
        return A @ w - b
        
    w_init = np.array([0.0, 0.0])
    traj = rmsprop_optimizer(w_init, grad_fn, lr=0.1, beta=0.9, max_steps=80)
    assert np.allclose(traj.weights[-1], w_star, atol=0.1)


def test_adam_optimizer_algorithm_7_4():
    """Verify Adam optimizer (Algorithm 7.4, Eq 7.43 - 7.47)."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([2.0, 5.0])
    w_star = np.linalg.solve(A, b)
    
    def grad_fn(w: np.ndarray) -> np.ndarray:
        return A @ w - b
        
    w_init = np.array([-2.0, 4.0])
    traj = adam_optimizer(w_init, grad_fn, lr=0.1, beta1=0.9, beta2=0.999, max_steps=120)
    assert np.allclose(traj.weights[-1], w_star, atol=0.05)


def test_figure_generators_7_3_to_7_6():
    """Verify that Figures 7.3, 7.4, 7.5, and 7.6 are successfully generated."""
    p3, r3 = generate_figure_7_3()
    assert os.path.exists(p3)
    
    p4, r4 = generate_figure_7_4()
    assert os.path.exists(p4)
    
    p5, r5 = generate_figure_7_5()
    assert os.path.exists(p5)
    
    p6, r6 = generate_figure_7_6()
    assert os.path.exists(p6)
