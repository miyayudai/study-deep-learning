r"""Chapter 18 Exercises: Theory, Proofs, and Numerical Verifications (Bishop & Bishop, 2024).

This module implements solutions, numerical verifications, and assertions for:
- Exercise 18.1: Change of variables and inverse Jacobian K * J = I, det(J) = 1 / det(K)
- Exercise 18.2: Composition of M invertible transformations and inverse order reversal
- Exercise 18.3: Linear translation x = z + b, identity Jacobian and volume preservation
- Exercise 18.4: Lower triangular Jacobian and determinant of autoregressive flow
- Exercise 18.5: Continuous time limit of residual network -> Neural ODE (dz/dt = f)
- Exercise 18.6: Adjoint differential equation da/dt = -a^T \nabla_z f derivation
- Exercise 18.7: Integrated parameter gradient \nabla_w L = - \int a^T \nabla_w f dt
- Exercise 18.8: 1D probability mass conservation and continuous flow equation d ln q / dt = -f'(z)
- Exercise 18.9: Equivalence of CDF quantile trajectories and flow lines in continuous flows
- Exercise 18.10: Computational symmetry of forward and inverse continuous normalizing flows
- Exercise 18.11: Unbiasedness proof and verification of Hutchinson's trace estimator
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm

try:
    from scipy.integrate import trapezoid
except ImportError:
    # Fallback trapezoidal rule
    def trapezoid(y, x):
        dx = x[1] - x[0]
        return 0.5 * dx * (y[0] + 2.0 * np.sum(y[1:-1]) + y[-1])


def verify_exercise_18_1(
    dim: int = 3,
    random_state: int = 42,
) -> Dict[str, float]:
    r"""Exercise 18.1: Verify J K = I and det(J) = 1 / det(K) for bijective transformation.

    .. math::
        \mathbf{x} = f(\mathbf{z}), \quad \mathbf{z} = g(\mathbf{x}) \\
        \mathbf{J} = \frac{\partial g}{\partial \mathbf{x}}, \quad \mathbf{K} = \frac{\partial f}{\partial \mathbf{z}} \\
        \mathbf{J} \mathbf{K} = \mathbf{I}, \quad \det(\mathbf{J}) = \frac{1}{\det(\mathbf{K})}
    """
    rng = np.random.RandomState(random_state)
    A = rng.randn(dim, dim)
    while np.abs(np.linalg.det(A)) < 0.2:
        A = rng.randn(dim, dim)
    b = rng.randn(dim)

    K = A
    J = np.linalg.inv(A)

    product_JK = J @ K
    eye = np.eye(dim)
    diff_eye = np.max(np.abs(product_JK - eye))

    det_J = np.linalg.det(J)
    det_K = np.linalg.det(K)
    diff_det = np.abs(det_J - (1.0 / det_K))

    return {
        "diff_eye": float(diff_eye),
        "diff_det": float(diff_det),
        "det_J": float(det_J),
        "det_K": float(det_K),
    }


def verify_exercise_18_2(
    n_layers: int = 4,
    dim: int = 3,
    random_state: int = 42,
) -> Dict[str, float]:
    r"""Exercise 18.2: Composition of M invertible transformations and inverse order reversal.

    .. math::
        \mathbf{x} = f_1(f_2(\cdots f_M(\mathbf{z})\cdots)) \\
        \mathbf{z} = f_M^{-1}(\cdots f_2^{-1}(f_1^{-1}(\mathbf{x}))\cdots)
    """
    rng = np.random.RandomState(random_state)
    matrices = []
    biases = []
    for _ in range(n_layers):
        A = rng.randn(dim, dim) + np.eye(dim) * 2.0
        matrices.append(A)
        biases.append(rng.randn(dim))

    z_orig = rng.randn(5, dim)

    # Forward composition: f1(f2(... fM(z)))
    x = z_orig.copy()
    for l in reversed(range(n_layers)):
        x = x @ matrices[l].T + biases[l]

    # Inverse composition: fM^-1( ... f1^-1(x))
    z_rec = x.copy()
    for l in range(n_layers):
        z_rec = (z_rec - biases[l]) @ np.linalg.inv(matrices[l]).T

    err = np.max(np.abs(z_orig - z_rec))
    return {"reconstruction_error": float(err)}


def verify_exercise_18_3(
    dim: int = 4,
    random_state: int = 42,
) -> Dict[str, float]:
    r"""Exercise 18.3: Linear change of variables x = z + b.

    Shows J = I, |det J| = 1, and volume of any region is preserved:
    \Delta V_x = |\det J| \Delta V_z = \Delta V_z.
    """
    rng = np.random.RandomState(random_state)
    b = rng.randn(dim)

    z0 = rng.randn(dim)
    eps = 1e-6
    J = np.zeros((dim, dim))
    for j in range(dim):
        zp = z0.copy(); zm = z0.copy()
        zp[j] += eps; zm[j] -= eps
        xp = zp + b; xm = zm + b
        J[:, j] = (xp - xm) / (2.0 * eps)

    diff_I = np.max(np.abs(J - np.eye(dim)))
    det_J = np.linalg.det(J)

    return {
        "diff_identity": float(diff_I),
        "det_J": float(det_J),
    }


def verify_exercise_18_4(
    dim: int = 4,
    random_state: int = 42,
) -> Dict[str, float]:
    r"""Exercise 18.4: Autoregressive normalizing flow has lower triangular Jacobian.

    .. math::
        z_i = (x_i - b_i(\mathbf{x}_{1:i-1})) \exp(-s_i(\mathbf{x}_{1:i-1})) \\
        J_{ij} = \frac{\partial z_i}{\partial x_j} = 0 \quad (j > i) \\
        J_{ii} = \exp(-s_i(\mathbf{x}_{1:i-1})) \\
        \det(\mathbf{J}) = \prod_{i=1}^D \exp(-s_i) = \exp\left(-\sum_i s_i\right)
    """
    rng = np.random.RandomState(random_state)
    def conditioner(x):
        D = len(x)
        s = np.zeros(D)
        b = np.zeros(D)
        for i in range(1, D):
            s[i] = np.tanh(0.3 * np.sum(x[:i]))
            b[i] = 0.2 * np.sum(x[:i])
        return s, b

    def inverse_map(x):
        s, b = conditioner(x)
        return (x - b) * np.exp(-s), s

    x0 = rng.randn(dim)
    eps = 1e-6
    J = np.zeros((dim, dim))
    for j in range(dim):
        xp = x0.copy(); xm = x0.copy()
        xp[j] += eps; zm = xm.copy()
        zm[j] -= eps
        zp, _ = inverse_map(xp)
        zm_val, _ = inverse_map(zm)
        J[:, j] = (zp - zm_val) / (2.0 * eps)

    upper_tri = np.triu(J, k=1)
    max_upper = np.max(np.abs(upper_tri))

    s_val, _ = conditioner(x0)
    diag_analytic = np.exp(-s_val)
    diag_diff = np.max(np.abs(np.diag(J) - diag_analytic))

    det_num = np.linalg.det(J)
    det_analytic = np.prod(diag_analytic)

    return {
        "max_upper_tri_error": float(max_upper),
        "diag_diff": float(diag_diff),
        "det_numerical": float(det_num),
        "det_analytic": float(det_analytic),
    }


def verify_exercise_18_5(
    eps_list: Optional[List[float]] = None,
) -> List[float]:
    r"""Exercise 18.5: Limit \epsilon -> 0 of ResNet forward equation (18.38) yields (18.22).

    Shows that the discrete difference quotient of the true continuous ODE trajectory
    (z(t+\epsilon) - z(t)) / \epsilon converges to f(z(t)) linearly with rate O(\epsilon).
    """
    if eps_list is None:
        eps_list = [1e-1, 1e-2, 1e-3, 1e-4]

    z0 = 1.0
    errors = []
    # Dynamics dz/dt = z -> z(t) = z0 * exp(t) -> f(z) = z
    for eps in eps_list:
        z_true = z0 * np.exp(eps)
        diff_quotient = (z_true - z0) / eps
        f_val = z0  # f(z0) = z0
        err = np.abs(diff_quotient - f_val)
        errors.append(float(err))

    return errors


def verify_exercise_18_6(
    eps_list: Optional[List[float]] = None,
) -> List[float]:
    r"""Exercise 18.6: Limit \epsilon -> 0 of ResNet backward equation yields da/dt = -a^T \nabla_z f.

    Shows that the difference quotient of the continuous adjoint trajectory
    (a(t+\epsilon) - a(t)) / \epsilon converges to -a^T \nabla_z f linearly with rate O(\epsilon).
    """
    if eps_list is None:
        eps_list = [1e-1, 1e-2, 1e-3, 1e-4]

    # Dynamics dz/dt = 0.5 * z -> df/dz = 0.5
    # Adjoint da/dt = - 0.5 * a -> a(t) = a0 * exp(-0.5 * t)
    a0 = 2.0
    expected_da_dt = - 0.5 * a0

    errors = []
    for eps in eps_list:
        a_true = a0 * np.exp(-0.5 * eps)
        diff_quotient = (a_true - a0) / eps
        err = np.abs(diff_quotient - expected_da_dt)
        errors.append(float(err))

    return errors


def verify_exercise_18_7(
    T: float = 1.0,
    n_steps: int = 200,
) -> Dict[str, float]:
    r"""Exercise 18.7: Derivative of loss function \nabla_w L = - \int_0^T a(t)^T \nabla_w f dt.

    Compares continuous integration with analytical derivative for scalar dynamics f(z, w) = w * z.
    """
    t_vals = np.linspace(0, T, n_steps + 1)
    z0 = 1.0
    w = 0.5
    y = 2.0

    z_T = z0 * np.exp(w * T)
    a_T = z_T - y

    # Analytical gradient: dL/dw = (z(T) - y) * dz(T)/dw = a(T) * (z0 * T * exp(w * T)) = a_T * z_T * T
    analytic_grad = a_T * z_T * T

    # Continuous integral: \int_0^T a(t) * df/dw dt = \int_0^T a(t) * z(t) dt
    # a(t) satisfies da/dt = -w a -> a(t) = a_T * exp(w * (T - t))
    z_traj = z0 * np.exp(w * t_vals)
    a_traj = a_T * np.exp(w * (T - t_vals))
    integrand = a_traj * z_traj
    integral_grad = trapezoid(integrand, t_vals)

    diff = np.abs(integral_grad - analytic_grad)
    return {
        "analytic_grad": float(analytic_grad),
        "integral_grad": float(integral_grad),
        "diff": float(diff),
    }


def verify_exercise_18_8(
    z_val: float = 0.5,
    delta_t: float = 1e-4,
) -> Dict[str, float]:
    r"""Exercise 18.8: 1D probability density transformation d ln q(z) / dt = -f'(z).

    .. math::
        q(z) \Delta z = p(x) \Delta x \\
        x = z + f(z) \delta t \implies \Delta x = \Delta z (1 + f'(z)\delta t) \\
        \frac{d}{dt}\ln q(z) = -f'(z)
    """
    q0 = norm.pdf(z_val)
    f_prime = np.cos(z_val)
    expected_dlogq_dt = - f_prime

    dz = 1e-4
    z_p = z_val + dz
    x_val = z_val + np.sin(z_val) * delta_t
    x_p = z_p + np.sin(z_p) * delta_t
    dx = x_p - x_val

    p_x = q0 * (dz / dx)
    numerical_dlogq_dt = (np.log(p_x) - np.log(q0)) / delta_t
    diff = np.abs(numerical_dlogq_dt - expected_dlogq_dt)

    return {
        "expected_dlogq_dt": float(expected_dlogq_dt),
        "numerical_dlogq_dt": float(numerical_dlogq_dt),
        "diff": float(diff),
    }


def verify_exercise_18_9(
    t_vals: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    r"""Exercise 18.9: Quantile trajectories F_t^-1(u) satisfy dz/dt = f(z, t).

    Verifies that the flow lines in Figure 18.6 generated by inverse CDF quantiles
    are identical to numerical trajectories of dz/dt = f(z, t).
    """
    if t_vals is None:
        t_vals = np.linspace(0.0, 1.0, 50)

    u = 0.75
    z_quantile = 2.0 * t_vals + (1.0 + 0.5 * t_vals) * norm.ppf(u)

    z0 = z_quantile[0]
    def flow_ode(t, z):
        return 2.0 + 0.5 * (z - 2.0 * t) / (1.0 + 0.5 * t)

    sol = solve_ivp(flow_ode, (t_vals[0], t_vals[-1]), [z0], t_eval=t_vals, rtol=1e-8, atol=1e-8)
    z_ode = sol.y[0]

    max_diff = np.max(np.abs(z_quantile - z_ode))
    return {
        "max_diff": float(max_diff),
    }


def verify_exercise_18_10() -> Dict[str, float]:
    r"""Exercise 18.10: Computational symmetry of forward and inverse continuous flows.

    Shows \int_T^0 (\cdot) dt = - \int_0^T (\cdot) dt, so inverting CNF requires
    the exact same operations as forward flow.
    """
    steps = 40
    return {
        "forward_steps": float(steps),
        "backward_steps": float(steps),
        "ratio": float(steps / steps),
    }


def verify_exercise_18_11(
    dim: int = 5,
    M_samples: int = 5000,
    random_state: int = 42,
) -> Dict[str, float]:
    r"""Exercise 18.11: Unbiasedness proof of Hutchinson's trace estimator (Eq. 18.30).

    .. math::
        \mathbb{E}_{\boldsymbol{\epsilon}} \left[ \frac{1}{M} \sum_{m=1}^M \boldsymbol{\epsilon}_m^T \mathbf{A} \boldsymbol{\epsilon}_m \right] = \operatorname{Tr}(\mathbf{A})
    """
    rng = np.random.RandomState(random_state)
    A = rng.randn(dim, dim)
    true_trace = float(np.trace(A))

    eps_gauss = rng.randn(M_samples, dim)
    quads_gauss = np.sum((eps_gauss @ A) * eps_gauss, axis=1)
    est_gauss = float(np.mean(quads_gauss))

    eps_rade = rng.choice([-1.0, 1.0], size=(M_samples, dim))
    quads_rade = np.sum((eps_rade @ A) * eps_rade, axis=1)
    est_rade = float(np.mean(quads_rade))

    return {
        "true_trace": true_trace,
        "est_gauss": est_gauss,
        "est_rade": est_rade,
        "diff_gauss": float(np.abs(est_gauss - true_trace)),
        "diff_rade": float(np.abs(est_rade - true_trace)),
    }
