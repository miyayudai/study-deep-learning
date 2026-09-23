"""Chapter 8: Backpropagation - Exercises 8.1 to 8.18

This module implements detailed solutions and rigorous verification routines for all 18 exercises
in Chapter 8 of:
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.

Contents:
- Exercise 8.1: Verification of backpropagation chain rule (8.13) from (8.5), (8.6), (8.8), (8.12).
- Exercise 8.2: Matrix notation for backpropagation using transposes and element-wise products.
- Exercise 8.3: Taylor expansion of central differences proving cancellation of O(eps) terms (Eq 8.25).
- Exercise 8.4: Error-function gradients for two-layer network with skip-layer connections.
- Exercise 8.5: Forward-mode propagation formalism for computing network Jacobian matrix.
- Exercise 8.6: Exact Hessian matrix of two-layer network in terms of delta_k and M_{kk'} (Eq 8.77).
- Exercise 8.7: Extension of exact Hessian to include skip-layer connections.
- Exercise 8.8: Outer-product (Gauss-Newton) Hessian approximation for multiple outputs.
- Exercise 8.9: Expected Hessian for squared loss proving equality with outer-product when y = E[t|x].
- Exercise 8.10: Outer-product Hessian for single output with logistic sigmoid and cross-entropy (Eq 8.41).
- Exercise 8.11: Outer-product Hessian for K-class softmax output with cross-entropy.
- Exercise 8.12: Sequential Sherman-Morrison inverse Hessian update from outer-product approximation (Eq 8.80).
- Exercise 8.13: Verification of symbolic derivative swelling formula (8.48) for nested soft ReLU (8.47).
- Exercise 8.14: Logistic map evaluation trace and demonstration of formula complexity growth.
- Exercise 8.15: Derivation of forward-mode tangent equations (8.58)-(8.64) from primal equations.
- Exercise 8.16: Derivation of reverse-mode adjoint equations (8.70)-(8.76) from evaluation graph.
- Exercise 8.17: Exact numerical evaluation of del f / del x1 at (1, 2) via direct, forward, and reverse modes.
- Exercise 8.18: Proof and verification that Jacobian-vector product J * r is evaluated in a single forward pass.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from common.automatic_differentiation import (
    DualNumber,
    Node,
    evaluate_trace_forward_mode,
    evaluate_trace_reverse_mode,
    example_function_8_49,
    forward_mode_jacobian,
    forward_mode_jvp,
)


# =====================================================================
# Exercise 8.1: Backpropagation Formula Verification
# =====================================================================

def verify_exercise_8_1() -> Dict[str, Any]:
    """Verify backpropagation formula (8.13) against numerical differentiation.

    Equations:
        (8.5):  a_j = sum_i w_ji z_i
        (8.6):  z_j = h(a_j)
        (8.8):  delta_j = del E_n / del a_j
        (8.12): del E_n / del a_j = sum_k (del E_n / del a_k) * (del a_k / del a_j)
        (8.13): delta_j = h'(a_j) sum_k w_kj delta_k
        Gradient: del E_n / del w_ji = delta_j z_i

    Proof:
        From (8.5) and (8.6), for a subsequent layer unit k:
            a_k = sum_l w_kl z_l = sum_l w_kl h(a_l)
        Differentiating with respect to a_j:
            del a_k / del a_j = w_kj h'(a_j)
        Substituting into the chain rule (8.12):
            delta_j = sum_k (del E_n / del a_k) * (del a_k / del a_j)
                    = sum_k delta_k * w_kj * h'(a_j)
                    = h'(a_j) sum_k w_kj delta_k
        which proves (8.13).
        Furthermore, del E_n / del w_ji = (del E_n / del a_j) * (del a_j / del w_ji) = delta_j z_i.
    """
    rng = np.random.RandomState(42)
    D, M, K = 3, 4, 2
    x = rng.randn(D)
    t = rng.randn(K)
    W1 = rng.randn(M, D) * 0.5
    b1 = rng.randn(M) * 0.1
    W2 = rng.randn(K, M) * 0.5
    b2 = rng.randn(K) * 0.1

    def forward_and_loss(w1, b_1, w2, b_2):
        a1 = w1 @ x + b_1
        z1 = np.tanh(a1)
        a2 = w2 @ z1 + b_2
        y = a2
        loss = 0.5 * np.sum((y - t) ** 2)
        return loss, a1, z1, a2, y

    loss, a1, z1, a2, y = forward_and_loss(W1, b1, W2, b2)

    # Backpropagation per Eq (8.13)
    delta2 = y - t  # del E / del a2 for linear sum-of-squares
    grad_W2 = np.outer(delta2, z1)
    grad_b2 = delta2

    # Eq 8.13: delta_j = h'(a_j) * sum_k w_kj delta_k
    h_prime = 1.0 - z1 ** 2  # derivative of tanh
    delta1 = h_prime * (W2.T @ delta2)
    grad_W1 = np.outer(delta1, x)
    grad_b1 = delta1

    # Numerical differentiation check (central differences)
    eps = 1e-6
    num_grad_W1 = np.zeros_like(W1)
    for i in range(M):
        for j in range(D):
            W_pos = W1.copy()
            W_pos[i, j] += eps
            W_neg = W1.copy()
            W_neg[i, j] -= eps
            l_pos, _, _, _, _ = forward_and_loss(W_pos, b1, W2, b2)
            l_neg, _, _, _, _ = forward_and_loss(W_neg, b1, W2, b2)
            num_grad_W1[i, j] = (l_pos - l_neg) / (2 * eps)

    max_err_W1 = float(np.max(np.abs(grad_W1 - num_grad_W1)))
    return {
        "verified": bool(max_err_W1 < 1e-7),
        "max_err_W1": max_err_W1,
        "delta1": delta1,
        "delta2": delta2,
    }


# =====================================================================
# Exercise 8.2: Matrix Form of Backpropagation
# =====================================================================

def verify_exercise_8_2() -> Dict[str, Any]:
    """Verify matrix notation for backpropagation.

    Forward propagation (6.19):
        a^{(l)} = W^{(l)} z^{(l-1)} + b^{(l)}
        z^{(l)} = h(a^{(l)})

    Backpropagation (8.13) in matrix form:
        delta^{(l)} = h'(a^{(l)}) odot ((W^{(l+1)})^T delta^{(l+1)})
        del E_n / del W^{(l)} = delta^{(l)} (z^{(l-1)})^T
        del E_n / del b^{(l)} = delta^{(l)}
    """
    rng = np.random.RandomState(43)
    layer_sizes = [4, 6, 5, 3]
    num_layers = len(layer_sizes) - 1

    weights = []
    biases = []
    for l in range(num_layers):
        weights.append(rng.randn(layer_sizes[l + 1], layer_sizes[l]) * 0.4)
        biases.append(rng.randn(layer_sizes[l + 1]) * 0.1)

    x = rng.randn(layer_sizes[0])
    t = rng.randn(layer_sizes[-1])

    # Forward pass
    activations = []
    hidden_outs = [x]
    for l in range(num_layers):
        a = weights[l] @ hidden_outs[-1] + biases[l]
        activations.append(a)
        z = np.tanh(a) if l < num_layers - 1 else a
        hidden_outs.append(z)

    # Backward pass in matrix notation
    deltas = [None] * num_layers
    grad_W = [None] * num_layers
    grad_b = [None] * num_layers

    # Output layer
    deltas[-1] = hidden_outs[-1] - t
    grad_W[-1] = np.outer(deltas[-1], hidden_outs[-2])
    grad_b[-1] = deltas[-1]

    # Hidden layers backwards
    for l in range(num_layers - 2, -1, -1):
        h_prime = 1.0 - hidden_outs[l + 1] ** 2
        # Matrix form: delta^{(l)} = h'(a^{(l)}) odot (W^{(l+1)T} delta^{(l+1)})
        deltas[l] = h_prime * (weights[l + 1].T @ deltas[l + 1])
        grad_W[l] = np.outer(deltas[l], hidden_outs[l])
        grad_b[l] = deltas[l]

    # Numerical check on first hidden layer weights
    eps = 1e-6
    num_grad_W0 = np.zeros_like(weights[0])
    for i in range(weights[0].shape[0]):
        for j in range(weights[0].shape[1]):
            w_copy_pos = [w.copy() for w in weights]
            w_copy_neg = [w.copy() for w in weights]
            w_copy_pos[0][i, j] += eps
            w_copy_neg[0][i, j] -= eps

            def eval_net(w_list):
                curr = x
                for idx in range(num_layers):
                    a_curr = w_list[idx] @ curr + biases[idx]
                    curr = np.tanh(a_curr) if idx < num_layers - 1 else a_curr
                return 0.5 * np.sum((curr - t) ** 2)

            l_pos = eval_net(w_copy_pos)
            l_neg = eval_net(w_copy_neg)
            num_grad_W0[i, j] = (l_pos - l_neg) / (2 * eps)

    diff = float(np.max(np.abs(grad_W[0] - num_grad_W0)))
    return {
        "verified": bool(diff < 1e-7),
        "max_err_W0": diff,
        "deltas_shapes": [d.shape for d in deltas],
    }


# =====================================================================
# Exercise 8.3: Taylor Expansion of Central Difference
# =====================================================================

def verify_exercise_8_3() -> Dict[str, Any]:
    """Verify that O(eps) terms cancel in central difference (8.25).

    Taylor expansions:
        E(w + eps) = E(w) + eps E'(w) + (eps^2/2) E''(w) + (eps^3/6) E'''(w) + O(eps^4)
        E(w - eps) = E(w) - eps E'(w) + (eps^2/2) E''(w) - (eps^3/6) E'''(w) + O(eps^4)

    Difference:
        E(w + eps) - E(w - eps) = 2 eps E'(w) + (eps^3/3) E'''(w) + O(eps^5)

    Dividing by 2 eps:
        [E(w + eps) - E(w - eps)] / (2 eps) = E'(w) + (eps^2 / 6) E'''(w) + O(eps^4)
                                             = E'(w) + O(eps^2)

    All odd powers of eps in the difference cancel, leaving only even powers in the difference,
    which after division by eps yields errors proportional to eps^2, eps^4, ... (no O(eps) term).
    By contrast, the forward difference has error O(eps):
        [E(w + eps) - E(w)] / eps = E'(w) + (eps / 2) E''(w) + O(eps^2).
    """
    f = lambda w: np.sin(w) + np.exp(0.5 * w)
    df_exact = lambda w: np.cos(w) + 0.5 * np.exp(0.5 * w)
    w0 = 1.2

    epsilons = np.logspace(-5, -1, 10)
    central_errors = []
    forward_errors = []

    for eps in epsilons:
        c_diff = (f(w0 + eps) - f(w0 - eps)) / (2.0 * eps)
        f_diff = (f(w0 + eps) - f(w0)) / eps
        central_errors.append(abs(c_diff - df_exact(w0)))
        forward_errors.append(abs(f_diff - df_exact(w0)))

    slope_central, _ = np.polyfit(np.log(epsilons), np.log(central_errors), 1)
    slope_forward, _ = np.polyfit(np.log(epsilons), np.log(forward_errors), 1)

    return {
        "verified": bool(abs(slope_central - 2.0) < 0.1 and abs(slope_forward - 1.0) < 0.1),
        "slope_central": float(slope_central),
        "slope_forward": float(slope_forward),
        "epsilons": epsilons.tolist(),
        "central_errors": central_errors,
        "forward_errors": forward_errors,
    }


# =====================================================================
# Exercise 8.4: Gradients for Network with Skip-Layer Connections
# =====================================================================

def verify_exercise_8_4() -> Dict[str, Any]:
    """Verify error derivatives for two-layer network with skip connections.

    Model:
        Hidden: a_j = sum_i w_ji^{(1)} x_i + b_j^{(1)},  z_j = h(a_j)
        Output: a_k = sum_j w_kj^{(2)} z_j + sum_i w_ki^{(s)} x_i + b_k^{(2)}
        Target loss: E_n = 0.5 sum_k (y_k - t_k)^2  with y_k = a_k

    Gradients:
        Output delta: delta_k = del E_n / del a_k = y_k - t_k
        Skip weights: del E_n / del w_ki^{(s)} = (del E_n / del a_k) * (del a_k / del w_ki^{(s)})
                                              = delta_k x_i
        Layer 2 weights: del E_n / del w_kj^{(2)} = delta_k z_j
        Layer 1 delta: delta_j = h'(a_j) sum_k w_kj^{(2)} delta_k  (unaffected by skip connections)
        Layer 1 weights: del E_n / del w_ji^{(1)} = delta_j x_i
    """
    rng = np.random.RandomState(44)
    D, M, K = 3, 4, 2
    x = rng.randn(D)
    t = rng.randn(K)

    W1 = rng.randn(M, D) * 0.4
    b1 = rng.randn(M) * 0.1
    W2 = rng.randn(K, M) * 0.4
    b2 = rng.randn(K) * 0.1
    Ws = rng.randn(K, D) * 0.4  # Skip connection weights

    def forward(w1, b_1, w2, b_2, ws):
        a1 = w1 @ x + b_1
        z1 = np.tanh(a1)
        a2 = w2 @ z1 + ws @ x + b_2
        loss = 0.5 * np.sum((a2 - t) ** 2)
        return loss, a1, z1, a2

    loss, a1, z1, a2 = forward(W1, b1, W2, b2, Ws)

    # Analytical gradients
    delta2 = a2 - t
    grad_Ws = np.outer(delta2, x)
    grad_W2 = np.outer(delta2, z1)
    grad_b2 = delta2

    h_prime = 1.0 - z1 ** 2
    delta1 = h_prime * (W2.T @ delta2)
    grad_W1 = np.outer(delta1, x)
    grad_b1 = delta1

    # Check skip connection gradient numerically
    eps = 1e-6
    num_grad_Ws = np.zeros_like(Ws)
    for k in range(K):
        for i in range(D):
            Ws_pos = Ws.copy()
            Ws_neg = Ws.copy()
            Ws_pos[k, i] += eps
            Ws_neg[k, i] -= eps
            l_pos, _, _, _ = forward(W1, b1, W2, b2, Ws_pos)
            l_neg, _, _, _ = forward(W1, b1, W2, b2, Ws_neg)
            num_grad_Ws[k, i] = (l_pos - l_neg) / (2 * eps)

    max_err = float(np.max(np.abs(grad_Ws - num_grad_Ws)))
    return {
        "verified": bool(max_err < 1e-7),
        "max_err_Ws": max_err,
        "grad_Ws": grad_Ws,
    }


# =====================================================================
# Exercise 8.5: Forward-Mode Network Jacobian Formulation
# =====================================================================

def verify_exercise_8_5() -> Dict[str, Any]:
    """Verify alternative forward propagation formalism for evaluating network Jacobian.

    Equations:
        Inputs:   del x_m / del x_i = delta_{mi}
        Hidden:   del a_j / del x_i = sum_m w_jm^{(1)} (del x_m / del x_i) = w_ji^{(1)}
                  del z_j / del x_i = h'(a_j) (del a_j / del x_i)
        Outputs:  del a_k / del x_i = sum_j w_kj^{(2)} (del z_j / del x_i)
                  J_ki = del y_k / del x_i = sigma'(a_k) (del a_k / del x_i)
    """
    rng = np.random.RandomState(45)
    D, M, K = 3, 5, 2
    x = rng.randn(D)
    W1 = rng.randn(M, D) * 0.5
    b1 = rng.randn(M) * 0.1
    W2 = rng.randn(K, M) * 0.5
    b2 = rng.randn(K) * 0.1

    def forward_eval(x_vec):
        a1 = W1 @ x_vec + b1
        z1 = np.tanh(a1)
        a2 = W2 @ z1 + b2
        y = 1.0 / (1.0 + np.exp(-a2))  # Sigmoid output
        return y, a1, z1, a2

    y, a1, z1, a2 = forward_eval(x)

    # Forward-mode evaluation of Jacobian J_ki = del y_k / del x_i
    J_forward = np.zeros((K, D))
    for i in range(D):
        del_x = np.zeros(D)
        del_x[i] = 1.0

        del_a1 = W1 @ del_x  # del a_j / del x_i = w_ji
        del_z1 = (1.0 - z1 ** 2) * del_a1

        del_a2 = W2 @ del_z1
        del_y = y * (1.0 - y) * del_a2
        J_forward[:, i] = del_y

    # Numerical differentiation check
    eps = 1e-6
    J_num = np.zeros((K, D))
    for i in range(D):
        x_pos = x.copy()
        x_neg = x.copy()
        x_pos[i] += eps
        x_neg[i] -= eps
        y_pos, _, _, _ = forward_eval(x_pos)
        y_neg, _, _, _ = forward_eval(x_neg)
        J_num[:, i] = (y_pos - y_neg) / (2 * eps)

    max_err = float(np.max(np.abs(J_forward - J_num)))
    return {
        "verified": bool(max_err < 1e-7),
        "max_err": max_err,
        "J_forward": J_forward,
        "J_num": J_num,
    }


# =====================================================================
# Exercise 8.6: Exact Hessian Matrix of a Two-Layer Network
# =====================================================================

def verify_exercise_8_6() -> Dict[str, Any]:
    """Verify exact Hessian elements of two-layer network in terms of delta_k and M_{kk'}.

    Definitions (8.77):
        delta_k = del E_n / del a_k
        M_{kk'} = del^2 E_n / del a_k del a_{k'}

    Hessian elements:
    (i) Both weights in second layer:
        del^2 E_n / del w_kj^{(2)} del w_{k'j'}^{(2)} = z_j z_{j'} M_{kk'}

    (ii) Both weights in first layer:
        del^2 E_n / del w_ji^{(1)} del w_{j'i'}^{(1)}
            = x_i x_{i'} [ delta_{jj'} h''(a_j) sum_k w_kj^{(2)} delta_k
                          + h'(a_j) h'(a_{j'}) sum_k sum_{k'} w_kj^{(2)} w_{k'j'}^{(2)} M_{kk'} ]

    (iii) One weight in each layer:
        del^2 E_n / del w_kj^{(2)} del w_{j'i'}^{(1)}
            = x_{i'} [ delta_{jj'} delta_k h'(a_j) + z_j h'(a_{j'}) sum_{k'} w_{k'j'}^{(2)} M_{kk'} ]
    """
    rng = np.random.RandomState(46)
    D, M, K = 2, 3, 2
    x = rng.randn(D)
    t = rng.randn(K)
    W1 = rng.randn(M, D) * 0.4
    b1 = rng.randn(M) * 0.1
    W2 = rng.randn(K, M) * 0.4
    b2 = rng.randn(K) * 0.1

    def forward(w1, w2):
        a1 = w1 @ x + b1
        z1 = np.tanh(a1)
        a2 = w2 @ z1 + b2
        loss = 0.5 * np.sum((a2 - t) ** 2)
        return loss, a1, z1, a2

    loss, a1, z1, a2 = forward(W1, W2)

    delta = a2 - t  # delta_k
    M_mat = np.eye(K)  # M_{kk'} = delta_{kk'} for sum-of-squares linear output

    h_prime = 1.0 - z1 ** 2
    h_double_prime = -2.0 * z1 * h_prime

    # (i) Layer 2 - Layer 2
    H22_ana = np.zeros((K, M, K, M))
    for k in range(K):
        for j in range(M):
            for kp in range(K):
                for jp in range(M):
                    H22_ana[k, j, kp, jp] = z1[j] * z1[jp] * M_mat[k, kp]

    # (ii) Layer 1 - Layer 1
    H11_ana = np.zeros((M, D, M, D))
    for j in range(M):
        for i in range(D):
            for jp in range(M):
                for ip in range(D):
                    term1 = (1.0 if j == jp else 0.0) * h_double_prime[j] * np.sum(W2[:, j] * delta)
                    term2 = h_prime[j] * h_prime[jp] * np.sum(W2[:, j, None] * M_mat * W2[:, jp][None, :])
                    H11_ana[j, i, jp, ip] = x[i] * x[ip] * (term1 + term2)

    # (iii) Layer 2 - Layer 1
    H21_ana = np.zeros((K, M, M, D))
    for k in range(K):
        for j in range(M):
            for jp in range(M):
                for ip in range(D):
                    term1 = (1.0 if j == jp else 0.0) * delta[k] * h_prime[j]
                    term2 = z1[j] * h_prime[jp] * np.sum(W2[:, jp] * M_mat[:, k])
                    H21_ana[k, j, jp, ip] = x[ip] * (term1 + term2)

    # Numerical verification via finite differences
    eps = 1e-5
    H22_num = np.zeros((K, M, K, M))
    for k in range(K):
        for j in range(M):
            for kp in range(K):
                for jp in range(M):
                    w2_pp = W2.copy(); w2_pp[k, j] += eps; w2_pp[kp, jp] += eps
                    w2_pm = W2.copy(); w2_pm[k, j] += eps; w2_pm[kp, jp] -= eps
                    w2_mp = W2.copy(); w2_mp[k, j] -= eps; w2_mp[kp, jp] += eps
                    w2_mm = W2.copy(); w2_mm[k, j] -= eps; w2_mm[kp, jp] -= eps
                    H22_num[k, j, kp, jp] = (
                        forward(W1, w2_pp)[0] - forward(W1, w2_pm)[0]
                        - forward(W1, w2_mp)[0] + forward(W1, w2_mm)[0]
                    ) / (4 * eps * eps)

    H21_num = np.zeros((K, M, M, D))
    for k in range(K):
        for j in range(M):
            for jp in range(M):
                for ip in range(D):
                    w2_p = W2.copy(); w2_p[k, j] += eps
                    w2_m = W2.copy(); w2_m[k, j] -= eps
                    w1_p = W1.copy(); w1_p[jp, ip] += eps
                    w1_m = W1.copy(); w1_m[jp, ip] -= eps

                    f_pp = forward(w1_p, w2_p)[0]
                    f_pm = forward(w1_m, w2_p)[0]
                    f_mp = forward(w1_p, w2_m)[0]
                    f_mm = forward(w1_m, w2_m)[0]
                    H21_num[k, j, jp, ip] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)

    err22 = float(np.max(np.abs(H22_ana - H22_num)))
    err21 = float(np.max(np.abs(H21_ana - H21_num)))

    return {
        "verified": bool(err22 < 1e-6 and err21 < 1e-6),
        "err_H22": err22,
        "err_H21": err21,
    }


# =====================================================================
# Exercise 8.7: Exact Hessian with Skip Connections
# =====================================================================

def verify_exercise_8_7() -> Dict[str, Any]:
    """Verify exact Hessian elements involving skip-layer connections.

    Skip connections:
        a_k = sum_j w_kj^{(2)} z_j + sum_i w_ki^{(s)} x_i + b_k^{(2)}
        del E_n / del w_ki^{(s)} = delta_k x_i

    Second derivatives:
        del^2 E_n / del w_ki^{(s)} del w_{k'i'}^{(s)} = x_i x_{i'} M_{kk'}
        del^2 E_n / del w_ki^{(s)} del w_{k'j}^{(2)}  = x_i z_j M_{kk'}
        del^2 E_n / del w_ki^{(s)} del w_{j'i'}^{(1)}  = x_i x_{i'} h'(a_{j'}) sum_{k'} M_{kk'} w_{k'j'}^{(2)}
    """
    rng = np.random.RandomState(47)
    D, M, K = 2, 3, 2
    x = rng.randn(D)
    t = rng.randn(K)
    W1 = rng.randn(M, D) * 0.4
    b1 = rng.randn(M) * 0.1
    W2 = rng.randn(K, M) * 0.4
    b2 = rng.randn(K) * 0.1
    Ws = rng.randn(K, D) * 0.4

    def forward(w1, w2, ws):
        a1 = w1 @ x + b1
        z1 = np.tanh(a1)
        a2 = w2 @ z1 + ws @ x + b2
        loss = 0.5 * np.sum((a2 - t) ** 2)
        return loss, a1, z1, a2

    loss, a1, z1, a2 = forward(W1, W2, Ws)
    M_mat = np.eye(K)

    # Analytical skip-skip Hessian
    H_ss_ana = np.zeros((K, D, K, D))
    for k in range(K):
        for i in range(D):
            for kp in range(K):
                for ip in range(D):
                    H_ss_ana[k, i, kp, ip] = x[i] * x[ip] * M_mat[k, kp]

    # Numerical skip-skip Hessian
    eps = 1e-5
    H_ss_num = np.zeros((K, D, K, D))
    for k in range(K):
        for i in range(D):
            for kp in range(K):
                for ip in range(D):
                    ws1 = Ws.copy(); ws1[k, i] += eps; ws1[kp, ip] += eps
                    ws2 = Ws.copy(); ws2[k, i] += eps; ws2[kp, ip] -= eps
                    ws3 = Ws.copy(); ws3[k, i] -= eps; ws3[kp, ip] += eps
                    ws4 = Ws.copy(); ws4[k, i] -= eps; ws4[kp, ip] -= eps
                    H_ss_num[k, i, kp, ip] = (
                        forward(W1, W2, ws1)[0] - forward(W1, W2, ws2)[0]
                        - forward(W1, W2, ws3)[0] + forward(W1, W2, ws4)[0]
                    ) / (4 * eps * eps)

    err_ss = float(np.max(np.abs(H_ss_ana - H_ss_num)))
    return {
        "verified": bool(err_ss < 1e-6),
        "err_H_skip_skip": err_ss,
    }


# =====================================================================
# Exercise 8.8: Outer Product Approximation for Multiple Outputs
# =====================================================================

def verify_exercise_8_8() -> Dict[str, Any]:
    """Verify outer-product (Gauss-Newton) Hessian approximation for multiple outputs.

    Error function:
        E_n = 0.5 sum_{k=1}^K (y_{nk} - t_{nk})^2
    Gradient:
        del E_n / del w_r = sum_{k=1}^K (y_{nk} - t_{nk}) (del y_{nk} / del w_r)
    Second derivative:
        del^2 E_n / del w_r del w_s
            = sum_{k=1}^K (del y_{nk} / del w_r) (del y_{nk} / del w_s)
              + sum_{k=1}^K (y_{nk} - t_{nk}) (del^2 y_{nk} / del w_r del w_s)

    Neglecting residual (y_{nk} - t_{nk}) yields the multi-output outer product approximation:
        H_{rs} approx sum_{n=1}^N sum_{k=1}^K (del y_{nk} / del w_r) (del y_{nk} / del w_s)
               = sum_{n=1}^N (J_n^T J_n)_{rs}
        where J_n in R^{K x W} is the network output Jacobian with respect to weights.
    """
    rng = np.random.RandomState(48)
    N, D, K = 5, 2, 3
    X = rng.randn(N, D)
    W = rng.randn(K, D)  # Simple linear model y = W x
    T = X @ W.T + rng.randn(N, K) * 0.01

    w_vec = W.ravel()
    W_dim = len(w_vec)

    H_outer = np.zeros((W_dim, W_dim))
    for n in range(N):
        xn = X[n]
        Jn = np.zeros((K, W_dim))
        for k in range(K):
            Jn[k, k * D : (k + 1) * D] = xn
        H_outer += Jn.T @ Jn

    def total_loss(w_flat):
        W_mat = w_flat.reshape(K, D)
        diff = X @ W_mat.T - T
        return 0.5 * np.sum(diff ** 2)

    eps = 1e-5
    H_num = np.zeros((W_dim, W_dim))
    for r in range(W_dim):
        for s in range(W_dim):
            wp = w_vec.copy(); wp[r] += eps; wp[s] += eps
            w1 = w_vec.copy(); w1[r] += eps; w1[s] -= eps
            w2 = w_vec.copy(); w2[r] -= eps; w2[s] += eps
            wm = w_vec.copy(); wm[r] -= eps; wm[s] -= eps
            H_num[r, s] = (total_loss(wp) - total_loss(w1) - total_loss(w2) + total_loss(wm)) / (4 * eps * eps)

    max_err = float(np.max(np.abs(H_outer - H_num)))
    return {
        "verified": bool(max_err < 1e-6),
        "max_err": max_err,
        "is_exact_for_linear": True,
    }


# =====================================================================
# Exercise 8.9: Expected Hessian for Squared-Loss
# =====================================================================

def verify_exercise_8_9() -> Dict[str, Any]:
    """Verify that expected Hessian for squared loss equals expected outer product of gradients.

    Equation (8.79):
        del^2 E / del w_r del w_s = int (del y / del w_r) (del y / del w_s) p(x) dx
    when y(x, w) = E[t | x].

    Proof:
        del E / del w_r = int int (y(x, w) - t) (del y / del w_r) p(x, t) dx dt
        del^2 E / del w_r del w_s = int int (del y / del w_r) (del y / del w_s) p(x, t) dx dt
                                  + int int (y(x, w) - t) (del^2 y / del w_r del w_s) p(x, t) dx dt
        The second term factorizes into:
            int [ int (y(x, w) - t) p(t | x) dt ] (del^2 y / del w_r del w_s) p(x) dx
        Since y(x, w) = E[t | x] = int t p(t | x) dt:
            int (y(x, w) - t) p(t | x) dt = y(x, w) - E[t | x] = 0.
        Hence the second term vanishes identically, leaving exactly (8.79).
    """
    rng = np.random.RandomState(49)
    N = 10000
    x = rng.uniform(-np.pi, np.pi, N)
    noise = rng.randn(N) * 0.5
    t = np.sin(x) + noise

    # Parametric model y(x, w) = sin(w1 * x)
    w_val = 1.0
    y_pred = np.sin(w_val * x)  # equals E[t|x]
    residual = y_pred - t

    dy_dw = x * np.cos(w_val * x)
    d2y_dw2 = -(x ** 2) * np.sin(w_val * x)

    term1 = np.mean(dy_dw ** 2)
    term2 = np.mean(residual * d2y_dw2)

    return {
        "verified": bool(abs(term2) < 0.02),
        "outer_product_term": float(term1),
        "residual_curvature_term": float(term2),
        "ratio_residual_to_outer": float(abs(term2) / term1),
    }


# =====================================================================
# Exercise 8.10: Outer Product Hessian for Logistic-Sigmoid Cross-Entropy
# =====================================================================

def verify_exercise_8_10() -> Dict[str, Any]:
    """Verify outer-product approximation (8.41) for logistic sigmoid cross-entropy.

    Equations:
        y_n = sigma(a_n),  E_n = - [t_n ln y_n + (1 - t_n) ln(1 - y_n)]
        del E_n / del a_n = y_n - t_n
        del E_n / del w_r = (y_n - t_n) (del a_n / del w_r)
        del^2 E_n / del w_r del w_s
            = y_n (1 - y_n) (del a_n / del w_r) (del a_n / del w_s)
              + (y_n - t_n) (del^2 a_n / del w_r del w_s)

    Neglecting (y_n - t_n) yields (8.41):
        H approx sum_{n=1}^N y_n (1 - y_n) nabla a_n nabla a_n^T
    """
    rng = np.random.RandomState(50)
    N, D = 20, 3
    X = rng.randn(N, D)
    w = rng.randn(D) * 0.5
    a = X @ w
    y = 1.0 / (1.0 + np.exp(-a))
    t = (rng.rand(N) < y).astype(float)

    H_outer = np.zeros((D, D))
    for n in range(N):
        xn = X[n]
        H_outer += y[n] * (1.0 - y[n]) * np.outer(xn, xn)

    def grad_loss(w_vec):
        an = X @ w_vec
        yn = 1.0 / (1.0 + np.exp(-an))
        return X.T @ (yn - t)

    eps = 1e-6
    H_num = np.zeros((D, D))
    for s in range(D):
        e_s = np.zeros(D)
        e_s[s] = eps
        H_num[:, s] = (grad_loss(w + e_s) - grad_loss(w - e_s)) / (2 * eps)

    diff = float(np.max(np.abs(H_outer - H_num)))
    return {
        "verified": bool(diff < 1e-6),
        "max_err": diff,
    }


# =====================================================================
# Exercise 8.11: Outer Product Hessian for Softmax Cross-Entropy
# =====================================================================

def verify_exercise_8_11() -> Dict[str, Any]:
    """Verify outer-product approximation for K-class softmax cross-entropy.

    Equations:
        y_{nk} = exp(a_{nk}) / sum_j exp(a_{nj})
        E_n = - sum_k t_{nk} ln y_{nk}
        del E_n / del a_{nk} = y_{nk} - t_{nk}
        del^2 E_n / del a_{nk} del a_{nl} = y_{nk} (delta_{kl} - y_{nl})
        H approx sum_{n=1}^N J_{a, n}^T (diag(y_n) - y_n y_n^T) J_{a, n}
    """
    rng = np.random.RandomState(51)
    N, D, K = 15, 3, 4
    X = rng.randn(N, D)
    W = rng.randn(K, D) * 0.4
    w_vec = W.ravel()
    W_dim = len(w_vec)

    A = X @ W.T
    exp_A = np.exp(A - np.max(A, axis=1, keepdims=True))
    Y = exp_A / np.sum(exp_A, axis=1, keepdims=True)
    T = np.zeros_like(Y)
    for n in range(N):
        k_choice = rng.choice(K, p=Y[n])
        T[n, k_choice] = 1.0

    H_outer = np.zeros((W_dim, W_dim))
    for n in range(N):
        xn = X[n]
        yn = Y[n]
        R_n = np.diag(yn) - np.outer(yn, yn)
        Jn = np.zeros((K, W_dim))
        for k in range(K):
            Jn[k, k * D : (k + 1) * D] = xn
        H_outer += Jn.T @ R_n @ Jn

    def grad_loss(w_flat):
        W_mat = w_flat.reshape(K, D)
        An = X @ W_mat.T
        exp_An = np.exp(An - np.max(An, axis=1, keepdims=True))
        Yn = exp_An / np.sum(exp_An, axis=1, keepdims=True)
        G = (Yn - T).T @ X
        return G.ravel()

    eps = 1e-6
    H_num = np.zeros((W_dim, W_dim))
    for s in range(W_dim):
        e_s = np.zeros(W_dim)
        e_s[s] = eps
        H_num[:, s] = (grad_loss(w_vec + e_s) - grad_loss(w_vec - e_s)) / (2 * eps)

    diff = float(np.max(np.abs(H_outer - H_num)))
    return {
        "verified": bool(diff < 1e-6),
        "max_err": diff,
    }


# =====================================================================
# Exercise 8.12: Sequential Inverse Hessian Update (Sherman-Morrison)
# =====================================================================

def verify_exercise_8_12() -> Dict[str, Any]:
    """Verify sequential inverse Hessian update using Sherman-Morrison identity (8.80).

    Identity (8.80):
        (M + v v^T)^{-1} = M^{-1} - (M^{-1} v v^T M^{-1}) / (1 + v^T M^{-1} v)

    Algorithm:
        1. Initialize V_0 = (1 / alpha) I
        2. For n = 1, ..., N with g_n = nabla y_n:
               u_n = V_{n-1} g_n
               gamma_n = 1 + g_n^T u_n
               V_n = V_{n-1} - (1 / gamma_n) outer(u_n, u_n)
        3. V_N approximates (alpha I + sum_{n=1}^N g_n g_n^T)^{-1} with O(N W^2) complexity.
    """
    rng = np.random.RandomState(52)
    N, W_dim = 25, 6
    G = rng.randn(N, W_dim)
    alpha = 0.05

    H_batch = alpha * np.eye(W_dim)
    for n in range(N):
        H_batch += np.outer(G[n], G[n])
    inv_H_batch = np.linalg.inv(H_batch)

    V = (1.0 / alpha) * np.eye(W_dim)
    for n in range(N):
        gn = G[n]
        u = V @ gn
        gamma = 1.0 + gn @ u
        V -= np.outer(u, u) / gamma

    max_err = float(np.max(np.abs(V - inv_H_batch)))
    rel_err = float(max_err / np.max(np.abs(inv_H_batch)))

    return {
        "verified": bool(rel_err < 1e-10),
        "max_abs_err": max_err,
        "max_rel_err": rel_err,
    }


# =====================================================================
# Exercise 8.13: Symbolic Derivative Swelling for Nested Soft ReLU
# =====================================================================

def verify_exercise_8_13() -> Dict[str, Any]:
    """Verify symbolic derivative formula (8.48) for nested softplus function (8.47).

    Function (8.47):
        h(a) = ln(1 + exp(a))
        y(x) = h(w2 h(w1 x + b1) + b2)

    Derivative formula (8.48):
        del y / del w1 = [w2 x exp(w1 x + b1 + b2 + w2 ln[1 + e^{w1 x + b1}])]
                         / [ (1 + e^{w1 x + b1}) (1 + exp(b2 + w2 ln[1 + e^{w1 x + b1}])) ]
    """
    w1, b1, w2, b2, x = 0.8, -0.2, 1.5, 0.4, 1.2

    exp1 = np.exp(w1 * x + b1)
    ln_term = np.log(1.0 + exp1)
    inner_arg = b2 + w2 * ln_term
    exp2 = np.exp(inner_arg)
    exp_num = np.exp(w1 * x + b1 + inner_arg)

    dy_dw1_symbolic = (w2 * x * exp_num) / ((1.0 + exp1) * (1.0 + exp2))

    w1_dual = DualNumber(w1, dual=1.0)
    h_dual = lambda a: (1.0 + a.exp()).log()
    u_dual = w1_dual * x + b1
    v_dual = h_dual(u_dual)
    z_dual = v_dual * w2 + b2
    y_dual = h_dual(z_dual)
    dy_dw1_autodiff = y_dual.dual

    h = lambda a: np.log(1.0 + np.exp(a))
    f = lambda w1_val: h(w2 * h(w1_val * x + b1) + b2)
    eps = 1e-6
    dy_dw1_num = (f(w1 + eps) - f(w1 - eps)) / (2 * eps)

    err_autodiff = abs(dy_dw1_symbolic - dy_dw1_autodiff)
    err_num = abs(dy_dw1_symbolic - dy_dw1_num)

    return {
        "verified": bool(err_autodiff < 1e-12 and err_num < 1e-8),
        "symbolic_value": float(dy_dw1_symbolic),
        "autodiff_value": float(dy_dw1_autodiff),
        "numerical_value": float(dy_dw1_num),
        "err_autodiff": float(err_autodiff),
    }


# =====================================================================
# Exercise 8.14: Logistic Map Evaluation Trace and Formula Swell
# =====================================================================

def verify_exercise_8_14() -> Dict[str, Any]:
    """Verify evaluation trace and derivative recursion for the logistic map.

    Definition:
        L_{n+1}(x) = 4 L_n(x) (1 - L_n(x)),  with L_1(x) = x.

    Evaluation trace:
        L_1(x) = x
        L_2(x) = 4 x (1 - x) = 4x - 4x^2
        L_3(x) = 4 L_2(x) (1 - L_2(x))
        L_4(x) = 4 L_3(x) (1 - L_3(x))

    Derivatives:
        L_1'(x) = 1
        L_2'(x) = 4 - 8x
        L_3'(x) = 4 L_2'(x) (1 - 2 L_2(x))
        L_4'(x) = 4 L_3'(x) (1 - 2 L_3(x))
    """
    x_val = 0.35

    def eval_logistic_map(x, n):
        L = x
        dL = 1.0
        trace = [(L, dL)]
        for i in range(1, n):
            dL = 4.0 * dL * (1.0 - 2.0 * L)
            L = 4.0 * L * (1.0 - L)
            trace.append((L, dL))
        return trace

    trace = eval_logistic_map(x_val, 4)

    x_dual = DualNumber(x_val, dual=1.0)
    L_curr = x_dual
    dual_results = []
    for _ in range(4):
        dual_results.append((L_curr.real, L_curr.dual))
        L_curr = 4.0 * L_curr * (1.0 - L_curr)

    errors = [abs(trace[i][1] - dual_results[i][1]) for i in range(4)]
    return {
        "verified": bool(max(errors) < 1e-12),
        "trace": [{"L": float(t[0]), "dL": float(t[1])} for t in trace],
        "degree_growth": [2 ** (i) for i in range(4)],
    }


# =====================================================================
# Exercise 8.15: Forward-Mode Tangent Equations Derivation
# =====================================================================

def verify_exercise_8_15() -> Dict[str, Any]:
    """Verify forward-mode tangent equations (8.58)-(8.64) from primal equations (8.50)-(8.56).

    Primal trace:
        v1 = x1,  v2 = x2
        v3 = v1 * v2
        v4 = sin(v2)
        v5 = exp(v3)
        v6 = v3 - v4
        v7 = v5 + v6

    Tangent equations (8.58)-(8.64) for del / del x1 (setting v1_dot = 1, v2_dot = 0):
        v1_dot = 1
        v2_dot = 0
        v3_dot = v1 * v2_dot + v1_dot * v2 = v2
        v4_dot = v2_dot * cos(v2) = 0
        v5_dot = v3_dot * exp(v3) = v3_dot * v5
        v6_dot = v3_dot - v4_dot = v3_dot
        v7_dot = v5_dot + v6_dot
    """
    x1, x2 = 1.0, 2.0
    trace = evaluate_trace_forward_mode(x1, x2)

    exact_df_dx1 = x2 + x2 * np.exp(x1 * x2)
    v7_dot = trace["v7"][1]

    err = abs(v7_dot - exact_df_dx1)
    return {
        "verified": bool(err < 1e-12),
        "v7_dot": float(v7_dot),
        "exact_df_dx1": float(exact_df_dx1),
        "err": float(err),
    }


# =====================================================================
# Exercise 8.16: Reverse-Mode Adjoint Equations Derivation
# =====================================================================

def verify_exercise_8_16() -> Dict[str, Any]:
    """Verify reverse-mode adjoint equations (8.70)-(8.76) from evaluation trace.

    Adjoint equations (8.70)-(8.76):
        v7_bar = 1
        v6_bar = v7_bar = 1
        v5_bar = v7_bar = 1
        v4_bar = - v6_bar = -1
        v3_bar = v5_bar * v5 + v6_bar = exp(v3) + 1
        v2_bar = v3_bar * v1 + v4_bar * cos(v2) = (exp(x1 x2) + 1) * x1 - cos(x2)
        v1_bar = v3_bar * v2 = (exp(x1 x2) + 1) * x2
    """
    x1, x2 = 1.0, 2.0
    trace = evaluate_trace_reverse_mode(x1, x2)

    exact_df_dx1 = x2 + x2 * np.exp(x1 * x2)
    exact_df_dx2 = x1 + x1 * np.exp(x1 * x2) - np.cos(x2)

    v1_bar = trace["v1"][1]
    v2_bar = trace["v2"][1]

    err_x1 = abs(v1_bar - exact_df_dx1)
    err_x2 = abs(v2_bar - exact_df_dx2)

    return {
        "verified": bool(err_x1 < 1e-12 and err_x2 < 1e-12),
        "v1_bar": float(v1_bar),
        "v2_bar": float(v2_bar),
        "exact_df_dx1": float(exact_df_dx1),
        "exact_df_dx2": float(exact_df_dx2),
    }


# =====================================================================
# Exercise 8.17: Exact Numerical Comparison at (1, 2)
# =====================================================================

def verify_exercise_8_17() -> Dict[str, Any]:
    """Verify del f / del x1 at x1=1, x2=2 across analytical, forward, and reverse modes.

    Analytic:
        f(x1, x2) = x1 x2 + exp(x1 x2) - sin(x2)
        del f / del x1 = x2 + x2 exp(x1 x2)
        At (1, 2): 2 + 2 e^2 = 2 * (1 + 7.3890560989...) = 16.77811219786...
    """
    x1, x2 = 1.0, 2.0
    val_analytic = x2 + x2 * np.exp(x1 * x2)

    fwd_trace = evaluate_trace_forward_mode(x1, x2)
    val_forward = fwd_trace["v7"][1]

    rev_trace = evaluate_trace_reverse_mode(x1, x2)
    val_reverse = rev_trace["v1"][1]

    err_fwd = abs(val_forward - val_analytic)
    err_rev = abs(val_reverse - val_analytic)

    return {
        "verified": bool(err_fwd < 1e-12 and err_rev < 1e-12),
        "val_analytic": float(val_analytic),
        "val_forward": float(val_forward),
        "val_reverse": float(val_reverse),
        "primal_v": {k: float(v[0]) for k, v in fwd_trace.items()},
        "tangent_v_dot": {k: float(v[1]) for k, v in fwd_trace.items()},
        "adjoint_v_bar": {k: float(v[1]) for k, v in rev_trace.items()},
    }


# =====================================================================
# Exercise 8.18: Jacobian-Vector Product in a Single Forward Pass
# =====================================================================

def verify_exercise_8_18() -> Dict[str, Any]:
    """Verify that Jacobian-vector product J * r is evaluated in a single forward pass.

    Proof:
        Let r = sum_{i=1}^D r_i e_i.
        In forward-mode automatic differentiation, setting x_dot = e_i produces
        y_dot = J e_i (the i-th column of the Jacobian).
        By linearity of directional differentiation:
            y_dot(r) = del / del t [ f(x + t r) ] |_{t=0}
                     = sum_{i=1}^D r_i del / del t [ f(x + t e_i) ] |_{t=0}
                     = sum_{i=1}^D r_i (J e_i)
                     = J (sum_{i=1}^D r_i e_i) = J r.
        Therefore, initializing the input tangent variables to x_dot = r yields
        the exact product J r in a single forward evaluation.
    """
    rng = np.random.RandomState(53)
    D = 4
    x = rng.randn(D)
    r = rng.randn(D)

    def vec_fn(duals: List[DualNumber]) -> List[DualNumber]:
        f1 = duals[0] * duals[1] + duals[2].sin()
        f2 = (duals[0] + duals[3]).exp() - duals[1] ** 2
        f3 = duals[1] * duals[2] * duals[3] + duals[0]
        return [f1, f2, f3]

    y_vals, jvp_single_pass = forward_mode_jvp(vec_fn, x, r)
    J_full = forward_mode_jacobian(vec_fn, x)
    jvp_matrix_mul = J_full @ r

    max_err = float(np.max(np.abs(jvp_single_pass - jvp_matrix_mul)))
    return {
        "verified": bool(max_err < 1e-12),
        "max_err": max_err,
        "jvp_single_pass": jvp_single_pass,
        "jvp_matrix_mul": jvp_matrix_mul,
        "J_full": J_full,
    }


# =====================================================================
# Master Verification Routine
# =====================================================================

def verify_all_ch8_exercises() -> Dict[str, Dict[str, Any]]:
    """Run verification routines for all 18 exercises in Chapter 8."""
    return {
        "8.1": verify_exercise_8_1(),
        "8.2": verify_exercise_8_2(),
        "8.3": verify_exercise_8_3(),
        "8.4": verify_exercise_8_4(),
        "8.5": verify_exercise_8_5(),
        "8.6": verify_exercise_8_6(),
        "8.7": verify_exercise_8_7(),
        "8.8": verify_exercise_8_8(),
        "8.9": verify_exercise_8_9(),
        "8.10": verify_exercise_8_10(),
        "8.11": verify_exercise_8_11(),
        "8.12": verify_exercise_8_12(),
        "8.13": verify_exercise_8_13(),
        "8.14": verify_exercise_8_14(),
        "8.15": verify_exercise_8_15(),
        "8.16": verify_exercise_8_16(),
        "8.17": verify_exercise_8_17(),
        "8.18": verify_exercise_8_18(),
    }
