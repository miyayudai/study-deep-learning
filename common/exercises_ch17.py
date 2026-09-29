"""Solutions and numerical verifications for Chapter 17 Exercises (Generative Adversarial Networks).

Exercises covered:
- Exercise 17.1: Optimal Discriminator, Effective Generator Objective, and Jensen-Shannon Divergence
- Exercise 17.2: Minimax Optimization Dynamics, Saddle Point Instability, and Limit Cycles
- Exercise 17.3: Mode Collapse, Optimal Discriminator Output Under Class Imbalance
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# Exercise 17.1: Optimal Discriminator & Jensen-Shannon Divergence
# =====================================================================

def gan_continuous_error(
    p_data: np.ndarray,
    p_g: np.ndarray,
    d: np.ndarray,
    dx: float = 0.01,
    eps: float = 1e-12,
) -> float:
    r"""Compute continuous GAN error function E(p_G, d) via numerical integration (Eq. 17.14).

    .. math::
        E(p_G, d) = -\int p_{\text{data}}(x) \ln d(x) \, dx - \int p_G(x) \ln(1 - d(x)) \, dx
    """
    d_clipped = np.clip(d, eps, 1.0 - eps)
    term1 = -np.sum(p_data * np.log(d_clipped)) * dx
    term2 = -np.sum(p_g * np.log(1.0 - d_clipped)) * dx
    return float(term1 + term2)


def optimal_discriminator_continuous(
    p_data: np.ndarray,
    p_g: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    r"""Compute optimal discriminator d*(x) for fixed p_data and p_G (Eq. 17.15).

    .. math::
        d^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}
    """
    denom = p_data + p_g + eps
    return p_data / denom


def effective_generator_objective(
    p_data: np.ndarray,
    p_g: np.ndarray,
    dx: float = 0.01,
    eps: float = 1e-12,
) -> float:
    r"""Compute effective generator error C(p_G) = E(p_G, d*) (Eq. 17.16).

    .. math::
        C(p_G) = -\int p_{\text{data}}(x) \ln\left(\frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_G(x)}\right) dx
                 -\int p_G(x) \ln\left(\frac{p_G(x)}{p_{\text{data}}(x)+p_G(x)}\right) dx
    """
    d_opt = optimal_discriminator_continuous(p_data, p_g, eps=eps)
    return gan_continuous_error(p_data, p_g, d_opt, dx=dx, eps=eps)


def gan_value_function_jsd(
    p_data: np.ndarray,
    p_g: np.ndarray,
    dx: float = 0.01,
    eps: float = 1e-12,
) -> float:
    r"""Compute Goodfellow / Bishop Eq. (17.17) formulation: -ln(4) + 2 * JSD(p_data || p_G).

    .. math::
        V(p_G) = -\ln(4) + \text{KL}\left(p_{\text{data}} \,\middle\|\, \frac{p_{\text{data}}+p_G}{2}\right)
                         + \text{KL}\left(p_G \,\middle\|\, \frac{p_{\text{data}}+p_G}{2}\right)
    """
    kl_p, kl_g, _ = jensen_shannon_divergence(p_data, p_g, dx=dx, eps=eps)
    return float(-np.log(4.0) + kl_p + kl_g)


def jensen_shannon_divergence(
    p: np.ndarray,
    q: np.ndarray,
    dx: float = 0.01,
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    r"""Compute KL divergences to midpoint distribution and Jensen-Shannon divergence (Eq. 17.17).

    .. math::
        M = \frac{p + q}{2}
        \text{KL}(p \| M) = \int p(x) \ln \frac{p(x)}{M(x)} \, dx
        \text{KL}(q \| M) = \int q(x) \ln \frac{q(x)}{M(x)} \, dx
        \text{JSD}(p \| q) = \frac{1}{2} \text{KL}(p \| M) + \frac{1}{2} \text{KL}(q \| M)

    Returns:
        Tuple of (KL(p || M), KL(q || M), JSD(p || q)).
    """
    m = 0.5 * (p + q)
    p_safe = np.where(p > 0, p, eps)
    q_safe = np.where(q > 0, q, eps)
    m_safe = np.where(m > 0, m, eps)

    kl_p_m = np.sum(np.where(p > 0, p * np.log(p_safe / m_safe), 0.0)) * dx
    kl_q_m = np.sum(np.where(q > 0, q * np.log(q_safe / m_safe), 0.0)) * dx
    jsd = 0.5 * (kl_p_m + kl_q_m)
    return float(kl_p_m), float(kl_q_m), float(jsd)


# =====================================================================
# Exercise 17.2: Minimax Training Dynamics & Saddle Point Instability
# =====================================================================

def bilinear_saddle_gradient_flow(
    a0: float = 1.0,
    b0: float = 0.0,
    eta: float = 1.0,
    t_max: float = 10.0,
    num_steps: int = 1000,
) -> Dict[str, np.ndarray]:
    r"""Compute analytical and numerical continuous-time gradient flow for E(a, b) = ab (Eq. 17.18 - 17.20).

    .. math::
        \frac{da}{dt} = \eta \frac{\partial E}{\partial a} = \eta b
        \frac{db}{dt} = -\eta \frac{\partial E}{\partial b} = -\eta a
        a(t) = C \cos(\eta t) + D \sin(\eta t)

    For a(0) = 1, b(0) = 0:
        C = 1, D = 0 => a(t) = \cos(\eta t), b(t) = -\sin(\eta t)
    """
    t = np.linspace(0, t_max, num_steps)
    # Analytical solution
    a_analytical = a0 * np.cos(eta * t) + b0 * np.sin(eta * t)
    b_analytical = b0 * np.cos(eta * t) - a0 * np.sin(eta * t)

    # Numerical Euler integration
    dt = t_max / (num_steps - 1)
    a_euler = np.zeros(num_steps)
    b_euler = np.zeros(num_steps)
    a_euler[0], b_euler[0] = a0, b0

    for i in range(num_steps - 1):
        da = eta * b_euler[i] * dt
        db = -eta * a_euler[i] * dt
        a_euler[i + 1] = a_euler[i] + da
        b_euler[i + 1] = b_euler[i] + db

    radius_analytical = np.sqrt(a_analytical**2 + b_analytical**2)
    radius_euler = np.sqrt(a_euler**2 + b_euler**2)

    return {
        "t": t,
        "a_analytical": a_analytical,
        "b_analytical": b_analytical,
        "radius_analytical": radius_analytical,
        "a_euler": a_euler,
        "b_euler": b_euler,
        "radius_euler": radius_euler,
    }


def discrete_simultaneous_vs_alternating(
    a0: float = 1.0,
    b0: float = 0.0,
    gamma: float = 0.1,
    num_steps: int = 100,
) -> Dict[str, np.ndarray]:
    """Compare discrete simultaneous vs alternating gradient updates for E(a, b) = ab."""
    # Simultaneous: a_{k+1} = a_k + gamma * b_k, b_{k+1} = b_k - gamma * a_k
    a_sim = np.zeros(num_steps)
    b_sim = np.zeros(num_steps)
    a_sim[0], b_sim[0] = a0, b0
    for k in range(num_steps - 1):
        a_sim[k + 1] = a_sim[k] + gamma * b_sim[k]
        b_sim[k + 1] = b_sim[k] - gamma * a_sim[k]

    # Alternating: a_{k+1} = a_k + gamma * b_k, b_{k+1} = b_k - gamma * a_{k+1}
    a_alt = np.zeros(num_steps)
    b_alt = np.zeros(num_steps)
    a_alt[0], b_alt[0] = a0, b0
    for k in range(num_steps - 1):
        a_alt[k + 1] = a_alt[k] + gamma * b_alt[k]
        b_alt[k + 1] = b_alt[k] - gamma * a_alt[k + 1]

    return {
        "a_sim": a_sim,
        "b_sim": b_sim,
        "radius_sim": np.sqrt(a_sim**2 + b_sim**2),
        "a_alt": a_alt,
        "b_alt": b_alt,
        "radius_alt": np.sqrt(a_alt**2 + b_alt**2),
    }


# =====================================================================
# Exercise 17.3: Optimal Discriminator Under Multi-modal Class Imbalance
# =====================================================================

def evaluate_dog_cat_discriminator_equilibrium(
    p_dog_weight: float = 0.5,
    p_cat_weight: float = 0.5,
    generator_dog_mode: float = 1.0,
    generator_cat_mode: float = 0.0,
) -> Dict[str, float]:
    r"""Evaluate theoretical discriminator output d*(x) when generator produces only dogs.

    Real data: p_data(x) = p_dog_weight * p_dog(x) + p_cat_weight * p_cat(x)
    Generator: p_G(x) = generator_dog_mode * p_dog(x) + generator_cat_mode * p_cat(x)

    When x is drawn from p_dog(x):
        d*(x_dog) = p_data(x_dog) / (p_data(x_dog) + p_G(x_dog))
                  = (0.5 * 1.0) / (0.5 * 1.0 + 1.0 * 1.0) = 0.5 / 1.5 = 1/3
    When x is drawn from p_cat(x):
        d*(x_cat) = p_data(x_cat) / (p_data(x_cat) + p_G(x_cat))
                  = (0.5 * 1.0) / (0.5 * 1.0 + 0.0) = 1.0
    """
    d_star_dog = p_dog_weight / (p_dog_weight + generator_dog_mode)
    denom_cat = p_cat_weight + generator_cat_mode
    d_star_cat = p_cat_weight / denom_cat if denom_cat > 0 else 1.0

    return {
        "d_star_dog": float(d_star_dog),
        "d_star_cat": float(d_star_cat),
    }
