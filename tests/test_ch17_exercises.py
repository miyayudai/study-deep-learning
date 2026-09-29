"""Unit tests for Chapter 17 Exercises (17.1 to 17.3)."""

import numpy as np
import pytest
from scipy.stats import norm

from common.exercises_ch17 import (
    gan_continuous_error,
    optimal_discriminator_continuous,
    effective_generator_objective,
    gan_value_function_jsd,
    jensen_shannon_divergence,
    bilinear_saddle_gradient_flow,
    discrete_simultaneous_vs_alternating,
    evaluate_dog_cat_discriminator_equilibrium,
)


def test_exercise_17_1_optimal_discriminator_and_jsd():
    """Verify Exercise 17.1: optimal d*(x), effective objective C(p_G), and JSD relation."""
    x = np.linspace(-6, 6, 1200)
    dx = x[1] - x[0]

    # Two 1D Gaussians
    p_data = norm.pdf(x, loc=0.0, scale=1.0)
    p_g_diff = norm.pdf(x, loc=1.5, scale=1.0)
    p_g_same = norm.pdf(x, loc=0.0, scale=1.0)

    # 1. Optimal discriminator formula: d*(x) = p_data(x) / (p_data(x) + p_g(x))
    d_opt_same = optimal_discriminator_continuous(p_data, p_g_same)
    assert np.allclose(d_opt_same, 0.5, atol=1e-4)

    # 2. When p_g == p_data, cross-entropy error C(p_G) = +ln(4)
    c_same = effective_generator_objective(p_data, p_g_same, dx=dx)
    assert np.isclose(c_same, np.log(4.0), atol=1e-3)

    # 3. Value function V(p_G) (Eq. 17.17): V(p_G) = -ln(4) + 2 * JSD
    v_same = gan_value_function_jsd(p_data, p_g_same, dx=dx)
    assert np.isclose(v_same, -np.log(4.0), atol=1e-3)

    # 4. When p_g != p_data, V(p_G) > -ln(4)
    v_diff = gan_value_function_jsd(p_data, p_g_diff, dx=dx)
    assert v_diff > -np.log(4.0)

    # 5. JSD relation: V(p_G) = -ln(4) + (kl_p + kl_g)
    kl_p, kl_g, jsd = jensen_shannon_divergence(p_data, p_g_diff, dx=dx)
    assert np.isclose(v_diff, -np.log(4.0) + (kl_p + kl_g), atol=1e-3)
    assert jsd > 0.0


def test_exercise_17_2_saddle_dynamics_and_orbit():
    """Verify Exercise 17.2: continuous gradient flow circular orbits and discrete divergence."""
    res = bilinear_saddle_gradient_flow(a0=1.0, b0=0.0, eta=2.0, t_max=10.0, num_steps=1000)

    # Unit radius conservation in analytical continuous flow: a(t)^2 + b(t)^2 == 1
    radii = res["radius_analytical"]
    assert np.allclose(radii, 1.0, atol=1e-10)

    # Initial conditions
    assert np.isclose(res["a_analytical"][0], 1.0)
    assert np.isclose(res["b_analytical"][0], 0.0)

    # Harmonic solution at t = pi / (2 * eta): a = cos(pi/2) = 0, b = -sin(pi/2) = -1
    idx_quarter = int(1000 * (np.pi / 4.0) / 10.0)
    assert np.isclose(res["a_analytical"][idx_quarter]**2 + res["b_analytical"][idx_quarter]**2, 1.0)

    # Discrete simultaneous updates spiral outwards (radius increases)
    disc = discrete_simultaneous_vs_alternating(a0=1.0, b0=0.0, gamma=0.1, num_steps=50)
    assert disc["radius_sim"][-1] > disc["radius_sim"][0]


def test_exercise_17_3_dog_cat_equilibrium():
    """Verify Exercise 17.3: optimal discriminator outputs 1/3 for dog images when generator produces only dogs."""
    res = evaluate_dog_cat_discriminator_equilibrium(
        p_dog_weight=0.5,
        p_cat_weight=0.5,
        generator_dog_mode=1.0,
        generator_cat_mode=0.0,
    )
    # Discriminator probability that dog image is real
    assert np.isclose(res["d_star_dog"], 1.0 / 3.0)
    # Discriminator probability that cat image is real
    assert np.isclose(res["d_star_cat"], 1.0)
