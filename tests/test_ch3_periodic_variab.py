"""
Unit tests for Chapter 3, Section 3.3 Periodic Variables (Von Mises Distribution).
"""
import os
import numpy as np
import pytest
from scipy import integrate
import matplotlib
matplotlib.use('Agg')

from common.probability import (
    VonMises,
    VonMisesDistribution,
    VonMisesMixture,
    plot_figure_3_9,
    plot_figure_3_10,
    plot_figure_3_11,
    plot_figure_3_12
)


def test_von_mises_normalization():
    """Verify that p(theta | theta_0, m) integrates to 1 over [0, 2*pi) for various m."""
    for m in [0.0, 0.2, 1.0, 5.0, 20.0]:
        for th0 in [0.0, np.pi/3, np.pi, 5*np.pi/3]:
            vm = VonMisesDistribution(theta_0=th0, m=m)
            integral, _ = integrate.quad(lambda t: vm.pdf(t), 0, 2 * np.pi)
            assert np.isclose(integral, 1.0, atol=1e-5), f"Failed for m={m}, th0={th0}: got {integral}"


def test_von_mises_periodicity():
    """Verify periodicity: p(theta + 2*k*pi) == p(theta)."""
    vm = VonMisesDistribution(theta_0=np.pi/4, m=3.0)
    thetas = np.array([0.1, 1.0, 2.5, 5.0])
    for k in [-2, -1, 1, 2]:
        assert np.allclose(vm.pdf(thetas + 2 * k * np.pi), vm.pdf(thetas), atol=1e-10)
        assert np.allclose(vm.log_pdf(thetas + 2 * k * np.pi), vm.log_pdf(thetas), atol=1e-10)


def test_von_mises_log_pdf_consistency():
    """Verify log_pdf matches ln(pdf)."""
    vm = VonMisesDistribution(theta_0=np.pi/2, m=2.5)
    thetas = np.linspace(0, 2 * np.pi, 50)
    pdf_vals = vm.pdf(thetas)
    log_pdf_vals = vm.log_pdf(thetas)
    assert np.allclose(np.log(pdf_vals), log_pdf_vals, atol=1e-7)


def test_von_mises_uniform_limit():
    """For m=0, von Mises becomes the circular uniform distribution 1/(2*pi)."""
    vm = VonMisesDistribution(theta_0=1.234, m=0.0)
    thetas = np.linspace(0, 2 * np.pi, 20)
    expected = 1.0 / (2 * np.pi)
    assert np.allclose(vm.pdf(thetas), expected, atol=1e-12)


def test_von_mises_large_m_gaussian_approximation():
    """For large m, von Mises approaches Gaussian with variance 1/m around theta_0."""
    m = 50.0
    th0 = np.pi
    vm = VonMisesDistribution(theta_0=th0, m=m)
    
    # Delta close to 0
    delta = np.linspace(-0.2, 0.2, 21)
    thetas = th0 + delta
    vm_density = vm.pdf(thetas)
    
    # Gaussian density: 1/sqrt(2*pi*sigma^2) * exp(-delta^2 / (2*sigma^2)) with sigma^2 = 1/m
    sigma = 1.0 / np.sqrt(m)
    gauss_density = np.exp(-0.5 * (delta / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)
    
    # Ratio should be very close to 1
    rel_error = np.abs(vm_density - gauss_density) / gauss_density
    assert np.all(rel_error < 0.05), f"Max relative error: {np.max(rel_error)}"


def test_circular_mean_boundary():
    """
    Test the key textbook motivating example (Bishop p. 89):
    Observations at 1 deg and 359 deg.
    Arithmetic mean is 180 deg (flawed), whereas circular mean is 0 deg (or 360 deg).
    """
    th1 = np.radians(1.0)
    th2 = np.radians(359.0)
    circ_mean = VonMisesDistribution.circular_mean([th1, th2])
    # circ_mean should be 0 (or within epsilon of 0 or 2*pi)
    assert np.isclose(circ_mean, 0.0, atol=1e-5) or np.isclose(circ_mean, 2*np.pi, atol=1e-5)


def test_circular_mean_origin_invariance():
    """Verify that circular mean shifts exactly by alpha when all angles are shifted by alpha."""
    angles = np.array([0.2, 0.5, 1.2, 1.8])
    base_mean = VonMisesDistribution.circular_mean(angles)
    
    shift = 1.35
    shifted_angles = (angles + shift) % (2 * np.pi)
    shifted_mean = VonMisesDistribution.circular_mean(shifted_angles)
    
    expected_mean = (base_mean + shift) % (2 * np.pi)
    assert np.isclose(shifted_mean, expected_mean, atol=1e-6)


def test_circular_resultant_and_variance():
    """Verify resultant vector length r_bar and circular variance V = 1 - r_bar."""
    # Identical angles -> r_bar = 1, V = 0
    angles_identical = np.array([1.0, 1.0, 1.0])
    assert np.isclose(VonMisesDistribution.circular_resultant_length(angles_identical), 1.0)
    assert np.isclose(VonMisesDistribution.circular_variance(angles_identical), 0.0)
    
    # Perfectly dispersed angles -> r_bar = 0, V = 1
    angles_orthogonal = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    assert np.isclose(VonMisesDistribution.circular_resultant_length(angles_orthogonal), 0.0, atol=1e-7)
    assert np.isclose(VonMisesDistribution.circular_variance(angles_orthogonal), 1.0, atol=1e-7)


def test_bessel_ratio_A_properties():
    """Verify A(0) = 0, A(m) monotonic and bounded in [0, 1)."""
    assert np.isclose(VonMisesDistribution.bessel_ratio_A(0.0), 0.0)
    m_vals = np.linspace(0.1, 20.0, 50)
    A_vals = VonMisesDistribution.bessel_ratio_A(m_vals)
    assert np.all(np.diff(A_vals) > 0), "A(m) must be strictly increasing"
    assert np.all((A_vals >= 0) & (A_vals < 1.0))
    # At m=10, A(10) ~ 0.949
    assert np.isclose(VonMisesDistribution.bessel_ratio_A(10.0), 0.94868, atol=1e-3)


def test_von_mises_mle():
    """Verify MLE parameter estimation recovers ground truth on generated sample."""
    true_th0 = 2.0
    true_m = 4.0
    vm_true = VonMisesDistribution(theta_0=true_th0, m=true_m)
    samples = vm_true.sample(size=10000, seed=42)
    
    vm_fit = VonMisesDistribution.fit_mle(samples)
    assert np.isclose(vm_fit.theta_0, true_th0, atol=0.08)
    assert np.isclose(vm_fit.m, true_m, atol=0.25)


def test_figures_3_9_to_3_12_execution(tmp_path):
    """Verify plotting functions execute properly and create valid figure files."""
    f9, a9 = plot_figure_3_9(save_paths=[str(tmp_path / 'fig3_9.png')])
    assert os.path.exists(tmp_path / 'fig3_9.png')
    
    f10, a10 = plot_figure_3_10(save_paths=[str(tmp_path / 'fig3_10.png')])
    assert os.path.exists(tmp_path / 'fig3_10.png')
    
    f11, (ax11_1, ax11_2) = plot_figure_3_11(save_paths=[str(tmp_path / 'fig3_11.png')])
    assert os.path.exists(tmp_path / 'fig3_11.png')
    
    f12, (ax12_1, ax12_2) = plot_figure_3_12(save_paths=[str(tmp_path / 'fig3_12.png')])
    assert os.path.exists(tmp_path / 'fig3_12.png')


def test_von_mises_alias():
    """Verify VonMises is an alias for VonMisesDistribution."""
    assert VonMises is VonMisesDistribution
    vm = VonMises(theta_0=1.0, m=2.0)
    assert isinstance(vm, VonMisesDistribution)


def test_von_mises_mixture():
    """Verify VonMisesMixture normalization, log_pdf, responsibilities, and sampling."""
    weights = [0.4, 0.6]
    theta_0s = [np.pi / 4, 5 * np.pi / 4]
    ms = [3.0, 5.0]
    mix = VonMisesMixture(weights=weights, theta_0s=theta_0s, ms=ms)

    # 1. Normalization
    integral, _ = integrate.quad(lambda t: mix.pdf(t), 0, 2 * np.pi)
    assert np.isclose(integral, 1.0, atol=1e-5)

    # 2. Periodicity
    thetas = np.array([0.2, 1.5, 3.8, 5.5])
    assert np.allclose(mix.pdf(thetas + 2 * np.pi), mix.pdf(thetas), atol=1e-10)
    assert np.allclose(mix.log_pdf(thetas + 2 * np.pi), mix.log_pdf(thetas), atol=1e-10)

    # 3. Log PDF consistency
    pdf_vals = mix.pdf(thetas)
    log_pdf_vals = mix.log_pdf(thetas)
    assert np.allclose(np.log(pdf_vals), log_pdf_vals, atol=1e-7)

    # 4. Responsibilities
    resp = mix.responsibilities(thetas)
    assert resp.shape == (len(thetas), 2)
    assert np.allclose(np.sum(resp, axis=1), 1.0)
    assert np.all(resp >= 0.0)

    # 5. Sampling
    samples = mix.sample(size=100, seed=42)
    assert len(samples) == 100
    assert np.all((samples >= 0.0) & (samples < 2 * np.pi))

    # 6. Scalar evaluation
    single_pdf = mix.pdf(np.pi / 4)
    assert isinstance(single_pdf, float)
    single_log = mix.log_pdf(np.pi / 4)
    assert isinstance(single_log, float)


def test_saved_figures_exist():
    """Verify that Figures 3.9 through 3.12 exist in both 3/result/ and result/."""
    expected_files = [
        'fig3_09_periodic_variables_mean.png',
        'fig3_10_gaussian_conditioned_unit_circle.png',
        'fig3_11_von_mises_cartesian_polar.png',
        'fig3_12_bessel_and_ratio_function.png'
    ]
    for fn in expected_files:
        assert os.path.exists(os.path.join('3', 'result', fn)), f"Missing 3/result/{fn}"
        assert os.path.exists(os.path.join('result', fn)), f"Missing result/{fn}"
        assert os.path.getsize(os.path.join('3', 'result', fn)) > 1000
        assert os.path.getsize(os.path.join('result', fn)) > 1000

