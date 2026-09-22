"""
Unit tests for Chapter 3, Section 3.5: Nonparametric Methods
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.

Covers:
- 3.5.1 Histograms (normalization, bin width scaling, curse of dimensionality)
- 3.5.2 Kernel densities (Parzen windows, Gaussian, boxcar, Epanechnikov, normalization, bandwidth trade-off)
- 3.5.3 Nearest-neighbours (KNN density estimation, adaptive volume, spatial divergence, KNN classifier, Bayes posterior, Cover-Hart bound)
- Figures 3.13, 3.14, 3.15, 3.16 reproduction and asset verification
"""
import os
import sys
import numpy as np
import pytest
from scipy import integrate, special
import matplotlib
matplotlib.use('Agg')

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from common.probability import (
    HistogramDensity, HistogramDensity1D,
    KernelDensityEstimator, KernelDensity1D, KernelDensityND,
    KNearestNeighborsDensity, KNNDensityEstimator,
    KNearestNeighborsClassifier, KNNClassifier,
    plot_figure_3_13, plot_figure_3_14, plot_figure_3_15, plot_figure_3_16
)


# ---------------------------------------------------------------------
# 3.5.1 Histograms Tests
# ---------------------------------------------------------------------

def test_histogram_density_normalization():
    """Verify that the histogram density integrates to exactly 1.0 (Eq 3.175)."""
    np.random.seed(42)
    data = np.random.normal(loc=0.5, scale=0.15, size=200)
    data = np.clip(data, 0.0, 1.0)
    
    for delta in [0.04, 0.08, 0.25]:
        hist = HistogramDensity(bin_width=delta, range_bounds=(0.0, 1.0)).fit(data)
        
        # Integral = sum_i p_i * Delta_i
        total_prob = np.sum(hist.density * hist.bin_widths)
        assert np.isclose(total_prob, 1.0, atol=1e-7), f"Failed normalization for delta={delta}"
        
        # Check non-negativity
        assert np.all(hist.density >= 0.0)
        
        # Evaluate on fine grid and check Riemann sum
        x_grid = np.linspace(0.001, 0.999, 1000)
        p_vals = hist.pdf(x_grid)
        dx = x_grid[1] - x_grid[0]
        riemann_integral = np.sum(p_vals) * dx
        assert np.isclose(riemann_integral, 1.0, atol=0.03)


def test_histogram_density_curse_of_dimensionality():
    """Verify exponential scaling of histogram bins M^D (Bishop 3.5.1, p. 100)."""
    M = 10  # 10 bins per dimension
    for D in [1, 2, 5, 10]:
        total_bins = M ** D
        assert total_bins == 10**D
    assert (10 ** 10) == 10_000_000_000  # 10 billion bins in 10D


def test_histogram_density_sampling():
    """Verify sampling from fitted HistogramDensity matches empirical bin frequencies."""
    data = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
    hist = HistogramDensity(bin_width=0.5, range_bounds=(0.0, 1.0)).fit(data)
    
    samples = hist.sample(size=10000, seed=42)
    assert len(samples) == 10000
    assert np.all((samples >= 0.0) & (samples <= 1.0))
    
    # 3 points in [0, 0.5] and 3 points in [0.5, 1.0], so probabilities are equal (0.5 each)
    frac_low = np.mean(samples < 0.5)
    assert np.isclose(frac_low, 0.5, atol=0.03)


# ---------------------------------------------------------------------
# 3.5.2 Kernel Densities Tests
# ---------------------------------------------------------------------

def test_kernel_density_normalization_1d():
    """Verify that Gaussian, Epanechnikov, and Boxcar kernels integrate to 1.0 (Eq 3.186)."""
    np.random.seed(42)
    data = np.array([0.2, 0.35, 0.7, 0.85])
    
    for kernel_name in ['gaussian', 'boxcar', 'epanechnikov']:
        kde = KernelDensity1D(h=0.1, kernel=kernel_name).fit(data)
        
        # Integrate from -1 to 2 using quad
        integral, _ = integrate.quad(lambda x: kde.evaluate(x), -1.0, 2.0, limit=100)
        assert np.isclose(integral, 1.0, atol=1e-4), f"Kernel {kernel_name} did not integrate to 1.0"
        
        # Test pdf and score_samples aliases
        p_eval = kde.pdf(0.35)
        log_p_eval = kde.score_samples(0.35)
        assert np.isclose(np.log(p_eval), log_p_eval, atol=1e-7)


def test_kernel_density_bandwidth_limits():
    """Verify behavior for small and large bandwidths h (Bishop p. 102)."""
    data = np.array([0.2, 0.8])
    
    # Very small h (sharp peaks at points, near zero midway)
    kde_small = KernelDensity1D(h=0.005, kernel="gaussian").fit(data)
    mid_val = kde_small.pdf(0.5)
    peak_val = kde_small.pdf(0.2)
    assert mid_val < 1e-10
    assert peak_val > 10.0
    
    # Very large h (over-smoothed, peak and mid-val become similar)
    kde_large = KernelDensity1D(h=2.0, kernel="gaussian").fit(data)
    mid_val_large = kde_large.pdf(0.5)
    peak_val_large = kde_large.pdf(0.2)
    assert np.isclose(mid_val_large, peak_val_large, rtol=0.1)


def test_kernel_density_nd():
    """Verify multidimensional Gaussian KDE (Eq 3.184)."""
    np.random.seed(42)
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    kde_2d = KernelDensityND(h=0.5).fit(X)
    
    # Symmetrical evaluation
    dens_origin = kde_2d.evaluate(np.array([[0.0, 0.0]]))
    dens_corner = kde_2d.evaluate(np.array([[1.0, 1.0]]))
    assert np.isclose(dens_origin, dens_corner, atol=1e-7)
    
    # Density decreases with distance
    dens_far = kde_2d.evaluate(np.array([[5.0, 5.0]]))
    assert dens_far < 1e-10


# ---------------------------------------------------------------------
# 3.5.3 Nearest Neighbours Tests
# ---------------------------------------------------------------------

def test_knn_density_estimator_1d():
    """Verify 1D KNN density formula p(x) = K / (2 * N * r_K(x)) (Eq 3.180)."""
    data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    N = len(data)
    K = 2
    knn = KNNDensityEstimator(K=K).fit(data)
    
    # At query point x = 0.25:
    # distances to data: |0.25 - 0.2| = 0.05, |0.25 - 0.3| = 0.05, |0.25 - 0.1| = 0.15...
    # Sorted distances: [0.05, 0.05, 0.15, ...]
    # 2nd nearest neighbor distance r_2 = 0.05
    # Volume in 1D: V = 2 * r_2 = 0.10
    # Expected density: p(0.25) = K / (N * V) = 2 / (5 * 0.10) = 4.0
    dens = knn.evaluate(0.25)
    assert np.isclose(dens, 4.0, atol=1e-5)
    
    # Test aliases
    assert np.isclose(knn.pdf(0.25), 4.0, atol=1e-5)
    assert np.isclose(knn.score_samples(0.25), np.log(4.0), atol=1e-5)


def test_knn_density_spatial_divergence():
    """Verify that KNN density tails decay as 1/|x|, so spatial integral diverges."""
    data = np.array([0.0])
    knn = KNNDensityEstimator(K=1).fit(data)
    
    # In 1D with 1 point at 0: p(x) = 1 / (2 * |x|)
    xs = np.array([10.0, 100.0, 1000.0])
    dens = knn.evaluate(xs)
    expected = 1.0 / (2.0 * xs)
    assert np.allclose(dens, expected, rtol=1e-5)
    # Integral of 1/|x| is logarithmic, diverging to infinity as x -> inf.


def test_knn_classifier_bayes_posteriors():
    """Verify KNN classifier posteriors sum to 1.0 (Eq 3.190)."""
    X = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],  # Class 0 (Red)
        [1.0, 1.0], [1.1, 1.0], [1.0, 1.1],  # Class 1 (Blue)
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    clf = KNNClassifier(K=3).fit(X, y)
    
    # Near Class 0: posterior for class 0 must be 1.0
    p0 = clf.predict_proba(np.array([[0.05, 0.05]]))
    assert np.isclose(p0[0, 0], 1.0)
    assert np.isclose(p0[0, 1], 0.0)
    assert np.sum(p0) == 1.0
    
    # Prediction matches majority vote
    pred = clf.predict(np.array([[0.05, 0.05]]))
    assert pred[0] == 0


def test_nearest_neighbour_rule_and_cover_hart_bound():
    """Verify K=1 rule and validate theoretical Cover-Hart (1967) bound."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    
    clf1 = KNNClassifier(K=1).fit(X, y)
    assert clf1.predict(np.array([[0.2]]))[0] == 0
    assert clf1.predict(np.array([[2.8]]))[0] == 1
    
    # Cover-Hart bound: P* <= P_NN <= 2*P* - (C/(C-1))*(P*)^2 <= 2*P*
    for P_star in [0.05, 0.10, 0.20]:
        C = 2
        upper_bound = 2 * P_star - (C / (C - 1)) * (P_star ** 2)
        assert upper_bound <= 2 * P_star
        assert P_star <= upper_bound


# ---------------------------------------------------------------------
# Figure Reproduction and Asset Verification Tests
# ---------------------------------------------------------------------

def test_figures_3_13_to_3_16_generation(tmp_path):
    """Verify that all four plotting functions execute cleanly and generate files."""
    f13, ax13 = plot_figure_3_13(save_paths=[str(tmp_path / "fig3_13.png")])
    assert os.path.exists(tmp_path / "fig3_13.png")
    assert len(ax13) == 3
    
    f14, ax14 = plot_figure_3_14(save_paths=[str(tmp_path / "fig3_14.png")])
    assert os.path.exists(tmp_path / "fig3_14.png")
    assert len(ax14) == 3
    
    f15, ax15 = plot_figure_3_15(save_paths=[str(tmp_path / "fig3_15.png")])
    assert os.path.exists(tmp_path / "fig3_15.png")
    assert len(ax15) == 3
    
    f16, (ax16_1, ax16_2) = plot_figure_3_16(save_paths=[str(tmp_path / "fig3_16.png")])
    assert os.path.exists(tmp_path / "fig3_16.png")


def test_saved_figures_exist():
    """Verify that Figures 3.13 through 3.16 exist in both 3/result/ and result/."""
    expected_files = [
        'fig3_13_histogram_density.png',
        'fig3_14_kernel_density.png',
        'fig3_15_knn_density.png',
        'fig3_16_knn_classifier.png'
    ]
    for fn in expected_files:
        p1 = os.path.join('3', 'result', fn)
        p2 = os.path.join('result', fn)
        assert os.path.exists(p1), f"Missing {p1}"
        assert os.path.exists(p2), f"Missing {p2}"
        assert os.path.getsize(p1) > 1000, f"File {p1} is too small"
        assert os.path.getsize(p2) > 1000, f"File {p2} is too small"
