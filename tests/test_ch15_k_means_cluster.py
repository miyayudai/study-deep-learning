"""
Tests for Chapter 15 Section 15.1: K-means Clustering
======================================================
Covers:
- KMeans class (fit, predict, transform, inertia, convergence)
- Monotonic decrease of distortion measure J (Eq. 15.1)
- Initializations: 'random', 'k-means++', custom centers
- Sequential / online K-means update (Eq. 15.4)
- 2D perpendicular bisector calculation
- Image segmentation and compression metrics (Section 15.1.1)
- Figure generation for Figures 15.1, 15.2, and 15.3
"""

import os
import tempfile
from pathlib import Path
import numpy as np
import pytest

from common.kmeans_clustering import (
    KMeans,
    compute_distortion,
    sequential_kmeans_update,
    get_perpendicular_bisector,
    image_segmentation_kmeans,
    load_faithful_dataset,
    generate_figure_15_1,
    generate_figure_15_2,
    generate_figure_15_3,
)


def test_kmeans_basic_clustering():
    """Test K-means on clearly separated 2D isotropic clusters."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(50, 2) + np.array([-5.0, -5.0])
    c2 = rng.randn(50, 2) + np.array([5.0, 5.0])
    X = np.vstack([c1, c2])

    kmeans = KMeans(n_clusters=2, random_state=42, init='k-means++')
    kmeans.fit(X)

    assert kmeans.cluster_centers_.shape == (2, 2)
    assert kmeans.labels_.shape == (100,)
    # Verify accurate separation (ground truth labels 0 and 1)
    acc = np.mean(kmeans.labels_[:50] == kmeans.labels_[0])
    assert acc > 0.95
    assert np.mean(kmeans.labels_[50:] != kmeans.labels_[0]) > 0.95

    # Check predict and transform
    test_pts = np.array([[-5.0, -5.0], [5.0, 5.0]])
    preds = kmeans.predict(test_pts)
    assert preds[0] != preds[1]

    dists = kmeans.transform(test_pts)
    assert dists.shape == (2, 2)
    assert dists[0, preds[0]] < dists[0, 1 - preds[0]]


def test_distortion_monotonic_decrease_on_faithful():
    """Verify that distortion J decreases monotonically at every half-step."""
    X_std, _ = load_faithful_dataset()
    mu1_init = np.array([-1.5, 1.0])
    mu2_init = np.array([1.5, -1.0])
    init_centers = np.array([mu1_init, mu2_init])

    kmeans = KMeans(n_clusters=2, max_iter=10, init=init_centers)
    kmeans.fit(X_std)

    distortions = [item[1] for item in kmeans.history_['distortion']]
    assert len(distortions) >= 8

    # Monotonic non-increasing property: J_{t+1} <= J_t
    for i in range(len(distortions) - 1):
        assert distortions[i + 1] <= distortions[i] + 1e-9

    # Check compute_distortion utility matches final inertia
    final_J = compute_distortion(X_std, kmeans.cluster_centers_, kmeans.labels_)
    assert np.isclose(final_J, kmeans.inertia_, atol=1e-5)


def test_kmeans_init_options():
    """Test initialization methods: random, k-means++, custom, and invalid."""
    X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]])

    km_rand = KMeans(n_clusters=2, init='random', random_state=0).fit(X)
    assert km_rand.cluster_centers_.shape == (2, 2)

    km_plus = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(X)
    assert km_plus.cluster_centers_.shape == (2, 2)

    custom_centers = np.array([[0.0, 0.0], [10.0, 10.0]])
    km_custom = KMeans(n_clusters=2, init=custom_centers).fit(X)
    assert km_custom.cluster_centers_.shape == (2, 2)

    with pytest.raises(ValueError):
        KMeans(n_clusters=3, init=custom_centers).fit(X)

    with pytest.raises(ValueError):
        KMeans(n_clusters=2, init='invalid_method').fit(X)


def test_unfitted_model_raises():
    """Calling predict or transform before fit should raise RuntimeError."""
    km = KMeans(n_clusters=2)
    with pytest.raises(RuntimeError):
        km.predict(np.zeros((5, 2)))
    with pytest.raises(RuntimeError):
        km.transform(np.zeros((5, 2)))


def test_sequential_kmeans_update():
    """Test online update formula mu = mu + eta * (x - mu)."""
    mu = np.array([2.0, 4.0])
    x = np.array([6.0, 8.0])
    eta = 0.5
    updated = sequential_kmeans_update(mu, x, eta)
    expected = np.array([4.0, 6.0])
    assert np.allclose(updated, expected)

    # For running average: mu_N = mu_{N-1} + (1/N) * (x_N - mu_{N-1})
    samples = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    online_mean = samples[0].copy()
    for n in range(1, len(samples)):
        online_mean = sequential_kmeans_update(online_mean, samples[n], 1.0 / (n + 1))
    assert np.allclose(online_mean, np.mean(samples, axis=0))


def test_perpendicular_bisector():
    """Test perpendicular bisector coordinate generation."""
    # Case 1: Centered horizontally symmetric centers (-1, 0) and (1, 0)
    mu1 = np.array([-1.0, 0.0])
    mu2 = np.array([1.0, 0.0])
    x_vals, y_vals = get_perpendicular_bisector(mu1, mu2, (-2, 2))
    # Midpoint is (0, 0), normal is (2, 0) along x-axis => vertical line x = 0
    assert np.allclose(x_vals, 0.0)

    # Case 2: 45 degree line: centers (-1, 1) and (1, -1)
    # Midpoint is (0, 0). Vector delta = (2, -2). Bisector line: y = x.
    mu1 = np.array([-1.0, 1.0])
    mu2 = np.array([1.0, -1.0])
    x_vals, y_vals = get_perpendicular_bisector(mu1, mu2, (-2, 2))
    assert np.allclose(x_vals, y_vals)


def test_image_segmentation():
    """Test image segmentation function on synthetic and real images."""
    # Create synthetic test image (20x20 with 2 distinct colors)
    test_img = np.zeros((20, 20, 3), dtype=np.uint8)
    test_img[:10, :] = [255, 0, 0]    # Red
    test_img[10:, :] = [0, 0, 255]    # Blue

    seg, palette, labels, info = image_segmentation_kmeans(test_img, K=2, random_state=42)

    assert seg.shape == (20, 20, 3)
    assert seg.dtype == np.uint8
    assert palette.shape == (2, 3)
    assert labels.shape == (20, 20)
    assert len(np.unique(palette, axis=0)) == 2
    assert info['compression_ratio'] > 1.0
    assert 0.0 <= info['space_savings_percent'] <= 100.0

    # Test with segmentation_source image if available
    img_path = Path('common/data/segmentation_source.png')
    if img_path.exists():
        seg_real, pal_real, _, info_real = image_segmentation_kmeans(
            img_path, K=3, max_iter=10, random_state=42
        )
        assert seg_real.shape == (480, 360, 3)
        assert pal_real.shape == (3, 3)
        assert info_real['compression_ratio'] > 5.0  # 24 bits down to 2 bits + palette


def test_figure_generators_save_valid_files():
    """Verify that Figure 15.1, 15.2, and 15.3 generate and save non-empty files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fig1_path = Path(tmpdir) / 'test_fig_15_1.png'
        fig2_path = Path(tmpdir) / 'test_fig_15_2.png'
        fig3_path = Path(tmpdir) / 'test_fig_15_3.png'

        generate_figure_15_1(fig1_path)
        assert fig1_path.exists()
        assert fig1_path.stat().st_size > 50000

        generate_figure_15_2(fig2_path)
        assert fig2_path.exists()
        assert fig2_path.stat().st_size > 10000

        generate_figure_15_3(fig3_path)
        assert fig3_path.exists()
        assert fig3_path.stat().st_size > 50000
