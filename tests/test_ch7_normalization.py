"""Tests for Chapter 7 Section 7.4: Normalization.

Tests cover:
- Section 7.4.1: Data Normalization (Eq 7.48 - 7.50)
- Section 7.4.2: Batch Normalization (Eq 7.51 - 7.57)
- Section 7.4.3: Layer Normalization (Eq 7.58 - 7.60)
- Figure 7.7 and Figure 7.8 generation
"""

import os
import numpy as np
import pytest

from common.normalization import (
    BatchNormalization,
    DataStandardizer,
    LayerNormalization,
    generate_figure_7_7,
    generate_figure_7_8,
    simulate_jacobian_norm_propagation,
)


def test_data_standardizer():
    """Verify input feature scaling to zero mean and unit variance (Eq 7.48 - 7.50)."""
    rng = np.random.default_rng(42)
    N = 100
    X = rng.normal(loc=[5.0, -10.0], scale=[2.0, 15.0], size=(N, 2))
    
    scaler = DataStandardizer()
    with pytest.raises(ValueError):
        scaler.transform(X)
        
    X_scaled = scaler.fit_transform(X)
    
    # Check zero mean and unit variance
    assert np.allclose(np.mean(X_scaled, axis=0), [0.0, 0.0], atol=1e-7)
    assert np.allclose(np.var(X_scaled, axis=0), [1.0, 1.0], atol=1e-7)
    
    # Check inverse transform
    X_reconstructed = scaler.inverse_transform(X_scaled)
    assert np.allclose(X, X_reconstructed, atol=1e-7)


def test_batch_normalization_forward_training():
    """Verify Batch Normalization forward pass during training (Eq 7.52 - 7.55)."""
    rng = np.random.default_rng(123)
    K = 64  # batch size
    M = 16  # hidden units
    
    # Arbitrary raw pre-activations
    a = rng.normal(loc=3.0, scale=4.0, size=(K, M))
    
    bn = BatchNormalization(num_features=M, delta=1e-5, momentum=0.9)
    out = bn.forward(a, training=True)
    
    assert out.shape == (K, M)
    # Mean across batch (axis 0) should be close to beta (0.0)
    assert np.allclose(np.mean(out, axis=0), np.zeros(M), atol=1e-4)
    # Variance across batch should be close to gamma^2 (1.0)
    assert np.allclose(np.var(out, axis=0), np.ones(M), atol=1e-3)
    
    # Running statistics should be initialized
    assert bn.initialized_running_stats is True
    assert np.allclose(bn.running_mean, np.mean(a, axis=0), atol=1e-4)


def test_batch_normalization_inference():
    """Verify Batch Normalization uses running statistics during inference (Eq 7.56, 7.57)."""
    rng = np.random.default_rng(456)
    K = 32
    M = 8
    
    bn = BatchNormalization(num_features=M, delta=1e-5, momentum=0.9)
    # Train on 10 batches to accumulate running statistics
    for _ in range(10):
        a_batch = rng.normal(loc=2.0, scale=3.0, size=(K, M))
        bn.forward(a_batch, training=True)
        
    running_mean_before = bn.running_mean.copy()
    
    # Single sample inference
    a_test = rng.normal(loc=2.0, scale=3.0, size=(1, M))
    out_test = bn.forward(a_test, training=False)
    
    assert out_test.shape == (1, M)
    # Running stats must NOT change during test time
    assert np.allclose(bn.running_mean, running_mean_before)


def test_batch_normalization_gradient_check():
    """Verify analytical gradients of BatchNorm vs finite differences."""
    rng = np.random.default_rng(789)
    K = 8
    M = 4
    
    bn = BatchNormalization(num_features=M)
    a = rng.normal(size=(K, M))
    
    # Forward pass
    out = bn.forward(a, training=True)
    dout = rng.normal(size=(K, M))
    da_analytic = bn.backward(dout)
    
    # Numerical gradient check for da
    eps = 1e-6
    da_numeric = np.zeros_like(a)
    for i in range(K):
        for j in range(M):
            a_plus = a.copy()
            a_plus[i, j] += eps
            out_plus = bn.forward(a_plus, training=True)
            loss_plus = np.sum(out_plus * dout)
            
            a_minus = a.copy()
            a_minus[i, j] -= eps
            out_minus = bn.forward(a_minus, training=True)
            loss_minus = np.sum(out_minus * dout)
            
            da_numeric[i, j] = (loss_plus - loss_minus) / (2.0 * eps)
            
    assert np.allclose(da_analytic, da_numeric, atol=1e-4, rtol=1e-3)


def test_layer_normalization_forward():
    """Verify Layer Normalization normalizes across hidden units per sample (Eq 7.58 - 7.60)."""
    rng = np.random.default_rng(101)
    K = 20
    M = 30
    
    a = rng.normal(loc=5.0, scale=8.0, size=(K, M))
    ln = LayerNormalization(num_features=M)
    out = ln.forward(a)
    
    assert out.shape == (K, M)
    # Mean across hidden units (axis 1) should be 0.0 for every sample
    assert np.allclose(np.mean(out, axis=1), np.zeros(K), atol=1e-4)
    # Variance across hidden units should be 1.0 for every sample
    assert np.allclose(np.var(out, axis=1), np.ones(K), atol=1e-3)


def test_layer_normalization_gradient_check():
    """Verify analytical gradients of LayerNorm vs finite differences."""
    rng = np.random.default_rng(202)
    K = 6
    M = 5
    
    ln = LayerNormalization(num_features=M)
    a = rng.normal(size=(K, M))
    
    out = ln.forward(a)
    dout = rng.normal(size=(K, M))
    da_analytic = ln.backward(dout)
    
    eps = 1e-6
    da_numeric = np.zeros_like(a)
    for i in range(K):
        for j in range(M):
            a_plus = a.copy()
            a_plus[i, j] += eps
            out_plus = ln.forward(a_plus)
            loss_plus = np.sum(out_plus * dout)
            
            a_minus = a.copy()
            a_minus[i, j] -= eps
            out_minus = ln.forward(a_minus)
            loss_minus = np.sum(out_minus * dout)
            
            da_numeric[i, j] = (loss_plus - loss_minus) / (2.0 * eps)
            
    assert np.allclose(da_analytic, da_numeric, atol=1e-4, rtol=1e-3)


def test_jacobian_norm_propagation_simulation():
    """Verify that normalization stabilizes deep gradient Jacobian product (Eq 7.51)."""
    # Without normalization and weight_scale = 1.3: gradients explode exponentially
    norms_unnorm = simulate_jacobian_norm_propagation(
        depth=10, width=32, weight_scale=1.3, normalize=False, num_trials=30
    )
    # With normalization: gradients are kept stably bounded
    norms_norm = simulate_jacobian_norm_propagation(
        depth=10, width=32, weight_scale=1.3, normalize=True, num_trials=30
    )
    
    assert norms_unnorm[-1] > 5.0 * norms_unnorm[0]  # exploded
    assert norms_norm[-1] < 10.0  # bounded


def test_generate_figures_7_7_and_7_8():
    """Verify that Figures 7.7 and 7.8 are generated and saved."""
    p7, r7 = generate_figure_7_7()
    assert os.path.exists(p7)
    
    p8, r8 = generate_figure_7_8()
    assert os.path.exists(p8)
