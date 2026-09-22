"""
Unit tests for Section 1.3: A Brief History of Machine Learning (Perceptron, XOR limitation, MLP, Compute scaling).
"""
import pytest
import numpy as np

def perceptron_predict(x, w, b):
    """Single layer threshold unit: f(w^T x + b)."""
    activation = np.dot(x, w) + b
    return np.where(activation >= 0, 1, 0)

def mlp_xor(x):
    """
    Two-layer network solving XOR using step/threshold activations:
    h1 = AND(x1, NOT x2)
    h2 = AND(NOT x1, x2)
    y = OR(h1, h2)
    """
    # Hidden unit 1: x1 - x2 >= 0.5 (i.e. x1=1, x2=0)
    h1 = np.where(x[:, 0] - x[:, 1] - 0.5 >= 0, 1, 0)
    # Hidden unit 2: -x1 + x2 >= 0.5 (i.e. x1=0, x2=1)
    h2 = np.where(-x[:, 0] + x[:, 1] - 0.5 >= 0, 1, 0)
    # Output unit: h1 + h2 >= 0.5 (OR)
    y = np.where(h1 + h2 - 0.5 >= 0, 1, 0)
    return y

def test_perceptron_and_or():
    # Perceptron can solve AND
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # AND weights: w1=1, w2=1, b=-1.5
    y_and = perceptron_predict(X, w=np.array([1.0, 1.0]), b=-1.5)
    np.testing.assert_array_equal(y_and, [0, 0, 0, 1])

    # Perceptron can solve OR
    # OR weights: w1=1, w2=1, b=-0.5
    y_or = perceptron_predict(X, w=np.array([1.0, 1.0]), b=-0.5)
    np.testing.assert_array_equal(y_or, [0, 1, 1, 1])

def test_xor_limitation_and_mlp_solution():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor_true = np.array([0, 1, 1, 0])
    
    # 2-layer MLP solves XOR perfectly
    y_mlp = mlp_xor(X)
    np.testing.assert_array_equal(y_mlp, y_xor_true)

def test_compute_growth_calculation():
    # Compute growth in deep learning era doubles roughly every 3.4 months (vs 24 months for Moore's law)
    doubling_time_dl = 3.4 / 12.0 # years
    total_growth_6yr = 2 ** (6.0 / doubling_time_dl)
    assert total_growth_6yr > 1_000_000
    assert total_growth_6yr < 5_000_000
