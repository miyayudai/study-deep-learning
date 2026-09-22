"""
Unit tests for Section 1.2: A Tutorial Example (Polynomial Curve Fitting & Regularization).
"""
import pytest
import numpy as np
from common.polynomial import (
    PolynomialFeatures,
    PolynomialRegression,
    generate_synthetic_data,
    k_fold_cross_validation
)

def test_synthetic_data_generation():
    x, t = generate_synthetic_data(n_samples=10, noise_std=0.2, random_state=42)
    assert len(x) == 10
    assert len(t) == 10
    assert x[0] == 0.0
    assert x[-1] == 1.0
    # Mean of targets should be roughly around 0 since sin(2*pi*x) averages to 0
    assert abs(np.mean(t)) < 0.5

def test_polynomial_features():
    poly = PolynomialFeatures(degree=3)
    x = np.array([0.0, 0.5, 2.0])
    Phi = poly.transform(x)
    assert Phi.shape == (3, 4)
    # Check powers: x^0, x^1, x^2, x^3
    np.testing.assert_allclose(Phi[0], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(Phi[1], [1.0, 0.5, 0.25, 0.125])
    np.testing.assert_allclose(Phi[2], [1.0, 2.0, 4.0, 8.0])

def test_polynomial_regression_exact_interpolation():
    # An N-point dataset can be perfectly interpolated by an (N-1) degree polynomial
    x = np.array([0.1, 0.4, 0.7, 0.9])
    t = np.array([1.2, -0.5, 2.3, 0.1])
    model = PolynomialRegression(degree=3, l2_reg=0.0).fit(x, t)
    y_pred = model.predict(x)
    np.testing.assert_allclose(y_pred, t, atol=1e-5)
    assert model.rmse(x, t) < 1e-5

def test_overfitting_and_weight_decay():
    x, t = generate_synthetic_data(n_samples=10, noise_std=0.25, random_state=0)
    # Unregularized high-degree model (M=9)
    model_unreg = PolynomialRegression(degree=9, l2_reg=0.0).fit(x, t)
    norm_unreg = np.linalg.norm(model_unreg.weights_)
    
    # Regularized model with lambda = 1.0
    model_reg = PolynomialRegression(degree=9, l2_reg=1.0).fit(x, t)
    norm_reg = np.linalg.norm(model_reg.weights_)
    
    # Regularization should drastically reduce the L2 norm of weights
    assert norm_reg < norm_unreg
    
    # Even moderate lambda = 1e-3 should reduce weight norm
    model_reg_small = PolynomialRegression(degree=9, l2_reg=1e-3).fit(x, t)
    norm_small = np.linalg.norm(model_reg_small.weights_)
    assert norm_small < norm_unreg

def test_k_fold_cross_validation():
    x, t = generate_synthetic_data(n_samples=12, noise_std=0.2, random_state=1)
    k = 4
    cv_res = k_fold_cross_validation(x, t, k=k, degree=3, l2_reg=1e-2, random_state=42)
    assert len(cv_res['train_rmses']) == k
    assert len(cv_res['val_rmses']) == k
    assert cv_res['mean_val_rmse'] > 0
    assert cv_res['mean_train_rmse'] > 0
