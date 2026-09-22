"""
Unit tests for Section 1.1: The Impact of Deep Learning (Medical diagnosis, Protein structure, Image synthesis).
"""
import pytest
import numpy as np

def compute_sensitivity_specificity(y_true, y_pred):
    """Compute sensitivity (recall) and specificity for medical diagnosis."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity

def compute_protein_rmsd(coords_true, coords_pred):
    """Compute Root-Mean-Square Deviation (RMSD) between 3D protein coordinates."""
    diff = coords_true - coords_pred
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=-1)))

def slerp(val, low, high):
    """Spherical linear interpolation for latent space exploration."""
    low_norm = low / np.linalg.norm(low)
    high_norm = high / np.linalg.norm(high)
    omega = np.arccos(np.clip(np.dot(low_norm, high_norm), -1.0, 1.0))
    so = np.sin(omega)
    if abs(so) < 1e-6:
        return (1.0 - val) * low + val * high
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high

def test_medical_diagnosis_metrics():
    # True labels: 1 = malignant melanoma, 0 = benign nevus
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    # Model predictions
    y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 0])
    
    sens, spec = compute_sensitivity_specificity(y_true, y_pred)
    assert sens == pytest.approx(2/3)
    assert spec == pytest.approx(4/5)

def test_protein_rmsd():
    coords_true = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    coords_pred = coords_true + 0.1
    rmsd = compute_protein_rmsd(coords_true, coords_pred)
    assert rmsd == pytest.approx(np.sqrt(0.03))

def test_slerp_interpolation():
    z1 = np.array([1.0, 0.0, 0.0])
    z2 = np.array([0.0, 1.0, 0.0])
    z_mid = slerp(0.5, z1, z2)
    assert np.linalg.norm(z_mid) == pytest.approx(1.0)
    assert z_mid[0] == pytest.approx(z_mid[1])
    assert z_mid[2] == pytest.approx(0.0)
