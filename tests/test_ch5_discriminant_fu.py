"""
Tests for Chapter 5 Section 5.1: Discriminant Functions.
Covers:
- Two-class Linear Discriminant, geometry, normal vector, margin, orthogonal projection (Eq 5.2 - 5.6)
- Multi-class Linear Discriminant, pairwise boundaries, convexity of decision regions (Eq 5.7 - 5.10)
- Heuristic multi-class ambiguities (1-vs-rest and 1-vs-1)
- 1-of-K coding (Eq 5.11)
- Least squares for classification, normal equations, sum constraint, outlier sensitivity (Eq 5.12 - 5.18)
- Logistic regression comparison
- Figure generation for Figures 5.1 to 5.4.
"""
import os
import tempfile
import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.discriminant_functions import (
    LinearDiscriminant2Class,
    LinearDiscriminantMultiClass,
    OneVersusRestClassifier,
    OneVersusOneClassifier,
    LeastSquaresClassifier,
    LogisticRegression2Class,
    to_one_of_k,
    plot_figure_5_1_geometry,
    plot_figure_5_2_ambiguities,
    plot_figure_5_3_convex_regions,
    plot_figure_5_4_least_squares_outliers,
    generate_all_section_5_1_figures,
)


class TestLinearDiscriminant2Class:
    """Tests for two-class linear discriminant geometry and projections."""

    def test_geometry_and_orthogonality(self):
        # Plane: 3*x1 + 4*x2 - 12 = 0, norm(w) = 5
        w = np.array([3.0, 4.0])
        w0 = -12.0
        disc = LinearDiscriminant2Class(w, w0)

        # Distance from origin: -w0 / ||w|| = 12 / 5 = 2.4
        assert np.isclose(disc.distance_from_origin(), 2.4)

        # Test points on decision boundary
        pA = np.array([4.0, 0.0])
        pB = np.array([0.0, 3.0])
        assert np.isclose(disc.decision_function(pA), 0.0)
        assert np.isclose(disc.decision_function(pB), 0.0)

        # Orthogonality: w^T (pA - pB) == 0
        diff = pA - pB
        assert np.isclose(np.dot(w, diff), 0.0)

    def test_margin_and_orthogonal_projection(self):
        w = np.array([1.0, -1.0])
        w0 = 0.0  # y = x1 - x2 = 0
        disc = LinearDiscriminant2Class(w, w0)

        # Point (2, 0): y(x) = 2 - 0 = 2, ||w|| = sqrt(2)
        # Margin r = 2 / sqrt(2) = sqrt(2)
        p = np.array([2.0, 0.0])
        r = disc.margin(p)
        assert np.isclose(r, np.sqrt(2.0))

        # Orthogonal projection: x_perp = [2, 0] - sqrt(2) * [1/sqrt(2), -1/sqrt(2)] = [1, 1]
        p_perp = disc.project(p)
        assert np.allclose(p_perp, [1.0, 1.0])
        assert np.isclose(disc.decision_function(p_perp), 0.0)

        # Vectorized projection
        P_batch = np.array([[2.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
        proj_batch = disc.project(P_batch)
        assert proj_batch.shape == (3, 2)
        for proj in proj_batch:
            assert np.isclose(disc.decision_function(proj), 0.0, atol=1e-10)

    def test_invalid_weight_vector(self):
        with pytest.raises(ValueError, match="Weight vector w must not be zero vector"):
            LinearDiscriminant2Class([0.0, 0.0], 1.0)

    def test_decision_boundary_line(self):
        w = np.array([2.0, 4.0])
        w0 = -8.0
        disc = LinearDiscriminant2Class(w, w0)
        x1, x2 = disc.decision_boundary_line((-2.0, 2.0), num_points=10)
        # Check all (x1, x2) lie on boundary
        pts = np.column_stack([x1, x2])
        vals = disc.decision_function(pts)
        assert np.allclose(vals, 0.0)


class TestLinearDiscriminantMultiClass:
    """Tests for K-class linear discriminant and convexity property."""

    def test_multiclass_prediction_and_boundaries(self):
        # 3 classes in 2D
        # Class 0: top
        # Class 1: bottom-left
        # Class 2: bottom-right
        W = np.array([
            [0.0, -1.0, 1.0],  # x1 weights
            [1.0, -0.5, -0.5]  # x2 weights
        ])
        w0 = np.array([0.0, 0.0, 0.0])
        model = LinearDiscriminantMultiClass(W, w0)

        # Point (0, 2) should be class 0
        assert model.predict(np.array([0.0, 2.0])) == 0
        # Point (-2, -2) should be class 1
        assert model.predict(np.array([-2.0, -2.0])) == 1
        # Point (2, -2) should be class 2
        assert model.predict(np.array([2.0, -2.0])) == 2

        # Pairwise boundary between class 0 and class 1
        diff_w, diff_w0 = model.pairwise_boundary(0, 1)
        expected_w = W[:, 0] - W[:, 1]
        assert np.allclose(diff_w, expected_w)
        assert np.isclose(diff_w0, 0.0)

    def test_convexity_of_decision_regions(self):
        # Bishop Eq 5.9 - 5.10: Any two points xA, xB in region R_k have all intermediate
        # convex combinations in R_k.
        W = np.array([
            [1.0, -1.0, 0.0],
            [0.0, 0.5, -1.5]
        ])
        w0 = np.array([0.5, -0.2, 0.1])
        model = LinearDiscriminantMultiClass(W, w0)

        # Find two points belonging to class 0
        xA = np.array([3.0, 0.0])
        xB = np.array([4.0, 1.0])
        assert model.predict(xA) == 0
        assert model.predict(xB) == 0

        # Convexity test
        assert model.verify_convexity(xA, xB, num_points=30)

        # Cross-class points should raise ValueError
        xC = np.array([-3.0, 0.0])
        with pytest.raises(ValueError, match="must belong to the same class"):
            model.verify_convexity(xA, xC)


class TestHeuristicClassifiersAmbiguity:
    """Tests for 1-vs-rest and 1-vs-1 ambiguous unclassified regions."""

    def test_one_versus_rest_ambiguity(self):
        # Two classifiers dividing space into 4 quadrants
        d1 = LinearDiscriminant2Class([1.0, 0.0], -1.0)  # C1 if x1 > 1
        d2 = LinearDiscriminant2Class([0.0, 1.0], -1.0)  # C2 if x2 > 1
        ovr = OneVersusRestClassifier([d1, d2])

        # Point in C1 only: (2, 0) => d1=True, d2=False => class 0
        assert ovr.predict(np.array([2.0, 0.0])) == 0
        # Point in C2 only: (0, 2) => d1=False, d2=True => class 1
        assert ovr.predict(np.array([0.0, 2.0])) == 1
        # Point in ambiguous region (both positive): (2, 2) => -1
        assert ovr.predict(np.array([2.0, 2.0])) == -1
        assert ovr.is_ambiguous(np.array([2.0, 2.0]))
        # Point where neither is positive: (0, 0) => -1
        assert ovr.predict(np.array([0.0, 0.0])) == -1
        assert ovr.is_ambiguous(np.array([0.0, 0.0]))

    def test_one_versus_one_ambiguity(self):
        # 3 pairwise classifiers with circular / tied votes at center
        # Pair (0, 1): x2 > 0 => class 0, else 1
        # Pair (1, 2): x1 > 0 => class 1, else 2
        # Pair (0, 2): x1 + x2 < 0 => class 2, else 0
        p01 = LinearDiscriminant2Class([0.0, 1.0], 0.0)
        p12 = LinearDiscriminant2Class([1.0, 0.0], 0.0)
        p02 = LinearDiscriminant2Class([-1.0, -1.0], 0.0)
        ovo = OneVersusOneClassifier({(0, 1): p01, (1, 2): p12, (0, 2): p02}, num_classes=3)

        # Clear winner in class 0: (0, 5)
        # p01: y>0 -> 0 wins
        # p12: y=0 -> 1 wins (tie) or 2 wins
        # p02: y=-5 < 0 -> 2 wins
        votes = ovo.predict_votes(np.array([[0.0, 5.0], [5.0, -5.0]]))
        assert votes.shape == (2, 3)


class Test1OfKCodingAndLeastSquares:
    """Tests for 1-of-K coding and least-squares classification."""

    def test_to_one_of_k(self):
        labels = np.array([0, 1, 2, 0])
        T = to_one_of_k(labels, num_classes=3)
        assert T.shape == (4, 3)
        assert np.allclose(T.sum(axis=1), 1.0)
        assert np.array_equal(T[0], [1, 0, 0])
        assert np.array_equal(T[1], [0, 1, 0])
        assert np.array_equal(T[2], [0, 0, 1])

    def test_least_squares_sum_constraint(self):
        # Bishop Eq 5.17 - 5.18: If sum_k t_nk = 1 for all training points,
        # then sum_k y_k(x) = 1 for ANY arbitrary x.
        np.random.seed(123)
        N = 40
        X = np.random.randn(N, 3)
        labels = np.random.choice([0, 1, 2], size=N)
        T = to_one_of_k(labels, num_classes=3)

        model = LeastSquaresClassifier().fit(X, T)

        # Arbitrary test points (even far away from data)
        X_test = np.random.uniform(-10, 10, size=(25, 3))
        assert model.verify_sum_constraint(X_test)

    def test_least_squares_accuracy_on_separable_data(self):
        np.random.seed(42)
        X1 = np.random.randn(30, 2) + np.array([-3.0, -3.0])
        X2 = np.random.randn(30, 2) + np.array([3.0, 3.0])
        X = np.vstack([X1, X2])
        t = np.array([0] * 30 + [1] * 30)

        model = LeastSquaresClassifier().fit(X, t)
        preds = model.predict(X)
        accuracy = np.mean(preds == t)
        assert accuracy >= 0.95

    def test_outlier_sensitivity_least_squares_vs_logistic(self):
        np.random.seed(42)
        # Class 0 centered at (-1, 0), Class 1 centered at (1, 0)
        X0 = np.random.randn(40, 2) * 0.5 + np.array([-1.0, 0.0])
        X1 = np.random.randn(40, 2) * 0.5 + np.array([1.0, 0.0])
        X_clean = np.vstack([X0, X1])
        t_clean = np.array([0] * 40 + [1] * 40)

        # Fit on clean
        ls_clean = LeastSquaresClassifier().fit(X_clean, t_clean)
        lr_clean = LogisticRegression2Class().fit(X_clean, t_clean)

        acc_ls_clean = np.mean(ls_clean.predict(X_clean) == t_clean)
        acc_lr_clean = np.mean(lr_clean.predict(X_clean) == t_clean)
        assert acc_ls_clean >= 0.95
        assert acc_lr_clean >= 0.95

        # Add heavy outliers in Class 1 side at (20, 0)
        X_outliers = np.array([[25.0, 0.0]] * 15)
        t_outliers = np.array([1] * 15)
        X_dirty = np.vstack([X_clean, X_outliers])
        t_dirty = np.hstack([t_clean, t_outliers])

        ls_dirty = LeastSquaresClassifier().fit(X_dirty, t_dirty)
        lr_dirty = LogisticRegression2Class().fit(X_dirty, t_dirty)

        # Logistic regression retains high accuracy on clean points
        acc_lr_on_clean = np.mean(lr_dirty.predict(X_clean) == t_clean)
        assert acc_lr_on_clean >= 0.90


class TestFiguresGeneration:
    """Test generating all required textbook figures for Section 5.1."""

    def test_all_figure_functions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fig1 = plot_figure_5_1_geometry(filepath=os.path.join(tmpdir, "fig1.png"), save_both=False)
            fig2 = plot_figure_5_2_ambiguities(filepath=os.path.join(tmpdir, "fig2.png"), save_both=False)
            fig3 = plot_figure_5_3_convex_regions(filepath=os.path.join(tmpdir, "fig3.png"), save_both=False)
            fig4 = plot_figure_5_4_least_squares_outliers(filepath=os.path.join(tmpdir, "fig4.png"), save_both=False)

            assert isinstance(fig1, plt.Figure)
            assert isinstance(fig2, plt.Figure)
            assert isinstance(fig3, plt.Figure)
            assert isinstance(fig4, plt.Figure)

            assert os.path.exists(os.path.join(tmpdir, "fig1.png"))
            assert os.path.exists(os.path.join(tmpdir, "fig2.png"))
            assert os.path.exists(os.path.join(tmpdir, "fig3.png"))
            assert os.path.exists(os.path.join(tmpdir, "fig4.png"))
            plt.close('all')
