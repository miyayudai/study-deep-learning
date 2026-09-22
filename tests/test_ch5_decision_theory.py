"""
Unit tests for Chapter 5, Section 5.2: Decision Theory for Classification.

Covers:
- Section 5.2.1: Misclassification Rate Minimization (Eq 5.20 - 5.21, Fig 5.5)
- Section 5.2.2: Expected Loss and Loss Matrix (Eq 5.22 - 5.23, Fig 5.6)
- Section 5.2.3: The Reject Option (Fig 5.7)
- Section 5.2.4: Inference & Decision Separation & Prior Compensation (Eq 5.24 - 5.27, Fig 5.8)
- Section 5.2.5: Classifier Accuracy & Confusion Matrix Metrics (Eq 5.28 - 5.37, Fig 5.9)
- Section 5.2.6: ROC Curves and AUC (Eq 5.38 - 5.39, Fig 5.10 - 5.11)
"""
import os
import pytest
import numpy as np

from common.classification_decision_theory import (
    optimal_decision_rule_misclassification,
    compute_expected_loss,
    optimal_decision_rule_expected_loss,
    decision_rule_with_reject,
    compensate_for_class_priors,
    ConfusionMatrix2Class,
    compute_roc_curve,
    plot_figure_5_5_joint_probabilities,
    plot_figure_5_6_loss_matrix,
    plot_figure_5_7_reject_option,
    plot_figure_5_8_class_densities_posteriors,
    plot_figure_5_9_confusion_matrix,
    plot_figure_5_10_roc_regions,
    plot_figure_5_11_roc_curve,
    generate_all_section_5_2_figures,
)


class TestMisclassificationMinimization:
    """Tests for Section 5.2.1: Misclassification rate minimization."""

    def test_optimal_decision_rule_misclassification_binary(self):
        # 3 points, 2 classes
        posteriors = np.array([
            [0.8, 0.2],
            [0.4, 0.6],
            [0.51, 0.49]
        ])
        decisions = optimal_decision_rule_misclassification(posteriors)
        assert np.array_equal(decisions, [0, 1, 0])

    def test_optimal_decision_rule_misclassification_multiclass(self):
        # 4 points, 3 classes
        posteriors = np.array([
            [0.2, 0.7, 0.1],
            [0.05, 0.15, 0.8],
            [0.4, 0.3, 0.3],
            [0.33, 0.34, 0.33]
        ])
        decisions = optimal_decision_rule_misclassification(posteriors)
        assert np.array_equal(decisions, [1, 2, 0, 1])

    def test_single_sample_misclassification(self):
        post = np.array([0.1, 0.9])
        decision = optimal_decision_rule_misclassification(post)
        assert decision == 1


class TestExpectedLoss:
    """Tests for Section 5.2.2: Expected loss and general loss matrices."""

    def test_zero_one_loss_equivalence(self):
        # Zero-one loss matrix: L_{kj} = 0 if k == j else 1
        K = 3
        zero_one_loss = 1.0 - np.eye(K)
        posteriors = np.array([
            [0.1, 0.7, 0.2],
            [0.4, 0.3, 0.3],
            [0.05, 0.1, 0.85]
        ])
        loss_decisions = optimal_decision_rule_expected_loss(posteriors, zero_one_loss)
        misclass_decisions = optimal_decision_rule_misclassification(posteriors)
        assert np.array_equal(loss_decisions, misclass_decisions)

    def test_asymmetric_cancer_loss_matrix(self):
        # Bishop Figure 5.6: L_{11}=0, L_{12}=1, L_{21}=100, L_{22}=0
        # Class 0: normal, Class 1: cancer
        loss_matrix = np.array([
            [0.0, 1.0],
            [100.0, 0.0]
        ])
        # Theoretical threshold for deciding cancer (class 1):
        # Expected loss for normal (j=0): 100 * p(cancer)
        # Expected loss for cancer (j=1): 1 * p(normal) = 1 * (1 - p(cancer))
        # Decide cancer if 1 - p(cancer) < 100 * p(cancer) <=> p(cancer) > 1 / 101 ~ 0.009901
        p_crit = 1.0 / 101.0

        # Slightly below threshold -> decide normal (0)
        p_below = np.array([1.0 - (p_crit - 0.002), p_crit - 0.002])
        dec_below = optimal_decision_rule_expected_loss(p_below, loss_matrix)
        assert dec_below == 0

        # Slightly above threshold -> decide cancer (1)
        p_above = np.array([1.0 - (p_crit + 0.002), p_crit + 0.002])
        dec_above = optimal_decision_rule_expected_loss(p_above, loss_matrix)
        assert dec_above == 1

    def test_invalid_loss_matrix_shape(self):
        with pytest.raises(ValueError):
            compute_expected_loss(np.array([0.5, 0.5]), np.zeros((2, 3)))


class TestRejectOption:
    """Tests for Section 5.2.3: The reject option."""

    def test_reject_option_thresholding(self):
        posteriors = np.array([
            [0.95, 0.05],  # max 0.95 >= 0.8 -> accept class 0
            [0.55, 0.45],  # max 0.55 < 0.8 -> reject
            [0.10, 0.90],  # max 0.90 >= 0.8 -> accept class 1
            [0.79, 0.21],  # max 0.79 < 0.8 -> reject
        ])
        decisions, rejected = decision_rule_with_reject(posteriors, theta=0.8, reject_label=-1)
        assert np.array_equal(rejected, [False, True, False, True])
        assert np.array_equal(decisions, [0, -1, 1, -1])

    def test_theta_extremes(self):
        posteriors = np.array([[0.6, 0.4], [0.8, 0.2]])
        # theta = 1.0 (strict) -> rejects all non-deterministic points
        _, rej_all = decision_rule_with_reject(posteriors, theta=1.0)
        assert np.all(rej_all)

        # theta <= 0.5 (binary) -> never rejects
        _, rej_none = decision_rule_with_reject(posteriors, theta=0.5)
        assert not np.any(rej_none)


class TestClassPriorCompensation:
    """Tests for Section 5.2.4: Compensating for modified class priors."""

    def test_compensate_for_class_priors_synthetic(self):
        # In training, classes are balanced 50:50: p_train = [0.5, 0.5]
        # In reality (screening), cancer is rare: p_target = [0.999, 0.001]
        p_train = np.array([0.5, 0.5])
        p_target = np.array([0.999, 0.001])

        # Model output from balanced training: p_train(C|x) = [0.5, 0.5]
        train_post = np.array([0.5, 0.5])
        target_post = compensate_for_class_priors(train_post, p_train, p_target)

        # Since model output is ambiguous (equal ratio), target posterior must equal target priors!
        assert np.allclose(target_post, p_target)
        assert np.isclose(np.sum(target_post), 1.0)

    def test_batch_prior_compensation(self):
        p_train = np.array([0.5, 0.5])
        p_target = np.array([0.8, 0.2])
        batch_posts = np.array([
            [0.5, 0.5],
            [0.8, 0.2]
        ])
        adj = compensate_for_class_priors(batch_posts, p_train, p_target)
        assert adj.shape == (2, 2)
        assert np.allclose(np.sum(adj, axis=1), [1.0, 1.0])


class TestConfusionMatrixAndMetrics:
    """Tests for Section 5.2.5: Classifier accuracy and confusion matrix metrics (Eq 5.28 - 5.39)."""

    def test_metrics_calculation(self):
        # 10 samples:
        # y_true: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0] (4 positive, 6 negative)
        # y_pred: [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
        # TP = 3, FN = 1, FP = 1, TN = 5
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])

        cm = ConfusionMatrix2Class(y_true, y_pred)
        assert cm.N == 10
        assert cm.TP == 3
        assert cm.FN == 1
        assert cm.FP == 1
        assert cm.TN == 5

        # Eq 5.28: N = TP + FP + TN + FN
        assert cm.N == cm.TP + cm.FP + cm.TN + cm.FN

        # Eq 5.29: Accuracy = (TP + TN) / N
        assert np.isclose(cm.accuracy, (3 + 5) / 10.0)

        # Eq 5.30: Precision = TP / (TP + FP)
        assert np.isclose(cm.precision, 3.0 / 4.0)

        # Eq 5.31: Recall / Sensitivity = TP / (TP + FN)
        assert np.isclose(cm.recall, 3.0 / 4.0)

        # Eq 5.32: Specificity = TN / (TN + FP)
        assert np.isclose(cm.specificity, 5.0 / 6.0)

        # Eq 5.33: False positive rate = FP / (TN + FP) = 1 - Specificity
        assert np.isclose(cm.false_positive_rate, 1.0 / 6.0)
        assert np.isclose(cm.false_positive_rate, 1.0 - cm.specificity)

        # Eq 5.38 - 5.39: F-score
        prec = 3.0 / 4.0
        rec = 3.0 / 4.0
        expected_f = (2 * prec * rec) / (prec + rec)
        assert np.isclose(cm.f_score, expected_f)
        assert np.isclose(cm.f_score, (2 * 3.0) / (2 * 3 + 1 + 1))


class TestROCCurveAndAUC:
    """Tests for Section 5.2.6: ROC Curve and AUC."""

    def test_perfect_classifier_roc_auc(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        fpr, tpr, thresholds, auc = compute_roc_curve(y_true, scores)

        assert np.isclose(auc, 1.0)
        assert np.all(fpr >= 0.0) and np.all(fpr <= 1.0)
        assert np.all(tpr >= 0.0) and np.all(tpr <= 1.0)

    def test_inverted_classifier_roc_auc(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        fpr, tpr, thresholds, auc = compute_roc_curve(y_true, scores)
        assert np.isclose(auc, 0.0)

    def test_roc_endpoints(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        scores = np.array([0.2, 0.4, 0.5, 0.6, 0.3, 0.8])
        fpr, tpr, _, _ = compute_roc_curve(y_true, scores)
        assert np.isclose(fpr[0], 0.0) and np.isclose(tpr[0], 0.0)
        assert np.isclose(fpr[-1], 1.0) and np.isclose(tpr[-1], 1.0)


class TestFigureGeneration:
    """Tests ensuring that all Section 5.2 figures generate and save properly."""

    def test_generate_all_figures(self, tmp_path):
        out_dir = str(tmp_path / "result")
        figs = generate_all_section_5_2_figures(result_dirs=[out_dir])
        assert len(figs) == 7
        for key in ['fig_5_5', 'fig_5_6', 'fig_5_7', 'fig_5_8', 'fig_5_9', 'fig_5_10', 'fig_5_11']:
            assert key in figs
            assert os.path.exists(os.path.join(out_dir, f"{key}_" + {
                'fig_5_5': 'joint_probabilities.png',
                'fig_5_6': 'loss_matrix.png',
                'fig_5_7': 'reject_option.png',
                'fig_5_8': 'class_densities_posteriors.png',
                'fig_5_9': 'confusion_matrix.png',
                'fig_5_10': 'roc_regions.png',
                'fig_5_11': 'roc_curve.png'
            }[key]))
        import matplotlib.pyplot as plt
        for fig in figs.values():
            plt.close(fig)

