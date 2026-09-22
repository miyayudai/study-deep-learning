"""
Unit tests for Section 5.3: Generative Classifiers.
Bishop & Bishop (2024), Chapter 5, Section 5.3.
"""
import os
import pytest
import numpy as np
import scipy.stats as stats

from common.generative_classifiers import (
    sigmoid,
    logit,
    probit,
    scaled_probit,
    GaussianDiscriminantAnalysis,
    BernoulliNaiveBayes,
    ExponentialFamilyClassifier,
    plot_figure_5_12_sigmoid_and_probit,
    plot_figure_5_13_two_class_gaussian_posteriors,
    plot_figure_5_14_multiclass_gaussian_boundaries,
    generate_all_section_5_3_figures,
)


class TestActivationFunctions:
    """Tests for sigmoid, logit, probit, and scaled probit."""

    def test_sigmoid_symmetry(self):
        a = np.array([-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0])
        sig = sigmoid(a)
        sig_neg = sigmoid(-a)
        # Symmetry property (Eq 5.43): sigma(-a) = 1 - sigma(a)
        assert np.allclose(sig_neg, 1.0 - sig)
        assert np.isclose(sigmoid(0.0), 0.5)

    def test_logit_is_inverse_of_sigmoid(self):
        a = np.linspace(-6.0, 6.0, 50)
        p = sigmoid(a)
        a_recovered = logit(p)
        assert np.allclose(a, a_recovered)

    def test_scaled_probit_derivative_matches_sigmoid_at_zero(self):
        # Analytical derivatives at a = 0:
        # d/da sigma(a)|_{a=0} = sigma(0) * (1 - sigma(0)) = 0.25
        # d/da Phi(lambda a)|_{a=0} = lambda * N(0|0, 1) = sqrt(pi/8) / sqrt(2*pi) = 1/4 = 0.25
        eps = 1e-6
        d_sig = (sigmoid(eps) - sigmoid(-eps)) / (2.0 * eps)
        d_prb = (scaled_probit(eps) - scaled_probit(-eps)) / (2.0 * eps)
        assert np.isclose(d_sig, 0.25, atol=1e-5)
        assert np.isclose(d_prb, 0.25, atol=1e-5)
        assert np.isclose(d_sig, d_prb, atol=1e-5)


class TestGaussianDiscriminantAnalysis:
    """Tests for GDA (LDA and QDA) with maximum likelihood solutions."""

    def test_maximum_likelihood_solution_2class(self):
        rng = np.random.RandomState(42)
        N1, N2 = 100, 200
        true_mu1 = np.array([2.0, 1.0])
        true_mu2 = np.array([-1.0, -2.0])
        true_cov = np.array([[1.5, 0.5], [0.5, 1.0]])

        X1 = rng.multivariate_normal(true_mu1, true_cov, size=N1)
        X2 = rng.multivariate_normal(true_mu2, true_cov, size=N2)
        X = np.vstack([X1, X2])
        y = np.hstack([np.zeros(N1), np.ones(N2)])

        gda = GaussianDiscriminantAnalysis(shared_cov=True).fit(X, y)

        # Prior estimate (Eq 5.56)
        assert np.isclose(gda.priors[0], 100 / 300)
        assert np.isclose(gda.priors[1], 200 / 300)

        # Mean estimates (Eq 5.58 - 5.59)
        assert np.allclose(gda.means[0], np.mean(X1, axis=0))
        assert np.allclose(gda.means[1], np.mean(X2, axis=0))

        # Shared covariance estimate (Eq 5.61)
        S1 = np.cov(X1, rowvar=False, bias=True)
        S2 = np.cov(X2, rowvar=False, bias=True)
        expected_cov = (100 / 300) * S1 + (200 / 300) * S2
        assert np.allclose(gda.covs, expected_cov)

    def test_lda_linear_form_matches_exact_posteriors(self):
        # For 2 classes with shared covariance, p(C1|x) = sigma(w^T x + w0) (Eq 5.48)
        X = np.array([
            [1.0, 2.0], [2.0, 1.0], [1.5, 2.5],
            [-1.0, -2.0], [-2.0, -1.0], [-1.5, -2.5]
        ])
        y = np.array([0, 0, 0, 1, 1, 1])

        gda = GaussianDiscriminantAnalysis(shared_cov=True).fit(X, y)
        w, w0 = gda.get_linear_weights_2class()

        # Query points
        X_query = np.array([[0.5, 0.5], [-0.5, -0.5], [2.0, 2.0], [-2.0, -2.0]])
        probs_gda = gda.predict_proba(X_query)[:, 0] # p(C1|x)
        probs_linear = sigmoid(X_query @ w + w0)

        assert np.allclose(probs_gda, probs_linear)

    def test_qda_multiclass_and_quadratic_nature(self):
        # 3 classes with distinct covariances
        rng = np.random.RandomState(42)
        X = np.vstack([
            rng.multivariate_normal([-2, -2], [[1.0, 0.0], [0.0, 1.0]], size=30),
            rng.multivariate_normal([2, -2], [[2.0, 0.5], [0.5, 0.5]], size=30),
            rng.multivariate_normal([0, 2], [[0.5, -0.3], [-0.3, 2.0]], size=30),
        ])
        y = np.repeat([0, 1, 2], 30)

        qda = GaussianDiscriminantAnalysis(shared_cov=False).fit(X, y)
        assert qda.covs.shape == (3, 2, 2)
        probs = qda.predict_proba(X)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.all(probs >= 0.0)


class TestBernoulliNaiveBayes:
    """Tests for discrete features under naive Bayes assumption."""

    def test_bernoulli_naive_bayes_training_and_inference(self):
        # Binary features D=3, K=2
        X = np.array([
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 1],
            [0, 1, 1],
            [0, 0, 0]
        ])
        y = np.array([0, 0, 0, 1, 1, 1])

        nb = BernoulliNaiveBayes(alpha=1.0).fit(X, y)
        assert nb.priors[0] == 0.5 and nb.priors[1] == 0.5
        assert nb.mu.shape == (2, 3)

        # Predict
        preds = nb.predict(X)
        probs = nb.predict_proba(X)
        assert np.allclose(probs.sum(axis=1), 1.0)
        # Class 0 has mostly feature 0 = 1, Class 1 has mostly feature 0 = 0
        assert preds[0] == 0
        assert preds[3] == 1


class TestExponentialFamilyClassifier:
    """Tests for exponential family class-conditionals (Section 5.3.4, Eq 5.66 - 5.68)."""

    def test_exponential_family_2class_linearity(self):
        # 2 classes with natural parameters lambda_1, lambda_2, scale s
        lambda1 = np.array([2.0, 1.0])
        lambda2 = np.array([-1.0, -2.0])
        log_g = np.array([-2.5, -2.5])
        priors = np.array([0.6, 0.4])
        scale = 1.5

        efc = ExponentialFamilyClassifier(
            natural_params=np.vstack([lambda1, lambda2]),
            log_g=log_g,
            priors=priors,
            scale=scale
        )

        w, w0 = efc.get_2class_linear_weights()
        expected_w = (lambda1 - lambda2) / scale
        expected_w0 = (log_g[0] - log_g[1]) + np.log(priors[0] / priors[1])

        assert np.allclose(w, expected_w)
        assert np.isclose(w0, expected_w0)

        # Test predict_proba matches sigmoid(w^T x + w0)
        X = np.array([[1.0, 0.5], [-0.5, 1.2], [2.0, -1.0]])
        probs = efc.predict_proba(X)
        probs_sig = sigmoid(X @ w + w0)

        assert np.allclose(probs[:, 0], probs_sig)
        assert np.allclose(probs[:, 1], 1.0 - probs_sig)


class TestFiguresGeneration:
    """Test generation of Figures 5.12 to 5.14."""

    def test_all_section_5_3_figures(self, tmp_path):
        figs = generate_all_section_5_3_figures(result_dirs=[str(tmp_path)])
        assert 'fig_5_12' in figs
        assert 'fig_5_13' in figs
        assert 'fig_5_14' in figs
        for k in ['fig_5_12', 'fig_5_13', 'fig_5_14']:
            assert hasattr(figs[k], 'savefig')
