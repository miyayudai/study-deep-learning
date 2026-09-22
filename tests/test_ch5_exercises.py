"""Unit tests for Chapter 5 Exercises: Single-layer Networks: Classification.

Exercises 5.1 to 5.24.
Bishop & Bishop (2024), Chapter 5, pp. 166-169.
"""

import numpy as np
import pytest
from scipy import integrate, special
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.stats import norm

from common.discriminative_classifiers import (
    GaussianBasisFunctions,
    LogisticRegression,
    ProbitRegression,
    SoftmaxRegression,
    erf_func,
    logit,
    probit,
    probit_deriv,
    sigmoid,
    sigmoid_deriv,
    softmax,
    softmax_jacobian,
)
from common.exercises_ch5 import (
    exercise_5_1_conditional_expectation,
    exercise_5_2_check_separability,
    exercise_5_3_verify_linear_constraint,
    exercise_5_4_verify_multiple_constraints,
    exercise_5_5_compute_f_score,
    exercise_5_6_compute_bhattacharyya_bound,
    exercise_5_7_zero_one_loss_decision,
    exercise_5_8_expected_loss_decision,
    exercise_5_9_average_posterior,
    exercise_5_10_reject_threshold,
    exercise_5_11_verify_sigmoid_properties,
    exercise_5_12_compute_gaussian_gda_params,
    exercise_5_13_mle_priors,
    exercise_5_14_mle_gaussian_shared_cov,
    exercise_5_15_mle_binary_naive_bayes,
    exercise_5_16_multistate_naive_bayes_ak,
    exercise_5_17_mle_multistate_naive_bayes,
    exercise_5_18_verify_sigmoid_derivative,
    exercise_5_19_verify_logistic_gradient,
    exercise_5_20_separable_growth_check,
    exercise_5_21_verify_softmax_jacobian,
    exercise_5_22_verify_softmax_gradient,
    exercise_5_23_verify_probit_erf,
    exercise_5_24_probit_sigmoid_matching_scale,
)


class TestChapter5Exercises:
    """Mathematical and numerical verification of Exercises 5.1 - 5.24."""

    def test_exercise_5_1_conditional_expectation(self):
        """Exercise 5.1: E[t|x] = p(Ck|x) for 1-of-K coding."""
        # Simulated posterior distribution
        p_Ck = np.array([0.2, 0.5, 0.3])
        # In 1-of-K coding, t in {e_1, e_2, e_3} with probabilities p_Ck
        E_t = np.zeros(3)
        for k in range(3):
            e_k = np.zeros(3)
            e_k[k] = 1.0
            E_t += e_k * p_Ck[k]
        assert np.allclose(E_t, p_Ck)

    def test_exercise_5_2_convex_hull_separation(self):
        """Exercise 5.2: Convex hull intersection vs linear separability."""
        # Case A: Non-intersecting convex hulls -> Linearly separable
        X = np.array([[1.0, 1.0], [2.0, 1.0], [1.5, 2.0]])
        Y = np.array([[-1.0, -1.0], [-2.0, -1.0], [-1.5, -2.0]])

        # Solve for separating hyperplane w^T x + w0 > 0 and w^T y + w0 < 0
        # Formulate linear program: max margin
        N_x, N_y = len(X), len(Y)
        c = np.zeros(3)  # [w1, w2, w0]
        A = np.vstack([
            np.hstack([-X, -np.ones((N_x, 1))]),
            np.hstack([Y, np.ones((N_y, 1))])
        ])
        b = -np.ones(N_x + N_y)
        res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
        assert res.success, "Non-intersecting hulls should be linearly separable"

        # Case B: Intersecting convex hulls (e.g. concentric points)
        X_int = np.array([[0.0, 0.0]])
        Y_int = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        A_int = np.vstack([
            np.hstack([-X_int, -np.ones((1, 1))]),
            np.hstack([Y_int, np.ones((4, 1))])
        ])
        b_int = -np.ones(5)
        res_int = linprog(c, A_ub=A_int, b_ub=b_int, bounds=(None, None))
        assert not res_int.success, "Intersecting hulls cannot be linearly separable"

    def test_exercise_5_3_linear_constraint_preservation(self):
        """Exercise 5.3: Sum-of-squares predictions preserve linear target constraints."""
        np.random.seed(42)
        N, D, K = 30, 2, 4
        X = np.random.randn(N, D)
        # Create targets satisfying a^T t_n + b = 0
        a = np.array([1.5, -2.0, 0.5, 1.0])
        b = -3.0
        # Generate random t_1..t_3 and solve for t_4
        T = np.random.randn(N, K)
        T[:, 3] = (-b - np.dot(T[:, :3], a[:3])) / a[3]

        # Verify training targets satisfy constraint
        assert np.allclose(T @ a + b, 0.0)

        # Least squares model with bias phi_0 = 1
        Phi = np.hstack([np.ones((N, 1)), X])
        # W = (Phi^T Phi)^-1 Phi^T T
        W = np.linalg.pinv(Phi) @ T

        # For any new test point x_test
        X_test = np.random.randn(10, D)
        Phi_test = np.hstack([np.ones((10, 1)), X_test])
        Y_pred = Phi_test @ W

        # Verify predictions satisfy a^T y(x) + b = 0
        assert np.allclose(Y_pred @ a + b, 0.0, atol=1e-10)

    def test_exercise_5_4_multiple_constraints(self):
        """Exercise 5.4: Multiple simultaneous linear constraints."""
        np.random.seed(42)
        N, D, K = 40, 3, 5
        X = np.random.randn(N, D)
        # 2 constraints: A^T t + b = 0
        A = np.array([[1.0, 0.0],
                      [0.0, 1.0],
                      [1.0, -1.0],
                      [2.0, 1.0],
                      [-1.0, 2.0]])
        b_vec = np.array([2.0, -1.5])

        # Generate targets satisfying both constraints
        T = np.random.randn(N, K)
        # Project onto nullspace to satisfy constraints
        # T_n A = -b_vec
        for n in range(N):
            T[n, :] = np.linalg.lstsq(A.T, -b_vec, rcond=None)[0] + (
                np.eye(K) - np.linalg.pinv(A.T) @ A.T
            ) @ T[n, :]

        assert np.allclose(T @ A + b_vec, 0.0)

        Phi = np.hstack([np.ones((N, 1)), X])
        W = np.linalg.pinv(Phi) @ T

        X_test = np.random.randn(15, D)
        Phi_test = np.hstack([np.ones((15, 1)), X_test])
        Y_pred = Phi_test @ W
        assert np.allclose(Y_pred @ A + b_vec, 0.0, atol=1e-10)

    def test_exercise_5_5_f_score_formula(self):
        """Exercise 5.5: Derivation of F_beta score formula."""
        TP, FP, FN = 80, 20, 10
        P = TP / (TP + FP)
        R = TP / (TP + FN)

        for beta in [0.5, 1.0, 2.0]:
            f_from_pr = (1.0 + beta ** 2) * P * R / (beta ** 2 * P + R)
            f_formula = (1.0 + beta ** 2) * TP / ((1.0 + beta ** 2) * TP + beta ** 2 * FN + FP)
            assert np.isclose(f_from_pr, f_formula)

    def test_exercise_5_6_bhattacharyya_bound(self):
        """Exercise 5.6: Misclassification error bound p(mistake) <= int sqrt(p(x,C1)p(x,C2)) dx."""
        # Non-negative numbers inequality: a <= b => a <= sqrt(ab)
        a, b = 2.0, 8.0
        assert a <= np.sqrt(a * b)

        # 1D Gaussians
        mu1, s1 = -1.0, 1.0
        mu2, s2 = 1.0, 1.0
        prior1, prior2 = 0.5, 0.5

        def p_joint1(x):
            return prior1 * norm.pdf(x, mu1, s1)

        def p_joint2(x):
            return prior2 * norm.pdf(x, mu2, s2)

        # True misclassification error: int min(p_joint1, p_joint2) dx
        def min_joint(x):
            return np.minimum(p_joint1(x), p_joint2(x))

        # Upper bound: int sqrt(p_joint1 * p_joint2) dx
        def bhattacharyya_integrand(x):
            return np.sqrt(p_joint1(x) * p_joint2(x))

        err_true, _ = integrate.quad(min_joint, -10, 10)
        err_bound, _ = integrate.quad(bhattacharyya_integrand, -10, 10)

        assert err_true <= err_bound

    def test_exercise_5_7_zero_one_loss(self):
        """Exercise 5.7: Loss matrix L_kj = 1 - I_kj minimizes risk by choosing max posterior."""
        p_post = np.array([0.1, 0.6, 0.3])
        K = len(p_post)
        L = 1.0 - np.eye(K)

        expected_losses = [np.sum(L[:, j] * p_post) for j in range(K)]
        # Expected loss for choosing j is 1 - p(C_j | x)
        assert np.allclose(expected_losses, 1.0 - p_post)
        best_class = np.argmin(expected_losses)
        assert best_class == np.argmax(p_post)

    def test_exercise_5_8_expected_loss_decision(self):
        """Exercise 5.8: Criterion for minimizing expected loss with general loss and priors."""
        L = np.array([[0.0, 10.0],
                      [1.0, 0.0]])  # asymmetric penalty
        p_post = np.array([0.85, 0.15])
        # Expected loss for action j:
        r0 = L[0, 0] * p_post[0] + L[1, 0] * p_post[1]  # = 0 + 1 * 0.15 = 0.15
        r1 = L[0, 1] * p_post[0] + L[1, 1] * p_post[1]  # = 10 * 0.85 + 0 = 8.5
        assert r0 < r1

    def test_exercise_5_9_average_posterior_convergence(self):
        """Exercise 5.9: Average of posterior probabilities approaches prior as N -> inf."""
        np.random.seed(42)
        N = 20000
        prior1 = 0.7
        # Draw labels
        labels = (np.random.rand(N) < prior1).astype(int)
        # Draw inputs
        X = np.where(labels == 1, np.random.randn(N) + 1.0, np.random.randn(N) - 1.0)

        # True posteriors
        p1 = prior1 * norm.pdf(X, 1.0, 1.0)
        p0 = (1.0 - prior1) * norm.pdf(X, -1.0, 1.0)
        post1 = p1 / (p1 + p0)

        mean_post1 = np.mean(post1)
        assert np.isclose(mean_post1, prior1, atol=0.01)

    def test_exercise_5_10_reject_option_loss(self):
        """Exercise 5.10: Reject threshold relationship theta = 1 - lambda."""
        lam = 0.25  # loss for rejecting
        # For 0-1 loss, reject when max_k p(C_k|x) <= theta = 1 - lambda = 0.75
        theta = 1.0 - lam
        assert np.isclose(theta, 0.75)

    def test_exercise_5_11_sigmoid_inverses(self):
        """Exercise 5.11: Sigmoid symmetry sigma(-a) = 1 - sigma(a) and logit inverse."""
        a_vals = np.linspace(-5, 5, 20)
        assert np.allclose(sigmoid(-a_vals), 1.0 - sigmoid(a_vals))
        assert np.allclose(logit(sigmoid(a_vals)), a_vals)

    def test_exercise_5_12_gaussian_gda_parameters(self):
        """Exercise 5.12: Verification of w and w0 in two-class Gaussian generative model."""
        mu1 = np.array([1.0, 2.0])
        mu2 = np.array([-1.0, -1.0])
        Sigma = np.array([[2.0, 0.5], [0.5, 1.5]])
        Sigma_inv = np.linalg.inv(Sigma)
        pi1 = 0.6
        pi2 = 0.4

        w = Sigma_inv @ (mu1 - mu2)
        w0 = -0.5 * mu1.T @ Sigma_inv @ mu1 + 0.5 * mu2.T @ Sigma_inv @ mu2 + np.log(pi1 / pi2)

        x_test = np.array([0.5, 0.5])
        a_analytical = np.dot(w, x_test) + w0

        # Bayes ratio
        p1 = pi1 * (1.0 / (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))) * np.exp(-0.5 * (x_test - mu1).T @ Sigma_inv @ (x_test - mu1))
        p2 = pi2 * (1.0 / (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))) * np.exp(-0.5 * (x_test - mu2).T @ Sigma_inv @ (x_test - mu2))
        a_bayes = np.log(p1 / p2)

        assert np.isclose(a_analytical, a_bayes)

    def test_exercise_5_13_multiclass_prior_mle(self):
        """Exercise 5.13: MLE for class priors pi_k = N_k / N."""
        targets = np.array([0, 0, 1, 1, 1, 2])
        N = len(targets)
        for k in range(3):
            assert np.isclose(np.sum(targets == k) / N, [2/6, 3/6, 1/6][k])

    def test_exercise_5_14_multiclass_gda_mle(self):
        """Exercise 5.14: MLE for means and shared covariance matrix."""
        np.random.seed(42)
        Phi = np.random.randn(60, 2)
        t = np.array([0]*20 + [1]*20 + [2]*20)
        K = 3
        N = len(t)

        mu_list = [np.mean(Phi[t == k], axis=0) for k in range(K)]
        # Shared covariance
        Sigma_shared = np.zeros((2, 2))
        for k in range(K):
            diff = Phi[t == k] - mu_list[k]
            Sigma_shared += diff.T @ diff / N

        assert Sigma_shared.shape == (2, 2)
        assert np.all(np.linalg.eigvalsh(Sigma_shared) > 0)

    def test_exercise_5_15_binary_naive_bayes_mle(self):
        """Exercise 5.15: MLE for Bernoulli Naive Bayes mu_ki = (1/N_k) sum t_nk x_ni."""
        X = np.array([[1, 0], [1, 1], [0, 0]])
        t = np.array([1, 1, 0])
        # For class 1 (N_1 = 2)
        mu_1 = np.mean(X[t == 1], axis=0)
        assert np.allclose(mu_1, [1.0, 0.5])

    def test_exercise_5_16_multistate_naive_bayes_linearity(self):
        """Exercise 5.16: Naive Bayes pre-activations a_k are linear functions of 1-of-L features."""
        # 1-of-L encoded features: phi in {0, 1}^(M*L)
        M, L, K = 3, 4, 2
        mu_kml = np.random.dirichlet(np.ones(L), size=(K, M))  # (K, M, L)
        # a_k = sum_m sum_l phi_ml ln mu_kml + ln pi_k
        phi = np.zeros((M, L))
        phi[0, 1] = 1.0
        phi[1, 3] = 1.0
        phi[2, 0] = 1.0

        ak_0 = np.sum(phi * np.log(mu_kml[0]))
        ak_linear = np.dot(phi.ravel(), np.log(mu_kml[0]).ravel())
        assert np.isclose(ak_0, ak_linear)

    def test_exercise_5_17_multistate_naive_bayes_mle(self):
        """Exercise 5.17: MLE for multi-state Naive Bayes mu_kml = N_kml / N_k."""
        # Feature states for class k
        states = np.array([0, 1, 1, 2, 1])  # 5 samples in class k
        L = 3
        counts = [np.sum(states == l) for l in range(L)]
        probs = [c / len(states) for c in counts]
        assert np.allclose(probs, [0.2, 0.6, 0.2])
        assert np.isclose(np.sum(probs), 1.0)

    def test_exercise_5_18_sigmoid_derivative(self):
        """Exercise 5.18: d sigma / da = sigma * (1 - sigma)."""
        a = np.array([-2.5, 0.0, 3.2])
        assert np.allclose(sigmoid_deriv(a), sigmoid(a) * (1.0 - sigmoid(a)))

    def test_exercise_5_19_logistic_gradient(self):
        """Exercise 5.19: Gradient of cross-entropy error is sum (y_n - t_n) phi_n."""
        np.random.seed(42)
        X = np.random.randn(10, 2)
        t = np.random.randint(0, 2, size=10)
        clf = LogisticRegression(fit_intercept=True)
        Phi = clf._prepare_features(X)
        w = np.random.randn(3)
        grad = clf.compute_gradient(Phi, t, w)
        y = sigmoid(Phi @ w)
        expected_grad = Phi.T @ (y - t)
        assert np.allclose(grad, expected_grad)

    def test_exercise_5_20_separable_divergence(self):
        """Exercise 5.20: Weight magnitude grows to infinity for linearly separable data without regularization."""
        X = np.array([[-2.0, -2.0], [2.0, 2.0]])
        t = np.array([0, 1])
        clf1 = LogisticRegression(reg=0.0, fit_intercept=False).fit(X, t, method="gd", lr=0.5, max_iter=50)
        clf2 = LogisticRegression(reg=0.0, fit_intercept=False).fit(X, t, method="gd", lr=0.5, max_iter=200)
        # Weights should continue growing without bound
        assert np.linalg.norm(clf2.w) > np.linalg.norm(clf1.w)
        assert np.linalg.norm(clf2.w) > 2.0

    def test_exercise_5_21_softmax_jacobian(self):
        """Exercise 5.21: dy_k / da_j = y_k * (I_kj - y_j)."""
        a = np.array([1.0, 2.0, 0.5])
        y = softmax(a)
        J = softmax_jacobian(y)
        K = len(a)
        for k in range(K):
            for j in range(K):
                assert np.isclose(J[k, j], y[k] * ((1.0 if k == j else 0.0) - y[j]))

    def test_exercise_5_22_softmax_gradient(self):
        """Exercise 5.22: Gradient of multiclass cross-entropy is sum (y_nj - t_nj) phi_n."""
        np.random.seed(42)
        X = np.random.randn(15, 2)
        t = np.random.randint(0, 3, size=15)
        clf = SoftmaxRegression(fit_intercept=True)
        Phi = clf._prepare_features(X)
        T = clf._to_one_hot(t, 3)
        W = np.random.randn(3, 3)
        grad = clf.compute_gradient(Phi, T, W)
        Y = softmax(Phi @ W)
        expected_grad = Phi.T @ (Y - T)
        assert np.allclose(grad, expected_grad)

    def test_exercise_5_23_probit_erf_relation(self):
        """Exercise 5.23: Phi(a) = 0.5 * (1 + erf(a / sqrt(2)))."""
        a_vals = np.linspace(-3, 3, 25)
        assert np.allclose(probit(a_vals), 0.5 * (1.0 + erf_func(a_vals / np.sqrt(2.0))))

    def test_exercise_5_24_probit_sigmoid_scaling(self):
        """Exercise 5.24: Equal derivatives of sigma(a) and Phi(lambda*a) at a=0 implies lambda^2 = pi/8."""
        # sigma'(0) = 1/4
        deriv_sig_0 = sigmoid_deriv(0.0)
        assert np.isclose(deriv_sig_0, 0.25)

        # Phi'(lambda * a)|_{a=0} * lambda = lambda * N(0|0, 1) = lambda / sqrt(2*pi)
        # Equal at 0: lambda / sqrt(2*pi) = 1/4 => lambda = sqrt(2*pi) / 4 => lambda^2 = 2*pi/16 = pi/8
        lambda_val = np.sqrt(np.pi / 8.0)
        deriv_probit_scaled_0 = lambda_val * norm.pdf(0.0)
        assert np.isclose(deriv_sig_0, deriv_probit_scaled_0)
        assert np.isclose(lambda_val ** 2, np.pi / 8.0)


class TestCommonExercisesCh5Module:
    """Explicit tests for all helper functions in common.exercises_ch5."""

    def test_helpers_ex1_to_ex6(self):
        # Ex 5.1
        e_t = exercise_5_1_conditional_expectation([0.1, 0.9])
        assert np.allclose(e_t, [0.1, 0.9])

        # Ex 5.2
        assert exercise_5_2_check_separability([[1, 1]], [[-1, -1]])
        assert not exercise_5_2_check_separability([[0, 0]], [[1, 0], [-1, 0]])

        # Ex 5.3
        X = np.array([[1.0], [2.0]])
        T = np.array([[1.0, 2.0], [2.0, 1.0]])
        a = np.array([1.0, 1.0])
        b = -3.0
        X_test = np.array([[1.5], [3.0]])
        train_viol, test_viol = exercise_5_3_verify_linear_constraint(X, T, a, b, X_test)
        assert np.isclose(train_viol, 0.0)
        assert np.isclose(test_viol, 0.0)

        # Ex 5.4
        A = np.array([[1.0], [1.0]])
        b_vec = np.array([-3.0])
        tr_v, te_v = exercise_5_4_verify_multiple_constraints(X, T, A, b_vec, X_test)
        assert np.isclose(tr_v, 0.0)
        assert np.isclose(te_v, 0.0)

        # Ex 5.5
        fpr, fform = exercise_5_5_compute_f_score(100, 10, 20, 1.0)
        assert np.isclose(fpr, fform)

        # Ex 5.6
        p1 = lambda x: 0.5 * norm.pdf(x, -1, 1)
        p2 = lambda x: 0.5 * norm.pdf(x, 1, 1)
        err_true, err_bound = exercise_5_6_compute_bhattacharyya_bound(p1, p2)
        assert err_true <= err_bound

    def test_helpers_ex7_to_ex17(self):
        # Ex 5.7
        cls_opt, losses = exercise_5_7_zero_one_loss_decision([0.2, 0.8])
        assert cls_opt == 1
        assert np.allclose(losses, [0.8, 0.2])

        # Ex 5.8
        L = np.array([[0.0, 5.0], [1.0, 0.0]])
        cls_gen, gen_losses = exercise_5_8_expected_loss_decision(L, [0.9, 0.1])
        assert cls_gen == 0

        # Ex 5.9
        avg_post = exercise_5_9_average_posterior(np.array([0.5, 0.7, 0.6]))
        assert np.isclose(avg_post, 0.6)

        # Ex 5.10
        assert np.isclose(exercise_5_10_reject_threshold(0.2), 0.8)

        # Ex 5.11
        sym, inv = exercise_5_11_verify_sigmoid_properties(np.array([-1.0, 0.0, 1.0]))
        assert sym and inv

        # Ex 5.12
        w, w0 = exercise_5_12_compute_gaussian_gda_params(np.array([1.0]), np.array([-1.0]), np.array([[1.0]]), 0.5, 0.5)
        assert np.isclose(w[0], 2.0)
        assert np.isclose(w0, 0.0)

        # Ex 5.13
        priors = exercise_5_13_mle_priors(np.array([0, 0, 1]), 2)
        assert np.allclose(priors, [2/3, 1/3])

        # Ex 5.14
        means, cov = exercise_5_14_mle_gaussian_shared_cov(np.array([[1.0], [2.0], [-1.0], [-2.0]]), np.array([0, 0, 1, 1]), 2)
        assert np.isclose(means[0][0], 1.5)
        assert np.isclose(means[1][0], -1.5)
        assert cov.shape == (1, 1)

        # Ex 5.15
        nb_means = exercise_5_15_mle_binary_naive_bayes(np.array([[1, 0], [1, 1]]), np.array([0, 0]), 1)
        assert np.allclose(nb_means[0], [1.0, 0.5])

        # Ex 5.16
        ak = exercise_5_16_multistate_naive_bayes_ak(np.array([[1, 0]]), np.array([[0.5, 0.5]]), 0.5)
        assert np.isclose(ak, np.log(0.5) + np.log(0.5))

        # Ex 5.17
        mle_nb = exercise_5_17_mle_multistate_naive_bayes(np.array([[[1, 0]], [[0, 1]]]))
        assert np.allclose(mle_nb, [[0.5, 0.5]])

    def test_helpers_ex18_to_ex24(self):
        # Ex 5.18
        assert exercise_5_18_verify_sigmoid_derivative(np.array([-2.0, 0.0, 2.0]))

        # Ex 5.19
        Phi = np.array([[1.0, 0.5], [1.0, -0.5]])
        t = np.array([1, 0])
        w = np.array([0.1, 0.2])
        ga, gn = exercise_5_19_verify_logistic_gradient(Phi, t, w)
        assert np.allclose(ga, gn, atol=1e-5)

        # Ex 5.20
        X = np.array([[-1.0], [1.0]])
        norms = exercise_5_20_separable_growth_check(X, np.array([0, 1]), [10, 50])
        assert norms[1] > norms[0]

        # Ex 5.21
        Ja, Jn = exercise_5_21_verify_softmax_jacobian(np.array([1.0, -1.0]))
        assert np.allclose(Ja, Jn, atol=1e-5)

        # Ex 5.22
        T = np.array([[1, 0], [0, 1]])
        W = np.array([[0.1, -0.1], [0.2, -0.2]])
        ga, gn = exercise_5_22_verify_softmax_gradient(Phi, T, W)
        assert np.allclose(ga, gn, atol=1e-4)

        # Ex 5.23
        assert exercise_5_23_verify_probit_erf(np.array([-1.0, 0.0, 1.0]))

        # Ex 5.24
        ds0, dp0, l2 = exercise_5_24_probit_sigmoid_matching_scale()
        assert np.isclose(ds0, dp0)
        assert np.isclose(l2, np.pi / 8.0)
