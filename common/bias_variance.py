"""
Bias-Variance Trade-off (Chapter 4: Single-layer Networks: Regression, Section 4.3).
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.

Covers:
- Expected squared loss decomposition into noise and model error (Eq 4.41 - 4.42)
- Frequentist thought experiment: Ensembles of datasets D^(1), ..., D^(L)
- Bias-Variance decomposition derivation:
    E_D[{f(x; D) - h(x)}^2] = (E_D[f(x; D)] - h(x))^2 + E_D[{f(x; D) - E_D[f(x; D)]}^2] (Eq 4.43 - 4.45)
- Integrated bias^2, variance, and expected loss (Eq 4.46 - 4.49, Eq 4.51 - 4.52)
- Simulation using Gaussian basis functions (M = 24 centers, s = 0.1) and Ridge regression
- Faithful reproduction of Figures 4.7 and 4.8.
"""
import os
from typing import Optional, List, Tuple, Dict, Any, Union
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from common.plot_utils import setup_style


class SinusoidalDataGenerator:
    """
    Generates synthetic regression datasets from h(x) = sin(2 * pi * x) + epsilon,
    where epsilon ~ N(0, noise_std^2) and x ~ U(0, 1).
    Bishop & Bishop (2024), Section 4.3, p. 126.
    """
    def __init__(self, noise_std: float = 0.3, random_state: Optional[int] = None):
        self.noise_std = float(noise_std)
        self.rng = np.random.RandomState(random_state)

    @staticmethod
    def true_function(x: np.ndarray) -> np.ndarray:
        """Target regression function h(x) = sin(2 * pi * x)."""
        return np.sin(2.0 * np.pi * x)

    def sample_dataset(self, n_samples: int = 25) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a single dataset of size n_samples."""
        x = self.rng.uniform(0.0, 1.0, size=n_samples)
        noise = self.rng.normal(0.0, self.noise_std, size=n_samples)
        t = self.true_function(x) + noise
        return x, t

    def sample_ensemble(self, n_datasets: int = 100, n_samples: int = 25) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Sample an ensemble of L independent datasets."""
        ensemble = []
        for _ in range(n_datasets):
            ensemble.append(self.sample_dataset(n_samples))
        return ensemble


class GaussianFeatureExtractor:
    """
    Gaussian radial basis function design matrix with M centers in [0, 1] and width s.
    Plus a constant bias feature phi_0(x) = 1 (total M + 1 features).
    """
    def __init__(self, m_centers: int = 24, s: float = 0.1):
        self.m_centers = m_centers
        self.s = float(s)
        self.centers = np.linspace(0.0, 1.0, m_centers)
        self.n_features = m_centers + 1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        diff = (x_arr[:, None] - self.centers[None, :]) / self.s
        rbf = np.exp(-0.5 * diff ** 2)
        bias = np.ones((len(x_arr), 1), dtype=np.float64)
        return np.hstack([bias, rbf])

    transform = __call__


class BiasVarianceSimulator:
    """
    Performs ensemble fitting across regularization coefficients lambda
    and computes empirical bias, variance, and test error.
    """
    def __init__(
        self,
        L: int = 100,
        N: int = 25,
        M: int = 24,
        s: float = 0.1,
        sigma: float = 0.3,
        random_state: int = 42,
        **kwargs
    ):
        if "n_datasets" in kwargs:
            L = kwargs["n_datasets"]
        if "n_samples" in kwargs:
            N = kwargs["n_samples"]
        if "n_basis" in kwargs:
            M = kwargs["n_basis"]
        if "spatial_scale" in kwargs:
            s = kwargs["spatial_scale"]
        if "noise_std" in kwargs:
            sigma = kwargs["noise_std"]

        self.L = L
        self.N = N
        self.M = M
        self.s = s
        self.sigma = sigma
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.true_func = lambda x: np.sin(2.0 * np.pi * x)

        # M Gaussian basis centers uniformly distributed in [0, 1]
        self.mu = np.linspace(0.0, 1.0, M)
        self.centers = self.mu

        # Generate L training datasets
        self.x_train = self.rng.uniform(0.0, 1.0, size=(L, N))
        self.t_train = np.sin(2.0 * np.pi * self.x_train) + self.rng.normal(0.0, sigma, size=(L, N))
        self.datasets = [(self.x_train[l], self.t_train[l]) for l in range(L)]

        # Precompute design matrices for all L datasets
        self.Phi_train = [self._design_matrix(self.x_train[l]) for l in range(L)]

        # Compatibility attributes for existing plotting functions
        self.n_datasets = L
        self.n_samples = N
        self.noise_std = sigma
        self.ensemble = self.datasets
        self.data_gen = SinusoidalDataGenerator(noise_std=sigma, random_state=random_state)
        self.feature_extractor = GaussianFeatureExtractor(m_centers=M, s=s)

        # Evaluation grid for computing bias and variance
        self.x_eval = np.linspace(0.0, 1.0, 1000)
        self.Phi_eval = self.feature_extractor(self.x_eval)
        self.h_eval = self.data_gen.true_function(self.x_eval)

        # Test set for evaluating generalization error
        rng_test = np.random.RandomState(random_state + 1000)
        self.x_test = rng_test.uniform(0.0, 1.0, size=1000)
        self.Phi_test = self.feature_extractor(self.x_test)
        self.t_test = self.data_gen.true_function(self.x_test) + rng_test.normal(0.0, sigma, size=1000)

    def evaluate_basis(self, x: np.ndarray) -> np.ndarray:
        """Alias for _design_matrix."""
        return self._design_matrix(x)

    def _design_matrix(self, x: np.ndarray) -> np.ndarray:
        """Construct design matrix Phi with bias column and M Gaussian basis functions."""
        x_flat = np.asarray(x, dtype=np.float64).ravel()
        cols = [np.ones_like(x_flat)]
        for m in self.mu:
            cols.append(np.exp(-0.5 * ((x_flat - m) / self.s) ** 2))
        return np.column_stack(cols)

    def fit_and_predict(self, lam: float, x: np.ndarray) -> np.ndarray:
        """Fit Ridge regression on each dataset and predict on evaluation points x."""
        Phi_eval = self._design_matrix(x)
        num_features = self.M + 1
        reg_matrix = lam * np.eye(num_features)
        preds = []
        for l in range(self.L):
            Phi_l = self.Phi_train[l]
            t_l = self.datasets[l][1]
            w = la.solve(reg_matrix + Phi_l.T @ Phi_l, Phi_l.T @ t_l, assume_a='pos')
            preds.append(Phi_eval @ w)
        return np.array(preds)

    def compute_tradeoff_for_lambda(
        self,
        lam: float,
        x_eval: np.ndarray,
        x_test: Optional[np.ndarray] = None,
        t_test: Optional[np.ndarray] = None
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Compute bias^2, variance, sum, and test error for given lambda."""
        preds = self.fit_and_predict(lam, x_eval)
        f_bar = np.mean(preds, axis=0)
        h_eval = self.true_func(x_eval)

        b2 = float(np.mean((f_bar - h_eval) ** 2))
        v = float(np.mean(np.var(preds, axis=0)))

        res = {
            "lambda": float(lam),
            "ln_lambda": float(np.log(lam)),
            "squared_bias": b2,
            "variance": v,
            "bias_variance_sum": b2 + v,
            "avg_prediction": f_bar,
            "individual_predictions": preds
        }
        if x_test is not None and t_test is not None:
            preds_test = self.fit_and_predict(lam, x_test)
            res["test_error"] = float(np.mean((preds_test - t_test[np.newaxis, :]) ** 2))
        return res

    def fit_ensemble_for_lambda(self, lam: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit Ridge regression on each dataset D^(l) with regularization parameter lambda.
        Returns:
            preds_eval: shape (L, len(x_eval))
            preds_test: shape (L, len(x_test))
        """
        preds_eval = []
        preds_test = []
        M = self.feature_extractor.n_features
        reg_matrix = lam * np.eye(M)

        for x_l, t_l in self.ensemble:
            Phi_l = self.feature_extractor(x_l)
            # Normal equations for regularized least squares: (lam * I + Phi^T Phi) w = Phi^T t
            w = la.solve(reg_matrix + Phi_l.T @ Phi_l, Phi_l.T @ t_l, assume_a='pos')
            preds_eval.append(self.Phi_eval @ w)
            preds_test.append(self.Phi_test @ w)

        return np.array(preds_eval), np.array(preds_test)

    def compute_metrics(self, lam: float) -> Dict[str, float]:
        """
        Compute integrated squared bias, variance, sum, and test error for a given lambda.
        """
        preds_eval, preds_test = self.fit_ensemble_for_lambda(lam)

        # Average prediction over ensemble: f_bar(x) = (1/L) sum_l f^(l)(x) (Eq 4.50)
        f_bar = np.mean(preds_eval, axis=0)

        # Integrated squared bias: (1/N) sum_n { f_bar(x_n) - h(x_n) }^2 (Eq 4.51)
        bias2 = float(np.mean((f_bar - self.h_eval) ** 2))

        # Integrated variance: (1/N) sum_n (1/L) sum_l { f^(l)(x_n) - f_bar(x_n) }^2 (Eq 4.52)
        variance = float(np.mean(np.var(preds_eval, axis=0, ddof=0)))

        # Average test error over ensemble
        test_err = float(np.mean((preds_test - self.t_test[None, :]) ** 2))

        return {
            'lambda': lam,
            'ln_lambda': float(np.log(lam)),
            'bias2': bias2,
            'variance': variance,
            'bias2_plus_variance': bias2 + variance,
            'test_error': test_err
        }

    def run_sweep(self, ln_lambdas: np.ndarray) -> Dict[str, np.ndarray]:
        """Sweep over an array of ln(lambda) values."""
        results = {
            'ln_lambda': [],
            'bias2': [],
            'variance': [],
            'bias2_plus_variance': [],
            'test_error': []
        }
        for ln_lam in ln_lambdas:
            res = self.compute_metrics(np.exp(ln_lam))
            results['ln_lambda'].append(res['ln_lambda'])
            results['bias2'].append(res['bias2'])
            results['variance'].append(res['variance'])
            results['bias2_plus_variance'].append(res['bias2_plus_variance'])
            results['test_error'].append(res['test_error'])

        return {k: np.array(v) for k, v in results.items()}


# =====================================================================
# Figure Plotters (Figure 4.7 and Figure 4.8)
# =====================================================================

def plot_figure_4_7_bias_variance_ensembles(
    simulator: Optional[BiasVarianceSimulator] = None,
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Faithfully reproduce Figure 4.7 from Bishop & Bishop (2024), page 127:
    Illustration of the bias-variance trade-off for sinusoidal data.
    3 rows for ln lambda = 3, 1, -3:
      Left column: 20 individual fit curves f^(l)(x) in red.
      Right column: True function h(x) in green and ensemble average f_bar(x) in red.
    """
    if simulator is None:
        simulator = BiasVarianceSimulator(n_datasets=100, n_samples=25, random_state=42)

    setup_style()
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 9.5), dpi=300)

    ln_lambdas = [3.0, 1.0, -3.0]
    n_display_curves = 20
    x_plot = np.linspace(0.0, 1.0, 200)
    Phi_plot = simulator.feature_extractor(x_plot)
    h_plot = simulator.data_gen.true_function(x_plot)

    for row_idx, ln_lam in enumerate(ln_lambdas):
        lam = np.exp(ln_lam)
        preds_eval = []
        M = simulator.feature_extractor.n_features
        reg_matrix = lam * np.eye(M)

        for l in range(simulator.n_datasets):
            x_l, t_l = simulator.ensemble[l]
            Phi_l = simulator.feature_extractor(x_l)
            w = la.solve(reg_matrix + Phi_l.T @ Phi_l, Phi_l.T @ t_l, assume_a='pos')
            preds_eval.append(Phi_plot @ w)

        preds_eval = np.array(preds_eval) # (100, 200)
        f_bar = np.mean(preds_eval, axis=0)

        ax_left = axes[row_idx, 0]
        ax_right = axes[row_idx, 1]

        # Left plot: 20 individual fits
        for l in range(n_display_curves):
            ax_left.plot(x_plot, preds_eval[l], color='#FF0000', linewidth=0.8, alpha=0.9)

        # Right plot: True function in green, ensemble mean in red
        ax_right.plot(x_plot, h_plot, color='#008800', linewidth=2.5, label='h(x)')
        ax_right.plot(x_plot, f_bar, color='#FF0000', linewidth=2.5, label=r'$\bar{f}(x)$')

        # Formatting matching textbook style
        for ax in (ax_left, ax_right):
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-1.5, 1.5)
            ax.set_xticks([0.0, 1.0])
            ax.set_xticklabels(['0', '1'], fontsize=11)
            ax.set_yticks([-1.0, 1.0])
            ax.set_yticklabels(['$-1$', '$1$'], fontsize=11)
            ax.set_xlabel('$x$', fontsize=12)
            ax.set_ylabel('$t$', fontsize=12, rotation=0, labelpad=8)
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)

        # Title/annotation for ln lambda on left plot
        sign_str = f"{int(ln_lam)}" if ln_lam > 0 else f"{int(ln_lam)}"
        ax_left.text(0.65, 0.95, rf"$\ln \lambda = {sign_str}$", fontsize=13,
                     transform=ax_left.transAxes, verticalalignment='top')

    plt.tight_layout(h_pad=2.0, w_pad=2.0)

    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 4.7 saved to: {p}")

    if show:
        plt.show()
    return fig, axes


def plot_figure_4_8_bias_variance_tradeoff(
    simulator: Optional[BiasVarianceSimulator] = None,
    save_paths: Optional[List[str]] = None,
    show: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Faithfully reproduce Figure 4.8 from Bishop & Bishop (2024), page 128:
    Plot of squared bias, variance, their sum, and test error as a function of ln lambda.
    """
    if simulator is None:
        simulator = BiasVarianceSimulator(n_datasets=100, n_samples=25, random_state=42)

    setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=300)

    ln_lambdas = np.linspace(-3.0, 3.0, 45)
    metrics = simulator.run_sweep(ln_lambdas)

    # Plot curves matching Bishop Figure 4.8 colors
    ax.plot(metrics['ln_lambda'], metrics['bias2'], color='#FF0000', linewidth=2.0, label=r'$(\mathrm{bias})^2$')
    ax.plot(metrics['ln_lambda'], metrics['variance'], color='#0000EE', linewidth=2.0, label=r'$\mathrm{variance}$')
    ax.plot(metrics['ln_lambda'], metrics['bias2_plus_variance'], color='#008800', linewidth=2.4, label=r'$(\mathrm{bias})^2 + \mathrm{variance}$')
    ax.plot(metrics['ln_lambda'], metrics['test_error'], color='#FF00FF', linewidth=2.0, label=r'$\mathrm{test\ error}$')

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-0.02, 0.25)
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([0, 0.25])
    ax.set_yticklabels(['0', '0.25'], fontsize=11)
    ax.set_xlabel(r'$\ln \lambda$', fontsize=12)

    # Match textbook layout with legend on the right side
    ax.legend(bbox_to_anchor=(1.02, 0.82), loc='upper left', frameon=True,
              framealpha=0.9, edgecolor='#CCCCCC', fontsize=11)

    plt.tight_layout()

    if save_paths:
        for p in save_paths:
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches='tight')
            print(f"Figure 4.8 saved to: {p}")

    if show:
        plt.show()
    return fig, ax


class BiasVarianceAnalysis:
    """
    High-level analyzer for the Bias-Variance trade-off experiments in Bishop Section 4.3.
    Provides compute_tradeoff_for_lambda matching the notebook API.
    """
    def __init__(
        self,
        n_basis: int = 24,
        spatial_scale: float = 0.1,
        noise_std: float = 0.3,
        n_samples: int = 25,
        n_datasets: int = 100,
        random_state: int = 42
    ):
        self.n_basis = n_basis
        self.spatial_scale = spatial_scale
        self.noise_std = noise_std
        self.n_samples = n_samples
        self.n_datasets = n_datasets
        self.simulator = BiasVarianceSimulator(
            n_datasets=n_datasets,
            n_samples=n_samples,
            noise_std=noise_std,
            m_centers=n_basis,
            s=spatial_scale,
            random_state=random_state
        )

    def true_func(self, x: np.ndarray) -> np.ndarray:
        return self.simulator.data_gen.true_function(x)

    def compute_tradeoff_for_lambda(
        self,
        lam: float,
        x_eval: np.ndarray,
        x_test: Optional[np.ndarray] = None,
        t_test: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        preds_eval = []
        preds_test = []
        Phi_eval = self.simulator.feature_extractor(x_eval)
        Phi_test = self.simulator.feature_extractor(x_test) if x_test is not None else None
        M = self.simulator.feature_extractor.n_features
        reg_matrix = lam * np.eye(M)

        for x_l, t_l in self.simulator.ensemble:
            Phi_l = self.simulator.feature_extractor(x_l)
            w = la.solve(reg_matrix + Phi_l.T @ Phi_l, Phi_l.T @ t_l, assume_a='pos')
            preds_eval.append(Phi_eval @ w)
            if Phi_test is not None:
                preds_test.append(Phi_test @ w)

        preds_eval = np.array(preds_eval)
        f_bar = np.mean(preds_eval, axis=0)
        h_eval = self.true_func(x_eval)

        squared_bias = float(np.mean((f_bar - h_eval) ** 2))
        variance = float(np.mean(np.var(preds_eval, axis=0, ddof=0)))
        bias_variance_sum = squared_bias + variance

        if preds_test and t_test is not None:
            preds_test_arr = np.array(preds_test)
            test_error = float(np.mean((preds_test_arr - t_test[None, :]) ** 2))
        else:
            test_error = bias_variance_sum + self.noise_std ** 2

        return {
            'lambda': lam,
            'ln_lambda': float(np.log(lam)),
            'squared_bias': squared_bias,
            'variance': variance,
            'bias_variance_sum': bias_variance_sum,
            'test_error': test_error
        }


# Convenience aliases
BiasVarianceExperiment = BiasVarianceSimulator
plot_figure_4_7_bias_variance = plot_figure_4_7_bias_variance_ensembles
plot_figure_4_7_bias_variance_fits = plot_figure_4_7_bias_variance_ensembles
plot_figure_4_8_bias_variance_curves = plot_figure_4_8_bias_variance_tradeoff
plot_figure_4_8_bias_variance_tradeoff = plot_figure_4_8_bias_variance_tradeoff
