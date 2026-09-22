"""
Polynomial curve fitting and regression models with analytical normal equations.
Based on Bishop & Bishop (2024), Chapter 1, Section 1.2.
"""
import numpy as np

class PolynomialFeatures:
    """Transform input scalar or 1D array x into polynomial feature matrix Phi."""
    def __init__(self, degree: int):
        self.degree = degree

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Compute design matrix Phi where Phi_{nj} = (x_n)^j for j = 0, ..., M.
        Args:
            x: 1D array of shape (N,) or (N, 1)
        Returns:
            Phi: 2D array of shape (N, degree + 1)
        """
        x_flat = np.asarray(x).reshape(-1)
        N = len(x_flat)
        Phi = np.empty((N, self.degree + 1), dtype=float)
        for j in range(self.degree + 1):
            Phi[:, j] = x_flat ** j
        return Phi

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.transform(x)


class PolynomialRegression:
    """
    Polynomial curve fitting model:
        y(x, w) = w_0 + w_1 x + w_2 x^2 + ... + w_M x^M
    Minimizes the regularized sum-of-squares error:
        E_tilde(w) = 1/2 * sum_{n=1}^N (y(x_n, w) - t_n)^2 + lambda/2 * ||w||^2
    """
    def __init__(self, degree: int = 3, l2_reg: float = 0.0):
        self.degree = degree
        self.l2_reg = l2_reg
        self.weights_ = None
        self.feature_extractor = PolynomialFeatures(degree)

    def fit(self, x: np.ndarray, t: np.ndarray) -> "PolynomialRegression":
        """
        Solve normal equation:
            (Phi^T Phi + lambda I) w = Phi^T t
        Note: Following Bishop, regularizing w_0 is standard in weight decay,
        or regularizing w_1 ... w_M without regularizing the bias w_0.
        Here we support full weight decay as in Eq. (1.4): ||w||^2 = sum_{j=0}^M w_j^2.
        """
        x_arr = np.asarray(x).reshape(-1)
        t_arr = np.asarray(t).reshape(-1)
        Phi = self.feature_extractor.transform(x_arr)
        
        A = Phi.T @ Phi
        if self.l2_reg > 0:
            A = A + self.l2_reg * np.eye(self.degree + 1)
        
        b = Phi.T @ t_arr
        self.weights_ = np.linalg.solve(A, b)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Compute model predictions y(x, w) = Phi w."""
        if self.weights_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        Phi = self.feature_extractor.transform(x)
        return Phi @ self.weights_

    def error(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        Sum-of-squares error (Eq. 1.2):
            E(w) = 1/2 * sum_{n=1}^N (y(x_n, w) - t_n)^2
        """
        y_pred = self.predict(x)
        t_arr = np.asarray(t).reshape(-1)
        return 0.5 * float(np.sum((y_pred - t_arr) ** 2))

    def rmse(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        Root-mean-square error (Eq. 1.3):
            E_RMS = sqrt(2 * E(w) / N) = sqrt(1/N * sum_{n=1}^N (y_n - t_n)^2)
        """
        t_arr = np.asarray(t).reshape(-1)
        N = len(t_arr)
        if N == 0:
            return 0.0
        return float(np.sqrt(2.0 * self.error(x, t) / N))


def generate_synthetic_data(n_samples: int = 10, noise_std: float = 0.25, random_state: int = 0):
    """
    Generate synthetic data set as described in Bishop Section 1.2.1:
    x_n uniformly spaced in [0, 1], t_n = sin(2*pi*x_n) + N(0, noise_std^2).
    """
    rng = np.random.RandomState(random_state)
    x = np.linspace(0.0, 1.0, n_samples)
    noise = rng.normal(0.0, noise_std, size=n_samples)
    t = np.sin(2.0 * np.pi * x) + noise
    return x, t


def k_fold_cross_validation(x: np.ndarray, t: np.ndarray, k: int = 4, degree: int = 3, l2_reg: float = 0.0, random_state: int = 42):
    """
    Perform S-fold cross-validation (Section 1.2.6, Figure 1.11).
    Returns list of (train_rmse, val_rmse) per fold, and the mean validation RMSE.
    """
    x_arr = np.asarray(x).reshape(-1)
    t_arr = np.asarray(t).reshape(-1)
    N = len(x_arr)
    
    indices = np.arange(N)
    rng = np.random.RandomState(random_state)
    rng.shuffle(indices)
    
    folds = np.array_split(indices, k)
    train_rmses = []
    val_rmses = []
    
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.setdiff1d(indices, val_idx)
        
        x_tr, t_tr = x_arr[train_idx], t_arr[train_idx]
        x_va, t_va = x_arr[val_idx], t_arr[val_idx]
        
        model = PolynomialRegression(degree=degree, l2_reg=l2_reg).fit(x_tr, t_tr)
        train_rmses.append(model.rmse(x_tr, t_tr))
        val_rmses.append(model.rmse(x_va, t_va))
        
    return {
        'train_rmses': train_rmses,
        'val_rmses': val_rmses,
        'mean_train_rmse': float(np.mean(train_rmses)),
        'mean_val_rmse': float(np.mean(val_rmses))
    }
