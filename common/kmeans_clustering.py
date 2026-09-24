"""
K-means Clustering Module (Chapter 15, Section 15.1)
====================================================
Implementation of K-means clustering, Lloyd's alternating minimization algorithm,
online/sequential Robbins-Monro updates, image segmentation, and figure reproduction
for Bishop's Deep Learning: Foundations and Concepts (2024).

Formulation:
- Objective function (distortion measure, Eq. 15.1):
    J = \sum_{n=1}^N \sum_{k=1}^K r_{nk} \|x_n - \mu_k\|^2
  where r_{nk} \in {0, 1} is 1-of-K coding with \sum_{k=1}^K r_{nk} = 1.
- E-step (Assignment, Eq. 15.2):
    r_{nk} = 1 if k = argmin_j \|x_n - \mu_j\|^2, else 0
- M-step (Update Centroids, Eq. 15.3):
    \mu_k = rac{\sum_{n=1}^N r_{nk} x_n}{\sum_{n=1}^N r_{nk}}
- Online/Sequential update (Robbins-Monro stochastic approximation, Eq. 15.4):
    \mu_k^{(	au)} = \mu_k^{(	au-1)} + \eta_	au (x_n - \mu_k^{(	au-1)})
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class KMeans:
    """K-means clustering via Lloyd's algorithm.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters K.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence (relative change in distortion or centroid movement).
    random_state : int or None, default=None
        Seed for reproducibility.
    init : str or np.ndarray, default='random'
        Initialization method: 'random', 'k-means++', or an array of shape (K, D).
    """

    def __init__(
        self,
        n_clusters: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        init: Union[str, np.ndarray] = "random",
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init

        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = 0.0
        self.n_iter_: int = 0
        self.history_: Dict[str, list] = {
            "centers": [],
            "assignments": [],
            "distortion": [],  # list of (step_float, J)
        }

    def _init_centers(self, X: np.ndarray) -> np.ndarray:
        """Initialize cluster centers."""
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape

        if isinstance(self.init, np.ndarray):
            centers = np.array(self.init, dtype=float).copy()
            if centers.shape != (self.n_clusters, n_features):
                raise ValueError(
                    f"init array must have shape ({self.n_clusters}, {n_features}), "
                    f"got {centers.shape}"
                )
            return centers

        if self.init == "random":
            indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[indices].astype(float).copy()

        if self.init == "k-means++":
            centers = np.empty((self.n_clusters, n_features), dtype=float)
            first_idx = rng.choice(n_samples)
            centers[0] = X[first_idx]
            for k in range(1, self.n_clusters):
                dists_sq = np.min(
                    [np.sum((X - centers[j]) ** 2, axis=1) for j in range(k)],
                    axis=0,
                )
                probs = dists_sq / np.sum(dists_sq)
                next_idx = rng.choice(n_samples, p=probs)
                centers[k] = X[next_idx]
            return centers

        raise ValueError(f"Unknown init method: {self.init}")

    def fit(self, X: np.ndarray) -> "KMeans":
        """Fit K-means model to X using alternating optimization.

        Parameters
        ----------
        X : np.ndarray of shape (N, D)
            Input data matrix.

        Returns
        -------
        self : KMeans
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        K = self.n_clusters

        centers = self._init_centers(X)
        self.history_ = {
            "centers": [centers.copy()],
            "assignments": [],
            "distortion": [],
        }

        labels = np.zeros(N, dtype=int)

        for iteration in range(1, self.max_iter + 1):
            # Step 1: E-step (Assignment of points to nearest centroid, Eq. 15.2)
            dists_sq = np.zeros((N, K))
            for k in range(K):
                dists_sq[:, k] = np.sum((X - centers[k]) ** 2, axis=1)
            labels = np.argmin(dists_sq, axis=1)

            # Distortion J after E-step
            J_e = float(np.sum(np.min(dists_sq, axis=1)))
            step_e = iteration - 0.5
            self.history_["distortion"].append((step_e, J_e))
            self.history_["assignments"].append(labels.copy())

            # Step 2: M-step (Update centroids, Eq. 15.3)
            new_centers = np.zeros_like(centers)
            for k in range(K):
                mask = labels == k
                if np.sum(mask) > 0:
                    new_centers[k] = np.mean(X[mask], axis=0)
                else:
                    new_centers[k] = centers[k]

            # Distortion J after M-step
            dists_sq_m = np.zeros((N, K))
            for k in range(K):
                dists_sq_m[:, k] = np.sum((X - new_centers[k]) ** 2, axis=1)
            J_m = float(np.sum(dists_sq_m[np.arange(N), labels]))
            step_m = float(iteration)
            self.history_["distortion"].append((step_m, J_m))
            self.history_["centers"].append(new_centers.copy())

            # Check convergence
            center_shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
            centers = new_centers.copy()
            self.n_iter_ = iteration

            if center_shift < self.tol:
                break

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = float(self.history_["distortion"][-1][1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster index for each sample in X."""
        if self.cluster_centers_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = np.asarray(X, dtype=float)
        dists_sq = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            dists_sq[:, k] = np.sum((X - self.cluster_centers_[k]) ** 2, axis=1)
        return np.argmin(dists_sq, axis=1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to a cluster-distance space."""
        if self.cluster_centers_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = np.asarray(X, dtype=float)
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(X - self.cluster_centers_[k], axis=1)
        return distances


def compute_distortion(
    X: np.ndarray, centers: np.ndarray, assignments: np.ndarray
) -> float:
    """Compute the K-means distortion measure J (Equation 15.1)."""
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    assignments = np.asarray(assignments, dtype=int)
    diff = X - centers[assignments]
    return float(np.sum(diff**2))


def sequential_kmeans_update(
    mu_k: np.ndarray, x_n: np.ndarray, eta: float
) -> np.ndarray:
    """Robbins-Monro sequential/online update for cluster center (Equation 15.4)."""
    return mu_k + eta * (x_n - mu_k)


def get_perpendicular_bisector(
    mu1: np.ndarray, mu2: np.ndarray, x_range: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the 2D perpendicular bisector dividing two cluster centers."""
    m = 0.5 * (mu1 + mu2)
    delta = mu2 - mu1
    dx, dy = delta[0], delta[1]

    x_vals = np.linspace(x_range[0], x_range[1], 100)
    if np.abs(dy) > 1e-9:
        y_vals = m[1] - (dx / dy) * (x_vals - m[0])
    else:
        x_vals = np.full(100, m[0])
        y_vals = np.linspace(-10, 10, 100)
    return x_vals, y_vals


def image_segmentation_kmeans(
    image: Union[np.ndarray, Image.Image, str, Path],
    K: int,
    max_iter: int = 30,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Union[int, float]]]:
    """Segment an RGB image into K color clusters using K-means."""
    if isinstance(image, (str, Path)):
        img_pil = Image.open(image).convert("RGB")
        img_arr = np.array(img_pil, dtype=np.uint8)
    elif isinstance(image, Image.Image):
        img_arr = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        img_arr = np.asarray(image)
        if img_arr.dtype != np.uint8:
            if img_arr.max() <= 1.0:
                img_arr = (img_arr * 255.0).astype(np.uint8)
            else:
                img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)

    H, W, C = img_arr.shape
    if C != 3:
        raise ValueError(f"Image must have 3 channels (RGB), got shape {img_arr.shape}")

    X = img_arr.reshape(-1, 3).astype(float)
    kmeans = KMeans(
        n_clusters=K, max_iter=max_iter, random_state=random_state, init="random"
    )
    kmeans.fit(X)

    centers_uint8 = np.clip(np.round(kmeans.cluster_centers_), 0, 255).astype(
        np.uint8
    )
    labels = kmeans.labels_.reshape(H, W)
    segmented = centers_uint8[kmeans.labels_].reshape(H, W, 3)

    bits_per_pixel_palette = int(np.ceil(np.log2(K))) if K > 1 else 1
    original_bits = H * W * 24
    compressed_bits = K * 24 + H * W * bits_per_pixel_palette
    compression_ratio = original_bits / compressed_bits
    space_savings = (1.0 - compressed_bits / original_bits) * 100.0

    compression_info = {
        "original_bits": original_bits,
        "compressed_bits": compressed_bits,
        "compression_ratio": compression_ratio,
        "space_savings_percent": space_savings,
        "bits_per_pixel": bits_per_pixel_palette,
    }

    return segmented, centers_uint8, labels, compression_info


def load_faithful_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Load and standardize the Old Faithful geyser dataset."""
    possible_paths = [
        Path("common/data/faithful.csv"),
        Path("../common/data/faithful.csv"),
        Path("/home/student/Documents/GitHub/my_DeepLearning/common/data/faithful.csv"),
    ]
    data_path = None
    for p in possible_paths:
        if p.exists():
            data_path = p
            break

    if data_path is None:
        raise FileNotFoundError(
            "Could not locate common/data/faithful.csv. Please ensure it exists."
        )

    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                data.append([float(parts[0]), float(parts[1])])

    X_raw = np.array(data, dtype=float)
    mean = np.mean(X_raw, axis=0)
    std = np.std(X_raw, axis=0)
    X_std = (X_raw - mean) / std
    return X_std, X_raw


def generate_figure_15_1(
    save_path: Optional[Union[str, Path]] = None, show: bool = False
) -> plt.Figure:
    """Generate and faithfully reproduce Figure 15.1 (Bishop, 2024)."""
    X_std, _ = load_faithful_dataset()

    mu1_init = np.array([-1.5, 1.0])
    mu2_init = np.array([1.5, -1.0])
    init_centers = np.array([mu1_init, mu2_init])

    kmeans = KMeans(n_clusters=2, max_iter=4, init=init_centers)
    kmeans.fit(X_std)

    c_green = "#00e600"
    c_blue = "#0000ff"
    c_red = "#ff0000"
    c_magenta = "#ff00ff"

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]

    history = kmeans.history_

    def plot_centers(ax, centers, has_border=False):
        if has_border:
            ax.scatter(
                centers[0, 0],
                centers[0, 1],
                marker="x",
                s=280,
                color="white",
                linewidth=7,
                zorder=4,
            )
            ax.scatter(
                centers[1, 0],
                centers[1, 1],
                marker="x",
                s=280,
                color="white",
                linewidth=7,
                zorder=4,
            )
        ax.scatter(
            centers[0, 0],
            centers[0, 1],
            marker="x",
            s=220,
            color=c_blue,
            linewidth=3.5,
            zorder=5,
        )
        ax.scatter(
            centers[1, 0],
            centers[1, 1],
            marker="x",
            s=220,
            color=c_red,
            linewidth=3.5,
            zorder=5,
        )

    for idx, ax in enumerate(axes.flat):
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.tick_params(
            direction="in", top=True, right=True, labelsize=12, length=5, width=1.2
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        ax.text(
            0.08,
            0.92,
            panel_labels[idx],
            transform=ax.transAxes,
            fontsize=15,
            fontweight="normal",
            verticalalignment="top",
        )

        if idx == 0:
            ax.scatter(
                X_std[:, 0],
                X_std[:, 1],
                color=c_green,
                s=25,
                alpha=1.0,
                edgecolors="none",
                zorder=2,
            )
            plot_centers(ax, history["centers"][0], has_border=False)

        elif idx % 2 == 1:
            iter_num = (idx + 1) // 2
            assign_idx = iter_num - 1
            labels = history["assignments"][assign_idx]
            centers = history["centers"][assign_idx]

            ax.scatter(
                X_std[labels == 0, 0],
                X_std[labels == 0, 1],
                color=c_blue,
                s=25,
                alpha=1.0,
                edgecolors="none",
                zorder=2,
            )
            ax.scatter(
                X_std[labels == 1, 0],
                X_std[labels == 1, 1],
                color=c_red,
                s=25,
                alpha=1.0,
                edgecolors="none",
                zorder=2,
            )

            bx, by = get_perpendicular_bisector(
                centers[0], centers[1], (-2.5, 2.5)
            )
            ax.plot(bx, by, color=c_magenta, linewidth=2.0, zorder=3)

            plot_centers(ax, centers, has_border=False)

        else:
            iter_num = idx // 2
            assign_idx = iter_num - 1
            labels = history["assignments"][assign_idx]
            centers = history["centers"][iter_num]

            ax.scatter(
                X_std[labels == 0, 0],
                X_std[labels == 0, 1],
                color=c_blue,
                s=25,
                alpha=1.0,
                edgecolors="none",
                zorder=2,
            )
            ax.scatter(
                X_std[labels == 1, 0],
                X_std[labels == 1, 1],
                color=c_red,
                s=25,
                alpha=1.0,
                edgecolors="none",
                zorder=2,
            )

            plot_centers(ax, centers, has_border=True)

    plt.tight_layout(pad=1.2)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def generate_figure_15_2(
    save_path: Optional[Union[str, Path]] = None, show: bool = False
) -> plt.Figure:
    """Generate and faithfully reproduce Figure 15.2 (Bishop, 2024)."""
    X_std, _ = load_faithful_dataset()
    mu1_init = np.array([-1.5, 1.0])
    mu2_init = np.array([1.5, -1.0])
    init_centers = np.array([mu1_init, mu2_init])

    kmeans = KMeans(n_clusters=2, max_iter=4, init=init_centers)
    kmeans.fit(X_std)

    steps = [item[0] for item in kmeans.history_["distortion"]]
    distortions = [item[1] for item in kmeans.history_["distortion"]]

    e_steps = [s for s in steps if s % 1.0 != 0]
    e_dists = [
        d for s, d in zip(steps, distortions) if s % 1.0 != 0
    ]

    m_steps = [s for s in steps if s % 1.0 == 0]
    m_dists = [
        d for s, d in zip(steps, distortions) if s % 1.0 == 0
    ]

    c_green = "#00e600"
    c_blue = "#0000ff"
    c_red = "#ff0000"

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(steps, distortions, color=c_green, linewidth=2.5, zorder=2)

    ax.scatter(
        e_steps,
        e_dists,
        s=120,
        facecolors="none",
        edgecolors=c_blue,
        linewidths=2.5,
        zorder=3,
    )

    ax.scatter(
        m_steps,
        m_dists,
        s=120,
        facecolors="none",
        edgecolors=c_red,
        linewidths=2.5,
        zorder=3,
    )

    ax.set_xlim(0.1, 4.3)
    ax.set_ylim(0, 1150)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_yticks([0, 500, 1000])

    ax.tick_params(
        direction="in", top=True, right=True, labelsize=14, length=6, width=1.2
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax.set_ylabel(
        r"$J$",
        fontsize=18,
        rotation=0,
        labelpad=15,
        verticalalignment="center",
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def generate_figure_15_3(
    save_path: Optional[Union[str, Path]] = None, show: bool = False
) -> plt.Figure:
    """Generate and faithfully reproduce Figure 15.3 (Bishop, 2024)."""
    possible_img_paths = [
        Path("common/data/segmentation_source.png"),
        Path("../common/data/segmentation_source.png"),
        Path(
            "/home/student/Documents/GitHub/my_DeepLearning/common/data/segmentation_source.png"
        ),
    ]
    img_path = None
    for p in possible_img_paths:
        if p.exists():
            img_path = p
            break

    if img_path is None:
        raise FileNotFoundError(
            "Could not locate common/data/segmentation_source.png"
        )

    orig_img = Image.open(img_path).convert("RGB")
    orig_arr = np.array(orig_img)

    seg_k2, _, _, _ = image_segmentation_kmeans(orig_arr, K=2, random_state=42)
    seg_k3, _, _, _ = image_segmentation_kmeans(orig_arr, K=3, random_state=42)
    seg_k10, _, _, _ = image_segmentation_kmeans(
        orig_arr, K=10, random_state=42
    )

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    panels = [
        (orig_arr, "Original image"),
        (seg_k2, r"$K = 2$"),
        (seg_k3, r"$K = 3$"),
        (seg_k10, r"$K = 10$"),
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=16, pad=10)
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
