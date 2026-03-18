from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import StandardScaler

from .kmeans import KMeansSegmentation


class SpectralClusteringSegmentation:
    """
    Spectral clustering model for image segmentation.

    This class expects a feature matrix of shape (H*W, D),
    where each row corresponds to one pixel and D is the
    number of features per pixel (for example Lab or Lab+x+y).

    Pipeline:
        features
          -> affinity matrix W
          -> normalized graph Laplacian L_sym
          -> first k eigenvectors
          -> row-normalized spectral embedding
          -> k-means on embedding
          -> labels
    """

    def __init__(
        self,
        n_clusters: int = 5,
        gamma: float = 1.0,
        affinity: str = "rbf",
        random_state: int = 42,
        n_init: int = 10,
        scale_features: bool = True,
        normalize_rows: bool = True,
    ) -> None:
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.affinity = affinity
        self.random_state = random_state
        self.n_init = n_init
        self.scale_features = scale_features
        self.normalize_rows = normalize_rows

        self.scaler: StandardScaler | None = None
        self.labels_: np.ndarray | None = None
        self.embedding_: np.ndarray | None = None
        self.affinity_matrix_: np.ndarray | None = None
        self.degree_matrix_: np.ndarray | None = None
        self.laplacian_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.kmeans_model_: KMeansSegmentation | None = None

    def _prepare_features(self, features: np.ndarray) -> np.ndarray:
        """
        Prepare features for spectral clustering, including optional scaling.
        """
        if features.ndim != 2:
            raise ValueError("features must have shape (N, D).")

        X = features.astype(np.float64, copy=True)

        if self.scale_features:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        return X

    def _compute_affinity_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise affinity/similarity matrix.
        """
        if self.affinity != "rbf":
            raise ValueError("Currently only affinity='rbf' is supported.")

        W = pairwise_kernels(
            X,
            metric="rbf",
            gamma=self.gamma,
            filter_params=True,
        )

        return W

    def _compute_degree_matrix(self, W: np.ndarray) -> np.ndarray:
        """
        Compute diagonal degree matrix D where D_ii = sum_j W_ij.
        """
        degrees = np.sum(W, axis=1)
        D = np.diag(degrees)
        return D

    def _compute_normalized_laplacian(self, W: np.ndarray) -> np.ndarray:
        """
        Compute symmetric normalized graph Laplacian:
            L_sym = I - D^{-1/2} W D^{-1/2}
        """
        degrees = np.sum(W, axis=1)

        eps = 1e-12
        inv_sqrt_degrees = 1.0 / np.sqrt(np.maximum(degrees, eps))
        D_inv_sqrt = np.diag(inv_sqrt_degrees)

        n = W.shape[0]
        I = np.eye(n)

        L_sym = I - D_inv_sqrt @ W @ D_inv_sqrt
        return L_sym

    def _compute_embedding(self, L_sym: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute spectral embedding from the eigenvectors corresponding
        to the smallest eigenvalues of the normalized Laplacian.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)

        embedding = eigenvectors[:, : self.n_clusters]

        if self.normalize_rows:
            row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            row_norms = np.maximum(row_norms, 1e-12)
            embedding = embedding / row_norms

        return eigenvalues, embedding

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run spectral clustering and return flat labels.
        """
        X = self._prepare_features(features)

        W = self._compute_affinity_matrix(X)
        D = self._compute_degree_matrix(W)
        L_sym = self._compute_normalized_laplacian(W)
        eigenvalues, embedding = self._compute_embedding(L_sym)

        kmeans_model = KMeansSegmentation(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            scale_features=False,
        )

        labels = kmeans_model.fit_predict(embedding)

        self.affinity_matrix_ = W
        self.degree_matrix_ = D
        self.laplacian_ = L_sym
        self.eigenvalues_ = eigenvalues
        self.embedding_ = embedding
        self.kmeans_model_ = kmeans_model
        self.labels_ = labels

        return labels

    def labels_to_image(
        self,
        image_shape: tuple[int, int] | tuple[int, int, int],
    ) -> np.ndarray:
        """
        Reshape flat labels back into image form.
        """
        if self.labels_ is None:
            raise ValueError("No labels found. Run fit_predict first.")

        h, w = image_shape[:2]
        expected_size = h * w

        if self.labels_.size != expected_size:
            raise ValueError(
                f"Label count {self.labels_.size} does not match image size {h}x{w}={expected_size}."
            )

        return self.labels_.reshape(h, w)

    def fit_predict_image(
        self,
        features: np.ndarray,
        image_shape: tuple[int, int] | tuple[int, int, int],
    ) -> np.ndarray:
        """
        Fit spectral clustering and return segmentation map directly.
        """
        self.fit_predict(features)
        return self.labels_to_image(image_shape)

    def evaluate_k_range(
        self,
        features: np.ndarray,
        image_shape: tuple[int, int] | tuple[int, int, int],
        k_values: range | list[int],
        show_segmentations: bool = True,
        show_variation_plot: bool = True,
        cmap: str = "nipy_spectral",
        figsize_segmentations: tuple[int, int] = (15, 8),
        figsize_plot: tuple[int, int] = (12, 5),
    ) -> dict:
        """
        Try multiple k values, store segmentation results, and plot
        a few quantitative summaries.

        For spectral clustering, there is not a direct 'inertia' in the
        original space that plays the same role as vanilla k-means.
        Instead we track:
          - k-means inertia in the spectral embedding space
          - sum of first k eigenvalues
        """
        X = self._prepare_features(features)

        h, w = image_shape[:2]
        k_values = list(k_values)

        if len(k_values) == 0:
            raise ValueError("k_values must contain at least one value.")

        segmentations: dict[int, np.ndarray] = {}
        labels_dict: dict[int, np.ndarray] = {}
        embedding_inertias: dict[int, float] = {}
        eigenvalue_sums: dict[int, float] = {}
        eigenvalues_dict: dict[int, np.ndarray] = {}

        W = self._compute_affinity_matrix(X)
        L_sym = self._compute_normalized_laplacian(W)
        all_eigenvalues, all_eigenvectors = np.linalg.eigh(L_sym)

        for k in k_values:
            embedding = all_eigenvectors[:, :k]

            if self.normalize_rows:
                row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
                row_norms = np.maximum(row_norms, 1e-12)
                embedding = embedding / row_norms

            kmeans_model = KMeansSegmentation(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.n_init,
                scale_features=False,
            )

            labels = kmeans_model.fit_predict(embedding)
            segmentation = labels.reshape(h, w)

            labels_dict[k] = labels
            segmentations[k] = segmentation
            embedding_inertias[k] = (
                kmeans_model.inertia_ if kmeans_model.inertia_ is not None else np.nan
            )
            eigenvalue_sums[k] = float(np.sum(all_eigenvalues[:k]))
            eigenvalues_dict[k] = all_eigenvalues[:k].copy()

        if show_segmentations:
            n = len(k_values)
            n_cols = min(3, n)
            n_rows = int(np.ceil(n / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize_segmentations)
            axes = np.array(axes).reshape(-1)

            for ax, k in zip(axes, k_values):
                ax.imshow(segmentations[k], cmap=cmap)
                ax.set_title(f"k = {k}")
                ax.axis("off")

            for ax in axes[len(k_values):]:
                ax.axis("off")

            fig.suptitle("Spectral Clustering Segmentation for Different k Values", fontsize=14)
            plt.tight_layout()
            plt.show()

        if show_variation_plot:
            fig, axes = plt.subplots(1, 2, figsize=figsize_plot)

            axes[0].plot(k_values, [embedding_inertias[k] for k in k_values], marker="o")
            axes[0].set_xlabel("k")
            axes[0].set_ylabel("K-means Inertia in Spectral Embedding")
            axes[0].set_title("Embedding Inertia vs k")
            axes[0].grid(True)

            axes[1].plot(k_values, [eigenvalue_sums[k] for k in k_values], marker="o")
            axes[1].set_xlabel("k")
            axes[1].set_ylabel("Sum of First k Eigenvalues")
            axes[1].set_title("Spectral Criterion vs k")
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()

        return {
            "k_values": k_values,
            "segmentations": segmentations,
            "labels": labels_dict,
            "embedding_inertias": embedding_inertias,
            "eigenvalue_sums": eigenvalue_sums,
            "eigenvalues": eigenvalues_dict,
        }

    def plot_eigenvalues(self, n_values: int = 20) -> None:
        """
        Plot the smallest eigenvalues of the fitted Laplacian.

        This is often useful because a spectral gap can suggest a good k.
        """
        if self.eigenvalues_ is None:
            raise ValueError("Model has not been fit yet.")

        n_plot = min(n_values, len(self.eigenvalues_))

        plt.figure(figsize=(7, 5))
        plt.plot(range(1, n_plot + 1), self.eigenvalues_[:n_plot], marker="o")
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
        plt.title("Smallest Laplacian Eigenvalues")
        plt.grid(True)
        plt.show()

    def summary(self) -> None:
        """
        Print a small summary of the fitted model.
        """
        if self.labels_ is None:
            print("Model has not been fit yet.")
            return

        unique, counts = np.unique(self.labels_, return_counts=True)

        print("SpectralClusteringSegmentation summary")
        print(f"n_clusters: {self.n_clusters}")
        print(f"gamma: {self.gamma}")
        print(f"affinity: {self.affinity}")
        print(f"normalize_rows: {self.normalize_rows}")
        print("cluster sizes:")
        for u, c in zip(unique, counts):
            print(f"  cluster {u}: {c}")

        if self.eigenvalues_ is not None:
            n_show = min(self.n_clusters + 3, len(self.eigenvalues_))
            print("smallest eigenvalues:")
            for i in range(n_show):
                print(f"  lambda_{i+1}: {self.eigenvalues_[i]:.6f}")