from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler


class MeanShiftSegmentation:
    """
    Mean shift clustering model for image segmentation.

    This class expects a feature matrix of shape (H*W, D),
    where each row corresponds to one pixel and D is the
    number of features per pixel (for example Lab or Lab+x+y).

    Unlike k-means or spectral clustering, mean shift does not
    require specifying the number of clusters in advance.
    Instead, clusters emerge from the estimated density and are
    strongly influenced by the bandwidth parameter.
    """

    def __init__(
        self,
        bandwidth: float | None = None,
        random_state: int = 42,
        scale_features: bool = True,
        bin_seeding: bool = True,
        cluster_all: bool = True,
        max_iter: int = 300,
    ) -> None:
        self.bandwidth = bandwidth
        self.random_state = random_state
        self.scale_features = scale_features
        self.bin_seeding = bin_seeding
        self.cluster_all = cluster_all
        self.max_iter = max_iter

        self.scaler: StandardScaler | None = None
        self.model: MeanShift | None = None
        self.labels_: np.ndarray | None = None
        self.cluster_centers_: np.ndarray | None = None
        self.n_clusters_: int | None = None
        self.estimated_bandwidth_: float | None = None

    def _prepare_features(self, features: np.ndarray) -> np.ndarray:
        """
        Prepare features for clustering, including optional scaling.
        """
        if features.ndim != 2:
            raise ValueError("features must have shape (N, D).")

        X = features.astype(np.float64, copy=True)

        if self.scale_features:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        return X

    def _get_bandwidth(
        self,
        X: np.ndarray,
        quantile: float = 0.2,
        n_samples: int | None = 1000,
    ) -> float:
        """
        Determine bandwidth. If self.bandwidth is provided, use it.
        Otherwise estimate it from the data.
        """
        if self.bandwidth is not None:
            self.estimated_bandwidth_ = float(self.bandwidth)
            return float(self.bandwidth)

        bw = estimate_bandwidth(X, quantile=quantile, n_samples=n_samples)

        # fallback in case estimate_bandwidth returns 0
        if bw <= 0:
            bw = 1.0

        self.estimated_bandwidth_ = float(bw)
        return float(bw)

    def fit_predict(
        self,
        features: np.ndarray,
        quantile: float = 0.2,
        n_samples_bandwidth: int | None = 1000,
    ) -> np.ndarray:
        """
        Fit mean shift to the feature matrix and return flat labels.
        If bandwidth is None, estimate it automatically.
        """
        X = self._prepare_features(features)

        bw = self._get_bandwidth(
            X,
            quantile=quantile,
            n_samples=n_samples_bandwidth,
        )

        self.model = MeanShift(
            bandwidth=bw,
            bin_seeding=self.bin_seeding,
            cluster_all=self.cluster_all,
            max_iter=self.max_iter,
        )

        self.labels_ = self.model.fit_predict(X)
        self.cluster_centers_ = self.model.cluster_centers_
        self.n_clusters_ = len(np.unique(self.labels_))

        return self.labels_

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
        quantile: float = 0.2,
        n_samples_bandwidth: int | None = 1000,
    ) -> np.ndarray:
        """
        Fit mean shift and return segmentation map directly.
        """
        self.fit_predict(
            features,
            quantile=quantile,
            n_samples_bandwidth=n_samples_bandwidth,
        )
        return self.labels_to_image(image_shape)

    def evaluate_bandwidth_range(
        self,
        features: np.ndarray,
        image_shape: tuple[int, int] | tuple[int, int, int],
        bandwidth_values: list[float],
        show_segmentations: bool = True,
        show_variation_plot: bool = True,
        cmap: str = "nipy_spectral",
        figsize_segmentations: tuple[int, int] = (15, 8),
        figsize_plot: tuple[int, int] = (12, 5),
    ) -> dict:
        """
        Try multiple bandwidth values, store segmentation results, and
        show how the number of discovered clusters changes.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (N, D).
        image_shape : tuple
            Original image shape, usually (H, W, 3) or (H, W).
        bandwidth_values : list[float]
            Bandwidth values to test.
        """
        X = self._prepare_features(features)
        h, w = image_shape[:2]

        if len(bandwidth_values) == 0:
            raise ValueError("bandwidth_values must contain at least one value.")

        segmentations: dict[float, np.ndarray] = {}
        labels_dict: dict[float, np.ndarray] = {}
        n_clusters_dict: dict[float, int] = {}
        centers_dict: dict[float, np.ndarray] = {}

        for bw in bandwidth_values:
            model = MeanShift(
                bandwidth=bw,
                bin_seeding=self.bin_seeding,
                cluster_all=self.cluster_all,
                max_iter=self.max_iter,
            )

            labels = model.fit_predict(X)
            segmentation = labels.reshape(h, w)

            labels_dict[bw] = labels
            segmentations[bw] = segmentation
            n_clusters_dict[bw] = len(np.unique(labels))
            centers_dict[bw] = model.cluster_centers_

        if show_segmentations:
            n = len(bandwidth_values)
            n_cols = min(3, n)
            n_rows = int(np.ceil(n / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize_segmentations)
            axes = np.array(axes).reshape(-1)

            for ax, bw in zip(axes, bandwidth_values):
                ax.imshow(segmentations[bw], cmap=cmap)
                ax.set_title(f"bw = {bw:.3f}\nclusters = {n_clusters_dict[bw]}")
                ax.axis("off")

            for ax in axes[len(bandwidth_values):]:
                ax.axis("off")

            fig.suptitle("Mean Shift Segmentation for Different Bandwidths", fontsize=14)
            plt.tight_layout()
            plt.show()

        if show_variation_plot:
            plt.figure(figsize=figsize_plot)
            plt.plot(
                bandwidth_values,
                [n_clusters_dict[bw] for bw in bandwidth_values],
                marker="o",
            )
            plt.xlabel("Bandwidth")
            plt.ylabel("Number of Clusters Found")
            plt.title("Mean Shift: Number of Clusters vs Bandwidth")
            plt.grid(True)
            plt.show()

        return {
            "bandwidth_values": bandwidth_values,
            "segmentations": segmentations,
            "labels": labels_dict,
            "n_clusters": n_clusters_dict,
            "cluster_centers": centers_dict,
        }

    def summary(self) -> None:
        """
        Print a small summary of the fitted model.
        """
        if self.labels_ is None:
            print("Model has not been fit yet.")
            return

        unique, counts = np.unique(self.labels_, return_counts=True)

        print("MeanShiftSegmentation summary")
        print(
            f"bandwidth: {self.estimated_bandwidth_:.4f}"
            if self.estimated_bandwidth_ is not None
            else f"bandwidth: {self.bandwidth}"
        )
        print(f"n_clusters_found: {self.n_clusters_}")
        print("cluster sizes:")
        for u, c in zip(unique, counts):
            print(f"  cluster {u}: {c}")