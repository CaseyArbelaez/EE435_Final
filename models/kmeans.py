from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class KMeansSegmentation:
    """
    K-means clustering model for image segmentation.

    This class expects a feature matrix of shape (H*W, D),
    where each row corresponds to one pixel and D is the
    number of features per pixel (for example RGB, Lab, or Lab+x+y).
    """

    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = 42,
        n_init: int = 10,
        scale_features: bool = True,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.scale_features = scale_features

        self.scaler: StandardScaler | None = None
        self.model: KMeans | None = None
        self.labels_: np.ndarray | None = None
        self.cluster_centers_: np.ndarray | None = None
        self.inertia_: float | None = None

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

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Fit k-means to the feature matrix and return flat labels.
        """
        X = self._prepare_features(features)

        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )

        self.labels_ = self.model.fit_predict(X)
        self.cluster_centers_ = self.model.cluster_centers_
        self.inertia_ = self.model.inertia_

        return self.labels_

    def labels_to_image(self, image_shape: tuple[int, int] | tuple[int, int, int]) -> np.ndarray:
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
        Fit k-means and return segmentation map directly.
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
        figsize_plot: tuple[int, int] = (7, 5),
    ) -> dict:
        """
        Try multiple k values, store each segmentation result, and plot
        total within-cluster variation (inertia) versus k.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (N, D).
        image_shape : tuple
            Original image shape, usually (H, W, 3) or (H, W).
        k_values : range or list[int]
            Values of k to test.
        show_segmentations : bool
            If True, display the labeled segmentation image for each k.
        show_variation_plot : bool
            If True, plot inertia vs k.
        cmap : str
            Colormap for segmentation visualization.
        figsize_segmentations : tuple[int, int]
            Figure size for segmentation grid.
        figsize_plot : tuple[int, int]
            Figure size for inertia plot.

        Returns
        -------
        dict
            Dictionary containing:
            - "k_values": list of k values
            - "segmentations": dict mapping k -> segmentation image
            - "inertias": dict mapping k -> inertia
            - "labels": dict mapping k -> flat labels
        """
        X = self._prepare_features(features)

        h, w = image_shape[:2]

        k_values = list(k_values)
        if len(k_values) == 0:
            raise ValueError("k_values must contain at least one value.")

        segmentations: dict[int, np.ndarray] = {}
        inertias: dict[int, float] = {}
        labels_dict: dict[int, np.ndarray] = {}

        for k in k_values:
            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.n_init,
            )

            labels = model.fit_predict(X)
            segmentation = labels.reshape(h, w)

            labels_dict[k] = labels
            segmentations[k] = segmentation
            inertias[k] = model.inertia_

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

            fig.suptitle("K-means Segmentation for Different k Values", fontsize=14)
            plt.tight_layout()
            plt.show()

        if show_variation_plot:
            plt.figure(figsize=figsize_plot)
            plt.plot(k_values, [inertias[k] for k in k_values], marker="o")
            plt.xlabel("k")
            plt.ylabel("Total Within-Cluster Variation (Inertia)")
            plt.title("Elbow Plot: Variation vs k")
            plt.grid(True)
            plt.show()

        return {
            "k_values": k_values,
            "segmentations": segmentations,
            "inertias": inertias,
            "labels": labels_dict,
        }

    def summary(self) -> None:
        """
        Print a small summary of the fitted model.
        """
        if self.labels_ is None:
            print("Model has not been fit yet.")
            return

        unique, counts = np.unique(self.labels_, return_counts=True)

        print("KMeansSegmentation summary")
        print(f"n_clusters: {self.n_clusters}")
        print(f"inertia: {self.inertia_:.4f}" if self.inertia_ is not None else "inertia: None")
        print("cluster sizes:")
        for u, c in zip(unique, counts):
            print(f"  cluster {u}: {c}")