import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import StandardScaler
import math
from copy import deepcopy
from sklearn.metrics import silhouette_score


class KernelKMeansSegmentation:
    def __init__(
        self,
        n_clusters=3,
        max_iter=50,
        tol=1e-4,
        gamma=None,
        kernel="rbf",
        random_state=None,
        scale_features=True,
        verbose=True,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = gamma
        self.kernel = kernel
        self.random_state = random_state
        self.scale_features = scale_features
        self.verbose = verbose

        self.labels_ = None
        self.objective_history_ = []
        self.scaler_ = None
        self.X_ = None
        self.K_ = None

    def _prepare_features(self, X):
        X = np.asarray(X, dtype=np.float64)

        if self.scale_features:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        return X

    def _compute_kernel(self, X):
        params = {}
        if self.kernel == "rbf" and self.gamma is not None:
            params["gamma"] = self.gamma

        K = pairwise_kernels(X, metric=self.kernel, filter_params=True, **params)
        return K

    def _initialize_labels(self, n_samples):
        rng = np.random.default_rng(self.random_state)
        labels = rng.integers(0, self.n_clusters, size=n_samples)

        # make sure every cluster has at least one point initially
        for k in range(self.n_clusters):
            labels[k] = k

        rng.shuffle(labels)
        return labels

    def _compute_distances(self, K, labels):
        """
        Compute kernel k-means distance from each point to each cluster.

        d^2(x_i, C_k) = K_ii
                        - 2/|C_k| sum_{j in C_k} K_ij
                        + 1/|C_k|^2 sum_{p,q in C_k} K_pq
        """
        n_samples = K.shape[0]
        distances = np.zeros((n_samples, self.n_clusters), dtype=np.float64)
        diag_K = np.diag(K)

        for k in range(self.n_clusters):
            cluster_mask = labels == k
            cluster_size = np.sum(cluster_mask)

            if cluster_size == 0:
                distances[:, k] = np.inf
                continue

            K_ic = K[:, cluster_mask]                         # (n_samples, cluster_size)
            term2 = -2.0 / cluster_size * np.sum(K_ic, axis=1)

            K_cc = K[np.ix_(cluster_mask, cluster_mask)]     # (cluster_size, cluster_size)
            term3 = np.sum(K_cc) / (cluster_size ** 2)

            distances[:, k] = diag_K + term2 + term3

        return distances

    def _compute_objective(self, distances, labels):
        return np.sum(distances[np.arange(len(labels)), labels])

    def fit(self, X):
        X = self._prepare_features(X)
        self.X_ = X

        n_samples = X.shape[0]
        K = self._compute_kernel(X)
        self.K_ = K

        labels = self._initialize_labels(n_samples)
        self.objective_history_ = []

        for it in range(self.max_iter):
            old_labels = labels.copy()

            distances = self._compute_distances(K, labels)
            labels = np.argmin(distances, axis=1)

            # handle empty clusters by reassigning random points
            present_clusters = np.unique(labels)
            missing_clusters = set(range(self.n_clusters)) - set(present_clusters)

            if missing_clusters:
                rng = np.random.default_rng(self.random_state + it if self.random_state is not None else None)
                for missing_k in missing_clusters:
                    idx = rng.integers(0, n_samples)
                    labels[idx] = missing_k

            distances = self._compute_distances(K, labels)
            objective = self._compute_objective(distances, labels)
            self.objective_history_.append(objective)

            changed = np.sum(labels != old_labels)

            if self.verbose:
                print(
                    f"iter {it + 1:02d} | objective = {objective:.6f} | changed = {changed}"
                )

            if changed == 0:
                if self.verbose:
                    print("converged: labels no longer changing")
                break

            if len(self.objective_history_) >= 2:
                improvement = abs(self.objective_history_[-2] - self.objective_history_[-1])
                if improvement < self.tol:
                    if self.verbose:
                        print(f"converged: objective improvement < tol ({self.tol})")
                    break

        self.labels_ = labels
        return self

    def predict(self, X):
        if self.labels_ is None:
            raise ValueError("Model must be fit before calling predict.")

        X = np.asarray(X, dtype=np.float64)

        if self.scale_features:
            if self.scaler_ is None:
                raise ValueError("Scaler was not fit.")
            X = self.scaler_.transform(X)

        # For now, this method assumes prediction on training data only.
        # True out-of-sample kernel k-means prediction is more involved.
        if X.shape[0] != self.X_.shape[0] or not np.allclose(X, self.X_):
            raise NotImplementedError(
                "This implementation currently supports prediction only on the fitted dataset."
            )

        return self.labels_

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def fit_predict_image(self, features, image_shape):
        labels = self.fit_predict(features)
        h, w = image_shape[:2]
        return labels.reshape(h, w)

    def summary(self):
        if self.labels_ is None:
            print("Model has not been fit yet.")
            return

        print("\nKernelKMeansSegmentation Summary")
        print("--------------------------------")
        print(f"n_clusters       : {self.n_clusters}")
        print(f"kernel           : {self.kernel}")
        print(f"gamma            : {self.gamma}")
        print(f"max_iter         : {self.max_iter}")
        print(f"scale_features   : {self.scale_features}")
        print(f"n_samples        : {len(self.labels_)}")
        print(f"cluster counts   : {[int(np.sum(self.labels_ == k)) for k in range(self.n_clusters)]}")

        if self.objective_history_:
            print(f"final objective  : {self.objective_history_[-1]:.6f}")
            print(f"iterations run   : {len(self.objective_history_)}")

    def plot_objective_history(self):
        if not self.objective_history_:
            print("No objective history to plot.")
            return

        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(self.objective_history_) + 1), self.objective_history_, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.title("Kernel K-means Objective History")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def evaluate_k_range(
        self,
        features,
        image_shape,
        k_values,
        show_segmentations=True,
        show_variation_plot=True,
        figsize_per_plot=(4, 4),
        max_cols=3,
        compute_silhouette=True,
    ):
        """
        Evaluate kernel k-means over a range of k values.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (n_pixels, n_features).

        image_shape : tuple
            Shape of original image, usually (H, W, C) or (H, W).

        k_values : iterable
            Range/list of cluster counts to evaluate.

        show_segmentations : bool
            Whether to display segmentation plates for each k.

        show_variation_plot : bool
            Whether to plot quantitative metrics versus k.

        figsize_per_plot : tuple
            Size per subplot in the segmentation plate.

        max_cols : int
            Maximum number of columns in the segmentation grid.

        compute_silhouette : bool
            Whether to compute silhouette scores on the input features.

        Returns
        -------
        dict
            Dictionary containing:
            - "k_values"
            - "objectives"
            - "normalized_objectives"
            - "silhouettes"
            - "cluster_sizes"
            - "segmentations"
            - "models"
        """
        features = np.asarray(features, dtype=np.float64)
        h, w = image_shape[:2]

        results = {
            "k_values": [],
            "objectives": [],
            "normalized_objectives": [],
            "silhouettes": [],
            "cluster_sizes": [],
            "segmentations": [],
            "models": [],
        }

        for k in k_values:
            if self.verbose:
                print(f"\nEvaluating k = {k}")

            model = KernelKMeansSegmentation(
                n_clusters=k,
                max_iter=self.max_iter,
                tol=self.tol,
                gamma=self.gamma,
                kernel=self.kernel,
                random_state=self.random_state,
                scale_features=self.scale_features,
                verbose=self.verbose,
            )

            segmentation = model.fit_predict_image(features, image_shape)
            labels = model.labels_

            final_objective = (
                model.objective_history_[-1] if len(model.objective_history_) > 0 else np.nan
            )
            normalized_objective = final_objective / len(labels)

            silhouette_val = np.nan
            unique_labels = np.unique(labels)

            if compute_silhouette and len(unique_labels) > 1 and len(unique_labels) < len(labels):
                try:
                    # silhouette_score can be expensive for large images
                    # for moderate sizes this is fine
                    silhouette_val = silhouette_score(features, labels)
                except Exception:
                    silhouette_val = np.nan

            cluster_sizes = [int(np.sum(labels == c)) for c in range(k)]

            results["k_values"].append(k)
            results["objectives"].append(final_objective)
            results["normalized_objectives"].append(normalized_objective)
            results["silhouettes"].append(silhouette_val)
            results["cluster_sizes"].append(cluster_sizes)
            results["segmentations"].append(segmentation)
            results["models"].append(model)

        if show_segmentations:
            n_plots = len(results["k_values"])
            n_cols = min(max_cols, n_plots)
            n_rows = math.ceil(n_plots / n_cols)

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
            )

            axes = np.array(axes).reshape(-1)

            for ax, k, seg, obj, sil in zip(
                axes,
                results["k_values"],
                results["segmentations"],
                results["normalized_objectives"],
                results["silhouettes"],
            ):
                ax.imshow(seg, cmap="nipy_spectral")
                if np.isnan(sil):
                    ax.set_title(f"k={k}\nobj/pix={obj:.4f}")
                else:
                    ax.set_title(f"k={k}\nobj/pix={obj:.4f}, sil={sil:.3f}")
                ax.axis("off")

            for ax in axes[n_plots:]:
                ax.axis("off")

            plt.tight_layout()
            plt.show()

        if show_variation_plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(results["k_values"], results["normalized_objectives"], marker="o")
            axes[0].set_xlabel("k")
            axes[0].set_ylabel("normalized objective")
            axes[0].set_title("Kernel K-means Objective vs k")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(results["k_values"], results["silhouettes"], marker="o")
            axes[1].set_xlabel("k")
            axes[1].set_ylabel("silhouette score")
            axes[1].set_title("Silhouette Score vs k")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        return results