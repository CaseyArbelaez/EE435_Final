from __future__ import annotations

import traceback
from skimage import data
from skimage.color import gray2rgb
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

from preprocessing.downsample import preprocess_downsample
from preprocessing.lab import preprocess_lab
from preprocessing.bilateral import preprocess_bilateral
from preprocessing.tv import preprocess_tv
from preprocessing.spatial import preprocess_spatial_features

from models import KMeansSegmentation
from models import KernelKMeansSegmentation
from models import SpectralClusteringSegmentation
from models import MeanShiftSegmentation


# ============================================================
# PREPROCESSING / FEATURE PIPELINES
# ============================================================

def build_feature_pipelines(image):
    """
    Create multiple preprocessing + feature extraction pipelines.

    Returns
    -------
    dict
        pipeline_name -> {
            "display_image": np.ndarray,
            "features": np.ndarray,
            "image_shape": tuple
        }
    """
    pipelines = {}

    # --------------------------------------------------------
    # BASELINE: Raw RGB
    # --------------------------------------------------------
    pipelines["raw_rgb"] = {
        "display_image": image,
        "features": image.reshape(-1, 3),
        "image_shape": image.shape
    }

    # --------------------------------------------------------
    # LAB ONLY (known good performer)
    # --------------------------------------------------------
    lab_img = preprocess_lab(image)
    pipelines["lab_only"] = {
        "display_image": image,
        "features": lab_img.reshape(-1, 3),
        "image_shape": lab_img.shape,
    }

    # --------------------------------------------------------
    # LAB + TV FILTERING (different weights)
    # --------------------------------------------------------
    for weight in [0.8]:
        tv_filtered = preprocess_tv(image, weight=weight)
        tv_lab = preprocess_lab(tv_filtered)
        pipelines[f"lab_tv_{weight}"] = {
            "display_image": image,
            "features": tv_lab.reshape(-1, 3),
            "image_shape": tv_lab.shape,
        }

    # --------------------------------------------------------
    # LAB + BILATERAL FILTERING
    # --------------------------------------------------------
    bilateral_filtered = preprocess_bilateral(image, sigma_color=0.08, sigma_spatial=7)
    bilateral_lab = preprocess_lab(bilateral_filtered)
    pipelines["lab_bilateral"] = {
        "display_image": image,
        "features": bilateral_lab.reshape(-1, 3),
        "image_shape": bilateral_lab.shape,
    }

    # --------------------------------------------------------
    # LAB + TV + BILATERAL (combined filtering)
    # --------------------------------------------------------
    tv_filtered = preprocess_tv(image, weight=0.12)
    tv_bilateral = preprocess_bilateral(tv_filtered, sigma_color=0.08, sigma_spatial=7)
    tv_bilateral_lab = preprocess_lab(tv_bilateral)
    pipelines["lab_tv_bilateral"] = {
        "display_image": image,
        "features": tv_bilateral_lab.reshape(-1, 3),
        "image_shape": tv_bilateral_lab.shape,
    }

    # --------------------------------------------------------
    # LAB + DOWNSCALING (different scales)
    # --------------------------------------------------------
    for scale in [0.10, 0.20]:
        downscaled = preprocess_downsample(image, scale)
        downscaled_lab = preprocess_lab(downscaled)
        pipelines[f"lab_downscale_{scale}"] = {
            "display_image": image,
            "features": downscaled_lab.reshape(-1, 3),
            "image_shape": downscaled_lab.shape,
        }

    # --------------------------------------------------------
    # LAB + TV + DOWNSCALING
    # --------------------------------------------------------
    tv_filtered = preprocess_tv(image, weight=1)
    tv_downscaled = preprocess_downsample(tv_filtered, 0.35)
    tv_downscaled_lab = preprocess_lab(tv_downscaled)
    pipelines["lab_tv_downscale"] = {
        "display_image": image,
        "features": tv_downscaled_lab.reshape(-1, 3),
        "image_shape": tv_downscaled_lab.shape,
    }

    # --------------------------------------------------------
    # LAB + BILATERAL + DOWNSCALING
    # --------------------------------------------------------
    bilateral_filtered = preprocess_bilateral(image, sigma_color=0.08, sigma_spatial=7)
    bilateral_downscaled = preprocess_downsample(bilateral_filtered, 0.8)
    bilateral_downscaled_lab = preprocess_lab(bilateral_downscaled)
    pipelines["lab_bilateral_downscale"] = {
        "display_image": image,
        "features": bilateral_downscaled_lab.reshape(-1, 3),
        "image_shape": bilateral_downscaled_lab.shape,
    }

    # --------------------------------------------------------
    # LAB + TV + BILATERAL + DOWNSCALING (comprehensive)
    # --------------------------------------------------------
    bilateral_filtered = preprocess_bilateral(image, sigma_color=0.08, sigma_spatial=7)
    bilateral_downsampled = preprocess_downsample(bilateral_filtered, 1)
    comprehensive_lab = preprocess_lab(bilateral_downsampled)
    pipelines["lab_comprehensive"] = {
        "display_image": image,
        "features": comprehensive_lab.reshape(-1, 3),
        "image_shape": comprehensive_lab.shape,
    }

    # --------------------------------------------------------
    # SPATIAL FEATURES (with LAB base)
    # --------------------------------------------------------
    lab_img = preprocess_lab(image)
    spatial_features = preprocess_spatial_features(lab_img, xy_weight=0.01)
    pipelines["lab_spatial"] = {
        "display_image": image,
        "features": spatial_features,
        "image_shape": lab_img.shape
    }

    # LAB + TV + Spatial
    tv_filtered = preprocess_tv(image, weight=0.12)
    tv_lab = preprocess_lab(tv_filtered)
    tv_spatial = preprocess_spatial_features(tv_lab, xy_weight=0.01)
    pipelines["lab_tv_spatial"] = {
        "display_image": image,
        "features": tv_spatial,
        "image_shape": tv_lab.shape
    }

    return pipelines



def plot_single_segmentation(original_image, segmentation, title):
    """
    Plot original image next to segmentation result.
    """
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap="gray" if original_image.ndim == 2 else None)
    plt.title("Original / Display Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmentation, cmap="nipy_spectral")
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.show()




# ============================================================
# EXPERIMENT RUNNERS
# ============================================================

def run_kmeans_single(features, image_shape, display_image, k):
    print(f"\nRunning KMeansSegmentation with k = {k}")

    model = KMeansSegmentation(
        n_clusters=k,
        random_state=42,
        n_init=10,
        scale_features=True,
    )

    segmentation = model.fit_predict_image(
        features=features,
        image_shape=image_shape,
    )

    model.summary()
    plot_single_segmentation(display_image, segmentation, f"K-Means Segmentation (k={k})")

    return {
        "model": model,
        "segmentation": segmentation,
    }

def run_kmeans_experiment(features, image_shape):
    print("\nRunning KMeansSegmentation over k = 2..10")

    model = KMeansSegmentation(
        random_state=42,
        n_init=10,
        scale_features=True,
    )

    return model.evaluate_k_range(
        features=features,
        image_shape=image_shape,
        k_values=range(2, 11),
        show_segmentations=True,
        show_variation_plot=False,
    )


def run_spectral_single(features, image_shape, display_image, k):
    print(f"\nRunning SpectralClusteringSegmentation with k = {k}")

    model = SpectralClusteringSegmentation(
        n_clusters=k,
        gamma=2.0,
        random_state=42,
        n_init=10,
        scale_features=True,
        normalize_rows=True,
    )

    segmentation = model.fit_predict_image(
        features=features,
        image_shape=image_shape,
    )

    model.summary()
    plot_single_segmentation(display_image, segmentation, f"Spectral Clustering (k={k})")

    return {
        "model": model,
        "segmentation": segmentation,
    }



def run_spectral_experiment(features, image_shape):
    print("\nRunning SpectralClusteringSegmentation over k = 2..10")

    model = SpectralClusteringSegmentation(
        n_clusters=3,   # placeholder, overwritten inside evaluate_k_range
        gamma=2.0,
        random_state=42,
        n_init=10,
        scale_features=True,
        normalize_rows=True,
    )

    return model.evaluate_k_range(
        features=features,
        image_shape=image_shape,
        k_values=range(2, 11),
        show_segmentations=True,
        show_variation_plot=False,
    )


def run_mean_shift_single(features, image_shape, display_image, bandwidth):
    print(f"\nRunning MeanShiftSegmentation with bandwidth = {bandwidth}")

    model = MeanShiftSegmentation(
        bandwidth=bandwidth,
        random_state=42,
        scale_features=True,
        bin_seeding=True,
        cluster_all=True,
        max_iter=300,
    )

    segmentation = model.fit_predict_image(
        features=features,
        image_shape=image_shape,
        quantile=0.2,
        n_samples_bandwidth=1000,
    )

    model.summary()
    plot_single_segmentation(display_image, segmentation, f"Mean Shift Segmentation (bw={bandwidth})")

    return {
        "model": model,
        "segmentation": segmentation,
    }



def run_mean_shift_experiment(features, image_shape):
    bandwidth_values = [0.025 * i + 0.2 for i in range(1, 10)]
    print(f"\nRunning MeanShiftSegmentation over bandwidths = {bandwidth_values}")

    model = MeanShiftSegmentation(
        bandwidth=None,
        random_state=42,
        scale_features=True,
        bin_seeding=True,
        cluster_all=True,
        max_iter=300,
    )

    return model.evaluate_bandwidth_range(
        features=features,
        image_shape=image_shape,
        bandwidth_values=bandwidth_values,
        show_segmentations=True,
        show_variation_plot=False,
    )




# ============================================================
# PIPELINE DRIVER
# ============================================================


def run_single_model_on_pipeline(
    pipeline_name,
    pipeline_data,
    model_name,
    k=None,
    bandwidth=None,
):
    """
    Run one specific model with one specific hyperparameter.
    """
    features = pipeline_data["features"]
    image_shape = pipeline_data["image_shape"]
    display_image = pipeline_data["display_image"]

    print(f"\n{'=' * 80}")
    print(f"PIPELINE: {pipeline_name}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 80}")
    print("image_shape:", image_shape)
    print("features shape:", features.shape)

    if model_name == "kmeans":
        if k is None:
            raise ValueError("k must be provided for kmeans.")
        return run_kmeans_single(features, image_shape, display_image, k)
    
    elif model_name == "spectral":
        if k is None:
            raise ValueError("k must be provided for spectral.")
        return run_spectral_single(features, image_shape, display_image, k)

    elif model_name == "mean_shift":
        if bandwidth is None:
            raise ValueError("bandwidth must be provided for mean_shift.")
        return run_mean_shift_single(features, image_shape, display_image, bandwidth)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def run_all_models_on_pipeline(pipeline_name, pipeline_data):
    """
    Run all 4 models on one preprocessing pipeline.
    """
    features = pipeline_data["features"]
    image_shape = pipeline_data["image_shape"]

    print(f"\n{'=' * 80}")
    print(f"PIPELINE: {pipeline_name}")
    print(f"{'=' * 80}")
    print("image_shape:", image_shape)
    print("features shape:", features.shape)

    results = {}

    # KMeans
    # try:
    #     results["kmeans"] = run_kmeans_experiment(features, image_shape)
    # except Exception as e:
    #     print(f"KMeans failed: {e}")
    #     traceback.print_exc()
    #     results["kmeans"] = {"error": str(e)}

    # # Spectral
    # try:
    #     results["spectral"] = run_spectral_experiment(features, image_shape)
    # except Exception as e:
    #     print(f"Spectral clustering failed: {e}")
    #     traceback.print_exc()
    #     results["spectral"] = {"error": str(e)}

    # Mean Shift
    try:
        results["mean_shift"] = run_mean_shift_experiment(features, image_shape)
    except Exception as e:
        print(f"Mean Shift failed: {e}")
        traceback.print_exc()
        results["mean_shift"] = {"error": str(e)}

    return results


def run_all_models_comparison(pipeline_name, pipeline_data, k=3, bandwidth=1.0):
    """
    Run all three models (K-means, Spectral, Mean Shift) on a pipeline
    and display results in a single 1x3 subplot.
    """
    features = pipeline_data["features"]
    image_shape = pipeline_data["image_shape"]
    display_image = pipeline_data["display_image"]

    print(f"\n{'=' * 80}")
    print(f"PIPELINE: {pipeline_name} - ALL MODELS COMPARISON")
    print(f"{'=' * 80}")
    print("image_shape:", image_shape)
    print("features shape:", features.shape)

    results = {}

    # Run K-means
    try:
        print(f"\nRunning K-means with k={k}")
        kmeans_model = KMeansSegmentation(
            n_clusters=k,
            random_state=42,
            n_init=10,
            scale_features=True,
        )
        kmeans_seg = kmeans_model.fit_predict_image(
            features=features,
            image_shape=image_shape,
        )
        results["kmeans"] = {"model": kmeans_model, "segmentation": kmeans_seg}
        print("K-means completed successfully")
    except Exception as e:
        print(f"K-means failed: {e}")
        results["kmeans"] = {"error": str(e)}

    # Run Spectral
    try:
        print(f"\nRunning Spectral with k={k}")
        # Downscale the display image using preprocessing_downsample, then extract LAB features
        scale = 0.2
        downscaled_image = preprocess_downsample(display_image, scale)
        downscaled_lab = preprocess_lab(downscaled_image)
        downscaled_features = downscaled_lab.reshape(-1, 3)
        downscaled_image_shape = downscaled_lab.shape
        
        spectral_model = SpectralClusteringSegmentation(
            n_clusters=k,
            gamma=2.0,
            random_state=42,
            n_init=10,
            scale_features=True,
            normalize_rows=True,
        )
        spectral_seg = spectral_model.fit_predict_image(
            features=downscaled_features,
            image_shape=downscaled_image_shape,
        )
        # Upscale segmentation back to original image size
        h, w = image_shape[:2]
        spectral_seg_upscaled = resize(spectral_seg.astype(float), (h, w), order=0, preserve_range=True, anti_aliasing=False).astype(int)
        results["spectral"] = {"model": spectral_model, "segmentation": spectral_seg_upscaled}
        print("Spectral completed successfully")
    except Exception as e:
        print(f"Spectral failed: {e}")
        results["spectral"] = {"error": str(e)}

    # Run Mean Shift
    try:
        print(f"\nRunning Mean Shift with bandwidth={bandwidth}")
        meanshift_model = MeanShiftSegmentation(
            bandwidth=bandwidth,
            random_state=42,
            scale_features=True,
            bin_seeding=True,
            cluster_all=True,
            max_iter=300,
        )
        meanshift_seg = meanshift_model.fit_predict_image(
            features=features,
            image_shape=image_shape,
            quantile=0.2,
            n_samples_bandwidth=1000,
        )
        results["mean_shift"] = {"model": meanshift_model, "segmentation": meanshift_seg}
        print("Mean Shift completed successfully")
    except Exception as e:
        print(f"Mean Shift failed: {e}")
        results["mean_shift"] = {"error": str(e)}

    # Create combined 1x3 subplot
    plot_all_models_comparison(display_image, results, pipeline_name)

    return results


def plot_all_models_comparison(display_image, results, pipeline_name):
    """
    Plot all three model results in a single 1x3 subplot.
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 7))

    model_names = ["image" ,"kmeans", "spectral", "mean_shift"]
    titles = ["Regular", "K-Means", "Spectral Clustering", "Mean Shift"]

    for i, (model_key, title) in enumerate(zip(model_names, titles)):
        if model_key in results and "segmentation" in results[model_key]:
            axes[i].imshow(results[model_key]["segmentation"], cmap="nipy_spectral")
            axes[i].set_title(f"{title}\n({pipeline_name})")
        else:
            # Show original image if model failed
            axes[i].imshow(display_image, cmap="gray" if display_image.ndim == 2 else None)
            axes[i].set_title(f"Regular Image")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    # Use coins image instead of astronaut - much smaller and better for testing
    # image = data.coins()
    # image = data.astronaut()  # Commented out - too large for spectral clustering
    image = imread("images/brain_tumor_glioma.jpg")
    if image.ndim == 2:
        image = gray2rgb(image)
    pipelines = build_feature_pipelines(image)
    print(f"select from the following pipeliens {pipelines.keys()}")


    # choose one pipeline - try these for best performance:
    # "lab_only" - LAB color space (good baseline)
    # "lab_tv_0.12" - LAB + TV filtering (edge-preserving smoothing)
    # "lab_bilateral" - LAB + bilateral filtering (noise reduction)
    # "lab_tv_bilateral" - LAB + TV + bilateral (comprehensive filtering)
    # "lab_comprehensive" - LAB + TV + bilateral + downscaling (all techniques)
    # "lab_spatial" - LAB + spatial coordinates (location-aware)
    # "lab_tv_spatial" - LAB + TV + spatial coordinates
    pipeline_name = "lab_comprehensive"  # Start with downscaled version for spectral clustering
    pipeline_data = pipelines[pipeline_name]

    run_all_models_on_pipeline(pipeline_name, pipeline_data)

    # Run all models on the selected pipeline and show results in 1x3 subplot
    run_all_models_comparison(
        pipeline_name=pipeline_name,
        pipeline_data=pipeline_data,
        k=5,  # number of clusters for k-means and spectral
        bandwidth=0.35  # bandwidth for mean shift
    )




    # all_results = {}

    # for pipeline_name, pipeline_data in pipelines.items():
    #     results = run_all_models_on_pipeline(pipeline_name, pipeline_data)
    #     all_results[pipeline_name] = results

    # print("\nFinished all experiments.")


if __name__ == "__main__":
    main()