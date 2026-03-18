# preprocess_spatial.py

import numpy as np


def preprocess_spatial_features(
    image: np.ndarray,
    xy_weight: float = 0.25
) -> np.ndarray:
    """
    Convert an image into a feature matrix [color_features, x, y]
    for spatially-aware clustering.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W, C), usually RGB or Lab already normalized.
    xy_weight : float
        Weight applied to normalized x and y coordinates.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (H*W, C+2).
    """
    if image.ndim != 3:
        raise ValueError("image must have shape (H, W, C).")

    h, w, c = image.shape

    yy, xx = np.meshgrid(
        np.linspace(0, 1, h),
        np.linspace(0, 1, w),
        indexing="ij"
    )

    color_features = image.reshape(-1, c)
    spatial_features = np.stack([xx, yy], axis=-1).reshape(-1, 2)

    features = np.concatenate(
        [color_features, xy_weight * spatial_features],
        axis=1
    )

    return features