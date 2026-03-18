# preprocess_bilateral.py

from skimage.restoration import denoise_bilateral
from skimage.util import img_as_float
import numpy as np


def preprocess_bilateral(
    image: np.ndarray,
    sigma_color: float = 0.08,
    sigma_spatial: float = 5.0,
    channel_axis: int = -1
) -> np.ndarray:
    """
    Apply bilateral filtering to smooth within objects while preserving edges.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W, 3).
    sigma_color : float
        Larger values smooth more across color differences.
    sigma_spatial : float
        Larger values smooth over larger spatial neighborhoods.
    channel_axis : int
        Channel axis for skimage.

    Returns
    -------
    np.ndarray
        Bilaterally filtered float image in [0, 1].
    """
    image = img_as_float(image)

    filtered = denoise_bilateral(
        image,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        channel_axis=channel_axis
    )

    return np.clip(filtered, 0.0, 1.0)