# preprocess_tv.py

from skimage.restoration import denoise_tv_chambolle
from skimage.util import img_as_float
import numpy as np


def preprocess_tv(
    image: np.ndarray,
    weight: float = 0.12,
    channel_axis: int = -1
) -> np.ndarray:
    """
    Apply total variation denoising to flatten regions while preserving edges.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W, 3).
    weight : float
        Larger values give stronger smoothing / flatter regions.
    channel_axis : int
        Channel axis for skimage.

    Returns
    -------
    np.ndarray
        TV-denoised float image in [0, 1].
    """
    image = img_as_float(image)

    denoised = denoise_tv_chambolle(
        image,
        weight=weight,
        channel_axis=channel_axis
    )

    return np.clip(denoised, 0.0, 1.0)