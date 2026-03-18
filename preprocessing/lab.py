# preprocess_lab.py

from skimage import color
from skimage.util import img_as_float
import numpy as np


def preprocess_lab(image: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Convert an RGB image to Lab color space.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image of shape (H, W, 3).
    normalize : bool
        If True, normalize Lab channels to roughly [0, 1] scale.

    Returns
    -------
    np.ndarray
        Lab image of shape (H, W, 3).
    """
    image = img_as_float(image)
    lab = color.rgb2lab(image)

    if not normalize:
        return lab

    # L in [0, 100], a and b are roughly in [-128, 127]
    L = lab[:, :, 0] / 100.0
    a = (lab[:, :, 1] + 128.0) / 255.0
    b = (lab[:, :, 2] + 128.0) / 255.0

    lab_norm = np.stack([L, a, b], axis=-1)
    return np.clip(lab_norm, 0.0, 1.0)