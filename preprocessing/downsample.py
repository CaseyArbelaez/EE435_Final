# preprocess_downsample.py

from skimage.transform import resize
from skimage.util import img_as_float
import numpy as np


def preprocess_downsample(image: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """
    Downsample an image to reduce small details and texture.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image of shape (H, W, 3).
    scale : float
        Scale factor in (0, 1]. Example: 0.5 halves the width and height.

    Returns
    -------
    np.ndarray
        Downsampled float image in [0, 1].
    """
    if scale <= 0 or scale > 1:
        raise ValueError("scale must be in the interval (0, 1].")

    image = img_as_float(image)
    h, w = image.shape[:2]
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    resized = resize(
        image,
        (new_h, new_w),
        anti_aliasing=True,
        preserve_range=True
    )

    return np.clip(resized, 0.0, 1.0)