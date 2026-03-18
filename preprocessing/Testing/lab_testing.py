import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lab import preprocess_lab
from bilateral import preprocess_bilateral
from tv import preprocess_tv
from downsample import preprocess_downsample
from skimage import data
import matplotlib.pyplot as plt
import numpy as np


image = data.astronaut()

# 1) LAB transformation
lab_image = preprocess_lab(image)

# 2) Bilateral filter (applied in RGB space)
bilateral_image = preprocess_bilateral(image, sigma_color=0.08, sigma_spatial=7)

# 3) Downscaling (on original RGB image)
down_image = preprocess_downsample(image, scale=0.25)

# Define the comparison pairs we want to save (title, processed image, output filename)
comparisons = [
    ("LAB Transform", lab_image, "comparison_lab.png"),
    ("Bilateral Filter", bilateral_image, "comparison_bilateral.png"),
    ("Downscaled Image", down_image, "comparison_downscale.png"),
]

out_dir = os.path.dirname(__file__)

for title, proc_img, out_name in comparisons:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(proc_img)
    axes[1].set_title(title)
    axes[1].axis("off")

    plt.tight_layout()

    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=150)
    print(f"Saved comparison: {out_path}")

    plt.show()
