import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bilateral import preprocess_bilateral
from skimage import data
import matplotlib.pyplot as plt
import numpy as np


image = data.astronaut()
filtered_image = preprocess_bilateral(image, sigma_color=0.085, sigma_spatial=7)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(filtered_image)
axes[1].set_title('Bilateral Filtered Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()
