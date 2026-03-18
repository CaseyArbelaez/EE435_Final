# Image Segmentation Experiments

This repository contains exploratory image segmentation experiments using scikit-image preprocessing pipelines and clustering algorithms.

## 🚀 What’s Included

- **`main.py`**: Runs preprocessing pipelines (LAB, TV filtering, bilateral filtering, spatial features, downsampling) and evaluates segmentation algorithms:
  - K-means
  - Spectral clustering
  - Mean shift
- **`testing.py`**: Quick viewer for built-in `skimage.data` images (shows them in a 2×2 grid and lets you close each figure to continue).
- **`preprocessing/`**: Custom preprocessing modules (TV denoising, bilateral filtering, Lab conversion, spatial features, downsampling).
- **`models/`**: Segmentation model wrappers (KMeans, Spectral, MeanShift, etc.).
- **`images/`**: Example image(s) used by the scripts.

## ✅ Setup

> This project assumes Python 3.8+.

1. **Create a virtual environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate
```

2. **Install dependencies**

```bash
pip install scikit-image matplotlib numpy scipy scikit-learn
```

> If you add a `requirements.txt`, you can run:
>
> ```bash
> pip install -r requirements.txt
> ```

## ▶️ Running the Scripts

### 1) Browse skimage sample images (2×2 grid)

```bash
python testing.py
```

Close each matplotlib window to move to the next set of images.

### 2) Run the segmentation experiment

```bash
python main.py
```

The default configuration uses `images/brain_tumor_glioma.jpg` and runs mean shift + comparison plots.

You can edit `main.py` to change which pipeline is used (see `pipeline_name`) or swap images.

## 🧠 Notes

- The code is designed for experimentation; feel free to tweak preprocessing parameters and algorithm hyperparameters.
- For faster results, use smaller images or downsample before clustering.

---

If you want, I can also help you add a `requirements.txt` and/or a simple `setup.py`/`pyproject.toml` for packaging.