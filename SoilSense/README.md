# 🌱 SoilSense v0.0 – 5‑Model Ensemble Baseline

**A foundational CNN ensemble for soil image classification (DIRT, GRASS, LEAF, MIX).**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🧠 What This Version Is

`v0.0` is the **first working prototype** of the SoilSense ensemble. It establishes the core architecture, training pipeline, and evaluation metrics that all later versions build upon.

**Key features:**
- **5‑model ensemble** with voting consensus
- **32×32 input resolution** (lightweight, fast to iterate)
- **CBAM attention** (channel + spatial)
- **Depthwise separable convolutions** + **DropPath** regularization
- **5‑stage residual backbone** with progressive channel widening
- **K‑Fold cross‑validation** (3 splits)
- **Class Activation Maps (CAM)** for interpretability
- **Automated wrong‑prediction tracking** (CSV logs)

---

## 💻 Hardware Requirements

| Component | Minimum |
|-----------|---------|
| **CPU** | Any modern CPU (4+ cores) – *training works on CPU, but is slower* |
| **GPU (optional)** | NVIDIA GPU with 8 GB+ VRAM (CUDA) – highly recommended for training |
| **RAM** | 8 GB+ |
| **Storage** | ~5 GB for dataset + models |

> ✅ **The augmentation script (`SetupMicrobiomeData.py`) runs entirely on CPU, even if you have a GPU.**  
> ✅ **Training and inference can also run on CPU** – expect 2‑5x longer training times.

---

## 📊 Training & Performance

| Metric                | Value                       |
|-----------------------|-----------------------------|
| Training images       | ~20 000 (augmented)         |
| Classes               | DIRT, GRASS, LEAF, MIX      |
| Models                | 5 (ensemble)                |
| Input size            | 32×32                       |
| Epochs per model      | 20 (early stopping)         |
| Cross‑validation      | 3‑fold                      |
| GPU (recommended)     | GTX 1080 Ti (or any CUDA GPU) |
| Expected accuracy     | 85‑90% (on clean test split)|

---

## 📁 Project Structure

```
v0.0/
├── MainMicrobiome.py          # training + inference + ensemble
├── SetupMicrobiomeData.py     # data augmentation pipeline (CPU only)
├── TestModel.py               # evaluation + CAM + error logging
└── data/
    ├── train/                 # DIRT, GRASS, LEAF, MIX subfolders
    └── test/                  # separate test set (not used during training)
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/artheidendureza-bit/soil-ai.git
cd soil-ai/v0.0
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare your dataset

Place your own images in `data/train/DIRT/`, `data/train/GRASS/`, etc.  
Each class needs **at least 20 original images** (augmentation will generate the rest).

### 4. Augment your dataset (CPU only)

```bash
python SetupMicrobiomeData.py
```

**What this does:**
- Reads original `.png` images from each class folder.
- Applies **15+ augmentation techniques** (rotations, flips, colour jitter, blur, noise, elastic deformation, weather effects, cutout, etc.).
- Generates ~500 augmented images per class (configurable).
- Runs in parallel across **all CPU cores** – no GPU needed.

**Expected output:**
```
[*] Processing 500 augmentations across ALL CPU CORES...
100%|██████████| 500/500 [00:36<00:00, 13.7 it/s]
[*] Done! Speed bottleneck resolved.
```

### 5. Train the ensemble

```bash
python MainMicrobiome.py
```

- **On GPU (recommended):** 2‑4 hours (GTX 1080 Ti)  
- **On CPU only:** 8‑12 hours (depending on your CPU)

### 6. Evaluate

```bash
python TestModel.py
```

This will run the ensemble on your test set, output accuracy, confusion matrix, and CAM visualisations.

---

## 🧪 Using the Trained Model

After training, you can classify a single image:

```python
from MainMicrobiome import load_ensemble, inference_transform
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = load_ensemble(device)

img = Image.open("your_image.jpg").convert("RGB")
tensor = inference_transform(img).unsqueeze(0).to(device)

probs = ensemble_predict(models, tensor)
pred_idx = probs.argmax().item()
print(CLASS_NAMES[pred_idx])
```

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{SoilSense_v0.0,
  author = {Artheiden Dureza},
  title = {SoilSense: 5‑Model Ensemble for Soil Classification},
  year = {2026},
  url = {https://github.com/artheidendureza-bit/soil-ai}
}
```

---

## ⚠️ Important Notes for `v0.0`

- This is the **baseline version**. Later versions (`v0.1`, `v0.11`) include more models, higher resolution, SWA, SAM, Focal Loss, and active learning.
- The test set must be **completely separate** from the training set (no image overlap).
- Training from scratch with **20 original images per class** is required; no pretrained weights are provided.
- The augmentation script **automatically deletes old `aug_*` files** and `microbiome.pth` to force a clean training run.

---

## 🤝 Contributing

Issues, suggestions, and improvements are welcome. Please open an issue or pull request.

---

**Author:** Artheiden Dureza – [GitHub](https://github.com/artheidendureza-bit)
