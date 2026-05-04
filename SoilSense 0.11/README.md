# 🌱 SoilSense v0.11 – Professional‑Grade Ensemble

**An 8‑model ensemble for soil classification (DIRT, GRASS, LEAF, MIX) with SAM optimizer, Focal Loss, and Sentinel gatekeeper.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-blue)](https://huggingface.co/spaces/DragonfireCoder/microbiome-ai-classifier)

---

## 🧠 What This Version Is

`v0.11` is the **professional‑grade evolution** of SoilSense. It adds Sharpness‑Aware Minimization (SAM), Focal Loss, a Sentinel gatekeeper for non‑soil rejection, and an auto‑evolution pipeline.

**Key features:**
- **8‑model ensemble** (7 soil classifiers + 1 Sentinel gatekeeper)
- **128×128 input resolution**
- **CBAM attention** (channel + spatial)
- **Stochastic Weight Averaging (SWA)**
- **SAM optimizer** (Sharpness‑Aware Minimization)
- **Focal Loss** with label smoothing
- **MixUp and CutMix** augmentation
- **Global RAM cache** – preloads all images for fast training
- **NOT_SOIL detection** – rejects non‑soil images
- **Auto‑evolution pipeline** – self‑improving training loop
- **Class Activation Maps (CAM)** for interpretability
- **Confidence thresholding** (UNSURE below 50%)
- **Gradio web interface** with live demo

---

## 💻 Hardware Requirements

| Component | Minimum |
|-----------|---------|
| **GPU** | NVIDIA GPU with 8 GB+ VRAM (tested on GTX 1080 Ti) |
| **CPU** | Any modern CPU (for data loading) |
| **RAM** | 16 GB+ (8 GB may work with smaller batch size) |
| **Storage** | ~10 GB for dataset + models |

> ⚠️ **This version requires a GPU.** CPU‑friendly versions (v0.0, v0.1‑CPU, v0.11‑CPU) are available separately.

---

## 📊 Training & Performance

| Metric                | Value                       |
|-----------------------|-----------------------------|
| Training images       | ~10 000 (augmented)         |
| Classes               | DIRT, GRASS, LEAF, MIX + NOT_SOIL |
| Models                | 8 (7 soil + 1 Sentinel)     |
| Input size            | 128×128                     |
| Epochs per model      | 50 (early stopping)         |
| Optimizer             | SAM (Sharpness‑Aware Minimization) |
| Loss                  | Focal Loss (gamma=2.0)      |
| GPU (required)        | GTX 1080 Ti or better       |
| Expected accuracy     | 90‑95% (on clean test split)|

---

## 📁 Project Structure
v0.11/
├── MainMicrobiome.py # training + inference + ensemble
├── MicrobiomeEngine.py # core architecture (SAM, CBAM, Sentinel)
├── SetupMicrobiomeData.py # data augmentation pipeline
├── TestModel.py # evaluation + CAM + error logging
├── ActiveLearning.py # retrains on hard examples
├── PseudoLabel.py # self‑labeling unlabeled data
├── MasterScript.py # full automated pipeline
├── app.py # Gradio web interface
└── data/
├── train/ # DIRT, GRASS, LEAF, MIX subfolders
└── test/ # separate test set


---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/artheidendureza-bit/soil-ai.git
cd soil-ai/v0.11
```

### 2. Install dependencies
```bash
 pip install -r requirements.
```

### 3. Prepare your dataset
Place your own images in data/train/DIRT/, data/train/GRASS/, etc.
Each class needs at least 20 original images (augmentation will generate the rest).

For Sentinel training, create a data/train/NOT_SOIL/ folder with non‑soil images (rocks, hands, random objects).

### 4. Augment your dataset.
```bash
python SetupMicrobiomeData.py
```

Expected output:

```text
[*] Processing 500 augmentations across ALL CPU CORES...
100%|██████████| 500/500 [00:36<00:00, 13.7 it/s]
[*] Done!
```

### 5. Train the ensemble (GPU required)
```bash
python MainMicrobiome.py
Training time: 4‑8 hours on GTX 1080 Ti
```
The pipeline automatically runs TestModel, ActiveLearning, and GenerateMetrics after training.

### 6. Evaluate
```bash
python TestModel.py
```
Outputs accuracy, confusion matrix, and CAM visualisations.

# 🧪 Using the Trained Model
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
🌐 Live Demo
Try SoilSense in your browser:
https://huggingface.co/spaces/DragonfireCoder/microbiome-ai-classifier

Scan the QR code in the app to share with friends.

📚 Citation
If you use this work in your research, please cite:

```bibtex
@software{SoilSense_v0.11,
  author = {Artheiden Dureza},
  title = {SoilSense: 8‑Model Ensemble with SAM, Focal Loss, and Sentinel},
  year = {2026},
  url = {https://github.com/artheidendureza-bit/soil-ai}
}
```
⚠️ Important Notes for v0.11
GPU required – this version will not train on CPU.
The Sentinel model requires a NOT_SOIL folder with non‑soil images.
The auto‑evolution pipeline runs TestModel.py and ActiveLearning.py automatically after training.
Pretrained weights are not provided – train from scratch with your own images.
