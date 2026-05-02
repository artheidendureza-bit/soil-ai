# Microbiome AI Classifier

**A 7-model ensemble for soil classification: DIRT, GRASS, LEAF, or MIX**

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-blue)](https://huggingface.co/spaces/DragonfireCoder/microbiome-ai-classifier)

## Quick Start
Upload a photo of soil. The ensemble of 7 AI models votes on DIRT, GRASS, LEAF, or MIX.

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/DragonfireCoder/microbiome-ai-classifier)

## Dataset
- 20 original images per class (balanced)
- Dual domain: lab (petri dish) + field (outdoor)
- Augmented to ~20,000 training images per class

## Model
- 7-model ensemble with voting consensus
- CBAM attention (channel + spatial)
- Stochastic Weight Averaging (SWA)
- 128×128 input resolution

## Version
**v0.1** - Current
- Dataset: 20 originals per class
- Model: 7 models, SWA, CBAM, 128×128
- Features: Confidence thresholding, CAM visualization

## Local Install
```bash
git clone https://github.com/DragonfireCoder/microbiome-ai.git
cd microbiome-ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
---
**Author:** Artheiden 

## HOW TO USE

### Web App (EASIEST)

1. Go to the [Hugging Face Space](https://huggingface.co/spaces/DragonfireCoder/microbiome-ai-classifier)
2. Wait for the app to load (it may take 30-60 seconds to start)
3. Upload a photo of soil, grass, leaves, or a mix
4. Click **"Analyze"**
5. View the results:
   - **Final prediction** (DIRT, GRASS, LEAF, MIX, or UNSURE)
   - **Confidence scores** for all 4 classes
   - **CAM heatmap** showing where the AI focused

*Note: If the Space is sleeping, the first load may take a minute. Refresh if needed.*

### Local (Run on Your Computer)

1. Clone the repository:
   ```bash
   git clone https://github.com/DragonfireCoder/microbiome-ai.git
   cd microbiome-ai

2. Create a virtual environment
    python -m venv venv
    source venv/bin/activate      # Mac/Linux
    venv\Scripts\activate         # Windows

3. Install the dependencies
    pip install -r requirements.txt

4. Run the app!
    python app.py