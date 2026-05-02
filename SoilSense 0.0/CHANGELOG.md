## UPDATES

**v.0: First public release of the 5-model ensemble soil classifier.**

### Features
- 5-model ensemble with voting consensus
- 128×128 input resolution
- CBAM attention (channel + spatial)
- Stochastic Weight Averaging (SWA)
- Confidence thresholding (UNSURE below 50%)
- CAM heatmap visualization
- Gradio web interface

### Dataset
- 20 original images per class (DIRT, GRASS, LEAF, MIX)
- Balanced across all 4 classes
- Augmented to ~2,000 training images

### Live Demo
[https://huggingface.co/spaces/DragonfireCoder/microbiome-ai-classifier](https://huggingface.co/spaces/DragonfireCoder/microbiome-ai-classifier)

### Installation
```bash
git clone https://github.com/DragonfireCoder/microbiome-ai.git
cd microbiome-ai
pip install -r requirements.txt
python app.py
```
**v.0.1: Bigger, Faster, Stronger**

### Added
- 7-model ensemble (up from 5 in v0.0)
- Stochastic Weight Averaging (SWA) for smoother convergence
- MixUp and CutMix augmentation
- Global RAM cache for faster training (preloads all images once)
- Victory lap: 5 extra epochs on 100% of data after validation

### Changed
- Input resolution: 32×32 → 128×128
- Base channels: 20-28 → 48-128 (wider models)
- Stage blocks: uniform → custom [2, 2, 3, 2, 1]
- Training split: K-Fold (3) → 80/20 fixed split
- Epochs: 20 → 50

### Fixed
- Inference transform missing Resize (now resizes to 128×128)

### Performance
- Better generalization (SWA, MixUp, CutMix)
- Higher accuracy on test set (see results)
