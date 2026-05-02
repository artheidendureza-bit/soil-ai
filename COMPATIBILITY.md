# Version Compatibility Guide

## v0.1 (GPU Required)

**Hardware:** NVIDIA GPU with 8GB+ VRAM (tested on GTX 1080 Ti)

**What you get:** 
- 7-model ensemble
- 128×128 resolution
- SWA, MixUp, CutMix
- Fast training (minutes to hours)

## v0.0 (CPU-FRIENDLY)

**Hardware:** CPU or low-end GPU

**What you get:**
- 5-model ensemble
- 32×32 resolution
- No SWA, no MixUp

**Dataset required:** User must provide 20 images per class

## How to Choose

| If you want... | Use this version |
|----------------|------------------|
| Best accuracy, have GPU | v0.1 |
| No GPU, just want to try | v0.0 |
| Retrain from scratch | v0.1 |
| Run immediately, no training | v0.0 |
| Make your own dataset | Any version (use SetupMicrobiomeData.py) |
