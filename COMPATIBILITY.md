# Version Compatibility Guide

## v0.1 (GPU Required)

**Hardware:** NVIDIA GPU with 8GB+ VRAM (tested on GTX 1080 Ti)

**What you get:** 
- 7-model ensemble
- 128×128 resolution
- SWA, MixUp, CutMix
- Fast training (minutes to hours)


## v0.1-CPU (CPU Only)

**Hardware:** Any modern CPU with 8GB+ RAM

**What you get:**
- 3-model ensemble (lighter)
- 64×64 resolution
- Slower training (hours to days)


## v0.0 (Baseline)

**Hardware:** CPU or low-end GPU

**What you get:**
- 5-model ensemble
- 32×32 resolution
- No SWA, no MixUp

**Dataset required:** User must provide 20 images per class


## v0.1-Pretrained (No Training Needed)

**Hardware:** Same as v0.1

**What you get:**
- Ready-to-use model weights
- Download from Hugging Face
- Run inference immediately

**No dataset needed.**


## How to Choose

| If you want... | Use this version |
|----------------|------------------|
| Best accuracy, have GPU | v0.1 |
| No GPU, just want to try | v0.1-CPU |
| Retrain from scratch | v0.0 |
| Run immediately, no training | v0.1-Pretrained |
| Make your own dataset | Any version (use SetupMicrobiomeData.py) |
