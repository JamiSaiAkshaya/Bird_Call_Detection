# AI-Powered Bird Call Detection — ML Extension

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Real-time detection of endangered bird species near power lines using deep learning and acoustic monitoring.  
**v2 (ML Extension)** adds: multi-class labelling, two-phase progressive fine-tuning, Focal Loss, Mixup/CutMix, stratified K-Fold CV, Grad-CAM, calibration analysis, and ONNX INT8 export.

---

## What's new in v2

| Area | v1 (last semester) | v2 (this semester) |
|---|---|---|
| Labels | Binary (target / background) | Per-species multi-class (9 classes) |
| Data volume | ~100 recordings/species | 500 recordings/species, quality A/B filtered |
| Augmentation | SpecAugment + Gaussian noise | + Mixup, CutMix, pitch shift, time stretch, gain |
| Model head | Global avg pool + linear | **Attention pooling** + BN + SiLU |
| Fine-tuning | Frozen backbone | **Two-phase**: head → unfreeze last 3 blocks |
| Loss | CrossEntropy | **Focal Loss** (γ=2) + label smoothing |
| LR schedule | ReduceLROnPlateau | **Linear warm-up** → cosine annealing |
| Validation | Single random split | **Stratified 5-Fold CV** |
| Evaluation | Accuracy + F1 | + per-class ROC/AUC, **ECE calibration**, Grad-CAM |
| Deployment | Flask stub | **ONNX FP32 + INT8** export + latency benchmark |

---

## Quick start

### 1. Clone and set up environment

```bash
git clone https://github.com/JamiSaiAkshaya/Bird_Call_Detection.git
cd Bird_Call_Detection
conda create -n bird_detection python=3.10
conda activate bird_detection
```

### 2. Install PyTorch (choose one)

```bash
# CPU only
pip install torch torchvision torchaudio

# GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

### 4. Install project in editable mode

```bash
pip install -e .
```

> **Note — Python 3.13 users:** numpy 2.x is fully supported. Do NOT manually pin numpy<2.0.

### 4. Run the notebook

```bash
jupyter notebook notebooks/bird_detection_ml_pipeline.ipynb
```

Run all cells top to bottom. The notebook handles:
- Xeno-Canto data collection + download
- Two-phase training (or K-Fold CV)
- Full evaluation with all metrics
- Grad-CAM visualisation
- ONNX export + INT8 quantization + latency test

---

## Project structure

```
Bird_Call_Detection/
├── config/
│   └── config.yaml                     # All hyperparameters and paths
├── notebooks/
│   └── bird_detection_ml_pipeline.ipynb  # Main notebook (v2)
├── src/
│   ├── data/
│   │   ├── audio_processor.py          # Load → normalise → mel-spectrogram
│   │   ├── dataset.py                  # PyTorch Dataset + DataLoader factory
│   │   └── xeno_canto_api.py           # API client with quality filter + caching
│   ├── models/
│   │   └── efficientnet_classifier.py  # EfficientNet-B1 + attention pooling
│   ├── training/
│   │   ├── augmentation.py             # SpecAugment, Mixup, CutMix, pitch/time
│   │   ├── losses.py                   # Focal Loss, soft-label CE
│   │   └── trainer.py                  # Two-phase trainer + K-Fold CV
│   ├── evaluation/
│   │   └── evaluator.py                # Metrics, ROC, calibration, Grad-CAM, ONNX
│   └── inference/
│       └── realtime_detector.py        # PyTorch + ONNX inference wrapper
├── experiments/
│   ├── results/                        # metrics.json
│   └── figures/                        # confusion matrix, ROC, Grad-CAM, calibration
├── models/saved_models/                # .pth checkpoints + .onnx files
├── tests/
│   ├── test_audio_processor.py
│   ├── test_model.py
│   └── test_augmentation.py
├── requirements.txt
└── setup.py
```

---

## Configuration

All settings live in `config/config.yaml`. Key knobs:

```yaml
training:
  phase1_epochs: 15        # head-only training
  phase2_epochs: 45        # backbone fine-tune
  phase1_lr: 0.001
  phase2_lr: 0.0001
  use_focal_loss: true
  focal_loss_gamma: 2.0
  use_kfold: true
  kfold_splits: 5
  warmup_epochs: 5

augmentation:
  mixup:   { enabled: true, alpha: 0.3, probability: 0.4 }
  cutmix:  { enabled: true, alpha: 0.5, probability: 0.3 }
  pitch_shift:  { enabled: true, steps_range: [-1.5, 1.5] }
  time_stretch: { enabled: true, rate_range: [0.85, 1.15] }
```

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Expected results

| Metric | v1 baseline | v2 target |
|---|---|---|
| Accuracy | 72.4% | 88–93% |
| F1 weighted | 0.667 | 0.87+ |
| ECE (calibration) | — | < 0.05 |
| ONNX latency (CPU) | — | < 100 ms |
| INT8 latency (CPU) | — | < 60 ms |

---

## Target species

| Species | Conservation status | Class ID |
|---|---|---|
| Whooping Crane | Endangered | 0 |
| California Condor | Critically Endangered | 1 |
| Ivory-billed Woodpecker | Critically Endangered | 2 |
| Attwater's Prairie-chicken | Critically Endangered | 3 |
| Loggerhead Shrike | Near Threatened | 4 |
| American Robin (background) | Least Concern | 5 |
| American Crow (background) | Least Concern | 6 |
| Northern Cardinal (background) | Least Concern | 7 |
| Carolina Chickadee (background) | Least Concern | 8 |

---

## Acknowledgements

- Cornell Lab of Ornithology — BirdNET models
- Xeno-Canto — bird sound database
- EfficientNet / timm — pretrained backbone
