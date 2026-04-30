# Installation Guide — Windows / Python 3.13

Follow these steps **in order**. Do not skip any step.

---

## Step 1 — Clone the repo

```cmd
git clone https://github.com/JamiSaiAkshaya/Bird_Call_Detection.git
cd Bird_Call_Detection
```

---

## Step 2 — Install PyTorch first (before anything else)

**CPU only (recommended if no NVIDIA GPU):**
```cmd
pip install torch torchvision torchaudio
```

**GPU — CUDA 12.1:**
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify PyTorch installed correctly:
```cmd
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

---

## Step 3 — Install all other dependencies

```cmd
pip install -r requirements.txt
```

If you see any individual package fail, install it separately:
```cmd
pip install librosa soundfile timm onnx onnxruntime scikit-learn pandas matplotlib seaborn tqdm pyyaml requests ipykernel jupyter
```

---

## Step 4 — Install the project package

```cmd
pip install -e .
```

---

## Step 5 — Verify everything works

```cmd
python -c "import torch, librosa, timm, onnx, sklearn, pandas, numpy; print('All imports OK')"
```

---

## Step 6 — Run the notebook

```cmd
jupyter notebook notebooks/bird_detection_ml_pipeline.ipynb
```

---

## Common errors and fixes

| Error | Fix |
|---|---|
| `numpy metadata-generation-failed` | You already have numpy 2.x — this is fine. Run `pip install -r requirements.txt` again, it will skip numpy. |
| `No module named 'timm'` | `pip install timm` |
| `No module named 'onnxruntime'` | `pip install onnxruntime` |
| `No module named 'librosa'` | `pip install librosa soundfile audioread` |
| `No module named 'cv2'` | `pip install opencv-python-headless` |
| Import error about `torch.cuda.amp` | Update PyTorch: `pip install --upgrade torch torchvision torchaudio` |
| Kernel not found in Jupyter | `pip install ipykernel` then `python -m ipykernel install --user --name bird_detection` |

---

## Minimum requirements

- Python 3.10, 3.11, 3.12, or 3.13
- 8 GB RAM (16 GB recommended)
- Internet connection (Xeno-Canto API downloads audio files)
- GPU optional — CPU training works but is slower
