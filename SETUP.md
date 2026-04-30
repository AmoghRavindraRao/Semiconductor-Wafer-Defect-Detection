# WaferVision — Setup & Run Guide

> **For:** Teammate running on Windows  
> **Time:** ~5 minutes

---

## Prerequisites

- Python 3.8+ installed
- Git installed with Git LFS
- The cloned repo

---

## Step 1 — Clone & pull checkpoints

```bash
git clone https://github.com/rohithraju-ops/Semiconductor-Wafer-Defect-Detection.git
cd Semiconductor-Wafer-Defect-Detection

# Pull the real checkpoint files (Git LFS)
git lfs install
git lfs pull
```

Verify the checkpoints are real (should be MBs, not bytes):

```bash
# Windows PowerShell
dir model_large\checkpoints\*.pth
# Expected: resnet_best.pth ~92MB, vit_best.pth ~10MB
```

---

## Step 2 — Create virtual environment

```bash
python -m venv swdd
swdd\Scripts\activate
```

---

## Step 3 — Install dependencies

```bash
pip install fastapi "uvicorn[standard]" databases aiosqlite sqlalchemy
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pillow opencv-python-headless
```

> **Note:** The `--index-url` flag installs the CPU-only version of PyTorch  
> (smaller download, works fine without a GPU).

---

## Step 4 — Generate the frontend files

```bash
python generate_wafer_files.py
```

This writes `api.py` and `dashboard.html` to the repo root.

---

## Step 5 — Start the server

```bash
# Windows PowerShell
$env:WAFER_MODEL="large"; uvicorn api:app --reload --port 8000
```

You should see:

```
[startup] model_large: ViT-192 (4L 4H) + ResNet-50  loaded OK
```

---

## Step 6 — Open the dashboard

Open your browser and go to:

```
http://localhost:8000
```

The navbar should show **🟢 LargeViT · ResNet-50**.

---

## Testing Live Inference

1. Go to the **Live Inference** tab
2. Click **Choose File** and upload any wafer PNG  
   *(test images are in `data/small_dataset/test/` — pick any subfolder)*
3. Select method: `Ensemble` (recommended)
4. Click **Run Inference**
5. Result appears with predicted class + confidence + per-class probabilities

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named 'cv2'` | `pip install opencv-python-headless` |
| `invalid load key, 'v'` | Checkpoint is LFS pointer — run `git lfs pull` |
| `Weights only load failed` | Already patched in api.py — ignore |
| `Model Offline` in navbar | Checkpoints missing — check Step 1 |
| Port 8000 already in use | Change `--port 8000` to `--port 8001` and open `localhost:8001` |
| `No module named 'data_utils'` | Make sure you are running from the repo root, not from inside model_large/ |

---

## Project Structure (relevant files)

```
Semiconductor-Wafer-Defect-Detection/
├── api.py                          ← FastAPI backend (run this)
├── dashboard.html                  ← Frontend (served automatically at /)
├── generate_wafer_files.py         ← Regenerates api.py + dashboard.html
│
├── model_large/
│   ├── predict.py                  ← Inference engine
│   ├── models.py                   ← ViT + ResNet architectures
│   ├── data_utils.py               ← Preprocessing
│   ├── checkpoints/
│   │   ├── vit_best.pth            ← ViT weights (~10MB)
│   │   ├── resnet_best.pth         ← ResNet-50 weights (~92MB)
│   │   ├── temperature_vit.npy     ← Calibration
│   │   ├── temperature_resnet.npy  ← Calibration
│   │   └── ensemble_weight.npy     ← Ensemble blend (ResNet=1.0)
│   └── results/
│       ├── from_scratch/
│       │   ├── comparison.md       ← F1 scores (feeds Performance page)
│       │   └── predictions.csv     ← Full test set predictions
│       └── threshold_tuning/
│           ├── chosen_thresholds.json  ← τ values (feeds threshold panel)
│           └── sweep_results.csv       ← 243 combos swept
```

---

## Quick Reference

| What | Command |
|------|---------|
| Start server | `$env:WAFER_MODEL="large"; uvicorn api:app --reload --port 8000` |
| Stop server | `Ctrl + C` |
| Open dashboard | `http://localhost:8000` |
| Check API health | `http://localhost:8000/health` |
| View results JSON | `http://localhost:8000/results` |
| View threshold data | `http://localhost:8000/threshold` |

---

*DSE 570 Capstone — Arizona State University — April 2026*
