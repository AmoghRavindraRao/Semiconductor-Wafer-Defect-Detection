# Wafer Defect Detection: Deep Learning Models

> A comprehensive machine learning project for automated wafer defect classification using Vision Transformers and ResNet models with supervised contrastive learning.

## 📋 Project Overview

This project implements a state-of-the-art wafer defect detection system designed to classify defects on semiconductor wafer maps. The system uses two parallel deep learning architectures (a custom Vision Transformer and ResNet) trained with supervised contrastive learning to achieve robust defect classification across multiple defect types.

**Course:** DSE 570 - Arizona State University, Semester 4

### Key Features

- **Dual Model Architecture**: SmallViT (custom Vision Transformer) and ResNet (50/18) models
- **Supervised Contrastive Learning**: Advanced loss function combining cross-entropy and contrastive loss
- **Multi-Stage Training Pipeline**: Stage 1 training with optional semi-supervised learning
- **Ensemble Predictions**: Combines both models for improved accuracy
- **Temperature Scaling & Calibration**: Calibrated probability estimates for reliability
- **Embedding Visualization**: Tools to analyze learned representations
- **LSWMD Dataset Integration**: Support for large-scale wafer defect dataset

### Dataset Classes

The project classifies wafer defects into 9 categories:

| Class | Description | Frequency |
|-------|-------------|-----------|
| **none** | No defects (background) | 85.3% |
| **Center** | Defect at wafer center | 2.5% |
| **Donut** | Ring-shaped defect | 0.3% |
| **Edge-Loc** | Defect localized at edge | 3.0% |
| **Edge-Ring** | Ring-shaped defect at edge | 5.6% |
| **Loc** | Localized defect | 2.1% |
| **Near-full** | Defect covering most of wafer | 0.1% |
| **Random** | Scattered random defects | 0.5% |
| **Scratch** | Linear scratch pattern | 0.7% |

## 📁 Project Structure

```
project_soft/
├── README.md                          # This file - project overview
│
├── data/
│   └── small_dataset/                 # Original small dataset (PNG images)
│       ├── train/                     # Training data by defect class
│       ├── validation/                # Validation data by defect class
│       └── test/                      # Test data by defect class
│
├── model_large/                       # Larger model configuration
│   ├── README.md                      # Model-specific documentation
│   ├── checkpoints/                   # Saved model weights
│   ├── embeddings/                    # Extracted embeddings & visualizations
│   ├── data_cache/                    # Cached preprocessed arrays
│   ├── pseudo_labels/                 # Generated pseudo-labels
│   ├── results/                       # Training and evaluation results
│   │
│   ├── models.py                      # SmallViT & ResNet architectures
│   ├── data_utils.py                  # Data loading and preprocessing
│   ├── datasets.py                    # PyTorch dataset classes
│   ├── losses.py                      # Supervised contrastive loss
│   ├── train_both.py                  # Main training script
│   ├── evaluate_both.py               # Model evaluation & ensemble
│   ├── predict.py                     # Inference script
│   ├── extract_embeddings.py          # Extract learned representations
│   ├── build_prototypes.py            # Build class prototypes
│   ├── calibrate.py                   # Temperature scaling calibration
│   ├── create_data_cache.py           # Cache creation utility
│   ├── tune_thresholds.py             # Threshold optimization
│   ├── pseudo_label.py                # Semi-supervised pseudo-labeling
│   └── LSWMD_INTEGRATION.md           # Dataset integration notes
│
├── model_small/                       # Smaller model configuration
│   ├── README.md                      # Model-specific documentation
│   ├── checkpoints/                   # Saved model weights
│   ├── embeddings/                    # Extracted embeddings & visualizations
│   ├── data_cache/                    # Cached preprocessed arrays
│   ├── pseudo_labels/                 # Generated pseudo-labels
│   ├── results/                       # Training and evaluation results
│   │
│   ├── [Same structure as model_large]
│   └── [Same Python scripts]
│
└── plot_embeddings.py                 # Visualization tool for embeddings
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch with CUDA support (recommended)
- NumPy, Pandas, Scikit-learn, Matplotlib

### Installation

1. **Navigate to project directory:**
   ```bash
   cd project_soft
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision torchaudio
   pip install numpy pandas scikit-learn matplotlib seaborn pillow
   ```

### Training

**Train both models (ViT + ResNet):**

```bash
cd model_large  # or model_small for smaller configuration

python train_both.py \
    --small_npz data_cache/small_arrays.npz \
    --epochs 50 \
    --resnet_backbone resnet50
```

**Optional: Mix in pseudo-labeled data (semi-supervised learning):**

```bash
python train_both.py \
    --small_npz data_cache/small_arrays.npz \
    --pkl_npz data_cache/pkl_arrays.npz \
    --pkl_mix_per_class 500 \
    --epochs 50
```

### Evaluation

**Evaluate trained models on test set:**

```bash
python evaluate_both.py
```

This will:
- Load both trained models
- Fit temperature scaling on validation data
- Sweep ensemble weights
- Generate predictions and metrics

### Inference

**Single image prediction:**

```bash
# Using ensemble (recommended)
python predict.py --image path/to/wafer.png

# Individual models
python predict.py --image path/to/wafer.png --model vit
python predict.py --image path/to/wafer.png --model resnet

# With test-time augmentation
python predict.py --image path/to/wafer.png --use_tta
```

**Batch prediction on directory:**

```bash
python predict.py --image_dir path/to/wafers/ --output results.csv
```

### Visualization

**Plot embedding distributions:**

```bash
python plot_embeddings.py
```

This generates 2D visualizations (PCA or t-SNE) of learned embeddings for all embedding sets.

## 🏗️ Model Architecture

### SmallViT (Custom Vision Transformer)

- **Input**: 64×64 wafer maps with values {0, 1, 2}
- **Architecture**: Patch-based Vision Transformer per PRD §6.2
- **Patch Size**: 8×8 patches
- **Embedding Dimension**: 192
- **Heads**: 4 attention heads
- **Depth**: 4 transformer blocks
- **Output**: 256-dim embeddings → 128-dim projection head

### ResNet (50 or 18)

- **Input**: 64×64 wafer maps with values {0, 1, 2}
- **Backbone**: ResNet-50 or ResNet-18 (trained from scratch, no ImageNet pretraining)
- **Output**: 256-dim embeddings → 128-dim projection head
- **Note**: Both models use identical interface for fair comparison

## 🎓 Training Pipeline

### Stage 1: Supervised Learning (Primary)

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW |
| **Learning Rate** | 3e-4 |
| **Weight Decay** | 0.05 |
| **Batch Size** | 128 |
| **Epochs** | 50 |
| **Warmup** | 5 epochs (linear) |
| **Schedule** | Cosine decay after warmup |
| **Loss** | CE (label smoothing=0.1) + 0.5 × SupCon |
| **Stopping Criterion** | Validation macro F1 (patience=10) |
| **Augmentation** | Flip, rotation, no color augmentation |

### Supervised Contrastive Loss

For each anchor sample, positives are all same-class samples in the batch, negatives are all other samples. Loss operates on L2-normalized 128-dim projection vectors with temperature=0.1.

### Optional: Semi-Supervised Mix-In

- Incorporates N samples per class from pseudo-labeled data
- Balances labeled and pseudo-labeled data in training
- Improves generalization on unlabeled splits

## 📊 Model Comparison: Large vs. Small

| Component | model_large | model_small |
|-----------|-------------|-------------|
| **ViT Embedding Dim** | 192 | 96 |
| **ViT Depth** | 4 | 2 |
| **ResNet Backbone** | resnet50 | resnet18 |
| **Total Parameters** | Higher | Lower |
| **Training Time** | Longer | Shorter |
| **Memory Usage** | Higher | Lower |
| **Expected Accuracy** | Better | Competitive |

**Use Cases:**
- **model_large**: Best accuracy, better for research/production
- **model_small**: Faster training/inference, edge deployment scenarios

## 📈 Evaluation Metrics

Models are evaluated using:

- **Macro F1 Score**: Primary metric (handles class imbalance)
- **Accuracy**: Overall correctness
- **Per-Class Precision/Recall**: Defect-specific performance
- **Confusion Matrix**: Error analysis
- **Calibration Error**: Probability estimate reliability

## 🔧 Key Utilities

### data_utils.py
- `load_small_arrays()`: Load cached PNG arrays
- `load_pkl_arrays()`: Load LSWMD.pkl data
- `pkl_mix_in()`: Sample balanced labeled data per class
- `to_canonical()`: Standardize wafer map format

### datasets.py
- `WaferArrayDataset`: PyTorch dataset with augmentation
- `train_augment()`: Discrete-preserving augmentations (flip, rotate)

### models.py
- `build_model()`: Factory for creating ViT or ResNet
- `SmallViT`: Custom vision transformer implementation
- `ResNetWafer`: ResNet adapter for wafer inputs

### utils.py
- Embedding extraction and caching
- Prototype building
- Temperature scaling calibration

## 📝 Preprocessing

All wafer maps are preprocessed to a canonical format:
1. **Canonical Size**: 64×64 pixels
2. **Value Space**: {0, 1, 2} (uint8)
  - 0 = Background
  - 1 = Die (wafer material)
  - 2 = Defect
3. **Bounding Box**: Cropped to non-background region (die + defects)
4. **No Normalization**: Discrete values preserved for ViT patch embedding

## 🎯 Results Structure

Each model directory contains:

```
results/
├── from_scratch/
│   ├── training_metrics.csv      # Loss/accuracy per epoch
│   ├── comparison.md             # Model comparison summary
│   └── predictions.csv           # Test set predictions
└── threshold_tuning/
    ├── threshold_sweep.csv       # F1 scores at different thresholds
    └── optimal_thresholds.json   # Best threshold per class
```

## 🔍 Embedding Analysis

### Extraction
```bash
python extract_embeddings.py
```

Generates:
- `train_embeddings.npy` / `train_labels.npy`: Training set embeddings
- `val_embeddings.npy` / `val_labels.npy`: Validation set
- `test_embeddings.npy` / `test_labels.npy`: Test set
- `unlabeled_embeddings.npy`: Unlabeled data representations
- `centroids.npy`: Class prototype vectors

### Visualization
```bash
python plot_embeddings.py
```

Creates 2D projections showing:
- Cluster separation by defect class
- Model representation quality
- Potential overlapping classes

## 🛠️ Model Calibration

### Temperature Scaling

```bash
python calibrate.py
```

Fits temperature parameter on validation set to calibrate confidence scores. Results saved in `checkpoints/temperature_*.npy`.

### Threshold Tuning

```bash
python tune_thresholds.py
```

Sweeps decision thresholds to optimize macro F1 score per class.

## 📦 Data Caching

Pre-processing is expensive; caching accelerates training:

```bash
python create_data_cache.py --output data_cache/small_arrays.npz
```

Cached format: `.npz` archive with:
- `X_train`, `y_train`: Training arrays and labels
- `X_val`, `y_val`: Validation split
- `X_test`, `y_test`: Test split

## 🌟 Advanced Features

### Ensemble Predictions
Combines ViT and ResNet predictions using learned ensemble weights:
```bash
python evaluate_both.py  # Computes ensemble_weight.npy
```

### Test-Time Augmentation (TTA)
Average predictions from 4 rotations (0°, 90°, 180°, 270°) for robustness:
```bash
python predict.py --image path/to/wafer.png --use_tta
```

### Pseudo-Labeling (Semi-Supervised)
Generate and refine labels for unlabeled data:
```bash
python pseudo_label.py --output pseudo_labels/pseudo_labels.csv
```

## 📚 References

- **Supervised Contrastive Learning**: Khosla et al. (2020)
- **Vision Transformers**: Dosovitskiy et al. (2020)
- **ResNet**: He et al. (2015)
- **Temperature Scaling**: Guo et al. (2017)

## 🤝 Contributing

This is a research/coursework project. For modifications:
1. Document changes in model-specific README
2. Update this main README if architecture changes
3. Test on both model_large and model_small configurations

## 📞 Contact & Support

**Course**: DSE 570 - Arizona State University  
**Semester**: 4

For questions or issues, refer to:
- Model-specific README files in `model_large/` and `model_small/`
- LSWMD_INTEGRATION.md for dataset details
- Code docstrings for implementation details

---

**Last Updated**: April 2026  
**Status**: Active Development
