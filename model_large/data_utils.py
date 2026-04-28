"""
data_utils.py

Single source of truth for:
  - to_canonical(arr): bbox-crop + pad-to-square + nearest-resize to 64x64
  - rgb_to_class_array(rgb): RGB PNG -> {0,1,2} array
  - WaferDataset: returns (vit_input, resnet_input, label)
  - pkl_mix_in(): grabs N-per-class pkl-labeled samples for Stage 1 training
  - load_lswmd_and_create_cache(): loads LSWMD.pkl, processes, and saves NPZ cache

Class mapping is hard-coded per PRD §5.
"""
from pathlib import Path
from typing import Tuple, Optional, Dict
import sys
import pickle

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

# ---- PRD §5: fixed mapping, never inferred from folder order ----
CLASS_TO_IDX = {
    "Center":    0, "Donut":     1, "Edge-Loc":  2, "Edge-Ring": 3,
    "Loc":       4, "Near-full": 5, "Random":    6, "Scratch":   7,
    "none":      8,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES  = 9
CANON_SIZE   = 64


# ---------------------------------------------------------------------------
# Canonical preprocessing (matches your latest Stage 1 to_canonical)
# ---------------------------------------------------------------------------
def to_canonical(wm: np.ndarray) -> np.ndarray:
    """Clip into {0,1,2}, crop to wafer bbox, pad to square, nearest-resize to 64x64."""
    wm = np.asarray(wm)
    wm = np.clip(wm, 0, 2).astype(np.uint8)

    # Bounding-box crop of non-background pixels (die + defect)
    nonbg = wm > 0
    if nonbg.any():
        ys, xs = np.where(nonbg)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        wm = wm[y0:y1, x0:x1]

    # Pad to square (centered)
    h, w = wm.shape[:2]
    if h != w:
        side = max(h, w)
        padded = np.zeros((side, side), dtype=np.uint8)
        y_off = (side - h) // 2
        x_off = (side - w) // 2
        padded[y_off:y_off + h, x_off:x_off + w] = wm
        wm = padded

    return cv2.resize(wm, (CANON_SIZE, CANON_SIZE), interpolation=cv2.INTER_NEAREST)


def rgb_to_class_array(rgb: np.ndarray) -> np.ndarray:
    """RGB PNG -> {0,1,2} array per PRD §4.2 colormap."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return np.where(b > 128, 2,
           np.where(g > 128, 1, 0)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Loaders for cached arrays (preferred — fast)
# ---------------------------------------------------------------------------
def load_small_arrays(npz_path: str, lswmd_pkl_path: Optional[str] = None):
    """
    Loads small-dataset cached arrays. 
    If NPZ doesn't exist and lswmd_pkl_path is provided, creates cache from LSWMD.pkl.
    Expects keys train_x, train_y, val_x, val_y, test_x, test_y.
    """
    npz_path = Path(npz_path)
    
    # If NPZ exists, load it
    if npz_path.exists():
        z = np.load(npz_path)
        return {
            "train_x": z["train_x"], "train_y": z["train_y"],
            "val_x":   z["val_x"],   "val_y":   z["val_y"],
            "test_x":  z["test_x"],  "test_y":  z["test_y"],
        }
    
    # If NPZ doesn't exist but LSWMD path is provided, create it
    if lswmd_pkl_path and Path(lswmd_pkl_path).exists():
        print(f"NPZ cache not found at {npz_path}")
        print(f"Creating cache from LSWMD.pkl...")
        output_dir = npz_path.parent
        return load_lswmd_and_create_cache(lswmd_pkl_path, str(output_dir))
    
    # Neither exists - try to find LSWMD.pkl in common locations
    if not lswmd_pkl_path:
        common_paths = [
            Path("LSWMD.pkl"),
            Path("../LSWMD.pkl"),
            Path("../../LSWMD.pkl"),
            Path(npz_path.parent.parent / "LSWMD.pkl"),
        ]
        for p in common_paths:
            if p.exists():
                print(f"Found LSWMD.pkl at {p}")
                lswmd_pkl_path = str(p)
                output_dir = npz_path.parent
                return load_lswmd_and_create_cache(lswmd_pkl_path, str(output_dir))
    
    raise FileNotFoundError(
        f"NPZ file not found at {npz_path} and no LSWMD.pkl available. "
        f"Please provide lswmd_pkl_path or ensure cache exists."
    )


def load_pkl_arrays(npz_path: str):
    """Loads pkl-labeled cached arrays. Expects keys labeled_x, labeled_y."""
    z = np.load(npz_path)
    return {"labeled_x": z["labeled_x"], "labeled_y": z["labeled_y"]}


def pkl_mix_in(pkl_data, per_class: int, rng_seed: int = 42):
    """Sample `per_class` labeled wafers from each class. Returns (x, y)."""
    rng = np.random.default_rng(rng_seed)
    keep = []
    for c in range(NUM_CLASSES):
        cidx = np.where(pkl_data["labeled_y"] == c)[0]
        if len(cidx) == 0:
            continue
        n = min(per_class, len(cidx))
        keep.append(rng.choice(cidx, size=n, replace=False))
    keep = np.concatenate(keep)
    return pkl_data["labeled_x"][keep], pkl_data["labeled_y"][keep]


def _load_lswmd_pkl(pkl_path: str) -> 'pd.DataFrame':
    """Load LSWMD.pkl with compatibility for older pandas versions."""
    import sys
    # Fix pandas.indexes import issue for older pickles
    import pandas as pd
    sys.modules['pandas.indexes'] = sys.modules['pandas.core.indexes']
    sys.modules['pandas.indexes.base'] = sys.modules['pandas.core.indexes.base']
    sys.modules['pandas.core.common'] = sys.modules['pandas.core']
    
    with open(pkl_path, 'rb') as f:
        for encoding in [None, 'latin1', 'utf-8', 'bytes']:
            try:
                f.seek(0)
                if encoding:
                    data = pickle.load(f, encoding=encoding)
                else:
                    data = pickle.load(f)
                return data
            except Exception:
                continue
    raise RuntimeError(f"Failed to load {pkl_path}")


def _extract_label_from_array(x: np.ndarray) -> Optional[str]:
    """Extract string label from nested array format."""
    if hasattr(x, 'flatten'):
        flat = x.flatten()
        if len(flat) > 0:
            val = flat[0]
            return str(val) if val is not None else None
    elif isinstance(x, (list, tuple)) and len(x) > 0:
        item = x[0]
        if isinstance(item, (list, tuple)) and len(item) > 0:
            val = item[0]
            return str(val) if val is not None else None
    return None


def _process_wafer_map(wm: np.ndarray) -> Optional[np.ndarray]:
    """
    Process wafer map to canonical format.
    Returns None if processing fails.
    """
    try:
        # Ensure it's uint8 in {0,1,2}
        wm = np.asarray(wm, dtype=np.uint8)
        wm = np.clip(wm, 0, 2)
        
        # Apply to_canonical preprocessing
        arr = to_canonical(wm)
        return arr
    except Exception:
        return None


def load_lswmd_and_create_cache(
    lswmd_pkl_path: str,
    output_dir: str = "data_cache",
    val_split: float = 0.1,
    test_split: float = 0.1,
    rng_seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Load LSWMD.pkl, process wafers, split into train/val/test, and save as NPZ.
    
    Returns dict with keys: train_x, train_y, val_x, val_y, test_x, test_y
    """
    import pandas as pd
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading LSWMD.pkl from {lswmd_pkl_path}...")
    df = _load_lswmd_pkl(lswmd_pkl_path)
    print(f"Loaded DataFrame: shape {df.shape}")
    
    # Extract labels
    print("Extracting labels and wafer maps...")
    def extract_label(x):
        return _extract_label_from_array(x)
    
    df['label'] = df['failureType'].apply(extract_label)
    df['train_test'] = df['trianTestLabel'].apply(extract_label)
    
    # Filter for labeled samples (where failureType is not None)
    df_labeled = df[df['label'].notna()].copy()
    print(f"Labeled samples: {len(df_labeled)} / {len(df)}")
    
    # Map labels to class indices
    print("Mapping labels to class indices...")
    label_to_idx = {cls_name: idx for cls_name, idx in CLASS_TO_IDX.items()}
    df_labeled['class_idx'] = df_labeled['label'].map(label_to_idx)
    
    # Remove any with unmapped labels
    df_labeled = df_labeled[df_labeled['class_idx'].notna()].copy()
    df_labeled['class_idx'] = df_labeled['class_idx'].astype(np.int64)
    print(f"Samples with valid class: {len(df_labeled)}")
    
    # Process wafer maps
    print("Processing wafer maps...")
    processed_maps = []
    valid_indices = []
    for pos, (idx, row) in enumerate(df_labeled.iterrows()):
        wm = _process_wafer_map(row['waferMap'])
        if wm is not None:
            processed_maps.append(wm)
            valid_indices.append(idx)  # Keep the original index for .loc access
        if (pos + 1) % 10000 == 0:
            print(f"  Processed {pos + 1} samples ({len(valid_indices)} valid)...")
    
    # Create arrays
    print(f"Creating arrays from {len(valid_indices)} valid samples...")
    df_valid = df_labeled.loc[valid_indices].copy()  # Use .loc with index labels
    
    x_all = np.stack(processed_maps, axis=0).astype(np.uint8)
    y_all = df_valid['class_idx'].values.astype(np.int64)
    
    print(f"All data: x shape {x_all.shape}, y shape {y_all.shape}")
    print(f"Class distribution: {np.bincount(y_all.astype(int))}")
    
    # Use train_test column if available, else random split
    print("Splitting data into train/val/test...")
    rng = np.random.default_rng(rng_seed)
    n = len(x_all)
    
    # Try to use the train/test split from the data
    train_test_labels = df_valid['train_test'].values
    train_mask = np.array([label == 'Training' for label in train_test_labels])
    test_mask = np.array([label == 'Test' for label in train_test_labels])
    
    if train_mask.sum() > 0 and test_mask.sum() > 0:
        print(f"  Using provided train/test split: train={train_mask.sum()}, test={test_mask.sum()}")
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        # Further split training data into train/val
        val_n = max(1, int(len(train_indices) * val_split))
        val_indices_in_train = rng.choice(len(train_indices), size=val_n, replace=False)
        val_indices = train_indices[val_indices_in_train]
        train_indices = np.setdiff1d(train_indices, val_indices)
    else:
        # Random split
        print(f"  Using random split: val_split={val_split}, test_split={test_split}")
        indices = np.arange(n)
        rng.shuffle(indices)
        
        test_n = max(1, int(n * test_split))
        val_n = max(1, int((n - test_n) * val_split))
        
        test_indices = indices[:test_n]
        val_indices = indices[test_n:test_n+val_n]
        train_indices = indices[test_n+val_n:]
    
    # Create split datasets
    train_x, train_y = x_all[train_indices], y_all[train_indices]
    val_x, val_y = x_all[val_indices], y_all[val_indices]
    test_x, test_y = x_all[test_indices], y_all[test_indices]
    
    print(f"Train: {len(train_x)} samples")
    print(f"Val:   {len(val_x)} samples")
    print(f"Test:  {len(test_x)} samples")
    
    # Save as NPZ
    small_npz_path = output_dir / "small_arrays.npz"
    print(f"\nSaving to {small_npz_path}...")
    np.savez(
        small_npz_path,
        train_x=train_x, train_y=train_y,
        val_x=val_x, val_y=val_y,
        test_x=test_x, test_y=test_y,
    )
    print(f"Saved successfully!")
    
    return {
        "train_x": train_x, "train_y": train_y,
        "val_x": val_x, "val_y": val_y,
        "test_x": test_x, "test_y": test_y,
    }


# ---------------------------------------------------------------------------
# Fallback loader: build arrays from PNG folders (slower, used if no cache)
# ---------------------------------------------------------------------------
def build_split_arrays_from_folder(split_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Walks split_dir/<class>/*.png, applies to_canonical, returns (x, y)."""
    split_dir = Path(split_dir)
    xs, ys = [], []
    for cls_name, cls_idx in CLASS_TO_IDX.items():
        cls_dir = split_dir / cls_name
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"missing class dir: {cls_dir}")
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            rgb = np.array(Image.open(p).convert("RGB"))
            arr = rgb_to_class_array(rgb)
            arr = to_canonical(arr)
            xs.append(arr)
            ys.append(cls_idx)
    return np.stack(xs).astype(np.uint8), np.array(ys, dtype=np.int64)


# ---------------------------------------------------------------------------
# Dataset returning BOTH model inputs + label
# ---------------------------------------------------------------------------
class WaferDataset(Dataset):
    """
    Single dataset feeding both ViT and ResNet branches.

    Returns:
      vit_input:    (64, 64) long tensor, values in {0,1,2}
      resnet_input: (3, 64, 64) float tensor, values in {0.0, 0.5, 1.0}
      label:        int

    Augmentation (train only): horizontal flip, vertical flip, k*90° rotation.
    Each is applied once to the canonical array, then the two views are derived.
    NEAREST-equivalent (the ops preserve {0,1,2} integers).
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool = False, seed: int = 0):
        assert x.dtype == np.uint8, f"expected uint8, got {x.dtype}"
        assert x.shape[1:] == (CANON_SIZE, CANON_SIZE), \
            f"expected (N,{CANON_SIZE},{CANON_SIZE}), got {x.shape}"
        self.x = x
        self.y = y.astype(np.int64)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.x)

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        if self.rng.random() < 0.5:
            arr = np.fliplr(arr)
        if self.rng.random() < 0.5:
            arr = np.flipud(arr)
        k = int(self.rng.integers(0, 4))
        if k > 0:
            arr = np.rot90(arr, k=k)
        return np.ascontiguousarray(arr)

    def __getitem__(self, i):
        arr = self.x[i]
        if self.augment:
            arr = self._augment(arr)

        vit_input = torch.from_numpy(arr.astype(np.int64))             # (64,64) long, {0,1,2}

        # ResNet branch: scale {0,1,2} -> {0.0, 0.5, 1.0}, replicate to 3 channels
        rn_arr = arr.astype(np.float32) / 2.0                          # (64,64), {0.0, 0.5, 1.0}
        rn_arr = np.stack([rn_arr, rn_arr, rn_arr], axis=0)            # (3,64,64)
        resnet_input = torch.from_numpy(rn_arr)

        return vit_input, resnet_input, int(self.y[i])
