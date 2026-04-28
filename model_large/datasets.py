"""
datasets.py
One dataset class, WaferArrayDataset, that yields canonical (64, 64) uint8
tensors with values in {0, 1, 2} regardless of source (small-folder PNG or
LSWMD.pkl). This is the mechanism that enforces the PRD §4 constraint.

Loaders that read from cached .npz arrays are the fast path. The dataset
can also be constructed directly from (arrays, labels) in memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import CANON_SIZE, CLASS_TO_IDX


# ---------------------------------------------------------------------------
# Train-time augmentation (PRD §8.1) - discrete only, preserves {0,1,2}
# ---------------------------------------------------------------------------
def train_augment(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        arr = np.ascontiguousarray(arr[:, ::-1])
    if rng.random() < 0.5:
        arr = np.ascontiguousarray(arr[::-1, :])
    k = int(rng.integers(0, 4))  # 0, 90, 180, 270
    if k:
        arr = np.ascontiguousarray(np.rot90(arr, k))
    return arr


# ---------------------------------------------------------------------------
# Unified dataset
# ---------------------------------------------------------------------------
class WaferArrayDataset(Dataset):
    """
    Arguments:
      arrays : (N, 64, 64) uint8 with values in {0, 1, 2}
      labels : (N,) int64   (use -1 for unlabeled)
      augment: True for training split, False otherwise
      return_source: if True, also return an int source-id (0=small, 1=pkl)
      source : optional (N,) int array to report alongside samples
    """

    def __init__(
        self,
        arrays: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        return_source: bool = False,
        source: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        seed: int = 0,
    ):
        assert arrays.ndim == 3 and arrays.shape[1:] == (CANON_SIZE, CANON_SIZE), (
            f"arrays must be (N, {CANON_SIZE}, {CANON_SIZE}); got {arrays.shape}"
        )
        assert arrays.dtype == np.uint8, f"arrays must be uint8; got {arrays.dtype}"
        unique = np.unique(arrays[:min(64, len(arrays))])
        assert set(unique.tolist()).issubset({0, 1, 2}), (
            f"arrays must contain only {{0,1,2}}; got {unique.tolist()}"
        )
        assert len(arrays) == len(labels), "arrays/labels length mismatch"

        self.arrays = arrays
        self.labels = labels.astype(np.int64)
        self.augment = augment
        self.return_source = return_source
        self.source = source if source is not None else np.zeros(len(arrays), dtype=np.int64)
        self.weights = (
            weights.astype(np.float32)
            if weights is not None
            else np.ones(len(arrays), dtype=np.float32)
        )
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.arrays)

    def __getitem__(self, idx: int):
        arr = self.arrays[idx]
        if self.augment:
            arr = train_augment(arr, self._rng)
        # long tensor so the model can use nn.Embedding(3, d) on pixel indices
        x = torch.from_numpy(arr.copy()).long()
        y = int(self.labels[idx])
        w = float(self.weights[idx])
        if self.return_source:
            return x, y, w, int(self.source[idx])
        return x, y, w


# ---------------------------------------------------------------------------
# NPZ loaders produced by convert_datasets.py
# ---------------------------------------------------------------------------
def load_small_split(npz_path: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load canonical small-dataset arrays for split in {'train','val','test'}."""
    z = np.load(npz_path)
    return z[f"{split}_x"], z[f"{split}_y"]


def load_pkl_arrays(npz_path: str | Path) -> dict[str, np.ndarray]:
    """
    Returns dict with:
      labeled_x   (N_l, 64, 64) uint8
      labeled_y   (N_l,) int64
      unlabeled_x (N_u, 64, 64) uint8
      unlabeled_id (N_u,) int64   - row index in original pkl (for provenance)
      labeled_id   (N_l,) int64   - row index in original pkl
    """
    z = np.load(npz_path)
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------
def build_small_datasets(
    npz_path: str | Path,
    augment_train: bool = True,
) -> tuple[WaferArrayDataset, WaferArrayDataset, WaferArrayDataset]:
    tx, ty = load_small_split(npz_path, "train")
    vx, vy = load_small_split(npz_path, "val")
    sx, sy = load_small_split(npz_path, "test")
    return (
        WaferArrayDataset(tx, ty, augment=augment_train, seed=0),
        WaferArrayDataset(vx, vy, augment=False, seed=1),
        WaferArrayDataset(sx, sy, augment=False, seed=2),
    )


def build_pkl_labeled_dataset(pkl_npz_path: str | Path) -> WaferArrayDataset:
    pkl = load_pkl_arrays(pkl_npz_path)
    return WaferArrayDataset(pkl["labeled_x"], pkl["labeled_y"], augment=False)


def build_pkl_unlabeled_dataset(pkl_npz_path: str | Path) -> WaferArrayDataset:
    pkl = load_pkl_arrays(pkl_npz_path)
    dummy = -np.ones(len(pkl["unlabeled_x"]), dtype=np.int64)
    return WaferArrayDataset(pkl["unlabeled_x"], dummy, augment=False)


# Sanity: CLASS_TO_IDX is imported so accidental re-ordering is caught at import.
assert list(CLASS_TO_IDX.values()) == list(range(9)), "CLASS_TO_IDX must be 0..8"
