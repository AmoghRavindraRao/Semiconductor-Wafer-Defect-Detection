"""
extract_embeddings.py
After Stage 1, freeze the encoder and extract L2-normalized, TTA-averaged
embeddings for every split. Writes the PRD §9 embedding bank.

Outputs (all under embeddings/):
  train_embeddings.npy        (N_train, 256) float32, L2-normalized
  train_labels.npy            (N_train,)     int64
  val_embeddings.npy          val_labels.npy
  test_embeddings.npy         test_labels.npy
  pkl_labeled_embeddings.npy  pkl_labeled_labels.npy
  unlabeled_embeddings.npy    (no labels - still useful for diagnostics)
  metadata.csv                id, split, source, label_or_-1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import (
    WaferArrayDataset,
    build_pkl_labeled_dataset,
    build_pkl_unlabeled_dataset,
    build_small_datasets,
)
from model import SmallViT, ViTConfig
from utils import (
    PATHS,
    get_device,
    get_logger,
    tta_forward,
)


@torch.no_grad()
def extract(
    model: SmallViT,
    loader: DataLoader,
    device,
    temperature: float,
    use_tta: bool,
) -> np.ndarray:
    model.eval()
    chunks = []
    for batch in loader:
        x, _, _ = batch[:3]
        x = x.to(device, non_blocking=True)
        if use_tta:
            _, emb, _ = tta_forward(model, x, temperature=temperature)
        else:
            _, emb = model(x)
            emb = F.normalize(emb, dim=-1)
        chunks.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 256), dtype=np.float32)


def loader_for(ds: WaferArrayDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_npz", type=Path,
                        default=PATHS["cache"] / "small_arrays.npz")
    parser.add_argument("--pkl_npz", type=Path,
                        default=PATHS["cache"] / "pkl_arrays.npz")
    parser.add_argument("--ckpt", type=Path,
                        default=PATHS["ckpt"] / "vit_best.pth")
    parser.add_argument("--temp_path", type=Path,
                        default=PATHS["ckpt"] / "temperature.npy")
    parser.add_argument("--out_dir", type=Path, default=PATHS["emb"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_tta", action="store_true",
                        help="skip TTA averaging at extraction time")
    args = parser.parse_args()

    device = get_device()
    logger = get_logger("extract", PATHS["results"] / "extract_embeddings.log")

    # Temperature (only used if model was calibrated; it doesn't affect embeddings
    # but we pass it to tta_forward for symmetry with pseudo_label.py)
    T = float(np.load(args.temp_path)[0]) if args.temp_path.exists() else 1.0
    logger.info("using T=%.4f (used only for TTA logit averaging; embeddings unaffected)", T)

    state = torch.load(args.ckpt, map_location=device)
    cfg = ViTConfig(**state.get("cfg", {})) if isinstance(state.get("cfg"), dict) else ViTConfig()
    model = SmallViT(cfg).to(device)
    # Handle both checkpoint formats: "model_state" and "state_dict"
    state_dict_key = "model_state" if "model_state" in state else "state_dict"
    model.load_state_dict(state[state_dict_key])
    model.eval()
    logger.info("loaded %s", args.ckpt)

    train_ds, val_ds, test_ds = build_small_datasets(args.small_npz, augment_train=False)
    pkl_lab = build_pkl_labeled_dataset(args.pkl_npz)
    pkl_unl = build_pkl_unlabeled_dataset(args.pkl_npz)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    use_tta = not args.no_tta

    def _go(name: str, ds: WaferArrayDataset, labels_path: Path | None = None):
        logger.info("extracting %s: %d samples (tta=%s)", name, len(ds), use_tta)
        embs = extract(model, loader_for(ds, args.batch_size, args.num_workers),
                       device, T, use_tta)
        np.save(out / f"{name}_embeddings.npy", embs)
        if labels_path is not None:
            np.save(labels_path, ds.labels.astype(np.int64))
        logger.info("  -> %s  shape=%s", out / f"{name}_embeddings.npy", embs.shape)
        return embs

    _go("train", train_ds, out / "train_labels.npy")
    _go("val", val_ds, out / "val_labels.npy")
    _go("test", test_ds, out / "test_labels.npy")
    _go("pkl_labeled", pkl_lab, out / "pkl_labeled_labels.npy")
    _go("unlabeled", pkl_unl)

    # metadata.csv
    rows = []
    sid = 0
    for split, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        for i in range(len(ds)):
            rows.append((sid, split, "small", int(ds.labels[i])))
            sid += 1
    for i in range(len(pkl_lab)):
        rows.append((sid, "pkl_labeled", "pkl", int(pkl_lab.labels[i])))
        sid += 1
    for i in range(len(pkl_unl)):
        rows.append((sid, "unlabeled", "pkl", -1))
        sid += 1
    df = pd.DataFrame(rows, columns=["id", "split", "source", "label_or_-1"])
    df.to_csv(out / "metadata.csv", index=False)
    logger.info("wrote %s  (rows=%d)", out / "metadata.csv", len(df))


if __name__ == "__main__":
    main()
