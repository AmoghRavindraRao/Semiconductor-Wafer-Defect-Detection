"""
pseudo_label.py
PRD §10. Pseudo-label the 638,507 unlabeled wafers with 5-view TTA + Tier-1
acceptance rule + per-class cap.

Writes pseudo_labels/pseudo_labels.csv with one row per unlabeled wafer:
  wafer_row_in_pkl, clf_class, clf_conf, clf_entropy, tta_agree,
  centroid_class, cosine_class, cos_top1, cos_margin,
  knn_class, knn_agree, accepted, final_label
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import build_pkl_unlabeled_dataset, load_pkl_arrays
from model import SmallViT, ViTConfig
from tune_thresholds import compute_signals, make_knn_search
from utils import NUM_CLASSES, PATHS, get_device, get_logger


def apply_acceptance(s: dict, th: dict) -> np.ndarray:
    """Tier-1 acceptance rule from §10.3."""
    agree4 = (
        (s["clf_class"] == s["centroid_class"])
        & (s["clf_class"] == s["cosine_class"])
        & (s["clf_class"] == s["knn_class"])
    )
    mask = (
        agree4
        & s["tta_agree"]
        & (s["clf_conf"] > th["tau_clf"])
        & (s["clf_entropy"] < th["tau_entropy"])
        & (s["cos_top1"] > th["tau_cos"])
        & (s["cos_margin"] > th["tau_margin"])
        & (s["knn_agree"] >= th["tau_knn"])
    )
    return mask


def apply_per_class_cap(accepted: np.ndarray, labels: np.ndarray,
                        cap: int, rng: np.random.Generator) -> np.ndarray:
    """§10.5 - randomly drop excess samples per class."""
    out = accepted.copy()
    for c in range(NUM_CLASSES):
        idx_c = np.where(out & (labels == c))[0]
        if len(idx_c) > cap:
            drop = rng.choice(idx_c, size=len(idx_c) - cap, replace=False)
            out[drop] = False
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_npz", type=Path,
                        default=PATHS["cache"] / "pkl_arrays.npz")
    parser.add_argument("--emb_dir", type=Path, default=PATHS["emb"])
    parser.add_argument("--ckpt", type=Path,
                        default=PATHS["ckpt"] / "vit_best.pth")
    parser.add_argument("--temp_path", type=Path,
                        default=PATHS["ckpt"] / "temperature.npy")
    parser.add_argument("--thresholds_json", type=Path,
                        default=PATHS["results"] / "threshold_tuning" / "chosen_thresholds.json")
    parser.add_argument("--out_dir", type=Path, default=PATHS["pseudo"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_per_class", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    logger = get_logger("pseudo", PATHS["results"] / "pseudo_label.log")

    T = float(np.load(args.temp_path)[0]) if args.temp_path.exists() else 1.0
    logger.info("T=%.4f", T)

    # Thresholds
    with open(args.thresholds_json) as f:
        th = json.load(f)
    logger.info("thresholds: %s", th)

    # Model
    state = torch.load(args.ckpt, map_location=device)
    cfg = ViTConfig(**state.get("cfg", {})) if isinstance(state.get("cfg"), dict) else ViTConfig()
    model = SmallViT(cfg).to(device)
    # Handle both checkpoint formats: "model_state" and "state_dict"
    state_dict_key = "model_state" if "model_state" in state else "state_dict"
    model.load_state_dict(state[state_dict_key])

    # Data
    pkl_unl = build_pkl_unlabeled_dataset(args.pkl_npz)
    loader = DataLoader(pkl_unl, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    logger.info("unlabeled: %d samples", len(pkl_unl))

    # kNN + centroids
    centroids = torch.from_numpy(np.load(args.emb_dir / "centroids.npy")).to(device)
    knn_search, knn_labels = make_knn_search(args.emb_dir, logger)

    # Compute signals (y is meaningless here; it will just be -1, which we ignore)
    logger.info("computing TTA signals on %d samples ...", len(pkl_unl))
    signals = compute_signals(model, loader, device, T, centroids,
                              knn_search, knn_labels, k=5)
    logger.info("signals computed.")

    # Acceptance
    accepted = apply_acceptance(signals, th)
    n_pre_cap = int(accepted.sum())
    logger.info("accepted pre-cap: %d (%.2f%%)", n_pre_cap,
                100.0 * n_pre_cap / max(1, len(accepted)))

    rng = np.random.default_rng(args.seed)
    accepted = apply_per_class_cap(accepted, signals["clf_class"],
                                   args.max_per_class, rng)
    logger.info("accepted post-cap: %d  (cap=%d)",
                int(accepted.sum()), args.max_per_class)

    # Per-class breakdown
    for c in range(NUM_CLASSES):
        n_all = int((signals["clf_class"] == c).sum())
        n_acc = int(((signals["clf_class"] == c) & accepted).sum())
        logger.info("  class %d: candidate=%d accepted=%d", c, n_all, n_acc)

    # Bring the original pkl row indices so downstream code can map back
    pkl_bundle = load_pkl_arrays(args.pkl_npz)
    row_ids = pkl_bundle["unlabeled_id"]
    assert len(row_ids) == len(signals["clf_class"]), "length mismatch"

    # Write CSV
    df = pd.DataFrame({
        "pkl_row": row_ids,
        "unl_idx": np.arange(len(row_ids), dtype=np.int64),
        "clf_class": signals["clf_class"],
        "clf_conf": signals["clf_conf"],
        "clf_entropy": signals["clf_entropy"],
        "tta_agree": signals["tta_agree"],
        "centroid_class": signals["centroid_class"],
        "cosine_class": signals["cosine_class"],
        "cos_top1": signals["cos_top1"],
        "cos_margin": signals["cos_margin"],
        "knn_class": signals["knn_class"],
        "knn_agree": signals["knn_agree"],
        "accepted": accepted,
        "final_label": np.where(accepted, signals["clf_class"], -1).astype(np.int64),
        "tier": np.where(accepted, 1, 0).astype(np.int64),
    })
    out_csv = args.out_dir / "pseudo_labels.csv"
    df.to_csv(out_csv, index=False)
    logger.info("wrote %s (rows=%d)", out_csv, len(df))


if __name__ == "__main__":
    main()
