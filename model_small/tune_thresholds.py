"""
tune_thresholds.py
PRD §11. Use the small val set as fake-unlabeled. For each of the 3x3x3x3x3 = 243
threshold combinations, compute precision and coverage against the true labels
(which the pseudo-labeler never sees). Select the setting with max coverage
subject to precision >= 0.95.

Signals used here are pre-computed once:
  - calibrated softmax probs from TTA-averaged logits
  - TTA-averaged L2-normalized embeddings
  - cosine similarities to centroids
  - k=5 cosine kNN against the FAISS index (but excluding the val set itself,
    which is fine since val was never added to the index)
  - TTA agreement flag

Output:
  results/threshold_tuning/sweep_results.csv
  results/threshold_tuning/precision_vs_coverage.png
  results/threshold_tuning/chosen_thresholds.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import build_small_datasets
from model import SmallViT, ViTConfig
from utils import NUM_CLASSES, PATHS, get_device, get_logger, tta_forward


# Sweep grid (PRD §11)
TAU_CLF = [0.85, 0.90, 0.95]
TAU_ENTROPY = [0.30, 0.40, 0.60]
TAU_COS = [0.70, 0.80, 0.90]
TAU_MARGIN = [0.10, 0.15, 0.20]
TAU_KNN = [0.6, 0.8, 1.0]


# ---------------------------------------------------------------------------
# Pre-compute signals on a loader
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_signals(
    model: SmallViT,
    loader: DataLoader,
    device,
    temperature: float,
    centroids: torch.Tensor,           # (C, D) on device, L2-normalized
    knn_search,                         # callable: (N,D)np -> (sims, idx) both (N,k)
    knn_labels: np.ndarray,             # (N_index,) int64, aligned with knn_search
    k: int = 5,
):
    model.eval()
    all_probs, all_emb, all_agree, all_y = [], [], [], []
    for batch in loader:
        x, y, _ = batch[:3]
        x = x.to(device, non_blocking=True)
        mean_logits, mean_emb, agree = tta_forward(model, x, temperature=temperature)
        probs = F.softmax(mean_logits, dim=-1)
        all_probs.append(probs.cpu().numpy().astype(np.float32))
        all_emb.append(mean_emb.cpu().numpy().astype(np.float32))
        all_agree.append(agree.cpu().numpy().astype(bool))
        all_y.append(y.numpy().astype(np.int64))

    probs = np.concatenate(all_probs, axis=0)
    emb = np.concatenate(all_emb, axis=0)
    agree = np.concatenate(all_agree, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Classifier
    clf_class = probs.argmax(axis=1)
    clf_conf = probs.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        clf_entropy = -np.nansum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)

    # Centroid: argmin Euclidean distance (equivalent to argmax cosine on normalized)
    C = centroids.detach().cpu().numpy().astype(np.float32)  # (num_classes, D)
    cos = emb @ C.T                                          # (N, C)  cosine (norm embeddings/centroids)
    # Euclidean distance on unit vectors: d^2 = 2 - 2*cos  ->  argmin(d) == argmax(cos)
    centroid_class = cos.argmax(axis=1).astype(np.int64)

    # Cosine signal (same matrix, report top-1 and margin)
    sorted_cos = np.sort(cos, axis=1)[:, ::-1]
    cosine_class = cos.argmax(axis=1).astype(np.int64)
    cos_top1 = sorted_cos[:, 0]
    cos_margin = sorted_cos[:, 0] - sorted_cos[:, 1]

    # kNN (cosine via inner product on normalized)
    sims, idx = knn_search(emb, k)       # (N,k) each
    knn_lbl = knn_labels[idx]            # (N,k)
    # majority vote
    knn_class = np.zeros(len(emb), dtype=np.int64)
    knn_agree = np.zeros(len(emb), dtype=np.float32)
    for i in range(len(emb)):
        bc = np.bincount(knn_lbl[i], minlength=NUM_CLASSES)
        knn_class[i] = int(bc.argmax())
        knn_agree[i] = float(bc[knn_class[i]]) / k

    return {
        "y": y,
        "clf_class": clf_class.astype(np.int64),
        "clf_conf": clf_conf,
        "clf_entropy": clf_entropy,
        "tta_agree": agree,
        "centroid_class": centroid_class,
        "cosine_class": cosine_class,
        "cos_top1": cos_top1,
        "cos_margin": cos_margin,
        "knn_class": knn_class,
        "knn_agree": knn_agree,
        "emb": emb,
        "probs": probs,
    }


# ---------------------------------------------------------------------------
# Apply a threshold combo -> (accepted_mask, predicted_labels)
# ---------------------------------------------------------------------------
def apply_thresholds(s: dict, tc: float, te: float, tcos: float, tm: float, tk: float):
    agree4 = (
        (s["clf_class"] == s["centroid_class"])
        & (s["clf_class"] == s["cosine_class"])
        & (s["clf_class"] == s["knn_class"])
    )
    accepted = (
        agree4
        & s["tta_agree"]
        & (s["clf_conf"] > tc)
        & (s["clf_entropy"] < te)
        & (s["cos_top1"] > tcos)
        & (s["cos_margin"] > tm)
        & (s["knn_agree"] >= tk)
    )
    return accepted, s["clf_class"]


# ---------------------------------------------------------------------------
# kNN search wrapper (faiss if available, else numpy brute force)
# ---------------------------------------------------------------------------
def make_knn_search(emb_dir: Path, logger):
    labels = np.load(emb_dir / "faiss_labels.npy")
    try:
        import faiss
        index_path = emb_dir / "faiss_index.bin"
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        index = faiss.read_index(str(index_path))
        logger.info("loaded faiss index  ntotal=%d", index.ntotal)

        def _search(q: np.ndarray, k: int):
            # q expected L2-normalized
            q = q.astype(np.float32)
            return index.search(q, k)

        return _search, labels
    except Exception as e:
        logger.warning("faiss unavailable (%s); using brute-force kNN", e)
        base = np.load(emb_dir / "faiss_labeled_norm.npy").astype(np.float32)

        def _search(q: np.ndarray, k: int):
            q = q.astype(np.float32)
            sims = q @ base.T                                  # (N, M)
            idx = np.argpartition(-sims, kth=k, axis=1)[:, :k]
            # sort top-k per row
            row = np.arange(len(q))[:, None]
            top_sims = sims[row, idx]
            order = np.argsort(-top_sims, axis=1)
            idx = idx[row, order]
            top_sims = top_sims[row, order]
            return top_sims, idx

        return _search, labels


# ---------------------------------------------------------------------------
# Plot precision vs coverage
# ---------------------------------------------------------------------------
def plot_pc(rows: list[dict], out_path: Path):
    try:
        import matplotlib.pyplot as plt
        prec = [r["precision"] for r in rows]
        cov = [r["coverage"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(cov, prec, s=12, alpha=0.6)
        ax.axhline(0.95, color="red", linestyle="--", label="precision=0.95 target")
        ax.set_xlabel("coverage (accepted / total)")
        ax.set_ylabel("precision (correct / accepted)")
        ax.set_title(f"Precision vs Coverage across {len(rows)} threshold combos")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"(plot skipped: {e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_npz", type=Path,
                        default=PATHS["cache"] / "small_arrays.npz")
    parser.add_argument("--emb_dir", type=Path, default=PATHS["emb"])
    parser.add_argument("--ckpt", type=Path,
                        default=PATHS["ckpt"] / "vit_best.pth")
    parser.add_argument("--temp_path", type=Path,
                        default=PATHS["ckpt"] / "temperature.npy")
    parser.add_argument("--out_dir", type=Path,
                        default=PATHS["results"] / "threshold_tuning")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--precision_target", type=float, default=0.95)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    logger = get_logger("tune", PATHS["results"] / "tune_thresholds.log")

    T = float(np.load(args.temp_path)[0]) if args.temp_path.exists() else 1.0
    logger.info("using T=%.4f", T)

    # Model
    state = torch.load(args.ckpt, map_location=device)
    cfg = ViTConfig(**state.get("cfg", {})) if isinstance(state.get("cfg"), dict) else ViTConfig()
    model = SmallViT(cfg).to(device)
    # Handle both checkpoint formats: "model_state" and "state_dict"
    state_dict_key = "model_state" if "model_state" in state else "state_dict"
    model.load_state_dict(state[state_dict_key])

    # Val loader (fake-unlabeled for the pseudo-labeler, true labels revealed only for scoring)
    _, val_ds, _ = build_small_datasets(args.small_npz, augment_train=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Centroids + kNN
    centroids = torch.from_numpy(np.load(args.emb_dir / "centroids.npy")).to(device)
    knn_search, knn_labels = make_knn_search(args.emb_dir, logger)

    logger.info("computing signals for %d val samples...", len(val_ds))
    signals = compute_signals(model, val_loader, device, T, centroids,
                              knn_search, knn_labels, k=5)
    logger.info("signals computed.")

    # Sweep
    rows = []
    for tc, te, tcos, tm, tk in itertools.product(
        TAU_CLF, TAU_ENTROPY, TAU_COS, TAU_MARGIN, TAU_KNN
    ):
        accepted, preds = apply_thresholds(signals, tc, te, tcos, tm, tk)
        n_acc = int(accepted.sum())
        coverage = n_acc / len(accepted) if len(accepted) else 0.0
        precision = (float((preds[accepted] == signals["y"][accepted]).mean())
                     if n_acc > 0 else 0.0)
        rows.append(dict(
            tau_clf=tc, tau_entropy=te, tau_cos=tcos, tau_margin=tm, tau_knn=tk,
            accepted=n_acc, coverage=coverage, precision=precision,
        ))

    # CSV
    try:
        import pandas as pd
        df = pd.DataFrame(rows).sort_values(["precision", "coverage"], ascending=[False, False])
        df.to_csv(args.out_dir / "sweep_results.csv", index=False)
        logger.info("wrote %s", args.out_dir / "sweep_results.csv")
    except ImportError:
        pass

    # Plot
    plot_pc(rows, args.out_dir / "precision_vs_coverage.png")

    # Chosen: max coverage s.t. precision >= target
    feasible = [r for r in rows if r["precision"] >= args.precision_target]
    if not feasible:
        logger.warning(
            "no combo meets precision >= %.2f; picking the highest-precision combo",
            args.precision_target,
        )
        chosen = max(rows, key=lambda r: r["precision"])
    else:
        chosen = max(feasible, key=lambda r: r["coverage"])

    logger.info("CHOSEN thresholds: %s", chosen)
    with open(args.out_dir / "chosen_thresholds.json", "w") as f:
        json.dump(chosen, f, indent=2)
    logger.info("wrote %s", args.out_dir / "chosen_thresholds.json")


if __name__ == "__main__":
    main()
