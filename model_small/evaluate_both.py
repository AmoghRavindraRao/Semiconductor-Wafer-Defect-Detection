"""
evaluate_both.py

Loads vit_best.pth and resnet_best.pth, fits temperature scaling on val,
sweeps ensemble weight, reports test macro F1 for ViT, ResNet, and ensemble.

Saves:
  checkpoints/temperature_vit.npy
  checkpoints/temperature_resnet.npy
  checkpoints/ensemble_weight.npy
  results/from_scratch/comparison.md
  results/from_scratch/predictions.csv

Usage:
    python evaluate_both.py
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from data_utils import (
    WaferDataset, NUM_CLASSES, IDX_TO_CLASS,
    load_small_arrays,
)
from models import build_model

logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO,
)
log = logging.getLogger("eval")


# ---------------------------------------------------------------------------
# Logits collection
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_logits(model, loader, model_name, device):
    model.eval()
    all_logits, all_y = [], []
    for vit_x, rn_x, y in loader:
        x = vit_x.to(device, non_blocking=True) if model_name == "vit" \
            else rn_x.to(device, non_blocking=True)
        logits, _ = model(x)
        all_logits.append(logits.cpu())
        all_y.append(y)
    return torch.cat(all_logits), torch.cat(all_y)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
def fit_temperature(logits: torch.Tensor, y: torch.Tensor, device: str) -> float:
    logits, y = logits.to(device), y.to(device)
    T = nn.Parameter(torch.ones(1, device=device))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T, y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().cpu().item())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def per_class_f1(y, pred):
    f1s = f1_score(y, pred, labels=list(range(NUM_CLASSES)),
                   average=None, zero_division=0)
    return {IDX_TO_CLASS[i]: float(f1s[i]) for i in range(NUM_CLASSES)}


def write_report(test_results, best_w, val_table, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    classes = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]

    L = []
    L.append("# From-Scratch Ensemble: ViT + ResNet (CE + SupCon)\n")
    L.append(f"Best ResNet weight: **w = {best_w:.2f}**  ")
    L.append(f"(ViT weight = 1 - w = {1 - best_w:.2f})\n")

    L.append("## Overall (test set)\n")
    L.append("| Model | macro F1 | accuracy |")
    L.append("|---|---:|---:|")
    for tag in ("vit", "resnet", "ensemble"):
        r = test_results[tag]
        L.append(f"| {tag} | {r['macro_f1']:.4f} | {r['acc']:.4f} |")
    L.append("")

    L.append("## Per-class F1 (test)\n")
    L.append("| class | ViT | ResNet | Ensemble | Δ(ens − best_solo) |")
    L.append("|---|---:|---:|---:|---:|")
    for c in classes:
        v = test_results["vit"]["per_class_f1"][c]
        r = test_results["resnet"]["per_class_f1"][c]
        e = test_results["ensemble"]["per_class_f1"][c]
        d = e - max(v, r)
        flag = " ✓" if d >= 0 else ""
        L.append(f"| {c} | {v:.4f} | {r:.4f} | {e:.4f} | {d:+.4f}{flag} |")
    L.append("")

    L.append("## Val sweep\n")
    L.append("| w (ResNet) | val macro F1 | val acc |")
    L.append("|---:|---:|---:|")
    for w, f1m, acc in val_table:
        marker = " ←" if abs(w - best_w) < 1e-9 else ""
        L.append(f"| {w:.2f} | {f1m:.4f} | {acc:.4f} |{marker}")
    L.append("")

    L.append("## Confusion matrix — ensemble (test)\n")
    cm = confusion_matrix(test_results["y"], test_results["ensemble"]["pred"],
                          labels=list(range(NUM_CLASSES)))
    L.append("| true ＼ pred | " + " | ".join(classes) + " |")
    L.append("|---|" + "|".join(["---:"] * NUM_CLASSES) + "|")
    for i, c in enumerate(classes):
        row = " | ".join(str(int(cm[i, j])) for j in range(NUM_CLASSES))
        L.append(f"| **{c}** | {row} |")
    L.append("")

    out_path.write_text("\n".join(L))
    log.info(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_npz", type=Path, default=Path("data_cache/small_arrays.npz"))
    parser.add_argument("--lswmd_pkl", type=Path, default=None,
                        help="Path to LSWMD.pkl for auto-creating cache if small_npz is missing")
    parser.add_argument("--ckpt_dir",  type=Path, default=Path("checkpoints"))
    parser.add_argument("--out_dir",   type=Path, default=Path("results/from_scratch"))
    parser.add_argument("--batch",     type=int, default=128)
    parser.add_argument("--workers",   type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device: {device}")

    # ---- Data ----
    small = load_small_arrays(str(args.small_npz), lswmd_pkl_path=str(args.lswmd_pkl) if args.lswmd_pkl else None)
    val_ds  = WaferDataset(small["val_x"],  small["val_y"],  augment=False)
    test_ds = WaferDataset(small["test_x"], small["test_y"], augment=False)
    val_loader  = DataLoader(val_ds,  batch_size=args.batch, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    log.info(f"val={len(val_ds)} test={len(test_ds)}")

    # ---- Load checkpoints ----
    vit_ckpt = torch.load(args.ckpt_dir / "vit_best.pth", map_location=device)
    rn_ckpt  = torch.load(args.ckpt_dir / "resnet_best.pth", map_location=device)
    rn_name  = rn_ckpt.get("model_name", "resnet50")
    log.info(f"  ViT epoch {vit_ckpt.get('epoch')}  val F1={vit_ckpt.get('val_macro_f1'):.4f}")
    log.info(f"  {rn_name} epoch {rn_ckpt.get('epoch')}  val F1={rn_ckpt.get('val_macro_f1'):.4f}")

    vit = build_model("vit", num_classes=NUM_CLASSES).to(device)
    vit.load_state_dict(vit_ckpt["state_dict"])

    resnet = build_model(rn_name, num_classes=NUM_CLASSES).to(device)
    resnet.load_state_dict(rn_ckpt["state_dict"])

    # ---- Calibrate on val ----
    log.info("collecting val logits...")
    vit_val_logits, y_val = collect_logits(vit, val_loader, "vit", device)
    rn_val_logits,  _     = collect_logits(resnet, val_loader, rn_name, device)

    T_vit = fit_temperature(vit_val_logits, y_val, device)
    T_rn  = fit_temperature(rn_val_logits,  y_val, device)
    log.info(f"  T_vit    = {T_vit:.4f}")
    log.info(f"  T_resnet = {T_rn:.4f}")
    np.save(args.ckpt_dir / "temperature_vit.npy",    np.array([T_vit], dtype=np.float32))
    np.save(args.ckpt_dir / "temperature_resnet.npy", np.array([T_rn],  dtype=np.float32))

    # ---- Sweep w on val ----
    pv_val = F.softmax(vit_val_logits / T_vit, dim=-1).numpy()
    pr_val = F.softmax(rn_val_logits  / T_rn,  dim=-1).numpy()
    yv     = y_val.numpy()

    log.info("=== val sweep ===")
    val_table = []
    for w in np.linspace(0.0, 1.0, 11):
        probs = w * pr_val + (1.0 - w) * pv_val
        pred = probs.argmax(axis=1)
        f1m = float(f1_score(yv, pred, average="macro", zero_division=0))
        acc = float((pred == yv).mean())
        val_table.append((float(w), f1m, acc))
        log.info(f"  w={w:.2f}  F1={f1m:.4f} acc={acc:.4f}")
    best_w = max(val_table, key=lambda r: r[1])[0]
    log.info(f"BEST w = {best_w:.2f}")
    np.save(args.ckpt_dir / "ensemble_weight.npy", np.array([best_w], dtype=np.float32))

    # ---- Test eval ----
    log.info("collecting test logits...")
    vit_te, yt = collect_logits(vit, test_loader, "vit", device)
    rn_te, _   = collect_logits(resnet, test_loader, rn_name, device)
    pv = F.softmax(vit_te / T_vit, dim=-1).numpy()
    pr = F.softmax(rn_te  / T_rn,  dim=-1).numpy()
    pe = best_w * pr + (1.0 - best_w) * pv
    yt = yt.numpy()

    test_results = {"y": yt}
    for tag, probs in [("vit", pv), ("resnet", pr), ("ensemble", pe)]:
        pred = probs.argmax(axis=1)
        test_results[tag] = {
            "macro_f1": float(f1_score(yt, pred, average="macro", zero_division=0)),
            "acc": float(accuracy_score(yt, pred)),
            "per_class_f1": per_class_f1(yt, pred),
            "pred": pred,
            "probs": probs,
        }

    log.info("=== test results ===")
    for tag in ("vit", "resnet", "ensemble"):
        r = test_results[tag]
        log.info(f"  {tag:9s}: macroF1={r['macro_f1']:.4f}  acc={r['acc']:.4f}")

    best_solo = max(test_results["vit"]["macro_f1"], test_results["resnet"]["macro_f1"])
    lift = test_results["ensemble"]["macro_f1"] - best_solo
    log.info("=" * 60)
    log.info(f"best solo: {best_solo:.4f}")
    log.info(f"ensemble:  {test_results['ensemble']['macro_f1']:.4f}")
    log.info(f"lift:      {lift:+.4f}")
    if lift < 0.005:
        log.warning("ensemble lift < 0.005 — consider shipping best solo model")
    log.info("=" * 60)

    # ---- Reports ----
    write_report(test_results, best_w, val_table, args.out_dir / "comparison.md")

    pred_csv = args.out_dir / "predictions.csv"
    pred_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "y_true":        yt,
        "y_true_name":   [IDX_TO_CLASS[i] for i in yt],
        "pred_vit":      test_results["vit"]["pred"],
        "pred_resnet":   test_results["resnet"]["pred"],
        "pred_ensemble": test_results["ensemble"]["pred"],
    })
    for i in range(NUM_CLASSES):
        df[f"prob_ens_{IDX_TO_CLASS[i]}"] = test_results["ensemble"]["probs"][:, i]
    df.to_csv(pred_csv, index=False)
    log.info(f"wrote {pred_csv}")


if __name__ == "__main__":
    main()
