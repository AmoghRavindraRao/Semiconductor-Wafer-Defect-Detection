"""
train_both.py

Trains BOTH the SmallViT and a ResNet (50 or 18) on identical data, splits,
preprocessing, optimizer, and schedule. Loss = CE(label_smoothing=0.1) + 0.5 * SupCon.

Stage 1 recipe (PRD §8.2):
  optimizer:  AdamW, lr=3e-4, wd=0.05
  schedule:   5-epoch linear warmup -> cosine decay
  batch:      128
  epochs:     50, early stop on val macro F1, patience 10
  checkpoint: max val macro F1

Optional pkl mix-in: --pkl_mix_per_class 500 (PRD §3.1 Option A relaxed).

Usage:
    python train_both.py \
        --small_npz   data_cache/small_arrays.npz \
        --pkl_npz     data_cache/pkl_arrays.npz \
        --pkl_mix_per_class 500 \
        --resnet_backbone resnet50 \
        --epochs 50

If --resnet_backbone resnet50 underperforms or fails, rerun with resnet18.
"""
import argparse
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from data_utils import (
    WaferDataset, NUM_CLASSES, IDX_TO_CLASS,
    load_small_arrays, load_pkl_arrays, pkl_mix_in,
)
from models import build_model
from losses import supcon_loss

logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO,
)
log = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Optimizer / schedule
# ---------------------------------------------------------------------------
def make_optimizer(model: nn.Module, lr: float, wd: float):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, model_name,
                    supcon_weight: float = 0.5, supcon_temp: float = 0.1):
    model.train()
    total_loss, total_n, total_correct = 0.0, 0, 0
    for vit_x, rn_x, y in loader:
        y = y.to(device, non_blocking=True)
        x = vit_x.to(device, non_blocking=True) if model_name == "vit" \
            else rn_x.to(device, non_blocking=True)

        logits, _emb, proj = model(x, return_projections=True)
        L_ce = F.cross_entropy(logits, y, label_smoothing=0.1)
        L_sc = supcon_loss(proj, y, temperature=supcon_temp)
        loss = L_ce + supcon_weight * L_sc

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        bs = y.size(0)
        total_loss    += loss.item() * bs
        total_n       += bs
        total_correct += (logits.argmax(-1) == y).sum().item()

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, device, model_name):
    model.eval()
    all_pred, all_y = [], []
    for vit_x, rn_x, y in loader:
        x = vit_x.to(device, non_blocking=True) if model_name == "vit" \
            else rn_x.to(device, non_blocking=True)
        logits, _ = model(x)
        all_pred.append(logits.argmax(-1).cpu())
        all_y.append(y)
    pred = torch.cat(all_pred).numpy()
    yy   = torch.cat(all_y).numpy()
    return {
        "macro_f1": f1_score(yy, pred, average="macro", zero_division=0),
        "acc":      (pred == yy).mean(),
    }


# ---------------------------------------------------------------------------
# Per-model training driver
# ---------------------------------------------------------------------------
def train_model(model_name: str, train_ds, val_ds, args, device, out_path):
    log.info("=" * 70)
    log.info(f"training {model_name}")
    log.info("=" * 70)

    model = build_model(model_name, num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  params: {n_params/1e6:.2f}M")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = steps_per_epoch * 5

    optimizer = make_optimizer(model, lr=args.lr, wd=args.wd)
    scheduler = make_scheduler(optimizer, total_steps, warmup_steps)

    best_f1, best_epoch, patience = -1.0, -1, 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler,
                                          device, model_name)
        val_metrics = evaluate(model, val_loader, device, model_name)
        log.info(f"  epoch {epoch:3d}  loss={tr_loss:.4f} acc={tr_acc:.4f}  "
                 f"val_macroF1={val_metrics['macro_f1']:.4f} val_acc={val_metrics['acc']:.4f}")

        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch, patience = val_metrics["macro_f1"], epoch, 0
            torch.save({"state_dict": model.state_dict(),
                        "model_name": model_name,
                        "epoch": epoch,
                        "val_macro_f1": best_f1},
                       out_path)
            log.info(f"    *** new best -> {out_path} (val_macroF1={best_f1:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                log.info(f"  early stop at epoch {epoch} (best={best_epoch}, F1={best_f1:.4f})")
                break

    log.info(f"DONE {model_name}: best val macroF1={best_f1:.4f} @ epoch {best_epoch}")
    return best_f1, best_epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_npz", type=Path, default=Path("data_cache/small_arrays.npz"))
    parser.add_argument("--pkl_npz",   type=Path, default=Path("data_cache/pkl_arrays.npz"))
    parser.add_argument("--lswmd_pkl", type=Path, default=None,
                        help="Path to LSWMD.pkl for auto-creating cache if small_npz is missing")
    parser.add_argument("--pkl_mix_per_class", type=int, default=500)
    parser.add_argument("--resnet_backbone", choices=["resnet50", "resnet18"], default="resnet50")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=128)
    parser.add_argument("--lr",     type=float, default=3e-4)
    parser.add_argument("--wd",     type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--ckpt_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--skip_vit", action="store_true")
    parser.add_argument("--skip_resnet", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device: {device}")
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    log.info(f"loading small arrays: {args.small_npz}")
    small = load_small_arrays(str(args.small_npz), lswmd_pkl_path=str(args.lswmd_pkl) if args.lswmd_pkl else None)
    log.info(f"  train={len(small['train_x'])}  val={len(small['val_x'])}  test={len(small['test_x'])}")

    train_x, train_y = small["train_x"], small["train_y"]
    if args.pkl_mix_per_class > 0:
        log.info(f"loading pkl arrays for mix-in: {args.pkl_npz}")
        pkl = load_pkl_arrays(str(args.pkl_npz))
        mix_x, mix_y = pkl_mix_in(pkl, args.pkl_mix_per_class, rng_seed=args.seed)
        log.info(f"  mixing in {len(mix_x)} pkl-labeled samples")
        for c in range(NUM_CLASSES):
            log.info(f"    class {c} ({IDX_TO_CLASS[c]:9s}): "
                     f"small={int((train_y == c).sum())} pkl_mix={int((mix_y == c).sum())}")
        train_x = np.concatenate([train_x, mix_x], axis=0)
        train_y = np.concatenate([train_y, mix_y], axis=0)
        log.info(f"  combined train size: {len(train_x)}")

    train_ds = WaferDataset(train_x, train_y, augment=True, seed=args.seed)
    val_ds   = WaferDataset(small["val_x"], small["val_y"], augment=False)

    # ---- Train both ----
    if not args.skip_vit:
        train_model("vit", train_ds, val_ds, args, device,
                    args.ckpt_dir / "vit_best.pth")

    if not args.skip_resnet:
        train_model(args.resnet_backbone, train_ds, val_ds, args, device,
                    args.ckpt_dir / "resnet_best.pth")

    log.info("all done.")


if __name__ == "__main__":
    main()
