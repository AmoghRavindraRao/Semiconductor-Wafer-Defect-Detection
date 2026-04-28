"""
calibrate.py
Stage 1b - temperature scaling (PRD §8.3).

Fit a single scalar T on the val set by minimizing NLL of softmax(logits / T)
via LBFGS. Save T to checkpoints/temperature.npy. Downstream code (pseudo-
labeling, threshold tuning) MUST divide logits by this T.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import build_small_datasets
from model import SmallViT, ViTConfig
from utils import PATHS, get_device, get_logger


@torch.no_grad()
def collect_val_logits(model: SmallViT, loader: DataLoader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        x, y, _ = batch[:3]
        x = x.to(device, non_blocking=True)
        logits, _ = model(x)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())
    return torch.cat(all_logits), torch.cat(all_y).long()


def fit_temperature(logits: torch.Tensor, y: torch.Tensor,
                    device, max_iter: int = 50, lr: float = 0.01) -> float:
    logits = logits.to(device)
    y = y.to(device)
    T = torch.nn.Parameter(torch.ones(1, device=device))
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp_min(1e-3), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().cpu().item())


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y).astype(np.float64)
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc = accuracies[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_npz", type=Path,
                        default=PATHS["cache"] / "small_arrays.npz")
    parser.add_argument("--ckpt", type=Path,
                        default=PATHS["ckpt"] / "vit_best.pth")
    parser.add_argument("--out_path", type=Path,
                        default=PATHS["ckpt"] / "temperature.npy")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = get_device()
    logger = get_logger("calibrate", PATHS["results"] / "calibrate.log")

    _, val_ds, _ = build_small_datasets(args.small_npz, augment_train=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    state = torch.load(args.ckpt, map_location=device)
    cfg = ViTConfig(**state.get("cfg", {})) if isinstance(state.get("cfg"), dict) else ViTConfig()
    model = SmallViT(cfg).to(device)
    # Handle both checkpoint formats: "model_state" and "state_dict"
    state_dict_key = "model_state" if "model_state" in state else "state_dict"
    model.load_state_dict(state[state_dict_key])
    logger.info("loaded %s", args.ckpt)

    logits, y = collect_val_logits(model, val_loader, device)
    logger.info("val logits shape=%s", tuple(logits.shape))

    probs_pre = F.softmax(logits, dim=-1).numpy()
    ece_pre = expected_calibration_error(probs_pre, y.numpy())
    logger.info("pre-calibration  ECE=%.4f  acc=%.4f",
                ece_pre, float((probs_pre.argmax(1) == y.numpy()).mean()))

    T = fit_temperature(logits, y, device)
    logger.info("fitted T=%.4f", T)

    probs_post = F.softmax(logits / T, dim=-1).numpy()
    ece_post = expected_calibration_error(probs_post, y.numpy())
    logger.info("post-calibration ECE=%.4f  acc=%.4f",
                ece_post, float((probs_post.argmax(1) == y.numpy()).mean()))

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_path, np.array([T], dtype=np.float32))
    logger.info("wrote %s (T=%.4f)", args.out_path, T)

    if T < 1.0 - 1e-3:
        logger.warning("T<1 is unusual; this would sharpen the distribution, "
                       "suggesting the model is underconfident.")
    if T > 5.0:
        logger.warning("T>5 is very large; model was extremely overconfident "
                       "or training didn't converge well.")


if __name__ == "__main__":
    main()
