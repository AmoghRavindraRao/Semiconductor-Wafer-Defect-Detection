"""
utils.py
Centralized utilities: paths, device, logging, and TTA.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ============================================================================
# Constants (from data_utils.py)
# ============================================================================
CLASS_TO_IDX = {
    "Center":    0, "Donut":     1, "Edge-Loc":  2, "Edge-Ring": 3,
    "Loc":       4, "Near-full": 5, "Random":    6, "Scratch":   7,
    "none":      8,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES  = 9
CANON_SIZE   = 64


# ============================================================================
# Paths (relative to project root)
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
PATHS = {
    "data":     PROJECT_ROOT / "data",
    "cache":    PROJECT_ROOT / "data_cache",
    "ckpt":     PROJECT_ROOT / "checkpoints",
    "emb":      PROJECT_ROOT / "embeddings",
    "results":  PROJECT_ROOT / "results",
    "pseudo":   PROJECT_ROOT / "pseudo_labels",
    "test":     PROJECT_ROOT / "test_data",
}


# ============================================================================
# Device Management
# ============================================================================
def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Logging
# ============================================================================
def get_logger(name: str, log_file: Path | str | None = None) -> logging.Logger:
    """
    Create a logger instance.
    
    Args:
        name: Logger name
        log_file: Optional path to log file for file output
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# Test-Time Augmentation (TTA)
# ============================================================================
@torch.no_grad()
def tta_forward(
    model,
    x: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass with 4-rotation TTA (0°, 90°, 180°, 270°).
    
    Args:
        model: SmallViT model (callable)
        x: Input tensor of shape (B, 64, 64) with values in {0, 1, 2}
        temperature: Temperature for logit scaling (default 1.0 = no scaling)
    
    Returns:
        Tuple of (mean_logits, mean_emb, agreement):
            - mean_logits: (B, 9) averaged logits across 4 rotations
            - mean_emb: (B, D) averaged embeddings across 4 rotations
            - agreement: (B,) boolean, True if all 4 rotations agree on argmax class
    """
    model.eval()
    B = x.shape[0]
    
    # Ensure x is long tensor with shape (B, 64, 64)
    x = x.long()
    if x.dim() == 2:
        # Single sample case, add batch dimension
        x = x.unsqueeze(0)
    
    # Generate 4 augmentations: identity, rot90, rot180, rot270
    augmentations = [
        x,                                         # identity, (B, 64, 64)
        torch.rot90(x, k=1, dims=[1, 2]),         # 90°, (B, 64, 64)
        torch.rot90(x, k=2, dims=[1, 2]),         # 180°, (B, 64, 64)
        torch.rot90(x, k=3, dims=[1, 2]),         # 270°, (B, 64, 64)
    ]
    
    all_logits = []
    all_emb = []
    all_preds = []
    
    for i, x_aug in enumerate(augmentations):
        logits, emb = model(x_aug)  # (B, 9), (B, D)
        
        # Rotate embeddings back to original orientation if they're spatial
        # Embeddings are typically (B, D) where D is feature dimension, not spatial
        # So we don't need to rotate them back
        
        # Scale logits by temperature and collect
        scaled_logits = logits / temperature if temperature != 1.0 else logits
        all_logits.append(scaled_logits)
        all_emb.append(emb)
        all_preds.append(logits.argmax(dim=-1))
    
    # Average logits and embeddings
    mean_logits = torch.stack(all_logits, dim=0).mean(dim=0)  # (B, 9)
    mean_emb = torch.stack(all_emb, dim=0).mean(dim=0)        # (B, D)
    
    # Compute agreement: True if all 4 rotations agree on argmax
    all_preds_stack = torch.stack(all_preds, dim=0)  # (4, B)
    agreement = (all_preds_stack == all_preds_stack[0]).all(dim=0)  # (B,)
    
    return mean_logits, mean_emb, agreement.float()
