"""
model.py
Wrapper module that re-exports from models.py and provides ViTConfig.
"""

from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
from models import SmallViT as _SmallViT, ResNetWafer, build_model


@dataclass
class ViTConfig:
    """Configuration for SmallViT model."""
    img_size: int = 64
    patch_size: int = 8
    vocab_size: int = 3
    token_dim: int = 64
    embed_dim: int = 192
    depth: int = 4
    heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1
    emb_out_dim: int = 256
    proj_dim: int = 128
    num_classes: int = 9


class SmallViT(_SmallViT):
    """SmallViT with support for ViTConfig parameter."""
    
    def __init__(self, cfg: Optional[ViTConfig] = None, **kwargs):
        """
        Initialize SmallViT from a ViTConfig object or individual parameters.
        
        Args:
            cfg: ViTConfig object (if provided, kwargs are ignored)
            **kwargs: Individual parameters (used if cfg is None)
        """
        if cfg is not None:
            if isinstance(cfg, ViTConfig):
                params = asdict(cfg)
            elif isinstance(cfg, dict):
                params = cfg
            else:
                raise TypeError(f"cfg must be ViTConfig or dict, got {type(cfg)}")
        else:
            params = kwargs if kwargs else asdict(ViTConfig())
        
        super().__init__(**params)


# Re-export for convenience
__all__ = [
    "ViTConfig",
    "SmallViT",
    "ResNetWafer",
    "build_model",
]
