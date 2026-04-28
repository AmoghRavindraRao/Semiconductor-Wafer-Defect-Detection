"""
models.py

Two architectures, each exposing the same interface for SupCon training:
  forward(x, return_projections=False) -> (logits, embeddings) or (logits, embeddings, projections)

  - SmallViT: PRD §6.2 spec. Patch embedding from {0,1,2} via nn.Embedding.
  - ResNetWafer: torchvision ResNet (50 or 18) trained from scratch on
    (3, 64, 64) inputs with values in {0.0, 0.5, 1.0}. NO ImageNet weights.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18


# ===========================================================================
# Small ViT (PRD §6.2)
# ===========================================================================
class SmallViT(nn.Module):
    def __init__(self,
                 img_size: int = 64,
                 patch_size: int = 8,
                 vocab_size: int = 3,
                 token_dim: int = 64,
                 embed_dim: int = 192,
                 depth: int = 4,
                 heads: int = 4,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 emb_out_dim: int = 256,
                 proj_dim: int = 128,
                 num_classes: int = 9):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Per-pixel token embedding {0,1,2} -> token_dim
        self.token_emb = nn.Embedding(vocab_size, token_dim)

        # Patchify: each patch becomes (patch_size*patch_size*token_dim) -> embed_dim
        self.patch_proj = nn.Linear(patch_size * patch_size * token_dim, embed_dim)

        # CLS + positional
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_emb   = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # Heads
        self.embed_head = nn.Linear(embed_dim, emb_out_dim)              # 256
        self.proj_head = nn.Sequential(                                   # 128, train-only
            nn.Linear(emb_out_dim, emb_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_out_dim, proj_dim),
        )
        self.classifier = nn.Linear(emb_out_dim, num_classes)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W) long, values in {0,1,2}
        B, H, W = x.shape
        tok = self.token_emb(x)                                          # (B, H, W, td)
        ps = self.patch_size
        # (B, H/ps, ps, W/ps, ps, td) -> (B, nP, ps*ps*td)
        tok = tok.unfold(1, ps, ps).unfold(2, ps, ps)                    # (B, H/ps, W/ps, td, ps, ps)
        tok = tok.permute(0, 1, 2, 4, 5, 3).contiguous()                 # (B, H/ps, W/ps, ps, ps, td)
        tok = tok.view(B, self.num_patches, -1)                          # (B, nP, ps*ps*td)
        return self.patch_proj(tok)                                       # (B, nP, embed_dim)

    def forward(self, x: torch.Tensor, return_projections: bool = False):
        B = x.size(0)
        patches = self._patchify(x)                                       # (B, nP, D)
        cls = self.cls_token.expand(B, -1, -1)                            # (B, 1, D)
        z = torch.cat([cls, patches], dim=1) + self.pos_emb               # (B, nP+1, D)
        z = self.encoder(z)
        z = self.norm(z)
        cls_out = z[:, 0]                                                 # (B, D)

        emb = self.embed_head(cls_out)                                    # (B, 256)
        emb_n = F.normalize(emb, dim=-1)
        logits = self.classifier(emb)

        if return_projections:
            proj = self.proj_head(emb)
            proj = F.normalize(proj, dim=-1)
            return logits, emb_n, proj
        return logits, emb_n


# ===========================================================================
# ResNet wafer model (from scratch, supports SupCon)
# ===========================================================================
class ResNetWafer(nn.Module):
    """
    torchvision ResNet (50 or 18) from scratch, with the same head structure
    as SmallViT for ensemble symmetry.

    Input: (B, 3, 64, 64) float in [0, 1]. NO ImageNet normalization.
    """
    def __init__(self, backbone: str = "resnet50",
                 emb_out_dim: int = 256,
                 proj_dim: int = 128,
                 num_classes: int = 9):
        super().__init__()
        if backbone == "resnet50":
            net = resnet50(weights=None)
            feat_dim = 2048
        elif backbone == "resnet18":
            net = resnet18(weights=None)
            feat_dim = 512
        else:
            raise ValueError(f"unknown backbone: {backbone}")

        # For 64x64 inputs: shrink the stem so we don't downsample to 1x1 too fast.
        # Original ResNet stem: conv(7,s=2) + maxpool(s=2) -> /4 in spatial.
        # Replace with conv(3,s=1) + no maxpool -> keep spatial.
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Identity()  # we attach our own head
        self.backbone = net

        self.embed_head = nn.Linear(feat_dim, emb_out_dim)
        self.proj_head  = nn.Sequential(
            nn.Linear(emb_out_dim, emb_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_out_dim, proj_dim),
        )
        self.classifier = nn.Linear(emb_out_dim, num_classes)

    def forward(self, x: torch.Tensor, return_projections: bool = False):
        feat = self.backbone(x)                                           # (B, feat_dim)
        emb = self.embed_head(feat)                                       # (B, 256)
        emb_n = F.normalize(emb, dim=-1)
        logits = self.classifier(emb)
        if return_projections:
            proj = self.proj_head(emb)
            proj = F.normalize(proj, dim=-1)
            return logits, emb_n, proj
        return logits, emb_n


def build_model(name: str, num_classes: int = 9) -> nn.Module:
    name = name.lower()
    if name == "vit":
        return SmallViT(num_classes=num_classes)
    if name == "resnet50":
        return ResNetWafer(backbone="resnet50", num_classes=num_classes)
    if name == "resnet18":
        return ResNetWafer(backbone="resnet18", num_classes=num_classes)
    raise ValueError(f"unknown model: {name}")
