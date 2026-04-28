"""
losses.py

Supervised contrastive loss (Khosla et al. 2020). Operates on L2-normalized
projection vectors. For each anchor, positives are same-class samples in the
batch; negatives are everything else.

Used at training time only. The projection head is discarded at inference.
"""
import torch
import torch.nn.functional as F


def supcon_loss(projections: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.1) -> torch.Tensor:
    """
    Args:
      projections: (B, D) — L2-normalized
      labels:      (B,)   — int class indices
      temperature: float
    Returns:
      scalar loss
    """
    device = projections.device
    B = projections.size(0)

    # Cosine sim (projections are unit-norm already)
    sim = projections @ projections.t()                        # (B, B)
    sim = sim / temperature

    # For numerical stability, subtract max per row
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    # Mask: exclude self from both numerator and denominator
    self_mask = torch.eye(B, dtype=torch.bool, device=device)

    # Same-class mask (positives)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()) & ~self_mask             # (B, B), bool

    exp_sim = torch.exp(sim)
    # Denominator: sum over all non-self samples
    denom = exp_sim.masked_fill(self_mask, 0).sum(dim=1, keepdim=True)  # (B, 1)
    log_prob = sim - torch.log(denom + 1e-12)                          # (B, B)

    # Average log-prob over positives, per anchor that HAS positives
    pos_count = pos_mask.sum(dim=1)                                    # (B,)
    has_pos = pos_count > 0
    if not has_pos.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    log_prob_pos = (log_prob * pos_mask).sum(dim=1)                    # (B,)
    mean_log_prob_pos = log_prob_pos[has_pos] / pos_count[has_pos].float()
    return -mean_log_prob_pos.mean()
