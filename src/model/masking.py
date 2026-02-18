from __future__ import annotations

import torch


def make_random_mask(
    B: int,
    N: int,
    mask_ratio: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Returns mask [B, N] boolean where True = masked.
    Deterministic if generator is provided with a fixed seed.
    """
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError("mask_ratio must be in (0,1)")
    num_mask = int(round(N * mask_ratio))
    if num_mask <= 0 or num_mask >= N:
        raise ValueError("mask_ratio leads to degenerate mask count")

    # Sample random scores and take top-k as masked
    scores = torch.rand((B, N), device=device, generator=generator)
    _, idx = torch.topk(scores, k=num_mask, dim=1, largest=True, sorted=False)

    mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    mask.scatter_(1, idx, True)
    return mask


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred:   [B, N, D]
    target: [B, N, D]
    mask:   [B, N] boolean (True=masked)
    Returns scalar loss averaged over masked positions and D.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")
    if mask.ndim != 2 or mask.shape[0] != pred.shape[0] or mask.shape[1] != pred.shape[1]:
        raise ValueError(f"mask shape {mask.shape} incompatible with pred shape {pred.shape}")

    diff2 = (pred - target) ** 2  # [B,N,D]
    m = mask.unsqueeze(-1).to(diff2.dtype)  # [B,N,1] float {0,1}
    denom = m.sum() * diff2.shape[-1]       # (#masked tokens total) * D

    if denom.item() == 0:
        raise RuntimeError("No masked tokens; check mask_ratio")

    # sum only masked tokens, then normalize
    loss = (diff2 * m).sum() / denom
    return loss

