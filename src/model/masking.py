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
    Returns scalar loss averaged over (B * num_masked * D)
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")
    if mask.ndim != 2 or mask.shape[0] != pred.shape[0] or mask.shape[1] != pred.shape[1]:
        raise ValueError(f"mask shape {mask.shape} incompatible with pred shape {pred.shape}")

    # expand mask to [B,N,1] for broadcasting
    diff2 = (pred - target) ** 2  # [B, N, D]
    m3 = mask.unsqueeze(-1).expand_as(diff2)  # [B, N, D]
    masked_diff2 = diff2[m3]  # 1D view of all masked entries

    if masked_diff2.numel() == 0:
        raise RuntimeError("No masked elements; check mask_ratio")

    return masked_diff2.mean()
