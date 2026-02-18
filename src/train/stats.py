from __future__ import annotations

import math
import torch


@torch.no_grad()
def global_grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(g.pow(2).sum().item())
    return math.sqrt(total)


@torch.no_grad()
def global_param_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        w = p.detach()
        total += float(w.pow(2).sum().item())
    return math.sqrt(total)


@torch.no_grad()
def pred_stats(pred_tokens: torch.Tensor) -> dict:
    return {
        "pred_mean": float(pred_tokens.float().mean().item()),
        "pred_std": float(pred_tokens.float().std().item()),
        "pred_absmax": float(pred_tokens.float().abs().max().item()),
    }

@torch.no_grad()
def latent_feat_std(x_latent: torch.Tensor) -> float:
    """
    x_latent: [B, Nv, d_model] (encoder output over visible tokens)
    Returns: average std across feature dims, computed over all (B*Nv) tokens.
    """
    if x_latent.ndim != 3:
        raise ValueError(f"Expected [B,Nv,d], got {tuple(x_latent.shape)}")
    x = x_latent.detach().float().reshape(-1, x_latent.shape[-1])  # [B*Nv, d]
    # unbiased=False for stability on small samples
    per_dim = x.std(dim=0, unbiased=False)
    return float(per_dim.mean().item())
