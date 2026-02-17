from __future__ import annotations

import torch


def assert_finite_tensor(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        bad = x[~torch.isfinite(x)]
        raise RuntimeError(f"{name} contains non-finite values. Example bad values: {bad[:5].detach().cpu().tolist()}")


def assert_finite_loss(loss: torch.Tensor) -> None:
    if not torch.isfinite(loss):
        raise RuntimeError(f"Loss is non-finite: {loss.item()}")


def assert_finite_grads(model: torch.nn.Module) -> None:
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            bad = p.grad[~torch.isfinite(p.grad)]
            raise RuntimeError(f"Non-finite grad in {n}. Example bad: {bad[:5].detach().cpu().tolist()}")


def assert_finite_params(model: torch.nn.Module) -> None:
    for n, p in model.named_parameters():
        if not torch.isfinite(p).all():
            bad = p[~torch.isfinite(p)]
            raise RuntimeError(f"Non-finite param in {n}. Example bad: {bad[:5].detach().cpu().tolist()}")
