from __future__ import annotations

import torch


def tubeletify(x: torch.Tensor, t: int, p: int) -> torch.Tensor:
    """
    x: [B, T, C, H, W]
    returns: tokens [B, N, D] where D = C*t*p*p and N = (T/t)*(H/p)*(W/p)
    """
    if x.ndim != 5:
        raise ValueError(f"Expected [B,T,C,H,W], got shape={tuple(x.shape)}")
    B, T, C, H, W = x.shape
    if T % t != 0:
        raise ValueError(f"T={T} not divisible by t={t}")
    if H % p != 0 or W % p != 0:
        raise ValueError(f"H,W=({H},{W}) not divisible by p={p}")

    Tt = T // t
    Hp = H // p
    Wp = W // p

    # [B, Tt, t, C, Hp, p, Wp, p]
    x = x.view(B, Tt, t, C, Hp, p, Wp, p)
    # [B, Tt, Hp, Wp, t, p, p, C]
    x = x.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
    # [B, N, D]
    tokens = x.view(B, Tt * Hp * Wp, t * p * p * C)
    return tokens


def untubeletify(tokens: torch.Tensor, t: int, p: int, T: int, H: int, W: int) -> torch.Tensor:
    """
    tokens: [B, N, D]
    returns: x [B, T, C, H, W]
    """
    if tokens.ndim != 3:
        raise ValueError(f"Expected [B,N,D], got shape={tuple(tokens.shape)}")
    B, N, D = tokens.shape
    if T % t != 0:
        raise ValueError(f"T={T} not divisible by t={t}")
    if H % p != 0 or W % p != 0:
        raise ValueError(f"H,W=({H},{W}) not divisible by p={p}")

    Tt = T // t
    Hp = H // p
    Wp = W // p
    expected_N = Tt * Hp * Wp
    expected_D = 3 * t * p * p  # C=3 for RGB

    if N != expected_N:
        raise ValueError(f"N={N} does not match expected_N={expected_N}")
    if D != expected_D:
        raise ValueError(f"D={D} does not match expected_D={expected_D}")

    x = tokens.view(B, Tt, Hp, Wp, t, p, p, 3)
    # invert permute: [B, Tt, t, 3, Hp, p, Wp, p]
    x = x.permute(0, 1, 4, 7, 2, 5, 3, 6).contiguous()
    x = x.view(B, T, 3, H, W)
    return x
