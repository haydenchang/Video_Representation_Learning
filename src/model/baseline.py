from __future__ import annotations

import torch
import torch.nn as nn


class BaselineReconstructor(nn.Module):
    """
    Minimal model: learn a single global vector for masked tokens, plus a linear projection
    from visible token space to output space (optional).

    For v1 plumbing validation, we simply output a learned mask token everywhere.
    """
    def __init__(self, D: int):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(D))

    def forward(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        # pred_tokens: [B, N, D]
        return self.mask_token.view(1, 1, -1).to(device).expand(B, N, -1)
