from __future__ import annotations

import numpy as np
import torch


def average_precision(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """
    y_true: [N] in {0,1}
    y_score: [N] real-valued (higher => more confident positive)

    Implements AP via sorting by score and integrating precision at positive hits.
    If a class has no positives in y_true, returns np.nan (we will ignore in mAP).
    """
    y_true = y_true.detach().cpu().float()
    y_score = y_score.detach().cpu().float()

    n_pos = int(y_true.sum().item())
    if n_pos == 0:
        return float("nan")

    order = torch.argsort(y_score, descending=True)
    y = y_true[order]

    tp = torch.cumsum(y, dim=0)
    fp = torch.cumsum(1.0 - y, dim=0)
    precision = tp / (tp + fp + 1e-12)

    # AP = mean precision at each true positive position
    ap = (precision * y).sum() / (n_pos + 1e-12)
    return float(ap.item())


def mean_average_precision(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """
    y_true: [N,K] in {0,1}
    y_score: [N,K] real-valued
    Returns mAP over classes with at least one positive in y_true.
    """
    K = y_true.shape[1]
    aps = []
    for k in range(K):
        ap = average_precision(y_true[:, k], y_score[:, k])
        if not np.isnan(ap):
            aps.append(ap)
    if len(aps) == 0:
        return float("nan")
    return float(np.mean(aps))
