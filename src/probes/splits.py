from __future__ import annotations

import random
from typing import Sequence

from src.data.clip_index import ClipSpec


def scene_disjoint_split(
    items: Sequence[ClipSpec],
    val_scene_frac: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """
    Split clip indices by scene so train/val are scene-disjoint.
    """
    if not (0.0 < val_scene_frac < 1.0):
        raise ValueError(f"val_scene_frac must be in (0,1), got {val_scene_frac}")
    if len(items) == 0:
        raise ValueError("Cannot split empty clip list")

    scenes = sorted({it.scene_idx for it in items})
    if len(scenes) < 2:
        raise ValueError("Need at least 2 scenes for scene-disjoint split")

    rng = random.Random(seed)
    rng.shuffle(scenes)

    n_val_scenes = max(1, int(round(len(scenes) * val_scene_frac)))
    n_val_scenes = min(n_val_scenes, len(scenes) - 1)
    val_scenes = set(scenes[:n_val_scenes])

    train_idx: list[int] = []
    val_idx: list[int] = []
    for i, spec in enumerate(items):
        if spec.scene_idx in val_scenes:
            val_idx.append(i)
        else:
            train_idx.append(i)

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("Scene split produced empty train/val")

    return train_idx, val_idx
