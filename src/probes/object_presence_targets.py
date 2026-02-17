from __future__ import annotations

import numpy as np
from nuscenes.nuscenes import NuScenes

from src.data.clip_sampler import sample_clip_from_start
from src.probes.object_presence_labels import LABELS, LABEL_TO_IDX, map_nuscenes_category_to_label


def clip_object_presence_multihot(
    nusc: NuScenes,
    start_sd_token: str,
    T: int,
    stride: int,
    keyframes_only: bool,
) -> np.ndarray:
    """
    Returns multi-hot vector [K] for the clip: OR over frames of mapped object classes.
    """
    K = len(LABELS)
    y = np.zeros((K,), dtype=np.float32)

    clip = sample_clip_from_start(
        nusc,
        start_sd_token=start_sd_token,
        T=T,
        stride=stride,
        keyframes_only=keyframes_only,
    )

    for fr in clip.frames:
        sample = nusc.get("sample", fr.sample_token)
        for ann_tok in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_tok)
            cat = ann["category_name"]
            lab = map_nuscenes_category_to_label(cat)
            if lab is not None:
                y[LABEL_TO_IDX[lab]] = 1.0

    return y
