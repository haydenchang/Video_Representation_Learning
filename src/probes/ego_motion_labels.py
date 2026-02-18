from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

EGO_LABELS = ["slow", "medium", "fast"]
EGO_LABEL_TO_IDX = {k: i for i, k in enumerate(EGO_LABELS)}

@dataclass(frozen=True)
class SpeedBuckets:
    """
    Thresholds in m/s.
    slow:   [0, slow_max)
    medium: [slow_max, med_max)
    fast:   [med_max, +inf)
    """
    slow_max: float = 2.0   # ~7.2 km/h
    med_max: float = 8.0    # ~28.8 km/h

def speed_to_bucket(speed_mps: float, buckets: SpeedBuckets = SpeedBuckets()) -> int:
    if speed_mps < buckets.slow_max:
        return EGO_LABEL_TO_IDX["slow"]
    if speed_mps < buckets.med_max:
        return EGO_LABEL_TO_IDX["medium"]
    return EGO_LABEL_TO_IDX["fast"]
