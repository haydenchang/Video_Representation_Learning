from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    T: int = 8
    stride: int = 1
    keyframes_only: bool = True

    # Locked training resolution
    out_h: int = 360
    out_w: int = 640

    # Pixel range policy
    to_float_01: bool = True


DATA_CFG = DataConfig()
