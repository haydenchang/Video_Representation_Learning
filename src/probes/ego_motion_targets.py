from __future__ import annotations

import math
import numpy as np
from nuscenes.nuscenes import NuScenes

from src.data.clip_sampler import sample_clip_from_start
from src.probes.ego_motion_labels import SpeedBuckets, speed_to_bucket, EGO_LABELS


def _l2(a, b) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def clip_mean_speed_mps(
    nusc: NuScenes,
    start_sd_token: str,
    T: int,
    stride: int,
    keyframes_only: bool,
) -> float:
    """
    Computes mean ego speed over consecutive frames in the clip (m/s).
    Uses ego_pose translations (meters) and timestamps (microseconds).
    """
    clip = sample_clip_from_start(
        nusc,
        start_sd_token=start_sd_token,
        T=T,
        stride=stride,
        keyframes_only=keyframes_only,
    )

    speeds = []
    for i in range(len(clip.frames) - 1):
        fr0 = clip.frames[i]
        fr1 = clip.frames[i + 1]

        sd0 = nusc.get("sample_data", fr0.sd_token)
        sd1 = nusc.get("sample_data", fr1.sd_token)

        ep0 = nusc.get("ego_pose", sd0["ego_pose_token"])
        ep1 = nusc.get("ego_pose", sd1["ego_pose_token"])

        p0 = ep0["translation"]  # meters
        p1 = ep1["translation"]  # meters

        dt = (fr1.timestamp - fr0.timestamp) / 1e6  # seconds (timestamps are in microseconds)
        if dt <= 0:
            raise RuntimeError(f"Non-positive dt={dt} between frames {i}->{i+1}")

        dist = _l2(p0, p1)  # meters
        speeds.append(dist / dt)  # m/s

    if len(speeds) == 0:
        raise RuntimeError("Clip too short to compute speed")

    return float(np.mean(speeds))


def clip_ego_motion_bucket_onehot(
    nusc: NuScenes,
    start_sd_token: str,
    T: int,
    stride: int,
    keyframes_only: bool,
    buckets: SpeedBuckets = SpeedBuckets(),
) -> np.ndarray:
    """
    Returns one-hot vector [3] for slow/medium/fast based on mean speed.
    """
    mean_speed = clip_mean_speed_mps(nusc, start_sd_token, T, stride, keyframes_only)
    k = speed_to_bucket(mean_speed, buckets=buckets)
    y = np.zeros((len(EGO_LABELS),), dtype=np.float32)
    y[k] = 1.0
    return y
