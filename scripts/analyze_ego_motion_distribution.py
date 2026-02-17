import numpy as np
from collections import Counter
from nuscenes.nuscenes import NuScenes

from src.data.clip_index import load_clip_index
from src.probes.ego_motion_targets import clip_mean_speed_mps
from src.probes.ego_motion_labels import SpeedBuckets, speed_to_bucket, EGO_LABELS

DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
items = load_clip_index(INDEX_PATH)

buckets = SpeedBuckets(slow_max=2.0, med_max=8.0)

speeds = []
bucket_ids = []

for spec in items:
    v = clip_mean_speed_mps(
        nusc,
        start_sd_token=spec.start_sd_token,
        T=spec.T,
        stride=spec.stride,
        keyframes_only=spec.keyframes_only,
    )
    speeds.append(v)
    bucket_ids.append(speed_to_bucket(v, buckets=buckets))

speeds = np.array(speeds, dtype=np.float32)

print("num clips:", len(speeds))
print("speed min/median/max:", float(speeds.min()), float(np.median(speeds)), float(speeds.max()))
for q in [5, 25, 50, 75, 95]:
    print(f"p{q}:", float(np.percentile(speeds, q)))

ctr = Counter(bucket_ids)
print("\nbucket counts:")
for i, name in enumerate(EGO_LABELS):
    print(name, ctr.get(i, 0))
