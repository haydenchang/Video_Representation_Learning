from nuscenes.nuscenes import NuScenes
from src.data.clip_index import load_clip_index
from src.probes.ego_motion_targets import clip_mean_speed_mps, clip_ego_motion_bucket_onehot
from src.probes.ego_motion_labels import EGO_LABELS, SpeedBuckets

DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
items = load_clip_index(INDEX_PATH)

buckets = SpeedBuckets(slow_max=2.0, med_max=8.0)

for i in [0, 1, 2, 10, 20]:
    spec = items[i]
    v = clip_mean_speed_mps(
        nusc,
        start_sd_token=spec.start_sd_token,
        T=spec.T,
        stride=spec.stride,
        keyframes_only=spec.keyframes_only,
    )
    y = clip_ego_motion_bucket_onehot(
        nusc,
        start_sd_token=spec.start_sd_token,
        T=spec.T,
        stride=spec.stride,
        keyframes_only=spec.keyframes_only,
        buckets=buckets,
    )
    lab = EGO_LABELS[int(y.argmax())]
    print(f"clip {i}: mean_speed_mps={v:.3f} -> {lab} onehot={y.tolist()}")
