from nuscenes.nuscenes import NuScenes
from src.data.clip_index import load_clip_index
from src.probes.object_presence_targets import clip_object_presence_multihot
from src.probes.object_presence_labels import LABELS

DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
items = load_clip_index(INDEX_PATH)

for i in [0, 1, 2]:
    spec = items[i]
    y = clip_object_presence_multihot(
        nusc,
        start_sd_token=spec.start_sd_token,
        T=spec.T,
        stride=spec.stride,
        keyframes_only=spec.keyframes_only,
    )
    present = [LABELS[j] for j in range(len(LABELS)) if y[j] > 0.5]
    print(f"clip {i}: present={present} y={y.tolist()}")
