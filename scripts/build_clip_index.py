from nuscenes.nuscenes import NuScenes
from src.data.clip_index import build_clip_index, save_clip_index

DATAROOT = r"C:\DS\TPV\nuScenes"
OUTPATH = r"artifacts\clip_index_T8_s1_keyframes.json"

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)

items = build_clip_index(
    nusc,
    T=8,
    stride=1,
    keyframes_only=True,
    max_scenes=5,           # small for now (local-only sanity)
    max_clips_per_scene=10, # cap work
)

print("clips:", len(items))
print("first:", items[0])
save_clip_index(items, OUTPATH)
print("saved:", OUTPATH)
