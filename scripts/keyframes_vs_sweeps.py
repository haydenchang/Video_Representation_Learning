from collections import Counter
from nuscenes.nuscenes import NuScenes
from src.data.framewalk import get_first_cam_front_sd_token, walk_cam_front

DATAROOT = r"C:\DS\TPV\nuScenes"
nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)

start = get_first_cam_front_sd_token(nusc, scene_idx=0)
frames = walk_cam_front(nusc, start_sd_token=start, max_steps=200)

k = sum(1 for f in frames if f.is_key_frame)
s = len(frames) - k

prefix_counts = Counter(f.filename.split("/")[0] for f in frames)

print("Total:", len(frames))
print("Keyframes (is_key_frame=True):", k)
print("Non-keyframes:", s)
print("Path prefix counts:", dict(prefix_counts))

# show first 30 as (key?, prefix, ts_delta)
for i in range(1, min(30, len(frames))):
    dt = frames[i].timestamp - frames[i-1].timestamp
    pref = frames[i].filename.split("/")[0]
    print(i, "key" if frames[i].is_key_frame else "sweep", pref, "dt=", dt)
