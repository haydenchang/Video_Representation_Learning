from nuscenes.nuscenes import NuScenes
from src.data.framewalk import get_first_cam_front_sd_token
from src.data.clip_sampler import sample_clip_from_start

DATAROOT = r"C:\DS\TPV\nuScenes"
nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)

start = get_first_cam_front_sd_token(nusc, scene_idx=0)

clip = sample_clip_from_start(
    nusc,
    start_sd_token=start,
    T=8,
    stride=1,
    keyframes_only=True,
)

print("OK: clip sampled")
print("T:", clip.T, "stride:", clip.stride, "keyframes_only:", clip.keyframes_only)
print("first:", clip.frames[0].filename)
print("last :", clip.frames[-1].filename)

ts = [f.timestamp for f in clip.frames]
deltas = [ts[i+1] - ts[i] for i in range(len(ts)-1)]
print("deltas:", deltas)

assert all(f.is_key_frame for f in clip.frames)
assert all(not f.filename.startswith("sweeps/") for f in clip.frames)
assert all(ts[i] < ts[i+1] for i in range(len(ts)-1))
print("OK: assertions passed")
