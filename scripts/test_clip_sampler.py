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

print("frames:", len(clip.frames))
print("all keyframes:", all(f.is_key_frame for f in clip.frames))
print("any sweeps path:", any(f.filename.startswith("sweeps/") for f in clip.frames))
