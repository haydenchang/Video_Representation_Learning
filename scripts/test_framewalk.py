print("TEST_FRAMEWALK STARTED")

from nuscenes.nuscenes import NuScenes
from src.data.framewalk import get_first_cam_front_sd_token, walk_cam_front

DATAROOT = r"C:\DS\TPV\nuScenes"

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)

start = get_first_cam_front_sd_token(nusc, scene_idx=0)
frames = walk_cam_front(nusc, start_sd_token=start, max_steps=20)

print("Got frames:", len(frames))
print("First ts:", frames[0].timestamp, "file:", frames[0].filename)
print("Last  ts:", frames[-1].timestamp, "file:", frames[-1].filename)

# Check monotonic timestamps
is_monotonic = all(frames[i].timestamp < frames[i+1].timestamp for i in range(len(frames)-1))
print("Timestamps strictly increasing:", is_monotonic)

# Print a few deltas (sanity)
deltas = [frames[i+1].timestamp - frames[i].timestamp for i in range(len(frames)-1)]
print("First 5 deltas:", deltas[:5])
