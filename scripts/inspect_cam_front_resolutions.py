from collections import Counter
from nuscenes.nuscenes import NuScenes

DATAROOT = r"C:\DS\TPV\nuScenes"
nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)

# Sample a limited number of scenes/samples to keep it fast
max_scenes = 20
max_samples_per_scene = 50

ctr = Counter()
count = 0

scene_limit = min(len(nusc.scene), max_scenes)
for scene_idx in range(scene_limit):
    scene = nusc.scene[scene_idx]
    sample_token = scene["first_sample_token"]

    k = 0
    while sample_token != "" and k < max_samples_per_scene:
        sample = nusc.get("sample", sample_token)
        sd_token = sample["data"]["CAM_FRONT"]
        sd = nusc.get("sample_data", sd_token)

        ctr[(sd["height"], sd["width"])] += 1
        count += 1

        sample_token = sample["next"]
        k += 1

print("samples inspected:", count)
print("unique resolutions:", len(ctr))
for (h, w), n in ctr.most_common(10):
    print(f"{h}x{w}: {n}")
