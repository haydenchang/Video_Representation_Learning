from collections import Counter
from nuscenes.nuscenes import NuScenes

from src.probes.object_presence_labels import map_nuscenes_category_to_label, LABELS

DATAROOT = r"C:\DS\TPV\nuScenes"
nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)

max_scenes = 20
max_samples_per_scene = 50

raw_ctr = Counter()
mapped_ctr = Counter()
unmapped_ctr = Counter()

scene_limit = min(len(nusc.scene), max_scenes)
for scene_idx in range(scene_limit):
    scene = nusc.scene[scene_idx]
    sample_token = scene["first_sample_token"]

    k = 0
    while sample_token != "" and k < max_samples_per_scene:
        sample = nusc.get("sample", sample_token)
        for ann_tok in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_tok)
            cat = ann["category_name"]
            raw_ctr[cat] += 1
            lab = map_nuscenes_category_to_label(cat)
            if lab is None:
                unmapped_ctr[cat] += 1
            else:
                mapped_ctr[lab] += 1
        sample_token = sample["next"]
        k += 1

print("Mapped labels:", LABELS)
print("Mapped counts:")
for k in LABELS:
    print(k, mapped_ctr[k])

print("\nTop raw categories:")
for k, v in raw_ctr.most_common(15):
    print(k, v)

print("\nTop UNMAPPED categories:")
for k, v in unmapped_ctr.most_common(15):
    print(k, v)
