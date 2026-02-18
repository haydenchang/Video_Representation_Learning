from collections import Counter
from nuscenes.nuscenes import NuScenes

DATAROOT = r"C:\DS\TPV\nuScenes"
nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)

scene = nusc.scene[0]
sample = nusc.get("sample", scene["first_sample_token"])

print("Sample keys:", sample.keys())
print("Number of ann tokens:", len(sample["anns"]))

# Inspect first 5 annotations
cats = []
for tok in sample["anns"][:5]:
    ann = nusc.get("sample_annotation", tok)
    print("Ann keys:", ann.keys())
    print("category_name:", ann["category_name"])
    cats.append(ann["category_name"])

# Count categories across this one sample (full list)
ctr = Counter()
for tok in sample["anns"]:
    ann = nusc.get("sample_annotation", tok)
    ctr[ann["category_name"]] += 1

print("Top categories in this sample:")
for k, v in ctr.most_common(10):
    print(k, v)
