import numpy as np
import torch
from collections import Counter
from nuscenes.nuscenes import NuScenes

from src.model.mvm import MaskedVideoModel
from src.probes.probe_dataset import ObjectPresenceProbeDataset

DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"

t, p = 2, 40
N = (8 // t) * (360 // p) * (640 // p)
D = 3 * t * p * p
device = torch.device("cpu")

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
enc = MaskedVideoModel(N=N, D=D, d_model=256, n_layers=2, n_heads=8).to(device).eval()

ds = ObjectPresenceProbeDataset(nusc, DATAROOT, INDEX_PATH, enc, t=t, p=p, mask_ratio_visible=0.0, device=device)

ctr = Counter()
for i in range(len(ds)):
    _, y = ds[i]
    key = tuple(int(v) for v in y.tolist())
    ctr[key] += 1

print("num_samples:", len(ds))
print("num_unique_label_vectors:", len(ctr))
print("top 10 label vectors:")
for k, v in ctr.most_common(10):
    print(v, k)
