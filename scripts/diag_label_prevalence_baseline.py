import random
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes

from src.data.clip_index import load_clip_index
from src.probes.object_presence_targets import clip_object_presence_multihot
from src.probes.object_presence_labels import LABELS

def macro_f1(y_true: torch.Tensor, y_prob: torch.Tensor, thr: float = 0.5) -> float:
    y_pred = (y_prob >= thr).float()
    eps = 1e-8
    f1s = []
    for k in range(y_true.shape[1]):
        tp = (y_pred[:, k] * y_true[:, k]).sum()
        fp = (y_pred[:, k] * (1 - y_true[:, k])).sum()
        fn = ((1 - y_pred[:, k]) * y_true[:, k]).sum()
        f1 = (2 * tp) / (2 * tp + fp + fn + eps)
        f1s.append(f1.item())
    return float(np.mean(f1s))

DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
items = load_clip_index(INDEX_PATH)

# Build labels for all clips
Y = []
for spec in items:
    y = clip_object_presence_multihot(nusc, spec.start_sd_token, spec.T, spec.stride, spec.keyframes_only)
    Y.append(torch.from_numpy(y).unsqueeze(0))
Y = torch.cat(Y, dim=0)  # [Nclips, K]

# deterministic split 80/20 like probe script
idxs = list(range(len(items)))
rng = random.Random(123)
rng.shuffle(idxs)
n_train = int(0.8 * len(idxs))
train_idx = idxs[:n_train]
val_idx = idxs[n_train:]

Ytr = Y[train_idx]
Yva = Y[val_idx]

# prevalence on train
prev = Ytr.float().mean(dim=0)  # probability each label is 1
print("train prevalence:")
for k, name in enumerate(LABELS):
    print(name, float(prev[k]))

# trivial baseline: predict prevalence probabilities for every val sample
Yprob = prev.view(1, -1).expand(Yva.shape[0], -1).clone()
f1 = macro_f1(Yva.float(), Yprob.float(), thr=0.5)
print("VAL size:", Yva.shape[0])
print("trivial prevalence baseline macro_f1:", f1)
