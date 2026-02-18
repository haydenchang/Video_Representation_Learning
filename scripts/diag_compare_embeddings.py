import random
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes

from src.model.mvm import MaskedVideoModel
from src.probes.probe_dataset import ObjectPresenceProbeDataset

DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"
CKPT_PATH = r"artifacts\checkpoints\mvm_day6_step50.pt"

device = torch.device("cpu")

ckpt = torch.load(CKPT_PATH, map_location="cpu")
N = ckpt["N"]; D = ckpt["D"]
t = ckpt["t"]; p = ckpt["p"]
d_model = ckpt["d_model"]

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)

def make_encoder(pretrained: bool) -> MaskedVideoModel:
    enc = MaskedVideoModel(N=N, D=D, d_model=d_model, n_layers=2, n_heads=8).to(device)
    if pretrained:
        enc.load_state_dict(ckpt["model_state"])
    enc.eval()
    return enc

enc_pre = make_encoder(True)
enc_rand = make_encoder(False)

ds_pre = ObjectPresenceProbeDataset(nusc, DATAROOT, INDEX_PATH, enc_pre, t=t, p=p, mask_ratio_visible=0.0, device=device)
ds_rand = ObjectPresenceProbeDataset(nusc, DATAROOT, INDEX_PATH, enc_rand, t=t, p=p, mask_ratio_visible=0.0, device=device)

# Compare embeddings on the same indices
idxs = [0, 1, 2, 10, 20]
for i in idxs:
    z_pre, y = ds_pre[i]
    z_rnd, _ = ds_rand[i]

    z_pre = z_pre.float()
    z_rnd = z_rnd.float()

    l2 = torch.norm(z_pre - z_rnd).item()
    cos = torch.nn.functional.cosine_similarity(z_pre, z_rnd, dim=0).item()

    print(f"idx {i}: l2={l2:.4f} cos={cos:.4f} y={y.tolist()}")

# Also sanity: pretrained embedding variability across samples
Z = []
for i in range(50):
    z, _ = ds_pre[i]
    Z.append(z.unsqueeze(0))
Z = torch.cat(Z, dim=0)  # [50, d_model]
print("pretrained z mean/std:", float(Z.mean()), float(Z.std()))
