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

# tubelet params
t, p = 2, 40
N = (8 // t) * (360 // p) * (640 // p)
D = 3 * t * p * p
d_model = 256

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)

ckpt = torch.load(CKPT_PATH, map_location="cpu")

def make_encoder(pretrained: bool):
    enc = MaskedVideoModel(N=N, D=D, d_model=d_model, n_layers=2, n_heads=8).to(device)
    if pretrained:
        missing, unexpected = enc.load_state_dict(ckpt["model_state"], strict=False)
        print("load_state_dict missing:", missing)
        print("load_state_dict unexpected:", unexpected)
    enc.eval()
    return enc

def collect_z(pretrained: bool, n_samples: int = 200, seed: int = 123):
    enc = make_encoder(pretrained=pretrained)
    ds = ObjectPresenceProbeDataset(
        nusc=nusc,
        dataroot=DATAROOT,
        index_path=INDEX_PATH,
        encoder_model=enc,
        t=t,
        p=p,
        mask_ratio_visible=0.0,
        device=device,
    )

    idxs = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    idxs = idxs[:min(n_samples, len(idxs))]

    Z = []
    for i in idxs:
        z, y = ds[i]
        Z.append(z.numpy())
    Z = np.stack(Z, axis=0)  # [M, d_model]
    return Z

def summarize(Z: np.ndarray, name: str):
    # feature variance across samples
    per_dim_std = Z.std(axis=0)
    print(f"{name}: Z shape:", Z.shape)
    print(f"{name}: mean(per-dim std):", float(per_dim_std.mean()))
    print(f"{name}: min/max per-dim std:", float(per_dim_std.min()), float(per_dim_std.max()))

    # cosine similarity between random pairs
    Zt = torch.from_numpy(Z).float()
    Zt = torch.nn.functional.normalize(Zt, dim=1)
    M = Zt.shape[0]
    pairs = 300
    rng = random.Random(999)
    coss = []
    for _ in range(pairs):
        a = rng.randrange(M)
        b = rng.randrange(M)
        if a == b:
            continue
        coss.append(float((Zt[a] * Zt[b]).sum().item()))
    print(f"{name}: cosine sim mean/std/min/max:", float(np.mean(coss)), float(np.std(coss)), float(np.min(coss)), float(np.max(coss)))

Z_pre = collect_z(pretrained=True, n_samples=200)
Z_rand = collect_z(pretrained=False, n_samples=200)

summarize(Z_pre, "pretrained")
summarize(Z_rand, "random")
