import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes

from src.data.clip_dataset import ClipDataset
from src.model.patchify import tubeletify
from src.model.masking import make_random_mask, masked_mse_loss
from src.model.mvm import MaskedVideoModel

# ---------- Determinism ----------
def seed_all(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

seed_all(123)

# ---------- Config ----------
DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"

t, p = 2, 40
T, H, W = 8, 360, 640
N = (T // t) * (H // p) * (W // p)  # 576
D = 3 * t * p * p                  # 9600
mask_ratio = 0.90

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ---------- Data ----------
nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
dataset = ClipDataset(nusc=nusc, dataroot=DATAROOT, index_path=INDEX_PATH)

# small, deterministic loader
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=True)

# ---------- Model ----------
model = MaskedVideoModel(N=N, D=D, d_model=256, n_layers=2, n_heads=8).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

# Mask RNG (seeded)
mask_gen = torch.Generator(device="cpu").manual_seed(999)

def collapse_signals(pred_tokens: torch.Tensor) -> dict:
    # pred_tokens: [B,N,D]
    with torch.no_grad():
        # token-wise mean/std (rough sanity; not a definitive collapse metric)
        std = pred_tokens.float().std().item()
        mean = pred_tokens.float().mean().item()
    return {"pred_mean": mean, "pred_std": std}

# ---------- Loop ----------
model.train()
max_steps = 50

it = iter(loader)
for step in range(max_steps):
    try:
        batch = next(it)
    except StopIteration:
        it = iter(loader)
        batch = next(it)

    video = batch["video"].to(device)              # [B,T,C,H,W]
    tokens_gt = tubeletify(video, t=t, p=p)        # [B,N,D]

    B = tokens_gt.shape[0]
    # generate on CPU for deterministic behavior across devices, then move to device
    mask_cpu = make_random_mask(B=B, N=N, mask_ratio=mask_ratio, device=torch.device("cpu"), generator=mask_gen)
    mask = mask_cpu.to(device)

    ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # [B,N]
    ids_visible = ids[~mask].view(B, -1)                             # [B,Nv]
    tokens_visible = tokens_gt[~mask].view(B, -1, D)                 # [B,Nv,D]

    pred = model(tokens_visible, ids_visible)                        # [B,N,D]
    loss = masked_mse_loss(pred, tokens_gt, mask)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if step % 10 == 0 or step == max_steps - 1:
        sig = collapse_signals(pred)
        print(f"Step {step} (Day 5): loss={loss.item():.6f} pred_std={sig['pred_std']:.6f} pred_mean={sig['pred_mean']:.6f}")

print("DONE")
