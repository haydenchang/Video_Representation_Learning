import torch
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes

from src.data.clip_dataset import ClipDataset
from src.model.patchify import tubeletify
from src.model.masking import make_random_mask, masked_mse_loss
from src.model.baseline import BaselineReconstructor
from src.config import DATA_CFG

DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"

# Tubelet params (locked)
t, p = 2, 40
D = 3 * t * p * p

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
dataset = ClipDataset(nusc=nusc, dataroot=DATAROOT, index_path=INDEX_PATH)
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

model = BaselineReconstructor(D=D).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

gen = torch.Generator(device="cpu").manual_seed(123)

batch = next(iter(loader))
video = batch["video"].to(device)  # [B,T,C,H,W]
B, T, C, H, W = video.shape
print("batch video shape:", (B, T, C, H, W))

tokens_gt = tubeletify(video, t=t, p=p)  # [B,N,D]
B2, N, D2 = tokens_gt.shape
print("tokens shape:", (B2, N, D2))

mask = make_random_mask(B=B, N=N, mask_ratio=0.9, device=device, generator=None)  # non-deterministic ok for plumbing
pred = model(B=B, N=N, device=device)

loss = masked_mse_loss(pred, tokens_gt, mask)
print("loss:", loss.detach().item())

opt.zero_grad(set_to_none=True)
loss.backward()
opt.step()

print("OK: forward/backward/step")
