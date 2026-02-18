import torch
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes

from src.data.clip_dataset import ClipDataset
from src.model.patchify import tubeletify
from src.model.masking import make_random_mask, masked_mse_loss
from src.model.mvm import MaskedVideoModel

DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"

t, p = 2, 40
N = (8 // t) * (360 // p) * (640 // p)  # 576
D = 3 * t * p * p                       # 9600

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
dataset = ClipDataset(nusc=nusc, dataroot=DATAROOT, index_path=INDEX_PATH)
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

batch = next(iter(loader))
video = batch["video"].to(device)  # [B,T,C,H,W]
tokens_gt = tubeletify(video, t=t, p=p)  # [B,N,D]

B, N2, D2 = tokens_gt.shape
assert N2 == N and D2 == D

gen = torch.Generator(device="cpu").manual_seed(123)

mask = make_random_mask(B=B, N=N, mask_ratio=0.9, device=device, generator=None)

# Build visible set
ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # [B,N]
ids_visible = ids[~mask].view(B, -1)                             # [B,Nv]
tokens_visible = tokens_gt[~mask].view(B, -1, D)                 # [B,Nv,D]

print("Nv:", tokens_visible.shape[1])

model = MaskedVideoModel(N=N, D=D, d_model=256, n_layers=2, n_heads=8).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

pred = model(tokens_visible, ids_visible)
print("pred shape:", tuple(pred.shape))

loss = masked_mse_loss(pred, tokens_gt, mask)
print("loss:", float(loss.detach()))

opt.zero_grad(set_to_none=True)
loss.backward()
opt.step()

print("OK: forward/backward/step")
