import torch
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes

from src.data.clip_dataset import ClipDataset
from src.model.patchify import tubeletify
from src.model.mvm import MaskedVideoModel

DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"
CKPT_PATH = r"artifacts\checkpoints\mvm_day6_step50.pt"

device = torch.device("cpu")

t, p = 2, 40
T, H, W = 8, 360, 640
N = (T // t) * (H // p) * (W // p)
D = 3 * t * p * p
d_model = 256

ckpt = torch.load(CKPT_PATH, map_location="cpu")

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
ds = ClipDataset(nusc=nusc, dataroot=DATAROOT, index_path=INDEX_PATH)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

# model
enc = MaskedVideoModel(N=N, D=D, d_model=d_model, n_layers=2, n_heads=8).to(device)
enc.load_state_dict(ckpt["model_state"], strict=False)
enc.eval()

batch = next(iter(loader))
video = batch["video"].to(device)              # [1,T,C,H,W]
video_zero = torch.zeros_like(video)

def get_z(v):
    tokens = tubeletify(v, t=t, p=p)           # [1,N,D]
    ids = torch.arange(N, device=device).unsqueeze(0)  # [1,N]
    with torch.no_grad():
        x = enc.encode_visible(tokens, ids)    # may include CLS
        # use token pooling consistent with your current setup:
        if x.shape[1] == tokens.shape[1] + 1:
            z = x[:, 0, :]                     # CLS
        else:
            z = x.mean(dim=1)
    return z.squeeze(0)

z_real = get_z(video)
z_zero = get_z(video_zero)

# compare
l2 = torch.norm(z_real - z_zero).item()
cos = torch.nn.functional.cosine_similarity(
    z_real.unsqueeze(0), z_zero.unsqueeze(0), dim=1
).item()

print("z_real mean/std:", z_real.mean().item(), z_real.std(unbiased=False).item())
print("z_zero mean/std:", z_zero.mean().item(), z_zero.std(unbiased=False).item())
print("real vs zero: l2 =", l2, "cos =", cos)
