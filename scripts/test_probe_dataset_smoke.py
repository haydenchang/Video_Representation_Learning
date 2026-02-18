import torch
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

encoder = MaskedVideoModel(N=N, D=D, d_model=256, n_layers=2, n_heads=8).to(device)
encoder.eval()

ds = ObjectPresenceProbeDataset(
    nusc=nusc,
    dataroot=DATAROOT,
    index_path=INDEX_PATH,
    encoder_model=encoder,
    t=t,
    p=p,
    mask_ratio_visible=0.0,
    device=device,
)

z, y = ds[0]
print("z shape:", tuple(z.shape))
print("y:", y.tolist())
