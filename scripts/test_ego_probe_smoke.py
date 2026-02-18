import torch
from nuscenes.nuscenes import NuScenes
from src.model.mvm import MaskedVideoModel
from src.probes.ego_motion_probe_dataset import EgoMotionProbeDataset
from src.probes.ego_motion_labels import EGO_LABELS

DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"

t, p = 2, 40
N = (8 // t) * (360 // p) * (640 // p)
D = 3 * t * p * p

device = torch.device("cpu")

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
encoder = MaskedVideoModel(N=N, D=D, d_model=256, n_layers=2, n_heads=8).to(device)
encoder.eval()

ds = EgoMotionProbeDataset(
    nusc=nusc,
    dataroot=DATAROOT,
    index_path=INDEX_PATH,
    encoder_model=encoder,
    t=t,
    p=p,
    device=device,
)

z, y = ds[0]
print("z shape:", tuple(z.shape))
print("ego label:", [EGO_LABELS[i] for i in range(len(EGO_LABELS)) if y[i] > 0.5])
print("onehot:", y.tolist())
