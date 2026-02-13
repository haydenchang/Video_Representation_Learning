from nuscenes.nuscenes import NuScenes
from torch.utils.data import DataLoader

from src.data.clip_dataset import ClipDataset

DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)

dataset = ClipDataset(
    nusc=nusc,
    dataroot=DATAROOT,
    index_path=INDEX_PATH,
)

print("dataset size:", len(dataset))

item = dataset[0]
video = item["video"]

print("video shape:", tuple(video.shape))
print("dtype:", video.dtype)
print("min/max:", float(video.min()), float(video.max()))

# test batching
loader = DataLoader(dataset, batch_size=2, shuffle=False)
batch = next(iter(loader))
print("batch video shape:", tuple(batch["video"].shape))
