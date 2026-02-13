from pathlib import Path
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes

from src.data.clip_dataset import ClipDataset

DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"
OUT_DIR = Path(r"artifacts\viz_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
dataset = ClipDataset(nusc=nusc, dataroot=DATAROOT, index_path=INDEX_PATH)

idx = 0
item = dataset[idx]
video = item["video"]  # [T, C, H, W], float [0,1]

T, C, H, W = video.shape
assert C == 3

imgs = []
for t in range(T):
    x = video[t].cpu().numpy()  # [C,H,W]
    x = np.transpose(x, (1, 2, 0))  # [H,W,C]
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    imgs.append(Image.fromarray(x))

canvas = Image.new("RGB", (W * T, H))
for i, im in enumerate(imgs):
    canvas.paste(im, (i * W, 0))

out_path = OUT_DIR / f"dataset_clip_{idx:03d}_T{T}.png"
canvas.save(out_path)
print("saved:", out_path)
print("start_token:", item["start_token"])
print("first/last filename:", item["filenames"][0], item["filenames"][-1])
