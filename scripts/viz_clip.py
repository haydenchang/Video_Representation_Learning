from pathlib import Path

from PIL import Image
from nuscenes.nuscenes import NuScenes

from src.data.clip_index import load_clip_index
from src.data.clip_sampler import sample_clip_from_start

DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"
OUT_DIR = Path(r"artifacts\viz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
items = load_clip_index(INDEX_PATH)

# Pick a few clips to visualize
pick = [0, 1, 2]

for idx in pick:
    spec = items[idx]
    clip = sample_clip_from_start(
        nusc,
        start_sd_token=spec.start_sd_token,
        T=spec.T,
        stride=spec.stride,
        keyframes_only=spec.keyframes_only,
    )

    imgs = []
    for fr in clip.frames:
        path = Path(DATAROOT) / fr.filename
        img = Image.open(path).convert("RGB")
        imgs.append(img)

    # Make a horizontal strip
    w, h = imgs[0].size
    canvas = Image.new("RGB", (w * len(imgs), h))
    for i, im in enumerate(imgs):
        canvas.paste(im, (i * w, 0))

    out_path = OUT_DIR / f"clip_{idx:03d}_T{spec.T}_s{spec.stride}.png"
    canvas.save(out_path)
    print("saved:", out_path)
