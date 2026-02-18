from pathlib import Path
from PIL import Image
from src.data.transforms import resize_to_tensor_float01

DATAROOT = r"C:\DS\TPV\nuScenes"
img_path = Path(DATAROOT) / r"samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530412470.jpg"

img = Image.open(img_path)
x = resize_to_tensor_float01(img, out_h=360, out_w=640)
print("shape:", tuple(x.shape))
print("min/max:", float(x.min()), float(x.max()))
