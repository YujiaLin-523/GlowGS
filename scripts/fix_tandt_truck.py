import json
from pathlib import Path
from PIL import Image

root = Path("data/tandt/truck")  # Path to the nerfstudio format tandt directory
tf_path = root / "transforms.json"

with open(tf_path, "r") as f:
    tf = json.load(f)

W_meta, H_meta = tf["w"], tf["h"]

# Read the resolution of a real image
first_frame = tf["frames"][0]
img_path = root / first_frame["file_path"]  # e.g., images/00001.jpg
with Image.open(img_path) as im:
    W_img, H_img = im.size

sx = W_img / W_meta
sy = H_img / H_meta

# Simple sanity check: ensure x and y scales are consistent
if abs(sx - sy) > 1e-3:
    raise ValueError("X and Y scale factors are inconsistent. Please check manually.")

scale = (sx + sy) / 2

# Update global resolution
tf["w"] = W_img
tf["h"] = H_img

# Scale intrinsic parameters proportionally
for k in ["fl_x", "fl_y", "cx", "cy"]:
    if k in tf:
        tf[k] *= scale

# Save backup and overwrite original
backup_path = tf_path.with_suffix(".orig.json")
tf_path.rename(backup_path)
with open(tf_path, "w") as f:
    json.dump(tf, f, indent=4)

print(f"Fixed: scale factor = {scale:.4f}")
