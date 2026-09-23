"""Create 20-byte LIDAR_TOP keyframe placeholder bins for the val scenes.

The converter (nuscenes_converter.py) requires each keyframe's LIDAR_TOP
file to exist (mmcv.check_file_exist) and nusc.get_sample_data reads the
point cloud; a 1-point (5 floats) bin satisfies both.  The eval pipeline
never reads lidar, so these placeholders are never used at inference.

Run AFTER blob extraction so real files (if any blob carried lidar) win:
    python tools/data_converter/make_lidar_placeholders.py
"""

import json
import os

ROOT = r"H:\datasets\nuscenes-trainval"
META = os.path.join(ROOT, "v1.0-trainval")

with open(os.path.join(META, "sample.json"), encoding="utf-8") as f:
    samples = json.load(f)
with open(os.path.join(META, "sample_data.json"), encoding="utf-8") as f:
    sds = json.load(f)
with open(os.path.join(META, "scene.json"), encoding="utf-8") as f:
    scenes = json.load(f)

from nuscenes.utils import splits as _splits  # noqa: E402

val_scenes = {s["token"] for s in scenes if s["name"] in set(_splits.val)}
val_sample_tokens = {s["token"] for s in samples
                     if s["scene_token"] in val_scenes}

placeholder = b"\x00" * 20  # one point, 5 float32
made = 0
for sd in sds:
    if (sd["is_key_frame"] and sd["sample_token"] in val_sample_tokens
            and sd["filename"].startswith("samples/LIDAR_TOP/")):
        dest = os.path.join(ROOT, sd["filename"])
        if not os.path.exists(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f2:
                f2.write(placeholder)
            made += 1
print(f"placeholders created: {made}")
