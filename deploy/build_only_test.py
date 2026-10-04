import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "tools"))

from mmcv import Config
import projects.mmdet3d_plugin  # noqa: F401 (registers plugin modules)
from mmdet.models import build_detector

cfg = Config.fromfile(os.path.join(ROOT, "projects/configs/sparsedrive_small_stage2.py"))
if cfg.get("custom_imports", None):
    from mmcv.utils import import_modules_from_strings
    import_modules_from_strings(**cfg["custom_imports"])
cfg.task_config["with_det"] = True
cfg.task_config["with_map"] = True
cfg.task_config["with_motion_plan"] = False
cfg.model.head.task_config = cfg.task_config

model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
model.eval()
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"built OK, params={n_params:.1f}M")
print("det head:", type(model.head.det_head).__name__)
print("map head:", type(model.head.map_head).__name__)
# check anchors loaded
ib = model.head.det_head.instance_bank
print("instance bank anchor:", getattr(ib.anchor, "shape", ib.anchor))
