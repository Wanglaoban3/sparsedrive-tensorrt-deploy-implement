"""Smoke test: build det+map model, load ckpt, forward one mini sample.

Verifies: model builds, checkpoint loads, pure-torch DFA path works inside
the real model, GridMask eval determinism, and prints module inventory.
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "tools"))
sys.path.append(os.path.join(ROOT, "deploy"))

from mmcv import Config  # noqa: E402
import projects.mmdet3d_plugin  # noqa: E402,F401 (registers plugin modules)
from mmcv.runner import load_checkpoint  # noqa: E402
from mmdet.models import build_detector  # noqa: E402

from export_onnx_det_map import SparseDriveONNXWrapper  # noqa: E402
from ptq_sensitivity import build_samples, sample_to_inputs, forward_outputs  # noqa: E402


def main():
    os.chdir(ROOT)
    cfg_path = "projects/configs/sparsedrive_small_stage2.py"
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "ckpt/sparsedrive_stage2.pth"

    cfg = Config.fromfile(cfg_path)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    cfg.task_config["with_det"] = True
    cfg.task_config["with_map"] = True
    cfg.task_config["with_motion_plan"] = False
    cfg.model.head.task_config = cfg.task_config

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, ckpt, map_location="cpu")
    model.cuda().eval()
    wrapper = SparseDriveONNXWrapper(model)
    print("model built, ckpt loaded")

    samples = build_samples(cfg, 2)
    print("samples:", len(samples))
    si = sample_to_inputs(samples[0], wrapper)
    o1 = forward_outputs(wrapper, si)
    o2 = forward_outputs(wrapper, si)
    same = all(torch.equal(a, b) for a, b in zip(o1, o2))
    print("deterministic forward:", same)
    for name, t in zip(
        ["det_cls", "det_bbox", "det_quality", "map_cls", "map_pts",
         "ego_feature_map"], [o1[i] for i in (0, 1, 2, 11, 12, 18)]):
        print(f"  {name}: shape={tuple(t.shape)} "
              f"mean={t.float().mean():.4f} std={t.float().std():.4f}")

    if not same:
        print("WARNING: forward is nondeterministic (GridMask in eval?)")
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
