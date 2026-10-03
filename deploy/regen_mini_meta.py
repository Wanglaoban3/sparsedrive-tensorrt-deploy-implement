# -*- coding: utf-8 -*-
"""regenerate mini_meta.npz only (tokens/ts/boundary) without image prep."""
import copy
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "deploy"))
os.chdir(ROOT)

from mmcv import Config  # noqa: E402
import projects.mmdet3d_plugin  # noqa: E402,F401
from projects.mmdet3d_plugin.datasets.nuscenes_3d_dataset import (  # noqa
    NuScenes3DDataset,
)

CFG = "projects/configs/sparsedrive_small_stage2.py"
BASE = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2")
OUT = os.path.join(BASE, "evaldata", "mini_eng")

cfg = Config.fromfile(CFG)
if cfg.get("custom_imports", None):
    from mmcv.utils import import_modules_from_strings
    import_modules_from_strings(**cfg["custom_imports"])
ds_cfg = copy.deepcopy(cfg.data.val)
ds_cfg["test_mode"] = True
ds_cfg.pop("type", None)
ds = NuScenes3DDataset(**ds_cfg)
N = len(ds)
tokens, tss, boundary = [], [], []
for i in range(N):
    info = ds.data_infos[i]
    tokens.append(info["token"])
    ts = info["timestamp"] / 1e6
    if i == 0 or abs(ts - tss[-1]) > 5.0:
        boundary.append(i)
    tss.append(ts)
np.savez(os.path.join(OUT, "mini_meta.npz"),
         tokens=np.array(tokens), ts=np.array(tss),
         boundary=np.array(boundary))
print("MINI_META_DONE", N, "boundaries", boundary)
