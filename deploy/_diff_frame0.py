# -*- coding: utf-8 -*-
"""1-frame diff: split-pipeline outs_00 vs engine baselines (out_00 = v3,
outv8_00 = v8). Reads outs_00_*.bin files fetched by board_split_poll."""
import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
BASE = (r"<REPO>"
        r"\work_dirs\sparsedrive_small_stage2\evaldata")
NEW = os.path.join(BASE, "mini_eng_v8")          # outs_00_*.bin live here
V8 = os.path.join(BASE, "mini_eng_v8", "outv8_00")
V3 = os.path.join(BASE, "mini_eng", "out_00")

SHAPES = {"det_cls": (900, 10), "det_bbox": (900, 11),
          "det_quality": (900, 2), "map_cls": (100, 3), "map_pts": (100, 40),
          "det_instance_feature": (900, 256), "det_anchor_embed": (900, 256),
          "map_instance_feature": (100, 256), "map_anchor_embed": (100, 256),
          "next_det_feat": (600, 256), "next_det_anchor": (600, 11),
          "next_det_conf": (600,), "next_det_instance_id": (600,),
          "next_id_count": (1,), "next_map_feat": (33, 256),
          "next_map_anchor": (33, 40), "next_map_conf": (33,),
          "ego_feature_map": (1, 256, 8, 22)}
DT = {"det_instance_id": np.int32, "next_det_instance_id": np.int32,
      "next_id_count": np.int32}


def load(path, name):
    if not os.path.exists(path):
        return None
    dt = DT.get(name, np.float32)
    n = np.prod(SHAPES[name])
    a = np.fromfile(path, dt)
    if a.size != n:
        print(f"  {name}: size {a.size} != {n}, skip")
        return None
    return a.reshape(SHAPES[name])


print(f"{'tensor':24s} {'vs_v8 maxabs':>14s} {'vs_v8 rel':>11s} "
      f"{'vs_v3 maxabs':>14s} {'vs_v3 rel':>11s}")
for name, shape in SHAPES.items():
    a = load(os.path.join(NEW, f"outs_00_{name}.bin"), name)
    if a is None:
        print(f"{name:24s} MISSING")
        continue
    row = f"{name:24s}"
    for tag, base in (("v8", os.path.join(V8, name + ".bin")),
                      ("v3", os.path.join(V3, name + ".bin"))):
        b = load(base, name)
        if b is None:
            row += f" {'n/a':>14s} {'n/a':>11s}"
            continue
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        den = max(np.abs(b).max(), 1e-9)
        row += f" {d.max():14.4e} {d.max()/den:11.3e}"
    print(row)
