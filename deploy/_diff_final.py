# -*- coding: utf-8 -*-
"""1-frame diff of final split configs (outs2b = bb2+hd, outs3b =
bb2+det+rest) vs outv8_00 (e_T6 single-engine reference)."""
import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
D = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2\evaldata")
SH = {"det_cls": (900, 10), "det_bbox": (900, 11), "det_quality": (900, 2),
      "map_cls": (100, 3), "map_pts": (100, 40),
      "det_instance_feature": (900, 256), "det_anchor_embed": (900, 256),
      "map_instance_feature": (100, 256), "map_anchor_embed": (100, 256),
      "next_det_feat": (600, 256), "next_det_anchor": (600, 11),
      "next_det_conf": (600,), "next_det_instance_id": (600,),
      "next_id_count": (1,), "next_map_feat": (33, 256),
      "next_map_anchor": (33, 40), "next_map_conf": (33,),
      "ego_feature_map": (1, 256, 8, 22)}
I32 = {"det_instance_id", "next_det_instance_id", "next_id_count"}


def ld(p, s, dt=np.float32):
    if not os.path.exists(p):
        return None
    a = np.fromfile(p, dt)
    return a.reshape(s) if a.size == int(np.prod(s)) else None


print(f"{'tensor':22s} {'2chain maxabs':>14s} {'2chain rel':>11s} "
      f"{'3chain maxabs':>14s} {'3chain rel':>11s}")
worst = 0
for name, s in SH.items():
    dt = np.int32 if name in I32 else np.float32
    v8 = ld(os.path.join(D, "mini_eng_v8", "outv8_00", name + ".bin"), s, dt)
    c2 = ld(os.path.join(D, f"outs2b_00_{name}.bin"), s, dt)
    c3 = ld(os.path.join(D, f"outs3b_00_{name}.bin"), s, dt)
    row = f"{name:22s}"
    for a in (c2, c3):
        if a is None or v8 is None:
            row += f" {'MISS':>14s} {'MISS':>11s}"
            continue
        dd = np.abs(a.astype(np.float64) - v8.astype(np.float64))
        den = max(np.abs(v8.astype(np.float64)).max(), 1e-9)
        if name not in I32:
            worst = max(worst, dd.max() / den)
        row += f" {dd.max():14.4e} {dd.max()/den:11.3e}"
    print(row)
print("\nworst float rel (2chain):", "%.3e" % worst)
