# -*- coding: utf-8 -*-
"""Three-way diagnosis of the split chain."""
import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
D = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2\evaldata")

p3 = np.fromfile(os.path.join(D, "dump_out",
                              "Reshape_9_output_0.__dump.bin"), np.float32)
bb = np.fromfile(os.path.join(D, "outbb_00_Reshape_9_output_0.bin"),
                 np.float16).astype(np.float32)
print("== 1. bb boundary vs e_T6 internal (P3 f32 dump) ==")
d = np.abs(bb - p3)
print("maxabs %.4e  rel %.3e  mean|d| %.3e" %
      (d.max(), d.max() / np.abs(p3).max(), d.mean()))

SH = {"det_cls": (900, 10), "det_bbox": (900, 11), "det_quality": (900, 2),
      "map_cls": (100, 3), "map_pts": (100, 40),
      "det_instance_feature": (900, 256), "det_anchor_embed": (900, 256),
      "map_instance_feature": (100, 256), "map_anchor_embed": (100, 256),
      "next_det_feat": (600, 256), "next_det_anchor": (600, 11),
      "next_det_conf": (600,), "next_det_instance_id": (600,),
      "next_id_count": (1,), "next_map_feat": (33, 256),
      "next_map_anchor": (33, 40), "next_map_conf": (33,),
      "ego_feature_map": (1, 256, 8, 22)}


def ld(p, s, dt=np.float32):
    a = np.fromfile(p, dt)
    return a.reshape(s) if a.size == int(np.prod(s)) else None


I = np.iinfo(np.int32)
print("\n== 2. hd isolated (e_T6 real boundary) vs outv8_00 ==")
print("== 3. chained run (outs_00b) vs outv8_00 ==")
print(f"{'tensor':22s} {'iso maxabs':>12s} {'iso rel':>10s} "
      f"{'chain maxabs':>12s} {'chain rel':>10s}")
for name, s in SH.items():
    v8 = ld(os.path.join(D, "mini_eng_v8", "outv8_00", name + ".bin"), s)
    iso = ld(os.path.join(D, f"outhd_iso_00_{name}.bin"), s)
    ch = ld(os.path.join(D, f"outs_00b_{name}.bin"), s)
    r = f"{name:22s}"
    for a, tag in ((iso, "iso"), (ch, "chain")):
        if a is None or v8 is None:
            r += f" {'MISS':>12s} {'MISS':>10s}"
            continue
        dd = np.abs(a.astype(np.float64) - v8.astype(np.float64))
        den = max(np.abs(v8).max(), 1e-9)
        r += f" {dd.max():12.4e} {dd.max()/den:10.3e}"
    print(r)
