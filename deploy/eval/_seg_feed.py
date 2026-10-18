# -*- coding: utf-8 -*-
"""Build segment-level A/B feeds for the map-head precision experiment.

Inputs (must already exist):
  work_dirs/.../seg_map_f.onnx, seg_f32_names.txt   (deploy/_cut_map_seg.py)
  evaldata/feat_flat_f16.bin                        (real FPN, fp16,
      level-major [24,256,H,W]: 64x176, 32x88, 16x44, 8x22, 6 cams)
  evaldata/mini_eng_v8/outv8_80/next_map_*.bin      (real recurrent prev)
  deploy/segsend/meta/{projection_mat,instance_t_matrix,time_interval}.bin
      (pulled from board vec/mini/in_00 by _seg_ab.py stage meta)

Writes deploy/segsend/{shared,in_a,in_b}: shared holds the 4 fpn f32 bins +
3 meta bins; in_a/in_b hold only their prev_map bins + manifest.tsv (board
stage symlinks the shared files in). Case A = scene-start (prev zero),
case B = recurrent (prev from frame 80 chain state).
"""
import io
import os
import shutil

import numpy as np

sys_stdout_guard = io  # noqa: F401
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2")
EV = os.path.join(W, "evaldata")
SEND = os.path.join(ROOT, "deploy", "segsend")

LEVELS = [("fpn0", 64, 176), ("fpn1", 32, 88), ("fpn2", 16, 44),
          ("fpn3", 8, 22)]

FPN_NAMES = ["/img_neck/fpn_convs.%d/conv/Conv_output_0" % i
             for i in range(4)]

os.makedirs(os.path.join(SEND, "shared"), exist_ok=True)
os.makedirs(os.path.join(SEND, "in_a"), exist_ok=True)
os.makedirs(os.path.join(SEND, "in_b"), exist_ok=True)

# ---- model + names ----
shutil.copy(os.path.join(W, "seg_map_f.onnx"),
            os.path.join(SEND, "seg_map_f.onnx"))
shutil.copy(os.path.join(W, "seg_f32_names.txt"),
            os.path.join(SEND, "seg_f32_names.txt"))

# ---- FPN: split feat_flat (fp16) into per-level f32 bins ----
flat = np.fromfile(os.path.join(EV, "feat_flat_f16.bin"), np.float16)
expect = sum(6 * 256 * h * w for _, h, w in LEVELS)
assert flat.size == expect, (flat.size, expect)
print("feat_flat fp16:", flat.shape, "finite:", np.isfinite(flat).all())
off = 0
for (nm, h, w), gin in zip(LEVELS, FPN_NAMES):
    n = 6 * 256 * h * w
    blk = flat[off:off + n].astype(np.float32).reshape(6, 256, h, w)
    off += n
    p = os.path.join(SEND, "shared", nm + ".bin")
    blk.tofile(p)
    print("%s f32%s std=%.4f mean=%.4f absmax=%.2f" %
          (nm, blk.shape, blk.std(), blk.mean(), np.abs(blk).max()))
assert off == flat.size

# ---- meta bins (pulled from board) ----
for nm, nbyte in (("projection_mat", 1 * 6 * 4 * 4 * 4),
                  ("instance_t_matrix", 1 * 4 * 4 * 4),
                  ("time_interval", 4)):
    p = os.path.join(SEND, "meta", nm + ".bin")
    assert os.path.getsize(p) == nbyte, (nm, os.path.getsize(p), nbyte)
    shutil.copy(p, os.path.join(SEND, "shared", nm + ".bin"))
    a = np.fromfile(os.path.join(SEND, "shared", nm + ".bin"), np.float32)
    print("meta", nm, a.reshape(-1)[:6])

# ---- prev states: A zeros (scene start), B real recurrent (frame 80) ----
PREV = [("prev_map_feat", (1, 33, 256)), ("prev_map_anchor", (1, 33, 40)),
        ("prev_map_conf", (1, 33))]
for nm, shape in PREV:
    np.zeros(shape, np.float32).tofile(
        os.path.join(SEND, "in_a", nm + "_a.bin"))
    src = os.path.join(EV, "mini_eng_v8", "outv8_80",
                       nm.replace("prev", "next") + ".bin")
    b = np.fromfile(src, np.float32).reshape(shape)
    b.tofile(os.path.join(SEND, "in_b", nm + "_b.bin"))
    print("prev %s: A=0, B std=%.4f absmax=%.3f" % (nm, b.std(),
                                                    np.abs(b).max()))

# ---- manifests ----
def manifest(case, prev_tag):
    rows = []
    for (nm, h, w), gin in zip(LEVELS, FPN_NAMES):
        rows.append((gin, "f32", "6,256,%d,%d" % (h, w), nm + ".bin"))
    for nm, shape in PREV:
        rows.append((nm, "f32", ",".join(map(str, shape)),
                     nm + "_" + prev_tag + ".bin"))
    rows.append(("projection_mat", "f32", "1,6,4,4", "projection_mat.bin"))
    rows.append(("instance_t_matrix", "f32", "1,4,4", "instance_t_matrix.bin"))
    rows.append(("time_interval", "f32", "1", "time_interval.bin"))
    d = os.path.join(SEND, case)
    with open(os.path.join(d, "manifest.tsv"), "w", newline="\n") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")
    return rows

ra = manifest("in_a", "a")
rb = manifest("in_b", "b")
for nm, _, _, fa in ra:
    assert os.path.exists(os.path.join(SEND, "shared", fa)) or \
        os.path.exists(os.path.join(SEND, "in_a", fa)), nm
print("manifests written; segsend ready")
