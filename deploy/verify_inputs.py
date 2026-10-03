# -*- coding: utf-8 -*-
"""verify input alignment: npz refs vs board vec bins"""
import csv
import numpy as np

BASE = (r"<REPO>\work_dirs"
        r"\sparsedrive_small_stage2")

fi = np.load(BASE + r"\mtq_v3_ref.npz.inputs.npz")
ff = np.load(BASE + r"\mtq_f32_ref.npz.inputs.npz")
fv = np.load(BASE + r"\mtq_v3_ref.npz")
ffo = np.load(BASE + r"\mtq_f32_ref.npz")
print("mtq_v3 inputs keys:", len(fi.files), "| mtq_f32 inputs keys:",
      len(ff.files))
print("mtq_v3 outputs:", sorted(fv.files)[:5], "...")
same = True
for k in fi.files:
    if k in ff.files:
        a, b = fi[k], ff[k]
        if a.shape != b.shape or not np.array_equal(a, b):
            same = False
            d = float(np.abs(a.astype(np.float64)
                             - b.astype(np.float64)).max()) \
                if a.shape == b.shape else -1
            print(f"  DIFF {k}: shape {a.shape} vs {b.shape}, maxd {d}")
print("inputs identical (v3 vs f32):", same)

# board vec inputs vs npz
import csv


def load_dir(d):
    out = {}
    with open(d + r"\manifest.tsv") as f:
        for row in csv.reader(f, delimiter="\t"):
            if not row:
                continue
            name, dt, shape, fn = row[0], row[1], row[2], row[3]
            shape = tuple(int(x) for x in shape.split(","))
            a = np.fromfile(d + "\\" + fn,
                            dtype=np.float32 if dt == "f32" else np.int32)
            out[name] = a.reshape(shape)
    return out


VB = load_dir(BASE + r"\evaldata\inputs")
same2 = True
for k in fi.files:
    if k in VB:
        if not np.array_equal(fi[k], VB[k]):
            same2 = False
            print(f"  board-vs-npz DIFF {k}: "
                  f"maxd {float(np.abs(fi[k].astype(np.float64) - VB[k].astype(np.float64)).max())}")
print("board vec/inputs_chain2_mtq == mtq_v3 inputs:", same2)

print()
print("npz f32 ref outputs:", sorted(ffo.files))
