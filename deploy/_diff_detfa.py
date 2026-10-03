"""A/B: det module outputs, baseline (e_det) vs ssa (e_det_fa).
Sentinels (upstream of attention) must match exactly; downstream diffs
expected small (plugin f32 softmax vs P2h f16-graph softmax)."""
import struct
import sys
from pathlib import Path

import numpy as np

REF = Path(r"<REPO>"
           r"\work_dirs\sparsedrive_small_stage2\outs_detref_00")
SSA = Path(r"<REPO>"
           r"\work_dirs\sparsedrive_small_stage2\outs_detfa2_00")

DT = {"f32": (np.float32, 4), "f16": (np.float16, 2), "b8": (np.int8, 1),
      "i32": (np.int32, 4), "i64": (np.int64, 8), "bool": (np.bool_, 1)}


def load_dir(d):
    out = {}
    for mf in d.glob("manifest.tsv"):
        for line in mf.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            name, dt, dims, fn = parts
            t, item = DT[dt]
            a = np.frombuffer((d / fn.lstrip("/")).read_bytes(), dtype=t)
            dims = [int(x) for x in dims.split(",") if x]
            out[name] = a.reshape(dims) if dims else a
    return out


ref, ssa = load_dir(REF), load_dir(SSA)
assert set(ref) == set(ssa), \
    f"tensor sets differ: {set(ref) ^ set(ssa)}"
print(f"{len(ref)} tensors")

EXACT = {"/LessOrEqual_output_0", "instance_t_matrix_to16",
         "/layers.0/MatMul_output_0_w2th", "/layers.0/Reshape_output_0"}
worst = 0.0
for name in sorted(ref):
    a, b = ref[name].astype(np.float64), ssa[name].astype(np.float64)
    if a.size == 0:
        continue
    den = np.linalg.norm(a)
    l2 = np.linalg.norm(a - b) / (den if den else 1.0)
    mx = np.abs(a - b).max() if a.size else 0.0
    tag = "EXACT-OK" if np.array_equal(a, b) else (
        "EXACT-FAIL" if name in EXACT else "diff")
    if name in EXACT and not np.array_equal(a, b):
        worst = max(worst, 1e9)
    print(f"  {name:42s} l2rel={l2:.3e} maxabs={mx:.3e} {tag}")
print(f"worst sentinel violation: {worst}")
