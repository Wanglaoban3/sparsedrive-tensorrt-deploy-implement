# -*- coding: utf-8 -*-
import glob
import io
import os
import sys

import numpy as np
import onnx
from onnx import shape_inference

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = r"<REPO>\work_dirs" \
    r"\sparsedrive_small_stage2"
p = os.path.join(W, "modwide_p2h.onnx")
m = onnx.load(p)
g = m.graph
print("boundary inputs:")
for vi in g.input:
    tt = vi.type.tensor_type
    dims = [d.dim_value for d in tt.shape.dim]
    print(f"  {vi.name}  dims={dims}")
print("boundary outputs:")
for vi in g.output:
    tt = vi.type.tensor_type
    dims = [d.dim_value for d in tt.shape.dim]
    print(f"  {vi.name}  dims={dims}")

print("== existing boundary npz in workdir ==")
known = {}
for f in glob.glob(os.path.join(W, "*.boundary.npz")) + \
        glob.glob(os.path.join(W, "mtq_v3_ref.npz.inputs.npz")):
    z = np.load(f)
    print(f"  {os.path.basename(f)}: {len(z.files)} arrays")
    for k in z.files:
        known.setdefault(k, (z[k].shape, str(z[k].dtype)))

zero = [vi.name for vi in list(g.input) + list(g.output)
        if any(d.dim_value == 0 for d in vi.type.tensor_type.shape.dim)]
print("tensors with 0-dims:", zero)
for t in zero:
    print(f"  {t}: known={known.get(t)}")
