# -*- coding: utf-8 -*-
"""Strip identity Cast nodes (in-dtype == out-dtype) — pure passthroughs
that ride multi-MB tensors (SSA Cast_4/Cast_5 on [1,8,900,900] f16)."""
import io
import os
import sys

import onnx
from onnx import TensorProto

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
SRC = os.path.join(W, "sp_head_det.onnx")
DST = os.path.join(W, "sp_head_det2.onnx")

m = onnx.load(SRC)
g = m.graph

def dtype_of(name):
    for src in (g.value_info, g.input, g.output):
        for v in src:
            if v.name == name:
                return v.type.tensor_type.elem_type
    return None

mi = onnx.shape_inference.infer_shapes(m)
def dtype_of(name):
    for src in (g.value_info, g.input, g.output,
                mi.graph.value_info, mi.graph.input, mi.graph.output):
        for v in src:
            if v.name == name:
                return v.type.tensor_type.elem_type
    return None

prod = {}
for i, n in enumerate(g.node):
    for o in n.output:
        prod[o] = i

removed = 0
bytes_saved = 0
for n in list(g.node):
    if n.op_type != "Cast" or len(n.input) != 1:
        continue
    to = next(a.i for a in n.attribute if a.name == "to")
    tin = dtype_of(n.input[0])
    if tin is None or to != tin:
        continue
    src_t = n.input[0]
    # graph outputs that point at this cast move to its source tensor
    for k in range(len(g.output)):
        if g.output[k].name == n.output[0]:
            old = g.output[k]
            new = onnx.ValueInfoProto()
            new.CopyFrom(old)
            new.name = src_t
            g.output[k].CopyFrom(new)
    for c in g.node:
        for k, x in enumerate(c.input):
            if x == n.output[0]:
                c.input[k] = src_t
    g.node.remove(n)
    removed += 1
    dims = []
    for src in (g.value_info, mi.graph.value_info):
        for v in src:
            if v.name == src_t:
                dims = [d.dim_value for d in
                        v.type.tensor_type.shape.dim]
    nbytes = 1
    for d in dims:
        nbytes *= max(d, 1)
    bytes_saved += nbytes * (2 if tin == TensorProto.FLOAT16 else 4)

print(f"removed {removed} identity casts, ~{bytes_saved/1e6:.1f}MB per-iter "
      f"tensor traffic eliminated (write+read x2 => "
      f"~{bytes_saved*2/1e6:.0f}MB)")
onnx.checker.check_model(m, False)
onnx.save(m, DST)
print("saved", DST)
