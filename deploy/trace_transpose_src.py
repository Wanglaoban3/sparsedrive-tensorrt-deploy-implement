# -*- coding: utf-8 -*-
"""probe: dump value_info entries (incl. duplicates) for the offending
tensor + the producer chain of the Transpose input in v5_P2h"""
import collections
import io
import os
import sys

import onnx
from onnx import TensorProto

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
SRC = os.path.join(W, "v5_P2h.onnx")
m = onnx.load(SRC)
g = m.graph
DTC = {TensorProto.FLOAT: "f32", TensorProto.FLOAT16: "f16"}

# duplicate value_info names
cnt = collections.Counter(vi.name for vi in g.value_info)
dups = {k: v for k, v in cnt.items() if v > 1}
print(f"value_info entries: {len(g.value_info)}, "
      f"duplicate names: {len(dups)}")
for k in list(dups)[:10]:
    print(f"  dup x{dups[k]}: {k}")

TARGET = "/layers.0/kps_generator/learnable_fc/Transpose_output_0"
for vi in g.value_info:
    if vi.name == TARGET:
        print(f"value_info: {TARGET} -> "
              f"{DTC.get(vi.type.tensor_type.elem_type, vi.type.tensor_type.elem_type)}")

prod = {}
for n in g.node:
    for o in n.output:
        prod[o] = n
init_dt = {i.name: i.data_type for i in g.initializer}

t = TARGET
for depth in range(12):
    p = prod.get(t)
    if p is None:
        print(f"{'  '*depth}{t}  [no producer; "
              f"{'INIT dt=' + str(init_dt.get(t)) if t in init_dt else 'graph in/out?'}]")
        break
    ins = []
    for x in p.input:
        if x in init_dt:
            ins.append(f"{x}[INIT dt={init_dt[x]}]")
        else:
            ins.append(x)
    print(f"{'  '*depth}{t}  <- {p.op_type} {p.name} ({ins})")
    nxt = [x for x in p.input if x not in init_dt]
    if not nxt:
        break
    t = nxt[0]
