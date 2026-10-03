# -*- coding: utf-8 -*-
"""probe the 4 remaining mixed MatMuls' input producers in v5_P2h"""
import io
import sys

import onnx
from onnx import TensorProto

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
m = onnx.load(W + r"\v5_P2h.onnx")
g = m.graph
prod = {}
for n in g.node:
    for o in n.output:
        prod[o] = n

TARGETS = ["/layers.0/attn/q_proj/MatMul",
           "/layers.0/kps_generator/learnable_fc/MatMul"]
for tname in TARGETS:
    n = next((n for n in g.node if n.name == tname), None)
    if n is None:
        print(f"{tname}: NOT FOUND")
        continue
    print(f"== {tname}")
    for x in n.input:
        p = prod.get(x)
        if p is None:
            print(f"   in {x}  (initializer/graph input)")
            continue
        attrs = ""
        if p.op_type == "Cast":
            attrs = f" to={next(a.i for a in p.attribute if a.name == 'to')}"
        print(f"   in {x}  <- {p.op_type} {p.name}{attrs}"
              f"  inputs={[list(p.input)[:3]]}")
