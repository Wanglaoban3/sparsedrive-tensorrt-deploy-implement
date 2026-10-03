# -*- coding: utf-8 -*-
"""probe: what actually feeds the 4 mixed layers.0 MatMuls (P1h graph)"""
import collections
import io
import os
import sys

import onnx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
SRC = os.path.join(W, "v5_P1h.onnx")
m = onnx.load(SRC)
g = m.graph

prod = {}
cons = collections.defaultdict(list)
for n in g.node:
    for o in n.output:
        prod[o] = n
    for x in n.input:
        if x:
            cons[x].append(n)
init_names = {i.name for i in g.initializer}
inits = {i.name: i for i in g.initializer}
graph_ins = {v.name for v in g.input}


def src_of(x, depth=0):
    pad = "  " * (depth + 1)
    if x in init_names:
        dt = inits[x].data_type
        print(f"{pad}{x}  [INIT dt={dt}]")
        return
    if x in graph_ins:
        print(f"{pad}{x}  [GRAPH-IN]")
        return
    p = prod.get(x)
    if p is None:
        print(f"{pad}{x}  [NO PRODUCER]")
        return
    print(f"{pad}{x}  <- {p.op_type} {p.name}")
    if depth < 3 and p.op_type in ("DequantizeLinear", "Cast", "Identity"):
        src_of(p.input[0], depth + 1)


targets = ["/layers.0/kps_generator/learnable_fc/MatMul",
           "/layers.0/attn/v_proj/MatMul",
           "/layers.0/attn/q_proj/MatMul",
           "/layers.0/attn/k_proj/MatMul"]
by_name = {n.name: n for n in g.node}
for t in targets:
    n = by_name[t]
    print("==", t)
    for k, x in enumerate(n.input):
        print(f"  in[{k}]:")
        src_of(x)

print()
print("QuantizeLinear nodes under /layers.0:",
      [n.name for n in g.node
       if n.op_type == "QuantizeLinear" and "/layers." in n.name])
print("DequantizeLinear nodes under /layers.0:",
      [n.name for n in g.node
       if n.op_type == "DequantizeLinear" and "/layers." in n.name])
dq_users = {n.output[0]: [c.name for c in cons[n.output[0]]]
            for n in g.node if n.op_type == "DequantizeLinear"
            and "/layers." in n.name}
for k, v in dq_users.items():
    print(f"DQ {k} -> {v}")
