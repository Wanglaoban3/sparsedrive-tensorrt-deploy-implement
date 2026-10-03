# -*- coding: utf-8 -*-
"""probe: why did the layers.0 QDQ pairs get skipped by the strip rule"""
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
graph_ins = {v.name for v in g.input}

names = ["/layers.0/kps_generator/learnable_fc/input_quantizer/QuantizeLinear",
         "/layers.0/attn/v_proj/input_quantizer/QuantizeLinear",
         "/layers.0/attn/q_proj/input_quantizer/QuantizeLinear"]
by_name = {n.name: n for n in g.node}
for nm in names:
    q = by_name[nm]
    print("==", nm)
    x = q.input[0]
    if x in init_names:
        print(f"  Q.input[0] {x}  [INITIALIZER]")
    elif x in graph_ins:
        print(f"  Q.input[0] {x}  [GRAPH INPUT]")
    else:
        p = prod.get(x)
        print(f"  Q.input[0] {x}  <- "
              f"{p.op_type + ' ' + p.name if p else 'NO PRODUCER'}")
    for k in range(1, len(q.input)):
        xx = q.input[k]
        print(f"  Q.input[{k}] {xx}  "
              f"[{'INIT' if xx in init_names else 'tensor'}]")
    for d in cons.get(q.output[0], []):
        print(f"  consumer: {d.op_type} {d.name}")
        for k, xx in enumerate(d.input):
            print(f"    DQ.input[{k}] {xx}  "
                  f"[{'INIT' if xx in init_names else 'tensor'}]")
