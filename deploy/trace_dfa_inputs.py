# -*- coding: utf-8 -*-
"""查 v5_P1h.onnx 中 DFA input[3]/[4] 的生产者链 (是否 Softmax),
以及图里有没有 softmax。"""
import io
import os
import sys

import onnx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
m = onnx.load(os.path.join(W, "v5_P1h.onnx"))
g = m.graph

prod = {}
for n in g.node:
    for o in n.output:
        prod[o] = n

# 全图 softmax 节点
sms = [n for n in g.node if n.op_type in ("Softmax",)]
print("Softmax nodes in graph:", len(sms))
for n in sms[:3]:
    print("  ", n.name, list(n.input), "->", list(n.output),
          "axes=", [a.i for a in n.attribute] if n.attribute else None)

dfas = [n for n in g.node if n.op_type == "DeformableAggregation"]
n0 = dfas[0]
for pos in (3, 4):
    t = n0.input[pos]
    chain = [t]
    cur = t
    for _ in range(6):
        n = prod.get(cur)
        if n is None:
            chain.append("<graph-input>")
            break
        chain.append(f"{n.op_type}({n.name})")
        cur = n.input[0] if n.input else None
        if cur is None:
            break
    print(f"input[{pos}] chain:", " <- ".join(chain))
