# -*- coding: utf-8 -*-
"""Who consumes the boundary tensor in sp_head, and does the ego chain
depend on it? Also: sanity-check the P3 dump provenance."""
import collections
import io
import os
import sys

import onnx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
m = onnx.load(os.path.join(W, "sp_head.onnx"))
g = m.graph
nodes = list(g.node)
prod, cons = {}, collections.defaultdict(list)
for i, n in enumerate(nodes):
    for o in n.output:
        prod[o] = i
    for x in n.input:
        if x:
            cons[x].append(i)
inits = {x.name for x in g.initializer}

B = "/Reshape_9_output_0"
print("head consumers of", B, "=", len(cons[B]))
for j in cons[B]:
    print(f"  [{j}] {nodes[j].op_type} {nodes[j].name!r}")

# upward closure of ego_feature_map producer: what graph inputs does it need?
target = "ego_feature_map"
need = set()
stack = [target]
need_t = set()
while stack:
    t = stack.pop()
    p = prod.get(t)
    if p is None or p in need:
        continue
    need.add(p)
    for x in nodes[p].input:
        if x and x not in inits:
            stack.append(x)
            need_t.add(x)
print("\nego chain nodes:", len(need))
ext = sorted(t for t in need_t if prod.get(t) is None)
print("ego external inputs (non-init, no producer in head):", ext)
reach_b = B in need_t or any(prod.get(t) is not None and
                             prod[t] in need for t in cons[B])
print("ego chain touches boundary consumers:",
      [nodes[j].name or j for j in cons[B] if j in need])
