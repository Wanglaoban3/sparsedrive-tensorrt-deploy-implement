# -*- coding: utf-8 -*-
"""P4-1: analyze v5_P1h.onnx for the engine split.
zone = backward closure of every QuantizeLinear float input (= backbone+neck
INT8 region, same rule as make_v5_p2h.py). head = everything else.
Boundary = tensors produced on the backbone side (zone ∪ Q/DQ) and consumed
by head nodes. Prints IO contract for both subgraphs."""
import collections
import io
import os
import re
import sys

import onnx
from onnx import TensorProto

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
SRC = (r"<REPO>\work_dirs"
       r"\sparsedrive_small_stage2\v5_P1h.onnx")

m = onnx.load(SRC, load_external_data=False)
g = m.graph
nodes = list(g.node)
print(f"graph: {len(nodes)} nodes, {len(g.initializer)} inits, "
      f"{len(g.input)} inputs, {len(g.output)} outputs, "
      f"{os.path.getsize(SRC)/1e6:.0f} MB")

prod = {}
cons = collections.defaultdict(list)
for n in nodes:
    for o in n.output:
        prod[o] = n
    for x in n.input:
        if x:
            cons[x].append(n)

init_names = {i.name for i in g.initializer}
# initializer as graph input (old-style) counts as constant, not engine input
graph_inputs = [vi.name for vi in g.input if vi.name not in init_names]

# zone: backward closure of Q float inputs
zone = set()
stack = [n for n in nodes if n.op_type == "QuantizeLinear"]
while stack:
    n = stack.pop()
    if n.name in zone:
        continue
    zone.add(n.name)
    for x in n.input:
        if not x or x in init_names:
            continue
        p = prod.get(x)
        if p is not None and p.name not in zone:
            stack.append(p)
qdz = {n.name for n in nodes if n.op_type in ("QuantizeLinear",
                                              "DequantizeLinear")}
backbone_side = lambda nm: nm in zone or nm in qdz
head = [n for n in nodes if not backbone_side(n.name)]
print(f"backbone zone nodes: {len(zone)}  (+{len(qdz)} Q/DQ), "
      f"head nodes: {len(head)}")

# boundary tensors: backbone-side producer -> head consumer
boundary = collections.defaultdict(list)  # tensor -> [head consumers]
for n in head:
    for x in n.input:
        if not x or x in init_names:
            continue
        p = prod.get(x)
        if p is not None and backbone_side(p.name):
            boundary[x].append(n)

# value_info shapes/dtypes
vi = {}
for src in (g.value_info, g.input, g.output):
    for v in src:
        try:
            t = v.type.tensor_type
            dims = [d.dim_value if d.HasField("dim_value") else -1
                    for d in t.dim]
        except (AttributeError, IndexError):
            continue
        vi[v.name] = (t.elem_type, dims)
ELEM = {TensorProto.FLOAT: "f32", TensorProto.FLOAT16: "f16",
        TensorProto.INT32: "i32", TensorProto.INT64: "i64",
        TensorProto.INT8: "i8", TensorProto.UINT8: "u8",
        TensorProto.BOOL: "bool"}

print(f"\n== boundary tensors ({len(boundary)}) ==")
btype = collections.Counter()
for t, cns in sorted(boundary.items()):
    p = prod[t].op_type
    et, dims = vi.get(t, (0, []))
    btype[ELEM.get(et, str(et))] += 1
    print(f"  {t}  {ELEM.get(et, et)}{dims}  via {p}  -> "
          f"{len(cns)} consumers e.g. {cns[0].name[:60]}")
print("boundary dtypes:", dict(btype))

# graph-input consumers per side
print(f"\n== graph inputs ({len(graph_inputs)}) ==")
for name in graph_inputs:
    users = cons.get(name, [])
    nb = sum(1 for u in users if backbone_side(u.name))
    nh = len(users) - nb
    et, dims = vi.get(name, (0, []))
    tag = "BACKBONE" if nb and not nh else ("HEAD" if nh and not nb else
                                            "BOTH" if users else "DEAD")
    print(f"  {tag:8s} {name}  {ELEM.get(et, et)}{dims}  "
          f"({nb} bb / {nh} head consumers)")

# graph outputs and their producer side
print(f"\n== graph outputs ({len(g.output)}) ==")
for o in g.output:
    p = prod.get(o.name)
    et, dims = vi.get(o.name, (0, []))
    side = "HEAD" if (p and not backbone_side(p.name)) else "BB"
    print(f"  {side:4s} {o.name}  {ELEM.get(et, et)}{dims}")

# head nodes by op type (top)
cnt = collections.Counter(n.op_type for n in head)
print("\n== head ops ==")
print(" ".join(f"{k}:{v}" for k, v in cnt.most_common(25)))
# zone nodes by op type
cntz = collections.Counter(prod[zn].op_type for zn in zone
                           if not (prod[zn].op_type in
                                   ("QuantizeLinear", "DequantizeLinear")))
print("== backbone ops ==")
print(" ".join(f"{k}:{v}" for k, v in cntz.most_common(15)))
