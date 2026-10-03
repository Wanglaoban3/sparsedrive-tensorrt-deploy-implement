# -*- coding: utf-8 -*-
"""T1: attn region -> pure float.
- strip activation QDQ pairs whose Q name contains '/attn/'
- dequantize int8-weight DQs whose consumers are all '/attn/' nodes
Everything else (backbone/fc/misc quantization) untouched."""
import re
import sys
import numpy as np
import onnx
from onnx import numpy_helper

W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
SRC = W + r"\sparsedrive_int8_v5_fold5.onnx"
import os
_tag = os.environ.get("TVAR", "T3")
DST = W + rf"\v5_{_tag}.onnx"
PAT = sys.argv[1] if len(sys.argv) > 1 else r"layers\.|fc_before|fc_after"

m = onnx.load(SRC)
g = m.graph
ini_map = {i.name: i for i in g.initializer}
inits = set(ini_map)
prod = {}
for n in g.node:
    for o in n.output:
        prod[o] = n
cons = {}
for n in g.node:
    for i in n.input:
        cons.setdefault(i, []).append(n)


def in_region(nname):
    return re.search(PAT, nname) is not None


VIA = {"Transpose", "Reshape", "Cast", "Squeeze", "Unsqueeze", "Identity"}


def is_weight_q(q):
    t = q.input[0]
    for _ in range(4):
        if t in inits:
            return True
        p = prod.get(t)
        if p is None or p.op_type not in VIA:
            return False
        t = p.input[0]
    return True


# --- 1. strip activation QDQ with name matching PAT (chain-safe) ---
qs = [n for n in g.node if n.op_type == "QuantizeLinear"
      and n.input[0] not in inits and not is_weight_q(n)
      and in_region(n.name)]
print(f"stripping {len(qs)} activation QDQ pairs matching {PAT}")
removed = 0
removed_set = set()


def resolve(t):
    while t in prod and prod[t].name in removed_set:
        t = prod[t].input[0]
    return t


for q in qs:
    qout = q.output[0]
    dqs = [d for d in cons.get(qout, [])
           if d.op_type == "DequantizeLinear"
           and d.input[1] == q.input[1] and d.input[2] == q.input[2]]
    if not dqs:
        continue
    x = resolve(q.input[0])
    for dq in dqs:
        dout = dq.output[0]
        for n in g.node:
            for k, i in enumerate(n.input):
                if i == dout:
                    n.input[k] = x
        g.node.remove(dq)
        removed_set.add(dq.name)
    g.node.remove(q)
    removed_set.add(q.name)
    removed += 1
print(f"  removed {removed}")

# --- 2. dequantize weight DQs consumed only by region nodes ---
new_inits = []
to_del = []
for d in g.node:
    if d.op_type != "DequantizeLinear" or d.input[0] not in inits:
        continue
    users = cons.get(d.output[0], [])
    if not users:
        continue
    cover = all(in_region(n.name) for n in users) or in_region(d.name)
    if not cover:
        continue
    w8 = numpy_helper.to_array(ini_map[d.input[0]])
    scale = numpy_helper.to_array(ini_map[d.input[1]]).astype(np.float32)
    axis = 0
    for a in d.attribute:
        if a.name == "axis":
            axis = a.i
    zp = None
    if len(d.input) > 2 and d.input[2] in inits:
        zp = numpy_helper.to_array(ini_map[d.input[2]])
    shape = [1] * w8.ndim
    shape[axis if axis >= 0 else w8.ndim + axis] = -1
    wf = w8.astype(np.float32) * scale.reshape(shape)
    if zp is not None:
        wf = wf + zp.astype(np.float32).reshape(shape)
    nm = d.input[0] + "._deq"
    new_inits.append(numpy_helper.from_array(wf, nm))
    for n in users:
        for k, i in enumerate(n.input):
            if i == d.output[0]:
                n.input[k] = nm
    to_del.append(d)
for d in to_del:
    g.node.remove(d)
g.initializer.extend(new_inits)
print(f"dequantized {len(to_del)} weight DQs in region")

live = set()
for n in g.node:
    live.update(n.input)
live.update(vi.name for vi in list(g.input) + list(g.output))
inis_live = set()
for n in g.node:
    inis_live.update(n.input)
keep = [i for i in list(g.initializer) if i.name in inis_live]
del g.initializer[:]
g.initializer.extend(keep)
for _ in range(8):
    changed = False
    for n in list(g.node):
        if all(o not in live for o in n.output):
            g.node.remove(n)
            changed = True
    if not changed:
        break

onnx.checker.check_model(m)
onnx.save(m, DST)
print("saved", DST)
