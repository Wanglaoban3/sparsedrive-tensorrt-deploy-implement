# -*- coding: utf-8 -*-
"""make dump onnx: expose every DFA node's loc/logits/output (+shared feat)
as graph outputs, so a purpose-built engine dumps engine-bit-exact data."""
import io
import sys

import onnx
from onnx import helper, shape_inference, TensorProto

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
SRC = W + r"\v5_P1h.onnx"
DST = W + r"\v5_dump.onnx"

m = onnx.load(SRC)
mi = shape_inference.infer_shapes(m)
vi = {v.name: v for v in list(mi.graph.value_info) + list(mi.graph.output)}
g = m.graph

existing = {o.name for o in g.output}
n_added = 0
feat_done = False

dfas = [n for n in g.node if n.op_type == "DeformableAggregation"]
print("DFA nodes:", len(dfas))


def add_out(tensor_name, suffix):
    global n_added
    out_name = tensor_name + suffix
    if out_name in existing or tensor_name in existing:
        return
    v = vi.get(tensor_name)
    if v is None:
        print("  !! no shape info for", tensor_name)
        return
    g.output.append(v)
    # rename to the suffixed name
    g.output[-1].name = out_name
    existing.add(out_name)
    n_added += 1


for k, n in enumerate(dfas):
    if not feat_done:
        add_out(n.input[0], ".dfa_feat")   # 共享 feat, 只输出一次
        feat_done = True
    add_out(n.input[3], f".dfa{k}_loc")
    add_out(n.input[4], f".dfa{k}_log")
    add_out(n.output[0], f".dfa{k}_out")

onnx.save(m, DST)
print("added", n_added, "outputs -> saved", DST)
