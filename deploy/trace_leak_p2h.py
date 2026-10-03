# -*- coding: utf-8 -*-
"""trace where f16 leaks into the temporal bookkeeping nodes of v5_P2h"""
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
F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT
import re
PAT = r"layers\.|fc_before|fc_after"
in_region = lambda name: re.search(PAT, name) is not None

prod = {}
for n in g.node:
    for o in n.output:
        prod[o] = n

PRESERVE = {
    "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Slice", "Identity",
    "Softmax", "Concat", "Mul", "Add", "Sub", "Div", "Pow", "Tanh",
    "Sigmoid", "Relu", "Erf", "Sqrt", "Reciprocal", "Neg", "Exp", "Log",
    "Max", "Min", "Pad", "Tile", "Expand", "Gather", "MatMul", "Gemm",
    "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "Flatten",
    "LeakyRelu", "HardSigmoid", "Shrink", "Sign", "Ceil", "Floor", "Round",
    "Where", "Dropout", "Clip", "ScatterND", "GatherElements", "Trilu",
}
dt = {}
for vi in list(g.value_info) + list(g.output) + list(g.input):
    t = vi.type.tensor_type.elem_type
    if t in (F16, F32):
        dt[vi.name] = t
for i in g.initializer:
    if i.data_type in (F16, F32):
        dt[i.name] = i.data_type
for n in g.node:
    if n.op_type == "Cast":
        to = next(a.i for a in n.attribute if a.name == "to")
        if to in (F16, F32):
            dt[n.output[0]] = to
        continue
    if n.op_type == "DeformableAggregation":
        for o in n.output:
            if o:
                dt[o] = F16
        continue
    if n.op_type == "DequantizeLinear":
        t = dt.get(n.input[1])
        if t in (F16, F32):
            dt[n.output[0]] = t
        continue
    if n.op_type in PRESERVE:
        fts = {t for t in (dt.get(x) for x in n.input if x)
               if t in (F16, F32)}
        if len(fts) == 1:
            dt[n.output[0]] = next(iter(fts))


def trace(t, depth=0, seen=None):
    """walk up from t, print the f16 entry point"""
    if seen is None:
        seen = set()
    if t in seen or depth > 12:
        return
    seen.add(t)
    p = prod.get(t)
    tag = "REG" if (p and in_region(p.name)) else "out"
    if dt.get(t) == F16 and (p is None or dt.get(p.input[0]) != F16
                             or True):
        pass
    if p is None:
        print(f"{'  '*depth}{t}  dt={dt.get(t)}  (graph input/init)")
        return
    print(f"{'  '*depth}{t}  dt={dt.get(t)}  <- {p.op_type} {p.name} [{tag}]")
    for x in p.input[:4]:
        if dt.get(x) in (F16, F32):
            trace(x, depth + 1, seen)


for start in ["/Concat_29", "/anchor_encoder/pos_fc/pos_fc.0_10/MatMul"]:
    n = next((n for n in g.node if n.name == start), None)
    if n is None:
        print(f"{start}: NOT FOUND")
        continue
    print(f"\n===== {start} ({n.op_type}) =====")
    for x in n.input:
        if dt.get(x) in (F16, F32):
            trace(x)
            print()
