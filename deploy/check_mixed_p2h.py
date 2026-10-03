# -*- coding: utf-8 -*-
"""quick mixed-dtype check for v5_P2h (single topological pass)"""
import io
import os
import sys

import onnx
from onnx import TensorProto

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
m = onnx.load(os.path.join(W, "v5_P2h.onnx"))
g = m.graph
F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT

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

mixed = []
for n in g.node:
    if n.op_type == "Cast":
        to = next(a.i for a in n.attribute if a.name == "to")
        if to in (F16, F32):
            dt[n.output[0]] = to
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
    elif n.op_type == "DeformableAggregation":
        # plugin contract: feat input forced kHALF -> output kHALF
        for o in n.output:
            if o:
                dt[o] = F16
    fts = {t for t in (dt.get(x) for x in n.input if x) if t in (F16, F32)}
    if len(fts) > 1 and n.op_type not in ("QuantizeLinear",
                                          "DequantizeLinear"):
        mixed.append((n.op_type, n.name,
                      [("f16" if dt.get(x) == F16 else "f32")
                       for x in n.input if x and dt.get(x) in (F16, F32)]))

print(f"mixed float-input nodes: {len(mixed)}")
for op, nm, dts in mixed[:20]:
    print(f"   {op} {nm} {dts}")
