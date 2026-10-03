# -*- coding: utf-8 -*-
"""probe: find tensors whose ORT-inferred dtype contradicts the declared
value_info dtype.  Inference starts ONLY from graph inputs + initializers
+ Cast/DequantizeLinear + PRESERVE propagation (no value_info seeding),
i.e. what ORT will actually see."""
import collections
import io
import os
import sys

import onnx
from onnx import TensorProto

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
SRC = os.environ.get("P2H_CHK", os.path.join(W, "v5_P2h.onnx"))
print(f"checking {SRC}")
m = onnx.load(SRC)
g = m.graph

PRESERVE = {
    "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Slice", "Identity",
    "Softmax", "Concat", "Mul", "Add", "Sub", "Div", "Pow", "Tanh",
    "Sigmoid", "Relu", "Erf", "Sqrt", "Reciprocal", "Neg", "Exp", "Log",
    "Max", "Min", "Pad", "Tile", "Expand", "Gather", "MatMul", "Gemm",
    "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "Flatten",
    "LeakyRelu", "HardSigmoid", "Shrink", "Sign", "Ceil", "Floor", "Round",
    "Where", "Dropout", "Clip", "ScatterND", "GatherElements", "Trilu",
}
F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT
DTC = {TensorProto.FLOAT: "f32", TensorProto.FLOAT16: "f16"}

init_dt = {i.name: i.data_type for i in g.initializer}
declared = {}
for vi in list(g.value_info) + list(g.output) + list(g.input):
    declared[vi.name] = vi.type.tensor_type.elem_type

dt = {}
for v in g.input:
    dt[v.name] = v.type.tensor_type.elem_type
dt.update(init_dt)

op_stats = collections.Counter()
for n in g.node:
    if n.op_type == "Cast":
        to = next(a.i for a in n.attribute if a.name == "to")
        dt[n.output[0]] = to
        continue
    if n.op_type == "DequantizeLinear":
        t = dt.get(n.input[1])
        if t is not None:
            dt[n.output[0]] = t
        continue
    if n.op_type == "Constant":
        continue
    if n.op_type == "DeformableAggregation":
        for o in n.output:
            if o:
                dt[o] = F16
        continue
    if n.op_type in PRESERVE:
        fts = {t for t in (dt.get(x) for x in n.input if x)
               if t in (F16, F32)}
        if len(fts) == 1:
            t = next(iter(fts))
            for o in n.output:
                if o:
                    dt[o] = t
        elif len(fts) > 1:
            op_stats[n.op_type] += 1
            print(f"MIXED per-inference: {n.op_type} {n.name} "
                  f"{sorted(DTC.get(t, t) for t in fts)}")
    else:
        op_stats["UNTRACKED:" + n.op_type] += 1

bad = 0
for name, decl in declared.items():
    act = dt.get(name)
    if act is not None and act != decl and decl in (F16, F32):
        p = None
        bad += 1
        print(f"DECLARED {DTC.get(decl, decl)} but INFERRED "
              f"{DTC.get(act, act)}: {name}")
print(f"declared-vs-inferred dtype mismatches: {bad}")
print("untracked op counts:",
      dict((k, v) for k, v in op_stats.items() if k.startswith("UNTRACKED")))
