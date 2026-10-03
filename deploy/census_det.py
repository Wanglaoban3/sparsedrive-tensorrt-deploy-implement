# -*- coding: utf-8 -*-
"""Census the ops between consecutive DFA calls in sp_head_det.onnx:
per segment op histogram, dtype mix, MatMul FLOPs, biggest tensors."""
import collections
import io
import os
import sys

import onnx
from onnx import TensorProto

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
m = onnx.load(os.path.join(W, "sp_head_det.onnx"))
g = m.graph
nodes = list(g.node)
prod = {}
for i, n in enumerate(nodes):
    for o in n.output:
        prod[o] = i
inits = {x.name for x in g.initializer}
mi = onnx.shape_inference.infer_shapes(m)
vi = {}
for src in (g.value_info, g.input, g.output,
            mi.graph.value_info, mi.graph.input, mi.graph.output):
    for v in src:
        tt = v.type.tensor_type
        dims = [d.dim_value if d.WhichOneof("value") == "dim_value" else -1
                for d in tt.shape.dim]
        vi[v.name] = (tt.elem_type, dims)
ELEM = {TensorProto.FLOAT: "f32", TensorProto.FLOAT16: "f16",
        TensorProto.INT32: "i32", TensorProto.INT64: "i64",
        TensorProto.BOOL: "b8"}

dfas = [i for i, n in enumerate(nodes)
        if n.op_type == "DeformableAggregation"]
print("DFA indices:", dfas, [nodes[i].name.split('/')[1] for i in dfas])
segs = []
lo = 0
for di in dfas + [len(nodes)]:
    segs.append((lo, di))
    lo = di + 1

for a, b in segs:
    seg = nodes[a:b + 1]
    ops = collections.Counter(n.op_type for n in seg)
    f32 = [n.name or n.op_type for n in seg for x in n.input
           if x in vi and vi[x][0] == TensorProto.FLOAT]
    i64 = [n.name for n in seg for x in n.input
           if x in vi and vi[x][0] == TensorProto.INT64]
    fl = 0
    for n in seg:
        if n.op_type in ("MatMul", "Gemm"):
            sa = vi.get(n.input[0], (0, []))[1]
            sb = vi.get(n.input[1], (0, []))[1] if len(n.input) > 1 else []
            if sa and sb and -1 not in sa and -1 not in sb and len(sa) >= 2:
                batch = 1
                for d in sa[:-2]:
                    batch *= d
                fl += 2 * batch * sa[-2] * sa[-1] * sb[-1]
    big = sorted(((int(np.prod([d for d in vi[o][1] if d > 0]) or 0) *
                   (2 if vi[o][0] == TensorProto.FLOAT16 else 4), o)
                  for n in seg for o in n.output if o in vi),
                 reverse=True)[:5]
    print(f"\n== nodes[{a}..{b}] n={len(seg)} MM={fl/1e9:.2f}GF "
          f"f32in={len(f32)} i64in={len(i64)}")
    print("   ops:", dict(ops.most_common(12)))
    if f32:
        print("   f32 inputs:", f32[:8])
    print("   big:", ", ".join(f"{s/1e6:.2f}MB {ELEM.get(vi[o][0])}"
                               f"{vi[o][1]}~{o[-45:]}" for s, o in big))
