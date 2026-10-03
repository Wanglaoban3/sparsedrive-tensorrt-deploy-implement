# -*- coding: utf-8 -*-
"""Dump one SSA block (layers.5/attn) node-by-node to design the rewrite."""
import io
import os
import sys

import onnx
from onnx import TensorProto

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
m = onnx.load(os.path.join(W, "sp_head_det.onnx"))
g = m.graph
nodes = list(g.node)
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
prod = {}
for i, n in enumerate(nodes):
    for o in n.output:
        prod[o] = i
ELEM = {TensorProto.FLOAT: "f32", TensorProto.FLOAT16: "f16",
        TensorProto.INT32: "i32", TensorProto.INT64: "i64",
        TensorProto.BOOL: "b8"}

start = next(i for i, n in enumerate(nodes)
             if n.name == "/layers.5/attn/inner_attn/MatMul")
end = next(i for i, n in enumerate(nodes)
           if n.name == "/layers.5/attn/outer_attn/MatMul_output_0_w2"
           ) if False else start + 40
for i in range(start - 6, min(start + 42, len(nodes))):
    n = nodes[i]
    ins = []
    for x in n.input:
        if x in inits:
            ins.append("init:" + x.split(".")[-1][:22])
        else:
            et, dims = vi.get(x, (0, []))
            ins.append(f"{ELEM.get(et, '?')}{dims}")
    outs = [f"{ELEM.get(vi.get(o, (0, []))[0], '?')}"
            f"{vi.get(o, (0, [])[1] if False else (0, []))[1] if o in vi else '?'}"
            for o in n.output]
    outs = []
    for o in n.output:
        et, dims = vi.get(o, (0, []))
        outs.append(f"{ELEM.get(et, '?')}{dims}")
    print(f"[{i}] {n.op_type} {n.name}")
    print(f"     in : {ins}")
    print(f"     out: {outs}")
