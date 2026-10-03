# -*- coding: utf-8 -*-
"""probe: diff the conv1->relu->maxpool-Q chain between P1h and P2h"""
import collections
import io
import os
import sys

import onnx
from onnx import TensorProto

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
DTC = {TensorProto.FLOAT: "f32", TensorProto.FLOAT16: "f16",
       TensorProto.INT8: "i8", TensorProto.INT32: "i32",
       TensorProto.INT64: "i64", TensorProto.BOOL: "bool"}

W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")


def dump(tag):
    src = os.path.join(W, tag)
    print(f"########## {tag}")
    m = onnx.load(src)
    g = m.graph
    prod = {}
    cons = collections.defaultdict(list)
    for n in g.node:
        for o in n.output:
            prod[o] = n
        for x in n.input:
            if x:
                cons[x].append(n)
    init_dt = {i.name: i.data_type for i in g.initializer}
    graph_ins = {v.name: v.type.tensor_type.elem_type for v in g.input}
    vi_dt = {v.name: v.type.tensor_type.elem_type for v in g.value_info}

    def desc(x):
        if x in init_dt:
            return f"{x} [INIT {DTC.get(init_dt[x], init_dt[x])}]"
        if x in graph_ins:
            return f"{x} [GRAPH-IN {DTC.get(graph_ins[x], graph_ins[x])}]"
        d = vi_dt.get(x)
        p = prod.get(x)
        ps = f"{p.op_type} {p.name}" if p else "NO-PROD"
        return f"{x} [{ps}; decl={DTC.get(d, d)}]"

    for nm in ("/img_backbone/relu/Relu", "/img_backbone/conv1/Conv",
               "/img_backbone/maxpool/input_quantizer/QuantizeLinear"):
        n = next((n for n in g.node if n.name == nm), None)
        print(f"== {nm}" + (" (MISSING)" if n is None else ""))
        if n is None:
            continue
        for k, x in enumerate(n.input):
            print(f"  in[{k}] {desc(x)}")
    # who consumes Relu output / conv1 output
    for tname in ("/img_backbone/relu/Relu_output_0",):
        for c in cons.get(tname, []):
            print(f"  consumer of {tname}: {c.op_type} {c.name}")
    print()


for tag in ("v5_P1h.onnx", "v5_P2h.onnx"):
    dump(tag)
