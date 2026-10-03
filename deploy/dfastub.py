# -*- coding: utf-8 -*-
"""Shared DFA-stub helper: replace DeformableAggregation custom nodes with
pure-ORT arithmetic that preserves input dtypes, so the graph loads on
CPUExecutionProvider (dtype mistakes fail at session load)."""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _graph_dtypes(gg):
    """Single topological dtype pass mirroring the graph's actual rules:
    Cast (to attr), DequantizeLinear (scale dtype), DeformableAggregation
    (kHALF contract), and same-dtype passthrough ops."""
    F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT
    dt = {v.name: v.type.tensor_type.elem_type for v in gg.input}
    dt.update({i.name: i.data_type for i in gg.initializer})
    for nd in gg.node:
        if nd.op_type == "Cast":
            to = next((a.i for a in nd.attribute if a.name == "to"), None)
            if to is not None and nd.output:
                dt[nd.output[0]] = to
            continue
        if nd.op_type == "DequantizeLinear":
            t = dt.get(nd.input[1])
            if t is not None and nd.output:
                dt[nd.output[0]] = t
            continue
        if nd.op_type == "DeformableAggregation":
            for o in nd.output:
                if o:
                    dt[o] = F16
            continue
        fts = {dt[x] for x in nd.input if x and dt.get(x) in (F16, F32)}
        if len(fts) == 1 and nd.output:
            for o in nd.output:
                if o:
                    dt[o] = next(iter(fts))
    return dt


def stub(src, dst):
    mm = onnx.load(src)
    gg = mm.graph
    keep = [o for o in mm.opset_import if o.domain != "SparseDrive"]
    del mm.opset_import[:]
    mm.opset_import.extend(keep)
    inits, new_nodes = [], []
    si = [0]
    dtm = _graph_dtypes(gg)

    def c(name, arr):
        inits.append(numpy_helper.from_array(arr, name=name))

    for n in list(gg.node):
        if n.op_type != "DeformableAggregation":
            continue
        feat, shp, ssi, loc, w = n.input[:5]
        out = n.output[0]
        t = (dtm.get(feat) or dtm.get(loc) or dtm.get(w)
             or TensorProto.FLOAT)
        p = f"ds{si[0]}_"
        si[0] += 1
        a1 = p + "a1"
        c(a1, np.array([1], np.int64))
        a234 = p + "a234"
        c(a234, np.array([2, 3, 4], np.int64))
        a2345 = p + "a2345"
        c(a2345, np.array([2, 3, 4, 5], np.int64))
        a2 = p + "a2"
        c(a2, np.array([2], np.int64))
        fs = p + "fs"
        new_nodes.append(helper.make_node("ReduceSum", [feat, a1], [fs],
                                          keepdims=0))
        fu = p + "fu"
        new_nodes.append(helper.make_node("Unsqueeze", [fs, a1], [fu]))
        ls = p + "ls"
        new_nodes.append(helper.make_node("ReduceSum", [loc, a234], [ls],
                                          keepdims=0))
        ws = p + "ws"
        new_nodes.append(helper.make_node("ReduceSum", [w, a2345], [ws],
                                          keepdims=0))
        lw = p + "lw"
        new_nodes.append(helper.make_node("Add", [ls, ws], [lw]))
        lwu = p + "lwu"
        new_nodes.append(helper.make_node("Unsqueeze", [lw, a2], [lwu]))
        s2 = p + "s2"
        c1 = p + "c1"
        new_nodes.append(helper.make_node("Cast", [shp], [c1], to=t))
        new_nodes.append(helper.make_node("ReduceSum", [c1], [s2],
                                          keepdims=0))
        s3 = p + "s3"
        c2 = p + "c2"
        new_nodes.append(helper.make_node("Cast", [ssi], [c2], to=t))
        new_nodes.append(helper.make_node("ReduceSum", [c2], [s3],
                                          keepdims=0))
        sc = p + "sc"
        new_nodes.append(helper.make_node("Add", [s2, s3], [sc]))
        o1 = p + "o1"
        new_nodes.append(helper.make_node("Mul", [fu, lwu], [o1]))
        new_nodes.append(helper.make_node("Mul", [o1, sc], [out]))
        gg.node.remove(n)
    for n in new_nodes:
        gg.node.append(n)
    gg.initializer.extend(inits)
    onnx.save(mm, dst)
    return dst
