# -*- coding: utf-8 -*-
"""P0-1b fallback: same kps rewrite but with the 3-D GEMM cast to FP16
(bypasses the board's broken fp32 NVRTC JIT path). Produces v4_P1h.onnx.
Stub equivalence vs v3_T3 with fp16 tolerance."""
import io
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference, TensorProto

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
SRC = W + r"\v6_T3.onnx"
DST = W + r"\v6_P1h.onnx"

m = onnx.load(SRC)
mi = shape_inference.infer_shapes(m)
vi = {v.name: v for v in list(mi.graph.value_info) + list(mi.graph.output)}
g = m.graph

targets = [n for n in g.node if n.op_type == "MatMul"
           and any("Unsqueeze_10" in x for x in n.input)]
print("kps matmuls:", len(targets))

inits_to_add = []


def iconst(name, arr):
    inits_to_add.append(helper.make_tensor(name, TensorProto.INT64,
                                           [len(arr)], arr))


def nidx(node):
    return next(i for i, nn in enumerate(list(g.node))
                if nn.name == node.name)


new_nodes_by_pos = {}
removed = set()
w2_done = False
for n in targets:
    out = n.output[0]
    dims = [d.dim_value for d in vi[out].type.tensor_type.shape.dim]
    lvls, Q, K = dims[1], dims[2], dims[3]
    x0, x1 = n.input[0], n.input[1]
    pos = nidx(n)
    add = []
    if not w2_done:
        w2_name = out + "_w2t"
        iconst(w2_name + "_sh", [lvls, 4, 4])
        add.append(helper.make_node("Reshape", [x0, w2_name + "_sh"],
                                    [w2_name + "_r"],
                                    name=out + "_w2_reshape"))
        add.append(helper.make_node("Transpose", [w2_name + "_r"],
                                    [w2_name], perm=[0, 2, 1],
                                    name=out + "_w2_transp"))
        add.append(helper.make_node("Cast", [w2_name], [w2_name + "h"],
                                    to=TensorProto.FLOAT16,
                                    name=out + "_w2_cast"))
        w2h = w2_name + "h"
        w2_done = True
    xr, ym = out + "_xr", out + "_ym"
    iconst(xr + "_sh", [1, Q * K, 4])
    iconst(ym + "_sh", dims)
    add.append(helper.make_node("Cast", [x1], [x1 + "h"],
                                to=TensorProto.FLOAT16,
                                name=out + "_xr_cast"))
    add.append(helper.make_node("Reshape", [x1 + "h", xr + "_sh"],
                                [xr + "h"], name=out + "_xr_reshape"))
    add.append(helper.make_node("MatMul", [xr + "h", w2h], [ym + "h"],
                                name=out + "_mm3d"))
    add.append(helper.make_node("Cast", [ym + "h"], [ym],
                                to=TensorProto.FLOAT,
                                name=out + "_mm3d_tof"))
    add.append(helper.make_node("Reshape", [ym, ym + "_sh"], [out],
                                name=out + "_back"))
    new_nodes_by_pos[pos] = add
    removed.add(n.name)

nl = []
for i, node in enumerate(list(g.node)):
    nl.extend(new_nodes_by_pos.get(i, []))
    if node.name in removed:
        continue
    nl.append(node)
del g.node[:]
g.node.extend(nl)
g.initializer.extend(inits_to_add)
onnx.save(m, DST)
print("saved", DST)

# -------- stub equivalence vs v3_T3 with fp16 tolerance --------


def stub(src, dst):
    mm = onnx.load(src)
    gg = mm.graph
    keep = [o for o in mm.opset_import if o.domain != "SparseDrive"]
    del mm.opset_import[:]
    mm.opset_import.extend(keep)
    inits, new_nodes = [], []
    si = [0]

    def c(name, arr):
        inits.append(numpy_helper.from_array(arr, name=name))

    for n in list(gg.node):
        if n.op_type != "DeformableAggregation":
            continue
        feat, shp, ssi, loc, w = n.input[:5]
        out = n.output[0]
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
        new_nodes.append(helper.make_node("Cast", [shp], [c1],
                                          to=TensorProto.FLOAT))
        new_nodes.append(helper.make_node("ReduceSum", [c1], [s2],
                                          keepdims=0))
        s3 = p + "s3"
        c2 = p + "c2"
        new_nodes.append(helper.make_node("Cast", [ssi], [c2],
                                          to=TensorProto.FLOAT))
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


sb = stub(DST, DST + ".smoke.onnx")
sa = stub(SRC, SRC + ".smoke.onnx")
inputs_npz = np.load(W + r"\mtq_v3_ref.npz.inputs.npz")
sessA = ort.InferenceSession(sa, None, providers=["CPUExecutionProvider"])
sessB = ort.InferenceSession(sb, None, providers=["CPUExecutionProvider"])
feed = {v.name: inputs_npz[v.name] for v in sessA.get_inputs()}
names = [o.name for o in sessA.get_outputs()]
va = dict(zip(names, sessA.run(names, feed)))
vb = dict(zip(names, sessB.run(names, feed)))
worst = 0.0
for nm in names:
    a, b = va[nm].astype(np.float64), vb[nm].astype(np.float64)
    d = float(np.abs(a - b).max())
    den = max(1e-6, float(np.abs(a).max()))
    worst = max(worst, d / den)
    if d > 1e-5:
        print(f"DIFF {nm} maxabs={d:.3e} rel={d / den:.3e}")
print("WORST rel:", worst, "(fp16 tolerance gate: 5e-3)")
