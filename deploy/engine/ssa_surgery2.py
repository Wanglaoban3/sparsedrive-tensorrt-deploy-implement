"""P5b surgery: inner_attn 的 6 节点链 (QK^T MatMul → Cast → Mul(scale) →
Softmax → Cast → PV MatMul_1) → 单个 FlashSDPA 插件节点 (domain SparseDrive)。

用法: python ssa_surgery2.py <in.onnx> <out.onnx>
插件输入 [Q, K^T, V, scale常量] → 输出沿用 PV MatMul_1 输出名 (下游
Transpose_4 零改动)。投影 (q/k/v_proj) 与 Transpose_3/Cast_1/Cast_3 保留。
"""
import sys
from collections import Counter

import onnx
import numpy as np
from onnx import TensorProto, helper, numpy_helper

ELEM = {1: "f32", 7: "i64", 9: "bool", 10: "f16"}
FLOAT, FLOAT16 = TensorProto.FLOAT, TensorProto.FLOAT16

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
nodes = list(g.node)
init = {x.name: x for x in g.initializer}

mi = onnx.shape_inference.infer_shapes(m)
vi = {}
for v in list(mi.graph.value_info) + list(mi.graph.output):
    tt = v.type.tensor_type
    dims = [d.dim_value if d.WhichOneof("value") == "dim_value" else -1
            for d in tt.shape.dim]
    vi[v.name] = (tt.elem_type, dims)

use = {}
for j, x in enumerate(nodes):
    for xinp in x.input:
        use.setdefault(xinp, []).append(j)

removed, sites = set(), []
for i, n in enumerate(nodes):
    if n.op_type != "MatMul" or not n.name.endswith("/attn/inner_attn/MatMul"):
        continue
    q_t, kt_t = n.input
    chain, cur = [], n.output[0]
    for _ in range(4):
        cons = use.get(cur, [])
        assert len(cons) == 1, f"fanout at {cur}"
        chain.append(nodes[cons[0]])
        cur = chain[-1].output[0]
    c4, mul, sm, c5 = chain
    assert c4.op_type == "Cast" and mul.op_type == "Mul" and \
        sm.op_type == "Softmax" and c5.op_type == "Cast"
    consts = [x for x in mul.input if x in init]
    assert len(consts) == 1
    sc = float(numpy_helper.to_array(init[consts[0]]).reshape(-1)[0])
    pv_cons = use.get(c5.output[0], [])
    assert len(pv_cons) == 1, f"PV fanout at {c5.output[0]}"
    pv = nodes[pv_cons[0]]
    assert pv.op_type == "MatMul" and \
        pv.name.endswith("/attn/inner_attn/MatMul_1")
    v_t = pv.input[1]
    assert v_t != c5.output[0]
    out_t = pv.output[0]
    out_cons = [nodes[c].op_type for c in use.get(out_t, [])]
    assert out_cons == ["Transpose"], f"{out_t} -> {out_cons}"
    et, dims = vi[q_t]
    assert et == FLOAT16 and len(dims) == 4 and dims[3] == 64, f"{q_t} {dims}"
    etv, dimv = vi[v_t]
    assert etv == FLOAT16, f"{v_t} {dimv}"
    sites.append((n.name, list(dims), list(dimv), sc))
    removed.update([n.output[0], c4.output[0], mul.output[0], sm.output[0],
                    c5.output[0], pv.output[0]])
    # 记录站点 → (q, kt, v, scale_const, out)
    sites[-1] = sites[-1] + (q_t, kt_t, v_t, consts[0], out_t)

print(f"sites: {len(sites)}")
shapes = Counter(tuple(s[1][:2]) + (s[2][2],) for s in sites)
for shp, c in sorted(shapes.items()):
    print(f"  Q{list(shp[:2])} Kn={shp[2]}: x{c}")

by_out = {s[8]: s for s in sites}
new_nodes = []
for n in nodes:
    if n.op_type == "MatMul" and n.name.endswith("/attn/inner_attn/MatMul_1") \
            and n.output[0] in by_out:
        # 插件放在 PV 原位置: 其时所有输入 (Q/K^T/V/scale) 均已产出
        s = by_out[n.output[0]]
        pre = n.name[: n.name.rindex("/")]
        pn = helper.make_node("FlashSDPA", [s[4], s[5], s[6], s[7]], [s[8]],
                              name=f"{pre}/FlashSDPA")
        pn.domain = "SparseDrive"
        new_nodes.append(pn)
        continue
    if n.output[0] in removed:
        continue
    new_nodes.append(n)
del g.node[:]
g.node.extend(new_nodes)

keep_vi = [v for v in g.value_info if v.name not in removed]
del g.value_info[:]
g.value_info.extend(keep_vi)

onnx.checker.check_model(m, False)
onnx.save(m, dst)

outs = set()
for n in g.node:
    outs.update(n.output)
gin = {x.name for x in g.input}
dangling = [t for n in g.node for t in n.input
            if t and t not in outs and t not in init and t not in gin]
assert not dangling, f"dangling: {dangling[:5]}"
nfa = sum(1 for n in g.node if n.op_type == "FlashSDPA")
print(f"removed {len(removed)} tensors, FlashSDPA nodes: {nfa}, saved {dst} "
      f"({len(g.node)} nodes)")
