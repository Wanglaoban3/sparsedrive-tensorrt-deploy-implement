# -*- coding: utf-8 -*-
"""P5 隔离探针: 对指定 layer 的 inner_attn 站点, 把 Q/K^T/V (+基线链全部中间
张量 / FA 图的插件输出) 加为图输出, 供引擎 dump 后 numpy 对拍。
用法: python probe_fa_io.py <in.onnx> <out.onnx> <layer 例 layers.19>
"""
import sys

import onnx
from onnx import TensorProto, helper, numpy_helper

src, dst, layer = sys.argv[1], sys.argv[2], sys.argv[3]
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

target = None
fa_node = None
for n in nodes:
    if n.op_type == "MatMul" and \
            n.name.endswith(f"/{layer}/attn/inner_attn/MatMul"):
        target = n
    if n.op_type == "FlashSDPA" and \
            n.name.endswith(f"/{layer}/attn/inner_attn/FlashSDPA"):
        fa_node = n
assert target is not None or fa_node is not None, f"site {layer} not found"
if fa_node is not None:  # FA 图: 链已被手术删除, 直接取插件 I/O
    q_t, kt_t, v_t = fa_node.input[0], fa_node.input[1], fa_node.input[2]
    out_t = fa_node.output[0]
    c4 = mul = sm = c5 = None
else:
    q_t, kt_t = target.input
    chain, cur = [], target.output[0]
    for _ in range(4):
        cons = use.get(cur, [])
        assert len(cons) == 1
        chain.append(nodes[cons[0]])
        cur = chain[-1].output[0]
    c4, mul, sm, c5 = chain
    pv = nodes[use.get(c5.output[0], [None])[0]]
    assert pv is not None and pv.op_type == "MatMul"
    v_t, out_t = pv.input[1], pv.output[0]

if mul is not None:
    consts = [x for x in mul.input if x in init]
    sc = float(numpy_helper.to_array(init[consts[0]]).reshape(-1)[0])
else:
    sc = 0.125  # FA 图: 链已删, hd=64 站点 scale 已知
print(f"site {layer}: q={q_t}\n  kt={kt_t}\n  v={v_t}\n  scale={sc}")


def add_out(name, fallback=None):
    if name in vi:
        et, dims = vi[name]
    else:
        assert fallback is not None, f"no shape info for {name}"
        et, dims = fallback
    tt = {1: TensorProto.FLOAT, 10: TensorProto.FLOAT16}.get(et)
    assert tt is not None, (name, et)
    g.output.append(helper.make_tensor_value_info(name, tt, dims))


add_out(q_t)
add_out(kt_t)
add_out(v_t)
live = {n.output[0] for n in g.node}
if c4 is not None and c4.output[0] in live:  # 基线图: 暴露全链
    for t in [target.output[0], c4.output[0], mul.output[0], sm.output[0],
              c5.output[0], out_t]:
        add_out(t)
    print("  baseline chain exposed (6 tensors)")
add_out(out_t, fallback=vi[q_t])  # FA 图: 插件输出与 Q 同形 f16

onnx.checker.check_model(m, False)
onnx.save(m, dst)
print("saved", dst)
