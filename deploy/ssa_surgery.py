"""P5 SSA 注意力手术: inner_attn 的 Cast(f32)→Mul(scale)→Softmax→Cast(f16)
四节点链 → 单个 ScaleSoftmax 插件节点 (domain SparseDrive)。

用法: python ssa_surgery.py <in.onnx> <out.onnx>
站点: *attn/inner_attn/MatMul 后恰跟 Cast(to=FLOAT)→Mul(1个标量常量)
→Softmax(axis=-1)→Cast(to=FLOAT16) 的链。插件输入 [scores, scale常量],
输出沿用原 Cast_5 输出名 (下游 MatMul_1 零改动)。GEMM (QK^T / PV) 保留。
"""
import sys
import numpy as np
import onnx
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
    chain, cur = [n], n.output[0]
    for _ in range(4):
        cons = use.get(cur, [])
        assert len(cons) == 1, f"fanout at {cur}"
        chain.append(nodes[cons[0]])
        cur = chain[-1].output[0]
    mm, c4, mul, sm, c5 = chain
    # P1h lineage: Cast(to=f32) ... Cast(to=f16); P2h lineage f16-izes the
    # chain -> Cast(to=f16) identity on both ends. Accept both.
    assert c4.op_type == "Cast" and \
        [a.i for a in c4.attribute if a.name == "to"] in ([FLOAT],
                                                         [FLOAT16]), \
        f"{c4.name}: to={[a.i for a in c4.attribute if a.name == 'to']}"
    assert mul.op_type == "Mul"
    consts = [x for x in mul.input if x in init]
    assert len(consts) == 1, f"{mul.name}: consts {consts}"
    w = numpy_helper.to_array(init[consts[0]])
    assert w.size == 1, f"{consts[0]} not scalar: {w.shape}"
    assert sm.op_type == "Softmax" and \
        [a.i for a in sm.attribute if a.name == "axis"] == [-1]
    assert c5.op_type == "Cast" and \
        [a.i for a in c5.attribute if a.name == "to"] == [FLOAT16]
    et, dims = vi[mm.output[0]]
    assert et == TensorProto.FLOAT16, f"{mm.output[0]} dtype {et}"
    et5, dims5 = vi[c5.output[0]]
    assert et5 == TensorProto.FLOAT16, f"{c5.output[0]} dtype {et5}"
    consumers = [nodes[c].op_type for c in use.get(c5.output[0], [])]
    assert consumers == ["MatMul"], f"{c5.output[0]} -> {consumers}"
    sites.append((n.name, dims5, float(w.reshape(-1)[0]),
                  consts[0], c5.output[0]))
    removed.update(x.output[0] for x in chain[1:])

print(f"sites: {len(sites)}")
from collections import Counter
shapes = Counter(tuple(s[1]) for s in sites)
for shp, c in sorted(shapes.items()):
    print(f"  {list(shp)}: x{c} (scale {sorted(set(round(s[2],6) for s in sites if tuple(s[1])==shp))})")

# rebuild node list in original order (preserves topo): mm -> append plugin
# right after; skip the 4 chain nodes
new_nodes = []
for n in nodes:
    if n.output[0] in removed:
        continue
    new_nodes.append(n)
    if n.op_type == "MatMul" and n.name.endswith("/attn/inner_attn/MatMul"):
        site = next(s for s in sites if s[0] == n.name)
        pre = n.name[: n.name.rindex("/")]
        pn = helper.make_node(
            "ScaleSoftmax", [n.output[0], site[3]], [site[4]],
            name=f"{pre}/ScaleSoftmax")
        pn.domain = "SparseDrive"
        new_nodes.append(pn)
del g.node[:]
g.node.extend(new_nodes)

# strip stale value_info of removed tensors (plugin output reuses Cast_5's vi)
keep_vi = [v for v in g.value_info if v.name not in removed]
del g.value_info[:]
g.value_info.extend(keep_vi)

onnx.checker.check_model(m, False)
onnx.save(m, dst)

# structural sanity: every consumed tensor is produced / init / graph input
outs = set()
for n in g.node:
    outs.update(n.output)
gin = {x.name for x in g.input}
dangling = [t for n in g.node for t in n.input
            if t and t not in outs and t not in init and t not in gin]
assert not dangling, f"dangling inputs: {dangling[:5]}"
nplugin = sum(1 for n in g.node if n.op_type == "ScaleSoftmax")
print(f"removed {len(removed)} tensors, plugin nodes: {nplugin}, "
      f"saved {dst} ({len(g.node)} nodes)")
