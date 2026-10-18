# -*- coding: utf-8 -*-
"""P4-6: second-stage split of sp_head.onnx:
  sp_head_det.onnx  : det decoder stack (upward closure of the decoder's
                      final instance feature / anchor embed outputs)
  sp_head_rest.onnx : map decoder + ego + cls/bbox/quality heads + state
                      updates (consumes sp_head_det outputs via boundary)
The det stack built standalone ran ~1.8ms in P1's module experiment
(modbig_p2h) vs ~11.5ms of degenerate ForeignNodes inside the full head -
Myelin falls back on high-rank tensors in large graphs (o5 log: rank
assertion + NVRTC failure)."""
import collections
import io
import os
import sys

import onnx
from onnx import TensorProto, helper

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
# argv[1]=src sp_head graph, argv[2]=output tag (default: P4 originals)
SRC = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 \
    else os.path.join(W, "sp_head.onnx")
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
OUT_DET = os.path.join(W, f"sp_head_det{TAG}.onnx")
OUT_REST = os.path.join(W, f"sp_head_rest{TAG}.onnx")

ELEM = {TensorProto.FLOAT: "f32", TensorProto.FLOAT16: "f16",
        TensorProto.INT32: "i32", TensorProto.INT64: "i64",
        TensorProto.BOOL: "b8"}

print(f"loading {SRC} ...")
m = onnx.load(SRC)
g = m.graph
nodes = list(g.node)
N = len(nodes)
prod, cons = {}, collections.defaultdict(list)
node_by_name = {}
for i, n in enumerate(nodes):
    node_by_name[n.name] = i
    for o in n.output:
        prod[o] = i
    for x in n.input:
        if x:
            cons[x].append(i)
init_names = {x.name for x in g.initializer}
graph_inputs = [v.name for v in g.input if v.name not in init_names]
print(f"nodes {N}, graph inputs {len(graph_inputs)}")

mi = onnx.shape_inference.infer_shapes(m)
vi = {}
for src in (g.value_info, g.input, g.output,
            mi.graph.value_info, mi.graph.input, mi.graph.output):
    for v in src:
        tt = v.type.tensor_type
        dims = [d.dim_value if d.WhichOneof("value") == "dim_value" else -1
                for d in tt.shape.dim]
        vi[v.name] = (tt.elem_type, dims)

# ---- det stack: upward closure of the det decoder's final states ----
ANCHORS = ["det_instance_feature", "det_anchor_embed"]
for a in ANCHORS:
    assert a in prod, f"anchor output {a} not found"
need = set()
stack = list(ANCHORS)
while stack:
    t = stack.pop()
    p = prod.get(t)
    if p is None or p in need:
        continue
    need.add(p)
    for x in nodes[p].input:
        if x and x not in init_names:
            stack.append(x)
det = set(need)
rest = {i for i in range(N)} - det
print(f"det stack {len(det)} nodes, rest {len(rest)} nodes")
dfas = [i for i in sorted(det) if nodes[i].op_type == "DeformableAggregation"]
print(f"DFA calls in det stack: {len(dfas)} "
      f"{[nodes[i].name.split('/')[1] for i in dfas]}")
dfas_all = sum(1 for n in nodes if n.op_type == "DeformableAggregation")
print(f"DFA calls total: {dfas_all} (rest keeps {dfas_all - len(dfas)})")

# boundary: det-produced tensors consumed by rest
boundary = sorted({o for i in det for o in nodes[i].output
                   if any(j in rest for j in cons.get(o, ()))})
print(f"boundary tensors: {len(boundary)}")
for t in boundary:
    et, dims = vi.get(t, (0, []))
    nb = [nodes[j].name.split('/')[-1] for j in cons.get(t, ()) if j in rest]
    print(f"  {ELEM.get(et, et)}{dims} {t}  -> {nb[:4]}")
assert all(prod.get(t) is not None for t in boundary)

# rest must not feed det (pure ancestry check)
bad = [(i, x) for i in det for x in nodes[i].input
       if x and x not in init_names and prod.get(x) is not None
       and prod[x] in rest]
print(f"det consuming rest tensors: {len(bad)}")
assert not bad, "cycle: det stack depends on rest"

# rest coverage: every rest node reachable from rest roots
reach = set()
reach_t = set(boundary) | set(graph_inputs) | init_names
st = list(reach_t)
while st:
    t = st.pop()
    for j in cons.get(t, ()):
        if j in reach or j in det:
            continue
        reach.add(j)
        for o in nodes[j].output:
            if o not in reach_t:
                reach_t.add(o)
                st.append(o)
orphans = [i for i in rest if i not in reach]
print(f"rest coverage {len(rest & reach)}/{len(rest)}; "
      f"orphans {[nodes[i].name or i for i in orphans[:5]]}")
assert not orphans, "rest orphans"

# ---- emit ----
def sub_graph(idxs, sub_inputs, sub_outputs, keep_inits, name):
    ng = onnx.GraphProto()
    for i in sorted(idxs):
        ng.node.extend([nodes[i]])
    for nm in sub_inputs:
        gi = next((v for v in g.input if v.name == nm), None)
        if gi is not None:
            ng.input.append(gi)
        else:
            et, dims = vi.get(nm, (TensorProto.FLOAT, []))
            ng.input.append(helper.make_tensor_value_info(nm, et, dims))
    for nm in sub_outputs:
        et, dims = vi.get(nm, (TensorProto.FLOAT, []))
        ng.output.append(helper.make_tensor_value_info(nm, et, dims))
    ng.initializer.extend(x for x in g.initializer if x.name in keep_inits)
    ng.name = name
    return ng

det_inits = {x for i in det for x in nodes[i].input if x in init_names}
rest_inits = {x for i in rest for x in nodes[i].input if x in init_names}
# rest graph inputs = original graph inputs that rest consumes directly
rest_gin = sorted({x for i in rest for x in nodes[i].input
                   if x in graph_inputs})
det_gin = sorted({x for i in det for x in nodes[i].input
                  if x in graph_inputs})
print(f"det graph inputs: {det_gin}")
print(f"rest graph inputs: {rest_gin}")

md = onnx.ModelProto()
md.CopyFrom(m)
outs_det = [o.name for o in g.output if prod.get(o.name) in det]
outs_rest = [o.name for o in g.output if prod.get(o.name) in rest]
print(f"graph outputs from det: {outs_det}")
print(f"graph outputs from rest: {len(outs_rest)}")
det_out_all = sorted(set(boundary) | set(outs_det))
# scalar-declared det internals whose true shapes shape-inference cannot
# track (they leave the det stack to the rest engine); verified from
# consumer weight shapes (cls/bbox heads, anchor_encoder)
OVERRIDE = {"/layers.38/Add_output_0": (TensorProto.FLOAT16, [1, 900, 256]),
            "det_bbox.pre_o": (TensorProto.FLOAT16, [1, 900, 11]),
            "det_instance_feature.pre_o": (TensorProto.FLOAT16,
                                           [1, 900, 256]),
            "det_anchor_embed.pre_o": (TensorProto.FLOAT16, [1, 900, 256])}
vi.update(OVERRIDE)
md.graph.CopyFrom(sub_graph(det, det_gin, det_out_all, det_inits,
                            "sp_head_det"))
onnx.checker.check_model(md, False)
onnx.save(md, OUT_DET)
print(f"saved {OUT_DET}")
for t in boundary:
    print(f"  boundary {t}: {ELEM.get(vi[t][0], vi[t][0])}{vi[t][1]}")

mr = onnx.ModelProto()
mr.CopyFrom(m)
ng = sub_graph(rest, rest_gin, outs_rest, rest_inits, "sp_head_rest")
# rest inputs: original protos + det boundary
inputs = [(nm, True) for nm in rest_gin] + [(nm, False) for nm in boundary]
del ng.input[:]
for nm, is_gi in inputs:
    if is_gi:
        ng.input.append(next(v for v in g.input if v.name == nm))
    else:
        et, dims = vi[nm]
        ng.input.append(helper.make_tensor_value_info(nm, et, dims))
mr.graph.CopyFrom(ng)
onnx.checker.check_model(mr, False)
onnx.save(mr, OUT_REST)
print(f"saved {OUT_REST}")
print(f"sizes: det={os.path.getsize(OUT_DET)/1e6:.1f}MB "
      f"rest={os.path.getsize(OUT_REST)/1e6:.1f}MB")
