# -*- coding: utf-8 -*-
"""P4-2: split v5_P2h.onnx into
  sp_backbone.onnx : img -> neck feature maps   (INT8 QDQ zone intact)
  sp_head.onnx     : feature maps + prev/metas -> 19 outputs  (pure fp16)

Dataflow classification keyed by NODE INDEX (the graph carries unnamed
nodes - name-keyed sets collide on ""):
  img_reach = forward closure from graph input `img`
  HEAD      = fixpoint {nodes consuming a non-img graph input or a HEAD
              output}  U  {nodes not in img_reach}
  BB        = img_reach \\ HEAD
  boundary  = BB-produced tensors consumed by HEAD nodes"""
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
# argv[1]=src graph, argv[2]=output tag (default: P4 originals, untouched)
SRC = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 \
    else os.path.join(W, "v5_P2h.onnx")
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
OUT_BB = os.path.join(W, f"sp_backbone{TAG}.onnx")
OUT_HD = os.path.join(W, f"sp_head{TAG}.onnx")

ELEM = {TensorProto.FLOAT: "f32", TensorProto.FLOAT16: "f16",
        TensorProto.INT32: "i32", TensorProto.INT64: "i64",
        TensorProto.INT8: "i8", TensorProto.BOOL: "bool"}

print(f"loading {SRC} ...")
m = onnx.load(SRC)
g = m.graph
nodes = list(g.node)
N = len(nodes)
prod = {}
cons = collections.defaultdict(list)
for i, n in enumerate(nodes):
    for o in n.output:
        prod[o] = i
    for x in n.input:
        if x:
            cons[x].append(i)
init_names = {i.name for i in g.initializer}
graph_inputs = [v.name for v in g.input if v.name not in init_names]
unnamed = sum(1 for n in nodes if not n.name)
print(f"nodes {N} (unnamed {unnamed}), inits {len(g.initializer)}")

vi = {}
for src in (g.value_info, g.input, g.output):
    for v in src:
        tt = v.type.tensor_type
        dims = [d.dim_value if d.WhichOneof("value") == "dim_value" else -1
                for d in tt.shape.dim]
        vi[v.name] = (tt.elem_type, dims)
# v5_P2h carries no value_info for most intermediates; one inference pass on
# the full graph annotates them (boundary tensors are internal there, so
# their shapes come out concrete)
mi = onnx.shape_inference.infer_shapes(m)
for v in list(mi.graph.value_info) + list(mi.graph.output):
    tt = v.type.tensor_type
    dims = [d.dim_value if d.WhichOneof("value") == "dim_value" else -1
            for d in tt.shape.dim]
    vi[v.name] = (tt.elem_type, dims)

# ---- img_reach ----
img_reach = set()
stack = list(cons.get("img", ()))
while stack:
    i = stack.pop()
    if i in img_reach:
        continue
    img_reach.add(i)
    for o in nodes[i].output:
        stack.extend(j for j in cons.get(o, ()) if j not in img_reach)

# ---- HEAD fixpoint ----
head_roots_t = {x for x in graph_inputs if x != "img"}
head_fwd = set()
q = collections.deque(i for i in range(N)
                      if any(x in head_roots_t for x in nodes[i].input))
q.extend(prod[o.name] for o in g.output if o.name in prod
         and prod[o.name] not in head_fwd)
head_fwd.update(q)
while q:
    i = q.popleft()
    for o in nodes[i].output:
        for j in cons.get(o, ()):
            if j not in head_fwd:
                head_fwd.add(j)
                q.append(j)
# bb: ALL ancestors of the feature handoff tensor (Reshape_9_output_0)
# within img_reach. The ego branch (graph-output-only side path) is then
# NOT in bb - it runs in the fp16 head engine, where it fuses properly
# (in the int8-flagged bb engine it degenerated into a 1.9ms ForeignNode).
ANCHOR = "/Reshape_9_output_0"
assert ANCHOR in prod, f"anchor tensor {ANCHOR} not found"
need_t = [ANCHOR]
need_n = set()
while need_t:
    t = need_t.pop()
    p = prod.get(t)
    if p is None or p in need_n:
        continue
    need_n.add(p)
    for x in nodes[p].input:
        if x and x not in init_names and x != "img":
            need_t.append(x)
bb = set(need_n)          # full ancestry: img-side compute + weight Q/DQ
                          # prep (Q/DQ consume inits only - cannot pull in
                          # head data; filtering by img_reach would drop
                          # them and strand bb convs without weights)
bb_core = {i for i in bb}
head = {i for i in range(N)} - bb
print(f"img_reach {len(img_reach)}, bb (anchor ancestry) {len(bb)}, "
      f"HEAD {len(head)}")

# ---- boundary: bb-produced tensors consumed by head ----
boundary = sorted({o for i in bb for o in nodes[i].output
                   if any(j in head for j in cons.get(o, ()))})
print(f"boundary tensors: {len(boundary)}")
for t in boundary:
    et, dims = vi.get(t, (0, []))
    nb = sum(1 for j in cons.get(t, ()) if j in head)
    p = nodes[prod[t]]
    print(f"  {ELEM.get(et, et)}{dims}  {t}\n"
          f"    via [{prod[t]}] {p.op_type} {p.name!r}  head_consumers={nb}")
assert all(prod[t] in img_reach for t in boundary), \
    "boundary producer not img-reachable"

# ---- checks ----
reach = set()
reach_t = set(boundary) | head_roots_t | init_names
stack = list(reach_t)
while stack:
    t = stack.pop()
    for j in cons.get(t, ()):
        if j in reach:
            continue
        reach.add(j)
        for o in nodes[j].output:
            if o not in reach_t:
                reach_t.add(o)
                stack.append(o)
orphans = [i for i in head if i not in reach]
print(f"head coverage {len(head & reach)}/{len(head)}; "
      f"orphans: {[nodes[i].name or f'#{i}' for i in orphans[:5]]}")
if orphans:
    for i in orphans[:3]:
        n = nodes[i]
        print(f"  orphan #{i} {n.op_type} {n.name!r} in={list(n.input)}")
    sys.exit("FATAL: head orphans")
qdq_head = [i for i in head if nodes[i].op_type in ("QuantizeLinear",
                                                    "DequantizeLinear")]
print(f"Q/DQ in HEAD: {len(qdq_head)}")
# no BB node may consume a head-produced tensor (only crossing = boundary)
bad = [(i, x) for i in bb for x in nodes[i].input
       if x and x not in init_names and x != "img"
       and prod.get(x) is not None and prod[x] in head]
print(f"BB consuming head tensors: {len(bad)} {bad[:3]}")
if bad:
    sys.exit("FATAL: second crossing")

# ---- emit ----
def sub_graph(idxs, sub_inputs, sub_outputs, keep_inits):
    ng = onnx.GraphProto()
    for i in sorted(idxs):                      # original topo order
        ng.node.extend([nodes[i]])              # extend copies; fine once
    for name in sub_inputs:
        gi = next((v for v in g.input if v.name == name), None)
        if gi is not None:
            ng.input.append(gi)
        else:
            et, dims = vi.get(name, (TensorProto.FLOAT, []))
            ng.input.append(helper.make_tensor_value_info(name, et, dims))
    for name in sub_outputs:
        et, dims = vi.get(name, (TensorProto.FLOAT, []))
        ng.output.append(helper.make_tensor_value_info(name, et, dims))
    ng.initializer.extend(x for x in g.initializer if x.name in keep_inits)
    return ng

bb_inits = {x for i in bb for x in nodes[i].input if x in init_names}
hd_inits = {x for i in head for x in nodes[i].input if x in init_names}
hd_inputs = [v.name for v in g.input
             if v.name not in init_names and v.name != "img"]

mb = onnx.ModelProto()
mb.CopyFrom(m)
mb.graph.CopyFrom(sub_graph(bb, ["img"], boundary, bb_inits))
mb.graph.name = "sp_backbone"
onnx.checker.check_model(mb, False)
bb_shapes = {t: vi[t] for t in boundary}
for t, (et, dims) in bb_shapes.items():
    print(f"bb output {t}: {ELEM.get(et, et)}{dims}")
assert bb_shapes.get(ANCHOR, (0, []))[0] == TensorProto.FLOAT16, \
    "feature boundary must be f16"
assert all(et in (TensorProto.FLOAT16, TensorProto.FLOAT)
           for et, _ in bb_shapes.values()), "unexpected boundary dtype"
onnx.save(mb, OUT_BB)
print(f"saved {OUT_BB}")

mb2 = onnx.ModelProto()
mb2.CopyFrom(m)
ng = sub_graph(head, hd_inputs, [o.name for o in g.output], hd_inits)
del ng.output[:]                     # original decls verbatim (dtypes)
ng.output.extend(g.output)
# head inputs: original protos for graph inputs + concrete bb outputs
del ng.input[:]
for name in hd_inputs:
    ng.input.append(next(v for v in g.input if v.name == name))
for name in boundary:
    et, dims = bb_shapes[name]
    ng.input.append(helper.make_tensor_value_info(name, et, dims))
mb2.graph.CopyFrom(ng)
mb2.graph.name = "sp_head"
onnx.checker.check_model(mb2, False)
onnx.save(mb2, OUT_HD)
print(f"saved {OUT_HD}")
print(f"sizes: bb={os.path.getsize(OUT_BB)/1e6:.0f}MB "
      f"hd={os.path.getsize(OUT_HD)/1e6:.0f}MB "
      f"(src {os.path.getsize(SRC)/1e6:.0f}MB)")

print("\n== sp_backbone: img ->", [t.split('/')[-1][:40] for t in boundary])
