# -*- coding: utf-8 -*-
"""P2h: head-wide fp16-ification on the deploy graph.

Input : work_dirs/sparsedrive_small_stage2/v5_P1h.onnx   (delivered graph)
Output: work_dirs/sparsedrive_small_stage2/v5_P2h.onnx

Root cause being fixed: T3 head-free-quant dequantized every head weight to
FP32 ("._deq" initializers).  The board's TRT 8.6.1.2 has a broken NVRTC
fp32 JIT, so every Myelin region that fused fp32 head ops degenerated to
tactic 0x0: all 5 map decoder layers + det layer-5/cls/quality fused into
~12.3 ms/iter of degenerate kernels, and det decoder layers 0-4 run as
plain fp32 layers.  This pass flips the whole head (nodes matching the T3
region PAT) to fp16, mirroring the kps-P1h fix at region scale:

  0. strip vestigial head QDQ pairs left by T3's head-free-quant (DQ
     output feeding only head-region float compute) -- otherwise the QDZ
     fence pins those consumers to f32 outside the region
  1. f32 initializers consumed by region nodes -> fp16 copies (originals
     pruned when no live consumer remains)
  2. in-region Cast(to=f32) whose consumers are all in-region -> flipped
     to Cast(to=f16) (no-op cast, folded by TRT)
  3. remaining f32 float tensors entering region nodes -> one shared
     Cast(to=f16) per source tensor, region consumers rewired only
  4. no-op casts inserted during earlier rounds (input already f16) get
     bypassed in a cleanup pass
  5. graph float outputs that would become fp16 -> Cast(to=f32) inserted
     so the engine IO contract (f32 bins / run_engine manifest / eval
     scripts) is unchanged

Dtype tracking is a single topological pass (ONNX node lists are
topologically ordered); the naive sweep version was O(depth x nodes) and
paged itself to death on this box.

Run deploy/fix_mixed_types.py afterwards as mop-up, then
deploy/verify_p2h_stub.py for the local numeric gate.
"""
import collections
import io
import os
import re
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
SRC = os.path.join(W, "v5_P1h.onnx")
DST = os.environ.get("P2H_DST", os.path.join(W, "v5_P2h.onnx"))
PAT = os.environ.get("P2H_PAT", r"layers\.|fc_before|fc_after")
F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT
DT_NAME = {F16: "f16", F32: "f32"}

PRESERVE = {
    "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Slice", "Identity",
    "Softmax", "Concat", "Mul", "Add", "Sub", "Div", "Pow", "Tanh",
    "Sigmoid", "Relu", "Erf", "Sqrt", "Reciprocal", "Neg", "Exp", "Log",
    "Max", "Min", "Pad", "Tile", "Expand", "Gather", "MatMul", "Gemm",
    "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "Flatten",
    "LeakyRelu", "HardSigmoid", "Shrink", "Sign", "Ceil", "Floor", "Round",
    "Where", "Dropout", "Clip", "ScatterND", "GatherElements", "Trilu",
    "Abs", "Split", "MaxPool",
}

print(f"loading {SRC} ...")
m = onnx.load(SRC)
g = m.graph
nodes = list(g.node)

# f16-ify EVERYTHING float except the INT8 QDQ zone (backbone/neck) and
# the graph-output boundary.  The zone is the full backward closure of
# every QuantizeLinear float input through producer edges (computed after
# the vestigial-head-pair strip below): Q/DQ, the INT8 convs, and ALL
# inter-Q glue (Conv/BN/Relu/MaxPool/Add/Resize chains) -- a one-level
# "output feeds Q" fence is not enough, it let the BN right after an INT8
# conv escape into the fp16 region and poison the next Q.  After T3 the
# head has no QDQ left, so the closure is exactly backbone/neck.
prod0 = {}
cons0 = {}
for n in nodes:
    for o in n.output:
        prod0[o] = n
    for x in n.input:
        if x:
            cons0.setdefault(x, []).append(n)
dq_out = set()
for n in nodes:
    if n.op_type == "DequantizeLinear":
        dq_out.add(n.output[0])

# --- strip vestigial QDQ pairs left over inside the head from T3's
# head-free-quant (3 weight quantizers on the layers.0 q/k/v/kps MatMuls,
# whose Q input is the f32 dequantized weight initializer, plus any
# activation leftovers).  A pair is stripped when EVERY consumer of its
# DQ output is head-region float compute (matches PAT) with no DQ-sourced
# inputs of its own -- the fake-quant only pins those consumers to f32
# (and QDZ-fences them out of the region).  Anything feeding the real
# INT8 zone stays untouched.  Must run before the QDZ classification so
# the freed consumers rejoin the fp16 region. ---
init_names0 = {i.name for i in g.initializer}
graph_outs0 = {o.name for o in g.output}
removed_names = set()
stripped_dq = stripped_q = 0
for q in list(nodes):
    if q.op_type != "QuantizeLinear" or not q.output:
        continue
    x = q.input[0]
    qout = q.output[0]
    dqs = [d for d in cons0.get(qout, [])
           if d.op_type == "DequantizeLinear"
           and d.input[1] == q.input[1]
           and (len(q.input) < 3 or d.input[2] == q.input[2])]
    for d in dqs:
        dout = d.output[0]
        if dout in graph_outs0:
            continue
        users = cons0.get(dout, [])
        if not users:
            continue
        if not all(re.search(PAT, c.name) for c in users):
            continue                   # outside the head - keep
        if not all(c.op_type not in ("QuantizeLinear", "DequantizeLinear")
                   and not any(xx in dq_out
                               for xx in c.input if xx and xx != dout)
                   for c in users):
            print(f"  keep head QDQ pair (non-float consumers): {q.name}")
            continue
        for c in users:
            for k, xx in enumerate(c.input):
                if xx == dout:
                    c.input[k] = x
        g.node.remove(d)
        removed_names.add(d.name)
        stripped_dq += 1
        print(f"  stripped {q.name} -> {d.name} "
              f"(Q src: {'INIT ' + x if x in init_names0 else x}; "
              f"users: {[c.name for c in users]})")
    if qout not in graph_outs0 and cons0.get(qout) and all(
            c.name in removed_names for c in cons0.get(qout, [])):
        g.node.remove(q)
        removed_names.add(q.name)
        stripped_q += 1
print(f"vestigial head QDQ pairs stripped: {stripped_dq} dq, "
      f"{stripped_q} q")

# --- zone = backward closure of every remaining Q's float inputs ---
zone = set()
stack = [n for n in nodes if n.op_type == "QuantizeLinear"]
while stack:
    n = stack.pop()
    if n.name in zone:
        continue
    zone.add(n.name)
    for x in n.input:
        if not x or x in init_names0:
            continue
        p = prod0.get(x)
        if p is not None and p.name not in zone:
            stack.append(p)

region = [n for n in nodes
          if n.name not in zone
          and n.op_type not in ("QuantizeLinear", "DequantizeLinear")]
region_names = {n.name for n in region}
print(f"region nodes (outside QDQ zone closure): {len(region)} "
      f"of {len(nodes)}, zone: {len(zone)}")

init_dtype = {}
for i in g.initializer:
    init_dtype[i.name] = i.data_type


def infer_dtypes():
    """single topological pass; ONNX node lists are topo-ordered"""
    dt = {}
    for vi in list(g.value_info) + list(g.output) + list(g.input):
        t = vi.type.tensor_type.elem_type
        if t in DT_NAME:
            dt[vi.name] = t
    for name, t in init_dtype.items():
        dt[name] = t
    for n in g.node:
        if n.op_type == "Cast":
            to = next(a.i for a in n.attribute if a.name == "to")
            if to in DT_NAME:
                dt[n.output[0]] = to
            continue
        if n.op_type == "DequantizeLinear":
            t = dt.get(n.input[1])
            if t in DT_NAME:
                dt[n.output[0]] = t
            continue
        if n.op_type == "Constant":
            continue
        if n.op_type == "ConstantOfShape":
            t = F32
            for a in n.attribute:
                if a.name == "value":
                    t = a.t.data_type
            for o in n.output:
                if o:
                    dt[o] = t
            continue
        if n.op_type == "TopK":
            if n.input and dt.get(n.input[0]) in DT_NAME:
                dt[n.output[0]] = dt[n.input[0]]
            continue
        if n.op_type == "DeformableAggregation":
            # plugin contract: feat input forced kHALF -> output kHALF
            for o in n.output:
                if o:
                    dt[o] = F16
            continue
        if n.op_type in PRESERVE:
            fts = {t for t in (dt.get(x) for x in n.input if x)
                   if t in DT_NAME}
            if len(fts) == 1:
                t = next(iter(fts))
                for o in n.output:
                    if o:
                        dt[o] = t
    return dt


def topo_sort():
    prod = {}
    for n in g.node:
        for o in n.output:
            prod[o] = n
    consumer_of = collections.defaultdict(list)
    node_by_id = {id(n): n for n in g.node}
    waiting = {}
    for n in g.node:
        waiting[id(n)] = len({x for x in n.input if x and x in prod})
    for n in g.node:
        for x in n.input:
            if x and x in prod:
                consumer_of[x].append(id(n))
    ready = collections.deque(n for n in g.node if waiting[id(n)] == 0)
    placed = []
    while ready:
        n = ready.popleft()
        placed.append(n)
        for o in n.output:
            for cid in consumer_of.get(o, ()):
                waiting[cid] -= 1
                if waiting[cid] == 0:
                    ready.append(node_by_id[cid])
    if len(placed) != len(g.node):
        raise RuntimeError(f"topo sort stuck: {len(placed)}/{len(g.node)}")
    del g.node[:]
    g.node.extend(placed)
    return {n.name for n in g.node if n.op_type == "Cast"}


def producer_map():
    prod = {}
    cons = collections.defaultdict(list)
    for n in g.node:
        for o in n.output:
            prod[o] = n
        for x in n.input:
            if x:
                cons[x].append(n)
    return prod, cons


new_inits = []
inserted_casts = []          # (node, src_tensor)
conv_cache = {}              # init name -> f16 copy name
cast_cache = {}              # source tensor -> f16 cast output name
n_conv = n_cast = n_flip = 0


def plugin_idx_pin(node, k):
    # DFA plugin accepts only kFLOAT/kINT32 on inputs 1,2
    # (spatial_shape / scale_start_index) -- never fp16-ify those
    return (node.op_type == "DeformableAggregation" and k in (1, 2))


for round_i in range(12):
    dt = infer_dtypes()
    prod, cons = producer_map()
    r_conv = r_cast = r_flip = 0

    # --- pass A: f32 initializers feeding region nodes ---
    for n in region:
        for k, x in enumerate(n.input):
            if plugin_idx_pin(n, k):
                continue
            if x and init_dtype.get(x) == F32:
                if x not in conv_cache:
                    arr = numpy_helper.to_array(
                        next(i for i in g.initializer if i.name == x))
                    nm = x + "._h16"
                    new_inits.append(numpy_helper.from_array(
                        arr.astype(np.float16), nm))
                    conv_cache[x] = nm
                    init_dtype[nm] = F16
                    n_conv += 1
                    r_conv += 1
                n.input[k] = conv_cache[x]

    # --- pass B/C: f32 tensors feeding region nodes ---
    for n in region:
        for k, x in enumerate(n.input):
            if plugin_idx_pin(n, k):
                continue
            if not x or x in init_dtype or dt.get(x) != F32:
                continue
            p = prod.get(x)
            if (p is not None and p.name in region_names
                    and p.op_type == "Cast"
                    and next(a.i for a in p.attribute if a.name == "to") == F32
                    and all(c.name in region_names for c in cons.get(x, []))):
                for a in p.attribute:
                    if a.name == "to":
                        a.i = F16
                n_flip += 1
                r_flip += 1
                continue
            if x not in cast_cache:
                cname = x + "_to16"
                node = helper.make_node("Cast", [x], [cname], to=F16,
                                        name=cname + "_cast")
                inserted_casts.append(node)
                cast_cache[x] = cname
                n_cast += 1
                r_cast += 1
            if n.input[k] == x:
                n.input[k] = cast_cache[x]

    print(f"round {round_i}: conv={r_conv} cast={r_cast} flip={r_flip}")
    if not (r_conv or r_cast or r_flip):
        print("fixpoint reached")
        break
else:
    print("WARNING: no fixpoint after 12 rounds")

g.node.extend(inserted_casts)
g.initializer.extend(new_inits)
topo_sort()

# --- cleanup: no-op casts (source already f16) get their consumers
# rewired to src, then are removed.  g.node.extend() COPIES protobuf
# messages, so inserted_casts holds orphaned clones: match casts inside
# g.node by name for the rewiring, and remove via output-tensor use
# counts. ---
dt = infer_dtypes()
prod, cons = producer_map()
cast_names = {node.name for node in inserted_casts}
bypassed = 0
for node in g.node:
    if node.op_type != "Cast" or node.name not in cast_names:
        continue
    src, out = node.input[0], node.output[0]
    if dt.get(src) != F16:
        continue
    for c in cons.get(out, []):
        for k, x in enumerate(c.input):
            if x == out:
                c.input[k] = src
    bypassed += 1
print(f"no-op casts bypassed: {bypassed}")
consumed = {x for n in g.node for x in n.input if x}
dead_names = {node.name for node in inserted_casts
              if dt.get(node.input[0]) == F16
              and node.output[0] not in consumed}
if dead_names:
    keep = [n for n in g.node if n.name not in dead_names]
    del g.node[:]
    g.node.extend(keep)
    print(f"dead cast nodes removed: {len(dead_names)}")

# --- graph-output boundary: f16 region outputs that ARE graph outputs get
# one Cast(f32) so the engine IO contract (f32 bins / run_engine manifest /
# eval scripts) is unchanged; everything downstream is region-internal ---
dt = infer_dtypes()
prod, cons = producer_map()
n_out = 0
for o in list(g.output):
    if o.type.tensor_type.elem_type != F32:
        continue
    t = o.name
    if dt.get(t) != F16:
        continue
    p = prod[t]
    oi = list(p.output).index(t)
    pre = t + ".pre_o"
    for c in cons.get(t, []):
        for k, xx in enumerate(c.input):
            if xx == t:
                c.input[k] = pre
    p.output[oi] = pre
    cast = helper.make_node("Cast", [pre], [t], to=F32,
                            name=t + "_o32_cast")
    pos = next(i for i, nn in enumerate(g.node) if nn.name == p.name)
    g.node.insert(pos + 1, cast)
    n_out += 1
print(f"graph f32 outputs restored from f16 producers: {n_out}")

# --- prune unreferenced initializers ---
live = set()
for n in g.node:
    live.update(n.input)
live.update(v.name for v in list(g.input) + list(g.output))
inis = [i for i in g.initializer if i.name in live]
print(f"initializers pruned: {len(g.initializer) - len(inis)}")
del g.initializer[:]
g.initializer.extend(inis)

# --- final verification report ---
dt = infer_dtypes()
residue = []
for n in g.node:
    if n.name not in region_names:
        continue
    fts = {dt.get(x) for x in n.input if x and dt.get(x) in DT_NAME}
    if F32 in fts:
        residue.append((n.op_type, n.name))
print(f"converted initializers: {n_conv}, casts inserted: {n_cast} "
      f"(bypassed {bypassed}), flipped casts: {n_flip}")
print(f"region nodes still consuming f32: {len(residue)}")
for op, nm in residue[:15]:
    print(f"   RESIDUE {op} {nm}")

print("graph outputs:",
      [(o.name, DT_NAME.get(o.type.tensor_type.elem_type,
                            str(o.type.tensor_type.elem_type)))
       for o in g.output])

# --- drop stale intermediate value_info: T3/P1h left f32 declarations on
# tensors this pass flipped to f16 (540 of them), and ORT's load-time type
# check rejects the mismatch.  value_info is optional metadata -- ORT and
# TRT infer shapes themselves; the graph IO contract lives in g.input /
# g.output and is untouched. ---
n_vi = len(g.value_info)
del g.value_info[:]
print(f"intermediate value_info cleared: {n_vi}")

onnx.checker.check_model(m, False)
onnx.save(m, DST)
print(f"saved {DST}  size={os.path.getsize(DST)/1e6:.1f} MB")

