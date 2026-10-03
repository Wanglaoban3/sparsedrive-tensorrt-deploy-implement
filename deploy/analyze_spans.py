# -*- coding: utf-8 -*-
"""Analyze e_hd ForeignNode spans: locate each span's node range in
sp_head.onnx, census op types / dtypes / MatMul FLOPs inside.
Usage: python analyze_spans.py <e_hd_prof.log>"""
import collections
import io
import os
import re
import sys

import onnx
from onnx import TensorProto

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
HD = os.path.join(W, "sp_head.onnx")

prof_path = sys.argv[1]
if not os.path.exists(prof_path):
    prof_path = os.path.join(r"<REPO>"
                             r"-implement\evaldata\mini_eng_v8",
                             os.path.basename(prof_path))
m = onnx.load(HD)
g = m.graph
nodes = list(g.node)
prod, cons = {}, collections.defaultdict(list)
node_by_name = {}
for i, n in enumerate(nodes):
    node_by_name[n.name] = i
    for o in n.output:
        prod[o] = i
    for x in n.input:
        if x:
            cons[x].append(i)
inits = {x.name for x in g.initializer}
init_cons = collections.defaultdict(list)
for i, n in enumerate(nodes):
    for x in n.input:
        if x in inits:
            init_cons[x].append(i)
msh = onnx.shape_inference.infer_shapes(m)
vi = {}
for src in (msh.graph.value_info, msh.graph.input, msh.graph.output):
    for v in src:
        tt = v.type.tensor_type
        dims = [d.dim_value if d.WhichOneof("value") == "dim_value" else -1
                for d in tt.shape.dim]
        vi[v.name] = (tt.elem_type, dims)
ELEM = {TensorProto.FLOAT: "f32", TensorProto.FLOAT16: "f16",
        TensorProto.INT32: "i32", TensorProto.INT64: "i64",
        TensorProto.BOOL: "b8", TensorProto.INT8: "i8"}

# ---- parse profile rows ----
# rows look like:
# [10/03/2026-09:24:34] [I]      13.07       0.1021  ...  {ForeignNode[a...b]}
row_re = re.compile(r"\[I\]\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.+)")
rows = []
for line in open(prof_path, encoding="utf-8", errors="replace"):
    mm = row_re.search(line)
    if not mm:
        continue
    total_ms = float(mm.group(1))
    name = mm.group(5).strip()
    if "ForeignNode" not in name:
        continue
    rows.append((total_ms, name))
rows.sort(key=lambda r: -r[0])
print(f"{len(rows)} ForeignNode rows, top 12 (total ms over 100 iters):")
for ms, nm in rows[:12]:
    print(f"  {ms:7.3f} ms  {nm[:150]}")

# ---- locate spans ----
tok_re = re.compile(r"[A-Za-z0-9_./\-\[\]]{4,}")


def span_range(label):
    label = label.replace("...", " ")
    idxs = []
    for tok in tok_re.findall(label):
        hit = None
        for cand in (tok, tok + "_output_0",
                     tok.replace("_output_0", "")):
            if cand in prod:
                hit = prod[cand]
                break
            if cand in node_by_name:
                hit = node_by_name[cand]
                break
        if hit is None and tok in init_cons:
            hit = init_cons[tok][0]
        if hit is not None:
            idxs.append(hit)
    return (min(idxs), max(idxs)) if idxs else None


def flops_mm(op):
    # MatMul/Gemm FLOPs: 2 * batch * M * K * N
    a = vi.get(op.input[0], (0, []))[1]
    b = vi.get(op.input[1], (0, []))[1] if len(op.input) > 1 else []
    if (not a or not b or -1 in a or -1 in b or len(a) < 2):
        return 0
    batch = 1
    for d in a[:-2]:
        batch *= d
    return 2 * batch * a[-2] * a[-1] * b[-1]


print("\nspan census (located spans only):")
for ms, nm in rows[:12]:
    rng = span_range(nm)
    if rng is None:
        print(f"  {ms:7.3f} ms  LOCATE-FAIL  {nm[:120]}")
        continue
    lo, hi = rng
    span = nodes[lo:hi + 1]
    ops = collections.Counter(n.op_type for n in span)
    nflt = sum(1 for n in span for x in n.input
               if x in vi and vi[x][0] == TensorProto.FLOAT)
    nf16 = sum(1 for n in span for x in n.input
               if x in vi and vi[x][0] == TensorProto.FLOAT16)
    fl = sum(flops_mm(n) for n in span if n.op_type in ("MatMul", "Gemm"))
    big = sorted(((int(np.prod([d for d in vi[o][1] if d > 0]) or 0) *
                   (2 if vi[o][0] == TensorProto.FLOAT16 else 4), o)
                  for n in span for o in n.output if o in vi),
                 reverse=True)[:4]
    print(f"  {ms:7.3f} ms  nodes[{lo}..{hi}] n={len(span)} "
          f"f32in={nflt} f16in={nf16} MM_FLOP={fl/1e9:.2f}G")
    print(f"      ops: {dict(ops)}")
    print(f"      big tensors: " + ", ".join(
        f"{sz/1e6:.1f}MB {ELEM.get(vi[o][0])}{vi[o][1]} {o[-60:]}"
        for sz, o in big))
