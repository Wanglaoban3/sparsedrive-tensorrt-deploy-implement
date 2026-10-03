# -*- coding: utf-8 -*-
"""Extract a decoder-region span as a standalone ONNX module for cheap
board-side compile tests (user directive: iterate on modules, not the full
engine).

For a span of NB nodes ending at the node whose output matches END_SUFFIX:
  - graph inputs  = span-consumed tensors produced outside the span
  - graph outputs = span-produced tensors consumed outside the span
  - initializers  = those referenced by span nodes
Shapes/values for the boundary tensors come from one full-graph stub-ORT
run, which also saves boundary values to <out>.inputs.npz for the module's
numeric self-check and later on-board input generation.

Usage:
  python deploy/extract_region.py --src .../v5_P1h.onnx \
      --end-suffix "/layers.7/Reshape_3" --nodes-back 401 \
      --out .../mod_p1h.onnx
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dfastub import stub  # noqa: E402

DT_NAME = {TensorProto.FLOAT: "f32", TensorProto.FLOAT16: "f16",
           TensorProto.INT32: "i32", TensorProto.INT64: "i64",
           TensorProto.BOOL: "bool", TensorProto.INT8: "i8"}
NAME_DT = {"float32": TensorProto.FLOAT, "float16": TensorProto.FLOAT16,
           "int32": TensorProto.INT32, "int64": TensorProto.INT64,
           "bool": TensorProto.BOOL, "int8": TensorProto.INT8}


def ort_of(d):
    return [int(x) for x in d]


ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--end-suffix", required=True)
ap.add_argument("--start-suffix", default=None,
                help="name-anchored span start (stable across graphs that "
                     "differ in inserted/removed nodes); falls back to "
                     "--nodes-back when omitted")
ap.add_argument("--nodes-back", type=int, default=401)
ap.add_argument("--out", required=True)
ap.add_argument("--skip-numeric", action="store_true")
args = ap.parse_args()

print(f"loading {args.src} ...")
m = onnx.load(args.src)
g = m.graph
nodes = list(g.node)


def find_idx(suffix):
    for i, n in enumerate(nodes):
        if n.name == suffix or any(o == suffix for o in n.output):
            return i
    hit = None
    for i, n in enumerate(nodes):
        if any(o.endswith(suffix) for o in n.output) or \
                n.name.endswith(suffix):
            hit = i
    return hit


hi = find_idx(args.end_suffix)
if hi is None:
    raise SystemExit(f"end suffix not found: {args.end_suffix}")
if args.start_suffix:
    lo = find_idx(args.start_suffix)
    if lo is None:
        raise SystemExit(f"start suffix not found: {args.start_suffix}")
    # DAG closure: identical functional span in graphs whose node lists
    # are ordered differently (index windows would drift)
    prod = {}
    cons = {}
    for n in nodes:
        for o in n.output:
            prod[o] = n
        for x in n.input:
            if x:
                cons.setdefault(x, []).append(n)

    def closure(seed, edges):
        seen, stack = set(), [seed]
        while stack:
            n = stack.pop()
            if n.name in seen:
                continue
            seen.add(n.name)
            for m in edges(n):
                if m.name not in seen:
                    stack.append(m)
        return seen

    anc = closure(nodes[hi], lambda n: (prod[x] for x in n.input
                                        if x in prod))
    desc = closure(nodes[lo], lambda n: (c for o in n.output
                                         for c in cons.get(o, ())))
    sel = anc & desc
    span = [n for n in nodes if n.name in sel]
    print(f"span closure {nodes[lo].name} .. {nodes[hi].name}: "
          f"{len(span)} nodes")
else:
    lo = hi - args.nodes_back + 1
    span = nodes[lo:hi + 1]
    print(f"span nodes [{lo}..{hi}] ({len(span)})")
has_dfa = any(n.op_type == "DeformableAggregation" for n in span)
print(f"DFA in span: {has_dfa}")

prod_in_span = {}
for n in span:
    for o in n.output:
        prod_in_span[o] = n
span_ids = {id(n) for n in span}
cons_in_span = set()
for n in span:
    for x in n.input:
        if x:
            cons_in_span.add(x)

init_names_all = {i.name for i in g.initializer}
boundary_in = [x for x in sorted(cons_in_span)
               if x not in prod_in_span and x not in init_names_all
               and not x.endswith(".smoke")]
boundary_out = []
for n in span:
    for o in n.output:
        outside_users = [c for c in g.node
                         if id(c) not in span_ids
                         and any(i == o for i in c.input)]
        if outside_users or o.endswith(args.end_suffix):
            boundary_out.append(o)
boundary_out = sorted(set(boundary_out))
print(f"boundary inputs: {len(boundary_in)}, outputs: {len(boundary_out)}")

# ---- shapes + values via one full stub-ORT run (always refreshed: a
# stale npz from a different span would poison the module IO) ----
W = os.path.dirname(args.out)
npz_in = os.path.join(W, os.path.basename(args.out) + ".boundary.npz")
if True:
    sm = onnx.load(args.src)
    sg = sm.graph
    vis = {}
    for vi in list(sg.value_info) + list(sg.output):
        vis[vi.name] = vi
    for nm in boundary_in + boundary_out:
        if nm in vis:
            sg.output.append(vis[nm])
        else:
            sg.output.append(helper.make_empty_tensor_value_info(nm))
    ssrc = args.src + ".extract.onnx"
    onnx.save(sm, ssrc)
    stub(ssrc, ssrc + ".stub.onnx")
    os.remove(ssrc)
    try:
        import onnxruntime as ort
    except ImportError:
        # no ORT here: shapes/dtypes from onnx.shape_inference; the npz
        # carries zero arrays (values unused unless numeric self-check runs)
        from onnx import shape_inference
        si = shape_inference.infer_shapes(onnx.load(ssrc + ".stub.onnx"))
        sv = {v.name: v for v in list(si.graph.value_info)
              + list(si.graph.output)}
        import numpy as _np
        DT = {"tensor(float)": _np.float32, "tensor(float16)": _np.float16,
              "tensor(int32)": _np.int32, "tensor(int64)": _np.int64,
              "tensor(bool)": _np.bool_, "tensor(int8)": _np.int8}
        out = {}
        for nm in boundary_in + boundary_out:
            vi = sv.get(nm)
            tt = vi.type.tensor_type
            dt = {int(onnx.TensorProto.FLOAT): _np.float32,
                  int(onnx.TensorProto.FLOAT16): _np.float16,
                  int(onnx.TensorProto.INT32): _np.int32,
                  int(onnx.TensorProto.INT64): _np.int64,
                  int(onnx.TensorProto.BOOL): _np.bool_,
                  int(onnx.TensorProto.INT8): _np.int8}.get(
                      tt.elem_type, _np.float32)
            shp = [d.dim_value for d in tt.shape.dim]
            out[nm] = _np.zeros(shp, dt)
        np.savez_compressed(npz_in, **out)
        os.remove(ssrc + ".stub.onnx")
        print(f"boundary shapes saved (no-ORT): {npz_in}")
    else:
        sess = ort.InferenceSession(ssrc + ".stub.onnx", None,
                                    providers=["CPUExecutionProvider"])
        inp_npz = np.load(os.path.join(
            os.path.dirname(args.src), "mtq_v3_ref.npz.inputs.npz"))
        feed = {v.name: inp_npz[v.name] for v in sess.get_inputs()}
        want = boundary_in + boundary_out
        vals = sess.run(want, feed)
        np.savez_compressed(npz_in, **{k: v for k, v in zip(want, vals)})
        os.remove(ssrc + ".stub.onnx")
        print(f"boundary values saved: {npz_in}")

bnd = np.load(npz_in)
shapes = {k: list(bnd[k].shape) for k in bnd.files}
dtypes = {k: str(bnd[k].dtype) for k in bnd.files}

# ---- build module ----
gm = helper.make_graph(
    [], "region_mod",
    [helper.make_tensor_value_info(x, NAME_DT[dtypes[x]], shapes[x])
     for x in boundary_in],
    [helper.make_tensor_value_info(x, NAME_DT[dtypes[x]], shapes[x])
     for x in boundary_out])
mm = helper.make_model(gm, opset_imports=[
    helper.make_operatorsetid(o.domain, o.version)
    for o in m.opset_import
    if o.domain != "SparseDrive" or has_dfa])
del mm.graph.node[:]
mm.graph.node.extend(span)

init_names = set()
for n in span:
    for x in n.input:
        init_names.add(x)
init_names -= set(shapes)  # boundary tensors are inputs, not initializers
kept = []
for init in g.initializer:
    if init.name in init_names:
        kept.append(numpy_helper.from_array(
            numpy_helper.to_array(init), init.name))
mm.graph.initializer.extend(kept)
print(f"module initializers: {len(kept)}")

onnx.checker.check_model(mm, False)
onnx.save(mm, args.out)
print(f"saved {args.out}  size={os.path.getsize(args.out)/1e6:.1f} MB")

# ---- module numeric self-check (needs pure-ORT span: no DFA plugin) ----
if not args.skip_numeric and not has_dfa:
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, None,
                                providers=["CPUExecutionProvider"])
    feed = {v.name: bnd[v.name] for v in sess.get_inputs()}
    outs = sess.run([o.name for o in sess.get_outputs()], feed)
    worst = 0.0
    for nm, v in zip([o.name for o in sess.get_outputs()], outs):
        d = float(np.abs(v.astype(np.float64) - bnd[nm].astype(np.float64)).max())
        worst = max(worst, d)
    print(f"module self-check worst abs diff vs full graph: {worst:.3e}")
