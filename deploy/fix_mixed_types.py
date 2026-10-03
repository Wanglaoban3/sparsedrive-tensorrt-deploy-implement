"""Fix mixed float32/float16 node inputs left by the attn-fp16 export.

The exporter cast Q,K,V to fp16 but fed the f32 softmax output directly to
attn @ V (mixed f32/f16 MatMul).  ORT rejects the model at load; TRT strict
typing would likely fail too.  This tool propagates float dtypes through the
graph (value_info + initializers + Cast attrs + dtype-preserving ops), finds
nodes with mixed f32/f16 inputs and inserts a Cast on the minority side:
  - if any input is a direct Cast(to=f16) output -> cast others to f16
    (matches torch autocast semantics / MTQ calibration intent)
  - otherwise -> upcast f16 inputs to f32 (lossless)

Usage: python deploy/fix_mixed_types.py <in.onnx> <out.onnx>
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT
FNAME = {F16: "f16", F32: "f32"}

# ops whose output dtype = the (uniform) float dtype of their inputs
PRESERVE = {
    "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Slice", "Identity",
    "Softmax", "Concat", "Mul", "Add", "Sub", "Div", "Pow", "Tanh",
    "Sigmoid", "Relu", "Erf", "Sqrt", "Reciprocal", "Neg", "Exp", "Log",
    "Max", "Min", "Pad", "Tile", "Expand", "Gather", "MatMul", "Gemm",
    "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "Flatten",
    "LeakyRelu", "HardSigmoid", "Shrink", "Sign", "Ceil", "Floor", "Round",
    "Where", "Dropout", "Clip",
}

path_in, path_out = sys.argv[1], sys.argv[2]
m = onnx.load(path_in)
g = m.graph

dtype = {}
for vi in list(g.input) + list(g.output) + list(g.value_info):
    t = vi.type.tensor_type.elem_type
    if t in FNAME:
        dtype[vi.name] = t
for i in g.initializer:
    if i.data_type in FNAME:
        dtype[i.name] = i.data_type

prod = {}
for n in g.node:
    for o in n.output:
        prod[o] = n
CAST_TO = {F16: F16, F32: F32, TensorProto.INT64: TensorProto.INT64,
           TensorProto.INT32: TensorProto.INT32,
           TensorProto.BOOL: TensorProto.BOOL,
           TensorProto.UINT8: TensorProto.UINT8,
           TensorProto.INT8: TensorProto.INT8}


def infer():
    changed = True
    while changed:
        changed = False
        for n in g.node:
            if n.op_type == "Cast":
                to = next((a.i for a in n.attribute if a.name == "to"), None)
                t = CAST_TO.get(to)
                if t in FNAME and dtype.get(n.output[0]) != t:
                    dtype[n.output[0]] = t
                    changed = True
                continue
            if n.op_type == "DequantizeLinear":
                sc = n.input[1]
                t = dtype.get(sc) or next(
                    (i.data_type for i in g.initializer if i.name == sc), None)
                if t in FNAME and dtype.get(n.output[0]) != t:
                    dtype[n.output[0]] = t
                    changed = True
                continue
            if n.op_type in PRESERVE:
                ts = [dtype.get(x) for x in n.input if x]
                fts = {t for t in ts if t in FNAME}
                if len(fts) == 1:
                    t = next(iter(fts))
                    for o in n.output:
                        if o and dtype.get(o) != t:
                            dtype[o] = t
                            changed = True


infer()
mixed = []
for n in g.node:
    if n.op_type in ("Cast", "QuantizeLinear", "DequantizeLinear"):
        continue
    fts = {t for t in (dtype.get(x) for x in n.input if x) if t in FNAME}
    if len(fts) > 1:
        mixed.append(n)

print(f"mixed float-input nodes: {len(mixed)}")
sigs = {}
for n in mixed:
    sig = tuple((x, FNAME.get(dtype.get(x))) for x in n.input if x)
    sigs.setdefault((n.op_type, tuple(t for _, t in sig if t)), []).append(n)

fixed = 0
for n in mixed:
    to = F16 if any(
        (prod.get(x) is not None and prod[x].op_type == "Cast"
         and next((a.i for a in prod[x].attribute if a.name == "to"), None) == F16)
        for x in n.input) else F32
    for k, x in enumerate(n.input):
        if x and dtype.get(x) in FNAME and dtype[x] != to:
            cname = f"{n.output[0]}_mixfix_{k}"
            g.node.append(helper.make_node(
                "Cast", [x], [cname], to=to, name=cname + "_cast"))
            n.input[k] = cname
            fixed += 1
    print(f"  fix {n.op_type} {n.name}: inputs casted to {FNAME[to]}")

# inserted Cast nodes were appended at the end; re-sort topologically
producers = {}
for n in g.node:
    for o in n.output:
        producers[o] = n
import collections

waiting = {id(n): len([x for x in n.input if x and x in producers])
           for n in g.node}
consumer_of = collections.defaultdict(list)
node_by_id = {id(n): n for n in g.node}
for n in g.node:
    for x in n.input:
        if x and x in producers:
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
    raise RuntimeError(f"topo sort stuck: placed {len(placed)} "
                       f"of {len(g.node)}")
del g.node[:]
g.node.extend(placed)

onnx.checker.check_model(m, False)
onnx.save(m, path_out)
print(f"inserted {fixed} casts -> {path_out}")

# re-verify no mixed remain
m2 = onnx.load(path_out)
g2 = m2.graph
dtype2 = {}
for vi in list(g2.input) + list(g2.output) + list(g2.value_info):
    t = vi.type.tensor_type.elem_type
    if t in FNAME:
        dtype2[vi.name] = t
for i in g2.initializer:
    if i.data_type in FNAME:
        dtype2[i.name] = i.data_type


def infer2():
    changed = True
    while changed:
        changed = False
        for n in g2.node:
            if n.op_type == "Cast":
                to = next((a.i for a in n.attribute if a.name == "to"), None)
                if to in FNAME and dtype2.get(n.output[0]) != to:
                    dtype2[n.output[0]] = to
                    changed = True
                continue
            if n.op_type == "DequantizeLinear":
                sc = n.input[1]
                t = dtype2.get(sc) or next(
                    (i.data_type for i in g2.initializer if i.name == sc), None)
                if t in FNAME and dtype2.get(n.output[0]) != t:
                    dtype2[n.output[0]] = t
                    changed = True
                continue
            if n.op_type in PRESERVE:
                fts = {t for t in (dtype2.get(x) for x in n.input if x)
                       if t in FNAME}
                if len(fts) == 1:
                    t = next(iter(fts))
                    for o in n.output:
                        if o and dtype2.get(o) != t:
                            dtype2[o] = t
                            changed = True


infer2()
left = [n for n in g2.node
        if n.op_type not in ("Cast", "QuantizeLinear", "DequantizeLinear")
        and len({t for t in (dtype2.get(x) for x in n.input if x)
                 if t in FNAME}) > 1]
print(f"mixed remaining after fix: {len(left)}")
for n in left[:10]:
    print("  STILL:", n.op_type, n.name,
          [FNAME.get(dtype2.get(x)) for x in n.input if x])
