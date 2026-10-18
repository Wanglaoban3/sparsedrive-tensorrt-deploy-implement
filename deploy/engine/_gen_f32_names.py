import io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import onnx
from onnx import TensorProto, numpy_helper
from collections import Counter

ROOT = r"H:\projects\sparsedrive-tensorrt-deploy-implement"
P = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2", "v5_P1h.onnx")
OUTF = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2",
                    "map_f32_names.txt")
m = onnx.load(P, load_external_data=False)
g = m.graph

FP16 = TensorProto.FLOAT16
dtype = {}
for vi in list(g.value_info) + list(g.output) + list(g.input):
    if vi.type.tensor_type.elem_type:
        dtype[vi.name] = vi.type.tensor_type.elem_type
for init in g.initializer:
    dtype[init.name] = init.data_type

# manual forward dtype propagation (Cast attrs exact; mixed -> f32;
# unknown -> None). Plugin/custom outputs unknown.
for n in g.node:
    ins = [dtype.get(i) for i in n.input]
    if n.op_type == "Cast":
        to = None
        for a in n.attribute:
            if a.name == "to":
                to = a.i
        out = to
    elif n.op_type == "QuantizeLinear":
        out = TensorProto.INT8
    elif n.op_type == "DequantizeLinear":
        out = TensorProto.FLOAT
    elif n.op_type == "Constant":
        out = None
        for a in n.attribute:
            if a.name == "value" and a.t and a.t.data_type:
                out = a.t.data_type
    else:
        kn = [d for d in ins if d is not None]
        if any(d == FP16 for d in kn):
            out = FP16 if n.op_type in ("MatMul", "Add", "Mul", "Sub",
                                        "Relu", "Div") else None
        else:
            out = (TensorProto.FLOAT if kn and
                   all(d in (TensorProto.FLOAT, TensorProto.FLOAT16,
                             TensorProto.INT64, TensorProto.INT32)
                       for d in kn) else None)
    for o in n.output:
        dtype.setdefault(o, out)

prod = {}
for n in g.node:
    for o in n.output:
        prod[o] = n

TARGETS = ["map_cls", "map_pts", "map_instance_feature",
           "map_anchor_embed", "next_map_feat", "next_map_anchor",
           "next_map_conf"]

seen, names, why = set(), [], Counter()
stack = list(TARGETS)
while stack:
    t = stack.pop()
    if t in seen or t not in prod:
        continue
    seen.add(t)
    n = prod[t]
    nm = n.name
    if nm.startswith("/img_backbone") or nm.startswith("/img_neck"):
        why["backbone/neck"] += 1
        continue
    if any(dtype.get(i) == FP16 for i in n.input):
        why["fp16-region keep"] += 1
        continue
    if n.op_type == "Cast":
        why["cast"] += 1
    elif not nm:
        why["anon"] += 1
    else:
        names.append(nm)
    stack.extend(n.input)

names_set = set(names)
opc = Counter()
for n in g.node:
    if n.name in names_set:
        opc[n.op_type] += 1
print("map f32 names: %d" % len(names))
print("excluded:", dict(why))
print("included op types:", dict(opc.most_common(18)))
# fp16-keep sample names for eyeball
with open(OUTF, "w", newline="\n") as f:
    f.write("\n".join(sorted(names_set)) + "\n")
print("written:", OUTF)
