"""Inspect the 5 input edges of every DeformableAggregation node in the
exported ONNX: what produces each input (raw FP vs Q->DQ pair)?"""

import os
import sys
from collections import Counter

import onnx

path = sys.argv[1] if len(sys.argv) > 1 else (
    "work_dirs/sparsedrive_small_stage2/sparsedrive_int8.onnx")
m = onnx.load(path)
g = m.graph

prod = {}
for n in g.node:
    for o in n.output:
        prod[o] = n

inits = {i.name for i in g.initializer}

print("DFA nodes:", sum(1 for n in g.node if n.op_type == "DeformableAggregation"))
input_kinds = Counter()
detail = {}


def trace(name, hops=12):
    """Walk upstream; report any Q/DQ node found within hops and the
    terminal producer kind."""
    cur = name
    for h in range(hops):
        p = prod.get(cur)
        if p is None:
            return "initializer/graph-input", None
        if p.op_type in ("QuantizeLinear", "DequantizeLinear"):
            return "FP-after-QDQ", p.op_type
        if not p.input:
            return f"terminal:{p.op_type}", None
        cur = p.input[0]
    return f"no-QDQ-within-{hops} (ends at {p.op_type})", None


for n in g.node:
    if n.op_type != "DeformableAggregation":
        continue
    for idx, inp in enumerate(n.input):
        kind, _ = trace(inp)
        input_kinds[(idx, kind)] += 1
        detail.setdefault(idx, Counter())[kind] += 1

names = ["mc_ms_feat", "spatial_shape", "scale_start_index",
         "sampling_location(loc)", "weights"]
print("\nper-input-edge producer kinds across all DFA nodes:")
for idx in sorted(detail):
    print(f"  input[{idx}] ~ {names[idx] if idx < 5 else '?'}:")
    for kind, cnt in detail[idx].most_common():
        print(f"     {cnt:3d}x  {kind}")
