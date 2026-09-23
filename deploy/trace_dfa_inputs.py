import sys

import onnx

path = ("work_dirs/sparsedrive_small_stage2/sparsedrive_int8.onnx")
m = onnx.load(path)
g = m.graph

prod = {}
for n in g.node:
    for o in n.output:
        prod[o] = n
inits = {i.name for i in g.initializer}
graph_inputs = {i.name for i in g.input}

for n in g.node:
    if n.op_type != "DeformableAggregation":
        continue
    print("DFA", n.name)
    for idx in (0, 3, 4):
        cur = n.input[idx]
        print(f"  input[{idx}] = {cur[:60]}")
        for h in range(24):
            if cur in inits:
                print("     -> initializer", cur[:50])
                break
            if cur in graph_inputs:
                print("     -> GRAPH INPUT", cur[:50])
                break
            p = prod.get(cur)
            if p is None:
                print("     -> no producer (dangling)", cur[:50])
                break
            tag = "   <<QDQ>>" if p.op_type in (
                "QuantizeLinear", "DequantizeLinear") else ""
            print(f"     <- {p.op_type:20s} {p.name[:44]} {tag}")
            cur = p.input[0]
    break
