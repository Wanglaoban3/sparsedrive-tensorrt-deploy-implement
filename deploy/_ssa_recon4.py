"""Trace full projection chain at big attention sites for the fused-SDPA
plugin design: q/k/v producers, weights, bias, layout, downstream."""
import sys

import onnx
from onnx import numpy_helper

path = (r"<REPO>\work_dirs"
        r"\sparsedrive_small_stage2\v5_P1h_ssa.onnx")
m = onnx.load(path)
g = m.graph
nodes = list(g.node)
init = {x.name: x for x in g.initializer}

mi = onnx.shape_inference.infer_shapes(m)
vi = {}
for v in list(mi.graph.value_info) + list(mi.graph.input) + \
        list(mi.graph.output):
    tt = v.type.tensor_type
    dims = [d.dim_value if d.WhichOneof("value") == "dim_value" else -1
            for d in tt.shape.dim]
    vi[v.name] = dims

prod = {}
for i, n in enumerate(nodes):
    for o in n.output:
        prod[o] = i


def shp(t):
    d = vi.get(t)
    return f"{d}" if d else "?"


def trace_back(t, depth=6):
    chain = []
    cur = t
    for _ in range(depth):
        p = prod.get(cur)
        if p is None:
            chain.append(f"    <graph input/init> {cur} {shp(cur)}")
            break
        n = nodes[p]
        w = ""
        for x in n.input:
            if x in init:
                a = numpy_helper.to_array(init[x])
                w += f" | W {x} {a.dtype}{list(a.shape)}"
        chain.append(f"    {n.op_type} {n.name} [{cur} {shp(cur)}]{w}")
        if not n.input:
            break
        cur = n.input[0]
    return chain


for site in ["/layers.5/attn", "/layers.4/attn"]:
    print(f"===== {site} =====")
    for proj in ["q_proj", "k_proj", "v_proj"]:
        t = f"{site}/{proj}/MatMul_output_0"
        print(f"  --- {proj} from {t} {shp(t)} ---")
        for line in trace_back(t, 4):
            print(line)
    # after PV: out reshape chain
    print("  --- PV out chain ---")
    pv = f"{site}/inner_attn/MatMul_1_output_0"
    cur = pv
    for _ in range(4):
        p = prod.get(cur)
        if p is None:
            break
        n = nodes[p]
        print(f"    {n.op_type} {n.name} [{cur} {shp(cur)}] "
              f"attrs={[ (a.name, a.ints if a.ints else a.i) for a in n.attribute]}")
        cur = n.output[0]
