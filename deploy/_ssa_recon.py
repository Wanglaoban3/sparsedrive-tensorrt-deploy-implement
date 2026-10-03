import sys, hashlib
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

ELEM = {1: "f32", 2: "u8", 3: "i8", 4: "u16", 5: "i16", 6: "i32",
        7: "i64", 9: "bool", 10: "f16", 11: "f64", 12: "u32", 13: "u64"}

path = sys.argv[1] if len(sys.argv) > 1 else \
    r"<REPO>\work_dirs\sparsedrive_small_stage2\v5_P1h.onnx"
m = onnx.load(path)
g = m.graph
init_names = {x.name for x in g.initializer}
inits = {x.name: x for x in g.initializer}

print("running shape inference ...")
mi = onnx.shape_inference.infer_shapes(m)
vi = {}
for v in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output):
    tt = v.type.tensor_type
    dims = [d.dim_value if d.WhichOneof("value") == "dim_value" else -1
            for d in tt.shape.dim]
    vi[v.name] = (ELEM.get(tt.elem_type, tt.elem_type), dims)

prod = {}
for i, n in enumerate(mi.graph.node):
    for o in n.output:
        prod[o] = (i, n)
nodes = list(mi.graph.node)

def show(t):
    return f"{t}[{vi.get(t, ('?', []))[0]}{vi.get(t, ('?', []))[1]}]"

# ---- find MatMul outputs shaped [1,8,900,900] ----
sites = []
for i, n in enumerate(nodes):
    if n.op_type != "MatMul":
        continue
    out = n.output[0]
    if out in vi and vi[out][1] == [1, 8, 900, 900]:
        sites.append(i)
print(f"\n=== {len(sites)} QK^T MatMul sites (out [1,8,900,900]) ===")

mask_hashes = {}
for si, i in enumerate(sites):
    n = nodes[i]
    q, k = n.input
    print(f"\n--- site {si}: node#{i} {n.name} ---")
    print(f"  Q in : {show(q)}   prod: {nodes[prod[q][0]].op_type} "
          f"({prod[q][1].name})")
    print(f"  K in : {show(k)}   prod: {nodes[prod[k][0]].op_type} "
          f"({prod[k][1].name})")
    print(f"  QK^T : {show(n.output[0])}")
    # walk forward from this node following the single-consumer chain
    cur = n.output[0]
    use = {}
    for j, x in enumerate(nodes):
        for xinp in x.input:
            use.setdefault(xinp, []).append(j)
    for step in range(6):
        cons = use.get(cur, [])
        if len(cons) != 1:
            print(f"  chain stop at {cur}: {len(cons)} consumers "
                  f"{[nodes[c].op_type for c in cons]}")
            break
        c = nodes[cons[0]]
        extra = ""
        if c.op_type == "Cast":
            to = [a.i for a in c.attribute if a.name == "to"]
            extra = f" to={to}"
        if c.op_type == "Softmax":
            ax = [(a.name, a.i) for a in c.attribute]
            extra = f" attrs={ax}"
        if c.op_type in ("Mul", "Add", "Sub") and \
                any(x in init_names for x in c.input):
            cn = [x for x in c.input if x in init_names][0]
            w = numpy_helper.to_array(inits[cn])
            u = np.unique(w)
            stats = (f"const {cn} {w.dtype}{list(w.shape)} "
                     f"min={w.min()} max={w.max()} nuniq={len(u)}")
            if len(u) <= 8:
                stats += f" vals={u[:8]}"
            h = hashlib.md5(w.tobytes()).hexdigest()[:10]
            mask_hashes.setdefault(h, []).append(cn)
            stats += f" md5={h}"
            extra = " | " + stats
        print(f"  step{step}: {c.op_type} {extra} -> {show(c.output[0])}")
        cur = c.output[0]
    # consumer after chain end
    cons = use.get(cur, [])
    print(f"  after chain: {cur} consumed by "
          f"{[(nodes[c].op_type, nodes[c].name) for c in cons]}")

print(f"\n=== mask const dedup ===")
for h, names in mask_hashes.items():
    print(f"  {h}: x{len(names)}  e.g. {names[0]}")

# also: any other [1,8,900,900]-ish producers (sanity: count all big tensors)
big = [(t, vi[t][0], vi[t][1]) for t in vi
       if vi[t][1] == [1, 8, 900, 900]]
print(f"\ntensors declared [1,8,900,900]: {len(big)} "
      f"({set(b[1] for b in big)} dtypes)")
