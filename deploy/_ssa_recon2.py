import sys
import onnx

path = sys.argv[1] if len(sys.argv) > 1 else \
    r"<REPO>\work_dirs\sparsedrive_small_stage2\v5_P1h.onnx"
m = onnx.load(path)
g = m.graph
mi = onnx.shape_inference.infer_shapes(m)
vi = {}
for v in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output):
    tt = v.type.tensor_type
    dims = [d.dim_value if d.WhichOneof("value") == "dim_value" else -1
            for d in tt.shape.dim]
    vi[v.name] = dims

def shp(t):
    return vi.get(t, "NO-VI")

nodes = list(mi.graph.node)

# 1) complete attention census: every Softmax node + input shape
print("=== all Softmax nodes ===")
for i, n in enumerate(nodes):
    if n.op_type == "Softmax":
        print(f"  #{i} {n.name}\n      in {n.input[0]} {shp(n.input[0])}")

# 2) enumerate all MatMul under an attn path
print("\n=== MatMul in */attn/* ===")
for i, n in enumerate(nodes):
    if n.op_type == "MatMul" and "/attn/" in n.name:
        ins = ", ".join(f"{t}{shp(t)}" for t in n.input)
        print(f"  #{i} {n.name}: {ins} -> {n.output[0]}{shp(n.output[0])}")

# 3) layer numbering: max layers.N and which exist
import re
lay = set()
for n in nodes:
    mm = re.match(r"/layers\.(\d+)/", n.name)
    if mm:
        lay.add(int(mm.group(1)))
print(f"\n=== /layers.N present: {min(lay)}..{max(lay)} "
      f"({len(lay)} distinct) ===")
missing = sorted(set(range(min(lay), max(lay) + 1)) - lay)
print(f"  missing indices: {missing}")
# what op_types live under each block of 7
for k in range(min(lay), max(lay) + 1, 7):
    ops = [n.op_type + ":" + n.name.split("/")[2] for n in nodes
           if re.match(rf"/layers\.{k}/", n.name)]
    from collections import Counter
    cnt = Counter(x.split(":")[0] for x in ops)
    subs = sorted({x.split(":")[1] for x in ops})
    print(f"  layers.{k}..{k+6}: {dict(cnt)}")
    print(f"     submodules: {subs}")
