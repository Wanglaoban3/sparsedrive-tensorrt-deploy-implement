import sys
import onnx

path = sys.argv[1] if len(sys.argv) > 1 else \
    r"<REPO>\work_dirs\sparsedrive_small_stage2\v5_P1h.onnx"
m = onnx.load(path)
g = m.graph

# 1) how is the existing plugin op declared in ONNX?
print("=== DeformableAggregation node proto (first) ===")
for n in g.node:
    if n.op_type == "DeformableAggregation":
        print(f"  name={n.name}")
        print(f"  domain='{n.domain}'")
        print(f"  inputs({len(n.input)}): {list(n.input)}")
        print(f"  outputs: {list(n.output)}")
        for a in n.attribute:
            print(f"  attr {a.name} type={a.type} i={a.i} "
                  f"f={a.f} s={a.s!r}")
        break

# 2) model opset imports (domain -> version)
print("\n=== opset imports ===")
for oi in m.opset_import:
    print(f"  domain='{oi.domain}' version={oi.version}")

# 3) verify the 10 target chains + collect surgery facts
nodes = list(g.node)
init_names = {x.name for x in g.initializer}
use = {}
for j, x in enumerate(nodes):
    for xinp in x.input:
        use.setdefault(xinp, []).append(j)

sites = []
for i, n in enumerate(nodes):
    if n.name.endswith("/attn/inner_attn/MatMul") and n.op_type == "MatMul":
        sites.append(i)
print(f"\n=== {len(sites)} inner_attn MatMul sites ===")
ok = True
for i in sites:
    chain = [nodes[i]]
    cur = nodes[i].output[0]
    for _ in range(4):
        cons = use.get(cur, [])
        assert len(cons) == 1, f"chain fanout at {cur}: {len(cons)}"
        chain.append(nodes[cons[0]])
        cur = chain[-1].output[0]
    ops = [c.op_type for c in chain]
    tag = chain[0].name.split("/")[1]
    mulnode = chain[2]
    consts = [x for x in mulnode.input if x in init_names]
    softmax_axis = [a.i for a in chain[3].attribute if a.name == "axis"]
    cast4_to = [a.i for a in chain[1].attribute if a.name == "to"]
    cast5_to = [a.i for a in chain[4].attribute if a.name == "to"]
    good = (ops == ["MatMul", "Cast", "Mul", "Softmax", "Cast"]
            and len(consts) == 1 and softmax_axis == [-1]
            and cast4_to == [1] and cast5_to == [10])
    ok &= good
    print(f"  {tag}: {ops} const={consts} axis={softmax_axis} "
          f"cast4={cast4_to} cast5={cast5_to} {'OK' if good else 'BAD'}")
print(f"all chains match pattern: {ok}")
