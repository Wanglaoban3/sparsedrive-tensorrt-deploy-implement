# -*- coding: utf-8 -*-
import io
import sys
import collections

import onnx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
SRC = (r"<REPO>\work_dirs"
       r"\sparsedrive_small_stage2\v5_P2h.onnx")
m = onnx.load(SRC, load_external_data=False)
g = m.graph
nodes = list(g.node)
prod = {}
node_by_name = {}
cons = collections.defaultdict(list)
for n in nodes:
    node_by_name[n.name] = n
    for o in n.output:
        prod[o] = n
    for x in n.input:
        if x:
            cons[x].append(n)
init_names = {i.name for i in g.initializer}

# replicate classification quickly
graph_inputs = [v.name for v in g.input if v.name not in init_names]
img_reach = set()
stack = list(cons.get("img", ()))
while stack:
    n = stack.pop()
    if n.name in img_reach:
        continue
    img_reach.add(n.name)
    for o in n.output:
        stack.extend(c for c in cons.get(o, ()) if c.name not in img_reach)
head_roots = {x for x in graph_inputs if x != "img"}
head = set()
q = collections.deque(n for n in nodes
                      if any(x in head_roots for x in n.input))
for n in q:
    head.add(n.name)
while q:
    n = q.popleft()
    for o in n.output:
        for c in cons.get(o, ()):
            if c.name not in head:
                head.add(c.name)
                q.append(c)
head |= {n.name for n in nodes if n.name not in img_reach}
bb = img_reach - head

print("== fpn convs ==")
for nm, n in node_by_name.items():
    if "fpn_convs" in nm and n.op_type == "Conv":
        print(f"  {'BB' if nm in bb else 'HEAD'} {nm}")
        for o in n.output:
            cs = cons.get(o, [])
            print(f"    out {o}: {len(cs)} consumers, "
                  f"{'BB' if all(c.name in bb for c in cs) else 'HEAD' if all(c.name in head for c in cs) else 'MIXED'}"
                  f" first={cs[0].name if cs else None}")

print("== /Transpose_3 chain ==")
t3 = node_by_name.get("/Transpose_3")
print("  /Transpose_3 in head:", "/Transpose_3" in head,
      "in img_reach:", "/Transpose_3" in img_reach,
      "in:", list(t3.input))
for x in t3.input:
    p = prod.get(x)
    if p:
        print(f"  producer of {x}: {p.name} ({p.op_type}) "
              f"inBB={p.name in bb} inHead={p.name in head} "
              f"inImgReach={p.name in img_reach}")
        for xx in p.input:
            pp = prod.get(xx)
            init_mark = "INIT" if xx in init_names else ""
            print(f"     in {xx[:60]} {init_mark} "
                  f"prod={pp.name[:60] if pp else None}")

print("== boundary-ish: tensors consumed by head, produced by BB, "
      "sample 30 ==")
cnt = 0
seen = set()
for nm in bb:
    for o in node_by_name[nm].output:
        outs = [c for c in cons.get(o, ()) if c.name in head]
        if outs and o not in seen:
            seen.add(o)
            cnt += 1
            if cnt <= 30:
                print(f"  {o[:80]}  consumers={len(outs)} "
                      f"first={outs[0].name[:60]}")
print("total:", cnt)
