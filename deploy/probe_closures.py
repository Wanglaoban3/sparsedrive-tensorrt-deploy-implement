# -*- coding: utf-8 -*-
import io
import sys

import onnx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = r"<REPO>\work_dirs" \
    r"\sparsedrive_small_stage2"
g = onnx.load(W + r"\v5_P1h.onnx").graph
nodes = list(g.node)
bynm = {n.name: n for n in nodes}
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
        stack.extend(edges(n))
    return seen


start = bynm["/layers.5/Add"]
desc = closure(start, lambda n: [c for o in n.output for c in cons.get(o, ())])
print("desc(/layers.5/Add):", len(desc))
for end in ("/layers.9/Add", "/layers.9/Add_1", "/layers.10/Add",
            "/layers.10/Add_1", "/layers.11/Add", "/layers.11/Add_1",
            "/layers.12/Add", "/layers.40/Add"):
    if end not in bynm:
        print(f"{end}: MISSING")
        continue
    anc = closure(bynm[end], lambda n: (prod[x] for x in n.input
                                        if x in prod))
    print(f"{end}: anc={len(anc)} closure={len(anc & desc)}")
# also: how many nodes in full det-side descendants overall
print("sample desc names:", sorted(list(desc))[:5])
