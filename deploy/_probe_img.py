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
cons = collections.defaultdict(list)
for n in g.node:
    for x in n.input:
        if x:
            cons[x].append(n)
print("consumers of img:")
for c in cons.get("img", ()):
    print("  node:", c.name, "| op:", c.op_type)
print("node name prefixes (first token after /):")
cnt = collections.Counter(
    n.name.split("/")[1] if n.name.startswith("/") and
    n.name.count("/") > 1 else "(raw)" for n in g.node)
for k, v in cnt.most_common(20):
    print(f"  {k}: {v}")
print("nodes named like Transpose_3 / raw names:")
for n in g.node:
    if n.name in ("/Transpose_3", "Transpose_3"):
        print("  ", n.name, n.op_type, "in:", list(n.input))
