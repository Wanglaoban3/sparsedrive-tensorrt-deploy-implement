# -*- coding: utf-8 -*-
import io
import sys

import onnx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
p = r"<REPO>\work_dirs" \
    r"\sparsedrive_small_stage2\v5_P2h.onnx"
g = onnx.load(p, load_external_data=False).graph
hits = [n.name for n in g.node if n.name.endswith("/Mul_2")]
print("nodes ending /Mul_2:", hits)
for n in g.node:
    if n.name == "/Mul_2":
        print("inputs:", list(n.input), "outputs:", list(n.output))
ends = [n.name for n in g.node if n.name.endswith("/layers.7/Reshape_3")]
print("end nodes:", ends)
