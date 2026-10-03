# -*- coding: utf-8 -*-
import io
import sys

import onnx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = r"<REPO>\work_dirs" \
    r"\sparsedrive_small_stage2"
ns = [n.name for n in onnx.load(W + r"\v5_P1h.onnx").graph.node]
for nm in ns:
    if ("layers.5" in nm or "layers.6" in nm) and nm.endswith(("Add_1", "/Add")):
        print(nm)
