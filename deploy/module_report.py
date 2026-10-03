# -*- coding: utf-8 -*-
import collections
import io
import sys

import onnx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = r"<REPO>\work_dirs" \
    r"\sparsedrive_small_stage2"
for f in (r"\mod_p1h.onnx", r"\mod_p2h.onnx"):
    g = onnx.load(W + f).graph
    cnt = collections.Counter(n.op_type for n in g.node)
    print(f, len(g.node), "nodes:", dict(cnt))
    print("  inputs :", [(v.name, v.type.tensor_type.elem_type)
                         for v in g.input])
    print("  outputs:", [(v.name, v.type.tensor_type.elem_type)
                         for v in g.output])
