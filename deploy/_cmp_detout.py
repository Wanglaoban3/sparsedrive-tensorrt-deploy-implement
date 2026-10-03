# -*- coding: utf-8 -*-
import io
import sys

import onnx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
for fn in ("sp_head_det.onnx", "sp_head_det2.onnx"):
    g = onnx.load(W + "\\" + fn).graph
    outs = [v.name for v in g.output]
    print(fn, len(outs), "outputs:")
    for o in outs:
        print("   ", o)
