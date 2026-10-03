# -*- coding: utf-8 -*-
"""Find map_cls producer node + its weight shapes in P1h vs P2h."""
import io
import os
import sys

import onnx
from onnx import numpy_helper

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")

for fn in ("v5_P1h.onnx", "v5_P2h.onnx"):
    m = onnx.load(os.path.join(W, fn))
    g = m.graph
    inits = {x.name: x for x in g.initializer}
    for out in ("map_cls", "map_pts"):
        p = next(n for n in g.node if out in n.output)
        print(f"{fn} {out}: [{p.op_type}] {p.name!r}")
        for x in p.input:
            if x in inits:
                t = numpy_helper.to_array(inits[x])
                print(f"    init {x}: {t.dtype}{t.shape}")
            else:
                q = next((n for n in g.node if x in n.output), None)
                print(f"    in {x} <- [{q.op_type if q is not None else '?'}] "
                      f"{q.name if q is not None else ''}")
    print()
