# -*- coding: utf-8 -*-
import io
import os
import sys

import onnx
from onnx import numpy_helper

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")

m = onnx.load(os.path.join(W, "sparsedrive_int8_v3_fold5.onnx"))
g = m.graph
inits = {x.name: x for x in g.initializer}
# find the map cls bias: same node path as P1h showed
p = next(n for n in g.node if "map_cls" in n.output)
print("fold5 map_cls producer:", p.op_type, repr(p.name))
for x in p.input:
    if x in inits:
        t = numpy_helper.to_array(inits[x])
        print("  init", x, t.dtype, t.shape)
    else:
        q = next((n for n in g.node if x in n.output), None)
        print("  in", x, "<-", q.op_type if q is not None else "?",
              q.name if q is not None else "")
# det cls bias for scale reference
pd = next(n for n in g.node if "det_cls" in n.output)
print("fold5 det_cls producer:", pd.op_type, repr(pd.name))
for x in pd.input:
    if x in inits:
        t = numpy_helper.to_array(inits[x])
        print("  init", x, t.dtype, t.shape)
