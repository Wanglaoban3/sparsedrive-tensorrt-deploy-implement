# -*- coding: utf-8 -*-
import io
import re
import sys

import onnx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = r"<REPO>\work_dirs" \
    r"\sparsedrive_small_stage2"
ns = [n.name for n in onnx.load(W + r"\v5_P1h.onnx").graph.node]
pat = re.compile(r"^/layers\.\d+(_1)?/(Add|Reshape|MatMul|Gemm)(_\d+)?$")
for nm in ns:
    if pat.match(nm):
        print(nm)
print("---- op counts of top-level /layers.N (no subpath) nodes ----")
import collections
c = collections.Counter()
for nm in ns:
    m = re.match(r"^/layers\.\d+(_1)?/[^/]*$", nm)
    if m:
        c[nm] += 1
print(len(c))
