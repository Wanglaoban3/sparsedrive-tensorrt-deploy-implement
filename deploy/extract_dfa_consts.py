# -*- coding: utf-8 -*-
"""从 ONNX initializer 提取 DFA 节点真实的 spatial_shape / scale_start_index,
重写 dfa_real_engine 的 shape.i32 / ssi.i32, 并打印质量分析。"""
import io
import os
import sys

import numpy as np
import onnx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
OUT = os.path.join(W, "dfa_real_engine")

for cand in ["v5_P1h.onnx", "v5_P2h.onnx", "v5.onnx", "model.onnx"]:
    p = os.path.join(W, cand)
    if os.path.exists(p):
        break
else:
    print("no onnx found; list:")
    for f in sorted(os.listdir(W)):
        if f.endswith(".onnx"):
            print(" ", f)
    sys.exit(1)
print("onnx:", cand)
m = onnx.load(p)
inits = {i.name: i for i in m.graph.initializer}
from onnx import numpy_helper  # noqa: E402

dfas = [n for n in m.graph.node if n.op_type == "DeformableAggregation"]
print("dfa nodes:", len(dfas))
n0 = dfas[0]
print("inputs:", list(n0.input))
for pos in (1, 2):
    tname = n0.input[pos]
    if tname in inits:
        arr = numpy_helper.to_array(inits[tname])
        print(f"input[{pos}] {tname}: shape={arr.shape} dtype={arr.dtype}")
        print(arr.reshape(-1))
    else:
        print(f"input[{pos}] {tname}: NOT initializer (graph input/computed)")

# 用节点0的真实值重写 (假定各 dfa 节点共享同一常量)
t1 = n0.input[1]
t2 = n0.input[2]
assert t1 in inits and t2 in inits
shp = numpy_helper.to_array(inits[t1]).astype(np.int32).reshape(-1)
ssi = numpy_helper.to_array(inits[t2]).astype(np.int32).reshape(-1)
shp.tofile(os.path.join(OUT, "shape.i32"))
ssi.tofile(os.path.join(OUT, "ssi.i32"))
print("written shape.i32", shp.shape, "ssi.i32", ssi.shape)
print("ssi =", ssi)
print("ONNX_CONST_DONE")
