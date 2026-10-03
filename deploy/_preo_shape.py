# -*- coding: utf-8 -*-
import collections
import io
import sys

import onnx
from onnx import numpy_helper

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
m = onnx.load(W + r"\sp_head.onnx")
g = m.graph
inits = {x.name: x for x in g.initializer}
cons = collections.defaultdict(list)
for n in g.node:
    for x in n.input:
        if x:
            cons[x].append(n)
for t in ("det_instance_feature.pre_o", "det_bbox.pre_o",
          "det_anchor_embed.pre_o", "/layers.38/Add_output_0"):
    print(t or "(scalar-tensor)")
    for n in cons.get(t, ()):
        winfo = []
        for x in n.input:
            if x in inits:
                a = numpy_helper.to_array(inits[x])
                winfo.append(f"init{x.split('.')[-1][:18]}{a.shape}")
        print("   <-", n.op_type, n.name, winfo)
