# -*- coding: utf-8 -*-
"""Compare graph output decls across v5 lineage: v5.onnx / P1h / P2h / splits."""
import io
import os
import sys

import onnx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")


def dims_of(v):
    tt = v.type.tensor_type
    return [d.dim_value if d.WhichOneof("value") == "dim_value" else -1
            for d in tt.shape.dim]


for fn in ("v5.onnx", "v5_T3.onnx", "v5_P1h.onnx", "v5_P2h.onnx",
           "sp_head.onnx", "sp_backbone.onnx"):
    p = os.path.join(W, fn)
    if not os.path.exists(p):
        print(f"{fn}: (missing)")
        continue
    g = onnx.load(p).graph
    outs = {v.name: dims_of(v) for v in g.output}
    key = {k: outs[k] for k in ("map_cls", "map_pts", "det_cls", "det_bbox")
           if k in outs}
    print(f"{fn}: {len(g.output)} outputs; map_cls={key.get('map_cls')} "
          f"map_pts={key.get('map_pts')} det_cls={key.get('det_cls')}")
    if fn == "sp_head.onnx":
        print("   ALL:", {k: v for k, v in sorted(outs.items())})
