# -*- coding: utf-8 -*-
import io
import sys

import onnx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
P = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2\sp_head_det.onnx")
m = onnx.load(P)
mi = onnx.shape_inference.infer_shapes(m)
names = {}
for src in (mi.graph.value_info, mi.graph.output):
    for v in src:
        names[v.name] = ([d.dim_value if d.WhichOneof("value") == "dim_value"
                          else -1 for d in v.type.tensor_type.shape.dim],
                         v.type.tensor_type.elem_type)
for k in ("det_instance_feature.pre_o", "det_bbox.pre_o",
          "det_anchor_embed.pre_o", "/layers.38/Add_output_0",
          "det_instance_feature"):
    print(k, names.get(k, "NOT-INFERRED"))
print("mi value_info:", len(mi.graph.value_info), "nodes:", len(m.graph.node))
