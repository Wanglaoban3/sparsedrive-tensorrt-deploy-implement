# -*- coding: utf-8 -*-
import io
import sys

import onnx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
P = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2\v5_P2h.onnx")
m = onnx.load(P, load_external_data=False)
mi = onnx.shape_inference.infer_shapes(m)
vi = {}
for v in list(mi.graph.value_info) + list(mi.graph.output):
    tt = v.type.tensor_type
    dims = [d.dim_value if d.WhichOneof("value") == "dim_value" else -1
            for d in tt.shape.dim]
    vi[v.name] = (tt.elem_type, dims)
print("value_info entries:", len(vi))
for t in ("/Reshape_9_output_0", "ego_feature_map.pre_o",
          "ego_feature_map"):
    print(t, "->", vi.get(t))
# producer node info for pre_o
for n in m.graph.node:
    if n.output and n.output[0] == "ego_feature_map.pre_o":
        print("pre_o producer:", n.op_type, n.name, "in:", list(n.input))
        break
# ego_feature_map original decl
for o in m.graph.output:
    if o.name == "ego_feature_map":
        tt = o.type.tensor_type
        dims = [(d.dim_value if d.WhichOneof("value") == "dim_value"
                 else d.dim_param) for d in tt.shape.dim]
        print("ego output decl elem", tt.elem_type, "dims", dims)
