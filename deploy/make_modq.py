# -*- coding: utf-8 -*-
"""modbig_p2h + synthetic QDQ pair on one boundary input -> modq_p2h.onnx
lets the board test kINT8 (explicit-quant mode) on a 381-node module."""
import onnx
from onnx import TensorProto, helper

SRC = r"<REPO>\work_dirs" \
      r"\sparsedrive_small_stage2\modbig_p2h.onnx"
DST = r"<REPO>\work_dirs" \
      r"\sparsedrive_small_stage2\modq_p2h.onnx"

g = onnx.load(SRC).graph
# pick a pure-float boundary input consumed by float compute:
target = "/layers.5/attn/out_proj/Add_output_0"  # [1,900,512] f16
idx = next(i for i, x in enumerate(g.input) if x.name == target)
scale = helper.make_tensor("modq_scale", TensorProto.FLOAT, [1], [0.05])
zp = helper.make_tensor("modq_zp", TensorProto.INT8, [1], [0])
q = helper.make_node("QuantizeLinear", [target, "modq_scale", "modq_zp"],
                     ["modq_q"], name="modq_Q")
dq = helper.make_node("DequantizeLinear", ["modq_q", "modq_scale", "modq_zp"],
                      ["modq_dq"], name="modq_DQ")
# rewire consumers of target to modq_dq
for n in g.node:
    for j, inp in enumerate(n.input):
        if inp == target:
            n.input[j] = "modq_dq"
# Q consumes the graph input directly; keep it declared in g.input.
# Q/DQ depend only on a graph input -> prepend to keep topo order
orig = list(g.node)
del g.node[:]
g.node.extend([q, dq])
g.node.extend(orig)
g.initializer.extend([scale, zp])

m = onnx.load(SRC)
m.graph.CopyFrom(g)
onnx.checker.check_model(m)
onnx.save(m, DST)
print("saved", DST, "nodes:", len(g.node))
