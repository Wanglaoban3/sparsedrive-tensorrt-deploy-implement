# -*- coding: utf-8 -*-
"""Re-quantize fpn_convs.N/conv back to INT8 in sp_backbone.onnx.
P2h had f16-ized them: DQ -> Cast(_to16) -> Conv(f16 weight). The Q/DQ
chains are still present - bypass the casts, rewire to the int8 DQs,
restore f32 bias from P1h, add an f16 Cast after the f32 conv output so
downstream f16 wiring is unchanged."""
import io
import os
import sys

import onnx
from onnx import TensorProto, helper, numpy_helper

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
SRC = os.path.join(W, "sp_backbone.onnx")
DST = os.path.join(W, "sp_backbone2.onnx")
P1H = os.path.join(W, "v5_P1h.onnx")

m = onnx.load(SRC)
g = m.graph
inits = {x.name for x in g.initializer}
p1 = onnx.load(P1H)
p1_inits = {x.name: x for x in p1.graph.initializer}

for N in range(4):
    pre = f"/img_neck/fpn_convs.{N}/conv"
    conv = next(n for n in g.node if n.name == f"{pre}/Conv")
    dq_in = f"{pre}/input_quantizer/DequantizeLinear_output_0"
    dq_w = f"{pre}/weight_quantizer/DequantizeLinear_output_0"
    for nm in (dq_in, dq_w):
        assert any(n.output and n.output[0] == nm for n in g.node), nm
    bias = f"model.img_neck.fpn_convs.{N}.conv.bias"
    if bias not in inits:
        assert bias in p1_inits
        g.initializer.append(p1_inits[bias])
        print(f"  copied f32 bias init {bias}")
    old_out = conv.output[0]
    conv.input[0] = dq_in
    conv.input[1] = dq_w
    conv.input[2] = bias
    conv.output[0] = old_out + "_i8f32"
    cast = helper.make_node("Cast", [old_out + "_i8f32"], [old_out],
                            name=old_out + "_requant_cast", to=int(
                                TensorProto.FLOAT16))
    idx = next(i for i, n in enumerate(g.node) if n.name == conv.name)
    g.node.insert(idx + 1, cast)
    print(f"fpn_convs.{N}: input<-{dq_in.split('/')[-1]}, "
          f"weight<-{dq_w.split('/')[-1]}, bias f32, out+cast")

onnx.checker.check_model(m, False)
onnx.save(m, DST)
print("saved", DST, f"{os.path.getsize(DST)/1e6:.1f}MB")
