# -*- coding: utf-8 -*-
"""probe: where do non-QDZ Conv/BN/Resize/MaxPool/TopK/Split/Abs live?"""
import collections
import io
import os
import re
import sys

import onnx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
m = onnx.load(os.path.join(W, "v5_P1h.onnx"))
g = m.graph
nodes = list(g.node)

dq_out = set()
q_in = set()
for n in nodes:
    if n.op_type == "QuantizeLinear":
        for x in n.input:
            q_in.add(x)
    elif n.op_type == "DequantizeLinear":
        dq_out.add(n.output[0])


def in_qdz(n):
    if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
        return True
    if any(x in dq_out for x in n.input):
        return True
    if any(o in q_in for o in n.output):
        return True
    return False


WATCH = {"Conv", "BatchNormalization", "Resize", "MaxPool", "TopK",
         "Split", "Abs", "GatherND", "NonMaxSuppression", "InstanceNormalization",
         "LayerNormalization", "Einsum", "CumSum", "Range", "OneHot",
         "SpaceToDepth", "DepthToSpace", "RoiAlign"}
cnt = collections.Counter()
by_prefix = collections.defaultdict(collections.Counter)
for n in nodes:
    if n.op_type in WATCH and not in_qdz(n):
        cnt[n.op_type] += 1
        pref = "/" + "/".join(n.name.split("/")[1:2]) if "/" in n.name else n.name
        by_prefix[n.op_type][pref] += 1
print("non-QDZ ops of interest:", dict(cnt))
for op, prefs in sorted(by_prefix.items()):
    print(f"  {op}: {dict(prefs)}")

# and the head-PAT coverage: which ops inside head PAT are NOT in PRESERVE
PRESERVE = {
    "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Slice", "Identity",
    "Softmax", "Concat", "Mul", "Add", "Sub", "Div", "Pow", "Tanh",
    "Sigmoid", "Relu", "Erf", "Sqrt", "Reciprocal", "Neg", "Exp", "Log",
    "Max", "Min", "Pad", "Tile", "Expand", "Gather", "MatMul", "Gemm",
    "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "Flatten",
    "LeakyRelu", "HardSigmoid", "Shrink", "Sign", "Ceil", "Floor", "Round",
    "Where", "Dropout", "Clip", "ScatterND", "GatherElements", "Trilu",
}
PAT = r"layers\.|fc_before|fc_after"
bad = collections.Counter()
for n in nodes:
    if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
        continue
    if re.search(PAT, n.name) and n.op_type not in PRESERVE \
            and n.op_type != "DeformableAggregation":
        bad[n.op_type] += 1
print("head-PAT ops outside PRESERVE:", dict(bad))
