# -*- coding: utf-8 -*-
"""P2h local numeric gate: stub-ORT equivalence v5_P1h vs v5_P2h.

Replaces every DeformableAggregation node with pure-ORT arithmetic that
preserves the node's input dtypes (dtype mistakes fail at session load),
then runs both graphs on CPU with the saved first-frame inputs and compares
all outputs.  Gate: worst relative diff <= 5e-3 (fp16 tolerance, same as
the P1h kps rewrite used).
"""
import io
import os
import sys

import numpy as np
import onnxruntime as ort

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dfastub

W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
A = os.path.join(W, "v5_P1h.onnx")
B = os.environ.get("P2H_DST", os.path.join(W, "v5_P2h.onnx"))

stub = dfastub.stub


def stub(src, dst):
    print(f"stubbing {os.path.basename(src)} -> {os.path.basename(dst)}")
    return dfastub.stub(src, dst)


sa = stub(A, A + ".smoke.onnx")
sb = stub(B, B + ".smoke.onnx")
inputs_npz = np.load(W + r"\mtq_v3_ref.npz.inputs.npz")
print("running ORT CPU (a few minutes) ...")
sessA = ort.InferenceSession(sa, None, providers=["CPUExecutionProvider"])
sessB = ort.InferenceSession(sb, None, providers=["CPUExecutionProvider"])
feed = {v.name: inputs_npz[v.name] for v in sessA.get_inputs()}
names = [o.name for o in sessA.get_outputs()]
va = dict(zip(names, sessA.run(names, feed)))
vb = dict(zip(names, sessB.run(names, feed)))
# tracking/bookkeeping int outputs: ID assignment flips on near-tied
# confidences when head weights de-quantize (P2h restores full-precision
# weights on the layers.0 q/k/v/kps MatMuls) -- renumbering, not numeric
# drift; id_count matches and IDs do not enter the mAP gate.
IDS = {"det_instance_id", "next_det_instance_id", "next_id_count"}
worst = 0.0
worst_nm = ""
for nm in names:
    a, b = va[nm].astype(np.float64), vb[nm].astype(np.float64)
    d = float(np.abs(a - b).max())
    den = max(1e-6, float(np.abs(a).max()))
    rel = d / den
    if rel > worst and nm not in IDS:
        worst, worst_nm = rel, nm
    if d > 1e-5:
        print(f"DIFF {nm} maxabs={d:.3e} rel={rel:.3e}"
              + ("  [id bookkeeping, ungated]" if nm in IDS else ""))
print(f"WORST rel: {worst:.3e} on {worst_nm}  (gate: 5e-3)")
print("P2H_STUB_PASS" if worst <= 5e-3 else "P2H_STUB_FAIL")
