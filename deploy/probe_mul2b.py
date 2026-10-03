# -*- coding: utf-8 -*-
import io
import sys

import onnx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
p = r"<REPO>\work_dirs" \
    r"\sparsedrive_small_stage2\v5_P2h.onnx"
m = onnx.load(p, load_external_data=False)
g = m.graph
tgt = "/Mul_2_output_0"
# any tensor with that exact name anywhere
for n in g.node:
    if tgt in n.output:
        print("produced by node:", n.name, n.op_type, "inputs:", list(n.input))
    if tgt in n.input:
        print("consumed by node:", n.name, n.op_type)
for io_list, tag in ((g.input, "graph-input"), (g.output, "graph-output")):
    for vi in io_list:
        if vi.name == tgt:
            print(tag, vi.name)
prods = [n.name for n in g.node if n.output and
         n.output[0].endswith("Mul_2_output_0")]
print("producers of *Mul_2_output_0:", prods)
