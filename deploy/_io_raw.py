# -*- coding: utf-8 -*-
import io
import sys

import onnx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
SRC = (r"<REPO>\work_dirs"
       r"\sparsedrive_small_stage2\v5_P1h.onnx")
m = onnx.load(SRC, load_external_data=False)
g = m.graph
print("== g.input raw ==")
for v in g.input:
    print(v.name, "type_case=", v.type.WhichOneof("value"),
          str(v)[:200].replace("\n", " "))
print("== g.output raw (first 3) ==")
for v in list(g.output)[:3]:
    print(v.name, str(v)[:200].replace("\n", " "))
print("== value_info count ==", len(g.value_info))
for v in list(g.value_info)[:3]:
    print(v.name, str(v)[:160].replace("\n", " "))
