# -*- coding: utf-8 -*-
"""derive deploy/eval_t6_mini_sp.py from eval_t6_mini_v8.py (split-pipeline
outputs outs_XX in mini_eng_sp)."""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
ROOT = (r"<REPO>")
src = open(ROOT + r"\deploy\eval_t6_mini_v8.py", encoding="utf-8").read()
s = src.replace("mini_eng_v8", "mini_eng_sp")
s = s.replace("outv8_", "outs_")
s = s.replace("eval_v8_mini", "eval_sp_mini")
s = s.replace('"""decode e_T6 + DFA-v8-plugin outputs',
              '"""decode split-pipeline (e_bb+e_hd, v8 plug) outputs')
s = s.replace("T3 engine + v8 plug", "split bb+hd (v8 plug)")
s = s.replace("v8 vs v3 delta", "split vs v3 delta")
s = s.replace("v8map", "spmap").replace("v8nds", "spnds")
assert s != src and "outv8_" not in s and "mini_eng_v8" not in s and \
    "eval_v8_mini" not in s
open(ROOT + r"\deploy\eval_t6_mini_sp.py", "w", encoding="utf-8").write(s)
print("wrote eval_t6_mini_sp.py", len(s.splitlines()), "lines")
