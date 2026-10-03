# -*- coding: utf-8 -*-
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
s = open(r"<REPO>"
         r"\deploy\eval_t6_mini_v8.py", encoding="utf-8").read()
for i, l in enumerate(s.splitlines()):
    if "outv8_" in l or "mini_eng_v8" in l:
        print(i + 1, l)
