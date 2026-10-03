# -*- coding: utf-8 -*-
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
s = open(r"<REPO>"
         r"\docs\OPTIMIZATION_SUMMARY.md", encoding="utf-8").read().splitlines()
print("lines", len(s))
for i, l in enumerate(s):
    if l.startswith("#"):
        print(i + 1, l)
