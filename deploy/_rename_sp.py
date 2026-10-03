# -*- coding: utf-8 -*-
"""Strip XX_ prefixes inside outs_XX frame dirs."""
import os
import re
import sys

D = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2\evaldata\mini_eng_sp")
pat = re.compile(r"^(\d+)_(.+)$")
n = 0
for d in sorted(os.listdir(D)):
    if not re.fullmatch(r"outs_\d+", d):
        continue
    kd = os.path.join(D, d)
    xx = d.split("_")[1]
    for f in os.listdir(kd):
        m = pat.match(f)
        if m and m.group(1) == xx:
            os.rename(os.path.join(kd, f), os.path.join(kd, m.group(2)))
            n += 1
print("renamed", n, "files")
