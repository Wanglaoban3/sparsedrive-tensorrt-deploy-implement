# -*- coding: utf-8 -*-
"""Rearrange flat outs_XX_name.bin into outs_XX/name.bin dirs."""
import collections
import os
import re
import shutil
import sys

D = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2\evaldata\mini_eng_sp")
pat = re.compile(r"^(outs_\d+)_(.+\.bin|manifest\.tsv)$")
groups = collections.defaultdict(list)
for f in os.listdir(D):
    m = pat.match(f)
    if m:
        groups[m.group(1)].append(f)
for k, files in groups.items():
    kd = os.path.join(D, k)
    os.makedirs(kd, exist_ok=True)
    for f in files:
        shutil.move(os.path.join(D, f),
                    os.path.join(kd, pat.match(f).group(2)))
print("organized", len(groups), "frame dirs")
