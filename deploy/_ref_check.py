# -*- coding: utf-8 -*-
"""悬空引用校验: 文档里提到的每个 *.py 文件名必须真实存在于仓库."""
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
on_disk = set()
for dp, _, fns in os.walk(ROOT):
    dirs = dp.split(os.sep)
    if any(x in dirs for x in (".git", "work_dirs", "evaldata")):
        continue
    for fn in fns:
        on_disk.add(fn)

docs = ["AGENTS.md"] + [os.path.join(dp, fn) for dp, _, fns in
                        os.walk(os.path.join(ROOT, "docs"))
                        for fn in fns if fn.endswith(".md")]
pat = re.compile(r"\b([A-Za-z0-9_]+\.(?:py|cpp|h|cu|sh|conf|json))\b")
missing = {}
for f in docs:
    txt = open(f, encoding="utf-8", errors="replace").read()
    for m in pat.findall(txt):
        if m not in on_disk and m not in missing:
            missing[m] = os.path.relpath(f, ROOT)

if missing:
    print("DANGLING REFS:")
    for k, v in sorted(missing.items()):
        print("  %-36s (referenced in %s)" % (k, v))
else:
    print("NO_DANGLING_REFS (all documented files exist)")
