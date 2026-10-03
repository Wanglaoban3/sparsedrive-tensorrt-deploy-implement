# -*- coding: utf-8 -*-
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
rows = []
tot = 0.0
for l in open(r"<REPO>\deploy"
              r"\artifacts\profile\t12_prof.log",
              encoding="utf-8", errors="replace"):
    m = re.search(r"\[I\]\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
                  r"([0-9.]+)\s+(.+?)\s*$", l.rstrip())
    if m and "Reformatting" in m.group(5):
        rows.append((float(m.group(2)), m.group(5)[:95]))
        tot += float(m.group(2))
rows.sort(reverse=True)
print(f"reformat total ms/iter: {tot:.3f}  rows: {len(rows)}")
for t, n in rows[:14]:
    print(f"{t:.4f}  {n}")
