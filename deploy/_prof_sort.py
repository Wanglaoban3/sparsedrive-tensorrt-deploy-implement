"""Dump sorted local profile rows (>=0.08ms) for e_det_ssa vs e_det."""
import re
from pathlib import Path

W = Path(r"<REPO>"
         r"\work_dirs\sparsedrive_small_stage2")
rx = re.compile(r"\[I\]\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.+)")
for f in ["e_detssa_prof.log", "e_det_prof2.log"]:
    rows = []
    for line in (W / f).read_text(errors="replace").splitlines():
        m = rx.search(line)
        if m:
            rows.append((float(m.group(2)), m.group(5)))
    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows)
    print(f"=== {f}: {len(rows)} rows, row-sum {tot:.2f} ms, top 26 ===")
    for ms, name in rows[:26]:
        print(f"  {ms:7.3f}  {name[:95]}")
