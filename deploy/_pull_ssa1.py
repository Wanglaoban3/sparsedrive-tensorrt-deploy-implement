# -*- coding: utf-8 -*-
"""Pull SSA1 artifacts: logs, profiles, both det dump dirs (leading-/ names
sanitized on the way over - P4 pitfall)."""
import os
import io
import sys
from pathlib import Path

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = Path(r"<REPO>"
         r"\work_dirs\sparsedrive_small_stage2")
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)
s = cli.open_sftp()

FILES = ["ssa1_run.log", "v9_build.log", "detssa_build.log",
         "e_detssa_prof.log", "e_det_prof2.log",
         "one_detref.log", "one_detssa.log"]
for f in FILES:
    try:
        s.get(f"/opt/m0/trt-dev/{f}", str(W / f))
        print("got", f)
    except FileNotFoundError:
        print("MISSING", f)

for rdir, ldir in [("outs_detref_00", "outs_detref_00"),
                   ("outs_detssa_00", "outs_detssa_00")]:
    (W / ldir).mkdir(exist_ok=True)
    try:
        names = s.listdir(f"/opt/m0/trt-dev/vec/mini/{rdir}")
    except FileNotFoundError:
        print("MISSING dir", rdir)
        continue
    for n in names:
        local = str(W / ldir / n.lstrip("/"))
        s.get(f"/opt/m0/trt-dev/vec/mini/{rdir}/{n}", local)
    print(f"got {rdir}: {len(names)} files")
s.close()
cli.close()
