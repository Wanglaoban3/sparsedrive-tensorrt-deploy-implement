# -*- coding: utf-8 -*-
"""Pull SSA5 artifacts: selftest log + probe dumps."""
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

for f in ["ssa5_run.log", "selftest_build.log"]:
    try:
        s.get(f"/opt/m0/trt-dev/{f}", str(W / f))
        print("got", f)
    except FileNotFoundError:
        print("MISSING", f)

for rdir, ldir in [("outs_probe_base", "outs_probe_base"),
                   ("outs_probe_fa", "outs_probe_fa")]:
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
