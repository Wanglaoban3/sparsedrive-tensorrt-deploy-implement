# -*- coding: utf-8 -*-
"""Pull SSA6 replay outputs."""
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
try:
    s.get("/opt/m0/trt-dev/ssa6_run.log", str(W / "ssa6_run.log"))
    print("got ssa6_run.log")
except FileNotFoundError:
    print("MISSING log")
(W / "outs_replay").mkdir(exist_ok=True)
names = s.listdir("/opt/m0/trt-dev/vec/mini/outs_replay")
for n in names:
    s.get(f"/opt/m0/trt-dev/vec/mini/outs_replay/{n}",
          str(W / "outs_replay" / n))
print(f"got outs_replay: {len(names)} files")
s.close()
cli.close()
