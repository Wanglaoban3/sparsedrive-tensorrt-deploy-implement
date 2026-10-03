# -*- coding: utf-8 -*-
"""inspect mini_00.log failure"""
import os
import paramiko

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)
cmd = (
    "tail -30 /opt/m0/trt-dev/repro/mini_00.log; "
    "echo ===SMOKE_TAIL===; tail -15 /opt/m0/trt-dev/repro/smoke.log"
)
_, out, err = cli.exec_command(cmd, timeout=60)
print(out.read().decode("utf-8", "replace"))
e = err.read().decode("utf-8", "replace")
if e.strip():
    print("STDERR:", e)
cli.close()
