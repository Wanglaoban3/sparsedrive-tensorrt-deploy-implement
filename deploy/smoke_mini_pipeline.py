# -*- coding: utf-8 -*-
"""smoke: run 2 frames of the fixed mini_pipeline.sh on board, check outputs"""
import os
import paramiko

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)
cmd = (
    "cd /opt/m0/trt-dev && rm -rf vec/mini/out_0 vec/mini/out_1 repro/mini_0.log "
    "repro/mini_1.log && rm -f repro/mini_pipeline.log && "
    "timeout 300 bash repro/mini_pipeline.sh 0 1 > repro/smoke.log 2>&1; rc=$?; "
    "echo rc=$rc; tail -3 repro/mini_pipeline.log; "
    "echo ---; ls vec/mini/out_00 | wc -l; ls vec/mini/out_01 | wc -l; "
    "echo ---; tail -6 repro/mini_01.log"
)
_, out, err = cli.exec_command(cmd, timeout=330)
print(out.read().decode("utf-8", "replace"))
e = err.read().decode("utf-8", "replace")
if e.strip():
    print("STDERR:", e)
cli.close()
