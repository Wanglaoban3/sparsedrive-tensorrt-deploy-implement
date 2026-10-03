# -*- coding: utf-8 -*-
"""poll e_T12 build completion on board; print tail of build log."""
import os
import io
import sys
import time

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=30):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


for i in range(40):  # up to ~40 min
    done = run("cat /opt/m0/trt-dev/repro/E_T12_DONE 2>/dev/null").strip()
    if done:
        print(f"DONE: {done}")
        print(run("tail -15 /opt/m0/trt-dev/repro/e_T12_build.log"))
        print(run("ls -la /opt/m0/trt-dev/models/e_T12.engine"))
        sys.exit(0)
    if i % 5 == 0:
        alive = run(
            "ps aux | grep onnx2engine2 | grep -v grep | wc -l").strip()
        logsz = run("stat -c %s /opt/m0/trt-dev/repro/e_T12_build.log "
                    "2>/dev/null").strip()
        print(f"[{i}m] alive={alive} log={logsz}B")
    time.sleep(60)
print("TIMEOUT after 40m")
print(run("tail -20 /opt/m0/trt-dev/repro/e_T12_build.log"))
sys.exit(1)
