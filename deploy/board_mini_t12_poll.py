# -*- coding: utf-8 -*-
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


def run(cmd, t=60):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


for i in range(60):
    log = run("cat /opt/m0/trt-dev/repro/mini_pipeline_t12.log 2>/dev/null")
    done = "MINI_DONE" in log
    if i % 3 == 0 or done:
        tail = log.strip().splitlines()
        print(f"[{i*30}s] frames done: {len(tail)} "
              f"last: {tail[-1] if tail else '-'}")
    if done:
        break
    time.sleep(30)
fails = run("grep FAIL /opt/m0/trt-dev/repro/mini_pipeline_t12.log "
            "2>/dev/null")
print("FAILS:", fails.strip() or "none")
print(run("ls /opt/m0/trt-dev/vec/mini/ | grep -c out7_"))
cli.close()
