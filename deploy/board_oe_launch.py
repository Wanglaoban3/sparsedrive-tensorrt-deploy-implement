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


def run(cmd, t=120):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


run("rm -f /opt/m0/trt-dev/mods/OE_DONE /opt/m0/trt-dev/mods/oe_run.log "
    "/opt/m0/trt-dev/mods/oe.out")
run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_oe.sh > oe.out 2>&1 "
    "< /dev/null & echo GO", t=10)
print(run("sleep 8; cat /opt/m0/trt-dev/mods/oe.out; "
          "cat /opt/m0/trt-dev/mods/oe_run.log 2>/dev/null; "
          "ps aux | grep onnx2engine2 | grep -v grep | wc -l"))
for i in range(36):
    time.sleep(20)
    if run("test -f /opt/m0/trt-dev/mods/OE_DONE && echo Y").strip() == "Y":
        print(f"[{i*20}s] ALL DONE")
        break
    if i % 3 == 0:
        log = run("cat /opt/m0/trt-dev/mods/oe_run.log 2>/dev/null") \
            .strip().replace("\n", " | ")
        print(f"[{i*20}s] {log}")
cli.close()
