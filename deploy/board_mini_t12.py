# -*- coding: utf-8 -*-
"""launch mini_pipeline_t12.sh (81-frame chained, e_T12 -> out7_*)."""
import os
import io
import sys

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=120):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


run("rm -f /opt/m0/trt-dev/repro/MINI_T12_DONE")
run("cd /opt/m0/trt-dev && setsid nohup bash repro/mini_pipeline_t12.sh "
    "> repro/mini_t12.out 2>&1 < /dev/null & echo GO", t=5)
print("launched mini_pipeline_t12 (81 frames)")
print(run("sleep 20; tail -3 /opt/m0/trt-dev/repro/mini_t12.out; "
          "cat /opt/m0/trt-dev/repro/mini_pipeline_t12.log 2>/dev/null"))
cli.close()
