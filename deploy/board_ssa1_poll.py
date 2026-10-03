# -*- coding: utf-8 -*-
"""Poll SSA job: argv[1]=local wait s, argv[2]=marker name (default SSA1)."""
import os
import io
import sys
import time

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
wait = float(sys.argv[1]) if len(sys.argv) > 1 else 0
marker = sys.argv[2] if len(sys.argv) > 2 else "SSA1"
log = sys.argv[3] if len(sys.argv) > 3 else "ssa1_run.log"
if wait:
    time.sleep(wait)
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=60):
    _, out, err = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace") + \
        err.read().decode("utf-8", "replace")


print(run(f"ls /opt/m0/trt-dev/mods/{marker}_DONE 2>/dev/null && echo "
          f"MARKER_YES || echo MARKER_NO"))
print(run(f"tail -25 /opt/m0/trt-dev/{log} 2>/dev/null"))
print(run("pgrep -af 'onnx2engine2|nvcc|trtexec|run_ssa|run_engines' "
          "| grep -v pgrep | head -5"))
cli.close()
