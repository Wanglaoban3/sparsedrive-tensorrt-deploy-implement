# -*- coding: utf-8 -*-
import os
import io
import sys

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=30):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


print(run("ps aux | grep onnx2engine | grep -v grep; "
          "ls -la /opt/m0/trt-dev/repro/e_T12_build.log "
          "/opt/m0/trt-dev/repro/E_T12_DONE 2>/dev/null; "
          "tail -5 /opt/m0/trt-dev/repro/e_T12_build.log 2>/dev/null"))
cli.close()
