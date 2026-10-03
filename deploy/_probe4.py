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


print(run("ps ax | grep -E 'onnx2engine2|trtexec|run_engine2|run_split3' "
          "| grep -v grep"))
print(run("cat /opt/m0/trt-dev/split3_run.log 2>&1"))
print(run("ls -la /opt/m0/trt-dev/models/e_*.engine 2>&1"))
cli.close()
