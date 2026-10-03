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


def run(cmd, t=60):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


print(run("ps aux | grep -E 'run_v8final|trtexec|mini_pipeline' | grep -v grep"))
print(run("tail -3 /opt/m0/trt-dev/v8final_run.log 2>/dev/null; "
          "ls -la /opt/m0/trt-dev/v8_prof4.log 2>/dev/null"))
cli.close()
