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


print("== x12.out ==")
print(run("cat /opt/m0/trt-dev/mods/x12.out"))
print("== full x12_run.log tail ==")
print(run("tail -6 /opt/m0/trt-dev/mods/x12_run.log"))
print("== bin ==")
print(run("ls -la /opt/m0/trt-dev/bin/"))
print("== try executing it ==")
print(run("/opt/m0/trt-dev/bin/onnx2engine3 2>&1 | head -3"))
cli.close()
