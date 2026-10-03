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


print("== onnx2engine usage ==")
print(run("/usr/local/bin/onnx2engine 2>&1 | head -50"))
print("== logs mentioning e_T6 build ==")
print(run("grep -l 'e_T6' /opt/m0/trt-dev/repro/*.log 2>/dev/null"))
print("== head of matching build logs ==")
print(run(
    "for f in $(grep -l 'Building e_T6\\|e_T6.engine' "
    "/opt/m0/trt-dev/repro/*.log 2>/dev/null | head -3); do "
    "echo \"--- $f\"; head -15 $f; done"))
print("== bash history / cmd files ==")
print(run(
    "grep -rh 'onnx2engine' /opt/m0/trt-dev/repro/*.sh "
    "/root/*.sh /tmp/*.sh 2>/dev/null | sort -u | head -20"))
cli.close()
