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


print("== trt-dev layout ==")
print(run("ls -la /opt/m0/trt-dev/ | head -40"))
print("== engines ==")
print(run("ls -la /opt/m0/trt-dev/*.engine /opt/m0/trt-dev/engines/ 2>/dev/null"))
print("== build scripts mentioning e_T6 / onnx2engine ==")
print(run(
    "grep -rl 'e_T6\\|onnx2engine' /opt/m0/trt-dev --include='*.sh' 2>/dev/null; "
    "ls /opt/m0/trt-dev/*.sh 2>/dev/null"))
print("== onnx2engine help (if binary exists) ==")
print(run("/opt/m0/trt-dev/onnx2engine --help 2>&1 | head -60"))
cli.close()
