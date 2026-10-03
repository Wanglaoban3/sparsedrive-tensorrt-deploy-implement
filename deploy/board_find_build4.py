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


print("== files containing onnx2engine2 ==")
print(run("grep -rl 'onnx2engine2' /opt/m0/trt-dev /root /tmp 2>/dev/null "
          "| head"))
print("== full build block ==")
print(run(
    "grep -rh -A8 'onnx2engine2 models' /opt/m0/trt-dev/repro/*.sh "
    "/opt/m0/trt-dev/*.sh /root/.bash_history /tmp/*.sh 2>/dev/null "
    "| head -30"))
print("== onnx2engine2 binary ==")
print(run("ls -la /usr/local/bin/onnx2engine* 2>/dev/null; "
          "/usr/local/bin/onnx2engine2 2>&1 | head -20"))
print("== bash_history onnx2engine lines ==")
print(run("grep -n 'onnx2engine' /root/.bash_history 2>/dev/null | tail -10"))
cli.close()
