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


print("== queue_fp32_temporal.sh ==")
print(run("cat /opt/m0/trt-dev/queue_fp32_temporal.sh"))
print("== bin/ ==")
print(run("ls -la /opt/m0/trt-dev/bin/"))
print("== models dir ==")
print(run("ls -la /opt/m0/trt-dev/models/ | head -30"))
print("== where is e_T6.engine ==")
print(run("find /opt/m0 -name 'e_T6*' 2>/dev/null | head"))
print("== mini_pipeline.sh (ENGINE/PLUGIN/build lines) ==")
print(run("grep -nE 'ENGINE|PLUGIN|trtexec|onnx2engine|e_T' "
          "/opt/m0/trt-dev/repro/mini_pipeline.sh | head -30"))
cli.close()
