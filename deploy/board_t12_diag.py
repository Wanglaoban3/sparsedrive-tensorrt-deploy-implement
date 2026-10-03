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


print("== e_T12 build log: precision/flags lines ==")
print(run("grep -inE 'flag|precision|fp16|half|int8|tf32|strict' "
          "/opt/m0/trt-dev/repro/e_T12_build.log | head -30"))
print("== t12 ForeignNode rows ==")
print(run("grep 'ForeignNode' /opt/m0/trt-dev/repro/t12_prof.log "
          "| grep '\\[I\\]' | grep -v Reformat"))
print("== onnx2engine2 source location ==")
print(run("ls -la /opt/m0/trt-dev/src/ && "
          "grep -rl 'onnx2engine2\\|f32-notq' /opt/m0/trt-dev/src/ "
          "/root 2>/dev/null | head"))
cli.close()
