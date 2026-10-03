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


print("== w1 log errors ==")
print(run("grep -iE 'error|invalid|failed|unsupported' "
          "/opt/m0/trt-dev/mods/w1_prof.log | head -10"))
print(run("tail -5 /opt/m0/trt-dev/mods/w1_prof.log"))
print("== w2 build errors ==")
print(run("grep -iE 'error|invalid|In node' "
          "/opt/m0/trt-dev/mods/w2_build.log | head -10"))
cli.close()
