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
    _, o, e = cli.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8", "replace") + \
        e.read().decode("utf-8", "replace")


print(run("cat /opt/m0/trt-dev/v8ab.log 2>/dev/null"))
print("=== v3 Deformable rows ===")
print(run("grep -B2 -A80 '=== Profile' /opt/m0/trt-dev/v8ab_v3_prof.log 2>/dev/null | grep -E 'Time|Deformable' | head -6"))
print("=== v8 Deformable rows ===")
print(run("grep -B2 -A80 '=== Profile' /opt/m0/trt-dev/v8ab_v8_prof.log 2>/dev/null | grep -E 'Time|Deformable' | head -6"))
print(run("ls /opt/m0/trt-dev/vec/mini/out_v8_00/ 2>/dev/null | head -5"))
print(run("which python python3 2>/dev/null; ls /usr/bin/python* 2>/dev/null"))
cli.close()
