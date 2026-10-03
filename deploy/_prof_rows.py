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


print("== v8 Deformable rows ==")
print(run("grep 'Deformable' /opt/m0/trt-dev/v8_prof2.log"))
print("== v8 Total ==")
print(run("grep ' Total' /opt/m0/trt-dev/v8_prof2.log"))
print("== v3 Deformable rows (baseline) ==")
print(run("grep 'Deformable' /opt/m0/trt-dev/v8ab_v3_prof.log"))
print("== v3 Total ==")
print(run("grep ' Total' /opt/m0/trt-dev/v8ab_v3_prof.log"))
cli.close()
