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


print("=== v3 Deformable rows ===")
print(run("grep 'Deformable' /opt/m0/trt-dev/v8ab_v3_prof.log"))
print("=== v8 prof log tail ===")
print(run("tail -15 /opt/m0/trt-dev/v8ab_v8_prof.log 2>/dev/null"))
print("=== v8 Deformable rows ===")
print(run("grep 'Deformable' /opt/m0/trt-dev/v8ab_v8_prof.log 2>/dev/null"))
print("=== v8ab.log GPU compute time lines ===")
print(run("grep -E 'GPU Compute Time|mean|median' /opt/m0/trt-dev/v8ab_v3_prof.log | tail -4"))
print(run("grep -E 'GPU Compute Time|mean|median' /opt/m0/trt-dev/v8ab_v8_prof.log 2>/dev/null | tail -4"))
cli.close()
