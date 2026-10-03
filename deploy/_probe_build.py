# -*- coding: utf-8 -*-
"""Probe: is the e_det_ssa build working or hung?"""
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
    _, out, err = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace") + \
        err.read().decode("utf-8", "replace")


print(run("date"))
print(run("wc -l /opt/m0/trt-dev/detssa_build.log "
          "/opt/m0/trt-dev/det_build.log 2>/dev/null"))
print(run("tail -2 /opt/m0/trt-dev/detssa_build.log"))
print(run("top -b -n1 | head -12"))
cli.close()
