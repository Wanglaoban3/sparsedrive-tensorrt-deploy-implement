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


print("== oe_a build error ==")
print(run("tail -12 /opt/m0/trt-dev/mods/oe_a_build.log"))
print("== oe_b profile ==")
print(run("grep -A30 '=== Profile' /opt/m0/trt-dev/mods/oe_b_prof.log "
          "| grep -E '\\[I\\] +[0-9]'"))
print("== oe_b build tail ==")
print(run("grep -E 'flags|OK|bytes' /opt/m0/trt-dev/mods/oe_b_build.log"))
cli.close()
