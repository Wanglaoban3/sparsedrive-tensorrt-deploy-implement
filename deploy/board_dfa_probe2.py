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


def run(cmd, t=120):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


print(run(
    "find /opt/m0 /data /tmp -maxdepth 4 \\( -name 'feat.f16' -o "
    "-name 'test_dfa_real2' -o -name '*.f16' \\) 2>/dev/null | head -20"))
print("== onevl dir (user's other work, read-only peek) ==")
print(run("ls /opt/m0/trt-dev/onevl/ 2>/dev/null"))
print("== repro subdir count ==")
print(run("ls /opt/m0/trt-dev/repro/ | head -30"))
cli.close()
