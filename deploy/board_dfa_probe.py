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


print("== find dfa dump dirs ==")
print(run("find /opt/m0/trt-dev -maxdepth 3 -name 'manifest.txt' "
          "-o -maxdepth 3 -name 'feat.f16' 2>/dev/null | head"))
print("== test binaries ==")
print(run("ls -la /opt/m0/trt-dev/vec/ 2>/dev/null | head -15"))
print(run("find /opt/m0/trt-dev -maxdepth 2 -name 'test_dfa*' "
          "-o -maxdepth 2 -name 'dfa_timer*' 2>/dev/null"))
cli.close()
