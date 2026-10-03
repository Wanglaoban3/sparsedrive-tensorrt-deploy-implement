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


print(run("echo PROBE_OK; date"))
print(run("ls -la /opt/m0/trt-dev/mods/ | grep -E 'v8ab|V8AB|run_v8' "))
print(run("ps aux | grep -E 'trtexec|run_v8ab' | grep -v grep | head"))
print(run("tail -c 800 /opt/m0/trt-dev/mods/v8ab_v3_prof.log 2>/dev/null"))
print(run("cat /opt/m0/trt-dev/mods/v8ab.out 2>/dev/null | head -5"))
cli.close()
