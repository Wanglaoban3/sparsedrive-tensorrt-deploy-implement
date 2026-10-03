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


print(run("ls -la /opt/m0/trt-dev/mods/ | grep -E 'flags|FLAG|run_flags'"))
print("== run_flags.sh ==")
print(run("cat /opt/m0/trt-dev/mods/run_flags.sh 2>/dev/null | head -8"))
print("== still running? ==")
print(run("ps aux | grep trtexec | grep -v grep"))
print("== first log tail ==")
print(run("tail -5 /opt/m0/trt-dev/mods/modbig_p2h_fp16i8.log 2>/dev/null"))
cli.close()
