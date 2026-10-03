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


print("== marker ==")
print(run("ls -la /opt/m0/trt-dev/mods/V8PROF2_DONE 2>&1"))
print("== procs ==")
print(run("ps aux | grep -E 'trtexec|run_engine|run_v8prof' | grep -v grep"))
print("== v8_prof2.log ==")
print(run("ls -la /opt/m0/trt-dev/v8_prof2.log 2>&1; "
          "tail -4 /opt/m0/trt-dev/v8_prof2.log 2>/dev/null"))
print("== runner stdout ==")
print(run("ls -la /opt/m0/trt-dev/v8prof2_run.out "
          "/opt/m0/trt-dev/mods/v8prof2_run.out 2>/dev/null; "
          "tail -6 /opt/m0/trt-dev/v8prof2_run.out 2>/dev/null; "
          "tail -6 /opt/m0/trt-dev/mods/v8prof2_run.out 2>/dev/null"))
print("== runner script ==")
print(run("ls -la /opt/m0/trt-dev/mods/run_v8prof.sh 2>&1; "
          "cat /opt/m0/trt-dev/mods/run_v8prof.sh 2>/dev/null"))
cli.close()
