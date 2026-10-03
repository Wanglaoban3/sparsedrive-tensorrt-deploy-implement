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


print("== leftover procs ==")
print(run("ps aux | grep -E 'trtexec|run_engine|run_v8|mini_pipeline' "
          "| grep -v grep; true"))
print("== deliverable ==")
print(run("md5sum /usr/local/lib/libdfaplug_v8.so "
          "/usr/local/lib/libdfaplug_v3.so; ls -la "
          "/usr/local/lib/libdfaplug_v8.so"))
print("== other processes (user's, untouched) ==")
print(run("ps aux | grep -E 'onevl|http.server' | grep -v grep | head -3"))
cli.close()
