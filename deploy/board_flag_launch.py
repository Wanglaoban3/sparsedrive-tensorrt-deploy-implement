# -*- coding: utf-8 -*-
import os
import io
import sys
import time

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=60):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


run("rm -f /opt/m0/trt-dev/mods/FLAGS_DONE /opt/m0/trt-dev/mods/flags_run.log")
run("cd /opt/m0/trt-dev/mods && setsid nohup ./run_flags.sh > run_flags.out "
    "2>&1 < /dev/null & echo GO", t=10)
for i in range(30):
    time.sleep(15)
    if run("test -f /opt/m0/trt-dev/mods/FLAGS_DONE && echo Y").strip() == "Y":
        print("DONE")
        break
    if i % 4 == 0:
        print(f"[{i*15}s] running:",
              run("ps aux | grep trtexec | grep -v grep | wc -l").strip())
cli.close()
