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


print("oe.out:", run("cat /opt/m0/trt-dev/mods/oe.out 2>/dev/null"))
for i in range(40):
    log = run("cat /opt/m0/trt-dev/mods/oe_run.log 2>/dev/null") \
        .strip().replace("\n", " | ")
    done = run("test -f /opt/m0/trt-dev/mods/OE_DONE && echo Y").strip()
    print(f"[{i*30}s] {log}{' DONE' if done == 'Y' else ''}")
    if done == "Y":
        break
    time.sleep(30)
cli.close()
