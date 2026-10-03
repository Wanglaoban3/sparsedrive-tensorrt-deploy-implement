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


for i in range(40):
    done = run("test -f /opt/m0/trt-dev/mods/W3_DONE && echo Y").strip()
    log = run("cat /opt/m0/trt-dev/mods/w3_run.log 2>/dev/null") \
        .strip().replace("\n", " | ")
    print(f"[{i*30}s] {log}{' DONE' if done == 'Y' else ''}")
    if done == "Y":
        break
    time.sleep(30)
for f in ("w3a_prof", "w3b_prof"):
    print(f"##### {f}")
    print(run(f"grep -E 'ForeignNode|Total' "
              f"/opt/m0/trt-dev/mods/{f}.log 2>/dev/null "
              f"| grep '\\[I\\]' | grep -v Reformat | grep -v Memory"))
cli.close()
