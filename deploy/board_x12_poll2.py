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


for i in range(44):
    done = run("test -f /opt/m0/trt-dev/mods/X12_DONE && echo Y").strip()
    log = run("grep -E 'rc=|===' /opt/m0/trt-dev/mods/x12_run.log 2>/dev/null"
              ) .strip().replace("\n", " | ")
    print(f"[{i*30}s] {log}{' DONE' if done == 'Y' else ''}")
    if done == "Y":
        break
    time.sleep(30)
print("== X1 FN rows ==")
print(run("grep -E 'ForeignNode|Total' "
          "/opt/m0/trt-dev/mods/x1_prof.log 2>/dev/null "
          "| grep '\\[I\\]' | grep -v Reformat"))
print("== X2 FN rows ==")
print(run("grep -E 'ForeignNode|Total' "
          "/opt/m0/trt-dev/mods/x2_prof.log 2>/dev/null "
          "| grep '\\[I\\]' | grep -v Reformat"))
print("== X2 build key lines ==")
print(run("grep -E 'flags|f16-notq|OK:|FAILED|Error' "
          "/opt/m0/trt-dev/mods/x2_build.log 2>/dev/null | head -8"))
print("== X1 build key lines ==")
print(run("grep -E 'flags|OK:|FAILED|Error' "
          "/opt/m0/trt-dev/mods/x1_build.log 2>/dev/null | head -8"))
cli.close()
