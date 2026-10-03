# -*- coding: utf-8 -*-
"""poll v8 engine A/B results"""
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


def run(cmd, t=120):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


t0 = time.time()
ok = False
while time.time() - t0 < 1500:
    if "YES" in run("ls /opt/m0/trt-dev/mods/V8AB_DONE 2>/dev/null && echo YES"):
        ok = True
        print("done in %.0fs" % (time.time() - t0))
        break
    time.sleep(20)
if not ok:
    print("TIMEOUT; tail:")
    print(run("tail -5 /opt/m0/trt-dev/mods/v8ab.out 2>/dev/null; "
              "tail -3 /opt/m0/trt-dev/mods/v8ab_v8_prof.log 2>/dev/null"))

print(run("cat /opt/m0/trt-dev/mods/v8ab.log"))
print("=== v3 profile rows ===")
print(run("grep -A80 '=== Profile' /opt/m0/trt-dev/mods/v8ab_v3_prof.log | "
          "grep -E 'Time\\(ms\\)|Deformable' | head -8"))
print("=== v8 profile rows ===")
print(run("grep -A80 '=== Profile' /opt/m0/trt-dev/mods/v8ab_v8_prof.log | "
          "grep -E 'Time\\(ms\\)|Deformable' | head -8"))
cli.close()
