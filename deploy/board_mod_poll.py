# -*- coding: utf-8 -*-
"""poll the board module A/B: DONE marker, per-module GPU compute means"""
import os
import io
import re
import sys
import time

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
BD = "/opt/m0/trt-dev/mods"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=30):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


while True:
    done = run(f"ls {BD}/DONE 2>/dev/null && echo YES || echo NO").strip()
    print(f"[{time.strftime('%H:%M:%S')}] DONE={done}")
    if done.endswith("YES"):
        break
    print(run(f"tail -3 {BD}/mods_run.log"))
    time.sleep(60)

for f in ("mod_p1h", "mod_p2h", "modbig_p1h", "modbig_p2h"):
    log = run(f"cat {BD}/{f}.log")
    rc = run(f"grep -o 'rc=[0-9]*' {BD}/mods_run.log | tail -4")
    mean = re.findall(r"GPU Compute Time: min ([0-9.]+) ms, mean ([0-9.]+) ms"
                      r".*max ([0-9.]+) ms", log)
    lat = re.findall(r"Latency: min ([0-9.]+) ms, mean ([0-9.]+) ms"
                     r".*max ([0-9.]+) ms", log)
    print(f"===== {f}  (rc lines: {rc.strip().splitlines()[-1:]})")
    print("  GPU compute mean:",
          mean[0][1] + " ms" if mean else "n/a")
    print("  Latency mean:", lat[0][1] + " ms" if lat else "n/a")
    for line in log.splitlines():
        if "Slowest" in line or "percentile" in line:
            print("  ", line.strip())
cli.close()
