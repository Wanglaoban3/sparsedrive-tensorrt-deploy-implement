# -*- coding: utf-8 -*-
"""Poll SPLIT3_DONE, print split3_run.log + profile tails, fetch outs_00."""
import io
import os
import sys
import time

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
MARK = "/opt/m0/trt-dev/mods/SPLIT3_DONE"
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=60):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


t0 = time.time()
while True:
    if run("test -f " + MARK + " && echo Y").strip() == "Y":
        break
    if time.time() - t0 > 1500:
        print("TIMEOUT; tail so far:")
        print(run("tail -20 /opt/m0/trt-dev/split3_run.log"))
        sys.exit(1)
    time.sleep(20)
print("marker found after %.0fs" % (time.time() - t0))
print("=" * 70)
print(run("cat /opt/m0/trt-dev/split3_run.log"))
print("=" * 70)
print("--- e_bb profile top ---")
print(run("grep -A40 '=== Profile ===' /opt/m0/trt-dev/e_bb_prof.log | "
          "sort -k2 -g -r | head -14"))
print("--- e_hd profile top ---")
print(run("grep -A40 '=== Profile ===' /opt/m0/trt-dev/e_hd_prof.log | "
          "sort -k2 -g -r | head -26"))
print("--- one_frame tail ---")
print(run("tail -15 /opt/m0/trt-dev/one_frame.log"))
print("=" * 70)
sftp = cli.open_sftp()
dst_dir = r"<REPO>\evaldata\mini_eng_v8"
os.makedirs(dst_dir, exist_ok=True)
try:
    names = sftp.listdir("/opt/m0/trt-dev/vec/mini/outs_00")
except IOError:
    print("fetch failed: no outs_00 dir")
    sys.exit(1)
for nm in names:
    sftp.get("/opt/m0/trt-dev/vec/mini/outs_00/" + nm,
             os.path.join(dst_dir, "outs_00_" + nm))
print("fetched", len(names), "files -> evaldata/mini_eng_v8/outs_00_*")
for lf in ("e_bb_prof.log", "e_hd_prof.log"):
    sftp.get("/opt/m0/trt-dev/" + lf, os.path.join(dst_dir, lf))
    print("fetched", lf)
sftp.close()
cli.close()
