# -*- coding: utf-8 -*-
"""Poll DET3_DONE, print logs, fetch prof logs + outs3_00."""
import io
import os
import sys
import time

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
DST = (r"<REPO>\work_dirs"
       r"\sparsedrive_small_stage2\evaldata")
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=60):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


t0 = time.time()
while True:
    if run("test -f /opt/m0/trt-dev/mods/DET3_DONE && echo Y").strip() == "Y":
        break
    if time.time() - t0 > 1500:
        print("TIMEOUT")
        print(run("tail -10 /opt/m0/trt-dev/det3_run.log"))
        sys.exit(1)
    time.sleep(20)
print("DET3 done after %.0fs" % (time.time() - t0))
print(run("cat /opt/m0/trt-dev/det3_run.log"))
sftp = cli.open_sftp()
for lf in ("e_det2_prof.log", "e_rest_prof.log"):
    try:
        sftp.get("/opt/m0/trt-dev/" + lf, os.path.join(DST, lf))
        print("fetched", lf)
    except IOError:
        print("no", lf)
try:
    names = sftp.listdir("/opt/m0/trt-dev/vec/mini/outs3_00")
    for nm in names:
        sftp.get("/opt/m0/trt-dev/vec/mini/outs3_00/" + nm,
                 os.path.join(DST, "outs3_00_" + nm))
    print("fetched", len(names), "outs3_00 files")
except IOError:
    print("no outs3_00")
sftp.close()
cli.close()
