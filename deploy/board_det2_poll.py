# -*- coding: utf-8 -*-
"""Poll DET2_DONE, report profiles, fetch prof logs."""
import io
import os
import sys
import time

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
DST = (r"<REPO>\evaldata")
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=60):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


t0 = time.time()
while True:
    if run("test -f /opt/m0/trt-dev/mods/DET2_DONE && echo Y").strip() == "Y":
        break
    if time.time() - t0 > 1500:
        print("TIMEOUT")
        print(run("tail -10 /opt/m0/trt-dev/det2_run.log"))
        sys.exit(1)
    time.sleep(20)
print("DET2 done after %.0fs" % (time.time() - t0))
print(run("cat /opt/m0/trt-dev/det2_run.log"))
sftp = cli.open_sftp()
os.makedirs(DST, exist_ok=True)
for lf in ("e_det_prof.log", "e_rest_prof.log"):
    try:
        sftp.get("/opt/m0/trt-dev/" + lf, os.path.join(DST, lf))
        print("fetched", lf)
    except IOError:
        print("no", lf)
sftp.close()
cli.close()
