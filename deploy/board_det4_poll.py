# -*- coding: utf-8 -*-
"""Poll DET4_DONE, fetch outs3_00."""
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
    if run("test -f /opt/m0/trt-dev/mods/DET4_DONE && echo Y").strip() == "Y":
        break
    if time.time() - t0 > 1200:
        print("TIMEOUT")
        print(run("tail -10 /opt/m0/trt-dev/det4_run.log"))
        sys.exit(1)
    time.sleep(15)
print("DET4 done after %.0fs" % (time.time() - t0))
print(run("cat /opt/m0/trt-dev/det4_run.log"))
sftp = cli.open_sftp()
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
