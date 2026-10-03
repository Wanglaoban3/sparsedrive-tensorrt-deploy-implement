# -*- coding: utf-8 -*-
"""Poll BB2_DONE, print logs, fetch prof + dumps."""
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
    if run("test -f /opt/m0/trt-dev/mods/BB2_DONE && echo Y").strip() == "Y":
        break
    if time.time() - t0 > 1500:
        print("TIMEOUT")
        print(run("tail -10 /opt/m0/trt-dev/bb2_run.log"))
        sys.exit(1)
    time.sleep(20)
print("BB2 done after %.0fs" % (time.time() - t0))
print(run("cat /opt/m0/trt-dev/bb2_run.log"))
sftp = cli.open_sftp()
try:
    sftp.get("/opt/m0/trt-dev/e_bb2_prof.log", os.path.join(DST,
                                                           "e_bb2_prof.log"))
    print("fetched e_bb2_prof.log")
except IOError:
    pass
for rdir, tag in (("vec/mini/outs3b_00", "outs3b_00_"),
                  ("vec/mini/outs2b_00", "outs2b_00_")):
    try:
        names = sftp.listdir("/opt/m0/trt-dev/" + rdir)
    except IOError:
        print("no", rdir)
        continue
    for nm in names:
        sftp.get("/opt/m0/trt-dev/" + rdir + "/" + nm,
                 os.path.join(DST, tag + nm))
    print("fetched", len(names), tag)
sftp.close()
cli.close()
