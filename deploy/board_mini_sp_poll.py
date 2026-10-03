# -*- coding: utf-8 -*-
"""Poll MINI_SP_DONE, fetch 81 frames of outs_XX, report progress."""
import io
import os
import re
import sys
import time

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
LOG = "/opt/m0/trt-dev/repro/mini_pipeline_sp.log"
DST = (r"<REPO>\work_dirs"
       r"\sparsedrive_small_stage2\evaldata\mini_eng_sp")
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)
sftp = cli.open_sftp()
os.makedirs(DST, exist_ok=True)

t0 = time.time()
done_k = set()
fail = None
while time.time() - t0 < 2400:
    try:
        with sftp.open(LOG, "r") as f:
            txt = f.read().decode("utf-8", "replace")
    except IOError:
        time.sleep(20)
        continue
    for m in re.finditer(r"done (\d+) files=(\d+)", txt):
        if m.group(1) not in done_k:
            done_k.add(m.group(1))
            print("frame %s done (%s files)" % (m.group(1), m.group(2)))
    if "FAIL" in txt:
        fail = [l for l in txt.splitlines() if "FAIL" in l][0]
        print(fail)
        break
    if "MINI_SP_DONE" in txt:
        print("ALL FRAMES DONE (%.0fs)" % (time.time() - t0))
        break
    # incremental fetch of completed frames
    for k in sorted(done_k - {"fetched"}):
        rdir = "/opt/m0/trt-dev/vec/mini/outs_" + k
        try:
            names = sftp.listdir(rdir)
        except IOError:
            continue
        for nm in names:
            sftp.get(rdir + "/" + nm, os.path.join(DST, "outs_" + k + "_" + nm))
        done_k.add("fetched") if len(done_k - {"fetched"}) == 81 else None
    if len([k for k in done_k if k != "fetched"]) == 81 and \
            "fetched" not in done_k:
        done_k.add("fetched")
        print("fetched all 81 frames")
    time.sleep(20)

# final fetch sweep for anything missed
n = 0
for k in ["%02d" % i for i in range(81)]:
    rdir = "/opt/m0/trt-dev/vec/mini/outs_" + k
    try:
        names = sftp.listdir(rdir)
    except IOError:
        print("missing frame", k)
        continue
    for nm in names:
        lp = os.path.join(DST, "outs_" + k + "_" + nm)
        if not os.path.exists(lp):
            sftp.get(rdir + "/" + nm, lp)
            n += 1
print("final sweep fetched", n, "new files")
sftp.close()
cli.close()
