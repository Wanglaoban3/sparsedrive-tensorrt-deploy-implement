# -*- coding: utf-8 -*-
"""Poll SPLIT4_DONE, fetch outbb_00 + outhd_iso_00, run comparisons."""
import io
import os
import sys
import time

import numpy as np
import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
MARK = "/opt/m0/trt-dev/mods/SPLIT4_DONE"
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
    if run("test -f " + MARK + " && echo Y").strip() == "Y":
        break
    if time.time() - t0 > 900:
        print("TIMEOUT")
        print(run("tail -20 /opt/m0/trt-dev/split4_run.log"))
        sys.exit(1)
    time.sleep(10)
print("marker after %.0fs" % (time.time() - t0))
print(run("cat /opt/m0/trt-dev/split4_run.log"))
sftp = cli.open_sftp()
for rdir, tag in (("vec/mini/outbb_00", "outbb_00_"),
                  ("vec/mini/outhd_iso_00", "outhd_iso_00_"),
                  ("vec/mini/outs_00b", "outs_00b_")):
    try:
        names = sftp.listdir("/opt/m0/trt-dev/" + rdir)
    except IOError:
        print("missing", rdir)
        continue
    for nm in names:
        sftp.get("/opt/m0/trt-dev/" + rdir + "/" + nm,
                 os.path.join(DST, tag + nm))
    print("fetched", len(names), tag)
sftp.close()
cli.close()

# ---- compare bb boundary vs P3 f32 dump ----
p3 = np.fromfile(os.path.join(DST, "dump_out",
                              "Reshape_9_output_0.__dump.bin"), np.float32)
bbf = os.path.join(DST, "outbb_00_/Reshape_9_output_0.bin")
if not os.path.exists(bbf):
    bbf = os.path.join(DST, "outbb_00_/Reshape_9_output_0.bin")
a16 = np.fromfile(bbf, np.float16).astype(np.float32)
print("\n== bb boundary vs e_T6 internal (P3 dump, f32) ==")
print("elems", p3.size, a16.size)
d = np.abs(a16 - p3)
den = np.abs(p3).max()
print("maxabs %.4e  rel %.3e  p99.9 %.3e" %
      (d.max(), d.max() / den, np.percentile(d, 99.9)))
