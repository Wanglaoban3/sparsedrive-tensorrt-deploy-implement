# -*- coding: utf-8 -*-
"""Poll SPLITS2_DONE, print split2_run.log, fetch outs_00 bins."""
import io
import os
import sys
import time

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
MARK = "/opt/m0/trt-dev/mods/SPLITS2_DONE"
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
    if time.time() - t0 > 1200:
        print("TIMEOUT waiting marker")
        print(run("tail -20 /opt/m0/trt-dev/split2_run.log"))
        sys.exit(1)
    time.sleep(15)
print("marker found after %.0fs" % (time.time() - t0))
print("=" * 70)
print(run("cat /opt/m0/trt-dev/split2_run.log"))
print("=" * 70)
# fetch outs_00
sftp = cli.open_sftp()
dst_dir = r"<REPO>\evaldata\mini_eng_v8"
os.makedirs(dst_dir, exist_ok=True)
try:
    names = sftp.listdir("/opt/m0/trt-dev/vec/mini/outs_00")
except IOError:
    print("fetch failed: no outs_00 dir")
    sys.exit(1)
n = 0
for nm in names:
    sftp.get("/opt/m0/trt-dev/vec/mini/outs_00/" + nm,
             os.path.join(dst_dir, "outs_00_" + nm))
    n += 1
print("fetched", n, "files to evaldata/mini_eng_v8/outs_00_*")
# also fetch manifest for shapes
try:
    sftp.get("/opt/m0/trt-dev/vec/mini/outs_00/manifest.tsv",
             os.path.join(dst_dir, "outs_00_manifest.tsv"))
    print("fetched manifest.tsv")
except IOError:
    print("no manifest.tsv")
sftp.close()
cli.close()
