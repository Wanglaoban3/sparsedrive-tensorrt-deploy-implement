# -*- coding: utf-8 -*-
"""poll V8WS_DONE; then print compile rc, debug lines, Deformable rows;
fetch 1-frame v8 output + v3 baseline and numpy-diff locally."""
import io
import os
import sys
import time

import numpy as np
import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
RDIR = "/opt/m0/trt-dev"
MARK = RDIR + "/mods/V8WS_DONE"
LDIR = (r"<REPO>"
        r"\work_dirs\sparsedrive_small_stage2\mini_eng")
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=60):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


deadline = time.time() + 25 * 60
while time.time() < deadline:
    if "V8WS_DONE" in run("ls " + MARK + " 2>/dev/null"):
        break
    time.sleep(30)
else:
    print("TIMEOUT; runlog tail:\n" + run("tail -20 " + RDIR + "/v8ws_run.log"))
    sys.exit(3)

print("== run log ==")
print(run("cat " + RDIR + "/v8ws_run.log"))

# fetch 1-frame outputs
sftp = cli.open_sftp()
os.makedirs(LDIR, exist_ok=True)
for d, names in (("outv8chk_00", ("det_cls", "det_bbox", "det_quality")),
                 ("out_00", ("det_cls", "det_bbox", "det_quality"))):
    try:
        fl = sftp.listdir(RDIR + "/vec/mini/" + d)
    except IOError:
        print("MISSING dir", d)
        continue
    for f in fl:
        if f.endswith(".bin"):
            sftp.get(RDIR + "/vec/mini/" + d + "/" + f,
                     os.path.join(LDIR, d + "_" + f))
sftp.close()
cli.close()

print("== local diff v8-1frame vs v3-1frame ==")
for name, shape in (("det_cls", (900, 10)), ("det_bbox", (900, 11)),
                    ("det_quality", (900, 2))):
    try:
        a = np.fromfile(os.path.join(LDIR, "outv8chk_00_" + name + ".bin"),
                        np.float32).reshape(shape)
        b = np.fromfile(os.path.join(LDIR, "out_00_" + name + ".bin"),
                        np.float32).reshape(shape)
    except (IOError, ValueError) as e:
        print(name, "skip:", e)
        continue
    d = np.abs(a - b)
    den = np.maximum(np.abs(b).max(), 1e-9)
    print(f"{name:12s} maxabs={d.max():.3e} rel={d.max()/den:.3e} "
          f"n_diff={(d > 0).sum()}/{a.size}")
print("POLL_DONE")
