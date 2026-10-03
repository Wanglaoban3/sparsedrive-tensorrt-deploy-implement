# -*- coding: utf-8 -*-
"""poll run_split.sh: SPLITS_DONE marker -> report build/prof/frame status."""
import io
import os
import sys
import time

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
RDIR = "/opt/m0/trt-dev"
LDIR = (r"<REPO>"
        r"\work_dirs\sparsedrive_small_stage2")
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=120):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


t0 = time.time()
while time.time() - t0 < 100 * 60:
    if "SPLITS_DONE" in run(f"ls {RDIR}/mods/SPLITS_DONE 2>/dev/null"):
        break
    time.sleep(60)
else:
    print("TIMEOUT:\n" + run(f"tail -5 {RDIR}/split_run.log"))
    sys.exit(3)

print("== split_run.log ==")
print(run(f"cat {RDIR}/split_run.log"))
print("== hd profile rows (top by time) ==")
print(run("grep -E '\\[I\\] +[0-9]' " + RDIR +
          "/e_hd_prof.log | sort -k1 -n -r | head -25"))
print("== bb profile rows (top by time) ==")
print(run("grep -E '\\[I\\] +[0-9]' " + RDIR +
          "/e_bb_prof.log | sort -k1 -n -r | head -12"))
print("== one_frame tail ==")
print(run(f"tail -8 {RDIR}/one_frame.log"))

# fetch 1-frame outputs for local diff
sftp = cli.open_sftp()
try:
    fl = sftp.listdir(RDIR + "/vec/mini/outs_00")
    os.makedirs(os.path.join(LDIR, "evaldata", "mini_eng_v8"), exist_ok=True)
    for f in fl:
        if f.endswith(".bin"):
            sftp.get(RDIR + "/vec/mini/outs_00/" + f,
                     os.path.join(LDIR, "evaldata", "mini_eng_v8",
                                  "outs_00_" + f))
    print("fetched", sum(1 for f in fl if f.endswith(".bin")), "bins")
except IOError as e:
    print("fetch failed:", e)
sftp.close()
cli.close()
print("POLL_DONE")
