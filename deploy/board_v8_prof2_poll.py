# -*- coding: utf-8 -*-
"""fetch 1-frame outputs + local numpy diff; then poll profile"""
import io
import os
import sys
import time

import numpy as np
import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2", "mini_eng")
LOCAL_V8 = os.path.join(BASE, "out_v8_00")
LOCAL_V3 = os.path.join(BASE, "out_00")

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=600):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


print("== launch poll (profile may still be running) ==")
if "YES" not in run("ls /opt/m0/trt-dev/mods/V8PROF2_DONE 2>/dev/null && echo YES"):
    print(run("ls /opt/m0/trt-dev/mods/run_v8prof.sh 2>/dev/null || echo NO_SCRIPT"))
    # 若脚本没启动 (上次 PipeTimeout 前是否执行不确定), 重新启动一次
    if "NO_SCRIPT" in run("ls /opt/m0/trt-dev/mods/run_v8prof.sh 2>/dev/null || echo NO_SCRIPT"):
        print("script missing!")
    run("cd /opt/m0/trt-dev/mods && (setsid nohup bash run_v8prof.sh > "
        "v8prof2.out 2>&1 < /dev/null &) ; echo RELAUNCHED", t=10)

t0 = time.time()
while time.time() - t0 < 900:
    time.sleep(15)
    if "YES" in run("ls /opt/m0/trt-dev/mods/V8PROF2_DONE 2>/dev/null && echo YES"):
        print("profile done in %.0fs" % (time.time() - t0))
        break
else:
    print("TIMEOUT")
print(run("cat /opt/m0/trt-dev/v8prof2_done.log 2>/dev/null"))
print("=== v8 Deformable rows ===")
print(run("grep 'Deformable' /opt/m0/trt-dev/v8_prof2.log 2>/dev/null"))

print("== fetch outputs ==")
sftp = cli.open_sftp()
os.makedirs(LOCAL_V8, exist_ok=True)
os.makedirs(LOCAL_V3, exist_ok=True)
for rdir, ldir in [("/opt/m0/trt-dev/vec/mini/out_v8_00", LOCAL_V8),
                   ("/opt/m0/trt-dev/vec/mini/out_00", LOCAL_V3)]:
    n = 0
    try:
        for fn in sftp.listdir(rdir):
            sftp.get(f"{rdir}/{fn}", os.path.join(ldir, fn))
            n += 1
        print(rdir, "fetched", n)
    except Exception as e:
        print(rdir, "fetch failed:", e)
sftp.close()
cli.close()

print("== local diff ==")
worst = []
for fn in sorted(os.listdir(LOCAL_V3)):
    pa, pb = os.path.join(LOCAL_V3, fn), os.path.join(LOCAL_V8, fn)
    if not os.path.exists(pb):
        continue
    x = np.fromfile(pa, np.float32)
    y = np.fromfile(pb, np.float32)
    if x.size != y.size:
        print(fn, "SIZE MISMATCH")
        continue
    d = np.abs(x - y)
    worst.append((d.max() / max(1e-6, np.abs(x).max()), fn, d.max(),
                  np.abs(x).max()))
worst.sort(reverse=True)
for rel, fn, mx, xa in worst[:12]:
    print(f"{fn:32s} maxabs={mx:.6f} |x|max={xa:.4f} rel={rel:.2e}")
print("DIFF_DONE")
