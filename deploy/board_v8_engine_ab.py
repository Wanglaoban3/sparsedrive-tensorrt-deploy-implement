# -*- coding: utf-8 -*-
"""P3-3: 引擎内 A/B —— 编 libdfaplug_v8.so, 不换引擎 (e_T6) 只换 .so:
   trtexec --loadEngine e_T6 + --plugins v8.so → profile;
   run_engine 一帧 → 与 v3 的 out_00 对拍输出误差。"""
import io
import os
import sys
import time

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEP = os.path.join(ROOT, "deploy")
LOCAL_OUT = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2",
                         "mini_eng", "out_v8_00")

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=900):
    _, out, err = cli.exec_command(cmd, timeout=t)
    o = out.read().decode("utf-8", "replace")
    e = err.read().decode("utf-8", "replace")
    return o + (("\n[stderr]\n" + e) if e.strip() else "")


print("== upload v8 source ==")
sftp = cli.open_sftp()
sftp.put(os.path.join(DEP, "dfaplug_v8.cu"), "/opt/m0/trt-dev/src/dfaplug_v8.cu")
sftp.close()
print("ok")

print("== compile libdfaplug_v8.so ==")
t0 = time.time()
print(run("cd /opt/m0/trt-dev/src && /usr/local/cuda/bin/nvcc -O3 -arch=sm_87 "
          "-shared -Xcompiler -fPIC -o /usr/local/lib/libdfaplug_v8.so "
          "dfaplug_v8.cu 2> nvcc8.err; echo NVCC_RC=$?; "
          "grep -E 'error' nvcc8.err | head; "
          "ls -la /usr/local/lib/libdfaplug_v8.so"))
print("compile %.0fs" % (time.time() - t0))

SH = r"""cat > /opt/m0/trt-dev/mods/run_v8ab.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
T=/usr/src/tensorrt/bin/trtexec
echo "=== v3 baseline" > v8ab.log
$T --loadEngine=models/e_T6.engine --plugins=/usr/local/lib/libdfaplug_v3.so \
   --dumpProfile --iterations=100 > v8ab_v3_prof.log 2>&1
echo "v3 prof rc=$?" >> v8ab.log
echo "=== v8" >> v8ab.log
$T --loadEngine=models/e_T6.engine --plugins=/usr/local/lib/libdfaplug_v8.so \
   --dumpProfile --iterations=100 > v8ab_v8_prof.log 2>&1
echo "v8 prof rc=$?" >> v8ab.log
echo "=== 1-frame accuracy diff" >> v8ab.log
rm -rf vec/mini/out_v8_00
run_engine models/e_T6.engine /usr/local/lib/libdfaplug_v8.so \
   vec/mini/in_00 vec/mini/in_00 --dump vec/mini/out_v8_00 \
   > v8ab_run.log 2>&1
echo "run rc=$?" >> v8ab.log
python3 - <<'PYEOF'
import os, struct
import numpy as np
a, b = "vec/mini/out_00", "vec/mini/out_v8_00"
worst = []
for fn in sorted(os.listdir(a)):
    pa, pb = os.path.join(a, fn), os.path.join(b, fn)
    if not os.path.exists(pb):
        continue
    x = np.fromfile(pa, np.float32)
    y = np.fromfile(pb, np.float32)
    if x.size != y.size:
        print(fn, "SIZE MISMATCH", x.size, y.size)
        continue
    d = np.abs(x - y)
    rel = d.max() / max(1e-6, np.abs(x).max())
    worst.append((rel, fn, d.max(), np.abs(x).max()))
worst.sort(reverse=True)
for rel, fn, mx, xa in worst[:12]:
    print(f"{fn:32s} maxabs={mx:.6f} |x|max={xa:.4f} rel={rel:.2e}")
print("DIFF_DONE")
PYEOF
touch V8AB_DONE
EOF"""
print(run(SH))
run("rm -f /opt/m0/trt-dev/mods/V8AB_DONE")
r = run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_v8ab.sh > v8ab.out "
        "2>&1 < /dev/null & echo GO", t=5)
print("launched:", r.strip())

print("== poll V8AB_DONE ==")
t0 = time.time()
while time.time() - t0 < 1500:
    time.sleep(20)
    r = run("ls /opt/m0/trt-dev/mods/V8AB_DONE 2>/dev/null && echo YES")
    if "YES" in r:
        print("done in %.0fs" % (time.time() - t0))
        break
else:
    print("TIMEOUT waiting")

print(run("cat /opt/m0/trt-dev/mods/v8ab.log"))
print(run("grep -A60 '=== Profile' /opt/m0/trt-dev/mods/v8ab_v3_prof.log | "
          "grep -E 'Time\\(ms\\)|Deformable|Plugin' | head -20"))
print(run("grep -A60 '=== Profile' /opt/m0/trt-dev/mods/v8ab_v8_prof.log | "
          "grep -E 'Time\\(ms\\)|Deformable|Plugin' | head -20"))

print("== fetch v8 1-frame outputs ==")
sftp = cli.open_sftp()
os.makedirs(LOCAL_OUT, exist_ok=True)
t0 = time.time()
try:
    for fn in sftp.listdir("/opt/m0/trt-dev/vec/mini/out_v8_00"):
        sftp.get(f"/opt/m0/trt-dev/vec/mini/out_v8_00/{fn}",
                 os.path.join(LOCAL_OUT, fn))
    print("fetched", time.time() - t0)
except Exception as e:
    print("fetch failed:", e)
sftp.close()
cli.close()
