# -*- coding: utf-8 -*-
"""重编 v8 .so (-lnvinfer), 重跑 trtexec profile, 拉一帧输出回本地"""
import io
import os
import sys
import time

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_V8 = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2",
                        "mini_eng", "out_v8_00")
LOCAL_V3 = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2",
                        "mini_eng", "out_00")

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=600):
    _, out, err = cli.exec_command(cmd, timeout=t)
    o = out.read().decode("utf-8", "replace")
    e = err.read().decode("utf-8", "replace")
    return o + (("\n[stderr]\n" + e) if e.strip() else "")


print("== recompile v8 .so with -lnvinfer ==")
print(run("cd /opt/m0/trt-dev/src && /usr/local/cuda/bin/nvcc -O3 -arch=sm_87 "
          "-shared -Xcompiler -fPIC -o /usr/local/lib/libdfaplug_v8.so "
          "dfaplug_v8.cu -lnvinfer 2> nvcc8.err; echo NVCC_RC=$?; "
          "grep error nvcc8.err | head -3; "
          "ls -la /usr/local/lib/libdfaplug_v8.so"))

SH = r"""cat > /opt/m0/trt-dev/mods/run_v8prof.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=models/e_T6.engine \
  --plugins=/usr/local/lib/libdfaplug_v8.so \
  --dumpProfile --iterations=100 > v8_prof2.log 2>&1
echo "v8 prof2 rc=$?" > v8prof2_done.log
touch V8PROF2_DONE
EOF"""
print(run(SH))
run("rm -f /opt/m0/trt-dev/mods/V8PROF2_DONE")
run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_v8prof.sh > v8prof2.out "
    "2>&1 < /dev/null & echo GO", t=5)
print("launched profile (engine load takes minutes)")

print("== fetch 1-frame outputs (v8 + v3) while profile runs ==")
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
print("FETCH_STAGE_DONE")
