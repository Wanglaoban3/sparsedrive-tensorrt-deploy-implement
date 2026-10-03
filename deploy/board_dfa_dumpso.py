# -*- coding: utf-8 -*-
"""编译 libdfaplug_dump.so (v3+dump钩子), 配 e_T6 跑一帧抓真实 DFA IO, 拉回本地"""
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
                         "dfa_eT6")
RDIR = "/opt/m0/trt-dev/dfa_eT6"
SRC = "/opt/m0/trt-dev/src"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=600):
    _, out, err = cli.exec_command(cmd, timeout=t)
    o = out.read().decode("utf-8", "replace")
    e = err.read().decode("utf-8", "replace")
    return o + (("\n[stderr]\n" + e) if e.strip() else "")


print("== upload source ==")
sftp = cli.open_sftp()
sftp.put(os.path.join(DEP, "dfaplug_dump.cu"), f"{SRC}/dfaplug_dump.cu")
sftp.close()
print("ok")

print("== compile .so ==")
print(run(f"cd {SRC} && /usr/local/cuda/bin/nvcc -O3 -arch=sm_87 -shared "
          f"-Xcompiler -fPIC -o /usr/local/lib/libdfaplug_dump.so "
          f"dfaplug_dump.cu 2>&1 | tail -5; ls -la /usr/local/lib/"
          f"libdfaplug_dump.so"))

print("== run 1 frame with dump ==")
print(run(f"mkdir -p {RDIR} && rm -f {RDIR}/dfa* {RDIR}/manifest.txt"))
t0 = time.time()
r = run(f"cd /opt/m0/trt-dev && DFA_DUMP_DIR={RDIR} DFA_DUMP_MAX=12 "
        f"run_engine models/e_T6.engine /usr/local/lib/libdfaplug_dump.so "
        f"vec/mini/in_00 vec/mini/in_00 --dump {RDIR}/engine_out "
        f"> {RDIR}/run.log 2>&1; echo RC=$?; tail -3 {RDIR}/run.log")
print(r)
print("engine 1 frame: %.0fs" % (time.time() - t0))

print("== dumped files ==")
print(run(f"ls {RDIR} | head -30; cat {RDIR}/manifest.txt 2>/dev/null"))
print(run(f"cd {RDIR} && md5sum dfa*_feat.f16 2>/dev/null | sort | "
          f"awk '{{print $1}}' | uniq -c"))

print("== fetch ==")
os.makedirs(LOCAL_OUT, exist_ok=True)
sftp = cli.open_sftp()
t0 = time.time()
flist = sftp.listdir(RDIR)
n = 0
for fn in flist:
    if not fn.startswith("dfa") and fn != "manifest.txt":
        continue
    sftp.get(f"{RDIR}/{fn}", os.path.join(LOCAL_OUT, fn))
    n += 1
sftp.close()
print(f"fetched {n} files ({time.time()-t0:.0f}s) -> {LOCAL_OUT}")
cli.close()
