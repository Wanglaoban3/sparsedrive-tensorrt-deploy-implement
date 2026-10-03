# -*- coding: utf-8 -*-
"""push v8 sources + real dump to board, compile test_dfa_ab, run v3-vs-v8 A/B"""
import io
import os
import sys
import time

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEP = os.path.join(ROOT, "deploy")
DUMP = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2",
                    "dfa_real_engine")
RDIR = "/opt/m0/trt-dev/dfa_eT6"
SRC = "/opt/m0/trt-dev/src"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=900):
    _, out, err = cli.exec_command(cmd, timeout=t)
    o = out.read().decode("utf-8", "replace")
    e = err.read().decode("utf-8", "replace")
    return o + (("\n[stderr]\n" + e) if e.strip() else "")


print("== mkdir ==")
print(run(f"mkdir -p {RDIR} {SRC}"))

print("== upload sources ==")
sftp = cli.open_sftp()
for fn in ["dfaplug_v3.cu", "dfaplug_v8.cu", "test_dfa_ab.cu"]:
    t0 = time.time()
    sftp.put(os.path.join(DEP, fn), f"{SRC}/{fn}")
    print(f"{fn} ok ({time.time()-t0:.1f}s)")

print("== dump data: 用板上 dfa_eT6 (libdfaplug_dump.so 抓的真实引擎 IO) ==")
print(run(f"ls {RDIR}/manifest.txt && head -2 {RDIR}/manifest.txt"))
sftp.close()

print("== compile ==")
t0 = time.time()
print(run(f"cd {SRC} && /usr/local/cuda/bin/nvcc -O3 -arch=sm_87 "
          f"-o /usr/local/bin/test_dfa_ab test_dfa_ab.cu 2> {SRC}/nvcc.err; "
          f"echo NVCC_RC=$?; grep -E 'error' {SRC}/nvcc.err | head -10"))
print("compile %.0fs" % (time.time() - t0))

print("== run A/B ==")
t0 = time.time()
print(run(f"/usr/local/bin/test_dfa_ab {RDIR} 50 10 2>&1; echo RUN_RC=$?",
          t=600))
print("run %.0fs" % (time.time() - t0))
cli.close()
