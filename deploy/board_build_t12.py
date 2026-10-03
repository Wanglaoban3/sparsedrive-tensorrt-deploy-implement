# -*- coding: utf-8 -*-
"""push v5_P2h.onnx to board and launch e_T12 full build (background)."""
import io
import os
import sys
import time

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
L = r"<REPO>"
SRC = os.path.join(L, "work_dirs", "sparsedrive_small_stage2", "v5_P2h.onnx")

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=60):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


print(run("df -h /opt/m0 | tail -1"))
sz = os.path.getsize(SRC)
print(f"local v5_P2h.onnx: {sz/1e6:.1f} MB")
t0 = time.time()
sftp = cli.open_sftp()
sftp.put(SRC, "/opt/m0/trt-dev/models/v5_P2h.onnx")
sftp.close()
print(f"pushed in {time.time()-t0:.1f}s")
print(run("ls -la /opt/m0/trt-dev/models/v5_P2h.onnx"))
# kill any stale waiter, then launch build detached with DONE marker
print(run(
    "cd /opt/m0/trt-dev && rm -f repro/e_T12_build.log repro/E_T12_DONE && "
    "nohup bash -c '/usr/local/bin/onnx2engine2 models/v5_P2h.onnx "
    "models/e_T12.engine --int8 --fp16 --ws-mb 2048 "
    "--plugins /usr/local/lib/libdfaplug_v3.so "
    "> repro/e_T12_build.log 2>&1; echo rc=$? > repro/E_T12_DONE' "
    ">/dev/null 2>&1 & echo LAUNCHED", t=30))
cli.close()
