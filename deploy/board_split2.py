# -*- coding: utf-8 -*-
"""Upload fixed run_engine2.cpp, recompile, 1-frame dump + 100-iter timing."""
import os
import io
import sys

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)
sftp = cli.open_sftp()

local = r"<REPO>\deploy\run_engine2.cpp"
data = open(local, "rb").read().replace(b"\r\n", b"\n")
with sftp.open("/opt/m0/trt-dev/src/run_engine2.cpp", "wb") as f:
    f.write(data)
print("uploaded run_engine2.cpp", len(data), "bytes")

script = r"""#!/bin/bash
cd /opt/m0/trt-dev
{
echo "=== compile runner ==="
g++ -O2 -I/usr/local/cuda/include -o /usr/local/bin/run_engine2 src/run_engine2.cpp -lnvinfer -L/usr/local/cuda/lib64 -lcudart -ldl 2> runner_build.log
rc=$?
echo "runner rc=$rc"
if [ $rc -ne 0 ]; then tail -30 runner_build.log; echo "SPLITS2_DONE" >> /opt/m0/trt-dev/mods/SPLITS2_DONE; exit 1; fi
echo "=== 1-frame dump ==="
rm -rf vec/mini/outs_00
/usr/local/bin/run_engine2 models/e_bb.engine models/e_hd.engine /opt/m0/trt-dev/libdfaplug_v8.so vec/mini/in_00 vec/mini/in_00 --dump vec/mini/outs_00 --iters 5
echo "1frame rc=$?"
echo "=== 100-iter timing ==="
/usr/local/bin/run_engine2 models/e_bb.engine models/e_hd.engine /opt/m0/trt-dev/libdfaplug_v8.so vec/mini/in_00 vec/mini/in_00 --iters 100 --warmup 10 2>&1 | grep -E "MEAN|IO|manifest|bb |hd "
echo "SPLITS2_DONE" >> /opt/m0/trt-dev/mods/SPLITS2_DONE
} > split2_run.log 2>&1
"""
data = script.replace("\r\n", "\n").encode("utf-8")
with sftp.open("/opt/m0/trt-dev/mods/run_split2.sh", "wb") as f:
    f.write(data)
sftp.close()
_, out, _ = cli.exec_command(
    "cd /opt/m0/trt-dev/mods && rm -f /opt/m0/trt-dev/mods/SPLITS2_DONE && "
    "setsid nohup bash run_split2.sh > /dev/null 2>&1 < /dev/null & echo LAUNCHED",
    timeout=20)
print(out.read().decode())
cli.close()
