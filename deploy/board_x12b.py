# -*- coding: utf-8 -*-
import os
import io
import sys

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=120):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


run("pkill -f run_x12.sh; pkill -f onnx2engine2; pkill -f onnx2engine3; "
    "pkill -f 'trtexec.*x[12].engine'; sleep 1")
SH = r"""cat > /opt/m0/trt-dev/mods/run_x12.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
echo "compile" > mods/x12_run.log
g++ -O2 -std=c++17 src/onnx2engine3.cpp -Iinclude \
  -I /usr/local/cuda/include \
  -L /usr/lib/aarch64-linux-gnu -L /usr/local/cuda/lib64 \
  -lnvinfer -lnvonnxparser -lcudart -ldl \
  -o bin/onnx2engine3 2>>mods/x12_run.log
echo "g++ rc=$?" >> mods/x12_run.log
[ -x bin/onnx2engine3 ] || { echo "compile failed" >> mods/x12_run.log; \
  touch mods/X12_DONE; exit 1; }
P=/usr/local/lib/libdfaplug_v3.so
T=/usr/src/tensorrt/bin/trtexec
cd mods
echo "=== X1 unpatched tool" >> x12_run.log
timeout 480 /usr/local/bin/onnx2engine2 modq_p2h.onnx x1.engine \
  --fp16 --int8 --ws-mb 2048 --plugins $P > x1_build.log 2>&1
echo "X1 build rc=$?" >> x12_run.log
if [ -f x1.engine ]; then
  $T --loadEngine=x1.engine --plugins=$P --dumpProfile --iterations=100 \
    > x1_prof.log 2>&1
  echo "X1 prof rc=$?" >> x12_run.log
fi
echo "=== X2 patched tool +f16-notq" >> x12_run.log
timeout 480 /opt/m0/trt-dev/bin/onnx2engine3 modq_p2h.onnx x2.engine \
  --fp16 --int8 --ws-mb 2048 --f16-notq --plugins $P > x2_build.log 2>&1
echo "X2 build rc=$?" >> x12_run.log
if [ -f x2.engine ]; then
  $T --loadEngine=x2.engine --plugins=$P --dumpProfile --iterations=100 \
    > x2_prof.log 2>&1
  echo "X2 prof rc=$?" >> x12_run.log
fi
touch X12_DONE
EOF"""
print(run(SH))
run("rm -f /opt/m0/trt-dev/mods/X12_DONE /opt/m0/trt-dev/mods/x12_run.log "
    "/opt/m0/trt-dev/mods/x1.engine /opt/m0/trt-dev/mods/x2.engine")
run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_x12.sh > x12.out 2>&1 "
    "< /dev/null & echo GO", t=5)
print("relaunched X1/X2")
cli.close()
