# -*- coding: utf-8 -*-
"""push modq module + patched tool src, compile onnx2engine3, run X1/X2."""
import io
import os
import sys

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
L = r"<REPO>"
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=120):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


sftp = cli.open_sftp()
sftp.put(os.path.join(L, "work_dirs", "sparsedrive_small_stage2",
                      "modq_p2h.onnx"), "/opt/m0/trt-dev/mods/modq_p2h.onnx")
sftp.put(os.path.join(L, "deploy", "onnx2engine3.cpp"),
         "/opt/m0/trt-dev/src/onnx2engine3.cpp")
sftp.close()
print("pushed modq_p2h.onnx + onnx2engine3.cpp")

print(run("ls /usr/include/aarch64-linux-gnu/NvInfer.h "
          "/usr/include/NvInfer.h /usr/src/tensorrt/include/NvInfer.h "
          "/opt/m0/trt-dev/include/*.h 2>/dev/null"))

SH = r"""cat > /opt/m0/trt-dev/mods/run_x12.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
INC=""
for d in /usr/include/aarch64-linux-gnu /usr/include /usr/local/cuda/include \
         /usr/src/tensorrt/include include; do
  [ -f "$d/NvInfer.h" ] && INC="$INC -I$d"
done
echo "compile inc:$INC" > mods/x12_run.log
g++ -O2 -std=c++17 src/onnx2engine3.cpp $INC \
  -L /usr/lib/aarch64-linux-gnu -lnvinfer \
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
run("rm -f /opt/m0/trt-dev/mods/X12_DONE /opt/m0/trt-dev/mods/x12_run.log")
run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_x12.sh > x12.out 2>&1 "
    "< /dev/null & echo GO", t=5)
print("launched X1/X2")
cli.close()
