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


print(run("cp /opt/m0/trt-dev/bin/onnx2engine3 /usr/local/bin/onnx2engine3 "
          "&& chmod +x /usr/local/bin/onnx2engine3 && "
          "/usr/local/bin/onnx2engine3 2>&1 | head -2"))
SH = r"""cat > /opt/m0/trt-dev/mods/run_x12.sh <<'EOF'
#!/bin/bash
P=/usr/local/lib/libdfaplug_v3.so
T=/usr/src/tensorrt/bin/trtexec
cd /opt/m0/trt-dev/mods
echo "=== X1 unpatched tool" > x12_run.log
timeout 480 /usr/local/bin/onnx2engine2 modq_p2h.onnx x1.engine \
  --fp16 --int8 --ws-mb 2048 --plugins $P > x1_build.log 2>&1
echo "X1 build rc=$?" >> x12_run.log
if [ -f x1.engine ]; then
  $T --loadEngine=x1.engine --plugins=$P --dumpProfile --iterations=100 \
    > x1_prof.log 2>&1
  echo "X1 prof rc=$?" >> x12_run.log
fi
echo "=== X2 patched tool +f16-notq" >> x12_run.log
timeout 480 /usr/local/bin/onnx2engine3 modq_p2h.onnx x2.engine \
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
run("rm -f /opt/m0/trt-dev/mods/X12_DONE /opt/m0/trt-dev/mods/x1.engine "
    "/opt/m0/trt-dev/mods/x2.engine")
run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_x12.sh > x12.out 2>&1 "
    "< /dev/null & echo GO", t=5)
print("relaunched")
cli.close()
