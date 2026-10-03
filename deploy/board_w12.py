# -*- coding: utf-8 -*-
"""wide-module experiment: W1 trtexec --fp16 (reproduce?), W2 patched tool
--f16-notq (rescue?)."""
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
                      "modwide_p2h.onnx"),
         "/opt/m0/trt-dev/mods/modwide_p2h.onnx")
sftp.close()

SH = r"""cat > /opt/m0/trt-dev/mods/run_w12.sh <<'EOF'
#!/bin/bash
P=/usr/local/lib/libdfaplug_v3.so
T=/usr/src/tensorrt/bin/trtexec
cd /opt/m0/trt-dev/mods
echo "=== W1 trtexec fp16" > w12_run.log
timeout 480 $T --onnx=modwide_p2h.onnx --fp16 --plugins=$P \
  --dumpProfile --iterations=100 > w1_prof.log 2>&1
echo "W1 rc=$?" >> w12_run.log
echo "=== W2 tool fp16 f16-notq" >> w12_run.log
timeout 480 /usr/local/bin/onnx2engine3 modwide_p2h.onnx w2.engine \
  --fp16 --ws-mb 2048 --f16-notq --plugins $P > w2_build.log 2>&1
echo "W2 build rc=$?" >> w12_run.log
if [ -f w2.engine ]; then
  $T --loadEngine=w2.engine --plugins=$P --dumpProfile --iterations=100 \
    > w2_prof.log 2>&1
  echo "W2 prof rc=$?" >> w12_run.log
fi
touch W12_DONE
EOF"""
print(run(SH))
run("rm -f /opt/m0/trt-dev/mods/W12_DONE /opt/m0/trt-dev/mods/w12_run.log "
    "/opt/m0/trt-dev/mods/w2.engine")
run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_w12.sh > w12.out 2>&1 "
    "< /dev/null & echo GO", t=5)
print("launched W1/W2")
cli.close()
