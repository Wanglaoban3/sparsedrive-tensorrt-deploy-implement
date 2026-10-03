# -*- coding: utf-8 -*-
"""wide module under small workspace: does tactic starvation reproduce the
in-engine 2.15ms zone?"""
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


SH = r"""cat > /opt/m0/trt-dev/mods/run_w3.sh <<'EOF'
#!/bin/bash
P=/usr/local/lib/libdfaplug_v3.so
T=/usr/src/tensorrt/bin/trtexec
cd /opt/m0/trt-dev/mods
echo "=== W3a ws=64MB" > w3_run.log
timeout 480 $T --onnx=modwide_p2h.onnx --fp16 --workspace=64 --plugins=$P \
  --dumpProfile --iterations=100 > w3a_prof.log 2>&1
echo "W3a rc=$?" >> w3_run.log
echo "=== W3b ws=256MB" >> w3_run.log
timeout 480 $T --onnx=modwide_p2h.onnx --fp16 --workspace=256 --plugins=$P \
  --dumpProfile --iterations=100 > w3b_prof.log 2>&1
echo "W3b rc=$?" >> w3_run.log
touch W3_DONE
EOF"""
print(run(SH))
run("rm -f /opt/m0/trt-dev/mods/W3_DONE /opt/m0/trt-dev/mods/w3_run.log")
run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_w3.sh > w3.out 2>&1 "
    "< /dev/null & echo GO", t=5)
print("launched W3a/W3b")
cli.close()
