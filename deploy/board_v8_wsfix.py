# -*- coding: utf-8 -*-
"""push ws-fix dfaplug_v8.cu, rebuild .so (-lnvinfer), launch verification job:
trtexec 5-iter profile with DFA_V8_DEBUG=1 + 1-frame run_engine, then DONE."""
import os
import io
import sys

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
SRC = r"<REPO>\deploy\dfaplug_v8.cu"
RDIR = "/opt/m0/trt-dev"
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=120):
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


sftp = cli.open_sftp()
sftp.put(SRC, RDIR + "/src/dfaplug_v8.cu")
sftp.close()
print("uploaded dfaplug_v8.cu")

SH = r"""cat > /opt/m0/trt-dev/mods/run_v8ws.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
echo "=== compile ===" > v8ws_run.log
nvcc -O3 -arch=sm_87 -shared -Xcompiler -fPIC \
  -o /usr/local/lib/libdfaplug_v8.so src/dfaplug_v8.cu -lnvinfer \
  > v8ws_nvcc.log 2>&1
echo "nvcc rc=$?" >> v8ws_run.log
ls -la /usr/local/lib/libdfaplug_v8.so >> v8ws_run.log
if [ $? -ne 0 ]; then echo V8WS_DONE >> /opt/m0/trt-dev/mods/V8WS_DONE; exit 1; fi

echo "=== trtexec 5-iter profile (DFA_V8_DEBUG=1) ===" >> v8ws_run.log
rm -f /opt/m0/trt-dev/mods/V8WS_DONE
DFA_V8_DEBUG=1 /usr/src/tensorrt/bin/trtexec \
  --loadEngine=models/e_T6.engine \
  --plugins=/usr/local/lib/libdfaplug_v8.so \
  --dumpProfile --iterations=5 > v8_prof3.log 2>&1
echo "trtexec rc=$?" >> v8ws_run.log
grep -c 'dfa_v8' v8_prof3.log >> v8ws_run.log
grep 'dfa_v8' v8_prof3.log | head -14 >> v8ws_run.log
grep 'DeformableAggregation' v8_prof3.log | grep -v Reformat >> v8ws_run.log
grep ' Total' v8_prof3.log | tail -1 >> v8ws_run.log

echo "=== 1-frame run_engine (DFA_V8_DEBUG=1) ===" >> v8ws_run.log
rm -rf vec/mini/outv8chk_00
DFA_V8_DEBUG=1 run_engine models/e_T6.engine \
  /usr/local/lib/libdfaplug_v8.so vec/mini/in_00 vec/mini/in_00 \
  --dump vec/mini/outv8chk_00 > v8_1frame.log 2>&1
echo "run_engine rc=$?" >> v8ws_run.log
grep 'dfa_v8' v8_1frame.log | head -14 >> v8ws_run.log
echo V8WS_DONE >> /opt/m0/trt-dev/mods/V8WS_DONE
EOF"""
print(run(SH))
run("chmod +x /opt/m0/trt-dev/mods/run_v8ws.sh")
run("rm -f /opt/m0/trt-dev/mods/V8WS_DONE")
run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_v8ws.sh > v8ws.out "
    "2>&1 < /dev/null & echo GO", t=5)
print("launched run_v8ws.sh")
cli.close()
