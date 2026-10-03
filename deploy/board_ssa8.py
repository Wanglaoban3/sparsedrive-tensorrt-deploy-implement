# -*- coding: utf-8 -*-
"""SSA8: 竞态定位 (合成 + 真实输入各一遍). Marker: SSA8_DONE."""
import os
import io
import sys

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
D = r"<REPO>\deploy"
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=120):
    _, out, err = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace") + \
        err.read().decode("utf-8", "replace")


s = cli.open_sftp()
s.put(D + r"\selftest_fa4.cu", "/opt/m0/trt-dev/src/selftest_fa4.cu")
s.close()
print("put selftest_fa4 ok")

SH = r"""cat > /opt/m0/trt-dev/mods/run_ssa8.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
LOG=ssa8_run.log
echo "=== build selftest_fa4 ===" > $LOG
cd src && nvcc -O3 -arch=sm_87 -o /usr/local/bin/selftest_fa4 \
  selftest_fa4.cu -lnvinfer > ../st4_build.log 2>&1
rc=$?
cd ..
echo "build rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then tail -25 st4_build.log >> $LOG
  echo "SSA8_DONE" >> /opt/m0/trt-dev/mods/SSA8_DONE; exit 1; fi
echo "--- synthetic ---" >> $LOG
/usr/local/bin/selftest_fa4 >> $LOG 2>&1
echo "--- real probe inputs ---" >> $LOG
PD=vec/mini/outs_probe_fa
/usr/local/bin/selftest_fa4 \
  $PD/_layers.19_attn_inner_attn_Cast_1_output_0.bin \
  $PD/_layers.19_attn_inner_attn_Transpose_3_output_0.bin \
  $PD/_layers.19_attn_inner_attn_Cast_3_output_0.bin >> $LOG 2>&1
echo "SSA8_DONE" >> /opt/m0/trt-dev/mods/SSA8_DONE
EOF"""
print(run(SH))
run("chmod +x /opt/m0/trt-dev/mods/run_ssa8.sh")
run("rm -f /opt/m0/trt-dev/mods/SSA8_DONE")
try:
    print(run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_ssa8.sh "
              "> ssa8.out 2>&1 < /dev/null & echo GO", t=8))
except Exception as e:
    print("read-timeout benign:", type(e).__name__)
cli.close()
print("ssa8 job launched")
