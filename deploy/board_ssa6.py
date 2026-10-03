# -*- coding: utf-8 -*-
"""SSA6: 分级内核调试 (micro wmma + S/P 导出 synth) + 真实数据重放.
Marker: SSA6_DONE."""
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
s.put(D + r"\selftest_fa2.cu", "/opt/m0/trt-dev/src/selftest_fa2.cu")
s.close()
print("put selftest_fa2 ok")

SH = r"""cat > /opt/m0/trt-dev/mods/run_ssa6.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
LOG=ssa6_run.log
echo "=== build selftest_fa2 ===" > $LOG
cd src && nvcc -O3 -arch=sm_87 -o /usr/local/bin/selftest_fa2 \
  selftest_fa2.cu -lnvinfer > ../st2_build.log 2>&1
rc=$?
cd ..
echo "build rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then tail -25 st2_build.log >> $LOG
  echo "SSA6_DONE" >> /opt/m0/trt-dev/mods/SSA6_DONE; exit 1; fi
/usr/local/bin/selftest_fa2 >> $LOG 2>&1

echo "=== replay real probe data ===" >> $LOG
PD=vec/mini/outs_probe_fa
mkdir -p vec/mini/outs_replay
QB=$PD/_layers.19_attn_inner_attn_Cast_1_output_0.bin
KB=$PD/_layers.19_attn_inner_attn_Transpose_3_output_0.bin
VB=$PD/_layers.19_attn_inner_attn_Cast_3_output_0.bin
ls -la $QB $KB $VB >> $LOG
/usr/local/bin/selftest_fa2 replay $QB $KB $VB vec/mini/outs_replay \
  >> $LOG 2>&1
echo "replay rc=$?" >> $LOG
ls -la vec/mini/outs_replay >> $LOG
echo "SSA6_DONE" >> /opt/m0/trt-dev/mods/SSA6_DONE
EOF"""
print(run(SH))
run("chmod +x /opt/m0/trt-dev/mods/run_ssa6.sh")
run("rm -f /opt/m0/trt-dev/mods/SSA6_DONE")
try:
    print(run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_ssa6.sh "
              "> ssa6.out 2>&1 < /dev/null & echo GO", t=8))
except Exception as e:
    print("read-timeout benign:", type(e).__name__)
cli.close()
print("ssa6 job launched")
