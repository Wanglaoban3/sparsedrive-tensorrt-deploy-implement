# -*- coding: utf-8 -*-
"""3-chain with unstripped e_det + 100-iter. Marker: DET5_DONE."""
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
    _, out, err = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace") + \
        err.read().decode("utf-8", "replace")


SH = r"""cat > /opt/m0/trt-dev/mods/run_det5.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
LOG=det5_run.log
PL=/usr/local/lib/libdfaplug_v8.so
echo "=== 3-chain 1-frame (e_det unstripped) ===" > $LOG
rm -rf vec/mini/outs3_00
/usr/local/bin/run_engines models/e_bb.engine models/e_det.engine \
  models/e_rest.engine $PL vec/mini/in_00 vec/mini/in_00 \
  --dump vec/mini/outs3_00 --iters 5 > one3.log 2>&1
rc=$?
echo "3chain rc=$rc" >> $LOG
tail -4 one3.log >> $LOG
echo "=== 100-iter ===" >> $LOG
/usr/local/bin/run_engines models/e_bb.engine models/e_det.engine \
  models/e_rest.engine $PL vec/mini/in_00 vec/mini/in_00 \
  --iters 100 --warmup 10 > tim3.log 2>&1
grep MEAN tim3.log >> $LOG
echo "DET5_DONE" >> /opt/m0/trt-dev/mods/DET5_DONE
EOF"""
print(run(SH))
run("chmod +x /opt/m0/trt-dev/mods/run_det5.sh")
run("rm -f /opt/m0/trt-dev/mods/DET5_DONE")
try:
    print(run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_det5.sh "
              "> det5.out 2>&1 < /dev/null & echo GO", t=8))
except Exception as e:
    print("read-timeout benign:", type(e).__name__)
cli.close()
print("det5 job launched")
