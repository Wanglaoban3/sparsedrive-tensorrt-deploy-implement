# -*- coding: utf-8 -*-
"""Compile run_engines, 3-chain 1-frame + 100-iter timing. Marker: DET4_DONE."""
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


sftp = cli.open_sftp()
print("put run_engines.cpp",
      sftp.put(r"<REPO>"
               r"\deploy\run_engines.cpp",
               "/opt/m0/trt-dev/src/run_engines.cpp") and "ok")
sftp.close()

SH = r"""cat > /opt/m0/trt-dev/mods/run_det4.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
LOG=det4_run.log
PL=/usr/local/lib/libdfaplug_v8.so
echo "=== compile run_engines ===" > $LOG
g++ -O2 -I/usr/local/cuda/include -o /usr/local/bin/run_engines \
  src/run_engines.cpp -lnvinfer -L/usr/local/cuda/lib64 -lcudart -ldl \
  > re_build.log 2>&1
rc=$?
echo "rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then tail -15 re_build.log >> $LOG
  echo "DET4_DONE" >> /opt/m0/trt-dev/mods/DET4_DONE; exit 1; fi
echo "=== 3-chain 1-frame ===" >> $LOG
rm -rf vec/mini/outs3_00
/usr/local/bin/run_engines models/e_bb.engine models/e_det2.engine \
  models/e_rest.engine $PL vec/mini/in_00 vec/mini/in_00 \
  --dump vec/mini/outs3_00 --iters 5 > one3.log 2>&1
rc=$?
echo "3chain rc=$rc" >> $LOG
tail -4 one3.log >> $LOG
echo "=== 100-iter ===" >> $LOG
/usr/local/bin/run_engines models/e_bb.engine models/e_det2.engine \
  models/e_rest.engine $PL vec/mini/in_00 vec/mini/in_00 \
  --iters 100 --warmup 10 > tim3.log 2>&1
grep MEAN tim3.log >> $LOG
echo "DET4_DONE" >> /opt/m0/trt-dev/mods/DET4_DONE
EOF"""
print(run(SH))
run("chmod +x /opt/m0/trt-dev/mods/run_det4.sh")
run("rm -f /opt/m0/trt-dev/mods/DET4_DONE")
try:
    print(run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_det4.sh "
              "> det4.out 2>&1 < /dev/null & echo GO", t=8))
except Exception as e:
    print("read-timeout benign:", type(e).__name__)
cli.close()
print("det4 job launched")
