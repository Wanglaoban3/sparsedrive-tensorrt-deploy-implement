# -*- coding: utf-8 -*-
"""P5-3b: hot-swap libdfaplug_v9.so (CTA/row kernel), rebuild run_engines
(dump filename sanitize), re-profile e_det_ssa, re-dump both 2-chains.
Engine binary unchanged - plugin binds by name at deserialize.
Marker: SSA2_DONE."""
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


def sftp_put(cli, local, remote):
    s = cli.open_sftp()
    s.put(local, remote)
    s.close()
    return "ok"


print("put dfaplug_v9.cu",
      sftp_put(cli, D + r"\dfaplug_v9.cu", "/opt/m0/trt-dev/src/dfaplug_v9.cu"))
print("put run_engines.cpp",
      sftp_put(cli, D + r"\run_engines.cpp",
               "/opt/m0/trt-dev/src/run_engines.cpp"))

SH = r"""cat > /opt/m0/trt-dev/mods/run_ssa2.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
LOG=ssa2_run.log
PL9=/usr/local/lib/libdfaplug_v9.so
echo "=== rebuild v9 .so (CTA/row kernel) ===" > $LOG
nvcc -O3 -arch=sm_87 -shared -Xcompiler -fPIC -o $PL9 \
  src/dfaplug_v9.cu -lnvinfer > v9_build2.log 2>&1
rc=$?
echo "v9 rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then tail -25 v9_build2.log >> $LOG
  echo "SSA2_DONE" >> /opt/m0/trt-dev/mods/SSA2_DONE; exit 1; fi
ls -la $PL9 >> $LOG

echo "=== rebuild run_engines (dump name sanitize) ===" >> $LOG
g++ -O2 -I/usr/local/cuda/include -o /usr/local/bin/run_engines \
  src/run_engines.cpp -lnvinfer -L/usr/local/cuda/lib64 -lcudart -ldl \
  > re_build2.log 2>&1
rc=$?
echo "run_engines rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then tail -20 re_build2.log >> $LOG
  echo "SSA2_DONE" >> /opt/m0/trt-dev/mods/SSA2_DONE; exit 1; fi

echo "=== re-profile e_det_ssa (same engine, new .so) ===" >> $LOG
/usr/src/tensorrt/bin/trtexec --loadEngine=models/e_det_ssa.engine \
  --staticPlugins=$PL9 --dumpProfile --iterations=100 \
  > e_detssa_prof2.log 2>&1
grep ' Total' e_detssa_prof2.log | tail -1 >> $LOG
echo "--- ScaleSoftmax rows ---" >> $LOG
grep "ScaleSoftmax" e_detssa_prof2.log >> $LOG

echo "=== 2-chain dump baseline (bb+det) ===" >> $LOG
rm -rf vec/mini/outs_detref_00
/usr/local/bin/run_engines models/e_bb.engine models/e_det.engine $PL9 \
  vec/mini/in_00 vec/mini/in_00 --dump vec/mini/outs_detref_00 \
  --iters 3 > one_detref.log 2>&1
rc=$?
echo "detref rc=$rc files:" >> $LOG
ls vec/mini/outs_detref_00 | wc -l >> $LOG

echo "=== 2-chain dump ssa (bb+det_ssa) ===" >> $LOG
rm -rf vec/mini/outs_detssa_00
/usr/local/bin/run_engines models/e_bb.engine models/e_det_ssa.engine $PL9 \
  vec/mini/in_00 vec/mini/in_00 --dump vec/mini/outs_detssa_00 \
  --iters 3 > one_detssa.log 2>&1
rc=$?
echo "detssa rc=$rc files:" >> $LOG
ls vec/mini/outs_detssa_00 | wc -l >> $LOG
echo "SSA2_DONE" >> /opt/m0/trt-dev/mods/SSA2_DONE
EOF"""
print(run(SH))
run("chmod +x /opt/m0/trt-dev/mods/run_ssa2.sh")
run("rm -f /opt/m0/trt-dev/mods/SSA2_DONE")
try:
    print(run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_ssa2.sh "
              "> ssa2.out 2>&1 < /dev/null & echo GO", t=8))
except Exception as e:
    print("read-timeout benign:", type(e).__name__)
cli.close()
print("ssa2 job launched")
