# -*- coding: utf-8 -*-
"""P5-3: push dfaplug_v9.cu + sp_head_det_ssa.onnx; compile libdfaplug_v9.so;
build e_det_ssa; trtexec profile e_det_ssa + baseline e_det (same session);
2-chain dumps (e_bb+e_det / e_bb+e_det_ssa) for local A/B. Marker: SSA1_DONE."""
import os
import io
import sys

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
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
print("put sp_head_det_ssa.onnx",
      sftp_put(cli, W + r"\sp_head_det_ssa.onnx",
               "/opt/m0/trt-dev/models/sp_head_det_ssa.onnx"))

SH = r"""cat > /opt/m0/trt-dev/mods/run_ssa1.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
LOG=ssa1_run.log
PL9=/usr/local/lib/libdfaplug_v9.so
echo "=== compile libdfaplug_v9.so ===" > $LOG
nvcc -O3 -arch=sm_87 -shared -Xcompiler -fPIC -o $PL9 \
  src/dfaplug_v9.cu -lnvinfer > v9_build.log 2>&1
rc=$?
echo "v9 rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then tail -25 v9_build.log >> $LOG
  echo "SSA1_DONE" >> /opt/m0/trt-dev/mods/SSA1_DONE; exit 1; fi
ls -la $PL9 >> $LOG

echo "=== build e_det_ssa ===" >> $LOG
/usr/local/bin/onnx2engine2 models/sp_head_det_ssa.onnx \
  models/e_det_ssa.engine --fp16 --ws-mb 2048 --plugins $PL9 \
  > detssa_build.log 2>&1
rc=$?
echo "det_ssa rc=$rc" >> $LOG
ls -la models/e_det_ssa.engine 2>/dev/null >> $LOG
if [ $rc -ne 0 ]; then tail -25 detssa_build.log >> $LOG
  echo "SSA1_DONE" >> /opt/m0/trt-dev/mods/SSA1_DONE; exit 1; fi

echo "=== profile e_det_ssa ===" >> $LOG
/usr/src/tensorrt/bin/trtexec --loadEngine=models/e_det_ssa.engine \
  --staticPlugins=$PL9 --dumpProfile --iterations=100 \
  > e_detssa_prof.log 2>&1
grep ' Total' e_detssa_prof.log | tail -1 >> $LOG
echo "--- e_det_ssa ScaleSoftmax rows ---" >> $LOG
grep "ScaleSoftmax" e_detssa_prof.log | head -12 >> $LOG
echo "--- e_det_ssa top ForeignNode rows ---" >> $LOG
grep "ForeignNode" e_detssa_prof.log | sort -k2 -g -r | head -8 >> $LOG

echo "=== profile e_det (baseline, same session) ===" >> $LOG
/usr/src/tensorrt/bin/trtexec --loadEngine=models/e_det.engine \
  --staticPlugins=$PL9 --dumpProfile --iterations=100 \
  > e_det_prof2.log 2>&1
grep ' Total' e_det_prof2.log | tail -1 >> $LOG
echo "--- e_det top ForeignNode rows ---" >> $LOG
grep "ForeignNode" e_det_prof2.log | sort -k2 -g -r | head -10 >> $LOG

echo "=== 2-chain dump baseline (bb+det) ===" >> $LOG
rm -rf vec/mini/outs_detref_00
/usr/local/bin/run_engines models/e_bb.engine models/e_det.engine $PL9 \
  vec/mini/in_00 vec/mini/in_00 --dump vec/mini/outs_detref_00 \
  --iters 3 > one_detref.log 2>&1
rc=$?
echo "detref rc=$rc" >> $LOG
ls vec/mini/outs_detref_00 2>/dev/null | wc -l >> $LOG

echo "=== 2-chain dump ssa (bb+det_ssa) ===" >> $LOG
rm -rf vec/mini/outs_detssa_00
/usr/local/bin/run_engines models/e_bb.engine models/e_det_ssa.engine $PL9 \
  vec/mini/in_00 vec/mini/in_00 --dump vec/mini/outs_detssa_00 \
  --iters 3 > one_detssa.log 2>&1
rc=$?
echo "detssa rc=$rc" >> $LOG
ls vec/mini/outs_detssa_00 2>/dev/null | wc -l >> $LOG
echo "SSA1_DONE" >> /opt/m0/trt-dev/mods/SSA1_DONE
EOF"""
print(run(SH))
run("chmod +x /opt/m0/trt-dev/mods/run_ssa1.sh")
run("rm -f /opt/m0/trt-dev/mods/SSA1_DONE")
try:
    print(run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_ssa1.sh "
              "> ssa1.out 2>&1 < /dev/null & echo GO", t=8))
except Exception as e:
    print("read-timeout benign:", type(e).__name__)
cli.close()
print("ssa1 job launched")
