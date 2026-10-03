# -*- coding: utf-8 -*-
"""P5b: push dfaplug_v10.cu (v8+ScaleSoftmax+FlashSDPA) + sp_head_det_fa.onnx;
compile libdfaplug_v10.so; build e_det_fa; profile; 2-chain dump.
Marker: SSA3_DONE."""
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


print("put dfaplug_v10.cu",
      sftp_put(cli, D + r"\dfaplug_v10.cu",
               "/opt/m0/trt-dev/src/dfaplug_v10.cu"))
print("put sp_head_det_fa.onnx",
      sftp_put(cli, W + r"\sp_head_det_fa.onnx",
               "/opt/m0/trt-dev/models/sp_head_det_fa.onnx"))

SH = r"""cat > /opt/m0/trt-dev/mods/run_ssa3.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
LOG=ssa3_run.log
PL0=/usr/local/lib/libdfaplug_v10.so
echo "=== compile libdfaplug_v10.so ===" > $LOG
nvcc -O3 -arch=sm_87 -shared -Xcompiler -fPIC -o $PL0 \
  src/dfaplug_v10.cu -lnvinfer > v10_build.log 2>&1
rc=$?
echo "v10 rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then tail -30 v10_build.log >> $LOG
  echo "SSA3_DONE" >> /opt/m0/trt-dev/mods/SSA3_DONE; exit 1; fi
ls -la $PL0 >> $LOG

echo "=== build e_det_fa ===" >> $LOG
/usr/local/bin/onnx2engine2 models/sp_head_det_fa.onnx \
  models/e_det_fa.engine --fp16 --ws-mb 2048 --plugins $PL0 \
  > detfa_build.log 2>&1
rc=$?
echo "det_fa rc=$rc" >> $LOG
ls -la models/e_det_fa.engine 2>/dev/null >> $LOG
if [ $rc -ne 0 ]; then tail -25 detfa_build.log >> $LOG
  echo "SSA3_DONE" >> /opt/m0/trt-dev/mods/SSA3_DONE; exit 1; fi

echo "=== profile e_det_fa ===" >> $LOG
/usr/src/tensorrt/bin/trtexec --loadEngine=models/e_det_fa.engine \
  --staticPlugins=$PL0 --dumpProfile --iterations=100 \
  > e_detfa_prof.log 2>&1
grep ' Total' e_detfa_prof.log | tail -1 >> $LOG
echo "--- FlashSDPA rows ---" >> $LOG
grep "FlashSDPA" e_detfa_prof.log >> $LOG
echo "--- top ForeignNode rows ---" >> $LOG
grep "ForeignNode" e_detfa_prof.log | sort -k2 -g -r | head -8 >> $LOG

echo "=== 2-chain dump fa (bb+det_fa) ===" >> $LOG
rm -rf vec/mini/outs_detfa_00
/usr/local/bin/run_engines models/e_bb.engine models/e_det_fa.engine $PL0 \
  vec/mini/in_00 vec/mini/in_00 --dump vec/mini/outs_detfa_00 \
  --iters 3 > one_detfa.log 2>&1
rc=$?
echo "detfa rc=$rc files:" >> $LOG
ls vec/mini/outs_detfa_00 | wc -l >> $LOG
echo "=== 100-iter 2-chain timing (bb+det_fa) ===" >> $LOG
/usr/local/bin/run_engines models/e_bb.engine models/e_det_fa.engine $PL0 \
  vec/mini/in_00 vec/mini/in_00 --iters 100 --warmup 10 > tim_fa.log 2>&1
grep MEAN tim_fa.log >> $LOG
echo "=== 100-iter 2-chain timing baseline (bb+det) ===" >> $LOG
/usr/local/bin/run_engines models/e_bb.engine models/e_det.engine \
  /usr/local/lib/libdfaplug_v8.so \
  vec/mini/in_00 vec/mini/in_00 --iters 100 --warmup 10 > tim_ref.log 2>&1
grep MEAN tim_ref.log >> $LOG
echo "SSA3_DONE" >> /opt/m0/trt-dev/mods/SSA3_DONE
EOF"""
print(run(SH))
run("chmod +x /opt/m0/trt-dev/mods/run_ssa3.sh")
run("rm -f /opt/m0/trt-dev/mods/SSA3_DONE")
try:
    print(run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_ssa3.sh "
              "> ssa3.out 2>&1 < /dev/null & echo GO", t=8))
except Exception as e:
    print("read-timeout benign:", type(e).__name__)
cli.close()
print("ssa3 job launched")
