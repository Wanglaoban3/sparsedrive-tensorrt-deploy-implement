# -*- coding: utf-8 -*-
"""Build cast-stripped e_det2 + e_rest, profile, compile run_engines,
1-frame 3-chain + timing. Marker: DET3_DONE."""
import os
import io
import sys

import paramiko

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2")
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(os.environ["BOARD_HOST"], username="root", password=os.environ["BOARD_PASS"], timeout=15)


def run(cmd, t=120):
    _, out, err = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace") + \
        err.read().decode("utf-8", "replace")


sftp = cli.open_sftp()
for local, remote in (
        (W + r"\sp_head_det.onnx",
         "/opt/m0/trt-dev/models/sp_head_det.onnx"),
        (W + r"\sp_head_det2.onnx",
         "/opt/m0/trt-dev/models/sp_head_det2.onnx"),
        (W + r"\sp_head_rest.onnx",
         "/opt/m0/trt-dev/models/sp_head_rest.onnx")):
    print("put", remote.split("/")[-1],
          sftp.put(local, remote) and "ok")
sftp.close()

SH = r"""cat > /opt/m0/trt-dev/mods/run_det3.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
LOG=det3_run.log
PL=/usr/local/lib/libdfaplug_v8.so
echo "=== build det2 (stripped casts) ===" > $LOG
/usr/local/bin/onnx2engine2 models/sp_head_det2.onnx models/e_det2.engine \
  --fp16 --ws-mb 2048 --plugins $PL > det2_build.log 2>&1
rc=$?
echo "det2 rc=$rc" >> $LOG
ls -la models/e_det2.engine 2>/dev/null >> $LOG
if [ $rc -ne 0 ]; then tail -15 det2_build.log >> $LOG
  echo "DET3_DONE" >> /opt/m0/trt-dev/mods/DET3_DONE; exit 1; fi
/usr/src/tensorrt/bin/trtexec --loadEngine=models/e_det2.engine \
  --staticPlugins=$PL --dumpProfile --iterations=100 > e_det2_prof.log 2>&1
grep ' Total' e_det2_prof.log | tail -1 >> $LOG
echo "--- det2 ForeignNode rows ---" >> $LOG
grep "ForeignNode" e_det2_prof.log | sort -k2 -g -r | head -7 >> $LOG

echo "=== build rest ===" >> $LOG
/usr/local/bin/onnx2engine2 models/sp_head_rest.onnx models/e_rest.engine \
  --fp16 --ws-mb 2048 --plugins $PL > rest_build.log 2>&1
rc=$?
echo "rest rc=$rc" >> $LOG
ls -la models/e_rest.engine 2>/dev/null >> $LOG
if [ $rc -ne 0 ]; then tail -15 rest_build.log >> $LOG
  echo "DET3_DONE" >> /opt/m0/trt-dev/mods/DET3_DONE; exit 1; fi
/usr/src/tensorrt/bin/trtexec --loadEngine=models/e_rest.engine \
  --staticPlugins=$PL --dumpProfile --iterations=100 > e_rest_prof.log 2>&1
grep ' Total' e_rest_prof.log | tail -1 >> $LOG
echo "--- rest ForeignNode rows ---" >> $LOG
grep "ForeignNode" e_rest_prof.log | sort -k2 -g -r | head -7 >> $LOG

echo "=== compile run_engines ===" >> $LOG
g++ -O2 -I/usr/local/cuda/include -o /usr/local/bin/run_engines \
  src/run_engines.cpp -lnvinfer -L/usr/local/cuda/lib64 -lcudart -ldl \
  > re_build.log 2>&1
rc=$?
echo "run_engines rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then tail -15 re_build.log >> $LOG
  echo "DET3_DONE" >> /opt/m0/trt-dev/mods/DET3_DONE; exit 1; fi

echo "=== 3-chain 1-frame ===" >> $LOG
rm -rf vec/mini/outs3_00
/usr/local/bin/run_engines models/e_bb.engine models/e_det2.engine \
  models/e_rest.engine $PL vec/mini/in_00 vec/mini/in_00 \
  --dump vec/mini/outs3_00 --iters 5 > one3.log 2>&1
rc=$?
echo "3chain rc=$rc" >> $LOG
tail -3 one3.log >> $LOG
echo "DET3_DONE" >> /opt/m0/trt-dev/mods/DET3_DONE
EOF"""
print(run(SH))
run("chmod +x /opt/m0/trt-dev/mods/run_det3.sh")
run("rm -f /opt/m0/trt-dev/mods/DET3_DONE")
try:
    print(run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_det3.sh "
              "> det3.out 2>&1 < /dev/null & echo GO", t=8))
except Exception as e:
    print("read-timeout benign:", type(e).__name__)
cli.close()
print("det3 job launched (~10 min)")
