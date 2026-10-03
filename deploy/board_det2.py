# -*- coding: utf-8 -*-
"""Build sp_head_det / sp_head_rest engines, profile each (module-level
test of the Myelin-fallback hypothesis). Marker: DET2_DONE."""
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
        (W + r"\sp_head_det.onnx", "/opt/m0/trt-dev/models/sp_head_det.onnx"),
        (W + r"\sp_head_rest.onnx",
         "/opt/m0/trt-dev/models/sp_head_rest.onnx"),
        (r"<REPO>"
         r"\deploy\run_engines.cpp",
         "/opt/m0/trt-dev/src/run_engines.cpp")):
    print("put", remote.split("/")[-1],
          sftp.put(local, remote) and "ok")
sftp.close()

SH = r"""cat > /opt/m0/trt-dev/mods/run_det2.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
LOG=det2_run.log
PL=/usr/local/lib/libdfaplug_v8.so
echo "=== build det ===" > $LOG
/usr/local/bin/onnx2engine2 models/sp_head_det.onnx models/e_det.engine \
  --fp16 --ws-mb 2048 --plugins $PL > det_build.log 2>&1
rc=$?
echo "det rc=$rc" >> $LOG
ls -la models/e_det.engine 2>/dev/null >> $LOG
if [ $rc -ne 0 ]; then tail -20 det_build.log >> $LOG
  echo "DET2_DONE" >> /opt/m0/trt-dev/mods/DET2_DONE; exit 1; fi
/usr/src/tensorrt/bin/trtexec --loadEngine=models/e_det.engine \
  --staticPlugins=$PL --dumpProfile --iterations=100 > e_det_prof.log 2>&1
grep ' Total' e_det_prof.log | tail -1 >> $LOG
echo "--- det ForeignNode rows ---" >> $LOG
grep "ForeignNode" e_det_prof.log | sort -k2 -g -r | head -8 >> $LOG

echo "=== build rest ===" >> $LOG
/usr/local/bin/onnx2engine2 models/sp_head_rest.onnx models/e_rest.engine \
  --fp16 --ws-mb 2048 --plugins $PL > rest_build.log 2>&1
rc=$?
echo "rest rc=$rc" >> $LOG
ls -la models/e_rest.engine 2>/dev/null >> $LOG
if [ $rc -ne 0 ]; then tail -20 rest_build.log >> $LOG
  echo "DET2_DONE" >> /opt/m0/trt-dev/mods/DET2_DONE; exit 1; fi
/usr/src/tensorrt/bin/trtexec --loadEngine=models/e_rest.engine \
  --staticPlugins=$PL --dumpProfile --iterations=100 > e_rest_prof.log 2>&1
grep ' Total' e_rest_prof.log | tail -1 >> $LOG
echo "--- rest ForeignNode rows ---" >> $LOG
grep "ForeignNode" e_rest_prof.log | sort -k2 -g -r | head -8 >> $LOG

echo "=== compile run_engines ===" >> $LOG
g++ -O2 -I/usr/local/cuda/include -o /usr/local/bin/run_engines \
  src/run_engines.cpp -lnvinfer -L/usr/local/cuda/lib64 -lcudart -ldl \
  > re_build.log 2>&1
rc=$?
echo "run_engines rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then tail -20 re_build.log >> $LOG
  echo "DET2_DONE" >> /opt/m0/trt-dev/mods/DET2_DONE; exit 1; fi
echo "DET2_DONE" >> /opt/m0/trt-dev/mods/DET2_DONE
EOF"""
print(run(SH))
run("chmod +x /opt/m0/trt-dev/mods/run_det2.sh")
run("rm -f /opt/m0/trt-dev/mods/DET2_DONE")
try:
    print(run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_det2.sh "
              "> det2.out 2>&1 < /dev/null & echo GO", t=8))
except Exception as e:
    print("read-timeout benign:", type(e).__name__)
cli.close()
print("det2 job launched (~8-12 min)")
