# -*- coding: utf-8 -*-
"""P4 redo with ego-in-head split: upload onnx + cpp, build both engines,
profile, compile runner, 1-frame dump, 100-iter timing. Marker: SPLIT3_DONE."""
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
        (W + r"\sp_backbone.onnx", "/opt/m0/trt-dev/models/sp_backbone.onnx"),
        (W + r"\sp_head.onnx", "/opt/m0/trt-dev/models/sp_head.onnx"),
        (r"<REPO>"
         r"\deploy\run_engine2.cpp", "/opt/m0/trt-dev/src/run_engine2.cpp")):
    print("put", remote.split("/")[-1],
          sftp.put(local, remote) and "ok")
sftp.close()

SH = r"""cat > /opt/m0/trt-dev/mods/run_split3.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
LOG=split3_run.log
echo "=== build bb ===" > $LOG
/usr/local/bin/onnx2engine2 models/sp_backbone.onnx models/e_bb.engine \
  --fp16 --int8 --ws-mb 2048 > bb_build.log 2>&1
rc=$?
echo "bb rc=$rc" >> $LOG
ls -la models/e_bb.engine 2>/dev/null >> $LOG
if [ $rc -ne 0 ]; then tail -20 bb_build.log >> $LOG
  echo "SPLIT3_DONE" >> /opt/m0/trt-dev/mods/SPLIT3_DONE; exit 1; fi

echo "=== build hd ===" >> $LOG
/usr/local/bin/onnx2engine2 models/sp_head.onnx models/e_hd.engine \
  --fp16 --ws-mb 2048 --plugins /usr/local/lib/libdfaplug_v8.so \
  > hd_build.log 2>&1
rc=$?
echo "hd rc=$rc" >> $LOG
ls -la models/e_hd.engine 2>/dev/null >> $LOG
if [ $rc -ne 0 ]; then tail -20 hd_build.log >> $LOG
  echo "SPLIT3_DONE" >> /opt/m0/trt-dev/mods/SPLIT3_DONE; exit 1; fi

echo "=== trtexec profiles ===" >> $LOG
/usr/src/tensorrt/bin/trtexec --loadEngine=models/e_bb.engine \
  --dumpProfile --iterations=100 > e_bb_prof.log 2>&1
grep ' Total' e_bb_prof.log | tail -1 >> $LOG
/usr/src/tensorrt/bin/trtexec --loadEngine=models/e_hd.engine \
  --plugins=/usr/local/lib/libdfaplug_v8.so \
  --dumpProfile --iterations=100 > e_hd_prof.log 2>&1
grep ' Total' e_hd_prof.log | tail -1 >> $LOG

echo "=== build runner ===" >> $LOG
g++ -O2 -I/usr/local/cuda/include -o /usr/local/bin/run_engine2 \
  src/run_engine2.cpp -lnvinfer -L/usr/local/cuda/lib64 -lcudart -ldl \
  > runner_build.log 2>&1
rc=$?
echo "runner rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then tail -20 runner_build.log >> $LOG
  echo "SPLIT3_DONE" >> /opt/m0/trt-dev/mods/SPLIT3_DONE; exit 1; fi

echo "=== 1-frame chained run ===" >> $LOG
rm -rf vec/mini/outs_00
/usr/local/bin/run_engine2 models/e_bb.engine models/e_hd.engine \
  /usr/local/lib/libdfaplug_v8.so vec/mini/in_00 vec/mini/in_00 \
  --dump vec/mini/outs_00 --iters 5 > one_frame.log 2>&1
rc=$?
echo "frame rc=$rc" >> $LOG
tail -8 one_frame.log >> $LOG

echo "=== 100-iter timing ===" >> $LOG
/usr/local/bin/run_engine2 models/e_bb.engine models/e_hd.engine \
  /usr/local/lib/libdfaplug_v8.so vec/mini/in_00 vec/mini/in_00 \
  --iters 100 --warmup 10 > timing.log 2>&1
grep MEAN timing.log >> $LOG
echo "SPLIT3_DONE" >> /opt/m0/trt-dev/mods/SPLIT3_DONE
EOF"""
print(run(SH))
run("chmod +x /opt/m0/trt-dev/mods/run_split3.sh")
run("rm -f /opt/m0/trt-dev/mods/SPLIT3_DONE")
run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_split3.sh > split3.out "
    "2>&1 < /dev/null & echo GO", t=5)
print("launched run_split3.sh (~10 min: 2 builds + profiles + runner + runs)")
cli.close()
