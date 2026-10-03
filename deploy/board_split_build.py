# -*- coding: utf-8 -*-
"""P4-3/P4-4: upload split onnx + run_engine2.cpp; build both engines,
profile each, compile runner, 1-frame chained run. Marker: SPLITS_DONE."""
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
    o = out.read().decode("utf-8", "replace")
    e = err.read().decode("utf-8", "replace")
    return o + e


sftp = cli.open_sftp()
for local, remote in (
        (W + r"\sp_backbone.onnx", "/opt/m0/trt-dev/models/sp_backbone.onnx"),
        (W + r"\sp_head.onnx", "/opt/m0/trt-dev/models/sp_head.onnx"),
        (r"<REPO>"
         r"\deploy\run_engine2.cpp", "/opt/m0/trt-dev/src/run_engine2.cpp")):
    print("put", remote.split("/")[-1],
          sftp.put(local, remote) and "ok")
sftp.close()

SH = r"""cat > /opt/m0/trt-dev/mods/run_split.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
LOG=split_run.log
echo "=== build bb ===" > $LOG
/usr/local/bin/onnx2engine2 models/sp_backbone.onnx models/e_bb.engine \
  --fp16 --int8 --ws-mb 2048 > bb_build.log 2>&1
echo "bb rc=$?" >> $LOG
ls -la models/e_bb.engine 2>/dev/null >> $LOG
if [ ! -f models/e_bb.engine ]; then
  echo "SPLITS_DONE" >> /opt/m0/trt-dev/mods/SPLITS_DONE; exit 1
fi

echo "=== build hd ===" >> $LOG
/usr/local/bin/onnx2engine2 models/sp_head.onnx models/e_hd.engine \
  --fp16 --ws-mb 2048 --plugins /usr/local/lib/libdfaplug_v8.so \
  > hd_build.log 2>&1
echo "hd rc=$?" >> $LOG
ls -la models/e_hd.engine 2>/dev/null >> $LOG
if [ ! -f models/e_hd.engine ]; then
  echo "SPLITS_DONE" >> /opt/m0/trt-dev/mods/SPLITS_DONE; exit 1
fi

echo "=== trtexec profiles ===" >> $LOG
/usr/src/tensorrt/bin/trtexec --loadEngine=models/e_bb.engine \
  --dumpProfile --iterations=100 > e_bb_prof.log 2>&1
echo "bb prof rc=$?" >> $LOG
grep ' Total' e_bb_prof.log | tail -1 >> $LOG
/usr/src/tensorrt/bin/trtexec --loadEngine=models/e_hd.engine \
  --plugins=/usr/local/lib/libdfaplug_v8.so \
  --dumpProfile --iterations=100 > e_hd_prof.log 2>&1
echo "hd prof rc=$?" >> $LOG
grep ' Total' e_hd_prof.log | tail -1 >> $LOG

echo "=== build runner ===" >> $LOG
g++ -O2 -I/usr/local/cuda/include -o /usr/local/bin/run_engine2 \
  src/run_engine2.cpp -lnvinfer -L/usr/local/cuda/lib64 -lcudart \
  > runner_build.log 2>&1
echo "runner rc=$?" >> $LOG
if [ $? -ne 0 ]; then
  tail -20 runner_build.log >> $LOG
  echo "SPLITS_DONE" >> /opt/m0/trt-dev/mods/SPLITS_DONE; exit 1
fi

echo "=== 1-frame chained run ===" >> $LOG
rm -rf vec/mini/outs_00
run_engine2 models/e_bb.engine models/e_hd.engine \
  /usr/local/lib/libdfaplug_v8.so vec/mini/in_00 vec/mini/in_00 \
  --dump vec/mini/outs_00 --iters 5 > one_frame.log 2>&1
echo "frame rc=$?" >> $LOG
tail -5 one_frame.log >> $LOG
echo "SPLITS_DONE" >> /opt/m0/trt-dev/mods/SPLITS_DONE
EOF"""
print(run(SH))
run("chmod +x /opt/m0/trt-dev/mods/run_split.sh")
run("rm -f /opt/m0/trt-dev/mods/SPLITS_DONE")
run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_split.sh > split.out "
    "2>&1 < /dev/null & echo GO", t=5)
print("launched run_split.sh")
cli.close()
