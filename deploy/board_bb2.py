# -*- coding: utf-8 -*-
"""Build re-quantized backbone e_bb2, profile, time both chain configs.
Marker: BB2_DONE."""
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


def sftp_put(cli, local, remote):
    s = cli.open_sftp()
    s.put(local, remote)
    s.close()
    return "ok"


print("put sp_backbone2.onnx",
      sftp_put(cli, W + r"\sp_backbone2.onnx",
               "/opt/m0/trt-dev/models/sp_backbone2.onnx"))


SH = r"""cat > /opt/m0/trt-dev/mods/run_bb2.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
LOG=bb2_run.log
PL=/usr/local/lib/libdfaplug_v8.so
echo "=== build bb2 (fpn requant) ===" > $LOG
/usr/local/bin/onnx2engine2 models/sp_backbone2.onnx models/e_bb2.engine \
  --fp16 --int8 --ws-mb 2048 > bb2_build.log 2>&1
rc=$?
echo "bb2 rc=$rc" >> $LOG
ls -la models/e_bb2.engine 2>/dev/null >> $LOG
if [ $rc -ne 0 ]; then tail -15 bb2_build.log >> $LOG
  echo "BB2_DONE" >> /opt/m0/trt-dev/mods/BB2_DONE; exit 1; fi
/usr/src/tensorrt/bin/trtexec --loadEngine=models/e_bb2.engine \
  --dumpProfile --iterations=100 > e_bb2_prof.log 2>&1
grep ' Total' e_bb2_prof.log | tail -1 >> $LOG
echo "--- bb2 fpn/quant rows ---" >> $LOG
grep -E "fpn_convs" e_bb2_prof.log | head -10 >> $LOG

echo "=== 2-chain timing (bb2+hd) ===" >> $LOG
/usr/local/bin/run_engines models/e_bb2.engine models/e_hd.engine $PL \
  vec/mini/in_00 vec/mini/in_00 --iters 100 --warmup 10 > t2b.log 2>&1
grep MEAN t2b.log >> $LOG
echo "=== 3-chain timing (bb2+det+rest) ===" >> $LOG
/usr/local/bin/run_engines models/e_bb2.engine models/e_det.engine \
  models/e_rest.engine $PL vec/mini/in_00 vec/mini/in_00 \
  --iters 100 --warmup 10 > t3b.log 2>&1
grep MEAN t3b.log >> $LOG
echo "=== 3-chain 1-frame dump ===" >> $LOG
rm -rf vec/mini/outs3b_00
/usr/local/bin/run_engines models/e_bb2.engine models/e_det.engine \
  models/e_rest.engine $PL vec/mini/in_00 vec/mini/in_00 \
  --dump vec/mini/outs3b_00 --iters 3 > one3b.log 2>&1
echo "rc=$?" >> $LOG
echo "=== 2-chain 1-frame dump ===" >> $LOG
rm -rf vec/mini/outs2b_00
/usr/local/bin/run_engines models/e_bb2.engine models/e_hd.engine $PL \
  vec/mini/in_00 vec/mini/in_00 --dump vec/mini/outs2b_00 --iters 3 \
  > one2b.log 2>&1
echo "rc=$?" >> $LOG
echo "BB2_DONE" >> /opt/m0/trt-dev/mods/BB2_DONE
EOF"""
print(run(SH))
run("chmod +x /opt/m0/trt-dev/mods/run_bb2.sh")
run("rm -f /opt/m0/trt-dev/mods/BB2_DONE")
try:
    print(run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_bb2.sh "
              "> bb2.out 2>&1 < /dev/null & echo GO", t=8))
except Exception as e:
    print("read-timeout benign:", type(e).__name__)
cli.close()
print("bb2 job launched (~4 min)")
