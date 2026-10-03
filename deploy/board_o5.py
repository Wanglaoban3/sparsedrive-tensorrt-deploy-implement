# -*- coding: utf-8 -*-
"""Build e_hd/e_bb with --builderOptimizationLevel=5 via trtexec, profile,
compare ForeignNode totals vs onnx2engine2 builds. Marker: O5_DONE."""
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


SH = r"""cat > /opt/m0/trt-dev/mods/run_o5.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
LOG=o5_run.log
T=/usr/src/tensorrt/bin/trtexec
PL=/usr/local/lib/libdfaplug_v8.so
echo "=== trtexec version check ===" > $LOG
$T --help 2>&1 | grep -i "builderOptimizationLevel" >> $LOG 2>&1
echo "=== build hd o5 ===" >> $LOG
$T --onnx=models/sp_head.onnx --fp16 --staticPlugins=$PL \
  --builderOptimizationLevel=5 --saveEngine=models/e_hd_o5.engine \
  --dumpProfile --iterations=100 > e_hd_o5.log 2>&1
rc=$?
echo "hd o5 rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then tail -15 e_hd_o5.log >> $LOG
  echo "O5_DONE" >> /opt/m0/trt-dev/mods/O5_DONE; exit 1; fi
ls -la models/e_hd_o5.engine >> $LOG
grep ' Total' e_hd_o5.log | tail -1 >> $LOG
echo "--- top ForeignNode rows (hd o5) ---" >> $LOG
grep "ForeignNode" e_hd_o5.log | sort -k2 -g -r | head -8 >> $LOG

echo "=== build bb o5 ===" >> $LOG
$T --onnx=models/sp_backbone.onnx --fp16 --int8 \
  --builderOptimizationLevel=5 --saveEngine=models/e_bb_o5.engine \
  --dumpProfile --iterations=100 > e_bb_o5.log 2>&1
rc=$?
echo "bb o5 rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then tail -15 e_bb_o5.log >> $LOG
  echo "O5_DONE" >> /opt/m0/trt-dev/mods/O5_DONE; exit 1; fi
ls -la models/e_bb_o5.engine >> $LOG
grep ' Total' e_bb_o5.log | tail -1 >> $LOG
echo "--- top rows (bb o5) ---" >> $LOG
grep -E "ForeignNode|Conv|Quantize" e_bb_o5.log | sort -k2 -g -r | head -8 >> $LOG
echo "O5_DONE" >> /opt/m0/trt-dev/mods/O5_DONE
EOF"""
print(run(SH))
run("chmod +x /opt/m0/trt-dev/mods/run_o5.sh")
run("rm -f /opt/m0/trt-dev/mods/O5_DONE")
try:
    print(run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_o5.sh "
              "> o5.out 2>&1 < /dev/null & echo GO", t=8))
except Exception as e:
    print("read-timeout benign:", type(e).__name__)
cli.close()
print("o5 job launched (hd build ~15-25 min + bb ~3 min)")
