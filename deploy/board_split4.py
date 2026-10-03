# -*- coding: utf-8 -*-
"""Diagnose split chain: dump bb boundary + hd isolated with e_T6's real
boundary features. Marker: SPLIT4_DONE."""
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
print("put run_engine2.cpp",
      sftp.put(r"<REPO>"
               r"\deploy\run_engine2.cpp",
               "/opt/m0/trt-dev/src/run_engine2.cpp") and "ok")
print("put feat_flat_f16.bin",
      sftp.put(r"<REPO>"
               r"\work_dirs\sparsedrive_small_stage2\evaldata"
               r"\feat_flat_f16.bin",
               "/opt/m0/trt-dev/vec/feat_flat_f16.bin") and "ok")
sftp.close()

SH = r"""cat > /opt/m0/trt-dev/mods/run_split4.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
LOG=split4_run.log
echo "=== compile ===" > $LOG
g++ -O2 -I/usr/local/cuda/include -o /usr/local/bin/run_engine2 \
  src/run_engine2.cpp -lnvinfer -L/usr/local/cuda/lib64 -lcudart -ldl \
  > runner_build.log 2>&1
rc=$?
echo "runner rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then tail -20 runner_build.log >> $LOG
  echo "SPLIT4_DONE" >> /opt/m0/trt-dev/mods/SPLIT4_DONE; exit 1; fi

echo "=== A: chained with bb dump ===" >> $LOG
rm -rf vec/mini/outs_00b vec/mini/outbb_00
/usr/local/bin/run_engine2 models/e_bb.engine models/e_hd.engine \
  /usr/local/lib/libdfaplug_v8.so vec/mini/in_00 vec/mini/in_00 \
  --dump vec/mini/outs_00b --dump-bb vec/mini/outbb_00 --iters 3 \
  > one_a.log 2>&1
rc=$?
echo "A rc=$rc" >> $LOG
tail -4 one_a.log >> $LOG

echo "=== B: hd isolated with e_T6 real boundary ===" >> $LOG
rm -rf vec/mini/in_hd00 vec/mini/outhd_iso_00
mkdir -p vec/mini/in_hd00
cp vec/mini/in_00/* vec/mini/in_hd00/
cp vec/feat_flat_f16.bin vec/mini/in_hd00/feat_flat_f16.bin
printf '/Reshape_9_output_0\tf16\t1,89760,256\tfeat_flat_f16.bin\n' \
  >> vec/mini/in_hd00/manifest.tsv
/usr/local/bin/run_engine2 models/e_hd.engine models/e_hd.engine \
  /usr/local/lib/libdfaplug_v8.so vec/mini/in_hd00 vec/mini/in_hd00 \
  --dump vec/mini/outhd_iso_00 --iters 2 > one_b.log 2>&1
rc=$?
echo "B rc=$rc" >> $LOG
tail -4 one_b.log >> $LOG
echo "SPLIT4_DONE" >> /opt/m0/trt-dev/mods/SPLIT4_DONE
EOF"""
print(run(SH))
run("chmod +x /opt/m0/trt-dev/mods/run_split4.sh")
run("rm -f /opt/m0/trt-dev/mods/SPLIT4_DONE")
try:
    run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_split4.sh "
        "> split4.out 2>&1 < /dev/null & echo GO", t=8)
except Exception as e:
    print("launch read-timeout (benign):", type(e).__name__)
cli.close()
print("job launched; poll SPLIT4_DONE")
