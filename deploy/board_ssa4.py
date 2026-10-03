# -*- coding: utf-8 -*-
"""P5b-v2: push 修正版 dfaplug_v10.cu (8warp + 合并K转置拷贝); 只重编 .so
热交换 (内核改动, 引擎不需重建); 重profile e_det_fa + 重dump + 2链计时.
Marker: SSA4_DONE."""
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


print("put dfaplug_v10.cu",
      cli.open_sftp().put(D + r"\dfaplug_v10.cu",
                          "/opt/m0/trt-dev/src/dfaplug_v10.cu") or "ok")

SH = r"""cat > /opt/m0/trt-dev/mods/run_ssa4.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
LOG=ssa4_run.log
PL0=/usr/local/lib/libdfaplug_v10.so
echo "=== compile libdfaplug_v10.so (v2 kernel) ===" > $LOG
nvcc -O3 -arch=sm_87 -shared -Xcompiler -fPIC -o $PL0 \
  src/dfaplug_v10.cu -lnvinfer > v10b_build.log 2>&1
rc=$?
echo "v10b rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then tail -30 v10b_build.log >> $LOG
  echo "SSA4_DONE" >> /opt/m0/trt-dev/mods/SSA4_DONE; exit 1; fi
ls -la $PL0 >> $LOG

echo "=== profile e_det_fa (hot-swap, no rebuild) ===" >> $LOG
/usr/src/tensorrt/bin/trtexec --loadEngine=models/e_det_fa.engine \
  --staticPlugins=$PL0 --dumpProfile --iterations=100 \
  > e_detfa_prof2.log 2>&1
grep ' Total' e_detfa_prof2.log | tail -1 >> $LOG
echo "--- FlashSDPA rows ---" >> $LOG
grep "FlashSDPA" e_detfa_prof2.log >> $LOG
echo "--- top ForeignNode rows ---" >> $LOG
grep "ForeignNode" e_detfa_prof2.log | sort -k2 -g -r | head -8 >> $LOG

echo "=== 2-chain dump fa (bb+det_fa) ===" >> $LOG
rm -rf vec/mini/outs_detfa2_00
/usr/local/bin/run_engines models/e_bb.engine models/e_det_fa.engine $PL0 \
  vec/mini/in_00 vec/mini/in_00 --dump vec/mini/outs_detfa2_00 \
  --iters 3 > one_detfa2.log 2>&1
rc=$?
echo "detfa2 rc=$rc files:" >> $LOG
ls vec/mini/outs_detfa2_00 | wc -l >> $LOG
echo "=== 100-iter 2-chain timing (bb+det_fa) ===" >> $LOG
/usr/local/bin/run_engines models/e_bb.engine models/e_det_fa.engine $PL0 \
  vec/mini/in_00 vec/mini/in_00 --iters 100 --warmup 10 > tim_fa2.log 2>&1
grep MEAN tim_fa2.log >> $LOG
echo "=== 100-iter 2-chain timing baseline (bb+det) ===" >> $LOG
/usr/local/bin/run_engines models/e_bb.engine models/e_det.engine \
  /usr/local/lib/libdfaplug_v8.so \
  vec/mini/in_00 vec/mini/in_00 --iters 100 --warmup 10 > tim_ref2.log 2>&1
grep MEAN tim_ref2.log >> $LOG
echo "SSA4_DONE" >> /opt/m0/trt-dev/mods/SSA4_DONE
EOF"""
print(run(SH))
run("chmod +x /opt/m0/trt-dev/mods/run_ssa4.sh")
run("rm -f /opt/m0/trt-dev/mods/SSA4_DONE")
try:
    print(run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_ssa4.sh "
              "> ssa4.out 2>&1 < /dev/null & echo GO", t=8))
except Exception as e:
    print("read-timeout benign:", type(e).__name__)
cli.close()
print("ssa4 job launched")
