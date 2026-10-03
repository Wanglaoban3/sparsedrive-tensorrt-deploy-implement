# -*- coding: utf-8 -*-
"""P5 隔离实验: (1) FlashSDPA 内核自测 (合成数据 CPU 对拍, 无引擎);
(2) 基线/FA 探针引擎 (暴露 layers.19 站点 Q/K^T/V/链输出) 各 dump 一次。
Marker: SSA5_DONE."""
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


def put(local, remote):
    s = cli.open_sftp()
    s.put(local, remote)
    s.close()
    return "ok"


print("put selftest", put(D + r"\selftest_fa.cu", "/opt/m0/trt-dev/src/selftest_fa.cu"))
print("put fapart", put(D + r"\_fa_part.cu", "/opt/m0/trt-dev/src/_fa_part.cu"))
print("put probe_base", put(W + r"\sp_head_det_probe.onnx",
                            "/opt/m0/trt-dev/models/sp_head_det_probe.onnx"))
print("put probe_fa", put(W + r"\sp_head_detfa_probe.onnx",
                          "/opt/m0/trt-dev/models/sp_head_detfa_probe.onnx"))

SH = r"""cat > /opt/m0/trt-dev/mods/run_ssa5.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
LOG=ssa5_run.log
echo "=== selftest_fa ===" > $LOG
cd src && nvcc -O3 -arch=sm_87 -o /usr/local/bin/selftest_fa \
  selftest_fa.cu -lnvinfer > ../selftest_build.log 2>&1
rc=$?
cd ..
echo "selftest build rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then tail -25 src/../selftest_build.log >> $LOG
  echo "SSA5_DONE" >> /opt/m0/trt-dev/mods/SSA5_DONE; exit 1; fi
/usr/local/bin/selftest_fa >> $LOG 2>&1

echo "=== build probe base ===" >> $LOG
/usr/local/bin/onnx2engine2 models/sp_head_det_probe.onnx \
  models/e_det_probe.engine --fp16 --ws-mb 2048 \
  --plugins /usr/local/lib/libdfaplug_v8.so > probeb_build.log 2>&1
rc=$?
echo "probe base rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then tail -25 probeb_build.log >> $LOG
  echo "SSA5_DONE" >> /opt/m0/trt-dev/mods/SSA5_DONE; exit 1; fi

echo "=== build probe fa ===" >> $LOG
/usr/local/bin/onnx2engine2 models/sp_head_detfa_probe.onnx \
  models/e_detfa_probe.engine --fp16 --ws-mb 2048 \
  --plugins /usr/local/lib/libdfaplug_v10.so > probef_build.log 2>&1
rc=$?
echo "probe fa rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then tail -25 probef_build.log >> $LOG
  echo "SSA5_DONE" >> /opt/m0/trt-dev/mods/SSA5_DONE; exit 1; fi

echo "=== dump probe base ===" >> $LOG
rm -rf vec/mini/outs_probe_base
/usr/local/bin/run_engines models/e_bb.engine models/e_det_probe.engine \
  /usr/local/lib/libdfaplug_v8.so vec/mini/in_00 vec/mini/in_00 \
  --dump vec/mini/outs_probe_base --iters 1 > probeb_run.log 2>&1
echo "probe base run rc=$? files:" >> $LOG
ls vec/mini/outs_probe_base | wc -l >> $LOG

echo "=== dump probe fa ===" >> $LOG
rm -rf vec/mini/outs_probe_fa
/usr/local/bin/run_engines models/e_bb.engine models/e_detfa_probe.engine \
  /usr/local/lib/libdfaplug_v10.so vec/mini/in_00 vec/mini/in_00 \
  --dump vec/mini/outs_probe_fa --iters 1 > probef_run.log 2>&1
echo "probe fa run rc=$? files:" >> $LOG
ls vec/mini/outs_probe_fa | wc -l >> $LOG
echo "SSA5_DONE" >> /opt/m0/trt-dev/mods/SSA5_DONE
EOF"""
print(run(SH))
run("chmod +x /opt/m0/trt-dev/mods/run_ssa5.sh")
run("rm -f /opt/m0/trt-dev/mods/SSA5_DONE")
try:
    print(run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_ssa5.sh "
              "> ssa5.out 2>&1 < /dev/null & echo GO", t=8))
except Exception as e:
    print("read-timeout benign:", type(e).__name__)
cli.close()
print("ssa5 job launched")
