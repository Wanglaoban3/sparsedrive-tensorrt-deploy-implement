# -*- coding: utf-8 -*-
"""launch final board job: 100-iter trtexec profile with fixed v8 .so,
then mini 81-frame chained pipeline (e_T6 + v8, outputs outv8_XX)."""
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
    _, out, _ = cli.exec_command(cmd, timeout=t)
    return out.read().decode("utf-8", "replace")


MINI = r"""cat > /opt/m0/trt-dev/repro/mini_pipeline_v8.sh <<'EOF'
#!/bin/bash
# mini 81 帧链式推理（e_T6.engine + libdfaplug_v8.so DFA 优化插件）
# 输出到 outv8_XX, 不覆盖 v3 基线 out_XX; prev_* 链与 v3 管线同构
set -x
cd /opt/m0/trt-dev
ENGINE=models/e_T6.engine
PLUGIN=/usr/local/lib/libdfaplug_v8.so
OUTD=outv8
K0=${1:-0}
K1=${2:-80}
rm -f repro/mini_pipeline_v8.log
for k in $(seq -f %02g $K0 $K1); do
  run_engine $ENGINE $PLUGIN \
    vec/mini/in_$k vec/mini/in_$k --dump vec/mini/${OUTD}_$k \
    > repro/mini_v8_$k.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAIL $k rc=$rc" >> repro/mini_pipeline_v8.log
    break
  fi
  if [ "$k" != "$(printf %02d $K1)" ]; then
    nk=$(printf "%02d" $((10#$k + 1)))
    cp vec/mini/${OUTD}_$k/next_det_feat.bin vec/mini/in_$nk/prev_det_feat.bin
    cp vec/mini/${OUTD}_$k/next_det_anchor.bin vec/mini/in_$nk/prev_det_anchor.bin
    cp vec/mini/${OUTD}_$k/next_det_conf.bin vec/mini/in_$nk/prev_det_conf.bin
    cp vec/mini/${OUTD}_$k/next_det_instance_id.bin vec/mini/in_$nk/prev_det_id.bin
    cp vec/mini/${OUTD}_$k/next_id_count.bin vec/mini/in_$nk/prev_id_count.bin
    cp vec/mini/${OUTD}_$k/next_map_feat.bin vec/mini/in_$nk/prev_map_feat.bin
    cp vec/mini/${OUTD}_$k/next_map_anchor.bin vec/mini/in_$nk/prev_map_anchor.bin
    cp vec/mini/${OUTD}_$k/next_map_conf.bin vec/mini/in_$nk/prev_map_conf.bin
    grep -q prev_det_feat vec/mini/in_$nk/manifest.tsv || \
      printf 'prev_det_feat\tf32\t1,600,256\tprev_det_feat.bin\nprev_det_anchor\tf32\t1,600,11\tprev_det_anchor.bin\nprev_det_conf\tf32\t1,600\tprev_det_conf.bin\nprev_det_id\ti32\t1,600\tprev_det_id.bin\nprev_id_count\ti32\t1,1\tprev_id_count.bin\nprev_map_feat\tf32\t1,33,256\tprev_map_feat.bin\nprev_map_anchor\tf32\t1,33,40\tprev_map_anchor.bin\nprev_map_conf\tf32\t1,33\tprev_map_conf.bin\n' \
      >> vec/mini/in_$nk/manifest.tsv
  fi
  echo "done $k files=$(ls vec/mini/${OUTD}_$k | wc -l)" \
    >> repro/mini_pipeline_v8.log
done
echo MINI_V8_DONE >> repro/mini_pipeline_v8.log
EOF"""

FINAL = r"""cat > /opt/m0/trt-dev/mods/run_v8final.sh <<'EOF'
#!/bin/bash
cd /opt/m0/trt-dev
echo "=== 100-iter profile with ws-fixed v8 .so ===" > v8final_run.log
DFA_V8_DEBUG=1 /usr/src/tensorrt/bin/trtexec \
  --loadEngine=models/e_T6.engine \
  --plugins=/usr/local/lib/libdfaplug_v8.so \
  --dumpProfile --iterations=100 > v8_prof4.log 2>&1
echo "trtexec rc=$?" >> v8final_run.log
grep 'dfa_v8' v8_prof4.log | head -12 >> v8final_run.log
grep 'DeformableAggregation' v8_prof4.log | grep -v Reformat >> v8final_run.log
grep ' Total' v8_prof4.log | tail -1 >> v8final_run.log
echo PROF4_DONE >> /opt/m0/trt-dev/mods/V8PROF4_DONE
df -h /opt/m0 | tail -1 >> v8final_run.log

echo "=== mini 81-frame v8 pipeline ===" >> v8final_run.log
bash repro/mini_pipeline_v8.sh >> v8final_run.log 2>&1
echo "mini rc=$?" >> v8final_run.log
echo V8FINAL_DONE >> /opt/m0/trt-dev/mods/V8FINAL_DONE
EOF"""
print(run(MINI))
print(run(FINAL))
run("chmod +x /opt/m0/trt-dev/repro/mini_pipeline_v8.sh "
    "/opt/m0/trt-dev/mods/run_v8final.sh")
run("rm -f /opt/m0/trt-dev/mods/V8PROF4_DONE "
    "/opt/m0/trt-dev/mods/V8FINAL_DONE /opt/m0/trt-dev/repro/MINI_V8_DONE")
print(run("md5sum /usr/local/lib/libdfaplug_v8.so"))
run("cd /opt/m0/trt-dev/mods && setsid nohup bash run_v8final.sh "
    "> v8final.out 2>&1 < /dev/null & echo GO", t=5)
print("launched run_v8final.sh (profile + mini)")
cli.close()
