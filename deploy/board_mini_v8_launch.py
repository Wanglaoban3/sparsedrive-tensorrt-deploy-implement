# -*- coding: utf-8 -*-
"""launch mini_pipeline_v8.sh on the board: 81-frame chained e_T6 + v8 plugin,
outputs to vec/mini/outv8_XX (v3 baseline out_XX untouched)."""
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


SH = r"""cat > /opt/m0/trt-dev/repro/mini_pipeline_v8.sh <<'EOF'
#!/bin/bash
# mini 81 帧链式推理（e_T6.engine + libdfaplug_v8.so DFA 优化插件）
# 用法: bash repro/mini_pipeline_v8.sh [起始帧] [结束帧]   默认 0..80
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

print(run(SH))
run("chmod +x /opt/m0/trt-dev/repro/mini_pipeline_v8.sh")
run("rm -f /opt/m0/trt-dev/repro/MINI_V8_DONE")
run("md5sum /usr/local/lib/libdfaplug_v8.so /opt/m0/trt-dev/models/e_T6.engine")
run("cd /opt/m0/trt-dev && setsid nohup bash repro/mini_pipeline_v8.sh "
    "> repro/mini_v8.out 2>&1 < /dev/null & echo GO", t=5)
print("launched mini_pipeline_v8 (81 frames)")
print(run("sleep 25; tail -3 /opt/m0/trt-dev/repro/mini_v8.out; "
          "cat /opt/m0/trt-dev/repro/mini_pipeline_v8.log 2>/dev/null"))
cli.close()
