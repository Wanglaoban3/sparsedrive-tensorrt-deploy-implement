#!/bin/bash
# mini 81 帧链式推理（交付组合: e_T6.engine + libdfaplug_v3.so）
# 用法: bash repro/mini_pipeline.sh [起始帧] [结束帧]   默认 0..80
# 逐帧: run_engine 推理 -> 把 next_* 时序输出拷成下一帧的 prev_*
set -x
cd /opt/m0/trt-dev
ENGINE=models/e_T6.engine
PLUGIN=/usr/local/lib/libdfaplug_v3.so
K0=${1:-0}
K1=${2:-80}
PREV_LINES='prev_det_feat\tf32\t1,600,256\tprev_det_feat.bin\nprev_det_anchor\tf32\t1,600,11\tprev_det_anchor.bin\nprev_det_conf\tf32\t1,600\tprev_det_conf.bin\nprev_det_id\ti32\t1,600\tprev_det_id.bin\nprev_id_count\ti32\t1,1\tprev_id_count.bin\nprev_map_feat\tf32\t1,33,256\tprev_map_feat.bin\nprev_map_anchor\tf32\t1,33,40\tprev_map_anchor.bin\nprev_map_conf\tf32\t1,33\tprev_map_conf.bin\n'
for k in $(seq -f %02g $K0 $K1); do
  run_engine $ENGINE $PLUGIN \
    vec/mini/in_$k vec/mini/in_$k --dump vec/mini/out_$k \
    > repro/mini_$k.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAIL $k rc=$rc" >> repro/mini_pipeline.log
    break
  fi
  if [ "$k" != "$(printf %02d $K1)" ]; then
    nk=$(printf "%02d" $((10#$k + 1)))
    cp vec/mini/out_$k/next_det_feat.bin vec/mini/in_$nk/prev_det_feat.bin
    cp vec/mini/out_$k/next_det_anchor.bin vec/mini/in_$nk/prev_det_anchor.bin
    cp vec/mini/out_$k/next_det_conf.bin vec/mini/in_$nk/prev_det_conf.bin
    cp vec/mini/out_$k/next_det_instance_id.bin vec/mini/in_$nk/prev_det_id.bin
    cp vec/mini/out_$k/next_id_count.bin vec/mini/in_$nk/prev_id_count.bin
    cp vec/mini/out_$k/next_map_feat.bin vec/mini/in_$nk/prev_map_feat.bin
    cp vec/mini/out_$k/next_map_anchor.bin vec/mini/in_$nk/prev_map_anchor.bin
    cp vec/mini/out_$k/next_map_conf.bin vec/mini/in_$nk/prev_map_conf.bin
    grep -q prev_det_feat vec/mini/in_$nk/manifest.tsv || \
      printf "$PREV_LINES" >> vec/mini/in_$nk/manifest.tsv
  fi
  echo "done $k files=$(ls vec/mini/out_$k | wc -l)" >> repro/mini_pipeline.log
done
echo MINI_DONE >> repro/mini_pipeline.log
