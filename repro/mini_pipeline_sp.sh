#!/bin/bash
# mini 81 帧链式推理（e_bb.engine + e_hd.engine 双引擎, D2D 链, v8 插件）
# 用法: bash repro/mini_pipeline_sp.sh [起始帧] [结束帧]   默认 0..80
# 输出到 outs_XX, 不覆盖 v3 out_XX / v8 outv8_XX
set -x
cd /opt/m0/trt-dev
PLUGIN=/usr/local/lib/libdfaplug_v8.so
OUTD=outs
K0=${1:-0}
K1=${2:-80}
rm -f repro/mini_pipeline_sp.log
for k in $(seq -f %02g $K0 $K1); do
  run_engine2 models/e_bb.engine models/e_hd.engine $PLUGIN \
    vec/mini/in_$k vec/mini/in_$k --dump vec/mini/${OUTD}_$k --iters 1 \
    > repro/mini_sp_$k.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAIL $k rc=$rc" >> repro/mini_pipeline_sp.log
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
    >> repro/mini_pipeline_sp.log
done
echo MINI_SP_DONE >> repro/mini_pipeline_sp.log
