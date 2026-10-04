#!/bin/bash
# mini 81 帧链式推理(场景重置修复版)。
# 与 mini_pipeline_v8.sh 唯一区别: 下一帧是场景首帧时不 cp prev_*(保留
# prep_mini_inputs 预置的零历史), 其余帧照常盲传。
# 用法: bash repro/mini_pipeline_reset.sh ENGINE PLUGIN OUTDIR_PARENT [K0] [K1] [BOUNDS]
#   OUTDIR_PARENT: dump 父目录, 每帧 dump 到 $OUTDIR_PARENT/outv8_$k
#   BOUNDS: 场景首帧(0 基)列表, 空格分隔, mini 为 "40"
set -x
cd /opt/m0/trt-dev
ENGINE=$1
PLUGIN=$2
OUTP=$3
K0=${4:-0}
K1=${5:-80}
BOUNDS=${6:-40}
mkdir -p "$OUTP"
LOG=repro/mini_reset_$(basename $OUTP).log
rm -f "$LOG"
for k in $(seq -f %02g $K0 $K1); do
  run_engine $ENGINE $PLUGIN vec/mini/in_$k vec/mini/in_$k \
    --dump "$OUTP/outv8_$k" > repro/mini_reset_frame_$k.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAIL $k rc=$rc" >> "$LOG"
    break
  fi
  if [ "$k" != "$(printf %02d $K1)" ]; then
    nk=$(printf "%02d" $((10#$k + 1)))
    skip=""
    for b in $BOUNDS; do
      [ "$nk" = "$(printf %02d $b)" ] && skip=yes
    done
    if [ -z "$skip" ]; then
      cp "$OUTP/outv8_$k/next_det_feat.bin" vec/mini/in_$nk/prev_det_feat.bin
      cp "$OUTP/outv8_$k/next_det_anchor.bin" vec/mini/in_$nk/prev_det_anchor.bin
      cp "$OUTP/outv8_$k/next_det_conf.bin" vec/mini/in_$nk/prev_det_conf.bin
      cp "$OUTP/outv8_$k/next_det_instance_id.bin" vec/mini/in_$nk/prev_det_id.bin
      cp "$OUTP/outv8_$k/next_id_count.bin" vec/mini/in_$nk/prev_id_count.bin
      cp "$OUTP/outv8_$k/next_map_feat.bin" vec/mini/in_$nk/prev_map_feat.bin
      cp "$OUTP/outv8_$k/next_map_anchor.bin" vec/mini/in_$nk/prev_map_anchor.bin
      cp "$OUTP/outv8_$k/next_map_conf.bin" vec/mini/in_$nk/prev_map_conf.bin
    fi
  fi
  echo "done $k" >> "$LOG"
done
echo "RESET_CHAIN_DONE $(basename $OUTP)" >> "$LOG"
