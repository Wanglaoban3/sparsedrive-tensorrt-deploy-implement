#!/bin/bash
# 第二级 motion/plan 链:每帧吃 e_T6(reset 链)输出 + 外部维护的 9 张量
# mp 状态,跑单一时序引擎 e_mp(场景首帧复位为零状态,与感知 reset 链同
# 约定),旋转 next_* 回状态目录,dump 5 个 mp 输出 + 4 个 decode 用状态
# 快照。
# 用法: bash repro/mini_pipeline_mp.sh ENGINE PLUGIN
#         SRCROOT OUTM [K0] [K1] [BOUNDS]
#   SRCROOT: e_T6 reset 链 dump 父目录(每帧 $SRCROOT/outv8_$k)
#   OUTM:    mp dump 父目录(每帧 $OUTM/outm_$k)
#   BOUNDS:  场景首帧(0 基)列表, mini 为 "0 40"
set -x
cd /opt/m0/trt-dev
ENG=$1
PLUGIN=$2
SRC=$3
OUTM=$4
K0=${5:-0}
K1=${6:-80}
BOUNDS=${7:-"0 40"}
STATE=vec/mini_mpstate
ZERO=vec/mini_mpstate0
FEED=vec/mini_mpfeed
mkdir -p "$STATE" "$FEED" "$OUTM"
LOG=repro/mini_mp_$(basename $OUTM).log
rm -f "$LOG"
# run_engine 的 inputs 目录必须含 manifest.tsv(name\tdtype\tdims\tfile),
# 19 输入形状静态,启动时生成一次,每帧拷入 FEED
MF=$STATE/manifest_base.tsv
{
  printf 'det_cls\tf32\t1,900,10\tdet_cls.bin\n'
  printf 'det_bbox\tf32\t1,900,11\tdet_bbox.bin\n'
  printf 'det_feat\tf32\t1,900,256\tdet_feat.bin\n'
  printf 'det_anchor_embed\tf32\t1,900,256\tdet_anchor_embed.bin\n'
  printf 'det_instance_id\ti32\t1,900\tdet_instance_id.bin\n'
  printf 'map_cls\tf32\t1,100,3\tmap_cls.bin\n'
  printf 'map_feat\tf32\t1,100,256\tmap_feat.bin\n'
  printf 'map_anchor_embed\tf32\t1,100,256\tmap_anchor_embed.bin\n'
  printf 'ego_feature_map\tf32\t1,256,8,22\tego_feature_map.bin\n'
  printf 't_matrix\tf32\t1,4,4\tt_matrix.bin\n'
  printf 'history_instance_feature\tf32\t1,900,4,256\thistory_instance_feature.bin\n'
  printf 'history_anchor\tf32\t1,900,4,11\thistory_anchor.bin\n'
  printf 'history_period\ti32\t1,900\thistory_period.bin\n'
  printf 'prev_instance_id\ti32\t1,900\tprev_instance_id.bin\n'
  printf 'prev_confidence\tf32\t1,900\tprev_confidence.bin\n'
  printf 'history_ego_feature\tf32\t1,1,4,256\thistory_ego_feature.bin\n'
  printf 'history_ego_anchor\tf32\t1,1,4,11\thistory_ego_anchor.bin\n'
  printf 'history_ego_period\ti32\t1,1\thistory_ego_period.bin\n'
  printf 'prev_ego_status\tf32\t1,1,10\tprev_ego_status.bin\n'
} > "$MF"
for k in $(seq -f %02g $K0 $K1); do
  for b in $BOUNDS; do
    if [ "$k" = "$(printf %02d $b)" ]; then
      cp $ZERO/*.bin $STATE/
    fi
  done
  rm -rf "$FEED"
  mkdir -p "$FEED" "$OUTM/outm_$k"
  cp "$MF" "$FEED/manifest.tsv"
  cp $SRC/outv8_$k/det_cls.bin $FEED/det_cls.bin
  cp $SRC/outv8_$k/det_bbox.bin $FEED/det_bbox.bin
  cp $SRC/outv8_$k/det_instance_feature.bin $FEED/det_feat.bin
  cp $SRC/outv8_$k/det_anchor_embed.bin $FEED/det_anchor_embed.bin
  cp $SRC/outv8_$k/det_instance_id.bin $FEED/det_instance_id.bin
  cp $SRC/outv8_$k/map_cls.bin $FEED/map_cls.bin
  cp $SRC/outv8_$k/map_instance_feature.bin $FEED/map_feat.bin
  cp $SRC/outv8_$k/map_anchor_embed.bin $FEED/map_anchor_embed.bin
  cp $SRC/outv8_$k/ego_feature_map.bin $FEED/ego_feature_map.bin
  cp vec/mini/in_$k/instance_t_matrix.bin $FEED/t_matrix.bin
  cp $STATE/history_instance_feature.bin $STATE/history_anchor.bin \
     $STATE/history_period.bin $STATE/prev_instance_id.bin \
     $STATE/prev_confidence.bin $STATE/history_ego_feature.bin \
     $STATE/history_ego_anchor.bin $STATE/history_ego_period.bin \
     $STATE/prev_ego_status.bin $FEED/
  run_engine $ENG $PLUGIN $FEED $FEED --dump "$OUTM/outm_$k" \
    > repro/mini_mp_frame_$k.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAIL $k rc=$rc" >> "$LOG"
    break
  fi
  D=$OUTM/outm_$k
  cp $D/next_history_instance_feature.bin $STATE/history_instance_feature.bin
  cp $D/next_history_anchor.bin $STATE/history_anchor.bin
  cp $D/next_history_period.bin $STATE/history_period.bin
  cp $D/next_prev_instance_id.bin $STATE/prev_instance_id.bin
  cp $D/next_prev_confidence.bin $STATE/prev_confidence.bin
  cp $D/next_history_ego_feature.bin $STATE/history_ego_feature.bin
  cp $D/next_history_ego_anchor.bin $STATE/history_ego_anchor.bin
  cp $D/next_history_ego_period.bin $STATE/history_ego_period.bin
  cp $D/next_prev_ego_status.bin $STATE/prev_ego_status.bin
  cp $D/next_history_anchor.bin $D/history_anchor.bin
  cp $D/next_history_period.bin $D/history_period.bin
  cp $D/next_history_ego_anchor.bin $D/history_ego_anchor.bin
  cp $D/next_history_ego_period.bin $D/history_ego_period.bin
  echo "done $k" >> "$LOG"
done
echo "MP_CHAIN_DONE $(basename $OUTM)" >> "$LOG"
