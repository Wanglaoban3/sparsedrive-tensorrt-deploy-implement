/* sp_rule_lead_hard_brake.c — L1: 前车急刹: 最近前车减速 <= -3.0 持续
 * >= 0.3s, 间距 < 20m, 窗内巡航速 > 2.
 * 出处: ad-data-engine events.py agent_events lead_hard_brake — 车端流式
 * 移植 (M9a 计划 Task 3). 终审 I1 修正: 原实现 min(vx)<=-3 检测的是
 * "目标倒车" (vx 带符号), 且上游 events.py 的 speed 非负使该式恒真退化;
 * 按 plan 契约改为 vx 差分减速序列 + sustained.
 * THR 段: [lead_hard_brake] (thr.conf); 默认值 = v0 种子, 上车前须分位数重标.
 */
#include "sp_rule_util.h"
#include <stdio.h>
#include <stdlib.h>

static double thr_dist = 20.0;
static double thr_decel = -3.0;
static double thr_v0 = 2.0;
static double thr_dur = 0.3;

static int init(const char* kv) {
  thr_dist = ru_thr(kv, "dist_thr", 20.0);
  thr_decel = ru_thr(kv, "decel_thr", -3.0);
  thr_v0 = ru_thr(kv, "v0_thr", 2.0);
  thr_dur = ru_thr(kv, "dur_s", 0.3);
  return 0;
}

static int eval(const sp_frame_ctx* c, sp_rule_event* ev) {
  if (!c || !ev || c->status != 0) return 0;  /* LATCH/旧帧不评 */
  int li = ru_pick_lead(c, 2.0, 40.0, 2.5);
  if (li < 0) return 0;
  double svx[SP_RULE_HIST], sts[SP_RULE_HIST];
  int k = ru_id_series(c, c->det[li].id, 0, 0, svx, 0, sts);
  if (k < 3) return 0;
  /* 减速序列 (chronological, 旧→新): a[i] = (v[i+1]-v[i]) / Δt */
  double a[SP_RULE_HIST];
  int64_t at[SP_RULE_HIST];
  for (int i = 0; i + 1 < k; ++i) {
    double dt = (sts[i + 1] - sts[i]) / 1e9;
    a[i] = dt > 1e-6 ? (svx[i + 1] - svx[i]) / dt : 0.0;
    at[i] = (int64_t)sts[i];
  }
  int na = k - 1;
  double mn = a[0], vmax = svx[0];
  for (int i = 0; i < na; ++i)
    if (a[i] < mn) mn = a[i];
  for (int i = 1; i < k; ++i)
    if (svx[i] > vmax) vmax = svx[i];
  double d0 = sqrt((double)c->det[li].x * c->det[li].x +
                   (double)c->det[li].y * c->det[li].y);
  if (d0 < thr_dist && vmax > thr_v0 &&
      ru_sustained(a, at, na, thr_decel, 0, thr_dur)) {
    snprintf(ev->name, sizeof(ev->name), "lead_hard_brake");
    ev->strength = (float)(-mn / 5.0);
    if (ev->strength > 1.0f) ev->strength = 1.0f;
    return 1;
  }
  return 0;
}

static void fini(void) {}

#ifdef __cplusplus
extern "C"
#endif
const sp_rule_desc* sp_rule_query(void) {
  static const sp_rule_desc d = {
      SP_RULE_ABI, "lead_hard_brake", "1.1.0", init, eval, fini};
  return &d;
}
