/* sp_rule_slow_lead.c — L1: 慢车压道: 前车均速 < 2.0 且 ego > 5.0, 间距 < 25m
 * 出处: ad-data-engine events.py agent_events slow_lead — 车端流式移植 (M9a 计划 Task 3).
 * THR 段: [slow_lead] (thr.conf); 默认值 = v0 种子, 上车前须分位数重标.
 */
#include "sp_rule_util.h"
#include <stdio.h>
#include <stdlib.h>

static double thr_dist = 25.0;
static double thr_speed = 2.0;
static double thr_ego = 5.0;

static int init(const char* kv) {
  thr_dist = ru_thr(kv, "dist_thr", 25.0);
  thr_speed = ru_thr(kv, "speed_thr", 2.0);
  thr_ego = ru_thr(kv, "ego_thr", 5.0);
  return 0;
}

static int eval(const sp_frame_ctx* c, sp_rule_event* ev) {
  if (!c || !ev || c->status != 0) return 0;  /* LATCH/旧帧不评 */
  int li = ru_pick_lead(c, 2.0, 40.0, 2.5);
  if (li < 0) return 0;
  double svx[SP_RULE_HIST];
  int k = ru_id_series(c, c->det[li].id, 0, 0, svx, 0, 0);
  if (k < 3) return 0;
  double sum = 0;
  for (int i = 0; i < k; ++i) sum += svx[i];
  double d0 = sqrt((double)c->det[li].x * c->det[li].x +
                   (double)c->det[li].y * c->det[li].y);
  if (d0 < thr_dist && sum / k < thr_speed && c->ego.speed > thr_ego) {
    snprintf(ev->name, sizeof(ev->name), "slow_lead");
    ev->strength = 1.0f;
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
      SP_RULE_ABI, "slow_lead", "1.0.0", init, eval, fini};
  return &d;
}
