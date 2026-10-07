/* sp_rule_stationary_approach.c — L1: 静止目标逼近: 前车均速 < 0.5 且 TTC < 3.0s
 * 出处: ad-data-engine events.py agent_events stationary_approach — 车端流式移植 (M9a 计划 Task 3).
 * THR 段: [stationary_approach] (thr.conf); 默认值 = v0 种子, 上车前须分位数重标.
 */
#include "sp_rule_util.h"
#include <stdio.h>
#include <stdlib.h>

static double thr_ttc = 3.0;
static double thr_vlead = 0.5;
static double thr_ego = 1.0;

static int init(const char* kv) {
  thr_ttc = ru_thr(kv, "ttc_thr", 3.0);
  thr_vlead = ru_thr(kv, "vlead_thr", 0.5);
  thr_ego = ru_thr(kv, "ego_thr", 1.0);
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
  if (c->ego.speed > thr_ego && sum / k < thr_vlead &&
      d0 / c->ego.speed < thr_ttc) {
    snprintf(ev->name, sizeof(ev->name), "stationary_approach");
    ev->strength = (float)(thr_ttc - d0 / c->ego.speed);
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
      SP_RULE_ABI, "stationary_approach", "1.0.0", init, eval, fini};
  return &d;
}
