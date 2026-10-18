/* sp_rule_construction_zone.c — L1: 施工/锥桶区: 锥桶最近距 < 30m
 * 出处: ad-data-engine events.py agent_events construction_zone — 车端流式移植 (M9a 计划 Task 3).
 * THR 段: [construction_zone] (thr.conf); 默认值 = v0 种子, 上车前须分位数重标.
 */
#include "sp_rule_util.h"
#include <stdio.h>
#include <stdlib.h>

static double thr_dist = 30.0;

static int init(const char* kv) {
  thr_dist = ru_thr(kv, "dist_thr", 30.0);
  return 0;
}

static int eval(const sp_frame_ctx* c, sp_rule_event* ev) {
  if (!c || !ev || c->status != 0) return 0;  /* LATCH/旧帧不评 */
  double dmin = 1e18;
  for (uint32_t i = 0; i < c->n_det; ++i) {
    const sp_trk* d = &c->det[i];
    if (!ru_is_cone(d->label)) continue;
    double dd = sqrt((double)d->x * d->x + (double)d->y * d->y);
    if (dd < dmin) dmin = dd;
  }
  if (dmin < thr_dist) {
    snprintf(ev->name, sizeof(ev->name), "construction_zone");
    ev->strength = (float)(1.0 - dmin / thr_dist);
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
      SP_RULE_ABI, "construction_zone", "1.0.0", init, eval, fini};
  return &d;
}
