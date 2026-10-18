/* sp_rule_cut_in.c — L1: 加塞 cut-in: 邻道目标横穿进入自车道, 纵距 < 20m
 * 出处: ad-data-engine events.py agent_events cut_in (现在时: 进入中) — 车端流式移植 (M9a 计划 Task 3).
 * THR 段: [cut_in] (thr.conf); 默认值 = v0 种子, 上车前须分位数重标.
 */
#include "sp_rule_util.h"
#include <stdio.h>
#include <stdlib.h>

static double thr_dist = 20.0;
static double lat_edge = 2.5;
static double lat_in = 2.0;

static int init(const char* kv) {
  thr_dist = ru_thr(kv, "dist_thr", 20.0);
  lat_edge = ru_thr(kv, "lat_edge", 2.5);
  lat_in = ru_thr(kv, "lat_in", 2.0);
  return 0;
}

static int eval(const sp_frame_ctx* c, sp_rule_event* ev) {
  if (!c || !ev || c->status != 0) return 0;  /* LATCH/旧帧不评 */
  for (uint32_t i = 0; i < c->n_det; ++i) {
    const sp_trk* d = &c->det[i];
    if (!ru_is_vehicle(d->label) || d->id < 0 || d->x > thr_dist) continue;
    double sx[SP_RULE_HIST], sy[SP_RULE_HIST];
    int k = ru_id_series(c, d->id, sx, sy, 0, 0, 0);
    if (k < 3) continue;
    if (fabs(sy[0]) >= lat_edge && fabs(sy[k - 1]) < lat_in &&
        (sy[k - 1] - sy[0]) * sy[0] < 0) {
      snprintf(ev->name, sizeof(ev->name), "cut_in");
      ev->strength = 1.0f;
      return 1;
    }
  }
  return 0;
}

static void fini(void) {}

#ifdef __cplusplus
extern "C"
#endif
const sp_rule_desc* sp_rule_query(void) {
  static const sp_rule_desc d = {
      SP_RULE_ABI, "cut_in", "1.0.0", init, eval, fini};
  return &d;
}
