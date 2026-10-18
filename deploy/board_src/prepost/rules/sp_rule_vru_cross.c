/* sp_rule_vru_cross.c — L1: VRU 横穿: 横向速度 > 1.5 且接近走廊 (|lat| < 3.0)
 * 出处: ad-data-engine events.py agent_events vru_cross — 车端流式移植 (M9a 计划 Task 3).
 * THR 段: [vru_cross] (thr.conf); 默认值 = v0 种子, 上车前须分位数重标.
 */
#include "sp_rule_util.h"
#include <stdio.h>
#include <stdlib.h>

static double lon_min = -5.0;
static double lon_max = 30.0;
static double thr_cross = 1.5;
static double thr_lat = 3.0;

static int init(const char* kv) {
  lon_min = ru_thr(kv, "lon_min", -5.0);
  lon_max = ru_thr(kv, "lon_max", 30.0);
  thr_cross = ru_thr(kv, "cross_thr", 1.5);
  thr_lat = ru_thr(kv, "lat_thr", 3.0);
  return 0;
}

static int eval(const sp_frame_ctx* c, sp_rule_event* ev) {
  if (!c || !ev || c->status != 0) return 0;  /* LATCH/旧帧不评 */
  for (uint32_t i = 0; i < c->n_det; ++i) {
    const sp_trk* d = &c->det[i];
    if (!ru_is_vru(d->label) || d->id < 0) continue;
    if (!(d->x > lon_min && d->x < lon_max)) continue;
    double sy[SP_RULE_HIST], sts[SP_RULE_HIST];
    int k = ru_id_series(c, d->id, 0, sy, 0, 0, sts);
    if (k < 3) continue;
    double lat_min = 1e18, lat_sp_max = 0;
    for (int j = 1; j < k; ++j) {
      double dt = (double)(sts[j] - sts[j - 1]) / 1e9;
      if (dt <= 0) continue;
      double sp = fabs((sy[j] - sy[j - 1]) / dt);
      if (sp > lat_sp_max) lat_sp_max = sp;
    }
    for (int j = 0; j < k; ++j)
      if (fabs(sy[j]) < lat_min) lat_min = fabs(sy[j]);
    if (lat_sp_max > thr_cross && lat_min < thr_lat) {
      snprintf(ev->name, sizeof(ev->name), "vru_cross");
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
      SP_RULE_ABI, "vru_cross", "1.0.0", init, eval, fini};
  return &d;
}
