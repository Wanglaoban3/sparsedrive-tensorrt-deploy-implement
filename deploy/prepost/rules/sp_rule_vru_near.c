/* sp_rule_vru_near.c — L1: VRU 近距: 行人/骑行者 纵向 (-5,30) 横向 < 2m
 * 出处: ad-data-engine events.py agent_events vru_near — 车端流式移植 (M9a 计划 Task 3).
 * THR 段: [vru_near] (thr.conf); 默认值 = v0 种子, 上车前须分位数重标.
 */
#include "sp_rule_util.h"
#include <stdio.h>
#include <stdlib.h>

static double lon_min = -5.0;
static double lon_max = 30.0;
static double thr_lat = 2.0;

static int init(const char* kv) {
  lon_min = ru_thr(kv, "lon_min", -5.0);
  lon_max = ru_thr(kv, "lon_max", 30.0);
  thr_lat = ru_thr(kv, "lat_thr", 2.0);
  return 0;
}

static int eval(const sp_frame_ctx* c, sp_rule_event* ev) {
  if (!c || !ev || c->status != 0) return 0;  /* LATCH/旧帧不评 */
  for (uint32_t i = 0; i < c->n_det; ++i) {
    const sp_trk* d = &c->det[i];
    if (!ru_is_vru(d->label)) continue;
    if (d->x > lon_min && d->x < lon_max && fabs((double)d->y) < thr_lat) {
      snprintf(ev->name, sizeof(ev->name), "vru_near");
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
      SP_RULE_ABI, "vru_near", "1.0.0", init, eval, fini};
  return &d;
}
