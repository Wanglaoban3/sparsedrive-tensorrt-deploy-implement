/* sp_rule_sharp_lat.c — L1: 急横: |横向加速度| > 2.0 m/s^2 持续 >= 0.3s
 * 出处: ad-data-engine events.py ego_events sharp_lat — 车端流式移植 (M9a 计划 Task 3).
 * THR 段: [sharp_lat] (thr.conf); 默认值 = v0 种子, 上车前须分位数重标.
 */
#include "sp_rule_util.h"
#include <stdio.h>
#include <stdlib.h>

static double thr_lat = 2.0;
static double thr_dur = 0.3;

static int init(const char* kv) {
  thr_lat = ru_thr(kv, "lat_thr", 2.0);
  thr_dur = ru_thr(kv, "dur_s", 0.3);
  return 0;
}

static int eval(const sp_frame_ctx* c, sp_rule_event* ev) {
  if (!c || !ev || c->status != 0) return 0;  /* LATCH/旧帧不评 */
  // 历史序列 chronological (a[0]=最老, a[n-1]=当前; ctx.hist_* 是新→旧)
  double a[SP_RULE_HIST];
  int64_t tsq[SP_RULE_HIST];
  int n = (int)c->n_hist;
  for (uint32_t i = 0; i < c->n_hist; ++i) {
    a[n - 1 - i] = (fabs((double)c->hist_ego[i].speed * (double)c->hist_ego[i].yaw_rate));
    tsq[n - 1 - i] = c->hist_ts[i];
  }
  double mx = 0;
  for (int i = 0; i < n; ++i)
    if (a[i] > mx) mx = a[i];
  if (n >= 2 && ru_sustained(a, tsq, n, thr_lat, 1, thr_dur)) {
    snprintf(ev->name, sizeof(ev->name), "sharp_lat");
    ev->strength = (float)(mx / 3.5);
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
      SP_RULE_ABI, "sharp_lat", "1.0.0", init, eval, fini};
  return &d;
}
