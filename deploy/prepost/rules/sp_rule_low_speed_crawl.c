/* sp_rule_low_speed_crawl.c — L1: 低速蠕行: 窗口均速 < 2.0 m/s 持续窗口 5s
 * 出处: ad-data-engine events.py ego_events low_speed_crawl (持续>=5s 口径=详设 §1.1) — 车端流式移植 (M9a 计划 Task 3).
 * THR 段: [low_speed_crawl] (thr.conf); 默认值 = v0 种子, 上车前须分位数重标.
 */
#include "sp_rule_util.h"
#include <stdio.h>
#include <stdlib.h>

static double thr_speed = 2.0;
static double win_s = 5.0;

static int init(const char* kv) {
  thr_speed = ru_thr(kv, "speed_thr", 2.0);
  win_s = ru_thr(kv, "win_s", 5.0);
  return 0;
}

static int eval(const sp_frame_ctx* c, sp_rule_event* ev) {
  if (!c || !ev || c->status != 0) return 0;  /* LATCH/旧帧不评 */
  // 历史序列 chronological (a[0]=最老, a[n-1]=当前; ctx.hist_* 是新→旧)
  double a[SP_RULE_HIST];
  int64_t tsq[SP_RULE_HIST];
  int n = (int)c->n_hist;
  for (uint32_t i = 0; i < c->n_hist; ++i) {
    a[n - 1 - i] = (c->hist_ego[i].speed);
    tsq[n - 1 - i] = c->hist_ts[i];
  }
  if (n < 2) return 0;
  // 尾窗: ts >= ts_last - win_s 的样本; 覆盖不足 win_s 不触发
  int64_t t_end = tsq[n - 1];
  int start = n - 1;
  while (start > 0 && (t_end - tsq[start - 1]) <= win_s * 1e9)
    --start;
  if ((double)(t_end - tsq[start]) / 1e9 < win_s) return 0;
  double sum = 0;
  for (int i = start; i < n; ++i) sum += a[i];
  double mean = sum / (n - start);
  if (mean < thr_speed) {
    snprintf(ev->name, sizeof(ev->name), "low_speed_crawl");
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
      SP_RULE_ABI, "low_speed_crawl", "1.0.0", init, eval, fini};
  return &d;
}
