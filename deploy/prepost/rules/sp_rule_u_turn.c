/* sp_rule_u_turn.c — L1: 掉头: 3s 内航向变化 > 120 度
 * 出处: ad-data-engine events.py ego_events u_turn (详设 §1.2: 3s 内 >120 度) — 车端流式移植 (M9a 计划 Task 3).
 * THR 段: [u_turn] (thr.conf); 默认值 = v0 种子, 上车前须分位数重标.
 */
#include "sp_rule_util.h"
#include <stdio.h>
#include <stdlib.h>

static double win_s = 3.0;
static double deg_thr = 120.0;

static int init(const char* kv) {
  win_s = ru_thr(kv, "win_s", 3.0);
  deg_thr = ru_thr(kv, "deg_thr", 120.0);
  return 0;
}

static int eval(const sp_frame_ctx* c, sp_rule_event* ev) {
  if (!c || !ev || c->status != 0) return 0;  /* LATCH/旧帧不评 */
  // 历史序列 chronological (a[0]=最老, a[n-1]=当前; ctx.hist_* 是新→旧)
  double a[SP_RULE_HIST];
  int64_t tsq[SP_RULE_HIST];
  int n = (int)c->n_hist;
  for (uint32_t i = 0; i < c->n_hist; ++i) {
    a[n - 1 - i] = (c->hist_ego[i].heading);
    tsq[n - 1 - i] = c->hist_ts[i];
  }
  if (n < 2) return 0;
  int64_t t_end = tsq[n - 1];
  int start = n - 1;
  while (start > 0 && (t_end - tsq[start - 1]) <= win_s * 1e9)
    --start;
  double sum = 0;
  for (int i = start + 1; i < n; ++i)
    sum += ru_wrap_pi(a[i] - a[i - 1]);
  if (fabs(sum) * 180.0 / M_PI > deg_thr) {
    snprintf(ev->name, sizeof(ev->name), "u_turn");
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
      SP_RULE_ABI, "u_turn", "1.0.0", init, eval, fini};
  return &d;
}
