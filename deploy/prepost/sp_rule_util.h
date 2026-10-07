/* sp_rule_util.h — 规则插件 SDK 静态内联助手 (M9a).
 * 每 TU 一份拷贝 (static inline), 无共享状态; 规则 = sp_rule.h + 本头.
 * 语义出处: ad-data-engine src/mine/events.py (L1 v0), 车端流式改写见计划.
 */
#ifndef SP_RULE_UTIL_H_
#define SP_RULE_UTIL_H_
#include <math.h>
#include <string.h>

#include "sp_rule.h"

/* nuScenes det label 集 (执行口径以 eval_t6_mini_v8.py class 序为准) */
static inline int ru_is_vehicle(int32_t l) {
  return l == 0 || l == 1 || l == 2 || l == 3 || l == 4; /* car/truck/cv/bus/trailer */
}
static inline int ru_is_vru(int32_t l) {
  return l == 6 || l == 7 || l == 8; /* motorcycle/bicycle/pedestrian */
}
static inline int ru_is_cone(int32_t l) { return l == 9; }

static inline double ru_wrap_pi(double a) {
  while (a > M_PI) a -= 2 * M_PI;
  while (a < -M_PI) a += 2 * M_PI;
  return a;
}

/* 历史序列 (chronological: vals[0]=最老, vals[n-1]=当前) 上的持续判定.
 * ts_chrono = 与 vals 同序的 ts 数组 (ctx.hist_* 是新→旧, 规则负责翻转).
 * mask = (ge ? v >= thr : v <= thr); run 时间 = 样本间 ts 实差和 >= dur_s. */
static inline int ru_sustained(const double* vals, const int64_t* ts_chrono,
                               int n, double thr, int ge, double dur_s) {
  if (n < 2) return 0;
  int run = 0;
  double t = 0;
  for (int i = 1; i < n; ++i) {
    int m = ge ? vals[i] >= thr : vals[i] <= thr;
    double dt = (double)(ts_chrono[i] - ts_chrono[i - 1]) / 1e9;
    if (dt < 0) dt = 0;
    if (m) {
      run += 1;
      t += dt;
    } else {
      run = 0;
      t = 0;
    }
    if (run >= 1 && t >= dur_s) return 1;
  }
  return 0;
}

/* 按 id 在历史里收集该目标的时序 (chronological), 返回样本数.
 * s_* 输出数组容量 n_hist; 只有 id>=0 的帧入列 (id 缺帧跳过不中断). */
static inline int ru_id_series(const sp_frame_ctx* c, int32_t id,
                               double* sx, double* sy, double* svx,
                               double* svy, double* sts) {
  if (!c || id < 0) return 0;
  int k = 0;
  for (uint32_t i = c->n_hist; i-- > 0;) {  // 最老 → 当前
    for (uint32_t j = 0; j < c->hist_n[i]; ++j) {
      const sp_trk* d = &c->hist_det[i][j];
      if (d->id == id) {
        if (sx) sx[k] = d->x;
        if (sy) sy[k] = d->y;
        if (svx) svx[k] = d->vx;
        if (svy) svy[k] = d->vy;
        if (sts) sts[k] = (double)c->hist_ts[i];
        ++k;
        break;
      }
    }
  }
  return k;
}

/* 当前帧同廊道最近前车 (events.py agent_events 的 ego 投影式; det 已在
 * ego 系: x 前向, y 左向). 返回下标或 -1. */
static inline int ru_pick_lead(const sp_frame_ctx* c, double lon_min,
                               double lon_max, double lat_max) {
  int best = -1;
  double best_d = 1e18;
  for (uint32_t i = 0; i < c->n_det; ++i) {
    const sp_trk* d = &c->det[i];
    if (!ru_is_vehicle(d->label)) continue;
    if (d->x > lon_min && d->x < lon_max && fabs((double)d->y) < lat_max) {
      double dist = sqrt((double)d->x * d->x + (double)d->y * d->y);
      if (dist < best_d) {
        best_d = dist;
        best = (int)i;
      }
    }
  }
  return best;
}

/* 简易 k=v 段解析: 取 key 的 double, 无则 dflt */
static inline double ru_thr(const char* kv, const char* key, double dflt) {
  if (!kv) return dflt;
  size_t klen = strlen(key);
  const char* p = kv;
  while (*p) {
    const char* eol = strchr(p, '\n');
    size_t len = eol ? (size_t)(eol - p) : strlen(p);
    if (len > klen + 1 && strncmp(p, key, klen) == 0 && p[klen] == '=') {
      char buf[64];
      size_t vlen = len - klen - 1;
      if (vlen >= sizeof(buf)) vlen = sizeof(buf) - 1;
      memcpy(buf, p + klen + 1, vlen);
      buf[vlen] = 0;
      return atof(buf);
    }
    p = eol ? eol + 1 : p + len;
  }
  return dflt;
}

#endif /* SP_RULE_UTIL_H_ */
