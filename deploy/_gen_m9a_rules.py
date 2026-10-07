# -*- coding: utf-8 -*-
"""一次性生成 deploy/prepost/rules/sp_rule_*.c ×14 (M9a Task 3).
规则 = events.py L1 v0 的车端流式移植; 用后即删."""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "deploy", "prepost", "rules")
os.makedirs(OUT, exist_ok=True)

HEAD = """/* sp_rule_{name}.c — L1: {desc}
 * 出处: ad-data-engine events.py {origin} — 车端流式移植 (M9a 计划 Task 3).
 * THR 段: [{name}] (thr.conf); 默认值 = v0 种子, 上车前须分位数重标.
 */
#include "sp_rule_util.h"
#include <stdio.h>
#include <stdlib.h>

"""

FOOT = """static void fini(void) {{}}

#ifdef __cplusplus
extern "C"
#endif
const sp_rule_desc* sp_rule_query(void) {{
  static const sp_rule_desc d = {{
      SP_RULE_ABI, "{name}", "1.0.0", init, eval, fini}};
  return &d;
}}
"""

CHRON = """  // 历史序列 chronological (a[0]=最老, a[n-1]=当前; ctx.hist_* 是新→旧)
  double a[SP_RULE_HIST];
  int64_t tsq[SP_RULE_HIST];
  int n = (int)c->n_hist;
  for (uint32_t i = 0; i < c->n_hist; ++i) {{
    a[n - 1 - i] = ({expr});
    tsq[n - 1 - i] = c->hist_ts[i];
  }}
"""


def chrono(expr):
    return CHRON.format(expr=expr)


RULES = {}

RULES["hard_brake"] = dict(
    desc="急刹: 纵向加速度 <= %.1f 持续 >= %.1fs" % (-3.0, 0.5),
    origin="ego_events hard_brake",
    thr='static double thr_acc = -3.0;\nstatic double thr_dur = 0.5;',
    init='  thr_acc = ru_thr(kv, "acc_thr", -3.0);\n'
         '  thr_dur = ru_thr(kv, "dur_s", 0.5);',
    body=chrono("c->hist_ego[i].acc") + """  double mn = 0;
  for (int i = 0; i < n; ++i)
    if (a[i] < mn) mn = a[i];
  if (n >= 2 && ru_sustained(a, tsq, n, thr_acc, 0, thr_dur)) {
    snprintf(ev->name, sizeof(ev->name), "hard_brake");
    ev->strength = (float)(-mn / 5.0);
    if (ev->strength > 1.0f) ev->strength = 1.0f;
    return 1;
  }
  return 0;""")

RULES["hard_accel"] = dict(
    desc="急加速: 纵向加速度 >= %.1f 持续 >= %.1fs" % (2.5, 0.5),
    origin="ego_events hard_accel",
    thr='static double thr_acc = 2.5;\nstatic double thr_dur = 0.5;',
    init='  thr_acc = ru_thr(kv, "acc_thr", 2.5);\n'
         '  thr_dur = ru_thr(kv, "dur_s", 0.5);',
    body=chrono("c->hist_ego[i].acc") + """  double mx = 0;
  for (int i = 0; i < n; ++i)
    if (a[i] > mx) mx = a[i];
  if (n >= 2 && ru_sustained(a, tsq, n, thr_acc, 1, thr_dur)) {
    snprintf(ev->name, sizeof(ev->name), "hard_accel");
    ev->strength = (float)(mx / 5.0);
    if (ev->strength > 1.0f) ev->strength = 1.0f;
    return 1;
  }
  return 0;""")

RULES["low_speed_crawl"] = dict(
    desc="低速蠕行: 窗口均速 < %.1f m/s 持续窗口 %.0fs" % (2.0, 5),
    origin="ego_events low_speed_crawl (持续>=5s 口径=详设 §1.1)",
    thr='static double thr_speed = 2.0;\nstatic double win_s = 5.0;',
    init='  thr_speed = ru_thr(kv, "speed_thr", 2.0);\n'
         '  win_s = ru_thr(kv, "win_s", 5.0);',
    body=chrono("c->hist_ego[i].speed") + """  if (n < 2) return 0;
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
  return 0;""")

RULES["reverse"] = dict(
    desc="倒车: 纵向速度 <= %.1f m/s 持续 >= %.1fs" % (-0.5, 0.3),
    origin="ego_events reverse",
    thr='static double thr_speed = -0.5;\nstatic double thr_dur = 0.3;',
    init='  thr_speed = ru_thr(kv, "speed_thr", -0.5);\n'
         '  thr_dur = ru_thr(kv, "dur_s", 0.3);',
    body=chrono("c->hist_ego[i].speed") + """  if (n >= 2 && ru_sustained(a, tsq, n, thr_speed, 0, thr_dur)) {
    snprintf(ev->name, sizeof(ev->name), "reverse");
    ev->strength = 1.0f;
    return 1;
  }
  return 0;""")

RULES["sharp_turn"] = dict(
    desc="急转: |yaw_rate| > %.1f rad/s 持续 >= %.1fs" % (0.3, 0.5),
    origin="ego_events sharp_turn",
    thr='static double thr_yaw = 0.3;\nstatic double thr_dur = 0.5;',
    init='  thr_yaw = ru_thr(kv, "yaw_thr", 0.3);\n'
         '  thr_dur = ru_thr(kv, "dur_s", 0.5);',
    body=chrono("fabs((double)c->hist_ego[i].yaw_rate)") + """  double mx = 0;
  for (int i = 0; i < n; ++i)
    if (a[i] > mx) mx = a[i];
  if (n >= 2 && ru_sustained(a, tsq, n, thr_yaw, 1, thr_dur)) {
    snprintf(ev->name, sizeof(ev->name), "sharp_turn");
    ev->strength = (float)(mx / 0.6);
    if (ev->strength > 1.0f) ev->strength = 1.0f;
    return 1;
  }
  return 0;""")

RULES["sharp_lat"] = dict(
    desc="急横: |横向加速度| > %.1f m/s^2 持续 >= %.1fs" % (2.0, 0.3),
    origin="ego_events sharp_lat",
    thr='static double thr_lat = 2.0;\nstatic double thr_dur = 0.3;',
    init='  thr_lat = ru_thr(kv, "lat_thr", 2.0);\n'
         '  thr_dur = ru_thr(kv, "dur_s", 0.3);',
    body=chrono("fabs((double)c->hist_ego[i].speed * "
                "(double)c->hist_ego[i].yaw_rate)") + """  double mx = 0;
  for (int i = 0; i < n; ++i)
    if (a[i] > mx) mx = a[i];
  if (n >= 2 && ru_sustained(a, tsq, n, thr_lat, 1, thr_dur)) {
    snprintf(ev->name, sizeof(ev->name), "sharp_lat");
    ev->strength = (float)(mx / 3.5);
    if (ev->strength > 1.0f) ev->strength = 1.0f;
    return 1;
  }
  return 0;""")

RULES["u_turn"] = dict(
    desc="掉头: %.0fs 内航向变化 > %.0f 度" % (3, 120),
    origin="ego_events u_turn (详设 §1.2: 3s 内 >120 度)",
    thr='static double win_s = 3.0;\nstatic double deg_thr = 120.0;',
    init='  win_s = ru_thr(kv, "win_s", 3.0);\n'
         '  deg_thr = ru_thr(kv, "deg_thr", 120.0);',
    body=chrono("c->hist_ego[i].heading") + """  if (n < 2) return 0;
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
  return 0;""")

RULES["lead_hard_brake"] = dict(
    desc="前车急刹: 最近前车减速 <= %.1f, 间距 < %.0fm" % (-3.0, 20),
    origin="agent_events lead_hard_brake (v0 口径=max巡航速>2 替代 GT v0)",
    thr='static double thr_dist = 20.0;\nstatic double thr_decel = -3.0;\n'
        'static double thr_v0 = 2.0;',
    init='  thr_dist = ru_thr(kv, "dist_thr", 20.0);\n'
         '  thr_decel = ru_thr(kv, "decel_thr", -3.0);\n'
         '  thr_v0 = ru_thr(kv, "v0_thr", 2.0);',
    body="""  int li = ru_pick_lead(c, 2.0, 40.0, 2.5);
  if (li < 0) return 0;
  double sx[SP_RULE_HIST], svx[SP_RULE_HIST];
  int k = ru_id_series(c, c->det[li].id, sx, 0, svx, 0, 0);
  if (k < 3) return 0;
  double mn = svx[0], vmax = svx[0];
  for (int i = 1; i < k; ++i) {
    if (svx[i] < mn) mn = svx[i];
    if (svx[i] > vmax) vmax = svx[i];
  }
  double d0 = sqrt((double)c->det[li].x * c->det[li].x +
                   (double)c->det[li].y * c->det[li].y);
  if (d0 < thr_dist && mn <= thr_decel && vmax > thr_v0) {
    snprintf(ev->name, sizeof(ev->name), "lead_hard_brake");
    ev->strength = (float)(-mn / 5.0);
    if (ev->strength > 1.0f) ev->strength = 1.0f;
    return 1;
  }
  return 0;""")

RULES["slow_lead"] = dict(
    desc="慢车压道: 前车均速 < %.1f 且 ego > %.1f, 间距 < %.0fm" %
         (2.0, 5.0, 25),
    origin="agent_events slow_lead",
    thr='static double thr_dist = 25.0;\nstatic double thr_speed = 2.0;\n'
        'static double thr_ego = 5.0;',
    init='  thr_dist = ru_thr(kv, "dist_thr", 25.0);\n'
         '  thr_speed = ru_thr(kv, "speed_thr", 2.0);\n'
         '  thr_ego = ru_thr(kv, "ego_thr", 5.0);',
    body="""  int li = ru_pick_lead(c, 2.0, 40.0, 2.5);
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
  return 0;""")

RULES["stationary_approach"] = dict(
    desc="静止目标逼近: 前车均速 < %.1f 且 TTC < %.1fs" % (0.5, 3.0),
    origin="agent_events stationary_approach",
    thr='static double thr_ttc = 3.0;\nstatic double thr_vlead = 0.5;\n'
        'static double thr_ego = 1.0;',
    init='  thr_ttc = ru_thr(kv, "ttc_thr", 3.0);\n'
         '  thr_vlead = ru_thr(kv, "vlead_thr", 0.5);\n'
         '  thr_ego = ru_thr(kv, "ego_thr", 1.0);',
    body="""  int li = ru_pick_lead(c, 2.0, 40.0, 2.5);
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
  return 0;""")

RULES["cut_in"] = dict(
    desc="加塞 cut-in: 邻道目标横穿进入自车道, 纵距 < %.0fm" % 20,
    origin="agent_events cut_in (现在时: 进入中)",
    thr='static double thr_dist = 20.0;\nstatic double lat_edge = 2.5;\n'
        'static double lat_in = 2.0;',
    init='  thr_dist = ru_thr(kv, "dist_thr", 20.0);\n'
         '  lat_edge = ru_thr(kv, "lat_edge", 2.5);\n'
         '  lat_in = ru_thr(kv, "lat_in", 2.0);',
    body="""  for (uint32_t i = 0; i < c->n_det; ++i) {
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
  return 0;""")

RULES["vru_near"] = dict(
    desc="VRU 近距: 行人/骑行者 纵向 (%.0f,%.0f) 横向 < %.0fm" %
         (-5, 30, 2.0),
    origin="agent_events vru_near",
    thr='static double lon_min = -5.0;\nstatic double lon_max = 30.0;\n'
        'static double thr_lat = 2.0;',
    init='  lon_min = ru_thr(kv, "lon_min", -5.0);\n'
         '  lon_max = ru_thr(kv, "lon_max", 30.0);\n'
         '  thr_lat = ru_thr(kv, "lat_thr", 2.0);',
    body="""  for (uint32_t i = 0; i < c->n_det; ++i) {
    const sp_trk* d = &c->det[i];
    if (!ru_is_vru(d->label)) continue;
    if (d->x > lon_min && d->x < lon_max && fabs((double)d->y) < thr_lat) {
      snprintf(ev->name, sizeof(ev->name), "vru_near");
      ev->strength = 1.0f;
      return 1;
    }
  }
  return 0;""")

RULES["vru_cross"] = dict(
    desc="VRU 横穿: 横向速度 > %.1f 且接近走廊 (|lat| < %.1f)" %
         (1.5, 3.0),
    origin="agent_events vru_cross",
    thr='static double lon_min = -5.0;\nstatic double lon_max = 30.0;\n'
        'static double thr_cross = 1.5;\nstatic double thr_lat = 3.0;',
    init='  lon_min = ru_thr(kv, "lon_min", -5.0);\n'
         '  lon_max = ru_thr(kv, "lon_max", 30.0);\n'
         '  thr_cross = ru_thr(kv, "cross_thr", 1.5);\n'
         '  thr_lat = ru_thr(kv, "lat_thr", 3.0);',
    body="""  for (uint32_t i = 0; i < c->n_det; ++i) {
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
  return 0;""")

RULES["construction_zone"] = dict(
    desc="施工/锥桶区: 锥桶最近距 < %.0fm" % 30,
    origin="agent_events construction_zone",
    thr='static double thr_dist = 30.0;',
    init='  thr_dist = ru_thr(kv, "dist_thr", 30.0);',
    body="""  double dmin = 1e18;
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
  return 0;""")

TPL_BODY = """static int init(const char* kv) {{
{init}
  return 0;
}}

static int eval(const sp_frame_ctx* c, sp_rule_event* ev) {{
  if (!c || !ev || c->status != 0) return 0;  /* LATCH/旧帧不评 */
{body}
}}

"""

for name, r in RULES.items():
    src = HEAD.format(name=name, desc=r["desc"], origin=r["origin"])
    src += r["thr"] + "\n\n"
    src += TPL_BODY.format(init=r["init"], body=r["body"])
    src += FOOT.format(name=name)
    with open(os.path.join(OUT, "sp_rule_%s.c" % name), "w",
              encoding="utf-8") as f:
        f.write(src)
    print("wrote sp_rule_%s.c" % name)
print("GEN_DONE")
