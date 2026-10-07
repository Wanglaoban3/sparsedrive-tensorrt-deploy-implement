// test_rule_load.cpp — M9a Task 1/3 RED/GREEN: 规则插件 ABI + 加载器 +
// L1 规则移植 + 去重/配额. 断言见 main() PASS 1..8; 退出码 0 = 全过.
// 规则夹具在板上运行时用 g++ -shared 现编 (模板 + 坏 ABI + 额外规则 + L1 组).
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include "sp_rule.h"
#include "sp_ruleload.h"
#include "sp_quota.h"
#include "sp_egoring.h"

static const char* kBadAbiSrc =
    "#include \"sp_rule.h\"\n"
    "static int ev0(const sp_frame_ctx*, sp_rule_event*) { return 0; }\n"
    "extern \"C\" const sp_rule_desc* sp_rule_query(void) {\n"
    "  static const sp_rule_desc d = {999, \"bad_abi\", \"1.0.0\", 0, ev0, 0};\n"
    "  return &d;\n"
    "}\n";

static const char* kExtraSrc =
    "#include \"sp_rule.h\"\n"
    "#include <string.h>\n"
    "#include <stdio.h>\n"
    "static int ev(const sp_frame_ctx* c, sp_rule_event* e) {\n"
    "  if (!c || c->status != 0) return 0;\n"
    "  if (c->ego.speed > 18.0f) {\n"
    "    snprintf(e->name, sizeof(e->name), \"demo_extra\");\n"
    "    e->strength = 1.0f;\n"
    "    return 1;\n"
    "  }\n"
    "  return 0;\n"
    "}\n"
    "extern \"C\" const sp_rule_desc* sp_rule_query(void) {\n"
    "  static const sp_rule_desc d = {SP_RULE_ABI, \"demo_extra\", \"1.0.0\","
    " 0, ev, 0};\n"
    "  return &d;\n"
    "}\n";

static int compile_src(const char* src, const char* out) {
  FILE* f = fopen("/tmp/m9a_t1/_fix.c", "w");
  if (!f) return -1;
  fputs(src, f);
  fclose(f);
  char cmd[512];
  snprintf(cmd, sizeof(cmd),
           "g++ -shared -fPIC -I/opt/m0/trt-dev/prepost "
           "/tmp/m9a_t1/_fix.c -o %s 2>/dev/null", out);
  return system(cmd) == 0 ? 0 : -1;
}

int main() {
  int passed = 0;
  system("rm -rf /tmp/m9a_t1 && mkdir -p /tmp/m9a_t1/rules");
  FILE* f = fopen("/tmp/m9a_t1/thr.conf", "w");
  fputs("[demo_speed_high]\nspeed_high=15.0\n", f);
  fclose(f);
  // 模板规则夹具 (交付物本体): 编进 rules 目录
  if (system("g++ -shared -fPIC -I/opt/m0/trt-dev/prepost "
             "/opt/m0/trt-dev/prepost/sp_rule_template.c "
             "-o /tmp/m9a_t1/rules/demo_speed_high.so 2>/dev/null") != 0) {
    printf("FAIL 0 (template compile)\n");
    return 1;
  }

  // ---- Case 1: 模板加载 + eval 命中 (THR 段注入 speed_high=15) ----
  {
    sp_rule_desc* rules[8] = {0};
    char err[256] = {0};
    int n = rule_scan("/tmp/m9a_t1/rules", "/tmp/m9a_t1/thr.conf", rules, 8,
                      err, sizeof(err));
    sp_frame_ctx c;
    memset(&c, 0, sizeof(c));
    c.abi_ver = SP_RULE_ABI;
    c.ego.speed = 16.0f;
    int ok = (n == 1 && rules[0] &&
              strcmp(rules[0]->name, "demo_speed_high") == 0);
    sp_rule_event ev;
    memset(&ev, 0, sizeof(ev));
    int ne = ok ? rules[0]->eval(&c, &ev) : -1;
    ok = ok && ne == 1 && strcmp(ev.name, "demo_speed_high") == 0 &&
         ev.strength > 0.799f && ev.strength < 0.801f;
    c.ego.speed = 10.0f;
    int ne2 = ok ? rules[0]->eval(&c, &ev) : -1;
    ok = ok && ne2 == 0;
    if (ok) { printf("PASS 1 (load+eval+thr)\n"); ++passed; }
    else printf("FAIL 1 (n=%d ne=%d ne2=%d name=%s str=%f)\n", n, ne, ne2,
                ev.name, (double)ev.strength);
    rule_unload_all();
  }

  // ---- Case 2: 坏 ABI 被拒且不计入 ----
  {
    if (compile_src(kBadAbiSrc, "/tmp/m9a_t1/rules/bad_abi.so") != 0) {
      printf("FAIL 2 (fixture compile)\n");
      return 1;
    }
    sp_rule_desc* rules[8] = {0};
    char err[256] = {0};
    int n = rule_scan("/tmp/m9a_t1/rules", "/tmp/m9a_t1/thr.conf", rules, 8,
                      err, sizeof(err));
    int ok = (n == 1);  // 只有 demo_speed_high; bad_abi 被 REJECT
    for (int i = 0; i < n; ++i)
      if (strcmp(rules[i]->name, "bad_abi") == 0) ok = 0;
    if (ok) { printf("PASS 2 (bad abi rejected)\n"); ++passed; }
    else printf("FAIL 2 (n=%d)\n", n);
    rule_unload_all();
  }

  // ---- Case 3: 垃圾文件 (非 ELF) 被拒, 好规则照常 ----
  {
    f = fopen("/tmp/m9a_t1/rules/garbage.so", "w");
    fputs("this is not an elf", f);
    fclose(f);
    sp_rule_desc* rules[8] = {0};
    char err[256] = {0};
    int n = rule_scan("/tmp/m9a_t1/rules", "/tmp/m9a_t1/thr.conf", rules, 8,
                      err, sizeof(err));
    int ok = (n == 1 &&
              strcmp(rules[0]->name, "demo_speed_high") == 0);
    if (ok) { printf("PASS 3 (garbage rejected, host alive)\n"); ++passed; }
    else printf("FAIL 3 (n=%d)\n", n);
    rule_unload_all();
  }

  // ---- Case 4: 新规则落目录 → rule_rescan() 加载 (drop-in) ----
  {
    if (compile_src(kExtraSrc, "/tmp/m9a_t1/rules/demo_extra.so") != 0) {
      printf("FAIL 4 (fixture compile)\n");
      return 1;
    }
    int n = rule_rescan("/tmp/m9a_t1/rules", "/tmp/m9a_t1/thr.conf");
    int ok = (n == 2);
    sp_frame_ctx c;
    memset(&c, 0, sizeof(c));
    c.abi_ver = SP_RULE_ABI;
    c.ego.speed = 19.0f;
    int fired = 0;
    for (int i = 0; i < n && ok; ++i) {
      sp_rule_event ev;
      memset(&ev, 0, sizeof(ev));
      if (rule_at(i)->eval(&c, &ev) > 0 &&
          strcmp(ev.name, "demo_extra") == 0) ++fired;
    }
    ok = ok && fired == 1;
    if (ok) { printf("PASS 4 (rescan picks up drop-in)\n"); ++passed; }
    else printf("FAIL 4 (n=%d fired=%d)\n", n, fired);
    rule_unload_all();
  }

  printf("== %d/4 ==\n", passed);
  int ok4 = passed == 4;

  // ================= Task 3: L1 规则移植 + 去重/配额 =================
  int passed3 = 0;
  // ---- Case 5: hard_brake 合成减速序列触发, strength = min(1, 4/5) ----
  {
    // 编译 L1 规则组 (deploy/prepost/rules/*.c 已被 driver push)
    system("rm -rf /tmp/m9a_t3 && mkdir -p /tmp/m9a_t3/rules");
    int n1 = system("for f in /opt/m0/trt-dev/prepost/rules/*.c; do "
                    "g++ -shared -fPIC -I/opt/m0/trt-dev/prepost $f "
                    "-o /tmp/m9a_t3/rules/$(basename $f .c).so || exit 1; "
                    "done");
    f = fopen("/tmp/m9a_t3/thr.conf", "w");
    fputs("[hard_brake]\nacc_thr=-3.0\ndur_s=0.5\n", f);
    fclose(f);
    if (n1 != 0) {
      printf("FAIL 5 (rule compile)\n");
      return 1;
    }
    int n = rule_rescan("/tmp/m9a_t3/rules", "/tmp/m9a_t3/thr.conf");
    // 合成减速: dt=0.1s, speed 每帧 -0.4 (acc=-4). 车前向 = R 第 1 列
    // (Task 4 90° 装转口径): 位移放 col1 (l2g[7]), heading=atan2(1,0)=pi/2
    Egoring er;
    int hb_idx = -1;
    sp_rule_desc* hb = 0;
    for (int i = 0; i < n; ++i)
      if (strcmp(rule_at(i)->name, "hard_brake") == 0) hb = rule_at(i);
    if (!hb) {
      printf("FAIL 5 (hard_brake not loaded, n=%d)\n", n);
      return 1;
    }
    double y = 0;
    for (int i = 0; i < 12; ++i) {
      double l2g[16] = {0};
      l2g[0] = 1; l2g[5] = 1; l2g[10] = 1; l2g[15] = 1;
      l2g[7] = y;
      TrigFrameLite fr{100 + i, 0, (int64_t)(1e9 * i * 0.1), l2g};
      er.push_frame(fr, 0, 0);
      y += (5.0 - 0.4 * i) * 0.1;  // 本帧速度决定本帧位移
      sp_frame_ctx ctx;
      er.build_ctx(&ctx, 0);
      sp_rule_event ev;
      memset(&ev, 0, sizeof(ev));
      if (hb->eval(&ctx, &ev) > 0 && hb_idx < 0) {
        hb_idx = i;
        if (strcmp(ev.name, "hard_brake") != 0 ||
            ev.strength < 0.799f || ev.strength > 0.801f) {
          printf("FAIL 5 (name=%s str=%f)\n", ev.name, (double)ev.strength);
          return 1;
        }
      }
    }
    if (hb_idx >= 0) { printf("PASS 5 (hard_brake at i=%d, str=0.8)\n", hb_idx); ++passed3; }
    else printf("FAIL 5 (never fired)\n");
    rule_unload_all();
  }

  // ---- Case 6: 配额/去重 (sp_quota.h): 冷却窗 / 事件类型配额 / 全局预算 ----
  {
    Quota q;  // 默认 dedup 5s, per-event 12/min, global 15/min
    int64_t t0 = (int64_t)1e9 * 1000;
    int a1 = q.allow(1, t0, "hard_brake");
    int a2 = q.allow(2, t0 + (int64_t)3e9, "hard_brake");   // 冷却窗内 → 吞
    int a3 = q.allow(3, t0 + (int64_t)3e9, "hard_accel");   // 异名 → 过
    Quota g;
    g.global_per_min = 2;
    int g1 = g.allow(1, t0, "e1");
    int g2 = g.allow(2, t0 + (int64_t)1e9, "e2");
    int g3 = g.allow(3, t0 + (int64_t)2e9, "e3");           // 全局超 → 吞
    Quota p;
    p.per_event_per_min = 2;
    p.dedup_window_s = 1.0;
    int p1 = p.allow(1, t0, "same");
    int p2 = p.allow(2, t0 + (int64_t)2e9, "same");         // 出冷却窗 → 过
    int p3 = p.allow(3, t0 + (int64_t)4e9, "same");         // 类型配额 → 吞
    int ok = a1 == 1 && a2 == 0 && a3 == 1 && q.suppressed == 1 &&
             g1 == 1 && g2 == 1 && g3 == 0 && g.suppressed == 1 &&
             p1 == 1 && p2 == 1 && p3 == 0 && p.suppressed == 1;
    if (ok) { printf("PASS 6 (quota dedup+per-event+global)\n"); ++passed3; }
    else printf("FAIL 6 (a=%d,%d,%d,%d g=%d,%d,%d,%llu p=%d,%d,%d,%llu)\n",
                a1, a2, a3, q.suppressed, g1, g2, g3,
                (unsigned long long)g.suppressed, p1, p2, p3,
                (unsigned long long)p.suppressed);
  }

  // ---- Case 7: 场景边界 l2g 阶跃 → ego 清零不差分, 无假事件 ----
  {
    Egoring er;
    // 场景 0: 匀速直行 8 帧 (speed=5); 车前向 = R 第 1 列 (col1 口径)
    double y = 0;
    for (int i = 0; i < 8; ++i) {
      double l2g[16] = {0};
      l2g[0] = 1; l2g[5] = 1; l2g[10] = 1; l2g[15] = 1;
      l2g[7] = y;
      TrigFrameLite fr{200 + i, 0, (int64_t)(1e9 * i * 0.1), l2g};
      er.push_frame(fr, 0, 0);
      y += 0.5;
    }
    sp_frame_ctx pre;
    er.build_ctx(&pre, 0);
    // 预检: 夹具本身有效 (跳变前速度确为 5, 防夹具静默失效)
    if (pre.ego.speed < 4.9f || pre.ego.speed > 5.1f) {
      printf("FAIL 7 (pre speed=%f, fixture broken)\n",
             (double)pre.ego.speed);
      rule_unload_all();
      return 1;
    }
    // 场景 1 首帧: l2g 阶跃 1000m (跨场景重定位)
    double lg2[16] = {0};
    lg2[0] = 1; lg2[5] = 1; lg2[10] = 1; lg2[15] = 1; lg2[7] = y + 1000;
    TrigFrameLite fr{208, 1, (int64_t)(1e9 * 8 * 0.1), lg2};
    er.push_frame(fr, 0, 0);
    sp_frame_ctx ctx;
    er.build_ctx(&ctx, 0);
    int ok = ctx.n_hist == 1 && ctx.ego.speed == 0 && ctx.ego.acc == 0 &&
             ctx.ego.yaw_rate == 0;
    if (ok) { printf("PASS 7 (scene jump neutralized)\n"); ++passed3; }
    else printf("FAIL 7 (n_hist=%u speed=%f)\n", ctx.n_hist,
                (double)ctx.ego.speed);
  }

  // ---- Case 8: LATCH 帧 (status!=0) 规则不评 ----
  {
    int n = rule_rescan("/tmp/m9a_t3/rules", "/tmp/m9a_t3/thr.conf");
    sp_rule_desc* hb = 0;
    for (int i = 0; i < n; ++i)
      if (strcmp(rule_at(i)->name, "hard_brake") == 0) hb = rule_at(i);
    Egoring er;
    double y = 0;
    for (int i = 0; i < 12; ++i) {  // 同 case 5 的减速序列 (col1 口径)
      double l2g[16] = {0};
      l2g[0] = 1; l2g[5] = 1; l2g[10] = 1; l2g[15] = 1;
      l2g[7] = y;
      TrigFrameLite fr{300 + i, 0, (int64_t)(1e9 * i * 0.1), l2g};
      er.push_frame(fr, 0, 0);
      y += (5.0 - 0.4 * i) * 0.1;
    }
    sp_frame_ctx ctx;
    er.build_ctx(&ctx, 0);
    ctx.status = 2;  // LATCH
    sp_rule_event ev;
    memset(&ev, 0, sizeof(ev));
    int ne = hb->eval(&ctx, &ev);
    if (ne == 0) { printf("PASS 8 (latch not evaluated)\n"); ++passed3; }
    else printf("FAIL 8 (ne=%d)\n", ne);
    rule_unload_all();
  }

  printf("== task1 %d/4, task3 %d/4 ==\n", ok4, passed3);
  return (ok4 && passed3 == 4) ? 0 : 1;
}
