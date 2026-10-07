// test_rule_load.cpp — M9a Task 1 RED/GREEN: 规则插件 ABI + 加载器.
// 断言见 main() PASS 1..6; 退出码 0 = 全过.
// 规则夹具在板上运行时用 g++ -shared 现编 (模板 + 坏 ABI + 额外规则).
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include "sp_rule.h"
#include "sp_ruleload.h"

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
  return passed == 4 ? 0 : 1;
}
