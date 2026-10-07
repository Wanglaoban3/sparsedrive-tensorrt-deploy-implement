/* sp_rule_template.c — M9a 规则插件模板 (drop-in 示范).
 * 拷贝本文件改 eval() 与 name 即为新规则; THR 段 = thr.conf 里的 [name].
 * 板上编译: g++ -shared -fPIC -I/opt/m0/trt-dev/prepost sp_rule_template.c \
 *           -o /usr/local/share/sp/rules/<name>.so
 * ABI 详见 sp_rule.h; 宿主加载纪律详见 docs/M9A_PLUGIN_GUIDE.md.
 */
#include "sp_rule.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 规则私有阈值 (init() 从自己的 [name] 段覆盖; 无段用默认) */
static float thr_speed_high = 12.0f;   /* m/s, demo: 超速 mining 触发 */

static int init(const char* kv_text) {
  /* k=v 行解析; 只认自己的键, 未知键忽略 */
  const char* p = kv_text;
  while (p && *p) {
    const char* eol = strchr(p, '\n');
    size_t len = eol ? (size_t)(eol - p) : strlen(p);
    if (len > 0 && (size_t)len < 256) {
      char line[256];
      memcpy(line, p, len);
      line[len] = 0;
      char* eq = strchr(line, '=');
      if (eq) {
        *eq = 0;
        if (strcmp(line, "speed_high") == 0)
          thr_speed_high = (float)atof(eq + 1);
      }
    }
    p = eol ? eol + 1 : 0;
  }
  return 0;
}

static int eval(const sp_frame_ctx* c, sp_rule_event* ev) {
  if (!c || !ev || c->status != 0) return 0;  /* LATCH/旧帧不评 */
  if (c->ego.speed > thr_speed_high) {
    snprintf(ev->name, sizeof(ev->name), "demo_speed_high");
    ev->strength = c->ego.speed / 20.0f > 1.0f ? 1.0f : c->ego.speed / 20.0f;
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
      SP_RULE_ABI, "demo_speed_high", "1.0.0", init, eval, fini};
  return &d;
}
