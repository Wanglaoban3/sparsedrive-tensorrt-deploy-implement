# M9a 插件开发指南 —— sp_trigger 规则插件

> 目标读者：下载本仓库、想给自己的场景加触发规则的工程师。**全程不改
> core 代码**：落一个 `.so` + 一段阈值配置即完成扩展（spec §4.2）。

## 1. 架构一页图

```
sp_result_m3 信箱(只读) ──> sp_trigger (10Hz tick)
                               │  manifest join (seq→l2g/ts) → ego 运动学
                               │  64 帧历史环 → sp_frame_ctx
                               ├──> rules/*.so 逐个 eval → 事件
                               │       ↑ THR 阈值: /etc/sp/thr.conf 按名字注入
                               └──> 去重/配额仲裁 → /var/lib/sp/trigger/events.jsonl
```

- **宿主 core 固定**：`sp_trigger.cpp` + `sp_ruleload.h`（dlopen/ABI 校验）
  + `sp_egoring.h`（ego/历史）+ `sp_quota.h`（仲裁）。
- **扩展点只有两个**：规则插件 `.so`（本指南）和 `thr.conf` 阈值段。
- 规则崩溃 = 进程死 → systemd 1s 拉起（spec §6 允许）；加载期坏插件被
  拒载，宿主不倒。

## 2. 规则插件 ABI（sp_rule.h，当前 SP_RULE_ABI = 1）

```c
typedef struct { float x,y,heading,speed,acc,yaw_rate; } sp_ego_state;
// heading 是连续 unwrap 值 (跨 ±π 不翻侧, 终审 C1); 做角度差仍须 wrap_pi,
// 不要拿相邻两帧 heading 直接相减判"转角"
typedef struct { float score; int32_t label,id;
                 float x,y,z,w,l,h,yaw,vx,vy; } sp_trk;   // ego 系, x 前向
typedef struct { char name[32]; float strength; } sp_rule_event;

typedef struct sp_frame_ctx {
  uint32_t abi_ver;  uint64_t seq;  int64_t ts_ns;  uint32_t scene;
  sp_ego_state ego;
  const sp_trk* det;  uint32_t n_det;
  uint32_t n_hist;                        // <= SP_RULE_HIST(64), 含当前帧
  const sp_trk* const* hist_det;          // hist[i] = i 帧前 (0=当前)
  const uint32_t* hist_n;  const sp_ego_state* hist_ego;
  const int64_t* hist_ts;
  const float* final_plan;  uint32_t status;
} sp_frame_ctx;

typedef struct {
  uint32_t abi_ver;  const char* name;  const char* version;
  int (*init)(const char* kv_text);     // 启动注入 thr.conf 的 [name] 段
  int (*eval)(const sp_frame_ctx*, sp_rule_event*);  // 返回 0/1 事件
  void (*fini)(void);
} sp_rule_desc;
// 必须导出:
#ifdef __cplusplus
extern "C"
#endif
const sp_rule_desc* sp_rule_query(void);
```

约定：
- `ctx->hist_*` 是**新→旧**（`hist[0]`=当前帧）；做时序窗口时自行翻转成
  旧→新（用 `sp_rule_util.h` 的助手，见 §3）。
- `ctx->status != 0`（LATCH/自检失败）时**必须不评**——那是旧结果重发。
- `ctx->ts_ns` 是 manifest 采集纪元 ts（与离线对拍同源）；一切时间窗按
  ts 实差，不要数帧。
- `strength ∈ [0,1]`；name 必须与 `thr.conf` 段名一致（也是事件名）。

## 3. SDK 助手（sp_rule_util.h，static inline 每规则一份）

- `ru_sustained(vals, ts_chrono, n, thr, ge, dur_s)` — 旧→新序列上的
  持续判定（run 的 ts 实差和 ≥ dur_s）。
- `ru_id_series(ctx, id, sx, sy, svx, svy, sts)` — 按跟踪 id 收集旧→新
  时序（缺帧跳过不中断）。
- `ru_pick_lead(ctx, lon_min, lon_max, lat_max)` — 当前帧同廊道最近车辆。
- `ru_thr(kv, key, dflt)` — 解析自己 THR 段的 k=v。
- 类别集：`ru_is_vehicle`(0-4) / `ru_is_vru`(6-8) / `ru_is_cone`(9)
  （nuScenes 序，改类别先核 `eval_t6_mini_v8.py` 的 class 列表）。

## 4. 五分钟写一条规则

从模板拷贝（模板本体 `sp_rule_template.c` = demo_speed_high）：

```c
/* sp_rule_my_rule.c */
#include "sp_rule_util.h"
#include <stdio.h>
#include <stdlib.h>
static double thr_gap = 1.0;                      /* 阈值, THR 段可覆盖 */

static int init(const char* kv) {
  thr_gap = ru_thr(kv, "gap_thr", 1.0);
  return 0;
}
static int eval(const sp_frame_ctx* c, sp_rule_event* ev) {
  if (!c || !ev || c->status != 0) return 0;
  /* ... 你的逻辑: 用 c->det / c->ego / 助手函数 ... */
  if (/* 命中 */) {
    snprintf(ev->name, sizeof(ev->name), "my_rule");
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
  static const sp_rule_desc d = {SP_RULE_ABI, "my_rule", "1.0.0",
                                 init, eval, fini};
  return &d;
}
```

编译（板上或交叉同 ABI）：

```bash
g++ -O2 -shared -fPIC -I/opt/m0/trt-dev/prepost sp_rule_my_rule.c \
    -o /usr/local/share/sp/rules/sp_rule_my_rule.so
```

阈值段（追加到 /etc/sp/thr.conf）：

```ini
[my_rule]
gap_thr=1.5
```

生效：`kill -HUP $(systemctl show sp-trigger@m3 -p MainPID --value)` ——
宿主重扫规则目录并重新读 thr.conf；日志出现 `loaded my_rule` 即成。
**注意 `.c` 文件被 g++ 按 C++ 编译，导出符号必须包 `extern "C"`**（模板
已处理好；裸写常见死法是 dlsym 报 sp_rule_query missing）。

## 5. 阈值标定纪律（spec §9，不要跳）

THR 种子值来自 ad-data-engine v0（GT 离线分布），上车感知输入分布会
漂移。新规则上线流程：

1. 观察模式：`SP_TRIG_OBSERVE=1` 跑 ≥3min，`observe.jsonl` 拿原始统计量
   分布（自己的统计量可临时加打点）；
2. 分位数定阈（物理量纲阈值除外，给观测证据即可）；
3. 门槛校准：`python deploy\_m9a_thr_calib.py 3`（现成 4 条已校准）；
4. 合成序列自测（对照 `test_rule_load.cpp` 用例 5 的形状）。

## 6. 事件流与仲裁

- 事件一行一 JSON：`{"ts_ns":..,"seq":..,"event":..,"strength":..}`，
  落 `/var/lib/sp/trigger/events.jsonl`（10MB×5 轮转，逐事件 flush）。
- 仲裁顺序：同名 5s 冷却 → 单事件 12/min → 全局 15/min（`[quota]` 段
  可调）；被吞事件只进 stderr 计数，不落盘。
- ctx-dump 模式（`SP_TRIG_CTX_DUMP=1`）另落逐帧二进制夹具 +
  `events_raw.jsonl`（免配额），用于 §7.5 一致性比对（numpy 镜像 =
  `deploy/_m9a_replay.py`）。

## 7. 常见死法（已踩实）

| 症状 | 根因 |
|---|---|
| dlsym: sp_rule_query missing | 忘了 `extern "C"`（g++ 名字修饰） |
| REJECT: abi mismatch | 改了 sp_rule.h 没 bump SP_RULE_ABI，或 .so 陈旧 |
| 事件迟迟不出现 | 忘了 `status!=0` 不评 / quota 冷却窗 / 阈值段名与规则名不一致 |
| ego.speed 恒 0 或怪异 | 车前向 = l2g R 第 1 列（lidar 装转 90°），别用 atan2(R10,R00) |
| trigger 单元起不来 start-limit-hit | 连续重启熔断，`systemctl reset-failed sp-trigger@m3` |
