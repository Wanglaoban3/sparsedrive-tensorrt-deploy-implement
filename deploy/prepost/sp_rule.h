/* sp_rule.h — M9a 规则插件统一 C ABI (spec §4.2).
 * 版本纪律同 RingMeta kVersion: 宿主校验 abi_ver 不匹配拒载 fail-visible.
 * 修改任何 struct 字段必须 bump SP_RULE_ABI 并同步宿主与全部规则.
 */
#ifndef SP_RULE_H_
#define SP_RULE_H_
#include <stdint.h>

#define SP_RULE_ABI 1
#define SP_RULE_HIST 64       /* ctx 历史深度 (帧): hist[0]=当前, hist[i]=i 帧前 */
#define SP_RULE_NAME_MAX 32

/* 自车运动学 (宿主从 l2g 链差分; 规则只读) */
typedef struct {
  float x, y;        /* 全局系位置 m (l2g 平移) */
  float heading;     /* 弧度, unwrap 后连续 */
  float speed;       /* m/s, 沿航向中点投影 (events.py human_ego_track 口径) */
  float acc;         /* m/s^2, speed 对时间差分 */
  float yaw_rate;    /* rad/s, heading 对时间差分 */
} sp_ego_state;

/* 检出目标 (镜像 res::DetBox, 48B 自然布局) */
typedef struct {
  float score;
  int32_t label;
  int32_t id;        /* 跟踪实例 id (-1 无) */
  float x, y, z, w, l, h, yaw, vx, vy;
} sp_trk;

/* 规则事件输出 */
typedef struct {
  char name[SP_RULE_NAME_MAX];
  float strength;    /* [0,1] */
} sp_rule_event;

/* 逐 tick 上下文 (宿主填充, 规则只读; 全部指针由宿主持有, 仅本 tick 有效) */
typedef struct sp_frame_ctx {
  uint32_t abi_ver;
  uint64_t seq;
  int64_t ts_ns;     /* manifest 采集 ts (与离线对拍同源, 非墙钟) */
  uint32_t scene;
  sp_ego_state ego;
  const sp_trk* det;         /* 当前帧检出 */
  uint32_t n_det;
  uint32_t n_hist;           /* 含当前帧的有效历史深度, <= SP_RULE_HIST */
  const sp_trk* const* hist_det;
  const uint32_t* hist_n;
  const sp_ego_state* hist_ego;
  const int64_t* hist_ts;
  const float* final_plan;   /* [6][2] (无则 0) */
  uint32_t status;           /* 信箱 status: !=0 (LATCH 等) 规则应不评 */
} sp_frame_ctx;

/* 插件导出: extern "C" const sp_rule_desc* sp_rule_query(void); */
typedef struct {
  uint32_t abi_ver;
  const char* name;      /* = 事件名 = thr.conf 段名 */
  const char* version;
  int (*init)(const char* kv_text);   /* 启动注入自己的 THR 段原文 (可空) */
  int (*eval)(const sp_frame_ctx* c, sp_rule_event* ev);  /* 返回事件数 0/1 */
  void (*fini)(void);
} sp_rule_desc;

#endif  /* SP_RULE_H_ */
