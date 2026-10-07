// sp_egoring.h — M9a 触发器 ego 历史环 (header-only; sp_trigger.cpp 的
// 核心逻辑提出, 板测可驱动). ego 运动学 = events.py human_ego_track 的
// 流式版: 速度沿航向中点投影, acc/yaw_rate 反向差分, 时间窗按 ts 实差.
// 场景边界 / ts 异常 / seq 倒退 → 窗口重建, 本帧速度置零 (假尖峰免疫).
#ifndef SP_EGORING_H_
#define SP_EGORING_H_
#include <cmath>
#include <cstdint>
#include <cstring>

#include "sp_result.h"   // kDetCap (检出容量与信箱一致)
#include "sp_rule.h"

// manifest 单帧最小集 (触发器/板测共同口径)
struct TrigFrameLite {
  uint64_t seq;
  uint32_t scene;
  int64_t ts_ns;
  const double* l2g;  // 16, 行主序
};

struct Egoring {
  sp_trk det[SP_RULE_HIST][sp::res::kDetCap];
  uint32_t n[SP_RULE_HIST];
  sp_ego_state ego[SP_RULE_HIST];
  int64_t ts[SP_RULE_HIST];
  uint32_t scene[SP_RULE_HIST];
  uint64_t seq[SP_RULE_HIST];
  int head = 0;        // 下一写入槽
  uint32_t depth = 0;  // 有效帧数 (<= SP_RULE_HIST)

  void reset() { head = 0; depth = 0; }

  // 处理一帧; false = 该帧被判定为边界/异常首帧 (速度置零, 窗口已重建)
  bool push_frame(const TrigFrameLite& fr, const sp_trk* d, uint32_t nd) {
    bool boundary = false;
    sp_ego_state eg;
    memset(&eg, 0, sizeof(eg));
    eg.x = (float)fr.l2g[3];
    eg.y = (float)fr.l2g[7];
    // 车前向 = R 第 1 列 (l2g[1], l2g[5]): nuScenes LIDAR_TOP 装转 90 度,
    // l2g 是 lidar 位姿, atan2(R10,R00) 给的是 lidar x 轴, 与行驶方向差
    // 90 度 (2026-10-07 板上实测: 位移投影 col0≈0, col1=8.45=hypot)
    double h_raw = atan2(fr.l2g[5], fr.l2g[1]);
    eg.heading = (float)h_raw;
    uint32_t prev = (head + SP_RULE_HIST - 1) % SP_RULE_HIST;
    if (depth == 0 || scene[prev] != fr.scene) {
      boundary = true;  // 首帧 / 场景切换: 窗口重建, 不差分
    } else {
      double dt = (double)(fr.ts_ns - ts[prev]) / 1e9;
      if (!(dt > 0 && dt < 10.0)) {
        boundary = true;  // ts 回退/长停
      } else {
        // heading 连续 unwrap (终审 C1): 裸 atan2 值域 (-pi,pi], 跨 ±pi
        // 边界时存裸值会让下方中点角偏差 pi → speed 投影翻负 (交付夹具
        // 实测 950 帧 24 处) → 假 reverse/acc 尖峰。存连续值; 规则侧
        // yaw_rate/u_turn 本就走 wrap_pi 差分, 不受影响。
        eg.heading =
            (float)((double)ego[prev].heading +
                    wrap_pi(h_raw - (double)ego[prev].heading));
        double dx = fr.l2g[3] - (double)ego[prev].x;
        double dy = fr.l2g[7] - (double)ego[prev].y;
        double mid = 0.5 * ((double)ego[prev].heading + eg.heading);
        eg.speed = (float)((dx * cos(mid) + dy * sin(mid)) / dt);
        eg.acc = (float)((eg.speed - (double)ego[prev].speed) / dt);
        double dh = wrap_pi(eg.heading - (double)ego[prev].heading);
        eg.yaw_rate = (float)(dh / dt);
      }
    }
    if (boundary) reset();
    n[head] = nd;
    if (nd) memcpy(det[head], d, nd * sizeof(sp_trk));
    ego[head] = eg;
    ts[head] = fr.ts_ns;
    scene[head] = fr.scene;
    seq[head] = fr.seq;
    head = (head + 1) % SP_RULE_HIST;
    if (depth < SP_RULE_HIST) depth += 1;
    return !boundary;
  }

  // ctx 组装: hist[i] = i 帧前 (含当前); 单线程约定 (静态暂存)
  void build_ctx(sp_frame_ctx* ctx, uint32_t status) {
    static sp_ego_state he[SP_RULE_HIST];
    static int64_t ht[SP_RULE_HIST];
    static uint32_t hn[SP_RULE_HIST];
    static sp_trk* hdp[SP_RULE_HIST];
    memset(ctx, 0, sizeof(*ctx));
    uint32_t cur = (head + SP_RULE_HIST - 1) % SP_RULE_HIST;
    for (uint32_t i = 0; i < depth; ++i) {
      uint32_t ix = (cur + SP_RULE_HIST - i) % SP_RULE_HIST;
      he[i] = ego[ix];
      ht[i] = ts[ix];
      hn[i] = n[ix];
      hdp[i] = det[ix];
    }
    ctx->abi_ver = SP_RULE_ABI;
    ctx->seq = seq[cur];
    ctx->ts_ns = ts[cur];
    ctx->scene = scene[cur];
    ctx->ego = ego[cur];
    ctx->det = det[cur];
    ctx->n_det = n[cur];
    ctx->n_hist = depth;
    ctx->hist_det = (const sp_trk* const*)hdp;
    ctx->hist_n = hn;
    ctx->hist_ego = he;
    ctx->hist_ts = ht;
    ctx->status = status;
  }

  static double wrap_pi(double a) {
    while (a > M_PI) a -= 2 * M_PI;
    while (a < -M_PI) a += 2 * M_PI;
    return a;
  }
};

#endif  // SP_EGORING_H_
