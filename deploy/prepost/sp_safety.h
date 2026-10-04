// sp_safety.h —— M-PROD B1: 运行时异常探测 (宿主侧检查 + 处置状态机)
// + B2 开机自检的容差比较原语.
// spec: docs/superpowers/specs/2026-10-03-mprod-hardening-design.md §6.
//
// 检查点 (每帧, 完成帧收割处, 全部 µs 级; 设备侧状态 absmax 见 sp_safety.cu):
//   NaN/Inf   det_cls/det_bbox/motion_reg/plan_reg/plan_status 非有限值
//   检出合理  解码后 BEV |x|,|y|>100m; 尺寸 w,l,h∈(0,20m] 外
//   检出数漂移 n_det 滚动 100 帧 z-score>6 (std≈0 跳过)
//   plan 合理  final_plan 相邻步 |a|>15m/s² (0.5s/步) 或 |κ|>1.0
//   状态发散   设备侧 absmax 非有限或 > SP_DIV_ABSMAX (默认 1e6)
//
// 处置状态机 (防掩盖真故障, 不做无限复位循环):
//   单帧异常 → nan_hits/div_hits++ + 模板复位该帧状态(复用 k=0/40 机制)
//            + 本帧信箱 status=DEGRADED_RESET (结果基于复位态, 标志明确)
//   60s 窗复位 > SP_DIV_LATCH_N → LATCH: 持续发布 last_valid + 真实旧化
//            → SP_DIV_GRACE_MS 宽限 → exit 14 (由节点执行)
// env 旋钮: SP_DIV_EN / SP_DIV_ABSMAX / SP_DIV_LATCH_N / SP_DIV_WIN_MS /
//           SP_DIV_GRACE_MS / SP_DIV_Z / SP_INJECT_NAN / SP_INJECT_NAN_PERIOD
#ifndef SP_SAFETY_H_
#define SP_SAFETY_H_

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include "sp_watch.h"

namespace sp {
namespace saf {

// ---- 命中位 ----
enum {
  kHitNone = 0,
  kHitNan = 1,     // 非有限值 (输出或状态)
  kHitRange = 2,   // 检出几何越界
  kHitDrift = 4,   // 检出数漂移
  kHitState = 8,   // 状态向量发散
  kHitPlan = 16    // plan 运动学越界
};

inline const char* hit_name(int bits) {
  if (bits & kHitNan) return "nan";
  if (bits & kHitState) return "state";
  if (bits & kHitRange) return "range";
  if (bits & kHitDrift) return "drift";
  if (bits & kHitPlan) return "plan";
  return "?";
}

struct Config {
  bool en = true;
  float absmax = 1e6f;
  int latch_n = 5;
  double win_ms = 60000;
  double grace_ms = 2000;
  double z = 6.0;
};

inline Config load_config() {
  Config c;
  const char* e = getenv("SP_DIV_EN");
  if (e && !atoi(e)) c.en = false;
  if ((e = getenv("SP_DIV_ABSMAX")) && atof(e) > 0) c.absmax = (float)atof(e);
  if ((e = getenv("SP_DIV_LATCH_N")) && atoi(e) > 0) c.latch_n = atoi(e);
  if ((e = getenv("SP_DIV_WIN_MS")) && atof(e) > 0) c.win_ms = atof(e);
  if ((e = getenv("SP_DIV_GRACE_MS")) && atof(e) > 0) c.grace_ms = atof(e);
  if ((e = getenv("SP_DIV_Z")) && atof(e) > 0) c.z = atof(e);
  return c;
}

// ---- 处置状态机 (每帧喂一次) ----
struct Verdict {
  bool anomalous = false;
  bool do_reset = false;    // 模板复位本帧状态
  bool latch = false;       // 复位次数超窗 → 节点应走 LATCH→exit14
  int reason = 0;           // res::kReason*
};

struct Safety {
  Config cf;
  uint32_t nan_hits = 0, div_hits = 0, resets_60s = 0;
  std::deque<int64_t> reset_ms;  // 复位事件时刻 (win_ms 窗)
  std::deque<float> ndet_win;    // 检出数滚动窗 (100)

  explicit Safety(const Config& c) : cf(c) {}

  // 检出数漂移 (z-score, 窗满 20 才生效)
  int check_ndet(int n_det) {
    ndet_win.push_back((float)n_det);
    if (ndet_win.size() > 100) ndet_win.pop_front();
    size_t n = ndet_win.size();
    if (n < 20) return 0;
    double mean = 0;
    for (float x : ndet_win) mean += x;
    mean /= n;
    double var = 0;
    for (float x : ndet_win) var += (x - mean) * (x - mean);
    var /= n;
    double sd = sqrt(var);
    if (sd < 1e-3) return 0;  // 恒定检出数 (thr=0 饱和 topk) 无漂移可言
    double z = fabs((double)n_det - mean) / sd;
    return z > cf.z ? kHitDrift : 0;
  }

  // hits = 各检查命中位 OR; now_ms = CLOCK_MONOTONIC ms
  Verdict frame(int hits, int64_t now_ms) {
    Verdict v;
    if (!cf.en) return v;
    if (hits & kHitNan) ++nan_hits;
    if (hits & (kHitRange | kHitDrift | kHitState | kHitPlan)) ++div_hits;
    if (!hits) {
      while (!reset_ms.empty() && (double)(now_ms - reset_ms.front()) > cf.win_ms)
        reset_ms.pop_front();
      resets_60s = (uint32_t)reset_ms.size();
      return v;
    }
    v.anomalous = true;
    v.reason = (hits & kHitNan) ? 1 : 2;  // res::kReasonNan / kReasonDivergence
    reset_ms.push_back(now_ms);
    while (!reset_ms.empty() && (double)(now_ms - reset_ms.front()) > cf.win_ms)
      reset_ms.pop_front();
    resets_60s = (uint32_t)reset_ms.size();
    v.do_reset = true;
    v.latch = (int)reset_ms.size() > cf.latch_n;
    return v;
  }
};

// ---- 宿主侧检查原语 ----
// 非有限扫描 (det_cls/det_bbox/motion_reg/plan_reg/plan_status 原始张量)
inline int scan_nonfinite(const float* p, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (!std::isfinite(p[i])) return kHitNan;
  return 0;
}

// 解码后检出合理性: BEV |x|,|y|≤100m; w,l,h∈(0,20m]; 全字段有限
inline int check_det_geo(const float* x, const float* y, const float* w,
                         const float* l, const float* h, int n) {
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(x[i]) || !std::isfinite(y[i]) || !std::isfinite(w[i]) ||
        !std::isfinite(l[i]) || !std::isfinite(h[i]))
      return kHitNan;
    if (fabsf(x[i]) > 100.f || fabsf(y[i]) > 100.f) return kHitRange;
    if (w[i] <= 0.f || w[i] > 20.f || l[i] <= 0.f || l[i] > 20.f ||
        h[i] <= 0.f || h[i] > 20.f)
      return kHitRange;
  }
  return 0;
}

// final_plan (6 点, 0.5s/步) 运动学: |a|>15m/s² 或 |κ|>1.0.
// 只应喂入置信模式的结果 —— 低置信 argmax 的 plan_reg 是噪声不是发散
// (板端实测: conf=0 帧的姿态超界属正常分布, 会造成复位误报/LATCH 误熔断).
// worst_a/worst_k 非空时带回本次最大值 (校准用).
inline int check_plan(const float* plan, int npts, float dt,
                      float* worst_a = nullptr, float* worst_k = nullptr) {
  if (worst_a) *worst_a = 0.f;
  if (worst_k) *worst_k = 0.f;
  if (npts < 3) return 0;
  float v_prev = 0.f;
  bool have_v = false;
  for (int i = 0; i + 1 < npts; ++i) {
    float dx = plan[(i + 1) * 2] - plan[i * 2];
    float dy = plan[(i + 1) * 2 + 1] - plan[i * 2 + 1];
    if (!std::isfinite(dx) || !std::isfinite(dy)) return kHitNan;
    float v = sqrtf(dx * dx + dy * dy) / dt;
    if (have_v && worst_a) {
      float a = fabsf(v - v_prev) / dt;
      if (a > *worst_a) *worst_a = a;
    }
    if (have_v) {
      float a = fabsf(v - v_prev) / dt;
      if (a > 15.f) return kHitPlan;
      if (i + 2 < npts) {
        // κ = 2·|cross(p1-p0, p2-p1)| / (|p1-p0|·|p2-p1|·|p2-p0|)
        float ax = plan[(i + 1) * 2] - plan[i * 2];
        float ay = plan[(i + 1) * 2 + 1] - plan[i * 2 + 1];
        float bx = plan[(i + 2) * 2] - plan[(i + 1) * 2];
        float by = plan[(i + 2) * 2 + 1] - plan[(i + 1) * 2 + 1];
        float la = sqrtf(ax * ax + ay * ay);
        float lb = sqrtf(bx * bx + by * by);
        if (la > 1e-6f && lb > 1e-6f) {
          float cx = plan[(i + 2) * 2] - plan[i * 2];
          float cy = plan[(i + 2) * 2 + 1] - plan[i * 2 + 1];
          float lc = sqrtf(cx * cx + cy * cy);
          float k = 2.f * fabsf(ax * by - ay * bx) / (la * lb * lc);
          if (worst_k && std::isfinite(k) && k > *worst_k) *worst_k = k;
          if (std::isfinite(k) && k > 1.0f) return kHitPlan;
        }
      }
    }
    v_prev = v;
    have_v = true;
  }
  return 0;
}

// ---- B2: 自检容差比较 ----
// tol 文件: 每行 "<name> <tol>"; 金标张量: <dir>/frame0/<name>.bin
struct GoldenTol {
  std::vector<std::string> names;
  std::vector<double> tol;
};

// int 位型 → float (absmax 标量 D2H 解码; 不依赖 CUDA 内建)
inline float int_as_float(int i) {
  float f;
  memcpy(&f, &i, 4);
  return f;
}

inline bool load_tol(const char* dir, GoldenTol* gt) {
  std::string path = std::string(dir) + "/tol.txt";
  FILE* f = fopen(path.c_str(), "r");
  if (!f) return false;
  char ln[256];
  while (fgets(ln, sizeof(ln), f)) {
    char nm[128];
    double t;
    if (sscanf(ln, "%127s %lf", nm, &t) == 2) {
      gt->names.push_back(nm);
      gt->tol.push_back(t < 1e-3 ? 1e-3 : t);
    }
  }
  fclose(f);
  return !gt->names.empty();
}

// 返回超差元素数 (0=过); worst 差值与首差下标带回诊断
inline int compare_bin(const char* dir, const std::string& name,
                       const void* got, size_t bytes, double tol,
                       double* worst, size_t* worst_at) {
  *worst = 0;
  *worst_at = 0;
  std::string path = std::string(dir) + "/frame0/" + name + ".bin";
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return -1;
  std::vector<char> ref(bytes);
  size_t rd = fread(ref.data(), 1, bytes, f);
  fclose(f);
  if (rd != bytes) return -1;
  const float* a = (const float*)got;
  const float* b = (const float*)ref.data();
  size_t n = bytes / 4;
  int bad = 0;
  for (size_t i = 0; i < n; ++i) {
    double d = fabs((double)a[i] - (double)b[i]);
    if (std::isnan(d)) {  // 任一侧 NaN → 差值 NaN: "d>tol" 恒 false 会被吞
      ++bad;
      continue;
    }
    if (d > *worst) {
      *worst = d;
      *worst_at = i;
    }
    if (d > tol) ++bad;
  }
  return bad;
}

}  // namespace saf
}  // namespace sp

#endif  // SP_SAFETY_H_
