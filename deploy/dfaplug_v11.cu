// ============================================================================
// DeformableAggregation TRT plugin v11 (v8 的 map-gather 向量化版, 2026-10-06)
// 语义与 v8 完全一致 (logits 入 / 内联 softmax / loc 有效性 / eps-skip /
// 双线性角点), 引擎按 名字/版本/命名空间 绑定, .so 热交换不重编引擎。
//
// 相对 v8 (deploy/dfaplug_v8.cu) 的两个结构变化 (P3 第六轮的剩余空间收尾):
//  K1 plan  pass2 每 (anchor,c,s,p) 至多写 **一条** per-anchor 条目 (v8 按
//    (anchor,group) 复制 G=8 份 32B 条目): EntryA 48B = {int off[4] 四角绝对
//    元素偏移, half wt[8] 组权重(wt<eps 记 0, 与"不参与聚合"数值等价:
//    fmaf(+0,f,acc)==acc), half cw[4] 角权重 + 8B pad}。
//    workspace: nA*(4B 计数 + NCSP*48B + split*nA*C*4B partial), map 头
//    184MB -> 35MB。
//  K2 gather 每 (anchor, SPLIT 分片) 一个 warp, **lane 拥有 8 连续通道**
//    (uint4=16B tap, warp 一次事务覆盖整条 512B 角点行; v8 为 lane=单通道
//    标量 half 读 + 条目按组复制 8 份广播)。lane 的 8 通道恰落单一 group
//    (8 | 32), 组权重每 lane 一个标量。SPLIT = ceil(768/nA) (目标 16SM×48
//    warp), SPLIT>1 走 fp32 partial + 确定性 finalize (固定序求和, 无原子)。
//    前例: SparseDriveV2 dfa_plugin.cu v4/v5 (entry half 化 + 8ch/lane
//    uint4 tap + SPLIT finalize), M2/M3 门禁同值通过。
//
// 生效条件: G==8 且 C%32==0 (det A=900 与 map A=100 均满足: G=8, C=256);
// 否则回落 v8 内核, 再回落 v3 (同一 .so)。
// 数值差异: 条目权重 half 化 (cw/wt 存 half, 乘积 gather 内 fp32 重算;
// v8 为 plan 内 fp32 折叠存 fp32) —— 权重相对误差 ~1e-3, 与 v3->v8 的
// "边界条目翻转" 同类 ulp 级; 累加顺序 = 单列表 slot 序 (atomicAdd 到达序,
// 与 v8 每组列表内到达序一致, 且零权重项 fmaf(+0,·,·) 精确不改累加器)。
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"

#ifndef DFA_W_EPS
#define DFA_W_EPS 1.0e-4f
#endif

// v8 (及其内嵌 v3) 作为回落路径; DFA_V8_EXTERNAL=外部已提供 v8 内核 (harness)
// 注: v8.cu 底部 registrar 引用字面量 token dfa_trt, 且其内部 include v3 时会
// #undef DFA_NO_REGISTER —— 故这里按 harness 同款方式自行包含 v3 (kb3) +
// 以 DFA_V8_NO_FALLBACK 包 v8, 保证 DFA_NO_REGISTER 存活到 v8 末尾,
// 只有 v11 自己的 registrar 在文件尾部注册。
#ifndef DFA_V8_CALL_NS
#define DFA_V8_CALL_NS dfa_k8
#endif
#ifndef DFA_V3_CALL_NS
#define DFA_V3_CALL_NS kb3
#endif
#ifndef DFA_V8_EXTERNAL
#define DFA_NO_REGISTER
#define dfa_kernel DFA_V3_CALL_NS
#define dfa_trt dfa_tb3
#define dfa_reg dfa_rb3
#include "dfaplug_v3.cu"
#undef dfa_kernel
#undef dfa_trt
#undef dfa_reg
#define DFA_V8_NO_FALLBACK
#define DFA_V8_NS DFA_V8_CALL_NS
#define DFA_TRT_NS dfa_t8
#define DFA_REG_NS dfa_r8
#include "dfaplug_v8.cu"
#undef DFA_NO_REGISTER
#undef DFA_V8_NO_FALLBACK
#endif

#ifndef DFA_V11_NO_FALLBACK
#define DFA_V11_MAYBE_FALLBACK
#endif

namespace DFA_V11_NS {

struct __align__(16) EntryA {  // 48B
  int off[4];                  // 四角绝对元素偏移; 无效角 = 0 (cw 也为 0)
  uint4 wt8;                   // 8 组 softmax 权重 (half), <eps 组 = 0
  uint4 cw4;                   // 4 角双线性权重 (half) + 8B pad(0)
};

// ---- plan: 与 v8 pass1 逐行相同, pass2 改写 per-anchor 条目 -----------------
template <typename IT>
__global__ void __launch_bounds__(256)
dfa_plan_kernel_v11(int* __restrict__ counts, EntryA* __restrict__ entries,
                    const IT* __restrict__ spatial_shape,
                    const IT* __restrict__ scale_start_index,
                    const __half* __restrict__ sample_location,
                    const __half* __restrict__ logits,
                    const int batch_size, const int num_cams, int num_feat,
                    const int num_embeds, const int num_scale,
                    const int num_anchors, const int num_pts,
                    const int num_groups) {
  (void)batch_size;
  const int na = blockIdx.x;               // b*A + a
  const int G = num_groups;                // ==8 (调用方保证)
  const int NCSP = num_cams * num_scale * num_pts;
  const int ncs = num_cams * num_scale;
  extern __shared__ float sm[];
  float* s_max = sm;                       // [8]
  float* s_sum = sm + 8;                   // [8]
  float* s_part = sm + 16;                 // [8 warps][8 g][2]
  int* sshape = reinterpret_cast<int*>(sm + 16 + 128);  // [ncs*3]
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = blockDim.x >> 5;      // 8

  for (int i = tid; i < ncs * 3; i += blockDim.x) {
    const int cs = i / 3;
    const int r = i - cs * 3;
    sshape[i] = (r == 0) ? (int)spatial_shape[cs * 2]
              : (r == 1) ? (int)spatial_shape[cs * 2 + 1]
                         : (int)scale_start_index[cs];
  }

  // ---- pass1: online softmax stats (与 v8 相同) ----
  const __half* lrow = logits + (long long)na * NCSP * G;
  float m[8], ss[8];
#pragma unroll
  for (int g = 0; g < 8; ++g) { m[g] = -3.0e30f; ss[g] = 0.f; }
  for (int i = tid; i < NCSP; i += blockDim.x) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(lrow + (long long)i * 8));
    const __half* hh = reinterpret_cast<const __half*>(&u);
#pragma unroll
    for (int g = 0; g < 8; ++g) {
      const float v = __half2float(hh[g]);
      const float mn = fmaxf(m[g], v);
      ss[g] = ss[g] * __expf(m[g] - mn) + __expf(v - mn);
      m[g] = mn;
    }
  }
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    for (int off = 16; off > 0; off >>= 1) {
      const float mo = __shfl_down_sync(0xffffffffu, m[g], off);
      const float so = __shfl_down_sync(0xffffffffu, ss[g], off);
      const float mn = fmaxf(m[g], mo);
      ss[g] = ss[g] * __expf(m[g] - mn) + so * __expf(mo - mn);
      m[g] = mn;
    }
    if (lane == 0) {
      s_part[(warp * 8 + g) * 2] = m[g];
      s_part[(warp * 8 + g) * 2 + 1] = ss[g];
    }
  }
  __syncthreads();
  if (tid < G) {
    float m2 = -3.0e30f, ss2 = 0.f;
    for (int w = 0; w < nwarps; ++w) {
      const float mw = s_part[(w * 8 + tid) * 2];
      const float sw = s_part[(w * 8 + tid) * 2 + 1];
      const float mn = fmaxf(m2, mw);
      ss2 = ss2 * __expf(m2 - mn) + sw * __expf(mw - mn);
      m2 = mn;
    }
    s_max[tid] = m2;
    s_sum[tid] = ss2;
  }
  __syncthreads();
  float inv[8];
#pragma unroll
  for (int g = 0; g < 8; ++g) inv[g] = 1.f / s_sum[g];

  // ---- pass2: 每 (c,s,p) 至多一条 per-anchor 条目 ----
  const int SP = num_scale * num_pts;
  const __half* sloc = sample_location + (long long)na * num_pts * num_cams * 2;
  const long long fbase = (long long)num_feat * num_embeds *
                          (long long)(na / num_anchors);
  for (int i = tid; i < NCSP; i += blockDim.x) {
    const int c = i / SP;
    const int rem = i - c * SP;
    const int s = rem / num_pts;
    const int p = rem - s * num_pts;
    const long long lo = ((long long)p * num_cams + c) * 2;
    const float2 lw2 = __half22float2(__ldg(reinterpret_cast<const __half2*>(
        sloc + lo)));
    const float loc_w = lw2.x, loc_h = lw2.y;
    if (!(loc_w > 0.f && loc_w < 1.f && loc_h > 0.f && loc_h < 1.f)) continue;
    const int cs3 = (c * num_scale + s) * 3;
    const int h = sshape[cs3];
    const int w = sshape[cs3 + 1];
    const long long fcs = fbase + (long long)sshape[cs3 + 2] * num_embeds;
    const float h_im = loc_h * h - 0.5f;
    const float w_im = loc_w * w - 0.5f;
    const int h_low = floorf(h_im), w_low = floorf(w_im);
    const float lh = h_im - h_low, lw_ = w_im - w_low;
    const float hh = 1.f - lh, hw = 1.f - lw_;
    const long long r0 = (long long)h_low * w * num_embeds;
    const long long r1 = r0 + (long long)w * num_embeds;
    const long long c0 = (long long)w_low * num_embeds;
    const long long c1 = c0 + num_embeds;
    int o0 = 0, o1 = 0, o2 = 0, o3 = 0;
    float w1 = 0.f, w2 = 0.f, w3 = 0.f, w4 = 0.f;
    if (h_low >= 0 && w_low >= 0) {
      w1 = hh * hw; o0 = (int)(fcs + r0 + c0);
    }
    if (h_low >= 0 && w_low + 1 <= w - 1) {
      w2 = hh * lw_; o1 = (int)(fcs + r0 + c1);
    }
    if (h_low + 1 <= h - 1 && w_low >= 0) {
      w3 = lh * hw; o2 = (int)(fcs + r1 + c0);
    }
    if (h_low + 1 <= h - 1 && w_low + 1 <= w - 1) {
      w4 = lh * lw_; o3 = (int)(fcs + r1 + c1);
    }
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(lrow + (long long)i * 8));
    const __half* lhh = reinterpret_cast<const __half*>(&u);
    float wt[8];
    bool any = false;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
      const float w_g = __expf(__half2float(lhh[g]) - s_max[g]) * inv[g];
      if (w_g < DFA_W_EPS) {
        wt[g] = 0.f;                        // 与 v8 "不参与聚合" 数值等价
      } else {
        wt[g] = w_g;
        any = true;
      }
    }
    if (!any) continue;
    const int slot = atomicAdd(&counts[na], 1);
    EntryA* e = entries + (long long)na * NCSP + slot;
    int4 a4; a4.x = o0; a4.y = o1; a4.z = o2; a4.w = o3;
    *reinterpret_cast<int4*>(&e->off[0]) = a4;
    uint4 wq;                               // wt[8] half
    __half* wh = reinterpret_cast<__half*>(&wq);
#pragma unroll
    for (int g = 0; g < 8; ++g) wh[g] = __float2half(wt[g]);
    e->wt8 = wq;
    uint4 cq;                               // cw[4] half + pad
    __half* ch4 = reinterpret_cast<__half*>(&cq);
    ch4[0] = __float2half(w1); ch4[1] = __float2half(w2);
    ch4[2] = __float2half(w3); ch4[3] = __float2half(w4);
    ch4[4] = __float2half(0.f); ch4[5] = ch4[4];
    ch4[6] = ch4[4]; ch4[7] = ch4[4];
    e->cw4 = cq;
  }
}

// ---- gather: 每 (anchor, SPLIT 分片) 一个 warp, lane 拥有 8 连续通道 -------
// 单角点 tap: uint4 (16B) = 8 连续通道, 配对 half2->float2 转换 + FMA
__device__ __forceinline__ void v11_tap8(const __half* __restrict__ feat,
                                         long long off, int c0, float w,
                                         float* acc) {
  const uint4 u = *reinterpret_cast<const uint4*>(feat + off + c0);
  const __half2* h2 = reinterpret_cast<const __half2*>(&u);
  const float2 f0 = __half22float2(h2[0]);
  const float2 f1 = __half22float2(h2[1]);
  const float2 f2 = __half22float2(h2[2]);
  const float2 f3 = __half22float2(h2[3]);
  acc[0] = fmaf(w, f0.x, acc[0]); acc[1] = fmaf(w, f0.y, acc[1]);
  acc[2] = fmaf(w, f1.x, acc[2]); acc[3] = fmaf(w, f1.y, acc[3]);
  acc[4] = fmaf(w, f2.x, acc[4]); acc[5] = fmaf(w, f2.y, acc[5]);
  acc[6] = fmaf(w, f3.x, acc[6]); acc[7] = fmaf(w, f3.y, acc[7]);
}

__global__ void __launch_bounds__(256)
dfa_gather_v11_kernel(__half* __restrict__ output,      // [nA, C] (split==1)
                      float* __restrict__ partial,      // [split, nA, C]
                      const __half* __restrict__ feat,
                      const int* __restrict__ counts,
                      const EntryA* __restrict__ entries,
                      const int nA, const int num_embeds, const int NCSP,
                      const int split) {
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int na = wid / split;
  const int sp = wid - na * split;
  if (na >= nA) return;
  const int lane = threadIdx.x & 31;
  const int cnt = counts[na];
  const EntryA* list = entries + (long long)na * NCSP;
  const int per = (cnt + split - 1) / split;   // cnt==0 时 per=0, 两端空区间
  const int k0 = sp * per;
  const int k1 = (k0 + per < cnt ? k0 + per : cnt);
  const int c0 = lane << 3;                    // 本 lane 首通道 (组 = c0>>5)
  const int gi = c0 >> 5;
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.f;
  for (int k = k0; k < k1; ++k) {
    const uint4 e0 = __ldg(reinterpret_cast<const uint4*>(&list[k].off[0]));
    const uint4 e1 = __ldg(reinterpret_cast<const uint4*>(&list[k].wt8));
    const uint4 e2 = __ldg(reinterpret_cast<const uint4*>(&list[k].cw4));
    const int o0 = e0.x, o1 = e0.y, o2 = e0.z, o3 = e0.w;
    // 组权重 = e1 的第 gi 个 half; gi 是运行期值 —— 必须用 select 链从寄存器
    // 取, 数组式动态下标会把 e1 降级到 local memory (16B stack frame, 每次
    // 迭代 local 往返, map 侧实测 gather 慢 2x 的根因)
    const __half2* e1h2 = reinterpret_cast<const __half2*>(&e1);
    const float h0 = __low2float(e1h2[0]), h1 = __high2float(e1h2[0]);
    const float h2_ = __low2float(e1h2[1]), h3 = __high2float(e1h2[1]);
    const float h4 = __low2float(e1h2[2]), h5 = __high2float(e1h2[2]);
    const float h6 = __low2float(e1h2[3]), h7 = __high2float(e1h2[3]);
    const float wg = gi == 0 ? h0 : gi == 1 ? h1 : gi == 2 ? h2_
                     : gi == 3 ? h3 : gi == 4 ? h4 : gi == 5 ? h5
                     : gi == 6 ? h6 : h7;
    const __half* cw = reinterpret_cast<const __half*>(&e2);
    v11_tap8(feat, (long long)o0, c0, __half2float(cw[0]) * wg, acc);
    v11_tap8(feat, (long long)o1, c0, __half2float(cw[1]) * wg, acc);
    v11_tap8(feat, (long long)o2, c0, __half2float(cw[2]) * wg, acc);
    v11_tap8(feat, (long long)o3, c0, __half2float(cw[3]) * wg, acc);
  }
  if (split > 1) {
    float* dst = partial + ((size_t)sp * nA + na) * num_embeds + c0;
#pragma unroll
    for (int j = 0; j < 8; ++j) dst[j] = acc[j];
  } else {
    __half2* dst = reinterpret_cast<__half2*>(output +
                                              (size_t)na * num_embeds + c0);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      dst[j] = __floats2half2_rn(acc[2 * j], acc[2 * j + 1]);
  }
}

// ---- finalize: 固定序求 partial 和 (SPLIT>1 时; 确定性, 无原子) -------------
__global__ void dfa_gather_v11_finalize_kernel(
    __half* __restrict__ output, const float* __restrict__ partial,
    const int nA, const int num_embeds, const int split) {
  const int total2 = nA * (num_embeds >> 1);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total2;
       i += gridDim.x * blockDim.x) {
    const int na = i / (num_embeds >> 1);
    const int c2 = i - na * (num_embeds >> 1);
    const float* p = partial + (size_t)na * num_embeds + 2 * c2;
    float s0 = 0.f, s1 = 0.f;
    for (int s = 0; s < split; ++s) {
      s0 += p[0];
      s1 += p[1];
      p += (size_t)nA * num_embeds;
    }
    reinterpret_cast<__half2*>(output + (size_t)na * num_embeds + 2 * c2)[0] =
        __floats2half2_rn(s0, s1);
  }
}

static inline int v11_split_for(int nA) {
  // DFA_V11_SPLIT 环境变量可强制 (诊断/实验用)
  static const int forced = [] {
    const char* e = getenv("DFA_V11_SPLIT");
    return e ? atoi(e) : 0;
  }();
  if (forced > 0) return forced > 64 ? 64 : forced;
  // 板上实测 (dfa_eT6, 2026-10-06): gather 是并行度受限 —— map (nA=100)
  // split 2/4/8/16/32/64 -> 0.79/0.47/0.37/0.31/0.30/0.32ms (32 平台);
  // det (nA=900) split 1/2/4 -> 0.268/0.239/0.247ms (2 最优) —— 胖迭代内核
  // 需更多 warp 喂访存管线.
  if (nA >= 512) return 2;
  int sp = 3072 / nA + ((3072 % nA) ? 1 : 0);
  return sp < 1 ? 1 : (sp > 32 ? 32 : sp);
}

// workspace 布局: [counts nA int][pad256][entries nA*NCSP*48B]
//                [pad256][partial nA*C*split*4 (split>1 时有效)]
template <typename IT>
int dfa_plan_v11(const IT* spatial_shape, const IT* scale_start_index,
                 const __half* loc, const __half* logits, int batch_size,
                 int num_cams, int num_feat, int num_embeds, int num_scale,
                 int num_anchors, int num_pts, int num_groups, int* split_out,
                 void* ws, size_t ws_size, cudaStream_t stream) {
  if (num_groups != 8 || (num_embeds & 31)) return 1;
  if (batch_size < 1 || num_cams < 1 || num_scale < 1 || num_pts < 1 ||
      num_anchors < 1) return 1;
  const int NCSP = num_cams * num_scale * num_pts;
  const int nA = batch_size * num_anchors;
  const int split = v11_split_for(nA);
  if (split_out) *split_out = split;
  const size_t cnt_bytes = (size_t)nA * 4;
  const size_t ent_off = (cnt_bytes + (size_t)255) & ~(size_t)255;
  const size_t ent_bytes = (size_t)nA * NCSP * sizeof(EntryA);
  const size_t part_off = (ent_off + ent_bytes + (size_t)255) & ~(size_t)255;
  const size_t part_bytes = (size_t)nA * num_embeds * split * 4;
  if (ws_size < part_off + part_bytes) return 1;
  int* counts = reinterpret_cast<int*>(ws);
  EntryA* entries = reinterpret_cast<EntryA*>(
      reinterpret_cast<char*>(ws) + ent_off);
  cudaError_t e0 = cudaMemsetAsync(counts, 0, cnt_bytes, stream);
  if (e0 != cudaSuccess) return (int)e0;
  const size_t smem = (16 + 128 + (size_t)num_cams * num_scale * 3) *
                      sizeof(float);
  dfa_plan_kernel_v11<IT><<<nA, 256, smem, stream>>>(
      counts, entries, spatial_shape, scale_start_index, loc, logits,
      batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors,
      num_pts, num_groups);
  return (int)cudaGetLastError();
}

int dfa_gather_v11(__half* output, const __half* feat, void* ws,
                   size_t ws_size, int batch_size, int num_anchors,
                   int num_embeds, int num_cams, int num_scale, int num_pts,
                   int split, cudaStream_t stream) {
  const int NCSP = num_cams * num_scale * num_pts;
  const int nA = batch_size * num_anchors;
  const size_t cnt_bytes = (size_t)nA * 4;
  const size_t ent_off = (cnt_bytes + (size_t)255) & ~(size_t)255;
  const size_t ent_bytes = (size_t)nA * NCSP * sizeof(EntryA);
  const size_t part_off = (ent_off + ent_bytes + (size_t)255) & ~(size_t)255;
  if (ws_size < part_off) return 1;
  int* counts = reinterpret_cast<int*>(ws);
  const EntryA* entries = reinterpret_cast<const EntryA*>(
      reinterpret_cast<char*>(ws) + ent_off);
  float* partial = reinterpret_cast<float*>(
      reinterpret_cast<char*>(ws) + part_off);
  const int warps = nA * split;
  const int grid = (warps * 32 + 255) / 256;
  dfa_gather_v11_kernel<<<grid, 256, 0, stream>>>(
      output, partial, feat, counts, entries, nA, num_embeds, NCSP, split);
  int rc = (int)cudaGetLastError();
  if (rc != 0) return rc;
  if (split > 1) {
    const int total2 = nA * (num_embeds >> 1);
    const int fgrid = (total2 + 255) / 256;
    dfa_gather_v11_finalize_kernel<<<fgrid, 256, 0, stream>>>(
        output, partial, nA, num_embeds, split);
    rc = (int)cudaGetLastError();
  }
  return rc;
}

template <typename IT>
int dfa_launch_v11(__half* output, const __half* feat, const IT* spatial_shape,
                   const IT* scale_start_index, const __half* loc,
                   const __half* logits, int batch_size, int num_cams,
                   int num_feat, int num_embeds, int num_scale,
                   int num_anchors, int num_pts, int num_groups,
                   void* ws, size_t ws_size, cudaStream_t stream) {
  int split = 0;
  int rc = dfa_plan_v11<IT>(spatial_shape, scale_start_index, loc, logits,
                            batch_size, num_cams, num_feat, num_embeds,
                            num_scale, num_anchors, num_pts, num_groups,
                            &split, ws, ws_size, stream);
  if (rc != 0) return rc;
  return dfa_gather_v11(output, feat, ws, ws_size, batch_size, num_anchors,
                        num_embeds, num_cams, num_scale, num_pts, split,
                        stream);
}

// v11 workspace 需求 (字节)
static size_t ws_need_v11(long long nA, long long NCSP, long long C) {
  const int split = v11_split_for((int)nA);
  const size_t cnt_bytes = (size_t)nA * 4;
  const size_t ent_off = (cnt_bytes + (size_t)255) & ~(size_t)255;
  const size_t ent_bytes = (size_t)nA * NCSP * sizeof(EntryA);
  const size_t part_off = (ent_off + ent_bytes + (size_t)255) & ~(size_t)255;
  return part_off + (size_t)nA * C * split * 4;
}

}  // namespace DFA_V11_NS

// ============================================================================
// TensorRT plugin (IPluginV2DynamicExt) — 名字/版本/命名空间与 v3/v8 相同,
// enqueue 优先 v11, 回落 v8, 再回落 v3。
// ============================================================================
using namespace nvinfer1;

namespace DFA_TRT11_NS {

class DfaPluginV11 : public IPluginV2DynamicExt {
 public:
  DfaPluginV11() = default;
  DfaPluginV11(const void* data, size_t size) { (void)data; (void)size; }
  ~DfaPluginV11() override {
    if (own_ws_) cudaFree(own_ws_);
  }

  const char* getPluginType() const noexcept override {
    return "DeformableAggregation";
  }
  const char* getPluginVersion() const noexcept override { return "1"; }
  int32_t getNbOutputs() const noexcept override { return 1; }

  DimsExprs getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs,
                                int32_t nbInputs,
                                IExprBuilder& exprBuilder) noexcept override {
    DimsExprs o{};
    o.nbDims = 3;
    o.d[0] = inputs[4].d[0];
    o.d[1] = inputs[4].d[1];
    o.d[2] = inputs[0].d[2];
    (void)outputIndex; (void)nbInputs; (void)exprBuilder;
    return o;
  }

  DataType getOutputDataType(int32_t index, DataType const* inputTypes,
                             int32_t nbInputs) const noexcept override {
    (void)index; (void)nbInputs;
    return inputTypes[0];
  }

  bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {
    (void)nbInputs; (void)nbOutputs;
    if (inOut[pos].format != PluginFormat::kLINEAR) return false;
    if (pos == 1 || pos == 2)
      return inOut[pos].type == DataType::kFLOAT ||
             inOut[pos].type == DataType::kINT32;
    return inOut[pos].type == DataType::kHALF;
  }

  int32_t initialize() noexcept override { return 0; }
  void terminate() noexcept override {}
  size_t getSerializationSize() const noexcept override { return 0; }
  void serialize(void* buffer) const noexcept override { (void)buffer; }
  void destroy() noexcept override { delete this; }
  void setPluginNamespace(const char* ns) noexcept override {
    snprintf(ns_, sizeof(ns_), "%s", ns ? ns : "");
  }
  const char* getPluginNamespace() const noexcept override { return ns_; }

  IPluginV2DynamicExt* clone() const noexcept override {
    auto* p = new DfaPluginV11();
    p->setPluginNamespace(ns_);
    return p;
  }

  void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
                       DynamicPluginTensorDesc const* out,
                       int32_t nbOutputs) noexcept override {
    (void)in; (void)nbInputs; (void)out; (void)nbOutputs;
  }

  // workspace 取 v8/v11 两路径较大者 (回落安全)
  size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs,
                          PluginTensorDesc const* outputs,
                          int32_t nbOutputs) const noexcept override {
    (void)nbInputs; (void)outputs; (void)nbOutputs;
    if (nbInputs < 5) return 0;
    const Dims& wd = inputs[4].dims;  // [bs, A, cams, S, P, G]
    const Dims& fd = inputs[0].dims;
    if (wd.nbDims < 6 || fd.nbDims < 3) return 0;
    const long long nA = (long long)wd.d[0] * wd.d[1];
    const long long G = wd.d[5];
    const long long NCSP = (long long)wd.d[2] * wd.d[3] * wd.d[4];
    const long long C = fd.d[2];
    const size_t v8 = (size_t)(((nA * G * 4) + 255) & ~255ll) +
                      (size_t)(nA * G * NCSP) * 32;
    const size_t v11 = DFA_V11_NS::ws_need_v11(nA, NCSP, C);
    return v8 > v11 ? v8 : v11;
  }

  int32_t enqueue(PluginTensorDesc const* inputDesc,
                  PluginTensorDesc const* outputDesc,
                  void const* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    (void)outputDesc;
    const Dims& fd = inputDesc[0].dims;
    const Dims& sd = inputDesc[1].dims;
    const Dims& wd = inputDesc[4].dims;  // [bs, A, cams, S, P, G]
    if (fd.nbDims < 3 || sd.nbDims < 2 || wd.nbDims < 6) return -1;
    const int bs = fd.d[0];
    const int num_feat = fd.d[1];
    const int C = fd.d[2];
    const int cams = wd.d[2];
    const int S = wd.d[3];
    const int A = wd.d[1];
    const int P = wd.d[4];
    const int G = wd.d[5];
    const bool idx_i32 = (inputDesc[1].type == DataType::kINT32 ||
                          inputDesc[2].type == DataType::kINT32);
    const size_t ws_size = getWorkspaceSize(inputDesc, 5, outputDesc, 1);
    // 引擎由 v3 (getWorkspaceSize=0) 构建, TRT 传 nullptr —— 实例自分配
    void* ws = workspace;
    size_t ws_cap = ws_size;
    if (ws_size > 0 && (ws == nullptr || ws_cap < ws_size)) {
      if (own_cap_ < ws_size) {
        if (own_ws_) cudaFree(own_ws_);
        own_ws_ = nullptr;
        if (cudaMalloc(&own_ws_, ws_size) == cudaSuccess)
          own_cap_ = ws_size;
      }
      if (own_ws_) {
        ws = own_ws_;
        ws_cap = own_cap_;
      }
    }
    static const bool dbg_ = [] {
      const char* e = getenv("DFA_V11_DEBUG");
      return e && e[0] == '1';
    }();
    if (dbg_)
      fprintf(stderr,
              "[dfa_v11] A=%d P=%d G=%d C=%d ws=%p cap=%zu\n",
              A, P, G, C, ws, ws_cap);

    const long long nA = (long long)bs * A;
    const long long NCSP = (long long)cams * S * P;
    const size_t need11 = DFA_V11_NS::ws_need_v11(nA, NCSP, C);
#define DFA_V11_TRY(IT)                                                       \
  do {                                                                        \
    if (ws != nullptr && ws_cap >= need11) {                                  \
      int rc = DFA_V11_NS::dfa_launch_v11<IT>(                                \
          (__half*)outputs[0], (const __half*)inputs[0], (const IT*)inputs[1],\
          (const IT*)inputs[2], (const __half*)inputs[3],                     \
          (const __half*)inputs[4], bs, cams, num_feat, C, S, A, P, G,        \
          ws, ws_cap, stream);                                                \
      if (rc == 0) return 0;                                                  \
      if (dbg_) fprintf(stderr, "[dfa_v11] rc=%d, fallthrough\n", rc);        \
      if (rc != 1) return -1;                                                 \
    }                                                                         \
  } while (0)

    if (idx_i32) {
      DFA_V11_TRY(int);
#ifdef DFA_V11_MAYBE_FALLBACK
      int rc = DFA_V8_CALL_NS::dfa_launch_v8<int>(
          (__half*)outputs[0], (const __half*)inputs[0], (const int*)inputs[1],
          (const int*)inputs[2], (const __half*)inputs[3],
          (const __half*)inputs[4], bs, cams, num_feat, C, S, A, P, G,
          ws, ws_cap, stream);
      if (rc == 0) return 0;
      rc = DFA_V3_CALL_NS::dfa_launch_h2<int>(
          (__half*)outputs[0], (const __half*)inputs[0], (const int*)inputs[1],
          (const int*)inputs[2], (const __half*)inputs[3],
          (const __half*)inputs[4], bs, cams, num_feat, C, S, A, P, G,
          stream);
      if (rc == 0) return 0;
#endif
      return -1;
    } else {
      DFA_V11_TRY(float);
#ifdef DFA_V11_MAYBE_FALLBACK
      int rc = DFA_V8_CALL_NS::dfa_launch_v8<float>(
          (__half*)outputs[0], (const __half*)inputs[0],
          (const float*)inputs[1], (const float*)inputs[2],
          (const __half*)inputs[3], (const __half*)inputs[4], bs, cams,
          num_feat, C, S, A, P, G, ws, ws_cap, stream);
      if (rc == 0) return 0;
      rc = DFA_V3_CALL_NS::dfa_launch_h2<float>(
          (__half*)outputs[0], (const __half*)inputs[0],
          (const float*)inputs[1], (const float*)inputs[2],
          (const __half*)inputs[3], (const __half*)inputs[4], bs, cams,
          num_feat, C, S, A, P, G, stream);
      if (rc == 0) return 0;
#endif
      return -1;
    }
#undef DFA_V11_TRY
  }

 private:
  char ns_[64] = {0};
  void* own_ws_ = nullptr;
  size_t own_cap_ = 0;
};

class DfaPluginV11Creator : public IPluginCreator {
 public:
  DfaPluginV11Creator() {
    fc_.nbFields = 0;
    fc_.fields = nullptr;
  }
  ~DfaPluginV11Creator() override = default;
  const char* getPluginName() const noexcept override {
    return "DeformableAggregation";
  }
  const char* getPluginVersion() const noexcept override { return "1"; }
  PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
  IPluginV2* createPlugin(const char* name,
                          PluginFieldCollection const* fc) noexcept override {
    (void)name; (void)fc;
    auto* p = new DfaPluginV11();
    p->setPluginNamespace(ns_);
    return p;
  }
  IPluginV2* deserializePlugin(const char* name, const void* serialData,
                               size_t serialLength) noexcept override {
    (void)name;
    auto* p = new DfaPluginV11(serialData, serialLength);
    p->setPluginNamespace(ns_);
    return p;
  }
  void setPluginNamespace(const char* ns) noexcept override {
    snprintf(ns_, sizeof(ns_), "%s", ns ? ns : "");
  }
  const char* getPluginNamespace() const noexcept override { return ns_; }

 private:
  char ns_[64] = {0};
  PluginFieldCollection fc_;
};

}  // namespace DFA_TRT11_NS

#ifndef DFA_NO_REGISTER
namespace DFA_REG11_NS {
struct Registrar {
  Registrar() {
    getPluginRegistry()->registerCreator(*new DFA_TRT11_NS::DfaPluginV11Creator(),
                                         "SparseDrive");
    getPluginRegistry()->registerCreator(*new DFA_TRT11_NS::DfaPluginV11Creator(),
                                         "");
  }
};
static Registrar g_registrar;
}
#endif
