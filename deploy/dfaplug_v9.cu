// ============================================================================
// DeformableAggregation TRT plugin v8 (logits 语义, plan+gather 两内核)
// 与 v3 相同的对外语义:
//  * weights 输入 = softmax 之前的 LOGITS [bs, A, cams, S, P, G], 插件内联
//    max-subtracted softmax;
//  * loc 无效 (w/h 不在 (0,1)) 或 softmax 概率 < DFA_W_EPS(1e-4) 的
//    (c,s,p,g) 不参与特征聚合 (误差上界与 v3 相同);
//  * 双线性四角点边界判定与 v3 逐条一致;
//  * fp16 IO, 输出 [bs, A, C]。
//
// 与 v3 (单内核, 每线程 (anchor,通道) 全量扫 NCSP) 的区别:
//  K1 plan  (每 anchor 一个 block):
//    pass1 寄存器 online-softmax (单遍同时得 max/sum, 运行修正),
//          warp 蝶形 + block 串行归约 (确定性);
//    pass2 重读 logits, 对每个 loc 合法且 wt>=eps 的 (c,s,p,g) 生成一条
//          32B 条目 {4×int32 角点绝对元素偏移(无效角点=0), 4×fp32 折叠权重
//          wk*wt}, atomicAdd 进 (anchor,group) 专属紧凑列表 (workspace)。
//  K2 gather (每 (anchor,group) 一个 warp, lane = 组内通道):
//    逐条目 2×int4 广播读 + 4× 连续 half 合并读 (32 lane = 64B) + FMA,
//    无分支 (死角点权重 0 + 偏移钳 0, 读 line0 无害)。
//
// 生效条件: G<=8, C%G==0, C/G<=32, logits 行宽 G==8 时按 uint4 向量读,
// workspace = nA*G*(4B + NCSP*32B) 可满足; 否则回落 v3 内核 (同一 .so)。
// 数值差异: softmax 归约树形与 v3 不同 (ulp 级), wt 边界条目可能翻转;
// 聚合按列表序累加 (顺序原子, 逐次运行 ulp 级抖动)。
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

// 命名空间可通过宏改名 (harness 双 include 用); 缺省即正式名字
#ifndef DFA_V8_NS
#define DFA_V8_NS dfa_kernel
#endif
#ifndef DFA_TRT_NS
#define DFA_TRT_NS dfa_trt
#endif
#ifndef DFA_REG_NS
#define DFA_REG_NS dfa_reg
#endif

// ---- v3 内核作为回落路径 (仅内核, 不注册 creator) --------------------------
#ifndef DFA_V8_NO_FALLBACK
#define DFA_NO_REGISTER
#define dfa_kernel kb3
#define dfa_trt tb3
#define dfa_reg rb3
#include "dfaplug_v3.cu"
#undef dfa_kernel
#undef dfa_trt
#undef dfa_reg
#undef DFA_NO_REGISTER
#endif

namespace DFA_V8_NS {

struct Entry {         // 32B, 16B 对齐
  int off[4];          // 角点绝对元素偏移; 无效角点 = 0 (权重也为 0)
  float w[4];          // 折叠权重 wk * wt (fp32)
};

__device__ __forceinline__ float2 ldg_h2(const __half2* p) {
  return __half22float2(__ldg(p));
}

// ---- K1: 每 anchor 一个 block, 产出紧凑条目列表 ----------------------------
template <typename IT>
__global__ void __launch_bounds__(256)
dfa_plan_kernel(int* __restrict__ counts, Entry* __restrict__ entries,
                const IT* __restrict__ spatial_shape,
                const IT* __restrict__ scale_start_index,
                const __half* __restrict__ sample_location,
                const __half* __restrict__ logits,
                const int batch_size, const int num_cams, const int num_feat,
                const int num_embeds, const int num_scale,
                const int num_anchors, const int num_pts,
                const int num_groups) {
  const int na = blockIdx.x;               // b*A + a
  const int b = na / num_anchors;
  const int G = num_groups;
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

  // ---- pass1: online softmax stats (max,sum 同时, 单遍) ----
  const __half* lrow = logits + (long long)na * NCSP * G;
  float m[8], ss[8];
#pragma unroll
  for (int g = 0; g < 8; ++g) { m[g] = -3.0e30f; ss[g] = 0.f; }
  for (int i = tid; i < NCSP; i += blockDim.x) {
    if (G == 8) {
      const uint4 u = __ldg(reinterpret_cast<const uint4*>(lrow + (long long)i * 8));
      const __half* hh = reinterpret_cast<const __half*>(&u);
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        const float v = __half2float(hh[g]);
        const float mn = fmaxf(m[g], v);
        ss[g] = ss[g] * __expf(m[g] - mn) + __expf(v - mn);
        m[g] = mn;
      }
    } else {
      for (int g = 0; g < G; ++g) {
        const float v = __half2float(__ldg(lrow + (long long)i * G + g));
        const float mn = fmaxf(m[g], v);
        ss[g] = ss[g] * __expf(m[g] - mn) + __expf(v - mn);
        m[g] = mn;
      }
    }
  }
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    if (g >= G) break;
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

  // ---- pass2: 生成条目 ----
  const int SP = num_scale * num_pts;
  const __half* sloc = sample_location + (long long)na * num_pts * num_cams * 2;
  const long long fbase = (long long)b * num_feat * num_embeds;
  for (int i = tid; i < NCSP; i += blockDim.x) {
    const int c = i / SP;
    const int rem = i - c * SP;
    const int s = rem / num_pts;
    const int p = rem - s * num_pts;
    const long long lo = ((long long)p * num_cams + c) * 2;
    const float2 lw2 = ldg_h2(reinterpret_cast<const __half2*>(sloc + lo));
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
    float v[8];
    if (G == 8) {
      const uint4 u = __ldg(reinterpret_cast<const uint4*>(lrow + (long long)i * 8));
      const __half* hh = reinterpret_cast<const __half*>(&u);
#pragma unroll
      for (int g = 0; g < 8; ++g) v[g] = __half2float(hh[g]);
    } else {
      for (int g = 0; g < G; ++g)
        v[g] = __half2float(__ldg(lrow + (long long)i * G + g));
    }
#pragma unroll
    for (int g = 0; g < 8; ++g) {
      if (g >= G) break;
      const float wt = __expf(v[g] - s_max[g]) * inv[g];
      if (wt < DFA_W_EPS) continue;
      const int slot = atomicAdd(&counts[na * G + g], 1);
      Entry* e = entries + ((long long)na * G + g) * NCSP + slot;
      int4 a4; a4.x = o0; a4.y = o1; a4.z = o2; a4.w = o3;
      *reinterpret_cast<int4*>(&e->off[0]) = a4;
      int4 b4; b4.x = __float_as_int(w1 * wt); b4.y = __float_as_int(w2 * wt);
      b4.z = __float_as_int(w3 * wt); b4.w = __float_as_int(w4 * wt);
      *reinterpret_cast<int4*>(&e->w[0]) = b4;
    }
  }
}

// ---- K2: 每 (anchor, group) 一个 warp, lane = 组内通道 ---------------------
__global__ void __launch_bounds__(256)
dfa_gather_v8_kernel(__half* __restrict__ output,
                     const __half* __restrict__ feat,
                     const int* __restrict__ counts,
                     const Entry* __restrict__ entries,
                     const int batch_size, const int num_anchors,
                     const int num_embeds, const int num_groups,
                     const int NCSP) {
  const int cg = num_embeds / num_groups;
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int na = gw / num_groups;
  const int g = gw - na * num_groups;
  if (na >= batch_size * num_anchors) return;
  const int lane = threadIdx.x & 31;
  if (lane >= cg) return;
  const int cnt = counts[na * num_groups + g];
  const Entry* list = entries + ((long long)na * num_groups + g) * NCSP;
  const int ch = g * cg + lane;
  float acc = 0.f;
  for (int k = 0; k < cnt; ++k) {
    const Entry* e = list + k;
    const int4 o4 = __ldg(reinterpret_cast<const int4*>(&e->off[0]));
    const int4 w4i = __ldg(reinterpret_cast<const int4*>(&e->w[0]));
    acc = fmaf(__int_as_float(w4i.x),
               __half2float(__ldg(feat + (long long)o4.x + ch)), acc);
    acc = fmaf(__int_as_float(w4i.y),
               __half2float(__ldg(feat + (long long)o4.y + ch)), acc);
    acc = fmaf(__int_as_float(w4i.z),
               __half2float(__ldg(feat + (long long)o4.z + ch)), acc);
    acc = fmaf(__int_as_float(w4i.w),
               __half2float(__ldg(feat + (long long)o4.w + ch)), acc);
  }
  output[(long long)na * num_embeds + ch] = __float2half(acc);
}

// plan 阶段 (memset + K1); gather 阶段单独可调, 便于分相计时
template <typename IT>
int dfa_plan_v8(const IT* spatial_shape,
                const IT* scale_start_index, const __half* loc,
                const __half* logits, int batch_size, int num_cams,
                int num_feat, int num_embeds, int num_scale,
                int num_anchors, int num_pts, int num_groups,
                void* ws, size_t ws_size, cudaStream_t stream) {
  if (batch_size < 1 || num_cams < 1 || num_scale < 1 || num_pts < 1 ||
      num_anchors < 1 || num_groups < 1 || num_groups > 8) return 1;
  if (num_embeds % num_groups) return 1;
  if (num_embeds / num_groups > 32) return 1;
  const int NCSP = num_cams * num_scale * num_pts;
  const int nA = batch_size * num_anchors;
  const size_t cnt_bytes = (size_t)nA * num_groups * sizeof(int);
  const size_t ent_off = (cnt_bytes + (size_t)255) & ~(size_t)255;
  const size_t ent_bytes = (size_t)nA * num_groups * NCSP * sizeof(Entry);
  if (ws_size < ent_off + ent_bytes) return 1;
  int* counts = reinterpret_cast<int*>(ws);
  Entry* entries = reinterpret_cast<Entry*>(reinterpret_cast<char*>(ws) + ent_off);

  cudaError_t e0 = cudaMemsetAsync(counts, 0, cnt_bytes, stream);
  if (e0 != cudaSuccess) return (int)e0;
  const size_t smem = (16 + 128 + (size_t)num_cams * num_scale * 3) * sizeof(float);
  dfa_plan_kernel<IT><<<nA, 256, smem, stream>>>(
      counts, entries, spatial_shape, scale_start_index, loc, logits,
      batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors,
      num_pts, num_groups);
  return (int)cudaGetLastError();
}

int dfa_gather_v8(__half* output, const __half* feat, void* ws,
                  size_t ws_size, int batch_size, int num_anchors,
                  int num_feat, int num_embeds, int num_cams, int num_scale,
                  int num_pts, int num_groups, cudaStream_t stream) {
  const int NCSP = num_cams * num_scale * num_pts;
  const int nA = batch_size * num_anchors;
  const size_t cnt_bytes = (size_t)nA * num_groups * sizeof(int);
  const size_t ent_off = (cnt_bytes + (size_t)255) & ~(size_t)255;
  if (ws_size < ent_off) return 1;
  int* counts = reinterpret_cast<int*>(ws);
  const Entry* entries = reinterpret_cast<const Entry*>(
      reinterpret_cast<char*>(ws) + ent_off);
  const int warps = nA * num_groups;
  const int grid2 = (warps * 32 + 255) / 256;
  dfa_gather_v8_kernel<<<grid2, 256, 0, stream>>>(
      output, feat, counts, entries, batch_size, num_anchors, num_embeds,
      num_groups, NCSP);
  return (int)cudaGetLastError();
}

template <typename IT>
int dfa_launch_v8(__half* output, const __half* feat, const IT* spatial_shape,
                  const IT* scale_start_index, const __half* loc,
                  const __half* logits, int batch_size, int num_cams,
                  int num_feat, int num_embeds, int num_scale,
                  int num_anchors, int num_pts, int num_groups,
                  void* ws, size_t ws_size, cudaStream_t stream) {
  int rc = dfa_plan_v8<IT>(spatial_shape, scale_start_index, loc, logits,
                           batch_size, num_cams, num_feat, num_embeds,
                           num_scale, num_anchors, num_pts, num_groups,
                           ws, ws_size, stream);
  if (rc != 0) return rc;
  return dfa_gather_v8(output, feat, ws, ws_size, batch_size, num_anchors,
                       num_feat, num_embeds, num_cams, num_scale, num_pts,
                       num_groups, stream);
}

}  // namespace DFA_V8_NS

// ============================================================================
// TensorRT plugin (IPluginV2DynamicExt) — 与 v3 相同的接口/名字/版本,
// enqueue 优先走 v8 内核, 不满足条件回落 v3。
// ============================================================================
using namespace nvinfer1;

namespace DFA_TRT_NS {

class DfaPlugin : public IPluginV2DynamicExt {
 public:
  DfaPlugin() = default;
  DfaPlugin(const void* data, size_t size) { (void)data; (void)size; }
  ~DfaPlugin() override {
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
    auto* p = new DfaPlugin();
    p->setPluginNamespace(ns_);
    return p;
  }

  void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
                       DynamicPluginTensorDesc const* out,
                       int32_t nbOutputs) noexcept override {
    (void)in; (void)nbInputs; (void)out; (void)nbOutputs;
  }

  // workspace 布局: [counts nA*G int][pad→256][entries nA*G*NCSP Entry(32B)]
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
    const size_t cnt = (size_t)(nA * G * (long long)sizeof(int));
    const size_t ent = (size_t)(nA * G * NCSP * 32);
    return ((cnt + 255) & ~(size_t)255) + ent;
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
    // 本引擎由 v3 插件 (getWorkspaceSize=0) 构建, TRT 只会传 nullptr;
    // 此时用实例自分配缓冲 (懒分配, 只增不减), 不重编 engine 也走 v8 路径。
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
      const char* e = getenv("DFA_V8_DEBUG");
      return e && e[0] == '1';
    }();
    if (dbg_)
      fprintf(stderr, "[dfa_v8] A=%d P=%d trt_ws=%p need=%zu use=%p cap=%zu\n",
              A, P, workspace, ws_size, ws, ws_cap);

#define DFA_V8_TRY(IT)                                                        \
  do {                                                                        \
    if (ws != nullptr && ws_cap >= ws_size) {                                 \
      int rc = DFA_V8_NS::dfa_launch_v8<IT>(                                  \
          (__half*)outputs[0], (const __half*)inputs[0], (const IT*)inputs[1],\
          (const IT*)inputs[2], (const __half*)inputs[3],                     \
          (const __half*)inputs[4], bs, cams, num_feat, C, S, A, P, G,        \
          ws, ws_cap, stream);                                                \
      if (rc == 0) return 0;                                                  \
      if (dbg_) fprintf(stderr, "[dfa_v8] v8 rc=%d, fallthrough\n", rc);      \
      if (rc != 1) return -1;                                                 \
    }                                                                         \
  } while (0)

    if (idx_i32) {
      DFA_V8_TRY(int);
#ifndef DFA_V8_NO_FALLBACK
      int rc = kb3::dfa_launch_h2<int>(
          (__half*)outputs[0], (const __half*)inputs[0], (const int*)inputs[1],
          (const int*)inputs[2], (const __half*)inputs[3],
          (const __half*)inputs[4], bs, cams, num_feat, C, S, A, P, G,
          stream);
      if (rc == 0) return 0;
#endif
      return -1;
    } else {
      DFA_V8_TRY(float);
#ifndef DFA_V8_NO_FALLBACK
      int rc = kb3::dfa_launch_h2<float>(
          (__half*)outputs[0], (const __half*)inputs[0],
          (const float*)inputs[1], (const float*)inputs[2],
          (const __half*)inputs[3], (const __half*)inputs[4], bs, cams,
          num_feat, C, S, A, P, G, stream);
      if (rc == 0) return 0;
#endif
      return -1;
    }
#undef DFA_V8_TRY
  }

 private:
  char ns_[64] = {0};
  void* own_ws_ = nullptr;  // TRT 不给 workspace 时的自分配缓冲
  size_t own_cap_ = 0;
};

class DfaPluginCreator : public IPluginCreator {
 public:
  DfaPluginCreator() {
    fc_.nbFields = 0;
    fc_.fields = nullptr;
  }
  ~DfaPluginCreator() override = default;
  const char* getPluginName() const noexcept override {
    return "DeformableAggregation";
  }
  const char* getPluginVersion() const noexcept override { return "1"; }
  PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
  IPluginV2* createPlugin(const char* name,
                          PluginFieldCollection const* fc) noexcept override {
    (void)name; (void)fc;
    auto* p = new DfaPlugin();
    p->setPluginNamespace(ns_);
    return p;
  }
  IPluginV2* deserializePlugin(const char* name, const void* serialData,
                               size_t serialLength) noexcept override {
    (void)name;
    auto* p = new DfaPlugin(serialData, serialLength);
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

}  // namespace DFA_TRT_NS

#ifndef DFA_NO_REGISTER
namespace DFA_REG_NS {
struct Registrar {
  Registrar() {
    getPluginRegistry()->registerCreator(*new dfa_trt::DfaPluginCreator(),
                                         "SparseDrive");
    getPluginRegistry()->registerCreator(*new dfa_trt::DfaPluginCreator(), "");
  }
};
static Registrar g_registrar;
}
#endif
// ============================================================================
// ScaleSoftmax TRT plugin (P5) —— 拼接在 dfaplug_v8.cu 之后组成 v9 .so。
//
// 背景: inner_attn 的 softmax 链 MatMul→Cast(f32)→Mul(scale)→Softmax(-1)
// →Cast(f16) 在 TRT 8.6 全图编译下每处物化 5 个 [1,H,Q,K] 中间张量
// (1×f16 + 3×f32 + 1×f16), 10 个 900-token 站点合计 ~1.7GB 读写 ≈ 9ms
// (P4 profile 实测)。本插件把 Cast/Mul/Softmax/Cast 四节点融成一个内核:
// 输入 TRT f16 GEMM 直出的 scores, f32 完成 scale+softmax, 输出 f16 概率,
// 流量降到 每站 2×f16 张量 (QK^T 写 + P 读) + PV 读。
//
// 输入: [0] scores f16 [1,H,Q,K] (K<=1024, 行由 warp 处理, 32 lane 均分)
//       [1] scale   f32 [1]    (图内 Pow 常量, 64 维站 0.125 / 32 维站 1/√32)
// 输出: f16 同形。
// 数值: 与参考链同序 (f16 scores → f32 ×scale → max-subtract softmax →
//       f16), 仅 exp/除法实现差 ulp; warp 蝶形归约定序, 逐次运行确定。
// ============================================================================
#ifndef SS_NO_REGISTER
namespace SS_NS {

// 一行一个 CTA (128 线程): 全合并 256B/指令读, 两轮 block 归约 (warp 蝶形
// + smem 跨 warp), 寄存器持行数据单遍完成 max/exp/sum/写。确定性定序。
// K <= 1024 → 每线程至多 8 个元素。
__global__ void scale_softmax_f16(const __half* __restrict__ x,
                                  const float* __restrict__ scale,
                                  __half* __restrict__ y, int rows, int K) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const __half* xr = x + (size_t)row * K;
  __half* yr = y + (size_t)row * K;
  const float sc = __ldg(scale);
  const int tid = threadIdx.x;
  const int nt = blockDim.x;
  constexpr int CH = 1024 / 128;  // 8
  float v[CH];
  float m = -3.4e38f;
#pragma unroll
  for (int j = 0; j < CH; ++j) {
    const int i = tid + j * nt;
    if (i < K) {
      const float f = __half2float(xr[i]) * sc;
      v[j] = f;
      m = fmaxf(m, f);
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  __shared__ float red[8];
  const int lane = tid & 31, warp = tid >> 5, nw = nt >> 5;
  if (lane == 0) red[warp] = m;
  __syncthreads();
  if (tid == 0) {
    float mm = red[0];
    for (int w = 1; w < nw; ++w) mm = fmaxf(mm, red[w]);
    red[0] = mm;
  }
  __syncthreads();
  m = red[0];
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < CH; ++j) {
    const int i = tid + j * nt;
    if (i < K) {
      const float e = expf(v[j] - m);
      v[j] = e;
      s += e;
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (lane == 0) red[warp] = s;
  __syncthreads();
  if (tid == 0) {
    float ss = red[0];
    for (int w = 1; w < nw; ++w) ss += red[w];
    red[0] = ss;
  }
  __syncthreads();
  const float inv = 1.f / red[0];
#pragma unroll
  for (int j = 0; j < CH; ++j) {
    const int i = tid + j * nt;
    if (i < K) yr[i] = __float2half(v[j] * inv);
  }
}

}  // namespace SS_NS

namespace SS_TRT_NS {

using namespace nvinfer1;

class ScaleSoftmaxPlugin : public IPluginV2DynamicExt {
 public:
  ScaleSoftmaxPlugin() = default;
  ScaleSoftmaxPlugin(const void* data, size_t size) { (void)data; (void)size; }
  ~ScaleSoftmaxPlugin() override = default;

  const char* getPluginType() const noexcept override { return "ScaleSoftmax"; }
  const char* getPluginVersion() const noexcept override { return "1"; }
  int32_t getNbOutputs() const noexcept override { return 1; }

  DimsExprs getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs,
                                int32_t nbInputs,
                                IExprBuilder& exprBuilder) noexcept override {
    (void)outputIndex; (void)nbInputs; (void)exprBuilder;
    return inputs[0];  // 同形
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
    if (pos == 1) return inOut[pos].type == DataType::kFLOAT;  // scale
    return inOut[pos].type == DataType::kHALF;                 // scores/out
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
    auto* p = new ScaleSoftmaxPlugin();
    p->setPluginNamespace(ns_);
    return p;
  }

  void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
                       DynamicPluginTensorDesc const* out,
                       int32_t nbOutputs) noexcept override {
    (void)in; (void)nbInputs; (void)out; (void)nbOutputs;
  }

  size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs,
                          PluginTensorDesc const* outputs,
                          int32_t nbOutputs) const noexcept override {
    (void)inputs; (void)nbInputs; (void)outputs; (void)nbOutputs;
    return 0;
  }

  int32_t enqueue(PluginTensorDesc const* inputDesc,
                  PluginTensorDesc const* outputDesc,
                  void const* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    (void)outputDesc; (void)workspace;
    const Dims& d = inputDesc[0].dims;
    if (d.nbDims != 4) return -1;
    const int H = d.d[1], Q = d.d[2], K = d.d[3];
    if (K > 1024 || K <= 0 || H <= 0 || Q <= 0) return -1;
    const int rows = H * Q;
    const int threads = 128;
    SS_NS::scale_softmax_f16<<<rows, threads, 0, stream>>>(
        static_cast<const __half*>(inputs[0]),
        static_cast<const float*>(inputs[1]),
        static_cast<__half*>(outputs[0]), rows, K);
    return cudaPeekAtLastError() != cudaSuccess ? -1 : 0;
  }

 private:
  char ns_[64] = {0};
};

class ScaleSoftmaxCreator : public IPluginCreator {
 public:
  ScaleSoftmaxCreator() {
    fc_.nbFields = 0;
    fc_.fields = nullptr;
  }
  ~ScaleSoftmaxCreator() override = default;
  const char* getPluginName() const noexcept override { return "ScaleSoftmax"; }
  const char* getPluginVersion() const noexcept override { return "1"; }
  PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
  IPluginV2* createPlugin(const char* name,
                          PluginFieldCollection const* fc) noexcept override {
    (void)name; (void)fc;
    auto* p = new ScaleSoftmaxPlugin();
    p->setPluginNamespace(ns_);
    return p;
  }
  IPluginV2* deserializePlugin(const char* name, const void* serialData,
                               size_t serialLength) noexcept override {
    (void)name;
    auto* p = new ScaleSoftmaxPlugin(serialData, serialLength);
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

}  // namespace SS_TRT_NS

namespace SS_REG_NS {
struct Registrar {
  Registrar() {
    getPluginRegistry()->registerCreator(*new SS_TRT_NS::ScaleSoftmaxCreator(),
                                         "SparseDrive");
    getPluginRegistry()->registerCreator(*new SS_TRT_NS::ScaleSoftmaxCreator(),
                                         "");
  }
};
static Registrar g_ss_registrar;
}
#endif  // SS_NO_REGISTER
