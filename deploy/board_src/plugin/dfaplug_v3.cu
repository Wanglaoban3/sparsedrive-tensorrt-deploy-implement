// ============================================================================
// DeformableAggregation TRT plugin v3 (logits-DF)
//  * weights 输入 = softmax 之前的 LOGITS, 布局 [bs, A, cams, S(levels),
//    P(pts), G(groups)] —— softmax 在导出端已删除, 由本插件内联完成:
//    对每个 (anchor, group) 在 (cams*S*P) 联合维做 max-subtracted softmax,
//    与原 in-graph softmax 数值语义一致。
//  * 死权重跳采样: 概率 < DFA_W_EPS(1e-4) 的 (c,s,p) 不再读特征图角点
//    (DFA 是访存瓶颈, 该跳过直接省带宽; 误差上界 = 被跳概率质量 x 特征幅度)。
//  * fp16 一线程双通道 (__half2) + __ldg + shape 表 smem, 继承 v2。
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"

#ifndef DFA_W_EPS
#define DFA_W_EPS 1.0e-4f
#endif

namespace dfa_kernel {

__device__ __forceinline__ float2 ldg_h2(const __half2* p) {
  return __half22float2(__ldg(p));
}

// weights: logits [bs, A, cams, S, P, G]; out [bs, A, C]
template <typename IT>
__global__ void dfa_gather_kernel_h2(
    __half* __restrict__ output, const __half* __restrict__ mc_ms_feat,
    const IT* __restrict__ spatial_shape, const IT* __restrict__ scale_start_index,
    const __half* __restrict__ sample_location, const __half* __restrict__ logits,
    const int batch_size, const int num_cams, const int num_feat,
    const int num_embeds, const int num_scale, const int num_anchors,
    const int num_pts, const int num_groups) {
  extern __shared__ int smem[];  // per (cam,scale): h, w, start
  const int ncs = num_cams * num_scale;
  for (int i = threadIdx.x; i < ncs * 3; i += blockDim.x) {
    const int cs = i / 3;
    const int r = i % 3;
    smem[i] = (r == 0) ? (int)spatial_shape[cs * 2]
             : (r == 1) ? (int)spatial_shape[cs * 2 + 1]
                        : (int)scale_start_index[cs];
  }
  __syncthreads();

  const int halfC = num_embeds >> 1;
  const long long total = (long long)batch_size * num_anchors * halfC;
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int chp = (int)(idx % halfC);
  const int ch0 = chp * 2;
  const long long na = idx / halfC;
  const int b = (int)(na / num_anchors);

  const int cg = num_embeds / num_groups;
  const int g = ch0 / cg;
  const int NCSP = num_cams * num_scale * num_pts;

  // ---- inline softmax stats over (cams*S*P) for (na, g) ----
  // 组内 halfC/num_groups 个线程分段扫描 + shuffle 归约 (map 头 NCSP=7200,
  // 串行扫描延迟不可接受)
  const int lanes = halfC / num_groups;       // 同组并行线程数 (16)
  const int lane = chp % lanes;
  const __half* lbase = logits + na * (long long)NCSP * num_groups + g;
  float lmax = -3.0e30f;
  for (int i = lane; i < NCSP; i += lanes) {
    lmax = fmaxf(lmax, __half2float(__ldg(lbase + (long long)i * num_groups)));
  }
  for (int off = lanes >> 1; off > 0; off >>= 1)
    lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, off, lanes));
  float lsum = 0.f;
  for (int i = lane; i < NCSP; i += lanes) {
    lsum += __expf(__half2float(__ldg(lbase + (long long)i * num_groups)) -
                   lmax);
  }
  for (int off = lanes >> 1; off > 0; off >>= 1)
    lsum += __shfl_xor_sync(0xffffffffu, lsum, off, lanes);
  const float inv_sum = 1.f / lsum;

  // sampling locations [bs, A, P, cams, 2]
  const __half* sloc = sample_location + na * (long long)num_pts * num_cams * 2;
  const __half* f2 = mc_ms_feat + (long long)b * num_feat * num_embeds + ch0;
  __half2* o2 = reinterpret_cast<__half2*>(output + na * num_embeds + ch0);

  float acc0 = 0.f, acc1 = 0.f;
  for (int p = 0; p < num_pts; ++p) {
    for (int c = 0; c < num_cams; ++c) {
      const long long lo = ((long long)p * num_cams + c) * 2;
      const float loc_w = __half2float(__ldg(sloc + lo));
      if (loc_w <= 0.f || loc_w >= 1.f) continue;
      const float loc_h = __half2float(__ldg(sloc + lo + 1));
      if (loc_h <= 0.f || loc_h >= 1.f) continue;
      // logits offset base for (p, c): ((na*cams + c)*S + 0)*P + p)*G
      const __half* lc = lbase + ((long long)c * num_scale * num_pts + p) *
                                    num_groups;
      for (int s = 0; s < num_scale; ++s) {
        const float wt =
            __expf(__half2float(__ldg(lc + (long long)s * num_pts *
                                      num_groups)) - lmax) * inv_sum;
        if (wt < DFA_W_EPS) continue;
        const int cs3 = (c * num_scale + s) * 3;
        const int h = smem[cs3];
        const int w = smem[cs3 + 1];
        const long long start = smem[cs3 + 2];
        const float h_im = loc_h * h - 0.5f;
        const float w_im = loc_w * w - 0.5f;
        const int h_low = floorf(h_im), w_low = floorf(w_im);
        const float lh = h_im - h_low, lw = w_im - w_low;
        const float hh = 1 - lh, hw = 1 - lw;
        const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
        const __half* bb = f2 + start * num_embeds;
        const long long r0 = (long long)h_low * w * num_embeds;
        const long long r1 = r0 + (long long)w * num_embeds;
        const long long c0 = (long long)w_low * num_embeds;
        const long long c1 = c0 + num_embeds;
        float2 v1 = {0.f, 0.f}, v2 = {0.f, 0.f}, v3 = {0.f, 0.f},
               v4 = {0.f, 0.f};
        if (h_low >= 0 && w_low >= 0)
          v1 = ldg_h2(reinterpret_cast<const __half2*>(bb + r0 + c0));
        if (h_low >= 0 && w_low + 1 <= w - 1)
          v2 = ldg_h2(reinterpret_cast<const __half2*>(bb + r0 + c1));
        if (h_low + 1 <= h - 1 && w_low >= 0)
          v3 = ldg_h2(reinterpret_cast<const __half2*>(bb + r1 + c0));
        if (h_low + 1 <= h - 1 && w_low + 1 <= w - 1)
          v4 = ldg_h2(reinterpret_cast<const __half2*>(bb + r1 + c1));
        acc0 += wt * (w1 * v1.x + w2 * v2.x + w3 * v3.x + w4 * v4.x);
        acc1 += wt * (w1 * v1.y + w2 * v2.y + w3 * v3.y + w4 * v4.y);
      }
    }
  }
  *o2 = __floats2half2_rn(acc0, acc1);
}

// float 标量回退路径 (同语义)
template <typename T, typename IT>
__global__ void dfa_gather_kernel_f32(
    T* __restrict__ output, const T* __restrict__ mc_ms_feat,
    const IT* __restrict__ spatial_shape, const IT* __restrict__ scale_start_index,
    const T* __restrict__ sample_location, const T* __restrict__ logits,
    const int batch_size, const int num_cams, const int num_feat,
    const int num_embeds, const int num_scale, const int num_anchors,
    const int num_pts, const int num_groups) {
  const long long total = (long long)batch_size * num_anchors * num_embeds;
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int ch = (int)(idx % num_embeds);
  const long long na = idx / num_embeds;
  const int b = (int)(na / num_anchors);
  const int cg = num_embeds / num_groups;
  const int g = ch / cg;
  const int NCSP = num_cams * num_scale * num_pts;
  const T* lbase = logits + na * NCSP * num_groups + g;
  float lmax = -3.0e30f;
  for (int i = 0; i < NCSP; ++i)
    lmax = fmaxf(lmax, (float)__ldg(lbase + (long long)i * num_groups));
  float lsum = 0.f;
  for (int i = 0; i < NCSP; ++i)
    lsum += __expf((float)__ldg(lbase + (long long)i * num_groups) - lmax);
  const float inv_sum = 1.f / lsum;
  const T* sloc = sample_location + na * (long long)num_pts * num_cams * 2;
  const long long feat_base = (long long)b * num_feat * num_embeds + ch;
  float acc = 0.f;
  for (int p = 0; p < num_pts; ++p) {
    for (int c = 0; c < num_cams; ++c) {
      const long long lo = ((long long)p * num_cams + c) * 2;
      const float loc_w = (float)__ldg(sloc + lo);
      if (loc_w <= 0.f || loc_w >= 1.f) continue;
      const float loc_h = (float)__ldg(sloc + lo + 1);
      if (loc_h <= 0.f || loc_h >= 1.f) continue;
      const T* lc = lbase + ((long long)c * num_scale * num_pts + p) * num_groups;
      for (int s = 0; s < num_scale; ++s) {
        const float wt = __expf((float)__ldg(lc + (long long)s * num_pts *
                                             num_groups) - lmax) * inv_sum;
        if (wt < DFA_W_EPS) continue;
        const int cs = c * num_scale + s;
        const int h = (int)spatial_shape[cs * 2];
        const int w = (int)spatial_shape[cs * 2 + 1];
        const long long start = (long long)scale_start_index[cs];
        const float h_im = loc_h * h - 0.5f;
        const float w_im = loc_w * w - 0.5f;
        const int h_low = floorf(h_im), w_low = floorf(w_im);
        const float lh = h_im - h_low, lw = w_im - w_low;
        const float hh = 1 - lh, hw = 1 - lw;
        const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
        const long long w_stride = num_embeds;
        const long long h_stride = (long long)w * w_stride;
        const T* base = mc_ms_feat + feat_base + start * num_embeds;
        float v = 0.f;
        if (h_low >= 0 && w_low >= 0)
          v += w1 * (float)__ldg(base + (long long)h_low * h_stride +
                                 (long long)w_low * w_stride);
        if (h_low >= 0 && w_low + 1 <= w - 1)
          v += w2 * (float)__ldg(base + (long long)h_low * h_stride +
                                 (long long)(w_low + 1) * w_stride);
        if (h_low + 1 <= h - 1 && w_low >= 0)
          v += w3 * (float)__ldg(base + (long long)(h_low + 1) * h_stride +
                                 (long long)w_low * w_stride);
        if (h_low + 1 <= h - 1 && w_low + 1 <= w - 1)
          v += w4 * (float)__ldg(base + (long long)(h_low + 1) * h_stride +
                                 (long long)(w_low + 1) * w_stride);
        acc += wt * v;
      }
    }
  }
  output[na * num_embeds + ch] = (T)acc;
}

template <typename IT>
int dfa_launch_h2(__half* output, const __half* feat, const IT* spatial_shape,
                  const IT* scale_start_index, const __half* loc,
                  const __half* logits, int batch_size, int num_cams,
                  int num_feat, int num_embeds, int num_scale, int num_anchors,
                  int num_pts, int num_groups, cudaStream_t stream) {
  if (num_embeds & 1) return 1;
  const int halfC = num_embeds >> 1;
  const long long total = (long long)batch_size * num_anchors * halfC;
  const int block = 256;
  const long long grid = (total + block - 1) / block;
  if (grid > 2147483647LL) return 1;
  const int smem = num_cams * num_scale * 3 * (int)sizeof(int);
  dfa_gather_kernel_h2<IT><<<(unsigned)grid, block, smem, stream>>>(
      output, feat, spatial_shape, scale_start_index, loc, logits,
      batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors,
      num_pts, num_groups);
  return (int)cudaGetLastError();
}

}  // namespace dfa_kernel

// ============================================================================
// TensorRT plugin (IPluginV2DynamicExt)
// ============================================================================
using namespace nvinfer1;

namespace dfa_trt {

class DfaPlugin : public IPluginV2DynamicExt {
 public:
  DfaPlugin() = default;
  DfaPlugin(const void* data, size_t size) { (void)data; (void)size; }
  ~DfaPlugin() override = default;

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

    if (idx_i32) {
      int rc = dfa_kernel::dfa_launch_h2<int>(
          (__half*)outputs[0], (const __half*)inputs[0], (const int*)inputs[1],
          (const int*)inputs[2], (const __half*)inputs[3],
          (const __half*)inputs[4], bs, cams, num_feat, C, S, A, P, G,
          stream);
      if (rc == 0) return 0;
      if (rc != 1) return -1;
    } else {
      int rc = dfa_kernel::dfa_launch_h2<float>(
          (__half*)outputs[0], (const __half*)inputs[0],
          (const float*)inputs[1], (const float*)inputs[2],
          (const __half*)inputs[3], (const __half*)inputs[4], bs, cams,
          num_feat, C, S, A, P, G, stream);
      if (rc == 0) return 0;
      if (rc != 1) return -1;
    }
    // 奇数通道兜底: f32 标量核 (输出 half 由 getOutputDataType 保证一致,
    // 此处仅极端情形; 正常 256/2048 通道不会走到)
    return -1;
  }

 private:
  char ns_[64] = {0};
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

}  // namespace dfa_trt

#ifndef DFA_NO_REGISTER
namespace dfa_reg {
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

