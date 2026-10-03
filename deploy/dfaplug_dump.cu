// ============================================================================
// DeformableAggregation TRT plugin v3 + DUMP 钩子 (调试专用)
// 与 libdfaplug_v3.so 语义/名字/版本完全一致; 设置环境变量 DFA_DUMP_DIR 时,
// 每次 enqueue 把全部输入/输出原样落盘 (真实引擎 IO, 无布局歧义):
//   dfa<k>_feat.f16  dfa<k>_shape.bin  dfa<k>_ssi.bin  dfa<k>_loc.f16
//   dfa<k>_log.f16   dfa<k>_out.f16    manifest.txt 追加行:
//   "<k> <bs> <num_feat> <C> <A> <cams> <S> <P> <G> <idx_dtype> <feat_ptr>"
// DFA_DUMP_MAX 限制落盘调用数 (默认 24)。不设 env 时零开销 (一次 getenv)。
// 用法: DFA_DUMP_DIR=/path DFA_DUMP_MAX=12 run_engine e_T6.engine
//         /usr/local/lib/libdfaplug_dump.so in in --dump out
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <atomic>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

using namespace nvinfer1;

namespace dfa_trt {

static std::atomic<int> g_dfa_call(0);

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
      if (rc == 0) {
        dfa_dump(inputDesc, inputs, outputs, stream);
        return 0;
      }
      if (rc != 1) return -1;
    } else {
      int rc = dfa_kernel::dfa_launch_h2<float>(
          (__half*)outputs[0], (const __half*)inputs[0],
          (const float*)inputs[1], (const float*)inputs[2],
          (const __half*)inputs[3], (const __half*)inputs[4], bs, cams,
          num_feat, C, S, A, P, G, stream);
      if (rc == 0) {
        dfa_dump(inputDesc, inputs, outputs, stream);
        return 0;
      }
      return -1;
    }
    return -1;
  }

 private:
  void dfa_dump(PluginTensorDesc const* inputDesc, void const* const* inputs,
                void* const* outputs, cudaStream_t stream) {
    const char* dd = getenv("DFA_DUMP_DIR");
    if (!dd) return;
    static const int max_calls = getenv("DFA_DUMP_MAX")
                                     ? atoi(getenv("DFA_DUMP_MAX")) : 24;
    const int k = g_dfa_call.fetch_add(1);
    if (k >= max_calls) return;
    const Dims& fd = inputDesc[0].dims;
    const Dims& sd = inputDesc[1].dims;
    const Dims& wd = inputDesc[4].dims;
    auto prod_dims = [](const Dims& d) {
      long long n = 1;
      for (int i = 0; i < d.nbDims; ++i) n *= d.d[i];
      return n;
    };
    auto cpy = [&](const void* dev, long long bytes, const char* name) {
      char p[512];
      snprintf(p, sizeof(p), "%s/dfa%d_%s", dd, k, name);
      FILE* f = fopen(p, "wb");
      if (!f) return;
      std::vector<char> h((size_t)bytes);
      cudaMemcpyAsync(h.data(), dev, (size_t)bytes, cudaMemcpyDeviceToHost,
                      stream);
      cudaStreamSynchronize(stream);
      fwrite(h.data(), 1, (size_t)bytes, f);
      fclose(f);
    };
    char mp[512];
    snprintf(mp, sizeof(mp), "%s/manifest.txt", dd);
    if (FILE* fm = fopen(mp, "a")) {
      fprintf(fm,
              "%d bs=%d num_feat=%d C=%d A=%d cams=%d S=%d P=%d G=%d "
              "idxt=%d featptr=%p shape_dims=%d,%d,%d\n",
              k, fd.d[0], fd.d[1], fd.d[2], wd.d[1], wd.d[2], wd.d[3],
              wd.d[4], wd.d[5], (int)inputDesc[1].type, inputs[0],
              sd.d[0], sd.nbDims > 1 ? sd.d[1] : 0,
              sd.nbDims > 2 ? sd.d[2] : 0);
      fclose(fm);
    }
    const bool i32 = (inputDesc[1].type == DataType::kINT32);
    const long long tsz = 2;  // half
    cpy(inputs[0], prod_dims(fd) * tsz, "feat.f16");
    cpy(inputs[1], prod_dims(sd) * 4, "shape.bin");
    cpy(inputs[2], prod_dims(inputDesc[2].dims) * 4, "ssi.bin");
    cpy(inputs[3], prod4(wd.d[0], wd.d[1], wd.d[4], wd.d[2]) * 2 * tsz,
        "loc.f16");
    cpy(inputs[4], prod_dims(wd) * tsz, "log.f16");
    cpy(outputs[0], prod4(fd.d[0], wd.d[1], fd.d[2], 1) * tsz, "out.f16");
    (void)i32;
  }

  static long long prod4(int a, int b, int c, int d) {
    return (long long)a * b * c * d;
  }
  static long long prod(const Dims& d) {
    long long n = 1;
    for (int i = 0; i < d.nbDims; ++i) n *= d.d[i];
    return n;
  }

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
