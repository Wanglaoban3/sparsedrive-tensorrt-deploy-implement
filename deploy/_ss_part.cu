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
