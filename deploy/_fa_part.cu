// ============================================================================
// FlashSDPA TRT plugin (P5b) —— 拼进 v10 .so (与 v8 DFA / ScaleSoftmax 共存)
//
// 动机 (P5 模块实测): inner_attn 6 节点链 QK^T MatMul → Cast(f32) → Mul(scale)
// → Softmax → Cast(f16) → PV MatMul 在板上全走退化 tactic: GEMM 无 tensor
// core (~1.6 TFLOPS), SS-only 插件路线被拆区碎片税吃掉 (+1.6ms)。本插件把
// 两个 GEMM + softmax 吸收为一个 wmma tensor-core 内核, 一站一片, 零碎片。
//
// 输入: [0] Q   f16 [1,H,Qn,64]  (inner_attn/Cast_1 输出)
//       [1] K^T f16 [1,H,64,Kn]  (inner_attn/Transpose_3 输出)
//       [2] V   f16 [1,H,Kn,64]  (inner_attn/Cast_3 输出)
//       [3] scale f32 [1]        (图内 Pow 常量, 0.125 / 1/sqrt(32))
// 输出: f16 [1,H,Qn,64]
//
// 数值链与参考逐段对齐: QK^T f32 累积 → f16 (参考 GEMM f16 出口) → f32
// ×scale + max-subtract softmax → f16 P (参考 Cast_5) → PV f32 累积 → f16。
// 定序固定 → 确定性。
//
// 几何 (v3): CTA = (head, 16 queries), 4 warp 每 warp 1 个 16x16 tile,
// Kn 按 64 一块, 双缓冲暂存 + 软件流水 (全局读早发, mma 吃延迟)。
//   pass1: K 块行主拷入 (满块 uint4) → col_major B 读行主暂存 = K^T
//          (无显式转置) → wmma Q·K^T → f32 acc frag 直写 sS f16 [16][904]。
//   softmax: 每 warp 4 行, f32 max/sum (lane 蝶形), P=exp(s-m)/sum 原位 f16。
//   pass2: V 块流水拷入 → wmma P·V (P tile 列随块推进) f32 跨块累积 →
//          acc frag 直写全局 (÷行和, 行守卫)。
//   smem ≈ 46KB → 2 CTA/SM; grid (ceil(Qn/16), H)。
//   演进: v1 (4warp, sK 全局转置直读) 2.7ms/站; v2 (8warp, smem 转置)
//   1.4ms/站; v3 双缓冲 + 2 CTA/SM, 修 k 维切片 bug (A tile 未随 k 步推进);
//   v3b: acc frag 直写改 store_matrix_sync (布局假设消除, A/B 无变化 →
//   布局本就对); v3c 修真根因: K^T 输入是 d 主布局 [d][key], 此前按 [key][d]
//   误读 —— 暂存改 [d][jl], 原生行连续读写皆合并。
// ============================================================================
#ifndef FA_NO_REGISTER
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <NvInfer.h>
#include <NvInferPlugin.h>

namespace FA_NS {

using namespace nvcuda;

#define FA_QT 16    // queries per CTA (smem 46KB → 2 CTA/SM)
#define FA_KB 64    // key block width
#define FA_KBKS 904 // sS row stride (>=900, 8 的倍数)

__global__ void __launch_bounds__(128)
flash_sdpa_f16(const __half* __restrict__ Q, const __half* __restrict__ Kt,
               const __half* __restrict__ V, const float* __restrict__ scale,
               __half* __restrict__ O, int H, int Qn, int Kn) {
  const int h = blockIdx.y;
  const int q0 = blockIdx.x * FA_QT;
  const int nq = min(FA_QT, Qn - q0);
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const float sc = __ldg(scale);

  extern __shared__ char smem_raw[];
  __half* sQ = reinterpret_cast<__half*>(smem_raw);      // [16][64]
  __half* sS = sQ + FA_QT * 64;                          // [16][KBKS]
  __half* sKV = sS + (size_t)FA_QT * FA_KBKS;            // [2][64][64] 双缓冲
  float* sR = reinterpret_cast<float*>(
      sKV + 2 * FA_KB * 64);                             // [FA_QT] 行和
  __shared__ float sT[4][256];  // 每 warp 16x16 f32 tile 暂存 (acc frag 落地)

  // Q tile (零垫)
  for (int i = tid; i < FA_QT * 64; i += 128) sQ[i] = __float2half(0.f);
  __syncthreads();
  const __half* Qp = Q + ((size_t)h * Qn + q0) * 64;
  for (int i = tid; i < nq * 64; i += 128) sQ[i] = Qp[i];
  // sS 垫列置 -inf (softmax 中 exp→0)
  for (int i = tid; i < FA_QT * FA_KBKS; i += 128)
    sS[i] = __ushort_as_half(0xfc00u);
  __syncthreads();

  const __half* Vp = V + (size_t)h * Kn * 64;

  // 16 行 × 4 列块 = 4 tile; 4 warp 每 warp 恰 1 tile (列块 = warp)
  const int n_id = warp;
  // B 一律 row_major: 板测外科验证 (SSA7) col_major 加载语义与预期不符,
  // row_major/mma/acc store 全部正确。sK[d][jl] 本就是 K^T 的行主形式。
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;

  // 双缓冲软件流水: 全局读早发, 写 smem 压后, 中间塞 tensor core
  // 工作吃掉内存延迟
  uint4 ra, rb, rc, rd;
  auto pf_load = [&](const __half* src, int js) {
    const uint4* s4 = reinterpret_cast<const uint4*>(src + (size_t)js * 64);
    ra = __ldg(s4 + tid);
    rb = __ldg(s4 + tid + 128);
    rc = __ldg(s4 + tid + 256);
    rd = __ldg(s4 + tid + 384);
  };
  auto pf_store = [&](int buf) {
    uint4* d4 = reinterpret_cast<uint4*>(sKV + buf * FA_KB * 64);
    d4[tid] = ra; d4[tid + 128] = rb; d4[tid + 256] = rc; d4[tid + 384] = rd;
  };
  // K^T 输入是 d 主布局 [d][key]: 原生行即连续段, 按 [d][jl] 暂存读写皆连续
  // (标量 half, 无对齐要求; v3 按 [key][d] 误读 d 主缓冲 —— A/B 抓出)
  __half hk[32];
  auto pf_load_k = [&](int js, int nj) {
#pragma unroll
    for (int t = 0; t < 32; ++t) {
      const int i = tid + 128 * t;
      const int d = i >> 6, jl = i & 63;
      hk[t] = (jl < nj)
                  ? __ldg(Kt + ((size_t)h * 64 + d) * Kn + js + jl)
                  : __float2half(0.f);
    }
  };
  auto pf_store_k = [&](int buf) {
    __half* dst = sKV + buf * FA_KB * 64;
#pragma unroll
    for (int t = 0; t < 32; ++t) dst[tid + 128 * t] = hk[t];
  };

  const int nblk = (Kn + FA_KB - 1) / FA_KB;

  // ---- pass 1: S = Q·K^T (f32 累积) → f16 sS (不乘 scale, 匹配参考舍入) ----
  // 暂存为 [d][jl] (d 主): B col_major base = k*16*64 + n*16, B[d][jl] 直取
  pf_load_k(0, min(FA_KB, Kn));
  pf_store_k(0);
  __syncthreads();
  for (int bi = 0; bi < nblk; ++bi) {
    const int jn0 = bi * FA_KB + FA_KB;
    const int cur = (bi & 1) * FA_KB * 64;
    const bool nxt_full = (bi + 1 < nblk) && (Kn - jn0 >= FA_KB);
    if (nxt_full) pf_load_k(jn0, FA_KB);
    const __half* stg = sKV + cur;
    wmma::fill_fragment(cf, 0.f);
    // A tile 列必须随 k 步推进: sQ 列 = k*16 (Q·K^T 对 d 维求和)
    for (int k = 0; k < 4; ++k) {
      wmma::load_matrix_sync(af, sQ + k * 16, 64);
      wmma::load_matrix_sync(bf, stg + k * 16 * FA_KB + n_id * 16, FA_KB);
      wmma::mma_sync(cf, af, bf, cf);
    }
    wmma::store_matrix_sync(sT[warp], cf, 16, wmma::mem_row_major);
    __syncwarp();
    for (int e = lane; e < 256; e += 32) {
      const int r = e >> 4, c = e & 15;
      // 列偏移必须带 key 块偏移 bi: 否则各块互相覆盖前 64 列 (v1-v3 同病灶)
      sS[(size_t)r * FA_KBKS + bi * FA_KB + n_id * 16 + c] =
          __float2half(sT[warp][e]);
    }
    if (nxt_full) {
      pf_store_k((bi & 1) ^ 1);
    } else if (bi + 1 < nblk) {
      pf_load_k(jn0, Kn - jn0);
      pf_store_k((bi & 1) ^ 1);
    }
    __syncthreads();
  }

  // ---- softmax (每 warp 4 行, f32 max/sum lane 蝶形) ----
  for (int r = warp; r < FA_QT; r += 4) {
    if (r >= nq) continue;
    __half* pr = sS + (size_t)r * FA_KBKS;
    float m = -3.4e38f;
    for (int j = lane; j < Kn; j += 32)
      m = fmaxf(m, __half2float(pr[j]) * sc);
#pragma unroll
    for (int o = 16; o; o >>= 1)
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float s = 0.f;
    for (int j = lane; j < Kn; j += 32)
      s += expf(__half2float(pr[j]) * sc - m);
#pragma unroll
    for (int o = 16; o; o >>= 1)
      s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) sR[r] = s;
    const float inv = 1.f / s;
    for (int j = lane; j < Kn; j += 32)
      pr[j] = __float2half(expf(__half2float(pr[j]) * sc - m) * inv);
    // 垫列清零: pass2 wmma 会读到 [Kn, 块尾) 的 P, -inf×V 会污染累积
    for (int j = Kn + lane; j < FA_KBKS; j += 32) pr[j] = __float2half(0.f);
  }
  __syncthreads();

  // ---- pass 2: O = P·V (f32 累积跨块) → frag 直写全局 (÷行和, 行守卫) ----
  // P tile 列偏移必须随块推进: sS 列 = bi*FA_KB + n_id*16
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af2;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf2;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf2;
  wmma::fill_fragment(cf2, 0.f);
  // V 原生 [key][d] 行主: 满块 uint4 向量化, 尾块标量守卫
  auto copy_kv = [&](const __half* src, int js, int nj, __half* dst) {
    if (nj == FA_KB) {
      const uint4* s4 = reinterpret_cast<const uint4*>(src + (size_t)js * 64);
      uint4* d4 = reinterpret_cast<uint4*>(dst);
      for (int i = tid; i < FA_KB * 8; i += 128) d4[i] = __ldg(s4 + i);
    } else {
      for (int i = tid; i < nj * 64; i += 128)
        dst[i] = src[(size_t)js * 64 + i];
      for (int i = nj * 64 + tid; i < FA_KB * 64; i += 128)
        dst[i] = __float2half(0.f);
    }
  };
  copy_kv(Vp, 0, min(FA_KB, Kn), sKV);
  __syncthreads();
  for (int bi = 0; bi < nblk; ++bi) {
    const int jn0 = bi * FA_KB + FA_KB;
    const int cur = (bi & 1) * FA_KB * 64;
    const bool nxt_full = (bi + 1 < nblk) && (Kn - jn0 >= FA_KB);
    if (nxt_full) pf_load(Vp, jn0);
    // A=P 列随 k 步推进 (key 维), 与 n_id 无关; B=V 行随 k 步推进
    for (int k = 0; k < 4; ++k) {
      wmma::load_matrix_sync(af2, sS + bi * FA_KB + k * 16, FA_KBKS);
      wmma::load_matrix_sync(bf2,
                             sKV + cur + k * 16 * FA_KB + n_id * 16, FA_KB);
      wmma::mma_sync(cf2, af2, bf2, cf2);
    }
    if (nxt_full) {
      pf_store((bi & 1) ^ 1);
    } else if (bi + 1 < nblk) {
      copy_kv(Vp, jn0, Kn - jn0, sKV + ((bi & 1) ^ 1) * FA_KB * 64);
    }
    __syncthreads();
  }
  wmma::store_matrix_sync(sT[warp], cf2, 16, wmma::mem_row_major);
  __syncwarp();
  for (int e = lane; e < 256; e += 32) {
    const int r = e >> 4, c = e & 15;
    if (r < nq) {
      __half* op = O + ((size_t)h * Qn + q0 + r) * 64 + n_id * 16 + c;
      *op = __float2half(sT[warp][e] / sR[r]);
    }
  }
}

}  // namespace FA_NS

namespace FA_TRT_NS {

using namespace nvinfer1;

class FlashSdpaPlugin : public IPluginV2DynamicExt {
 public:
  FlashSdpaPlugin() = default;
  FlashSdpaPlugin(const void* data, size_t size) { (void)data; (void)size; }
  ~FlashSdpaPlugin() override = default;

  const char* getPluginType() const noexcept override { return "FlashSDPA"; }
  const char* getPluginVersion() const noexcept override { return "1"; }
  int32_t getNbOutputs() const noexcept override { return 1; }

  DimsExprs getOutputDimensions(int32_t outputIndex, DimsExprs const* inputs,
                                int32_t nbInputs,
                                IExprBuilder&) noexcept override {
    (void)outputIndex; (void)nbInputs;
    return inputs[0];
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
    if (pos == 3) return inOut[pos].type == DataType::kFLOAT;  // scale
    return inOut[pos].type == DataType::kHALF;
  }

  int32_t initialize() noexcept override {
    return cudaFuncSetAttribute(FA_NS::flash_sdpa_f16,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                100u * 1024u) == cudaSuccess
               ? 0
               : -1;
  }
  void terminate() noexcept override {}
  size_t getSerializationSize() const noexcept override { return 0; }
  void serialize(void* buffer) const noexcept override { (void)buffer; }
  void destroy() noexcept override { delete this; }
  void setPluginNamespace(const char* ns) noexcept override {
    snprintf(ns_, sizeof(ns_), "%s", ns ? ns : "");
  }
  const char* getPluginNamespace() const noexcept override { return ns_; }

  IPluginV2DynamicExt* clone() const noexcept override {
    auto* p = new FlashSdpaPlugin();
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
    if (d.nbDims != 4 || d.d[2] > 900) return -1;
    const int H = d.d[1], Qn = d.d[2], Kn = inputDesc[1].dims.d[3];
    if (Kn > 900 || Kn <= 0 || Qn <= 0 || H <= 0) return -1;
    const int grid_x = (Qn + FA_QT - 1) / FA_QT;
    dim3 grid(grid_x, H);
    const size_t smem =
        ((FA_QT * 64 + FA_QT * FA_KBKS + 2 * FA_KB * 64) * sizeof(__half)) +
        (FA_QT + 1) * sizeof(float);
    FA_NS::flash_sdpa_f16<<<grid, 128, smem, stream>>>(
        static_cast<const __half*>(inputs[0]),
        static_cast<const __half*>(inputs[1]),
        static_cast<const __half*>(inputs[2]),
        static_cast<const float*>(inputs[3]),
        static_cast<__half*>(outputs[0]), H, Qn, Kn);
    return cudaPeekAtLastError() != cudaSuccess ? -1 : 0;
  }

 private:
  char ns_[64] = {0};
};

class FlashSdpaCreator : public IPluginCreator {
 public:
  FlashSdpaCreator() {
    fc_.nbFields = 0;
    fc_.fields = nullptr;
  }
  ~FlashSdpaCreator() override = default;
  const char* getPluginName() const noexcept override { return "FlashSDPA"; }
  const char* getPluginVersion() const noexcept override { return "1"; }
  PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
  IPluginV2* createPlugin(const char* name,
                          PluginFieldCollection const* fc) noexcept override {
    (void)name; (void)fc;
    auto* p = new FlashSdpaPlugin();
    p->setPluginNamespace(ns_);
    return p;
  }
  IPluginV2* deserializePlugin(const char* name, const void* serialData,
                               size_t serialLength) noexcept override {
    (void)name;
    auto* p = new FlashSdpaPlugin(serialData, serialLength);
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

}  // namespace FA_TRT_NS

namespace FA_REG_NS {
struct Registrar {
  Registrar() {
    getPluginRegistry()->registerCreator(*new FA_TRT_NS::FlashSdpaCreator(),
                                         "SparseDrive");
    getPluginRegistry()->registerCreator(*new FA_TRT_NS::FlashSdpaCreator(),
                                         "");
  }
};
static Registrar g_fa_registrar;
}
#endif  // FA_NO_REGISTER
