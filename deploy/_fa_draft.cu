// ============================================================================
// FlashSDPA TRT plugin (P5b) —— 拼接进 v10 .so (与 v8 DFA / ScaleSoftmax 共存)
//
// 替换 inner_attn 的 6 节点: QK^T MatMul → Cast → Mul(scale) → Softmax →
// Cast → PV MatMul_1。wmma 16x16x16 f16 / f32 累积 —— 板上退化 tactic 跑
// 这两个 GEMM 只有 ~1.6 TFLOPS (无 TC), 本内核走 tensor core。
//
// 输入: [0] Q   f16 [1,H,Qn,64]   (inner_attn/Cast_1 输出)
//       [1] K^T f16 [1,H,64,Kn]   (inner_attn/Transpose_3 输出)
//       [2] V   f16 [1,H,Kn,64]   (inner_attn/Cast_3 输出)
//       [3] scale f32 [1]         (图内 Pow 常量)
// 输出: f16 [1,H,Qn,64]
//
// 数值链与参考一致: QK^T f32 累积 → f16 (参考 GEMM f16 出) → f32 ×scale +
// max-subtract softmax → f16 P → PV f32 累积 → f16。定序固定, 确定性。
//
// 几何: CTA = (head, 32 queries), Kn 按 64 一块流过。
//   pass1: 每 64-key 块 wmma Q(32x64)×K^T(64x64) → S f32 frag → 存 smem
//          S_f16[32][KBKS] (KBKS=960 ≥ Kn); 块后每 warp 归并 4 行算
//          max/sum (f32, 全局行归约一次)。
//   pass2: smem S 原位 exp→P(f16); 每 64-key 块 wmma P(32x64)×V(64x64)
//          累积 O frag (f32); 末尾 ×1/sum → out。
//   V pass2 从全局重读 (0.9MB, L2 命中)。smem ~78KB (需 opt-in)。
// ============================================================================
#ifndef FA_NO_REGISTER
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace FA_NS {

using namespace nvcuda;

#define FA_QT 32            // queries per CTA
#define FA_KB 64            // key block
#define FA_KBKS 960         // padded row width in smem S (>= Kn <= 900)

__global__ void __launch_bounds__(128)
flash_sdpa_f16(const __half* __restrict__ Q, const __half* __restrict__ Kt,
               const __half* __restrict__ V, const float* __restrict__ scale,
               __half* __restrict__ O, int H, int Qn, int Kn) {
  const int h = blockIdx.y;
  const int q0 = blockIdx.x * FA_QT;
  const int nq = min(FA_QT, Qn - q0);
  if (nq <= 0) return;
  const int tid = threadIdx.x;
  const float sc = __ldg(scale);

  extern __shared__ char smem_raw[];
  __half* sQ = reinterpret_cast<__half*>(smem_raw);              // [32][64]
  __half* sS = sQ + FA_QT * 64;                                  // [32][KBKS]
  __half* sV = sS + FA_QT * FA_KBKS;                             // [64][64]
  float* sRed = reinterpret_cast<float*>(sV + FA_KB * 64);       // [32] max/sum

  // Q tile (rows q0..q0+31), zero-pad
  for (int i = tid; i < FA_QT * 64; i += blockDim.x) sQ[i] = __float2half(0.f);
  __syncthreads();
  const __half* Qp = Q + ((size_t)h * Qn + q0) * 64;
  for (int i = tid; i < nq * 64; i += blockDim.x) sQ[i] = Qp[i];
  __syncthreads();

  // ---- pass 1: S = Q·K^T · scale → smem (f16) ----
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
  const int nw = blockDim.x / 32;  // 4 warps
  const int warp = tid >> 5;
  const __half* Kp = Kt + (size_t)h * 64 * Kn;
  for (int j0 = 0; j0 < Kn; j0 += FA_KB) {
    const int nj = min(FA_KB, Kn - j0);
    wmma::fill_fragment(cf, 0.f);
    for (int m = 0; m < 2; ++m)
      for (int n = 0; n < 4; ++n) {
        wmma::load_matrix_sync(af, sQ + m * 16 * 64, 64);
        wmma::load_matrix_sync(bf, Kp + (size_t)(j0 + n * 16), Kn);
        wmma::mma_sync(cf, af, bf, cf);
      }
    __syncthreads();  // cf 各 warp 独立 (16 行组), 无需; 保留 sS 写栅栏
    for (int e = tid; e < FA_QT * FA_KB; e += blockDim.x) {
      const int r = e / FA_KB, c = e % FA_KB;
      __half v = __float2half(0.f);
      if (r < nq && c < nj) {
        // accumulator 元素 (r,c): 每 fragment 16x16, cf 是单 16x16 →
        // 需按 frag 布局取出; 改用 store 到暂存区再读 (简单正确)
      }
      sS[(size_t)r * FA_KBKS + c] = v;
    }
    // 上面的直接取值不可行 —— 用 store_matrix_sync 落 smem 暂存再转 f16
    __shared__ __half sC[16 * 64];  // 2(m)*4(n) 块 → 32x64
    for (int m = 0; m < 2; ++m)
      for (int n = 0; n < 4; ++n)
        wmma::store_matrix_sync(
            reinterpret_cast<__half*>(sC) + 0, cf, 16,
            wmma::mem_row_major);  // placeholder (见下方修正版)
    __syncthreads();
  }
}
}  // namespace FA_NS
#endif  // FA_NO_REGISTER
