// FlashSDPA 分级调试: (1) 微缩 wmma 语义测试; (2) 带 S/P 中间导出的调试内核;
// synth 合成对拍 / replay 真实数据重放。
// 编译: nvcc -O3 -arch=sm_87 -o selftest_fa2 selftest_fa2.cu -lnvinfer
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <mma.h>

using namespace nvcuda;

// ---------- 微缩 wmma: 验证 A(row_major)/B(col_major)/B(row_major)/store ----
__global__ void micro_wmma(float* outA, float* outBcol, float* outBrow) {
  __shared__ __half A[256], B[256];
  const int l = threadIdx.x;
  if (l < 256) {
    A[l] = __float2half(((l >> 4) * 0.25f) + (l & 15) * 0.01f);  // [m][k]
    B[l] = __float2half(((l & 15) * 0.125f) - (l >> 4) * 0.02f); // [k][n] 行主
  }
  __syncthreads();
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fbc;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fbr;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
  wmma::load_matrix_sync(fa, A, 16);
  wmma::load_matrix_sync(fbc, B, 16);  // col_major 读行主缓冲 = B^T
  wmma::load_matrix_sync(fbr, B, 16);
  wmma::fill_fragment(fc, 0.f);
  wmma::mma_sync(fc, fa, fbc, fc);
  wmma::store_matrix_sync(outA, fc, 16, wmma::mem_row_major);
  wmma::fill_fragment(fc, 0.f);
  wmma::mma_sync(fc, fa, fbr, fc);
  wmma::store_matrix_sync(outBcol, fc, 16, wmma::mem_row_major);
  // 交叉验证: A 用 col_major 读行主缓冲 (即 A^T)
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> fac;
  wmma::load_matrix_sync(fac, B, 16);
  wmma::fill_fragment(fc, 0.f);
  wmma::mma_sync(fc, fac, fbr, fc);
  wmma::store_matrix_sync(outBrow, fc, 16, wmma::mem_row_major);
}

static int micro_test() {
  printf("== micro wmma ==\n");
  float *dA, *dBc, *dBr, *hA, *hBc, *hBr;
  cudaMallocManaged(&dA, 256 * 4); cudaMallocManaged(&dBc, 256 * 4);
  cudaMallocManaged(&dBr, 256 * 4);
  micro_wmma<<<1, 32>>>(dA, dBc, dBr);
  cudaDeviceSynchronize();
  float A[256], B[256];
  for (int i = 0; i < 256; ++i) {
    A[i] = ((i >> 4) * 0.25f) + (i & 15) * 0.01f;
    B[i] = ((i & 15) * 0.125f) - (i >> 4) * 0.02f;
  }
  double w1 = 0, w2 = 0, w3 = 0;
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 16; ++n) {
      float e1 = 0, e2 = 0, e3 = 0;
      for (int k = 0; k < 16; ++k) {
        e1 += A[m * 16 + k] * B[n * 16 + k];  // col_major 读行主 = B^T 参与积
        e2 += A[m * 16 + k] * B[k * 16 + n];  // row_major 正常
        e3 += B[m * 16 + k] * B[k * 16 + n];  // A col_major 读行主 = A^T
      }
      w1 = fmax(w1, fabs(dA[m * 16 + n] - e1));
      w2 = fmax(w2, fabs(dBc[m * 16 + n] - e2));
      w3 = fmax(w3, fabs(dBr[m * 16 + n] - e3));
    }
  printf("  Bcol(=A@B^T) maxerr=%.2e\n  Brow(=A@B)   maxerr=%.2e\n"
         "  Acol(=B@B)   maxerr=%.2e\n", w1, w2, w3);
  int bad = (w1 > 1e-3) + (w2 > 1e-3) + (w3 > 1e-3);
  printf("  %s\n", bad ? "SEMANTICS MISMATCH" : "semantics OK");
  return bad;
}

// ---------- 调试内核: 与插件内核同一份逻辑 + S/P 中间导出 ----------
#define QT 16
#define KB 64
#define KBKS 904

__global__ void __launch_bounds__(128)
fa_dbg(const __half* __restrict__ Q, const __half* __restrict__ Kt,
       const __half* __restrict__ V, const float* __restrict__ scale,
       __half* __restrict__ O, int H, int Qn, int Kn,
       __half* __restrict__ dbgS, __half* __restrict__ dbgP) {
  const int h = blockIdx.y;
  const int q0 = blockIdx.x * QT;
  const int nq = min(QT, Qn - q0);
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const float sc = __ldg(scale);

  extern __shared__ char smem_raw[];
  __half* sQ = reinterpret_cast<__half*>(smem_raw);
  __half* sS = sQ + QT * 64;
  __half* sKV = sS + (size_t)QT * KBKS;
  float* sR = reinterpret_cast<float*>(sKV + 2 * KB * 64);
  __shared__ float sT[4][256];

  for (int i = tid; i < QT * 64; i += 128) sQ[i] = __float2half(0.f);
  __syncthreads();
  const __half* Qp = Q + ((size_t)h * Qn + q0) * 64;
  for (int i = tid; i < nq * 64; i += 128) sQ[i] = Qp[i];
  for (int i = tid; i < QT * KBKS; i += 128)
    sS[i] = __ushort_as_half(0xfc00u);
  __syncthreads();

  const __half* Vp = V + (size_t)h * Kn * 64;
  const int n_id = warp;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;

  uint4 ra, rb, rc, rd;
  auto pf_load = [&](const __half* src, int js) {
    const uint4* s4 = reinterpret_cast<const uint4*>(src + (size_t)js * 64);
    ra = __ldg(s4 + tid);
    rb = __ldg(s4 + tid + 128);
    rc = __ldg(s4 + tid + 256);
    rd = __ldg(s4 + tid + 384);
  };
  auto pf_store = [&](int buf) {
    uint4* d4 = reinterpret_cast<uint4*>(sKV + buf * KB * 64);
    d4[tid] = ra; d4[tid + 128] = rb; d4[tid + 256] = rc; d4[tid + 384] = rd;
  };
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
    __half* dst = sKV + buf * KB * 64;
#pragma unroll
    for (int t = 0; t < 32; ++t) dst[tid + 128 * t] = hk[t];
  };

  const int nblk = (Kn + KB - 1) / KB;
  pf_load_k(0, min(KB, Kn));
  pf_store_k(0);
  __syncthreads();
  for (int bi = 0; bi < nblk; ++bi) {
    const int jn0 = bi * KB + KB;
    const int cur = (bi & 1) * KB * 64;
    const bool nxt_full = (bi + 1 < nblk) && (Kn - jn0 >= KB);
    if (nxt_full) pf_load_k(jn0, KB);
    const __half* stg = sKV + cur;
    wmma::fill_fragment(cf, 0.f);
    for (int k = 0; k < 4; ++k) {
      wmma::load_matrix_sync(af, sQ + k * 16, 64);
      wmma::load_matrix_sync(bf, stg + k * 16 * KB + n_id * 16, KB);
      wmma::mma_sync(cf, af, bf, cf);
    }
    wmma::store_matrix_sync(sT[warp], cf, 16, wmma::mem_row_major);
    __syncwarp();
    for (int e = lane; e < 256; e += 32) {
      const int r = e >> 4, c = e & 15;
      sS[(size_t)r * KBKS + bi * KB + n_id * 16 + c] =
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
  // 导出 S (f16, 原样): dbgS[(h*Qn+q0+r)*Kn + c]
  if (dbgS) {
    for (int e = tid; e < nq * Kn; e += 128) {
      const int r = e / Kn, c = e % Kn;
      dbgS[((size_t)(h * Qn + q0 + r)) * Kn + c] = sS[(size_t)r * KBKS + c];
    }
    __syncthreads();
  }

  for (int r = warp; r < QT; r += 4) {
    if (r >= nq) continue;
    __half* pr = sS + (size_t)r * KBKS;
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
    for (int j = Kn + lane; j < KBKS; j += 32) pr[j] = __float2half(0.f);
  }
  __syncthreads();
  if (dbgP) {
    for (int e = tid; e < nq * Kn; e += 128) {
      const int r = e / Kn, c = e % Kn;
      dbgP[((size_t)(h * Qn + q0 + r)) * Kn + c] = sS[(size_t)r * KBKS + c];
    }
    __syncthreads();
  }

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af2;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf2;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf2;
  wmma::fill_fragment(cf2, 0.f);
  auto copy_kv = [&](const __half* src, int js, int nj, __half* dst) {
    if (nj == KB) {
      const uint4* s4 = reinterpret_cast<const uint4*>(src + (size_t)js * 64);
      uint4* d4 = reinterpret_cast<uint4*>(dst);
      for (int i = tid; i < KB * 8; i += 128) d4[i] = __ldg(s4 + i);
    } else {
      for (int i = tid; i < nj * 64; i += 128)
        dst[i] = src[(size_t)js * 64 + i];
      for (int i = nj * 64 + tid; i < KB * 64; i += 128)
        dst[i] = __float2half(0.f);
    }
  };
  copy_kv(Vp, 0, min(KB, Kn), sKV);
  __syncthreads();
  for (int bi = 0; bi < nblk; ++bi) {
    const int jn0 = bi * KB + KB;
    const int cur = (bi & 1) * KB * 64;
    const bool nxt_full = (bi + 1 < nblk) && (Kn - jn0 >= KB);
    if (nxt_full) pf_load(Vp, jn0);
    for (int k = 0; k < 4; ++k) {
      wmma::load_matrix_sync(af2, sS + bi * KB + k * 16, KBKS);
      wmma::load_matrix_sync(bf2, sKV + cur + k * 16 * KB + n_id * 16, KB);
      wmma::mma_sync(cf2, af2, bf2, cf2);
    }
    if (nxt_full) {
      pf_store((bi & 1) ^ 1);
    } else if (bi + 1 < nblk) {
      copy_kv(Vp, jn0, Kn - jn0, sKV + ((bi & 1) ^ 1) * KB * 64);
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

// ---------- 驱动 ----------
static size_t smem_bytes() {
  return ((QT * 64 + QT * KBKS + 2 * KB * 64) * sizeof(__half)) +
         (QT + 1) * sizeof(float);
}

static int replay(const char* qf, const char* kf, const char* vf,
                  const char* od) {
  const int H = 8, Qn = 900, Kn = 900;
  size_t nq = (size_t)H * Qn * 64, nk = (size_t)H * 64 * Kn;
  size_t nv = (size_t)H * Kn * 64, ns = (size_t)H * Qn * Kn;
  std::vector<__half> hq(nq), hkt(nk), hv(nv), ho(nq), hs(ns), hp(ns);
  FILE* f;
  f = fopen(qf, "rb"); if (!f) { printf("no %s\n", qf); return 1; }
  fread(hq.data(), 2, nq, f); fclose(f);
  f = fopen(kf, "rb"); if (!f) { printf("no %s\n", kf); return 1; }
  fread(hkt.data(), 2, nk, f); fclose(f);
  f = fopen(vf, "rb"); if (!f) { printf("no %s\n", vf); return 1; }
  fread(hv.data(), 2, nv, f); fclose(f);
  float sc = 0.125f;
  __half *dq, *dkt, *dv, *doo, *ds, *dp;
  float* dsc;
  cudaMalloc(&dq, nq * 2); cudaMalloc(&dkt, nk * 2); cudaMalloc(&dv, nv * 2);
  cudaMalloc(&doo, nq * 2); cudaMalloc(&ds, ns * 2); cudaMalloc(&dp, ns * 2);
  cudaMalloc(&dsc, 4);
  cudaMemcpy(dq, hq.data(), nq * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dkt, hkt.data(), nk * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dv, hv.data(), nv * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dsc, &sc, 4, cudaMemcpyHostToDevice);
  cudaFuncSetAttribute(fa_dbg, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       100u << 10);
  dim3 grid((Qn + QT - 1) / QT, H);
  fa_dbg<<<grid, 128, smem_bytes()>>>(dq, dkt, dv, dsc, doo, H, Qn, Kn,
                                      ds, dp);
  cudaError_t e = cudaDeviceSynchronize();
  printf("replay launch %s\n", cudaGetErrorString(e));
  cudaMemcpy(ho.data(), doo, nq * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(hs.data(), ds, ns * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(hp.data(), dp, ns * 2, cudaMemcpyDeviceToHost);
  char p[512];
  snprintf(p, sizeof(p), "%s/O_fa2.bin", od);
  f = fopen(p, "wb"); fwrite(ho.data(), 2, nq, f); fclose(f);
  snprintf(p, sizeof(p), "%s/S_fa2.bin", od);
  f = fopen(p, "wb"); fwrite(hs.data(), 2, ns, f); fclose(f);
  snprintf(p, sizeof(p), "%s/P_fa2.bin", od);
  f = fopen(p, "wb"); fwrite(hp.data(), 2, ns, f); fclose(f);
  printf("replay dumped O/S/P to %s\n", od);
  return 0;
}

static int synth_one(int Qn, int Kn) {
  const int H = 8;
  size_t nq = (size_t)H * Qn * 64, nk = (size_t)H * 64 * Kn;
  size_t nv = (size_t)H * Kn * 64, ns = (size_t)H * Qn * Kn;
  std::vector<__half> hq(nq), hkt(nk), hv(nv), ho(nq), hs(ns), hp(ns);
  srand(11 + Qn);
  auto rh = []() {
    return __float2half((rand() / (float)RAND_MAX - 0.5f) * 2.f);
  };
  for (auto& x : hq) x = rh();
  for (auto& x : hkt) x = rh();
  for (auto& x : hv) x = rh();
  float sc = 0.125f;
  __half *dq, *dkt, *dv, *doo, *ds, *dp;
  float* dsc;
  cudaMalloc(&dq, nq * 2); cudaMalloc(&dkt, nk * 2); cudaMalloc(&dv, nv * 2);
  cudaMalloc(&doo, nq * 2); cudaMalloc(&ds, ns * 2); cudaMalloc(&dp, ns * 2);
  cudaMalloc(&dsc, 4);
  cudaMemcpy(dq, hq.data(), nq * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dkt, hkt.data(), nk * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dv, hv.data(), nv * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dsc, &sc, 4, cudaMemcpyHostToDevice);
  cudaFuncSetAttribute(fa_dbg, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       100u << 10);
  dim3 grid((Qn + QT - 1) / QT, H);
  fa_dbg<<<grid, 128, smem_bytes()>>>(dq, dkt, dv, dsc, doo, H, Qn, Kn,
                                      ds, dp);
  cudaDeviceSynchronize();
  cudaMemcpy(ho.data(), doo, nq * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(hs.data(), ds, ns * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(hp.data(), dp, ns * 2, cudaMemcpyDeviceToHost);
  // CPU 分级参考
  double wS = 0, wP = 0, wO = 0;
  std::vector<float> Prow(Kn > 0 ? Kn : 1), Oref(64);
  for (int h = 0; h < H; ++h) {
    const __half* Q = &hq[(size_t)h * Qn * 64];
    const __half* KT = &hkt[(size_t)h * 64 * Kn];
    const __half* V = &hv[(size_t)h * Kn * 64];
    for (int q = 0; q < Qn; ++q) {
      for (int j = 0; j < Kn; ++j) {
        float s = 0;
        for (int d = 0; d < 64; ++d)
          s += __half2float(Q[(size_t)q * 64 + d]) *
               __half2float(KT[(size_t)d * Kn + j]);
        Prow[j] = s;
      }
      const __half* Sg = &hs[((size_t)h * Qn + q) * Kn];
      for (int j = 0; j < Kn; ++j) {
        double a = Prow[j], b = __half2float(Sg[j]);
        wS = fmax(wS, fabs(a - b) / (fabs(a) + 1.0));
      }
      float m = -3.4e38f;
      for (int j = 0; j < Kn; ++j) m = fmaxf(m, Prow[j] * sc);
      float sum = 0;
      for (int j = 0; j < Kn; ++j) {
        Prow[j] = expf(Prow[j] * sc - m);
        sum += Prow[j];
      }
      for (int j = 0; j < Kn; ++j) Prow[j] /= sum;
      const __half* Pg = &hp[((size_t)h * Qn + q) * Kn];
      for (int j = 0; j < Kn; ++j) {
        double a = Prow[j], b = __half2float(Pg[j]);
        wP = fmax(wP, fabs(a - b));
      }
      for (int d = 0; d < 64; ++d) {
        float s = 0;
        for (int j = 0; j < Kn; ++j)
          s += Prow[j] * __half2float(V[(size_t)j * 64 + d]);
        Oref[d] = s;
      }
      const __half* Og = &ho[((size_t)h * Qn + q) * 64];
      for (int d = 0; d < 64; ++d) {
        double a = Oref[d], b = __half2float(Og[d]);
        wO = fmax(wO, fabs(a - b) / (fabs(a) + 0.01));
      }
    }
  }
  printf("  (%d,%d) S maxrel=%.3e P maxerr=%.3e O maxrel=%.3e\n",
         Qn, Kn, wS, wP, wO);
  int bad = (wS > 5e-2) + (wP > 5e-3) + (wO > 5e-2);
  return bad;
}

int main(int argc, char** argv) {
  if (argc >= 5 && argv[1][0] == 'r') return replay(argv[2], argv[3],
                                                    argv[4], argv[5]);
  printf("== micro ==\n");
  int bad = micro_test();
  printf("== synth (S/P/O 分级) ==\n");
  bad += synth_one(64, 64);
  bad += synth_one(900, 900);
  printf("== %s ==\n", bad ? "FAILED" : "ALL PASS");
  return bad ? 1 : 0;
}
