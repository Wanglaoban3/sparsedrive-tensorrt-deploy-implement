// 竞态定位: 同进程连续两次启动 fa_dbg, 设备端逐张量对拍.
// 编译: nvcc -O3 -arch=sm_87 -o selftest_fa4 selftest_fa4.cu -lnvinfer
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <mma.h>

using namespace nvcuda;

#define QT 16
#define KB 64
#define KBKS 904

// ---- 与插件内核同逻辑 (含 row_major B + bi*KB 散射), dump 可选 ----
__global__ void __launch_bounds__(128)
fa_dbg(const __half* __restrict__ Q, const __half* __restrict__ Kt,
       const __half* __restrict__ V, const float* __restrict__ scale,
       __half* __restrict__ O, int H, int Qn, int Kn,
       __half* __restrict__ dbgS) {
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

__global__ void cmp_kernel(const __half* a, const __half* b, long long n,
                           int* bad, float* ref) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = __half2float(a[i]), y = __half2float(b[i]);
    if (x != y) {
      atomicAdd(bad, 1);
      if (fabs(x - y) > fabs(ref[0])) {
        ref[0] = x - y;
        ref[1] = (float)i;
      }
    }
  }
}

int main(int argc, char** argv) {
  const int H = 8, Qn = 900, Kn = 900;
  size_t nq = (size_t)H * Qn * 64, nk = (size_t)H * 64 * Kn;
  size_t nv = (size_t)H * Kn * 64, ns = (size_t)H * Qn * Kn;
  std::vector<__half> hq(nq), hkt(nk), hv(nv);
  if (argc >= 4) {
    FILE* f = fopen(argv[1], "rb"); fread(hq.data(), 2, nq, f); fclose(f);
    f = fopen(argv[2], "rb"); fread(hkt.data(), 2, nk, f); fclose(f);
    f = fopen(argv[3], "rb"); fread(hv.data(), 2, nv, f); fclose(f);
    printf("inputs from probe files\n");
  } else {
    srand(5);
    auto rh = []() {
      return __float2half((rand() / (float)RAND_MAX - 0.5f) * 2.f);
    };
    for (auto& x : hq) x = rh();
    for (auto& x : hkt) x = rh();
    for (auto& x : hv) x = rh();
    printf("inputs synthetic\n");
  }
  float sc = 0.125f;
  __half *dq, *dkt, *dv, *o1, *o2, *s1, *s2;
  float *dsc, *st;
  int* bad;
  cudaMalloc(&dq, nq * 2); cudaMalloc(&dkt, nk * 2); cudaMalloc(&dv, nv * 2);
  cudaMalloc(&o1, nq * 2); cudaMalloc(&o2, nq * 2);
  cudaMalloc(&s1, ns * 2); cudaMalloc(&s2, ns * 2);
  cudaMalloc(&dsc, 4); cudaMalloc(&st, 8 * sizeof(float)); cudaMalloc(&bad, 4);
  cudaMemcpy(dq, hq.data(), nq * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dkt, hkt.data(), nk * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dv, hv.data(), nv * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dsc, &sc, 4, cudaMemcpyHostToDevice);
  const size_t smem =
      ((QT * 64 + QT * KBKS + 2 * KB * 64) * sizeof(__half)) +
      (QT + 1) * sizeof(float);
  cudaFuncSetAttribute(fa_dbg, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       100u << 10);
  dim3 grid((Qn + QT - 1) / QT, H);

  // S 两连跑对拍
  cudaMemset(st, 0, 32); cudaMemset(bad, 0, 4);
  fa_dbg<<<grid, 128, smem>>>(dq, dkt, dv, dsc, o1, H, Qn, Kn, s1);
  fa_dbg<<<grid, 128, smem>>>(dq, dkt, dv, dsc, o2, H, Qn, Kn, s2);
  cudaDeviceSynchronize();
  cmp_kernel<<<(ns + 255) / 256, 256>>>(s1, s2, ns, bad, st);
  float hst[2]; int hbad;
  cudaMemcpy(hst, st, 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(&hbad, bad, 4, cudaMemcpyDeviceToHost);
  printf("S  : diff cells=%d maxdiff=%f at %ld\n", hbad, hst[0],
         (long)hst[1]);

  // O 两连跑对拍
  cudaMemset(st, 0, 32); cudaMemset(bad, 0, 4);
  cmp_kernel<<<(nq + 255) / 256, 256>>>(o1, o2, nq, bad, st);
  cudaMemcpy(hst, st, 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(&hbad, bad, 4, cudaMemcpyDeviceToHost);
  printf("O  : diff cells=%d maxdiff=%f at %ld\n", hbad, hst[0],
         (long)hst[1]);

  // 第三跑 (S -> s1) 与第一跑对拍 (确认 S 的不稳定是否跨多次)
  cudaMemset(st, 0, 32); cudaMemset(bad, 0, 4);
  fa_dbg<<<grid, 128, smem>>>(dq, dkt, dv, dsc, o2, H, Qn, Kn, s2);
  cudaDeviceSynchronize();
  cmp_kernel<<<(ns + 255) / 256, 256>>>(s1, s2, ns, bad, st);
  cudaMemcpy(hst, st, 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(&hbad, bad, 4, cudaMemcpyDeviceToHost);
  printf("S' : diff cells=%d maxdiff=%f at %ld\n", hbad, hst[0],
         (long)hst[1]);
  return 0;
}
