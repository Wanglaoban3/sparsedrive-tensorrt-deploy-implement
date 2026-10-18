// FlashSDPA 内核自测: 合成数据 + CPU 参考对拍 (无引擎, 隔离内核数学)
// 编译: nvcc -O3 -arch=sm_87 -o selftest_fa selftest_fa.cu -lnvinfer
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "_fa_part.cu"

static int check(int Qn, int Kn) {
  const int H = 8;
  size_t nq = (size_t)H * Qn * 64, nk = (size_t)H * 64 * Kn;
  size_t nv = (size_t)H * Kn * 64;
  std::vector<__half> hq(nq), hkt(nk), hv(nv), ho(nq);
  srand(7 + Qn);
  auto rh = []() {
    return __float2half((rand() / (float)RAND_MAX - 0.5f) * 2.f);
  };
  for (auto& x : hq) x = rh();
  for (auto& x : hkt) x = rh();
  for (auto& x : hv) x = rh();

  __half *dq, *dkt, *dv, *doo;
  float* dsc;
  cudaMalloc(&dq, nq * 2); cudaMalloc(&dkt, nk * 2);
  cudaMalloc(&dv, nv * 2); cudaMalloc(&doo, nq * 2);
  cudaMalloc(&dsc, 4);
  float sc = 0.125f;
  cudaMemcpy(dq, hq.data(), nq * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dkt, hkt.data(), nk * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dv, hv.data(), nv * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dsc, &sc, 4, cudaMemcpyHostToDevice);

  const size_t smem =
      ((FA_QT * 64 + FA_QT * FA_KBKS + 2 * FA_KB * 64) * sizeof(__half)) +
      (FA_QT + 1) * sizeof(float);
  cudaFuncSetAttribute(FA_NS::flash_sdpa_f16,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, 100u << 10);
  dim3 grid((Qn + FA_QT - 1) / FA_QT, H);
  FA_NS::flash_sdpa_f16<<<grid, 128, smem>>>(dq, dkt, dv, dsc, doo,
                                             H, Qn, Kn);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    printf("  (%d,%d) launch FAIL %s\n", Qn, Kn, cudaGetErrorString(e));
    return 1;
  }
  cudaMemcpy(ho.data(), doo, nq * 2, cudaMemcpyDeviceToHost);

  // CPU 参考: f32 全程 (内核 f16 舍入差应在 ~1e-2 量级)
  double worst = 0; double gnorm = 0, dnorm = 0;
  std::vector<float> p(Kn), oref(64);
  for (int h = 0; h < H; ++h) {
    const __half* Q = &hq[(size_t)h * Qn * 64];
    const __half* KT = &hkt[(size_t)h * 64 * Kn];
    const __half* V = &hv[(size_t)h * Kn * 64];
    for (int q = 0; q < Qn; ++q) {
      float m = -3.4e38f;
      for (int j = 0; j < Kn; ++j) {
        float s = 0;
        for (int d = 0; d < 64; ++d)
          s += __half2float(Q[(size_t)q * 64 + d]) *
               __half2float(KT[(size_t)d * Kn + j]);
        s *= sc;
        p[j] = s;
        if (s > m) m = s;
      }
      float sum = 0;
      for (int j = 0; j < Kn; ++j) { p[j] = expf(p[j] - m); sum += p[j]; }
      for (int j = 0; j < Kn; ++j) p[j] /= sum;
      for (int d = 0; d < 64; ++d) {
        float s = 0;
        for (int j = 0; j < Kn; ++j)
          s += p[j] * __half2float(V[(size_t)j * 64 + d]);
        oref[d] = s;
      }
      const __half* O = &ho[((size_t)h * Qn + q) * 64];
      for (int d = 0; d < 64; ++d) {
        double a = oref[d], b = __half2float(O[d]);
        gnorm += a * a; dnorm += (a - b) * (a - b);
        double rel = fabs(a - b) / (fabs(a) + 1e-3);
        if (rel > worst) worst = rel;
      }
    }
  }
  double l2 = sqrt(dnorm) / sqrt(gnorm);
  printf("  (%d,%d) l2rel=%.3e maxrel=%.3e %s\n", Qn, Kn, l2, worst,
         l2 < 2e-2 ? "PASS" : "FAIL");
  return l2 < 2e-2 ? 0 : 1;
}

int main() {
  printf("== FlashSDPA kernel selftest ==\n");
  int f = 0;
  f += check(900, 900);
  f += check(900, 600);
  f += check(17, 97);   // 尾块/守卫
  f += check(64, 64);   // 单块
  printf("== %s ==\n", f ? "FAILED" : "ALL PASS");
  return f ? 1 : 0;
}
