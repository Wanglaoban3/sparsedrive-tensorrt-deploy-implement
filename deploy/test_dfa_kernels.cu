// ============================================================================
// DFA 内核离线单测: 同一份合成输入, v3(2通道/half2) vs v5(4通道/uint2)
//   1) 对拍 (预期逐位一致 — 逐通道累加顺序相同)
//   2) cudaEvent 计时 (200 iters)
// 用法: ./test_dfa_kernels
// 编译: nvcc -O3 -arch=sm_87 test_dfa_kernels.cu -o test_dfa_kernels
//       (与 dfaplug_v3.cu / dfaplug_v5.cu 同目录)
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define dfa_kernel kv3
#define dfa_trt tv3
#define dfa_reg rg3
#include "dfaplug_v3.cu"
#undef dfa_kernel
#undef dfa_trt
#undef dfa_reg


#define dfa_kernel kv6
#define dfa_trt tv6
#define dfa_reg rg6
#include "dfaplug_v6.cu"
#undef dfa_kernel
#undef dfa_trt
#undef dfa_reg

#define dfa_kernel kv5
#define dfa_trt tv5
#define dfa_reg rg5
#include "dfaplug_v5.cu"
#undef dfa_kernel
#undef dfa_trt
#undef dfa_reg

struct Cfg {
  int bs = 1, cams = 6, S = 4, C = 256, A = 900, P = 13, G = 8;
  const char* tag;
};

int main() {
  Cfg det;
  det.tag = "det";
  Cfg map;
  map.tag = "map";
  map.A = 100;
  map.P = 300;

  for (Cfg cfg : {det, map}) {
    // 特征图: 6 cam x (64x176 + 32x88 + 16x44 + 8x22)
    int per_cam = 64 * 176 + 32 * 88 + 16 * 44 + 8 * 22;
    int num_feat = cfg.cams * per_cam;
    const long long n_feat = (long long)num_feat * cfg.C;
    const long long n_loc = (long long)cfg.A * cfg.P * cfg.cams * 2;
    const long long n_log = (long long)cfg.A * cfg.cams * cfg.S * cfg.P *
                            cfg.G;

    std::vector<__half> h_feat(n_feat), h_loc(n_loc), h_log(n_log);
    std::vector<int> h_shape(cfg.cams * cfg.S * 2), h_ssi(cfg.cams * cfg.S);
    srand(42);
    auto rnd01 = []() { return (rand() % 20001) / 20000.0f; };
    for (auto& v : h_feat) v = __float2half((rand() % 2000 / 1000.f - 1.f)
                                            * (rnd01() * 30.f));
    int off = 0;
    for (int c = 0; c < cfg.cams; ++c) {
      const int hs[4] = {64, 32, 16, 8}, ws[4] = {176, 88, 44, 22};
      for (int s = 0; s < cfg.S; ++s) {
        h_shape[(c * cfg.S + s) * 2] = hs[s];
        h_shape[(c * cfg.S + s) * 2 + 1] = ws[s];
        h_ssi[c * cfg.S + s] = off;
        off += hs[s] * ws[s];
      }
    }
    for (long long i = 0; i < n_loc; ++i) {
      float v = rnd01();
      if (i % 3 == 0) v = v * 2 - 0.5f;  // 1/3 样本落到图外, 练 skip 分支
      h_loc[i] = __float2half(v);
    }
    for (auto& v : h_log) v = __float2half((rand() % 2000 / 1000.f - 1.f)
                                           * 4.f);

    __half *d_feat, *d_loc, *d_log, *d_ref, *d_new;
    int *d_shape, *d_ssi;
    cudaMalloc(&d_feat, n_feat * 2);
    cudaMalloc(&d_loc, n_loc * 2);
    cudaMalloc(&d_log, n_log * 2);
    cudaMalloc(&d_shape, h_shape.size() * 4);
    cudaMalloc(&d_ssi, h_ssi.size() * 4);
    cudaMalloc(&d_ref, (long long)cfg.A * cfg.C * 2);
    cudaMalloc(&d_new, (long long)cfg.A * cfg.C * 2);
    cudaMemcpy(d_feat, h_feat.data(), n_feat * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_loc, h_loc.data(), n_loc * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_log, h_log.data(), n_log * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, h_shape.data(), h_shape.size() * 4,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ssi, h_ssi.data(), h_ssi.size() * 4, cudaMemcpyHostToDevice);

    // 正确性
    kv3::dfa_launch_h2<int>(d_ref, d_feat, d_shape, d_ssi, d_loc, d_log,
                              cfg.bs, cfg.cams, num_feat, cfg.C, cfg.S,
                              cfg.A, cfg.P, cfg.G, 0);
    kv5::dfa_launch_h4<int>(d_new, d_feat, d_shape, d_ssi, d_loc, d_log,
                              cfg.bs, cfg.cams, num_feat, cfg.C, cfg.S,
                              cfg.A, cfg.P, cfg.G, 0);
    cudaDeviceSynchronize();
    std::vector<__half> h_ref(cfg.A * cfg.C), h_new(cfg.A * cfg.C);
    cudaMemcpy(h_ref.data(), d_ref, h_ref.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_new.data(), d_new, h_new.size() * 2,
               cudaMemcpyDeviceToHost);
    long long diff = 0;
    float maxd = 0;
    for (size_t i = 0; i < h_ref.size(); ++i) {
      float a = __half2float(h_ref[i]), b = __half2float(h_new[i]);
      if (a != b) {
        ++diff;
        maxd = fmaxf(maxd, fabsf(a - b));
      }
    }
    printf("[%s] diff_elems=%lld/%zu maxabs=%.6f\n", cfg.tag, diff,
           h_ref.size(), maxd);

    // 计时
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int which = 0; which < 2; ++which) {
      for (int w = 0; w < 20; ++w) {
        if (!which)
          kv3::dfa_launch_h2<int>(d_ref, d_feat, d_shape, d_ssi, d_loc,
                                    d_log, cfg.bs, cfg.cams, num_feat,
                                    cfg.C, cfg.S, cfg.A, cfg.P, cfg.G, 0);
        else
          kv5::dfa_launch_h4<int>(d_new, d_feat, d_shape, d_ssi, d_loc,
                                    d_log, cfg.bs, cfg.cams, num_feat,
                                    cfg.C, cfg.S, cfg.A, cfg.P, cfg.G, 0);
      }
      cudaEventRecord(e0);
      for (int it = 0; it < 200; ++it) {
        if (!which)
          kv3::dfa_launch_h2<int>(d_ref, d_feat, d_shape, d_ssi, d_loc,
                                    d_log, cfg.bs, cfg.cams, num_feat,
                                    cfg.C, cfg.S, cfg.A, cfg.P, cfg.G, 0);
        else
          kv5::dfa_launch_h4<int>(d_new, d_feat, d_shape, d_ssi, d_loc,
                                    d_log, cfg.bs, cfg.cams, num_feat,
                                    cfg.C, cfg.S, cfg.A, cfg.P, cfg.G, 0);
      }
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      float ms = 0;
      cudaEventElapsedTime(&ms, e0, e1);
      printf("[%s] %s: %.4f ms/iter\n", cfg.tag,
             which ? "v5(4ch)" : "v3(2ch)", ms / 200.f);
    }
    // v6: 需要 workspace; NCSP>1024 才走 3 内核路径
    const int ncsp = cfg.cams * cfg.S * cfg.P;
    if (ncsp > 1024) {
      void* d_ws = nullptr;
      const size_t ws_bytes = (size_t)cfg.A * cfg.G * 2 * 4 +
                              (size_t)cfg.A * kv6::DFA_MAP_CHUNKS * cfg.C * 4;
      cudaMalloc(&d_ws, ws_bytes);
      kv6::dfa_launch_h4<int>(d_new, d_feat, d_shape, d_ssi, d_loc, d_log,
                              cfg.bs, cfg.cams, num_feat, cfg.C, cfg.S,
                              cfg.A, cfg.P, cfg.G, 0, d_ws);
      cudaDeviceSynchronize();
      cudaMemcpy(h_new.data(), d_new, h_new.size() * 2,
                 cudaMemcpyDeviceToHost);
      long long diff6 = 0;
      float maxd6 = 0;
      for (size_t i = 0; i < h_ref.size(); ++i) {
        float a = __half2float(h_ref[i]), b = __half2float(h_new[i]);
        if (a != b) {
          ++diff6;
          maxd6 = fmaxf(maxd6, fabsf(a - b));
        }
      }
      printf("[%s] v6(3kernel) diff=%lld maxabs=%.6f err=%s\n", cfg.tag,
             diff6, maxd6, cudaGetErrorString(cudaGetLastError()));
      for (int w = 0; w < 20; ++w)
        kv6::dfa_launch_h4<int>(d_new, d_feat, d_shape, d_ssi, d_loc, d_log,
                                cfg.bs, cfg.cams, num_feat, cfg.C, cfg.S,
                                cfg.A, cfg.P, cfg.G, 0, d_ws);
      cudaEventRecord(e0);
      for (int it = 0; it < 200; ++it)
        kv6::dfa_launch_h4<int>(d_new, d_feat, d_shape, d_ssi, d_loc, d_log,
                                cfg.bs, cfg.cams, num_feat, cfg.C, cfg.S,
                                cfg.A, cfg.P, cfg.G, 0, d_ws);
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      float ms6 = 0;
      cudaEventElapsedTime(&ms6, e0, e1);
      printf("[%s] v6(3kernel): %.4f ms/iter\n", cfg.tag, ms6 / 200.f);
      printf("[%s] v6 err: %s\n", cfg.tag,
             cudaGetErrorString(cudaGetLastError()));
      cudaFree(d_ws);
    }
    cudaFree(d_feat);
    cudaFree(d_loc);
    cudaFree(d_log);
    cudaFree(d_shape);
    cudaFree(d_ssi);
    cudaFree(d_ref);
    cudaFree(d_new);
  }
  return 0;
}
