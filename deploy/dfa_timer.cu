// ============================================================================
// 最简 DFA 真实数据计时: 引擎 dump 数据 → v3 内核 → 计时
// 用法: ./dfa_timer <data_dir> <iters>
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define dfa_kernel kv3
#define dfa_trt tv3
#define dfa_reg rg3
#include "dfaplug_v3.cu"
#undef dfa_kernel
#undef dfa_trt
#undef dfa_reg

static std::vector<char> rd(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) { fprintf(stderr, "miss %s\n", p.c_str()); exit(1); }
  fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<char> b(s);
  if (fread(b.data(), 1, s, f) != (size_t)s) exit(1);
  fclose(f); return b;
}

int main(int argc, char** argv) {
  std::string dir = argv[1];
  int iters = argc > 2 ? atoi(argv[2]) : 200;
  auto sb = rd(dir + "/shape.i32");
  auto sib = rd(dir + "/ssi.i32");
  auto fb = rd(dir + "/feat.f16");
  auto mb = rd(dir + "/manifest.txt");

  int *d_sh, *d_si;
  cudaMalloc(&d_sh, sb.size()); cudaMalloc(&d_si, sib.size());
  cudaMemcpy(d_sh, sb.data(), sb.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_si, sib.data(), sib.size(), cudaMemcpyHostToDevice);
  __half* d_ft;
  cudaMalloc(&d_ft, fb.size());
  cudaMemcpy(d_ft, fb.data(), fb.size(), cudaMemcpyHostToDevice);
  const int num_feat = (int)(fb.size() / 2 / 256);

  char* ln = strtok(mb.data(), "\n");
  float tot = 0;
  while (ln) {
    int rank, A; char tag[8], ck[16];
    if (sscanf(ln, "%d %7s %d %15s", &rank, tag, &A, ck) != 4) break;
    int P = (A == 900) ? 13 : 300;
    auto lb = rd(dir + "/" + std::string(ck) + "_loc.f16");
    auto wb = rd(dir + "/" + std::string(ck) + "_w3.f16");
    __half *d_lc, *d_w;
    cudaMalloc(&d_lc, lb.size()); cudaMalloc(&d_w, wb.size());
    cudaMemcpy(d_lc, lb.data(), lb.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, wb.data(), wb.size(), cudaMemcpyHostToDevice);
    __half* d_out;
    cudaMalloc(&d_out, (long long)A * 256 * 2);

    // warmup
    for (int i = 0; i < 10; ++i)
      kv3::dfa_launch_h2<int>(d_out, d_ft, d_sh, d_si, d_lc,
                              (const __half*)d_w, 1, 6, num_feat, 256, 4,
                              A, P, 8, 0);
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i)
      kv3::dfa_launch_h2<int>(d_out, d_ft, d_sh, d_si, d_lc,
                              (const __half*)d_w, 1, 6, num_feat, 256, 4,
                              A, P, 8, 0);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1); ms /= iters;
    printf("[%s %s] A=%d P=%d  %.3f ms\n", tag, ck, A, P, ms);
    tot += ms;
    cudaFree(d_lc); cudaFree(d_w); cudaFree(d_out);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    ln = strtok(nullptr, "\n");
  }
  printf("TOTAL v3: %.2f ms\n", tot);
  return 0;
}
