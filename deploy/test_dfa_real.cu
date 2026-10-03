// ============================================================================
// DFA 真实数据测试台: 逐调用点加载真实 dump, 对拍 + 计时 + 跳过率统计
// 用法: ./test_dfa_real <frame_dir> <iters>
// 数据: feat.f16, shape.i32, ssi.i32, c<k>_loc.f16 / c<k>_log.f16 /
//       c<k>_out.f16 (torch 参考), manifest.txt: "rank tag A c<k>"
// 内核: v3(half2 基线) 编在 kv3; 候选内核编在 kc (compact-live-list v7)
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
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

#define dfa_kernel kc
#define dfa_trt tc
#define dfa_reg rc
#include "dfaplug_v7.cu"
#undef dfa_kernel
#undef dfa_trt
#undef dfa_reg

static std::vector<char> read_file(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(1); }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<char> buf(sz);
  if (fread(buf.data(), 1, sz, f) != (size_t)sz) exit(1);
  fclose(f);
  return buf;
}

int main(int argc, char** argv) {
  std::string dir = argv[1];
  int iters = (argc > 2) ? atoi(argv[2]) : 200;
  auto shape_buf = read_file(dir + "/shape.i32");
  auto ssi_buf = read_file(dir + "/ssi.i32");
  auto feat_buf = read_file(dir + "/feat.f16");
  auto manifest = read_file(dir + "/manifest.txt");
  const int* h_shape = (const int*)shape_buf.data();
  const int* h_ssi = (const int*)ssi_buf.data();
  const __half* h_feat = (const __half*)feat_buf.data();

  int *d_shape, *d_ssi;
  cudaMalloc(&d_shape, shape_buf.size());
  cudaMalloc(&d_ssi, ssi_buf.size());
  cudaMemcpy(d_shape, h_shape, shape_buf.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ssi, h_ssi, ssi_buf.size(), cudaMemcpyHostToDevice);
  __half* d_feat;
  cudaMalloc(&d_feat, feat_buf.size());
  cudaMemcpy(d_feat, h_feat, feat_buf.size(), cudaMemcpyHostToDevice);
  const long long num_feat = feat_buf.size() / 2 / 256;

  char* line = strtok(manifest.data(), "\n");
  double tot_v3 = 0, tot_c = 0;
  while (line) {
    int rank, A;
    char tag[8], ck[16];
    if (sscanf(line, "%d %7s %d %15s", &rank, tag, &A, ck) != 4) break;
    const long long P = (A == 900) ? 13 : 300;
    auto loc_buf = read_file(dir + "/" + std::string(ck) + "_loc.f16");
    auto log_buf = read_file(dir + "/" + std::string(ck) + "_log.f16");
    auto w3_buf = read_file(dir + "/" + std::string(ck) + "_w3.f16");
    auto out_buf = read_file(dir + "/" + std::string(ck) + "_ref.f16");

    __half *d_loc, *d_log, *d_w3, *d_ref, *d_o3, *d_oc;
    cudaMalloc(&d_loc, loc_buf.size());
    cudaMalloc(&d_log, log_buf.size());
    cudaMalloc(&d_w3, w3_buf.size());
    cudaMalloc(&d_ref, (long long)A * 256 * 2);
    cudaMalloc(&d_o3, (long long)A * 256 * 2);
    cudaMalloc(&d_oc, (long long)A * 256 * 2);
    cudaMemcpy(d_loc, loc_buf.data(), loc_buf.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_log, log_buf.data(), log_buf.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3, w3_buf.data(), w3_buf.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref, out_buf.data(), out_buf.size(), cudaMemcpyHostToDevice);

    // 基线 v3
    kv3::dfa_launch_h2<int>(d_o3, d_feat, d_shape, d_ssi, d_loc, (const __half*)d_w3, 1,
                              6, (int)num_feat, 256, 4, A, (int)P, 8, 0);
    cudaDeviceSynchronize();
    std::vector<__half> h_o3(A * 256), h_ref(A * 256);
    cudaMemcpy(h_o3.data(), d_o3, h_o3.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), d_ref, h_ref.size() * 2,
               cudaMemcpyDeviceToHost);
    double denom = 0, dsum = 0;
    float maxd = 0;
    for (size_t i = 0; i < h_ref.size(); ++i) {
      double a = __half2float(h_ref[i]), b = __half2float(h_o3[i]);
      denom += fabs(a);
      dsum += fabs(a - b);
      maxd = fmaxf(maxd, (float)fabs(a - b));
    }
    const double rel = dsum / std::max(1e-9, denom);

    // 计时 v3
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int w = 0; w < 20; ++w)
      kv3::dfa_launch_h2<int>(d_o3, d_feat, d_shape, d_ssi, d_loc, (const __half*)d_w3, 1,
                                6, (int)num_feat, 256, 4, A, (int)P, 8, 0);
    cudaEventRecord(e0);
    for (int it = 0; it < iters; ++it)
      kv3::dfa_launch_h2<int>(d_o3, d_feat, d_shape, d_ssi, d_loc, (const __half*)d_w3, 1,
                                6, (int)num_feat, 256, 4, A, (int)P, 8, 0);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms3 = 0;
    cudaEventElapsedTime(&ms3, e0, e1);
    ms3 /= iters;

    // 候选 compact-live-list v7
    void* d_ws = nullptr;
    const int maxe = 7200;  // min(DFA_MAXE, NCSP): map 7200 / det 312
    const size_t ws_bytes = (size_t)A * 8 * (2 * 4 + 4 + maxe * 4) +
                            4096;
    cudaMalloc(&d_ws, ws_bytes);
    for (int w = 0; w < 20; ++w)
      kc::dfa_launch_live<int>(d_oc, d_feat, d_shape, d_ssi, d_loc, d_log,
                                 1, 6, (int)num_feat, 256, 4, A, (int)P, 8,
                                 d_ws, 0);
    cudaEventRecord(e0);
    for (int it = 0; it < iters; ++it)
      kc::dfa_launch_live<int>(d_oc, d_feat, d_shape, d_ssi, d_loc, d_log,
                                 1, 6, (int)num_feat, 256, 4, A, (int)P, 8,
                                 d_ws, 0);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float msc = 0;
    cudaEventElapsedTime(&msc, e0, e1);
    msc /= iters;

    std::vector<__half> h_oc(A * 256);
    cudaMemcpy(h_oc.data(), d_oc, h_oc.size() * 2, cudaMemcpyDeviceToHost);
    double csum = 0;
    float cmaxd = 0;
    for (size_t i = 0; i < h_ref.size(); ++i) {
      double a = __half2float(h_ref[i]), b = __half2float(h_oc[i]);
      csum += fabs(a - b);
      cmaxd = fmaxf(cmaxd, (float)fabs(a - b));
    }
    const double relc = csum / std::max(1e-9, denom);

    printf("[%s %s] A=%lld P=%lld  v3: %.3fms rel=%.4f | live: %.3fms "
           "rel=%.4f maxabs=%.4f | speedup x%.2f\n",
           tag, ck, (long long)A, P, ms3, rel, msc, relc, cmaxd,
           ms3 / std::max(0.001f, msc));
    tot_v3 += ms3;
    tot_c += msc;
    cudaFree(d_loc);
    cudaFree(d_log);
    cudaFree(d_w3);
    cudaFree(d_ref);
    cudaFree(d_o3);
    cudaFree(d_oc);
    cudaFree(d_ws);
    line = strtok(nullptr, "\n");
  }
  printf("TOTAL: v3=%.2fms  live=%.2fms  (target 5.0ms)\n", tot_v3, tot_c);
  return 0;
}
