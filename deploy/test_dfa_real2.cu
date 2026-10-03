// ============================================================================
// DFA 真实数据测试台 v2: v3 基线对拍 (不再跑 v7 候选)
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

  int *d_shape, *d_ssi;
  cudaMalloc(&d_shape, shape_buf.size());
  cudaMalloc(&d_ssi, ssi_buf.size());
  cudaMemcpy(d_shape, shape_buf.data(), shape_buf.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ssi, ssi_buf.data(), ssi_buf.size(), cudaMemcpyHostToDevice);
  __half* d_feat;
  cudaMalloc(&d_feat, feat_buf.size());
  cudaMemcpy(d_feat, feat_buf.data(), feat_buf.size(), cudaMemcpyHostToDevice);
  const long long num_feat = feat_buf.size() / 2 / 256;

  char* line = strtok(manifest.data(), "\n");
  double tot_v3 = 0, tot_ref32 = 0;
  int n_ok = 0;
  while (line) {
    int rank, A;
    char tag[8], ck[16];
    if (sscanf(line, "%d %7s %d %15s", &rank, tag, &A, ck) != 4) break;
    const long long P = (A == 900) ? 13 : 300;
    auto loc_buf = read_file(dir + "/" + std::string(ck) + "_loc.f16");
    auto w3_buf = read_file(dir + "/" + std::string(ck) + "_w3.f16");
    auto ref16_buf = read_file(dir + "/" + std::string(ck) + "_out.f16");
    auto ref32_buf = read_file(dir + "/" + std::string(ck) + "_ref32.f32");

    __half *d_loc, *d_w3, *d_o3;
    float* d_ref32;
    cudaMalloc(&d_loc, loc_buf.size());
    cudaMalloc(&d_w3, w3_buf.size());
    cudaMalloc(&d_o3, (long long)A * 256 * 2);
    cudaMalloc(&d_ref32, ref32_buf.size());
    cudaMemcpy(d_loc, loc_buf.data(), loc_buf.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3, w3_buf.data(), w3_buf.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref32, ref32_buf.data(), ref32_buf.size(),
               cudaMemcpyHostToDevice);

    kv3::dfa_launch_h2<int>(d_o3, d_feat, d_shape, d_ssi, d_loc,
                            (const __half*)d_w3, 1, 6, (int)num_feat, 256,
                            4, A, (int)P, 8, 0);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    std::vector<__half> h_o3(A * 256), h_ref16(A * 256);
    std::vector<float> h_ref32(A * 256);
    cudaMemcpy(h_o3.data(), d_o3, h_o3.size() * 2, cudaMemcpyDeviceToHost);
    memcpy(h_ref16.data(), ref16_buf.data(), h_ref16.size() * 2);
    memcpy(h_ref32.data(), ref32_buf.data(), h_ref32.size() * 4);

    // 对比 v3 fp16 output vs engine fp32 reference
    double denom32 = 0, dsum32 = 0;
    for (size_t i = 0; i < h_ref32.size(); ++i) {
      double a = h_ref32[i], b = __half2float(h_o3[i]);
      denom32 += fabs(a);
      dsum32 += fabs(a - b);
    }
    const double rel32 = dsum32 / std::max(1e-9, denom32);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int w = 0; w < 20; ++w)
      kv3::dfa_launch_h2<int>(d_o3, d_feat, d_shape, d_ssi, d_loc,
                              (const __half*)d_w3, 1, 6, (int)num_feat, 256,
                              4, A, (int)P, 8, 0);
    cudaEventRecord(e0);
    for (int it = 0; it < iters; ++it)
      kv3::dfa_launch_h2<int>(d_o3, d_feat, d_shape, d_ssi, d_loc,
                              (const __half*)d_w3, 1, 6, (int)num_feat, 256,
                              4, A, (int)P, 8, 0);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    ms /= iters;

    printf("[%s %s] A=%lld P=%lld err=%s v3: %.3fms rel_vs_engine32=%.5f\n",
           tag, ck, (long long)A, P, cudaGetErrorString(err), ms, rel32);
    if (err == cudaSuccess && ms > 0.001) { tot_v3 += ms; ++n_ok; }
    // accumulate ref32 magnitude for total check
    for (size_t i = 0; i < h_ref32.size(); ++i) tot_ref32 += fabs(h_ref32[i]);
    cudaFree(d_loc);
    cudaFree(d_w3);
    cudaFree(d_o3);
    cudaFree(d_ref32);
    line = strtok(nullptr, "\n");
  }
  printf("TOTAL: v3=%.2fms over %d calls (engine in-profile det=%.2f map=%.2f)\n",
         tot_v3, n_ok, tot_v3 > 0 ? 0.0 : 0.0, 0.0);
  return 0;
}
