// ============================================================================
// test_dfa_v11: 真实 dump 数据上 v3 / v8 / v11 (per-anchor 条目 + 向量化
// gather) 逐调用 A/B + 分相计时。dump 格式与 test_dfa_ab 完全相同
// (libdfaplug_dump.so / convert_engine_dump2.py 产物, 裸 logits)。
// 用法: test_dfa_v11 <dump_dir> [iters=50] [warmup=10] [only_rank=-1]
// 判读: v11vseng / v11vsv8 应为 ulp 级 (条目权重 half 化 ~1e-3/项, 均值后
// 输出差异 ~1e-4); 计时关注 map 侧 gather 相对 v8 的收益。
// ============================================================================
#define DFA_NO_REGISTER
#define dfa_kernel kv3
#define dfa_trt tv3
#define dfa_reg rg3
#include "dfaplug_v3.cu"
#undef DFA_NO_REGISTER
#undef dfa_kernel
#undef dfa_trt
#undef dfa_reg

#define DFA_NO_REGISTER
#define DFA_V8_NO_FALLBACK
#define DFA_V8_NS kv8
#define DFA_TRT_NS tv8
#define DFA_REG_NS rg8
#include "dfaplug_v8.cu"

#define DFA_NO_REGISTER
#define DFA_V11_NO_FALLBACK
#define DFA_V8_EXTERNAL
#define DFA_V8_CALL_NS kv8
#define DFA_V11_NS kv11
#define DFA_TRT11_NS tv11
#define DFA_REG11_NS rg11
#include "dfaplug_v11.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

static std::vector<__half> read_f16(std::string path, size_t* n_half) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { printf("MISSING %s\n", path.c_str()); exit(2); }
  fseek(f, 0, SEEK_END);
  long bytes = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<__half> v(bytes / 2);
  if (bytes && fread(v.data(), 1, bytes, f) != (size_t)bytes) {
    printf("SHORT READ %s\n", path.c_str()); exit(2);
  }
  fclose(f);
  if (n_half) *n_half = v.size();
  return v;
}

struct Stats { double l2rel, maxabs, maxrel; };

// a=候选(half), b=参考(fp32); maxrel 用 max(|b|,0.05) 做分母
static Stats compare(const std::vector<__half>& a, const std::vector<float>& b) {
  Stats s{0, 0, 0};
  double num = 0, den = 0;
  for (size_t i = 0; i < b.size(); ++i) {
    const double x = __half2float(a[i]), y = b[i];
    const double d = x - y;
    num += d * d;
    den += y * y;
    if (std::abs(d) > s.maxabs) s.maxabs = std::abs(d);
    const double r = std::abs(d) / std::max(std::abs(y), 0.05);
    if (r > s.maxrel) s.maxrel = r;
  }
  s.l2rel = std::sqrt(num) / std::max(1e-12, std::sqrt(den));
  return s;
}

static size_t ws_need11(long long nA, long long NCSP, long long C) {
  const int split = kv11::v11_split_for((int)nA);
  const size_t cnt_bytes = (size_t)nA * 4;
  const size_t ent_off = (cnt_bytes + (size_t)255) & ~(size_t)255;
  const size_t ent_bytes = (size_t)nA * NCSP * sizeof(kv11::EntryA);
  const size_t part_off = (ent_off + ent_bytes + (size_t)255) & ~(size_t)255;
  return part_off + (size_t)nA * C * split * 4;
}

static float time_launch(int kind, int calls, __half** d_out, __half** d_feat,
                         int* d_shape, int* d_ssi, __half** d_loc,
                         __half** d_log, int bs, int cams, int num_feat, int C,
                         int S, const int* vA, const int* vP, int G,
                         void* d_ws8, size_t ws8, void* d_ws11, size_t ws11,
                         int iters, int warmup) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
#define ONE_CALL(i)                                                           \
  do {                                                                        \
    const int A = vA[i], P = vP[i];                                           \
    if (kind == 3)                                                            \
      kv3::dfa_launch_h2<int>(d_out[i], d_feat[i], d_shape, d_ssi, d_loc[i],  \
                              d_log[i], bs, cams, num_feat, C, S, A, P, G, 0);\
    else if (kind == 80)                                                      \
      kv8::dfa_plan_v8<int>(d_shape, d_ssi, d_loc[i], d_log[i], bs, cams,     \
                            num_feat, C, S, A, P, G, d_ws8, ws8, 0);          \
    else if (kind == 81)                                                      \
      kv8::dfa_gather_v8(d_out[i], d_feat[i], d_ws8, ws8, bs, A, num_feat, C, \
                         cams, S, P, G, 0);                                   \
    else if (kind == 8)                                                       \
      kv8::dfa_launch_v8<int>(d_out[i], d_feat[i], d_shape, d_ssi, d_loc[i],  \
                              d_log[i], bs, cams, num_feat, C, S, A, P, G,    \
                              d_ws8, ws8, 0);                                 \
    else if (kind == 110)                                                     \
      kv11::dfa_plan_v11<int>(d_shape, d_ssi, d_loc[i], d_log[i], bs, cams,   \
                              num_feat, C, S, A, P, G, nullptr, d_ws11, ws11, \
                              0);                                             \
    else if (kind == 111)                                                     \
      kv11::dfa_gather_v11(d_out[i], d_feat[i], d_ws11, ws11, bs, A, C, cams, \
                           S, P, kv11::v11_split_for(bs * A), 0);             \
    else                                                                      \
      kv11::dfa_launch_v11<int>(d_out[i], d_feat[i], d_shape, d_ssi,          \
                                d_loc[i], d_log[i], bs, cams, num_feat, C, S, \
                                A, P, G, d_ws11, ws11, 0);                    \
  } while (0)
  for (int w = 0; w < warmup; ++w)
    for (int i = 0; i < calls; ++i) ONE_CALL(i);
  cudaEventRecord(e0);
  for (int it = 0; it < iters; ++it)
    for (int i = 0; i < calls; ++i) ONE_CALL(i);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return ms / iters;
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: %s <dump_dir> [iters] [warmup] [only]\n", argv[0]); return 1; }
  const std::string dir = argv[1];
  const int iters = argc > 2 ? atoi(argv[2]) : 50;
  const int warmup = argc > 3 ? atoi(argv[3]) : 10;
  const int only = argc > 4 ? atoi(argv[4]) : -1;

  const int bs = 1, C = 256;
  std::vector<int> vA, vP, vG, vcams, vS, vk;
  {
    FILE* f = fopen((dir + "/manifest.txt").c_str(), "r");
    if (!f) { printf("bad manifest\n"); return 2; }
    char line[512];
    while (fgets(line, sizeof(line), f)) {
      int k, bs_, num_feat_, C_, A, cams, S, P, G, idxt;
      void* fptr = nullptr;
      if (sscanf(line,
                 "%d bs=%d num_feat=%d C=%d A=%d cams=%d S=%d P=%d G=%d "
                 "idxt=%d featptr=%p",
                 &k, &bs_, &num_feat_, &C_, &A, &cams, &S, &P, &G, &idxt,
                 &fptr) == 11) {
        if (only >= 0 && k != only) continue;
        vk.push_back(k);
        vA.push_back(A);
        vP.push_back(P);
        vG.push_back(G);
        vcams.push_back(cams);
        vS.push_back(S);
      }
    }
    fclose(f);
  }
  const int ncalls = (int)vA.size();
  if (!ncalls) { printf("no manifest entries\n"); return 2; }
  printf("calls=%d iters=%d warmup=%d\n", ncalls, iters, warmup);
  const int cams = vcams[0], S = vS[0], G = vG[0];

  char k0[64];
  snprintf(k0, sizeof(k0), "%d", vk[0]);
  size_t nfeat = 0;
  std::vector<__half> feat = read_f16(dir + "/dfa" + k0 + "_feat.f16", &nfeat);
  const int num_feat = (int)(nfeat / C);
  std::vector<int> shape, ssi;
  {
    auto read_i32 = [&](const char* fn, std::vector<int>& out) {
      FILE* f = fopen((dir + "/" + fn).c_str(), "rb");
      if (!f) { printf("MISSING %s\n", fn); exit(2); }
      fseek(f, 0, SEEK_END);
      long bytes = ftell(f);
      fseek(f, 0, SEEK_SET);
      std::vector<int> v(bytes / 4);
      if (bytes && fread(v.data(), 1, (size_t)bytes, f) != (size_t)bytes) {
        printf("SHORT READ %s\n", fn);
        exit(2);
      }
      fclose(f);
      out = v;
    };
    read_i32(("dfa" + std::string(k0) + "_shape.bin").c_str(), shape);
    read_i32(("dfa" + std::string(k0) + "_ssi.bin").c_str(), ssi);
  }
  printf("feat [%d,256]  shape(%zu)=[%d,%d,%d,%d]  ssi(%zu)=[%d,%d,%d,%d...]\n",
         num_feat, shape.size(), shape[0], shape[2], shape[4], shape[6],
         ssi.size(), ssi[0], ssi[1], ssi[2], ssi[3]);

  std::vector<std::vector<__half>> h_loc(ncalls), h_log(ncalls);
  std::vector<std::vector<float>> h_ref(ncalls);
  for (int i = 0; i < ncalls; ++i) {
    char kb[64];
    snprintf(kb, sizeof(kb), "%d", vk[i]);
    const std::string pre = dir + "/dfa" + kb;
    h_loc[i] = read_f16(pre + "_loc.f16", nullptr);
    h_log[i] = read_f16(pre + "_log.f16", nullptr);
    {
      std::vector<__half> o = read_f16(pre + "_out.f16", nullptr);
      h_ref[i].resize(o.size());
      for (size_t j = 0; j < o.size(); ++j) h_ref[i][j] = __half2float(o[j]);
    }
    const size_t want_loc = (size_t)bs * vA[i] * vP[i] * cams * 2;
    const size_t want_log = (size_t)bs * vA[i] * cams * S * vP[i] * G;
    const size_t want_ref = (size_t)bs * vA[i] * C;
    if (h_loc[i].size() != want_loc || h_log[i].size() != want_log ||
        h_ref[i].size() != want_ref) {
      printf("dfa%d size mismatch: loc %zu/%zu log %zu/%zu out %zu/%zu\n",
             vk[i], h_loc[i].size(), want_loc, h_log[i].size(), want_log,
             h_ref[i].size(), want_ref);
      return 2;
    }
  }

  // device
  std::vector<__half*> d_feat(ncalls), d_loc(ncalls), d_log(ncalls),
      d_out8(ncalls), d_out11(ncalls);
  int *d_shape = nullptr, *d_ssi = nullptr;
  cudaMalloc(&d_shape, shape.size() * 4);
  cudaMalloc(&d_ssi, ssi.size() * 4);
  cudaMemcpy(d_shape, shape.data(), shape.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ssi, ssi.data(), ssi.size() * 4, cudaMemcpyHostToDevice);
  {
    __half* df = nullptr;
    cudaMalloc(&df, feat.size() * 2);
    cudaMemcpy(df, feat.data(), feat.size() * 2, cudaMemcpyHostToDevice);
    for (int i = 0; i < ncalls; ++i) d_feat[i] = df;
  }
  for (int i = 0; i < ncalls; ++i) {
    cudaMalloc(&d_loc[i], h_loc[i].size() * 2);
    cudaMemcpy(d_loc[i], h_loc[i].data(), h_loc[i].size() * 2,
               cudaMemcpyHostToDevice);
    cudaMalloc(&d_log[i], h_log[i].size() * 2);
    cudaMemcpy(d_log[i], h_log[i].data(), h_log[i].size() * 2,
               cudaMemcpyHostToDevice);
    cudaMalloc(&d_out8[i], (size_t)vA[i] * C * 2);
    cudaMalloc(&d_out11[i], (size_t)vA[i] * C * 2);
  }
  // v8 与 v11 workspace 布局不同, 分开分配 (各取所有调用最大值)
  size_t ws8_max = 0, ws11_max = 0;
  for (int i = 0; i < ncalls; ++i) {
    const long long nA = (long long)bs * vA[i];
    const long long NCSP = (long long)cams * S * vP[i];
    size_t n8 = ((size_t)(nA * G * 4) + 255) & ~(size_t)255;
    n8 += (size_t)(nA * G * NCSP) * 32;
    if (n8 > ws8_max) ws8_max = n8;
    size_t n11 = ws_need11(nA, NCSP, C);
    if (n11 > ws11_max) ws11_max = n11;
  }
  void* d_ws8 = nullptr;
  void* d_ws11 = nullptr;
  cudaMalloc(&d_ws8, ws8_max);
  cudaMalloc(&d_ws11, ws11_max);
  printf("ws8_max=%.1fMB ws11_max=%.1fMB (%.2fx)\n", ws8_max / 1048576.0,
         ws11_max / 1048576.0, (double)ws8_max / (double)ws11_max);

  // 正确性: v8 / v11 vs 引擎真值 (out.f16 = v3 引擎输出), v11 vs v8
  for (int i = 0; i < ncalls; ++i) {
    const int A = vA[i], P = vP[i];
    int rc8 = kv8::dfa_launch_v8<int>(d_out8[i], d_feat[i], d_shape, d_ssi,
                                      d_loc[i], d_log[i], bs, cams, num_feat,
                                      C, S, A, P, G, d_ws8, ws8_max, 0);
    int rc11 = kv11::dfa_launch_v11<int>(d_out11[i], d_feat[i], d_shape, d_ssi,
                                         d_loc[i], d_log[i], bs, cams,
                                         num_feat, C, S, A, P, G, d_ws11,
                                         ws11_max, 0);
    cudaDeviceSynchronize();
    std::vector<__half> o8(A * C), o11(A * C);
    cudaMemcpy(o8.data(), d_out8[i], o8.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(o11.data(), d_out11[i], o11.size() * 2, cudaMemcpyDeviceToHost);
    Stats s8 = compare(o8, h_ref[i]);
    Stats s11 = compare(o11, h_ref[i]);
    Stats s118 = compare(o11, std::vector<float>(o8.begin(), o8.end()));
    printf("dfa%-2d %-3s A=%-4d rc8=%d rc11=%d | v8vseng l2=%.2e maxabs=%.2e"
           " | v11vseng l2=%.2e maxabs=%.2e | v11vsv8 l2=%.2e maxabs=%.2e\n",
           vk[i], (A == 900 ? "det" : "map"), A, rc8, rc11,
           s8.l2rel, s8.maxabs, s11.l2rel, s11.maxabs, s118.l2rel, s118.maxabs);
  }

  // v11 条目统计 (per-anchor): 最后一次 plan 后的 counts
  {
    const int i = ncalls - 1;
    const int A = vA[i];
    kv11::dfa_plan_v11<int>(d_shape, d_ssi, d_loc[i], d_log[i], bs, cams,
                            num_feat, C, S, A, vP[i], G, nullptr, d_ws11,
                            ws11_max, 0);
    cudaDeviceSynchronize();
    std::vector<int> hcnt((size_t)bs * A);
    cudaMemcpy(hcnt.data(), d_ws11, hcnt.size() * 4, cudaMemcpyDeviceToHost);
    long long tot = 0;
    int mx = 0;
    for (int v : hcnt) { tot += v; if (v > mx) mx = v; }
    printf("v11 entries (last call A=%d): total=%lld anchor-max=%d NCSP=%d"
           " (v8 布局下等效 total*8)\n", A, tot, mx, cams * S * vP[i]);
    // 每调用 map 的 per-anchor 分布 (偏斜诊断: split 分片按 ceil(cnt/split))
    for (int q = 0; q < ncalls; ++q) {
      if (vA[q] == 900) continue;
      kv11::dfa_plan_v11<int>(d_shape, d_ssi, d_loc[q], d_log[q], bs, cams,
                              num_feat, C, S, vA[q], vP[q], G, nullptr,
                              d_ws11, ws11_max, 0);
      cudaDeviceSynchronize();
      std::vector<int> hc((size_t)bs * vA[q]);
      cudaMemcpy(hc.data(), d_ws11, hc.size() * 4, cudaMemcpyDeviceToHost);
      std::vector<int> sorted = hc;
      std::sort(sorted.begin(), sorted.end());
      const long long t2 =
          std::accumulate(sorted.begin(), sorted.end(), 0ll);
      printf("MAPCOUNT dfa%-2d A=%d min=%d p50=%d p90=%d max=%d avg=%lld"
             " split=%d maxwarp=%d\n",
             vk[q], vA[q], sorted.front(),
             sorted[sorted.size() / 2], sorted[(sorted.size() * 9) / 10],
             sorted.back(), t2 / (long long)sorted.size(),
             kv11::v11_split_for(vA[q]),
             (sorted.back() + kv11::v11_split_for(vA[q]) - 1) /
                 kv11::v11_split_for(vA[q]));
    }
  }

  // 计时: v3 / v8 (plan+gather) / v11 (plan+gather+finalize)
  float tot8 = 0, tot11 = 0;
  for (int i = 0; i < ncalls; ++i) {
    const int a1[1] = {vA[i]}, p1[1] = {vP[i]};
    float t8 = time_launch(8, 1, &d_out8[i], &d_feat[i], d_shape, d_ssi,
                           &d_loc[i], &d_log[i], bs, cams, num_feat, C, S,
                           a1, p1, G, d_ws8, ws8_max, d_ws11, ws11_max,
                           iters, warmup);
    float t11 = time_launch(11, 1, &d_out11[i], &d_feat[i], d_shape, d_ssi,
                            &d_loc[i], &d_log[i], bs, cams, num_feat, C, S,
                            a1, p1, G, d_ws8, ws8_max, d_ws11, ws11_max,
                            iters, warmup);
    float tp8 = time_launch(80, 1, &d_out8[i], &d_feat[i], d_shape, d_ssi,
                            &d_loc[i], &d_log[i], bs, cams, num_feat, C, S,
                            a1, p1, G, d_ws8, ws8_max, d_ws11, ws11_max,
                            iters, warmup);
    float tg8 = time_launch(81, 1, &d_out8[i], &d_feat[i], d_shape, d_ssi,
                            &d_loc[i], &d_log[i], bs, cams, num_feat, C, S,
                            a1, p1, G, d_ws8, ws8_max, d_ws11, ws11_max,
                            iters, warmup);
    float tp11 = time_launch(110, 1, &d_out11[i], &d_feat[i], d_shape, d_ssi,
                             &d_loc[i], &d_log[i], bs, cams, num_feat, C, S,
                             a1, p1, G, d_ws8, ws8_max, d_ws11, ws11_max,
                             iters, warmup);
    float tg11 = time_launch(111, 1, &d_out11[i], &d_feat[i], d_shape, d_ssi,
                             &d_loc[i], &d_log[i], bs, cams, num_feat, C, S,
                             a1, p1, G, d_ws8, ws8_max, d_ws11, ws11_max,
                             iters, warmup);
    tot8 += t8;
    tot11 += t11;
    printf("TIME dfa%-2d %-3s A=%-4d P=%-3d v8=%.4f (p=%.4f g=%.4f)"
           " v11=%.4f (p=%.4f g=%.4f) g-speedup=%.2fx full=%.2fx\n",
           vk[i], (vA[i] == 900 ? "det" : "map"), vA[i], vP[i], t8, tp8, tg8,
           t11, tp11, tg11,
           tg8 / (tg11 > 0 ? tg11 : 1e-9f), t8 / (t11 > 0 ? t11 : 1e-9f));
  }
  float seq8 = time_launch(8, ncalls, d_out8.data(), d_feat.data(), d_shape,
                           d_ssi, d_loc.data(), d_log.data(), bs, cams,
                           num_feat, C, S, vA.data(), vP.data(), G, d_ws8,
                           ws8_max, d_ws11, ws11_max, iters, warmup);
  float seq11 = time_launch(11, ncalls, d_out11.data(), d_feat.data(), d_shape,
                            d_ssi, d_loc.data(), d_log.data(), bs, cams,
                            num_feat, C, S, vA.data(), vP.data(), G, d_ws8,
                            ws8_max, d_ws11, ws11_max, iters, warmup);
  printf("TOTAL per-call-sum v8=%.4fms v11=%.4fms x%.2f | sequence v8=%.4fms"
         " v11=%.4fms x%.2f | saved=%.3fms/frame\n",
         tot8, tot11, tot8 / tot11, seq8, seq11, seq8 / seq11, seq8 - seq11);
  printf("DFA_V11_AB_DONE\n");
  return 0;
}
