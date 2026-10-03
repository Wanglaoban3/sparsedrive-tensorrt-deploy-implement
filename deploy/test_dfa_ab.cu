// ============================================================================
// test_dfa_ab: 真实 dump 数据上 v3 vs v8 (plan+gather) 逐调用 A/B。
// 用法: test_dfa_ab <dump_dir> [iters=50] [warmup=10] [only_rank=-1]
// 数据 (deploy/convert_engine_dump2.py 产物):
//   feat.f16 [89760,256], shape.i32 [6,4,2], ssi.i32 [6,4]
//   c{k}_loc.f16 [A,P,6,2]   c{k}_log.f16 [A,6,4,P,8]  (裸 logits!)
//   c{k}_ref32.f32 [A,256]
//   manifest.txt 行: "<rank> <tag det|map> <A> c<k>"
// 注意: c{k}_w3.f16 是后验 softmax 权重, v3/v8 语义要的是裸 logits,
//       这里一律喂 _log.f16。
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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static std::vector<float> read_f32(std::string path, size_t* n) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { printf("MISSING %s\n", path.c_str()); exit(2); }
  fseek(f, 0, SEEK_END);
  long bytes = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<float> v(bytes / 4);
  if (bytes && fread(v.data(), 1, bytes, f) != (size_t)bytes) {
    printf("SHORT READ %s\n", path.c_str()); exit(2);
  }
  fclose(f);
  if (n) *n = v.size();
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

static float time_launch(int kind, int calls, __half** d_out, __half** d_feat,
                         int* d_shape, int* d_ssi, __half** d_loc,
                         __half** d_log, int bs, int cams, int num_feat, int C,
                         int S, const int* vA, const int* vP, int G,
                         void* d_ws, size_t ws_size, int iters, int warmup) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  for (int w = 0; w < warmup; ++w) {
    for (int i = 0; i < calls; ++i) {
      const int A = vA[i], P = vP[i];
      if (kind == 3)
        kv3::dfa_launch_h2<int>(d_out[i], d_feat[i], d_shape, d_ssi, d_loc[i],
                                d_log[i], bs, cams, num_feat, C, S, A, P, G, 0);
      else if (kind == 80)
        kv8::dfa_plan_v8<int>(d_shape, d_ssi, d_loc[i], d_log[i], bs, cams,
                              num_feat, C, S, A, P, G, d_ws, ws_size, 0);
      else if (kind == 81)
        kv8::dfa_gather_v8(d_out[i], d_feat[i], d_ws, ws_size, bs, A,
                           num_feat, C, cams, S, P, G, 0);
      else
        kv8::dfa_launch_v8<int>(d_out[i], d_feat[i], d_shape, d_ssi, d_loc[i],
                                d_log[i], bs, cams, num_feat, C, S, A, P, G,
                                d_ws, ws_size, 0);
    }
  }
  cudaEventRecord(e0);
  for (int it = 0; it < iters; ++it) {
    for (int i = 0; i < calls; ++i) {
      const int A = vA[i], P = vP[i];
      if (kind == 3)
        kv3::dfa_launch_h2<int>(d_out[i], d_feat[i], d_shape, d_ssi, d_loc[i],
                                d_log[i], bs, cams, num_feat, C, S, A, P, G, 0);
      else if (kind == 80)
        kv8::dfa_plan_v8<int>(d_shape, d_ssi, d_loc[i], d_log[i], bs, cams,
                              num_feat, C, S, A, P, G, d_ws, ws_size, 0);
      else if (kind == 81)
        kv8::dfa_gather_v8(d_out[i], d_feat[i], d_ws, ws_size, bs, A,
                           num_feat, C, cams, S, P, G, 0);
      else
        kv8::dfa_launch_v8<int>(d_out[i], d_feat[i], d_shape, d_ssi, d_loc[i],
                                d_log[i], bs, cams, num_feat, C, S, A, P, G,
                                d_ws, ws_size, 0);
    }
  }
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

  // manifest (libdfaplug_dump.so 产物):
  // "<k> bs=1 num_feat=89760 C=256 A=900 cams=6 S=4 P=13 G=8 idxt=3 featptr=.."
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

  // host 数据 (feat/shape/ssi 取 dfa<first>; e_T6 12 调用共享同一 feat 指针)
  char k0[64];
  snprintf(k0, sizeof(k0), "%d", vk[0]);
  size_t nfeat = 0;
  std::vector<__half> feat = read_f16(dir + "/dfa" + k0 + "_feat.f16", &nfeat);
  const int num_feat = (int)(nfeat / C);
  const bool idx_i32 = true;  // e_T6 idxt=3 (kINT32); f32 时转 int (整数值)
  std::vector<int> shape, ssi;
  {
    // shape.bin/ssi.bin 是 int32 裸字节 (idxt=3); 若引擎用 f32 索引则转 int
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
  std::vector<__half*> d_feat(ncalls), d_loc(ncalls), d_log(ncalls), d_out3(ncalls), d_out8(ncalls);
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
    cudaMemcpy(d_loc[i], h_loc[i].data(), h_loc[i].size() * 2, cudaMemcpyHostToDevice);
    cudaMalloc(&d_log[i], h_log[i].size() * 2);
    cudaMemcpy(d_log[i], h_log[i].data(), h_log[i].size() * 2, cudaMemcpyHostToDevice);
    cudaMalloc(&d_out3[i], (size_t)vA[i] * C * 2);
    cudaMalloc(&d_out8[i], (size_t)vA[i] * C * 2);
  }
  // v8 workspace: 取所有调用最大值
  size_t ws_max = 0;
  for (int i = 0; i < ncalls; ++i) {
    const long long nA = (long long)bs * vA[i];
    const long long NCSP = (long long)cams * S * vP[i];
    size_t need = ((size_t)(nA * G * 4) + 255) & ~(size_t)255;
    need += (size_t)(nA * G * NCSP) * 32;
    if (need > ws_max) ws_max = need;
  }
  void* d_ws = nullptr;
  cudaMalloc(&d_ws, ws_max);
  printf("ws_max=%.1fMB\n", ws_max / 1048576.0);

  // 正确性 (先各跑一遍)
  for (int i = 0; i < ncalls; ++i) {
    const int A = vA[i], P = vP[i];
    int rc3 = kv3::dfa_launch_h2<int>(d_out3[i], d_feat[i], d_shape, d_ssi,
                                      d_loc[i], d_log[i], bs, cams, num_feat,
                                      C, S, A, P, G, 0);
    int rc8 = kv8::dfa_launch_v8<int>(d_out8[i], d_feat[i], d_shape, d_ssi,
                                      d_loc[i], d_log[i], bs, cams, num_feat,
                                      C, S, A, P, G, d_ws, ws_max, 0);
    cudaDeviceSynchronize();
    std::vector<__half> o3(A * C), o8(A * C);
    cudaMemcpy(o3.data(), d_out3[i], o3.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(o8.data(), d_out8[i], o8.size() * 2, cudaMemcpyDeviceToHost);
    Stats s3 = compare(o3, h_ref[i]);
    Stats s8 = compare(o8, h_ref[i]);
    Stats s83 = compare(o8, std::vector<float>(o3.begin(), o3.end()));
    printf("dfa%-2d %-3s A=%-4d rc3=%d rc8=%d | v3vseng l2=%.2e maxabs=%.2e"
           " | v8vseng l2=%.2e maxabs=%.2e | v8vsv3 l2=%.2e\n",
           vk[i], (A == 900 ? "det" : "map"), A, rc3, rc8,
           s3.l2rel, s3.maxabs,
           s8.l2rel, s8.maxabs, s83.l2rel);
  }

  // 条目数 sanity (第一次 v8 跑完后的 counts, 取最后一个调用的)
  if (ncalls > 0) {
    const int A = vA[ncalls - 1];
    std::vector<int> hcnt((size_t)bs * A * G);
    // 重跑一次以刷新 counts
    kv8::dfa_launch_v8<int>(d_out8[ncalls - 1], d_feat[ncalls - 1], d_shape,
                            d_ssi, d_loc[ncalls - 1], d_log[ncalls - 1], bs,
                            cams, num_feat, C, S, A, vP[ncalls - 1], G, d_ws,
                            ws_max, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(hcnt.data(), d_ws, hcnt.size() * 4, cudaMemcpyDeviceToHost);
    long long tot = 0;
    int mx = 0;
    for (int v : hcnt) { tot += v; if (v > mx) mx = v; }
    printf("entry sanity (last call A=%d): total=%lld per-g-max=%d NCSP=%d\n",
           A, tot, mx, cams * S * vP[ncalls - 1]);
  }

  // 计时: 每调用分开 (v3 / v8全 / v8 plan / v8 gather) + 条目统计
  float tot3 = 0, tot8 = 0;
  for (int i = 0; i < ncalls; ++i) {
    const int a1[1] = {vA[i]}, p1[1] = {vP[i]};
    float t3 = time_launch(3, 1, &d_out3[i], &d_feat[i], d_shape, d_ssi,
                           &d_loc[i], &d_log[i], bs, cams, num_feat, C, S,
                           a1, p1, G, nullptr, 0, iters, warmup);
    float t8 = time_launch(8, 1, &d_out8[i], &d_feat[i], d_shape, d_ssi,
                           &d_loc[i], &d_log[i], bs, cams, num_feat, C, S,
                           a1, p1, G, d_ws, ws_max, iters, warmup);
    float tp = time_launch(80, 1, &d_out8[i], &d_feat[i], d_shape, d_ssi,
                           &d_loc[i], &d_log[i], bs, cams, num_feat, C, S,
                           a1, p1, G, d_ws, ws_max, iters, warmup);
    float tg = time_launch(81, 1, &d_out8[i], &d_feat[i], d_shape, d_ssi,
                           &d_loc[i], &d_log[i], bs, cams, num_feat, C, S,
                           a1, p1, G, d_ws, ws_max, iters, warmup);
    tot3 += t3;
    tot8 += t8;
    // 条目统计 (最后一次 plan 的 counts)
    const int A = vA[i];
    std::vector<int> hcnt((size_t)bs * A * G);
    cudaMemcpy(hcnt.data(), d_ws, hcnt.size() * 4, cudaMemcpyDeviceToHost);
    long long tot = 0;
    int mx = 0;
    for (int v : hcnt) { tot += v; if (v > mx) mx = v; }
    printf("TIME dfa%-2d %-3s A=%-4d P=%-3d v3=%.4f v8=%.4f "
           "(plan=%.4f gather=%.4f) x%.2f | entries=%lld gmax=%d\n",
           vk[i], (A == 900 ? "det" : "map"), A, vP[i], t3, t8, tp, tg,
           t3 / t8, tot, mx);
  }
  // 全序列 (模拟一帧 12 调用背靠背)
  float seq3 = time_launch(3, ncalls, d_out3.data(), d_feat.data(), d_shape,
                           d_ssi, d_loc.data(), d_log.data(), bs, cams,
                           num_feat, C, S, vA.data(), vP.data(), G, nullptr, 0,
                           iters, warmup);
  float seq8 = time_launch(8, ncalls, d_out8.data(), d_feat.data(), d_shape,
                           d_ssi, d_loc.data(), d_log.data(), bs, cams,
                           num_feat, C, S, vA.data(), vP.data(), G, d_ws,
                           ws_max, iters, warmup);
  printf("TOTAL per-call-sum v3=%.4fms v8=%.4fms x%.2f | sequence v3=%.4fms"
         " v8=%.4fms x%.2f | saved=%.3fms/frame\n",
         tot3, tot8, tot3 / tot8, seq3, seq8, seq3 / seq8, seq3 - seq8);
  printf("DFA_AB_DONE\n");
  return 0;
}
