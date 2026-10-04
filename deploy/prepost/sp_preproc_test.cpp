// sp_preproc_test — M2 数值验证工具: 读 R0 帧 k 的 6 路 NV12 → CUDA 前处理
// → [1,6,3,256,704] f32 落盘, 本地与 numpy/PIL 参考 (work_dirs/preproc_ref)
// 逐像素对比. 不走总线, 隔离数值与链路变量.
// 用法: sp_preproc_test <manifest.jsonl> <data_root> <frame_k> <out.bin>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "preproc.h"

using namespace sp;

// —— 极简 json 数字扫描: 取 key 后面的前 n 个数 (容忍嵌套/字符串) ——
static bool scan_nums(const std::string& s, const char* key, double* out,
                      int n) {
  std::string pat = "\"" + std::string(key) + "\"";
  size_t p = s.find(pat);
  if (p == std::string::npos) return false;
  int got = 0;
  bool in_str = false;
  while (p < s.size() && got < n) {
    char ch = s[p];
    if (ch == '"') in_str = !in_str;
    if (!in_str && ((ch >= '0' && ch <= '9') || ch == '-' || ch == '+' ||
                    ch == '.' || ch == 'e' || ch == 'E')) {
      char* end = nullptr;
      out[got++] = strtod(s.c_str() + p, &end);
      p = end - s.c_str();
      continue;
    }
    ++p;
  }
  return got == n;
}

int main(int argc, char** argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s <manifest.jsonl> <data_root> <frame_k> "
            "<out.bin>\n", argv[0]);
    return 2;
  }
  const char* manifest = argv[1];
  std::string root = argv[2];
  int k = atoi(argv[3]);
  const char* out_path = argv[4];

  FILE* f = fopen(manifest, "rb");
  if (!f) { fprintf(stderr, "open manifest failed\n"); return 1; }
  char line[65536];
  // 首行 header: w,h,resize,crop,mean,std
  if (!fgets(line, sizeof(line), f)) return 1;
  std::string header(line);
  double hw[2], crop[4], mean[3], stdv[3];
  if (!scan_nums(header, "w", hw, 2) ||
      !scan_nums(header, "crop", crop, 4) ||
      !scan_nums(header, "mean", mean, 3) ||
      !scan_nums(header, "std", stdv, 3)) {
    fprintf(stderr, "bad header\n");
    return 1;
  }
  for (int i = 0; i <= k; ++i) {
    if (!fgets(line, sizeof(line), f)) { fprintf(stderr, "no frame %d\n", k); return 1; }
  }
  std::string frameline(line);
  fclose(f);
  // cams: 6 个字符串
  std::string pat = "\"cams\"";
  size_t p = frameline.find(pat);
  if (p == std::string::npos) { fprintf(stderr, "no cams\n"); return 1; }
  std::vector<std::string> cams;
  size_t at = frameline.find('[', p);
  for (int c = 0; c < 6; ++c) {
    size_t q0 = frameline.find('"', at);
    size_t q1 = frameline.find('"', q0 + 1);
    cams.push_back(frameline.substr(q0 + 1, q1 - q0 - 1));
    at = q1 + 1;
  }

  const uint32_t w = (uint32_t)hw[0], h = (uint32_t)hw[1];
  const size_t cam_bytes = (size_t)w * h * 3 / 2;
  PreprocParams pp;
  pp.src_w = w;
  pp.src_h = h;
  pp.resized_w = (uint32_t)crop[2];
  pp.resized_h = (uint32_t)crop[3];
  pp.out_w = pp.resized_w - (uint32_t)crop[0];
  pp.out_h = pp.resized_h - (uint32_t)crop[1];
  pp.crop_x = (uint32_t)crop[0];
  pp.crop_y = (uint32_t)crop[1];
  for (int i = 0; i < 3; ++i) {
    pp.mean[i] = (float)mean[i];
    pp.std[i] = (float)stdv[i];
  }

  char err[256];
  Preproc pre;
  if (!pre.init(pp, err, sizeof(err))) {
    fprintf(stderr, "preproc init: %s\n", err);
    return 1;
  }
  printf("preproc: %ux%u -> resized %ux%u -> %ux%u crop(%u,%u)\n", w, h,
         pp.resized_w, pp.resized_h, pp.out_w, pp.out_h, pp.crop_x,
         pp.crop_y);

  // 6 路 NV12 → 设备 (布局同 slot payload)
  uint8_t* d_slot = nullptr;
  cudaMalloc(&d_slot, cam_bytes * 6);
  uint8_t* h_slot = (uint8_t*)malloc(cam_bytes * 6);
  for (int c = 0; c < 6; ++c) {
    std::string path = root + "/" + cams[c];
    FILE* g = fopen(path.c_str(), "rb");
    if (!g) { fprintf(stderr, "open %s failed\n", path.c_str()); return 1; }
    if (fread(h_slot + c * cam_bytes, 1, cam_bytes, g) != cam_bytes) {
      fprintf(stderr, "short read %s\n", path.c_str());
      return 1;
    }
    fclose(g);
  }
  cudaMemcpy(d_slot, h_slot, cam_bytes * 6, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  float* d_out = nullptr;
  cudaMalloc(&d_out, pre.out_bytes());
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  // warmup + timed
  pre.run(d_slot, cam_bytes, d_out, stream);
  cudaStreamSynchronize(stream);
  cudaEventRecord(e0, stream);
  pre.run(d_slot, cam_bytes, d_out, stream);
  cudaEventRecord(e1, stream);
  cudaEventSynchronize(e1);
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  printf("preproc run: %.3f ms (after warmup)\n", ms);

  std::vector<float> host(pre.out_bytes() / 4);
  cudaMemcpy(host.data(), d_out, pre.out_bytes(), cudaMemcpyDeviceToHost);
  FILE* o = fopen(out_path, "wb");
  fwrite(host.data(), 4, host.size(), o);
  fclose(o);
  printf("PREPROC_TEST_DONE %s\n", out_path);
  return 0;
}
