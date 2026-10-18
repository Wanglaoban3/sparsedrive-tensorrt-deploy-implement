// 通用 TensorRT engine 运行器: 按 manifest 读输入 -> 推理 -> 对照参考输出 + 性能统计
// 用法:
//   run_engine <engine.plan> <plugins.so|-> <inputs_dir> <ref_dir>
//              [--iters N] [--warmup W]
// inputs_dir / ref_dir 各含 manifest.tsv (name\tdtype\tdims\tfile) + *.bin
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>

#define CU(call)                                                         \
  do {                                                                   \
    cudaError_t e_ = (call);                                             \
    if (e_ != cudaSuccess) {                                             \
      printf("CUDA error %s @%d\n", cudaGetErrorString(e_), __LINE__);   \
      return 1;                                                          \
    }                                                                    \
  } while (0)

class Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) printf("[TRT] %s\n", msg);
  }
};

struct Tensor {
  std::string name, dtype, file;
  std::vector<long long> dims;
  std::vector<char> data;
};

static bool load_manifest(const std::string& dir, std::vector<Tensor>& ts) {
  FILE* f = fopen((dir + "/manifest.tsv").c_str(), "r");
  if (!f) return false;
  char line[1024];
  while (fgets(line, sizeof(line), f)) {
    Tensor t;
    char* p = line;
    auto field = [&p]() {
      char* s = p;
      while (*p && *p != '\t' && *p != '\n') ++p;
      std::string out(s, p - s);
      if (*p) ++p;
      while (!out.empty() && (out.back() == '\r' || out.back() == ' '))
        out.pop_back();  // Windows CRLF 免疫
      return out;
    };
    t.name = field();
    t.dtype = field();
    std::string dstr = field();
    t.file = field();
    if (t.name.empty()) continue;
    for (size_t i = 0; i < dstr.size();) {
      size_t c = dstr.find(',', i);
      if (c == std::string::npos) c = dstr.size();
      t.dims.push_back(atoll(dstr.substr(i, c - i).c_str()));
      i = c + 1;
    }
    FILE* b = fopen((dir + "/" + t.file).c_str(), "rb");
    if (!b) { printf("missing %s/%s\n", dir.c_str(), t.file.c_str()); return false; }
    fseek(b, 0, SEEK_END);
    long sz = ftell(b);
    fseek(b, 0, SEEK_SET);
    t.data.resize(sz);
    if (fread(t.data.data(), 1, sz, b) != (size_t)sz) { fclose(b); return false; }
    fclose(b);
    ts.push_back(std::move(t));
  }
  fclose(f);
  return true;
}

static size_t dtype_size(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kBOOL: return 1;
    default: return 0;
  }
}

static size_t dims_volume(const nvinfer1::Dims& d) {
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) v *= (size_t)d.d[i];
  return v;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    printf("usage: %s <engine.plan> <plugins.so|-> <inputs_dir> <ref_dir>"
           " [--iters N] [--warmup W]\n", argv[0]);
    return 2;
  }
  const char* enginePath = argv[1];
  const char* pluginSo = strcmp(argv[2], "-") ? argv[2] : nullptr;
  std::string inDir = argv[3], refDir = argv[4];
  int iters = 0, warmup = 0;
  const char* dumpDir = nullptr;
  for (int i = 5; i < argc; ++i) {
    if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dumpDir = argv[++i];
  }

  if (pluginSo) {
    void* lib = dlopen(pluginSo, RTLD_NOW);
    if (!lib) { printf("dlopen(%s): %s\n", pluginSo, dlerror()); return 1; }
    printf("plugin loaded: %s\n", pluginSo);
  }

  Logger logger;
  auto rt = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  if (!rt) return 1;

  FILE* f = fopen(enginePath, "rb");
  if (!f) { printf("cannot open engine %s\n", enginePath); return 1; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<char> blob(sz);
  if (fread(blob.data(), 1, sz, f) != (size_t)sz) { fclose(f); return 1; }
  fclose(f);
  auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
      rt->deserializeCudaEngine(blob.data(), sz));
  if (!engine) { printf("deserialize engine FAILED\n"); return 1; }
  auto ctx = std::unique_ptr<nvinfer1::IExecutionContext>(
      engine->createExecutionContext());
  if (!ctx) return 1;
  printf("engine loaded: %ld bytes, %d IO tensors\n", sz,
         engine->getNbIOTensors());

  // 读参考/输入文件
  std::vector<Tensor> ins, refs;
  if (!load_manifest(inDir, ins)) { printf("bad inputs dir\n"); return 1; }
  bool haveRef = load_manifest(refDir, refs);
  if (iters > 0 && !haveRef) { /* 只测性能可以没有 ref */ }

  cudaStream_t stream;
  CU(cudaStreamCreate(&stream));

  const int nb = engine->getNbIOTensors();
  std::vector<void*> devBuf(nb, nullptr);
  struct OutInfo { std::string name; nvinfer1::Dims dims; nvinfer1::DataType dt; std::vector<char> host; };
  std::vector<OutInfo> outs;

  for (int i = 0; i < nb; ++i) {
    const char* name = engine->getIOTensorName(i);
    const nvinfer1::Dims dims = engine->getTensorShape(name);
    const nvinfer1::DataType dt = engine->getTensorDataType(name);
    const nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name);
    printf("  IO[%d] %-24s %s dims=[", i, name,
           mode == nvinfer1::TensorIOMode::kINPUT ? "IN " : "OUT");
    for (int d = 0; d < dims.nbDims; ++d) printf("%d%s", dims.d[d], d + 1 < dims.nbDims ? "," : "");
    printf("] %s\n", (int)dt == 0 ? "f32" : (int)dt == 1 ? "f16" : (int)dt == 2 ? "i8" : "i32");

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      Tensor* t = nullptr;
      for (auto& x : ins)
        if (x.name == name) { t = &x; break; }
      if (!t) { printf("input %s not in manifest\n", name); return 1; }
      CU(cudaMalloc(&devBuf[i], t->data.size()));
      CU(cudaMemcpyAsync(devBuf[i], t->data.data(), t->data.size(),
                         cudaMemcpyHostToDevice, stream));
    } else {
      size_t bytes = dims_volume(dims) * dtype_size(dt);
      CU(cudaMalloc(&devBuf[i], bytes));
      OutInfo oi; oi.name = name; oi.dims = dims; oi.dt = dt;
      oi.host.resize(bytes);
      outs.push_back(std::move(oi));
    }
    ctx->setTensorAddress(name, devBuf[i]);
  }
  CU(cudaStreamSynchronize(stream));

  // ---- 正确性: 一次推理 + 对比 ----
  if (!ctx->enqueueV3(stream)) { printf("enqueueV3 FAILED\n"); return 1; }
  CU(cudaStreamSynchronize(stream));

  // 逐输出 D2H (通过 tensor 名找 buffer index)
  for (int i = 0; i < nb; ++i) {
    const char* name = engine->getIOTensorName(i);
    if (engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kOUTPUT) continue;
    OutInfo* oi = nullptr;
    for (auto& o : outs)
      if (o.name == name) { oi = &o; break; }
    if (!oi) continue;
    CU(cudaMemcpy(oi->host.data(), devBuf[i], oi->host.size(), cudaMemcpyDeviceToHost));
  }
  CU(cudaStreamSynchronize(stream));

  // 可选: 落盘 engine 原始输出
  if (dumpDir) {
    std::string manifest;
    char mp[512];
    snprintf(mp, sizeof(mp), "mkdir -p %s", dumpDir);
    if (system(mp) != 0) { printf("mkdir %s failed\n", dumpDir); return 1; }
    for (auto& oi : outs) {
      std::string fn = oi.name + ".bin";
      FILE* g = fopen((std::string(dumpDir) + "/" + fn).c_str(), "wb");
      if (!g) { printf("cannot write %s\n", fn.c_str()); return 1; }
      fwrite(oi.host.data(), 1, oi.host.size(), g);
      fclose(g);
      const char* dts = oi.dt == nvinfer1::DataType::kHALF ? "f16"
                        : oi.dt == nvinfer1::DataType::kINT32 ? "i32" : "f32";
      std::string ds;
      for (int d = 0; d < oi.dims.nbDims; ++d)
        ds += (d ? "," : "") + std::to_string(oi.dims.d[d]);
      manifest += oi.name + "\t" + dts + "\t" + ds + "\t" + fn + "\n";
    }
    FILE* g = fopen((std::string(dumpDir) + "/manifest.tsv").c_str(), "wb");
    fwrite(manifest.data(), 1, manifest.size(), g);
    fclose(g);
    printf("outputs dumped to %s (%d tensors)\n", dumpDir, (int)outs.size());
  }

  if (haveRef) {
    printf("\n== 端到端对比 (engine vs torch 参考) ==\n");
    printf("%-24s %12s %12s %10s  %s\n", "output", "rel_l2", "max_abs", "cosine", "verdict");
    int nPass = 0, nTot = 0;
    for (auto& oi : outs) {
      Tensor* r = nullptr;
      for (auto& x : refs)
        if (x.name == oi.name) { r = &x; break; }
      if (!r) { printf("%-24s (无参考, 跳过)\n", oi.name.c_str()); continue; }
      const size_t n = r->data.size() / 4;
      // 参考恒为 f32; engine 输出可能是 f16/i32
      std::vector<float> ov(n);
      if (oi.dt == nvinfer1::DataType::kHALF) {
        const unsigned short* h = (const unsigned short*)oi.host.data();
        for (size_t i = 0; i < n; ++i) { __half hh; memcpy(&hh, &h[i], 2); ov[i] = __half2float(hh); }
      } else {
        memcpy(ov.data(), oi.host.data(), n * 4);
      }
      const float* rp = (const float*)r->data.data();
      if (oi.dt == nvinfer1::DataType::kINT32) {
        // 整数输出: 与参考逐 int32 一致率 (参考同为 int32 原始字节)
        if (r->dtype != "i32") {
          printf("%-24s (参考非 i32, 跳过)\n", oi.name.c_str());
          continue;
        }
        long same = 0;
        const int* ip = (const int*)oi.host.data();
        const int* ir = (const int*)r->data.data();
        for (size_t i = 0; i < n; ++i) same += (ip[i] == ir[i]);
        printf("%-24s %12s %12s %10s  match=%ld/%zu (%.2f%%)\n",
               oi.name.c_str(), "-", "-", "-",
               same, n, 100.0 * same / n);
        continue;
      }
      double sqd = 0, sqr = 0, mx = 0, dot = 0, no = 0, nr = 0;
      for (size_t i = 0; i < n; ++i) {
        double o = ov[i], rr = rp[i];
        sqd += (o - rr) * (o - rr); sqr += rr * rr;
        if (fabs(o - rr) > mx) mx = fabs(o - rr);
        dot += o * rr; no += o * o; nr += rr * rr;
      }
      double rel = sqrt(sqd) / (sqrt(sqr) + 1e-12);
      double cos = dot / (sqrt(no) * sqrt(nr) + 1e-12);
      bool ok = rel < 0.05 && cos > 0.99;
      nTot++; nPass += ok;
      printf("%-24s %12.3e %12.3e %10.6f  %s\n", oi.name.c_str(), rel, mx, cos,
             ok ? "PASS" : "FAIL");
    }
    printf("==> %d/%d outputs PASS\n", nPass, nTot);
  }

  // ---- 性能 ----
  if (iters > 0) {
    printf("\n== 性能 (%d warmup + %d iters) ==\n", warmup, iters);
    cudaEvent_t e0, e1;
    CU(cudaEventCreate(&e0));
    CU(cudaEventCreate(&e1));
    for (int i = 0; i < warmup; ++i) ctx->enqueueV3(stream);
    CU(cudaStreamSynchronize(stream));
    std::vector<float> ms(iters);
    for (int i = 0; i < iters; ++i) {
      CU(cudaEventRecord(e0, stream));
      ctx->enqueueV3(stream);
      CU(cudaEventRecord(e1, stream));
      CU(cudaEventSynchronize(e1));
      CU(cudaEventElapsedTime(&ms[i], e0, e1));
    }
    // 统计
    double sum = 0; float mn = 1e30f, mxv = 0;
    for (float v : ms) { sum += v; if (v < mn) mn = v; if (v > mxv) mxv = v; }
    double mean = sum / iters;
    double var = 0;
    for (float v : ms) var += (v - mean) * (v - mean);
    var /= iters;
    std::vector<float> sorted(ms);
    std::sort(sorted.begin(), sorted.end());
    printf("latency ms: mean=%.3f std=%.3f min=%.3f p50=%.3f p99=%.3f max=%.3f  fps=%.1f\n",
           mean, sqrt(var), mn,
           sorted[iters / 2], sorted[(int)(iters * 0.99) % iters], mxv,
           1000.0 / mean);
  }
  printf("RUN_ENGINE_DONE\n");
  return 0;
}
