// sp_modelnode — M3/M4: 总线消费进程 = 前处理 + 引擎推理 + 时序状态闭环.
// 链路: acquire(shm 槽, fence 语义持引用) → P2 投影矩阵/host t_matrix(double)
// → M2 前处理 kernel(读 registered shm 设备指针) → 引擎 enqueue →
// next_* D2D 回写 prev_* → dump 19 路输出(与链路二 outv8 同名同布局).
// 场景边界口径与链路二一致: 只重置 t_matrix(identity)+dt(0.5), 实例状态
// 跨场景保留; 首帧零状态(f32=0/id=-1/count=0); time_interval=帧间 ts 差.
//
// M5 拆分链(--hd): 主引擎位换成 e_hd, 前面加 e_bb2 —— bb2 输出张量按名字
// 全等 D2D 直供 hd 输入(run_engine2 同款零拷贝, col_feats 按 parity 固定
// 双份); graph 按 bb2/hd×2 parity 捕两张(事件夹中间, 分段计时); 感知状态
// 反馈不变(全来自 hd, bb2 无状态).
// M6a 二级(--mp): e_mp 同进程第三级, 9 个感知输入按名字映射直绑 hd 输出
// 缓冲(零拷贝), t_matrix 直绑 d_tmat[p]; 9 张历史状态单缓冲外旋 D2D,
// 场景首帧全清零(与感知"只重置 tmat/dt"两套口径并存); mp 不捕 graph
// (M7 再议), 逐帧 setTensorAddress(parity 别名); 输出 dump 到 outm_XX.
//
// M4 双缓冲流水线(默认): 3 stream, 2 帧在飞 ——
//   pre_stream:  H2D 参数 + 前处理(帧 k)        ┐
//   eng_stream:  等前处理事件 → 推理 → 反馈 D2D  ├ 帧 k-1 推理与帧 k 前处理重叠
//   post_stream: 等推理事件 → 19 路 D2H(pinned) ┘
//   CPU 只在"帧 k-2 后处理完成"处同步; 槽位引用挂 ev_preB 事件由释放队列
//   异步归还(fence 语义不变, CPU 不等前处理); 输出/输入全按奇偶双份,
//   引擎写 out[p] 前先等 post 上次用完(parity p 的 D2H 完成).
//   --serial 退回 M3 串行模式; --graph 把"推理+反馈"按 parity 各捕获成
//   CUDA graph 反复 launch; --loop 允许帧数超过清单(清单序回绕, 稳定性压测).
//
// 用法: sp_modelnode <ring> <engine|bb2.engine> <plugin.so> <manifest.jsonl>
//                    <out_dir> [--hd hd.engine] [--mp mp.engine]
//                    [--frames N] [--warmup W] [--dump-img N]
//                    [--img-from <dir|file|tpl>] [--serial] [--graph]
//                    [--loop] [--no-dump] [--det-thr F] [--map-thr F]
//                    [--det-topk N] [--mailbox NAME]
// Q1-Q3: 每帧后处理完成后 det/map decode (口径=离线评测脚本, postproc.h),
// 结果发布到单槽 latest-wins 信箱 (sp_result.h, 默认 sp_result_<ring>),
// 并逐帧追加 JSON 旁路 out_dir/result.jsonl (--no-dump 时跳过 JSON 落盘).
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <signal.h>
#include <sys/stat.h>
#include <time.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <NvInfer.h>

#include "postproc.h"
#include "preproc.h"
#include "sp_bus.h"
#include "sp_dmapool.h"
#include "sp_result.h"
#include "sp_watch.h"

using namespace sp;

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

class Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) printf("[TRT] %s\n", msg);
  }
};

// ---------- 极简 manifest v2 解析 ----------
struct FrameMetaV2 {
  uint64_t ts_ns;
  uint32_t scene;
  std::string cam_path[6];
  double l2i[6][16];
  double l2g[16];
};

struct ManifestV2 {
  PreprocParams pp;
  double resize = 0.0;  // aug resize (P2 用)
  std::vector<FrameMetaV2> frames;
};

static bool scan_nums_after(const std::string& s, size_t from,
                            const char* key, double* out, int n) {
  std::string pat = "\"" + std::string(key) + "\"";
  size_t p = s.find(pat, from);
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

static bool scan_u64_after(const std::string& s, size_t from,
                           const char* key, uint64_t* out) {
  std::string pat = "\"" + std::string(key) + "\"";
  size_t p = s.find(pat, from);
  if (p == std::string::npos) return false;
  p = s.find(':', p + pat.size());
  if (p == std::string::npos) return false;
  ++p;
  while (p < s.size() && (s[p] == ' ')) ++p;
  uint64_t v = 0;
  if (p >= s.size() || s[p] < '0' || s[p] > '9') return false;
  while (p < s.size() && s[p] >= '0' && s[p] <= '9')
    v = v * 10 + (uint64_t)(s[p++] - '0');
  *out = v;
  return true;
}

static bool load_manifest_v2(const char* path, ManifestV2* m) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  char line[65536];
  if (!fgets(line, sizeof(line), f)) { fclose(f); return false; }
  std::string header(line);
  double hw[2], crop[4], mean[3], stdv[3], resize[1];
  if (!scan_nums_after(header, 0, "w", hw, 2) ||
      !scan_nums_after(header, 0, "resize", resize, 1) ||
      !scan_nums_after(header, 0, "crop", crop, 4) ||
      !scan_nums_after(header, 0, "mean", mean, 3) ||
      !scan_nums_after(header, 0, "std", stdv, 3)) {
    fclose(f);
    return false;
  }
  m->pp.src_w = (uint32_t)hw[0];
  m->pp.src_h = (uint32_t)hw[1];
  m->pp.resized_w = (uint32_t)crop[2];
  m->pp.resized_h = (uint32_t)crop[3];
  m->pp.out_w = m->pp.resized_w - (uint32_t)crop[0];
  m->pp.out_h = m->pp.resized_h - (uint32_t)crop[1];
  m->pp.crop_x = (uint32_t)crop[0];
  m->pp.crop_y = (uint32_t)crop[1];
  for (int i = 0; i < 3; ++i) {
    m->pp.mean[i] = (float)mean[i];
    m->pp.std[i] = (float)stdv[i];
  }
  m->resize = resize[0];
  int lineno = 0;
  while (fgets(line, sizeof(line), f)) {
    lineno += 1;
    std::string s(line);
    if (s.size() < 3) continue;
    FrameMetaV2 fm;
    memset(&fm, 0, sizeof(fm));
    if (!scan_u64_after(s, 0, "ts_ns", &fm.ts_ns)) { fclose(f); return false; }
    double sc;
    if (!scan_nums_after(s, 0, "scene", &sc, 1)) { fclose(f); return false; }
    fm.scene = (uint32_t)sc;
    std::string pat = "\"cams\"";
    size_t p = s.find(pat);
    size_t at = s.find('[', p);
    for (int c = 0; c < 6; ++c) {
      size_t q0 = s.find('"', at);
      size_t q1 = s.find('"', q0 + 1);
      fm.cam_path[c] = s.substr(q0 + 1, q1 - q0 - 1);
      at = q1 + 1;
    }
    if (!scan_nums_after(s, 0, "l2i", &fm.l2i[0][0], 96)) {
      fclose(f);
      return false;
    }
    if (!scan_nums_after(s, 0, "l2g", &fm.l2g[0], 16)) {
      fclose(f);
      return false;
    }
    m->frames.push_back(fm);
  }
  fclose(f);
  return true;
}

// ---------- host 4x4 (double, row-major) ----------
static void mat4_mul(const double* a, const double* b, double* o) {
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) {
      double v = 0;
      for (int k = 0; k < 4; ++k) v += a[r * 4 + k] * b[k * 4 + c];
      o[r * 4 + c] = v;
    }
}

// 行主序 4x4 求逆 —— 刚体变换口径 ([R t; 0 1], l2g 由四元数构造, R 正交):
// inv = [R^T  -R^T t; 0 1], 平移在【列 3】(扁平下标 3/7/11), 行 3 恒 [0 0 0 1].
// 本项目两度踩坑, 都在这 16 个下标上: 先是抄 MESA gluInvertMatrix(列主序约定)
// 且抄错 inv[9]/inv[13]/inv[14], l2g 每帧 Z 平移差 -1.55m, det mAP -10pt;
// 后是解析式把 -R^T t 误写进行 3 (下标 12/13/14), tmat 底行爆炸到 1.5e6.
// numpy 二维下标验证过≠C 一维扁平下标写对, 必须按扁平下标复算. 2026-10-04 定案.
static bool mat4_inv(const double* m, double* o) {
  double tx = m[3], ty = m[7], tz = m[11];
  o[0] = m[0];
  o[1] = m[4];
  o[2] = m[8];
  o[3] = -(m[0] * tx + m[4] * ty + m[8] * tz);
  o[4] = m[1];
  o[5] = m[5];
  o[6] = m[9];
  o[7] = -(m[1] * tx + m[5] * ty + m[9] * tz);
  o[8] = m[2];
  o[9] = m[6];
  o[10] = m[10];
  o[11] = -(m[2] * tx + m[6] * ty + m[10] * tz);
  o[12] = 0;
  o[13] = 0;
  o[14] = 0;
  o[15] = 1.0;
  return true;
}

// P2: ida(参数化 scale/crop) @ lidar2img, double → f32
static void make_projection(const ManifestV2& man, int frame_idx,
                            float* out /*6*16 f32*/) {
  double ext[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  ext[0] = ext[5] = man.resize;  // aug resize (来自 manifest header)
  ext[2] = -(double)man.pp.crop_x;
  ext[6] = -(double)man.pp.crop_y;
  const FrameMetaV2& fm = man.frames[frame_idx];
  for (int c = 0; c < 6; ++c) {
    double tmp[16];
    mat4_mul(ext, &fm.l2i[c][0], tmp);
    for (int i = 0; i < 16; ++i) out[c * 16 + i] = (float)tmp[i];
  }
}

static double now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static uint64_t fnv1a(const std::string& s) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

static double pct(std::vector<double>& v, double q) {
  if (v.empty()) return 0;
  size_t i = (size_t)(q * (v.size() - 1) + 0.5);
  std::vector<double> s = v;
  std::sort(s.begin(), s.end());
  return s[std::min(i, s.size() - 1)];
}

enum { kPar = 2 };  // 流水线深度 = 缓冲奇偶份数

struct Bind {
  std::string name;
  void* dev[kPar] = {nullptr, nullptr};
  size_t bytes = 0;
  bool is_input = false;
  bool alias = false;  // M5: bb2 输出别名到 hd 输入缓冲 (不分配不落盘)
  nvinfer1::DataType dt;
  char* host[kPar] = {nullptr, nullptr};  // pinned, 输出用
};

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc < 6) {
    fprintf(stderr,
            "usage: %s <ring> <engine|bb2.engine> <plugin> <manifest.jsonl> "
            "<out_dir> [--hd hd.engine] [--mp mp.engine] [--frames N] "
            "[--warmup W] [--dump-img N] [--img-from P] [--serial] [--graph] "
            "[--dual] [--loop] [--no-dump] [--det-thr F] [--map-thr F] "
            "[--det-topk N] [--mailbox NAME] [--cmd N] [--dma]\n",
            argv[0]);
    return 2;
  }
  const char* ring = argv[1];
  const char* engine_path = argv[2];
  const char* plugin_so = argv[3];
  const char* manifest_path = argv[4];
  std::string out_dir = argv[5];
  const char* hd_path = nullptr;   // M5: 拆分链 hd 引擎 (给定则 argv[2]=bb2)
  const char* mp_path = nullptr;   // M6a: e_mp 二级引擎
  long n_frames = -1;
  int warmup = 0;
  int dump_img_n = 0;   // 前 N 帧落盘实际喂给引擎的 img (对质用)
  const char* img_from = nullptr;  // 隔离模式: 目录根/模板/单文件
  bool serial = false, use_graph = false, loop = false, no_dump = false;
  bool dual = false;  // M7a: bb2 独立流与 hd+mp 重叠
  bool use_dma = false;  // M8: 采集源走设备池 (fd 导入), 绕开 mapped-shm 读
  float det_thr = 0.0f, map_thr = 0.0f;  // Q2: 默认=离线评测口径(全保留)
  int det_topk = 300;
  int plan_cmd = 2;  // M6b: final_plan 便捷解码的 cmd (2=直行; 真 cmd 车辆给)
  std::string mailbox_name;
  for (int i = 6; i < argc; ++i) {
    if (!strcmp(argv[i], "--frames") && i + 1 < argc) n_frames = atol(argv[++i]);
    else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--hd") && i + 1 < argc) hd_path = argv[++i];
    else if (!strcmp(argv[i], "--mp") && i + 1 < argc) mp_path = argv[++i];
    else if (!strcmp(argv[i], "--dump-img") && i + 1 < argc)
      dump_img_n = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--img-from") && i + 1 < argc)
      img_from = argv[++i];
    else if (!strcmp(argv[i], "--serial")) serial = true;
    else if (!strcmp(argv[i], "--dual")) dual = true;
    else if (!strcmp(argv[i], "--dma")) use_dma = true;
    else if (!strcmp(argv[i], "--graph")) use_graph = true;
    else if (!strcmp(argv[i], "--loop")) loop = true;
    else if (!strcmp(argv[i], "--no-dump")) no_dump = true;
    else if (!strcmp(argv[i], "--det-thr") && i + 1 < argc)
      det_thr = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--map-thr") && i + 1 < argc)
      map_thr = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--det-topk") && i + 1 < argc)
      det_topk = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--cmd") && i + 1 < argc)
      plan_cmd = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--mailbox") && i + 1 < argc)
      mailbox_name = argv[++i];
  }
  if (mailbox_name.empty()) mailbox_name = std::string("sp_result_") + ring;
  if (det_topk > res::kDetCap) det_topk = res::kDetCap;
  if (dual && serial) {
    printf("modelnode: --dual 与 --serial 互斥, 退回串行\n");
    dual = false;
  }
  if (dual && !hd_path) {
    printf("modelnode: --dual 需要 --hd 拆分链, 忽略\n");
    dual = false;
  }
  signal(SIGINT, on_sig);
  signal(SIGTERM, on_sig);

  ManifestV2 man;
  if (!load_manifest_v2(manifest_path, &man)) {
    fatal_exit(10, "init", "bad manifest %s", manifest_path);
  }
  printf("modelnode: manifest %zu frames, %ux%u -> %ux%u\n",
         man.frames.size(), man.pp.src_w, man.pp.src_h, man.pp.out_w,
         man.pp.out_h);
  const long nman = (long)man.frames.size();
  if (!loop && (n_frames < 0 || n_frames > nman)) n_frames = nman;
  if (n_frames < 0) n_frames = nman;
  printf("modelnode: mode=%s%s%s run=%ld frames\n", serial ? "serial" : "pipe",
         use_graph ? "+graph" : "", dual ? "+dual" : "", n_frames);

  // ---- bus attach (consumer, CUDA Mapped 注册; 等发布端建环) ----
  char err[256];
  Bus* bus = nullptr;
  for (int t = 0; t < 240 && !bus; ++t) {
    bus = Bus::open(ring, man.pp.src_w, man.pp.src_h, false, false, err,
                    sizeof(err));
    if (bus) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  if (!bus) {
    fatal_exit(10, "init", "bus open: %s", err);
  }
  const RingMeta* rm = bus->meta();
  cudaError_t ce = cudaHostRegister(bus->base(), bus->mapped_bytes(),
                                    cudaHostRegisterMapped);
  if (ce != cudaSuccess) {
    fatal_exit(12, "init", "register: %s", cudaGetErrorString(ce));
  }
  uint8_t* dev_base = nullptr;
  ce = cudaHostGetDevicePointer((void**)&dev_base, bus->base(), 0);
  if (ce != cudaSuccess) {
    fatal_exit(12, "init", "getdevptr: %s", cudaGetErrorString(ce));
  }
  printf("modelnode: registered %zu bytes, dev=%p\n", bus->mapped_bytes(),
         (void*)dev_base);

  // ---- M8: --dma 设备池导入 (fd 旁路, 消费端 cudaExternalMemory) ----
  const uint8_t* dma_dev[kRingDepth] = {};
  if (use_dma || rm->dma_present) {
    if (!rm->dma_present || !rm->dma_slot_bytes) {
      fatal_exit(10, "init", "--dma 但发布端未注册设备池");
    }
    if (!use_dma) {
      fatal_exit(10, "init", "发布端为 --dma 池 (shm payload 空), 节点必须加 --dma");
    }
    DmaPoolInfo want = {};
    want.magic = 0x53445031;  // "SDP1" (sp_dmapool.h)
    want.n_slots = rm->dma_n_slots;
    want.slot_bytes = rm->dma_slot_bytes;
    want.width = rm->width;
    want.height = rm->height;
    DmaPoolSub* sub = new DmaPoolSub();
    if (!sub->attach(ring, want, err, sizeof(err))) {
      fatal_exit(10, "init", "dma attach: %s", err);
    }
    if (want.n_slots > kRingDepth) {
      fatal_exit(10, "init", "dma slots %u > ring depth", want.n_slots);
    }
    for (uint32_t i = 0; i < want.n_slots; ++i)
      dma_dev[i] = sub->dev(i);
  }

  // ---- preproc ----
  Preproc pre;
  if (!pre.init(man.pp, err, sizeof(err))) {
    fatal_exit(12, "init", "preproc: %s", err);
  }
  cudaStream_t pre_stream, eng_stream, post_stream;
  cudaStream_t eng_streamB = nullptr;  // M7a: bb2 专用流 (--dual)
  cudaStreamCreate(&pre_stream);
  cudaStreamCreate(&eng_stream);
  cudaStreamCreate(&post_stream);
  const char* os_flag = getenv("SP_ONE_STREAM");
  if (os_flag && atoi(os_flag)) {
    eng_stream = pre_stream;  // 排查用: 前处理与引擎同流
    printf("modelnode: ONE-STREAM mode\n");
    if (dual) {
      printf("modelnode: SP_ONE_STREAM 与 --dual 冲突, 退回单流\n");
      dual = false;
    }
  }
  if (dual) cudaStreamCreate(&eng_streamB);

  // 双缓冲: img/参数按奇偶各 kPar 份; H2D 源用 pinned 暂存
  // (异步 H2D 不能读栈上临时值 —— 提交后 CPU 立即返回, 栈会失效)
  float* d_img[kPar];
  float* d_proj[kPar];
  float* d_tmat[kPar];
  float* d_dt[kPar];
  float* h_proj[kPar];
  float* h_tmat[kPar];
  float* h_dt[kPar];
  for (int p = 0; p < kPar; ++p) {
    cudaMalloc(&d_img[p], pre.out_bytes());
    cudaMalloc(&d_proj[p], 6 * 16 * 4);
    cudaMalloc(&d_tmat[p], 16 * 4);
    cudaMalloc(&d_dt[p], 4);
    cudaMallocHost(&h_proj[p], 6 * 16 * 4);
    cudaMallocHost(&h_tmat[p], 16 * 4);
    cudaMallocHost(&h_dt[p], 4);
  }

  // ---- engine: 双 context 按奇偶固定绑定 (无逐帧重绑) ----
  // M5: --hd 时 argv[2]=e_bb2, 主引擎位换成 e_hd; bb2 输出按名字全等
  // D2D 直供 hd 输入 (run_engine2 同款), 边界缓冲按 parity 双份固定.
  const bool use_hd = hd_path != nullptr;
  const char* main_path = use_hd ? hd_path : engine_path;
  void* plug = dlopen(plugin_so, RTLD_NOW);
  if (!plug) {
    fatal_exit(11, "init", "dlopen(%s): %s", plugin_so, dlerror());
  }
  printf("modelnode: plugin %s\n", plugin_so);
  Logger logger;
  auto rt = std::unique_ptr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(logger));
  auto load_eng = [&](const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) fatal_exit(11, "init", "open engine %s failed", path);
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> blob(sz);
    if (fread(blob.data(), 1, sz, f) != (size_t)sz)
      fatal_exit(11, "init", "short read engine %s", path);
    fclose(f);
    return std::unique_ptr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(blob.data(), sz));
  };

  // -- bb2 侧 (拆分模式): img→col_feats, 输出别名到 hd 输入缓冲 --
  std::unique_ptr<nvinfer1::ICudaEngine> engineBB;
  nvinfer1::IExecutionContext* ctxBB[kPar] = {nullptr, nullptr};
  std::vector<Bind> bindsBB;
  std::string bb_out_name;
  if (use_hd) {
    engineBB = load_eng(engine_path);
    if (!engineBB) fatal_exit(11, "init", "bb2 deserialize FAILED");
    for (int p = 0; p < kPar; ++p) ctxBB[p] = engineBB->createExecutionContext();
    if (!ctxBB[0] || !ctxBB[1]) {
      fatal_exit(11, "init", "bb2 createExecutionContext FAILED");
    }
    int nbB = engineBB->getNbIOTensors();
    bindsBB.resize(nbB);
    int n_out = 0;
    for (int i = 0; i < nbB; ++i) {
      const char* name = engineBB->getIOTensorName(i);
      Bind& b = bindsBB[i];
      b.name = name;
      b.dt = engineBB->getTensorDataType(name);
      b.is_input = engineBB->getTensorIOMode(name) ==
                   nvinfer1::TensorIOMode::kINPUT;
      size_t vol = 1;
      auto dims = engineBB->getTensorShape(name);
      for (int d = 0; d < dims.nbDims; ++d) vol *= (size_t)dims.d[d];
      size_t es = b.dt == nvinfer1::DataType::kHALF
                      ? 2
                      : (b.dt == nvinfer1::DataType::kINT8 ? 1 : 4);
      b.bytes = vol * es;
      if (b.is_input) {
        // bb2 输入只应有 img (projection_mat 若在 bb2 侧也共享同一缓冲)
        if (!strcmp(name, "img")) {
          for (int p = 0; p < kPar; ++p) b.dev[p] = d_img[p];
        } else if (!strcmp(name, "projection_mat")) {
          for (int p = 0; p < kPar; ++p) b.dev[p] = d_proj[p];
        } else {
          fatal_exit(11, "init", "bb2 unexpected input %s", name);
        }
      } else {
        bb_out_name = name;
        n_out++;
        b.alias = true;  // 缓冲在 hd 侧分配后回填
      }
    }
    if (n_out != 1) {
      fatal_exit(11, "init", "bb2 outputs=%d (期望 1)", n_out);
    }
    printf("modelnode: bb2 %d tensors, boundary out %s\n", nbB,
           bb_out_name.c_str());
  }

  auto engine = load_eng(main_path);
  if (!engine) fatal_exit(11, "init", "deserialize FAILED %s", main_path);
  nvinfer1::IExecutionContext* ctx[kPar] = {nullptr, nullptr};
  for (int p = 0; p < kPar; ++p) ctx[p] = engine->createExecutionContext();
  if (!ctx[0] || !ctx[1]) {
    fatal_exit(11, "init", "createExecutionContext FAILED");
  }

  const int nb = engine->getNbIOTensors();
  std::vector<Bind> binds(nb);
  struct StateBind { Bind* in; Bind* out; };
  std::vector<StateBind> states;  // next_* → prev_* 反馈对
  for (int i = 0; i < nb; ++i) {
    const char* name = engine->getIOTensorName(i);
    Bind& b = binds[i];
    b.name = name;
    b.dt = engine->getTensorDataType(name);
    b.is_input = engine->getTensorIOMode(name) ==
                 nvinfer1::TensorIOMode::kINPUT;
    size_t vol = 1;
    auto dims = engine->getTensorShape(name);
    for (int d = 0; d < dims.nbDims; ++d) vol *= (size_t)dims.d[d];
    size_t es = b.dt == nvinfer1::DataType::kHALF
                    ? 2
                    : (b.dt == nvinfer1::DataType::kINT8 ? 1 : 4);
    b.bytes = vol * es;
    if (b.is_input) {
      if (!strcmp(name, "img")) {
        for (int p = 0; p < kPar; ++p) b.dev[p] = d_img[p];
      } else if (!strcmp(name, "projection_mat")) {
        if (b.bytes != 6 * 16 * 4) fatal_exit(11, "init", "proj size %zu", b.bytes);
        for (int p = 0; p < kPar; ++p) b.dev[p] = d_proj[p];
      } else if (!strcmp(name, "instance_t_matrix")) {
        if (b.bytes != 16 * 4) fatal_exit(11, "init", "tmat size %zu", b.bytes);
        for (int p = 0; p < kPar; ++p) b.dev[p] = d_tmat[p];
      } else if (!strcmp(name, "time_interval")) {
        if (b.bytes != 4) fatal_exit(11, "init", "dt size %zu", b.bytes);
        for (int p = 0; p < kPar; ++p) b.dev[p] = d_dt[p];
      } else if (use_hd && b.name == bb_out_name) {
        // M5 边界张量: 按 parity 固定双份 (graph 地址要求), bb2 输出直写
        for (int p = 0; p < kPar; ++p) {
          ce = cudaMalloc(&b.dev[p], b.bytes);
          if (ce != cudaSuccess) {
            fatal_exit(12, "init", "boundary malloc: %s",
                       cudaGetErrorString(ce));
          }
        }
        printf("modelnode: boundary %s (%zu B x%d, D2D 直供)\n", name,
               b.bytes, kPar);
      } else {
        // 时序状态: 单份 (推理全在 eng_stream 上定序, 反馈后下一帧才读)
        cudaMalloc(&b.dev[0], b.bytes);
        b.dev[1] = b.dev[0];
        states.push_back({&b, nullptr});
      }
    } else {
      for (int p = 0; p < kPar; ++p) {
        cudaMalloc(&b.dev[p], b.bytes);
        cudaMallocHost(&b.host[p], b.bytes);
      }
    }
  }
  if (use_hd) {
    // 边界回填: bb2 输出 dev = hd 输入 dev; bb2 ctx 绑定
    Bind* bb_out = nullptr;
    for (auto& b : bindsBB)
      if (!b.is_input) bb_out = &b;
    Bind* bnd = nullptr;
    for (auto& b : binds)
      if (b.is_input && b.name == bb_out_name) bnd = &b;
    if (!bb_out || !bnd) fatal_exit(11, "init", "boundary wire fail");
    if (bb_out->bytes != bnd->bytes || bb_out->dt != bnd->dt) {
      fatal_exit(11, "init", "boundary mismatch bb %zuB/%d vs hd %zuB/%d",
                 bb_out->bytes, (int)bb_out->dt, bnd->bytes, (int)bnd->dt);
    }
    for (int p = 0; p < kPar; ++p) bb_out->dev[p] = bnd->dev[p];
    for (int p = 0; p < kPar; ++p)
      for (auto& b : bindsBB)
        if (!ctxBB[p]->setTensorAddress(b.name.c_str(), b.dev[p])) {
          fatal_exit(11, "init", "bb2 setTensorAddress %s ctx%d failed",
                     b.name.c_str(), p);
        }
  }
  for (int p = 0; p < kPar; ++p)
    for (auto& b : binds)
      if (!ctx[p]->setTensorAddress(b.name.c_str(), b.dev[p])) {
        fatal_exit(11, "init", "setTensorAddress %s ctx%d failed",
                   b.name.c_str(), p);
      }
  printf("modelnode: %d tensors bound x%d ctx (%s)\n", nb, kPar,
         use_hd ? "split bb2+hd" : "single");
  // 反馈对: prev_x 输入 ↔ next_x 输出; 特例 prev_det_id ↔ next_det_instance_id
  for (auto& st : states) {
    std::string next = "next_" + st.in->name.substr(5);
    if (st.in->name == "prev_det_id") next = "next_det_instance_id";
    for (auto& b : binds)
      if (!b.is_input && b.name == next) st.out = &b;
    if (!st.out) fatal_exit(11, "init", "no output %s", next.c_str());
  }
  // 状态零填 (首帧): f32 → 0; i32 → -1 (id) / 0 (count)
  for (auto& st : states) {
    bool is_id = st.in->name == "prev_det_id";  // 链路二零状态: id 全 -1
    bool is_cnt = st.in->name == "prev_id_count";
    std::vector<char> z(st.in->bytes);
    if (is_id) {
      int* p = (int*)z.data();
      for (size_t i = 0; i < st.in->bytes / 4; ++i) p[i] = -1;
    } else if (!is_cnt) {
      memset(z.data(), 0, z.size());
    }
    cudaMemcpy(st.in->dev[0], z.data(), st.in->bytes, cudaMemcpyHostToDevice);
    if (is_id)
      printf("modelnode: zero state %s -> all -1 (%zu B)\n",
             st.in->name.c_str(), st.in->bytes);
  }

  std::vector<Bind*> outs;
  for (auto& b : binds)
    if (!b.is_input) outs.push_back(&b);

  // ---- M6a: e_mp 二级引擎 (--mp) ----
  // 9 个感知输入按名字映射直绑感知输出缓冲(零拷贝, parity 别名), t_matrix
  // 直绑 d_tmat[p]; 9 张历史状态单缓冲外旋, 场景首帧全清零(模板 D2D);
  // mp 不捕 graph(M7 再议), 逐帧 setTensorAddress; 输出落 outm_XX.
  std::unique_ptr<nvinfer1::ICudaEngine> engineM;
  nvinfer1::IExecutionContext* ctxM = nullptr;
  std::vector<Bind> bindsM;
  std::vector<StateBind> mstates;
  std::vector<Bind*> outs_mp;
  std::vector<void*> d_mrst;  // 场景复位模板(设备), 与 mstates 一一对应
  const bool use_mp = mp_path != nullptr;
  if (use_mp) {
    // mp 输入名 → 感知输出名 (全等匹配, 不做子串)
    static const std::pair<const char*, const char*> mp_alias[] = {
        {"det_cls", "det_cls"},
        {"det_bbox", "det_bbox"},
        {"det_feat", "det_instance_feature"},
        {"det_anchor_embed", "det_anchor_embed"},
        {"det_instance_id", "det_instance_id"},
        {"map_cls", "map_cls"},
        {"map_feat", "map_instance_feature"},
        {"map_anchor_embed", "map_anchor_embed"},
        {"ego_feature_map", "ego_feature_map"},
    };
    auto find_bind = [&](const std::string& nm) -> Bind* {
      for (auto& b : binds)
        if (b.name == nm) return &b;
      return nullptr;
    };
    engineM = load_eng(mp_path);
    if (!engineM) fatal_exit(11, "init", "mp deserialize FAILED");
    ctxM = engineM->createExecutionContext();
    if (!ctxM) fatal_exit(11, "init", "mp ctx FAILED");
    int nbM = engineM->getNbIOTensors();    bindsM.resize(nbM);
    for (int i = 0; i < nbM; ++i) {
      const char* name = engineM->getIOTensorName(i);
      Bind& b = bindsM[i];
      b.name = name;
      b.dt = engineM->getTensorDataType(name);
      b.is_input = engineM->getTensorIOMode(name) ==
                   nvinfer1::TensorIOMode::kINPUT;
      size_t vol = 1;
      auto dims = engineM->getTensorShape(name);
      for (int d = 0; d < dims.nbDims; ++d) vol *= (size_t)dims.d[d];
      size_t es = b.dt == nvinfer1::DataType::kHALF
                      ? 2
                      : (b.dt == nvinfer1::DataType::kINT8 ? 1 : 4);
      b.bytes = vol * es;
      if (b.is_input) {
        if (!strcmp(name, "t_matrix")) {
          if (b.bytes != 16 * 4) fatal_exit(11, "init", "mp tmat size %zu", b.bytes);
          for (int p = 0; p < kPar; ++p) b.dev[p] = d_tmat[p];
          continue;
        }
        const char* src = nullptr;
        for (auto& al : mp_alias)
          if (!strcmp(name, al.first)) { src = al.second; break; }
        if (!src) {
          // mp 状态输入 (history_*/prev_*): 独立单缓冲, 两 parity 绑同一
          // 地址 —— 复位模板与反馈外旋都写 dev[0], 同流内定序, 与感知
          // 状态口径一致; 反馈对/零填在下面 mstates 一节接
          void* st1 = nullptr;
          if (cudaMalloc(&st1, b.bytes) != cudaSuccess) {
            fatal_exit(12, "init", "mp state %s alloc FAILED", name);
          }
          b.dev[0] = b.dev[1] = st1;
          continue;
        }
        Bind* sb = find_bind(src);
        if (!sb || sb->is_input) {
          fatal_exit(11, "init", "mp input %s: 感知输出 %s 不存在", name, src);
        }
        if (sb->bytes != b.bytes || sb->dt != b.dt) {
          fatal_exit(11, "init", "mp %s mismatch mp %zuB/%d vs per %zuB/%d",
                     name, b.bytes, (int)b.dt, sb->bytes, (int)sb->dt);
        }
        for (int p = 0; p < kPar; ++p) b.dev[p] = sb->dev[p];
      } else if (!strncmp(name, "next_", 5)) {
        // mp 状态反馈输出: 缓冲随输出分配, 反馈对在下面接
        for (int p = 0; p < kPar; ++p) {
          cudaMalloc(&b.dev[p], b.bytes);
          cudaMallocHost(&b.host[p], b.bytes);
        }
        outs_mp.push_back(&b);
      } else {
        for (int p = 0; p < kPar; ++p) {
          cudaMalloc(&b.dev[p], b.bytes);
          cudaMallocHost(&b.host[p], b.bytes);
        }
        outs_mp.push_back(&b);
      }
    }
    // mp 反馈对: prev/历史 输入 → next_ 输出 (规则: "next_"+输入名, 全部成立)
    for (auto& b : bindsM) {
      if (!b.is_input) continue;
      if (!strcmp(b.name.c_str(), "t_matrix")) continue;
      bool aliased = false;
      for (auto& al : mp_alias)
        if (b.name == al.first) { aliased = true; break; }
      if (aliased) continue;  // 感知直供输入, 非状态
      std::string next = "next_" + b.name;
      Bind* ob = nullptr;
      for (auto& x : bindsM)
        if (!x.is_input && x.name == next) { ob = &x; break; }
      if (!ob) fatal_exit(11, "init", "no mp output %s", next.c_str());
      mstates.push_back({&b, ob});
    }
    // 状态零填 + 场景复位模板 (prev id 全 -1, 其余全 0)
    for (auto& st : mstates) {
      bool idm1 = st.in->name == "prev_instance_id";
      std::vector<char> z(st.in->bytes, 0);
      if (idm1) {
        int* q = (int*)z.data();
        for (size_t i = 0; i < st.in->bytes / 4; ++i) q[i] = -1;
      }
      void* rst = nullptr;
      cudaMalloc(&rst, st.in->bytes);
      cudaMemcpy(rst, z.data(), st.in->bytes, cudaMemcpyHostToDevice);
      d_mrst.push_back(rst);
      cudaMemcpy(st.in->dev[0], z.data(), st.in->bytes, cudaMemcpyHostToDevice);
      printf("modelnode: mp state %s (%zu B, reset=%s)\n",
             st.in->name.c_str(), st.in->bytes, idm1 ? "-1" : "0");
    }
  }
  // mp 全量绑定 (init 一次 + 每帧 parity 重绑走同一函数)
  auto mp_bind_all = [&](int p) -> bool {
    for (auto& b : bindsM)
      if (!ctxM->setTensorAddress(b.name.c_str(), b.dev[p])) return false;
    return true;
  };
  if (use_mp) {
    if (!mp_bind_all(0)) fatal_exit(11, "init", "mp bind failed");
    printf("modelnode: mp %d tensors (%zu outs, %zu states)\n",
           (int)bindsM.size(), outs_mp.size(), mstates.size());
  }

  // ---- Q1-Q3 decode 源张量 (名字全等; 布局按口径定长断言) ----
  auto find_out = [&](const char* nm) -> Bind* {
    for (auto* b : outs)
      if (b->name == nm) return b;
    return nullptr;
  };
  Bind* b_det_cls = find_out("det_cls");
  Bind* b_det_q = find_out("det_quality");
  Bind* b_det_bbox = find_out("det_bbox");
  Bind* b_det_id = find_out("det_instance_id");
  Bind* b_map_cls = find_out("map_cls");
  Bind* b_map_pts = find_out("map_pts");
  bool can_decode = b_det_cls && b_det_q && b_det_bbox && b_det_id &&
                    b_map_cls && b_map_pts;
  if (can_decode) {
    struct { Bind* b; size_t want; const char* nm; } chk[] = {
        {b_det_cls, (size_t)post::kDetAnchors * post::kDetCls * 4, "det_cls"},
        {b_det_q, (size_t)post::kDetAnchors * 2 * 4, "det_quality"},
        {b_det_bbox, (size_t)post::kDetAnchors * 11 * 4, "det_bbox"},
        {b_det_id, (size_t)post::kDetAnchors * 4, "det_instance_id"},
        {b_map_cls, (size_t)post::kMapAnchors * post::kMapCls * 4, "map_cls"},
        {b_map_pts, (size_t)post::kMapAnchors * post::kMapPts * 2 * 4,
         "map_pts"},
    };
    for (auto& c : chk)
      if (c.b->bytes != c.want) {
        fatal_exit(11, "init", "%s bytes=%zu want=%zu (decode 口径失配)",
                   c.nm, c.b->bytes, c.want);
      }
    printf("modelnode: decode on (det_thr=%.3f topk=%d map_thr=%.3f)\n",
           det_thr, det_topk, map_thr);
  } else {
    printf("modelnode: decode tensors missing, mailbox/json disabled\n");
  }
  const uint64_t cfg_hash = fnv1a(
      std::string(engine_path) + "|" + (hd_path ? hd_path : "") + "|" +
      (mp_path ? mp_path : "") + "|" + plugin_so + "|det_thr=" +
      std::to_string(det_thr) + "|map_thr=" + std::to_string(map_thr) +
      "|topk=" + std::to_string(det_topk));

  // dump 目录预建 (落盘移出关键路径); out_dir 本身无论是否 dump 都要建
  {
    std::string cmd = "mkdir -p " + out_dir;
    if (system(cmd.c_str()) != 0)
      fatal_exit(10, "init", "mkdir %s failed", out_dir.c_str());
  }
  if (!no_dump) {
    for (long k = 0; k < n_frames; ++k) {
      char d[512];
      snprintf(d, sizeof(d), "mkdir -p %s/out_%02ld", out_dir.c_str(), k);
      if (system(d) != 0) fatal_exit(10, "init", "mkdir %s failed", d);
      if (use_mp) {
        snprintf(d, sizeof(d), "mkdir -p %s/outm_%02ld", out_dir.c_str(), k);
        if (system(d) != 0) fatal_exit(10, "init", "mkdir %s failed", d);
      }
    }
  }
  // dump helper: 从 pinned host[p] 落盘 (GPU 之外, 纯 CPU)
  auto dump_frame = [&](long k, int p) {
    char d[512];
    snprintf(d, sizeof(d), "%s/out_%02ld", out_dir.c_str(), k);
    std::string manifest;
    for (auto* b : outs) {
      std::string fn = b->name + ".bin";
      FILE* g = fopen((std::string(d) + "/" + fn).c_str(), "wb");
      fwrite(b->host[p], 1, b->bytes, g);
      fclose(g);
      const char* dts = b->dt == nvinfer1::DataType::kINT32 ? "i32" : "f32";
      manifest += b->name + "\t" + dts + "\t" + fn + "\n";
    }
    FILE* g = fopen((std::string(d) + "/manifest.tsv").c_str(), "wb");
    fwrite(manifest.data(), 1, manifest.size(), g);
    fclose(g);
    if (use_mp) {
      snprintf(d, sizeof(d), "%s/outm_%02ld", out_dir.c_str(), k);
      std::string mf;
      for (auto* b : outs_mp) {
        std::string fn = b->name + ".bin";
        FILE* q = fopen((std::string(d) + "/" + fn).c_str(), "wb");
        fwrite(b->host[p], 1, b->bytes, q);
        fclose(q);
        const char* dts = b->dt == nvinfer1::DataType::kINT32 ? "i32" : "f32";
        mf += b->name + "\t" + dts + "\t" + fn + "\n";
      }
      FILE* q = fopen((std::string(d) + "/manifest.tsv").c_str(), "wb");
      fwrite(mf.data(), 1, mf.size(), q);
      fclose(q);
    }
  };

  // ---- events: 每 parity 一组, 读上一轮再复用 ----
  cudaEvent_t ev_preA[kPar], ev_preB[kPar], ev_inf0[kPar], ev_inf1[kPar],
      ev_post[kPar], ev_b2[kPar], ev_hd[kPar];
  for (int p = 0; p < kPar; ++p) {
    cudaEventCreate(&ev_preA[p]);
    cudaEventCreate(&ev_preB[p]);
    cudaEventCreate(&ev_inf0[p]);
    cudaEventCreate(&ev_inf1[p]);
    cudaEventCreate(&ev_post[p]);
    cudaEventCreate(&ev_b2[p]);
    cudaEventCreate(&ev_hd[p]);
  }

  // ---- warmup: 零图交替预热两个 ctx (不消费) ----
  if (warmup > 0) {
    for (int i = 0; i < warmup; ++i) {
      int p = i & 1;
      cudaMemset(d_img[p], 0, pre.out_bytes());
      cudaMemset(d_proj[p], 0, 6 * 16 * 4);
      cudaMemset(d_tmat[p], 0, 16 * 4);
      cudaMemset(d_dt[p], 0, 4);
      if (use_hd && !ctxBB[p]->enqueueV3(dual ? eng_streamB : eng_stream)) {
        fatal_exit(12, "warmup", "bb2 enqueue FAILED");
      }
      if (!ctx[p]->enqueueV3(eng_stream)) {
        fatal_exit(12, "warmup", "enqueue FAILED");
      }
      if (use_mp) {
        if (!mp_bind_all(p) || !ctxM->enqueueV3(eng_stream)) {
          fatal_exit(12, "warmup", "mp enqueue FAILED");
        }
      }
      cudaStreamSynchronize(eng_stream);
      if (dual) cudaStreamSynchronize(eng_streamB);
    }
    printf("modelnode: warmup %d done\n", warmup);
  }

  // ---- CUDA graph: 推理+反馈按 parity 各捕获 (M4 实测项) ----
  // 拆分模式两张: gB[p]=bb2, gH[p]=hd+感知反馈 —— 事件可夹中间分段计时;
  // 单引擎模式一张: gB[p]=推理+反馈 (M4 原语义). mp 不捕 (M7 再议).
  cudaGraphExec_t gB[kPar] = {nullptr, nullptr};
  cudaGraphExec_t gH[kPar] = {nullptr, nullptr};
  if (use_graph) {
    for (int p = 0; p < kPar; ++p) {
      cudaStream_t sB = dual ? eng_streamB : eng_stream;
      cudaStreamSynchronize(sB);
      cudaStreamBeginCapture(sB, cudaStreamCaptureModeGlobal);
      bool ok = use_hd ? ctxBB[p]->enqueueV3(sB) : ctx[p]->enqueueV3(sB);
      if (ok && !use_hd)
        for (auto& st : states)
          cudaMemcpyAsync(st.in->dev[0], st.out->dev[p], st.in->bytes,
                          cudaMemcpyDeviceToDevice, sB);
      cudaGraph_t g = nullptr;
      if (cudaStreamEndCapture(sB, &g) != cudaSuccess || !ok) {
        printf("modelnode: graph capture FAILED (p=%d), fallback non-graph\n",
               p);
        use_graph = false;
        break;
      }
      size_t nn = 0;
      cudaGraphGetNodes(g, nullptr, &nn);
      if (cudaGraphInstantiate(&gB[p], g, nullptr, nullptr, 0) !=
              cudaSuccess ||
          nn == 0) {
        printf("modelnode: graph instantiate FAILED (nodes=%zu)\n", nn);
        use_graph = false;
        break;
      }
      cudaGraphDestroy(g);
      printf("modelnode: graphB[%d] captured (%zu nodes)\n", p, nn);
      if (use_hd) {
        cudaStreamSynchronize(eng_stream);
        cudaStreamBeginCapture(eng_stream, cudaStreamCaptureModeGlobal);
        ok = ctx[p]->enqueueV3(eng_stream);
        if (ok)
          for (auto& st : states)
            cudaMemcpyAsync(st.in->dev[0], st.out->dev[p], st.in->bytes,
                            cudaMemcpyDeviceToDevice, eng_stream);
        g = nullptr;
        if (cudaStreamEndCapture(eng_stream, &g) != cudaSuccess || !ok) {
          printf("modelnode: hd graph capture FAILED (p=%d)\n", p);
          use_graph = false;
          break;
        }
        nn = 0;
        cudaGraphGetNodes(g, nullptr, &nn);
        if (cudaGraphInstantiate(&gH[p], g, nullptr, nullptr, 0) !=
                cudaSuccess ||
            nn == 0) {
          printf("modelnode: hd graph instantiate FAILED (nodes=%zu)\n", nn);
          use_graph = false;
          break;
        }
        cudaGraphDestroy(g);
        printf("modelnode: graphH[%d] captured (%zu nodes)\n", p, nn);
      }
    }
  }

  // 引擎就绪后才注册为消费者 —— filesrc --wait-cons 以此为发布起点,
  // 保证 node 从 seq 1 开始消费, seq↔manifest 严格对齐.
  // watchdog 在图捕获完成后、进主循环前启动 (冷启动/加载期不设防).
  watch_start();
  const int32_t cid = bus->register_consumer();
  if (cid < 0) fatal_exit(11, "init", "no consumer slot");
  printf("modelnode: consumer registered, ready\n");

  // ---- 主循环 (默认流水线: 提交不阻塞, CPU 只在完成帧处同步) ----
  uint64_t last_seq = 0;
  uint64_t seq_base = 0;  // resync 基线: 中途接入时设为 首帧seq-1, 之后
                          // 期望 seq = seq_base + submitted + 1
  uint32_t cur_scene = 0xFFFFFFFFu;
  bool have_prev = false;
  double prev_l2g[16];
  int64_t prev_ts = 0;
  long done = 0;  // 已完成(后处理读完)帧数
  std::deque<std::pair<FrameView, cudaEvent_t>> relq;  // fence 异步释放
  std::string logpath = out_dir + "/frame_log.tsv";
  FILE* flog = fopen(logpath.c_str(), "w");
  if (flog) setvbuf(flog, nullptr, _IOLBF, 0);
  if (flog)
    fprintf(flog, "seq\tpre_ms\tinfer_ms\tpost_ms\tgpu_ms\tsvc_ms\tacq_ms"
                  "\tready_ns\tbb2_ms\thd_ms\tmp_ms\n");

  // ---- 结果信箱 + JSON 旁路 ----
  char rerr[256];
  res::Mailbox* mailbox = res::Mailbox::create(mailbox_name.c_str(), rerr,
                                               sizeof(rerr));
  if (!mailbox) {
    fatal_exit(12, "init", "mailbox: %s", rerr);
  }
  printf("modelnode: mailbox sp_res_%s ready\n", mailbox_name.c_str());
  FILE* fjson = nullptr;
  if (!no_dump) {
    fjson = fopen((out_dir + "/result.jsonl").c_str(), "w");
    if (fjson) setvbuf(fjson, nullptr, _IOLBF, 0);
  }

  // 每帧提交时留档 (收割时填进结果消息; complete_frame 只看帧号)
  std::vector<uint64_t> f_seq(n_frames, 0);
  std::vector<int64_t> f_ts(n_frames, 0);
  std::vector<int64_t> f_acq_ns(n_frames, 0);  // acquire 墙钟 (frame_age 基准)
  std::vector<uint32_t> f_scene(n_frames, 0), f_flags(n_frames, 0);

  std::vector<double> s_pre, s_inf, s_post, s_svc, s_dec, s_json;
  std::vector<double> s_b2, s_hdms, s_mp;
  std::vector<double> t_acq(n_frames, 0.0), acq_ms_v(n_frames, 0.0);
  double t_first_ready = 0, t_last_ready = 0;
  long submitted = 0;

  // 完成帧 j: 等 post D2H → 读事件耗时 → dump → 记日志
  auto complete_frame = [&](long j) {
    watch_touch(kWsPost);
    int64_t wpost0 = watch_now_ms();
    int p = (int)(j & 1);
    if (!no_dump) cudaEventSynchronize(ev_post[p]);
    else cudaEventSynchronize(ev_inf1[p]);
    float pre_ms = 0, inf_ms = 0, post_ms = 0, gpu_ms = 0;
    float b2_ms = 0, hd_ms = 0, mp_ms = 0;
    cudaEventElapsedTime(&pre_ms, ev_preA[p], ev_preB[p]);
    cudaEventElapsedTime(&inf_ms, ev_inf0[p], ev_inf1[p]);
    cudaEventElapsedTime(&b2_ms, ev_inf0[p], ev_b2[p]);
    cudaEventElapsedTime(&hd_ms, ev_b2[p], ev_hd[p]);
    cudaEventElapsedTime(&mp_ms, ev_hd[p], ev_inf1[p]);
    if (!no_dump) cudaEventElapsedTime(&post_ms, ev_inf1[p], ev_post[p]);
    if (!no_dump) cudaEventElapsedTime(&gpu_ms, ev_preA[p], ev_post[p]);
    else cudaEventElapsedTime(&gpu_ms, ev_preA[p], ev_inf1[p]);
    // 设防段只盖 GPU 同步等待 (hang 在这里现形); 落盘/解码/发布是 IO
    watch_record(kWsPost, (double)(watch_now_ms() - wpost0));
    watch_touch(kWsIo);
    double ready = now_ns();
    if (!t_first_ready) t_first_ready = ready;
    t_last_ready = ready;
    if (!no_dump) dump_frame(j, p);
    // ---- Q1-Q3: decode → 信箱发布 → JSON 旁路 (纯 CPU, 从 pinned host 读) ----
    if (can_decode) {
      double t_d0 = now_ns();
      static post::DetOut dets[res::kDetCap];
      static post::MapOut maps[res::kMapCap];
      int nd = post::det_decode((const float*)b_det_cls->host[p],
                                (const float*)b_det_q->host[p],
                                (const float*)b_det_bbox->host[p],
                                (const int32_t*)b_det_id->host[p],
                                det_topk, det_thr, dets);
      int nm = post::map_decode((const float*)b_map_cls->host[p],
                                (const float*)b_map_pts->host[p],
                                map_thr, maps);
      res::ResultMsg msg;
      memset(&msg, 0, sizeof(msg));
      msg.magic = res::kMagic;
      msg.version = res::kVer;
      msg.header_size = (uint16_t)offsetof(res::ResultMsg, det);
      msg.seq = f_seq[j];
      // acquire 墙钟, 不用 manifest group_ts_ns (nuscenes 采集纪元, 与
      // now 相减是 8 年, age 恒钳 65535); 评估口径仍走 f_ts
      msg.ts_capture_ns = f_acq_ns[j];
      msg.frame_id = (uint32_t)j;
      // 打包: scene 高 24b; 低 8b = 源状态位重映射 — bit0 source_ok,
      // bit1..6 cam[0..5] ok (原 flags bits 8..13), bit7 备用
      msg.scene_flags = (f_scene[j] << 8) | (f_flags[j] & 1u) |
                        ((f_flags[j] >> 7) & 0x7Eu);
      msg.config_hash = cfg_hash;
      snprintf(msg.plugin_path, res::kPluginPathMax, "%s", plugin_so);
      msg.pre_ms = pre_ms;
      msg.infer_ms = inf_ms;
      msg.post_ms = post_ms;
      msg.e2e_ms = gpu_ms;
      msg.n_det = (uint32_t)nd;
      msg.n_map = (uint32_t)nm;
      for (int i = 0; i < nd; ++i) {
        res::DetBox& b = msg.det[i];
        b.score = dets[i].score;
        b.label = dets[i].label;
        b.x = dets[i].x; b.y = dets[i].y; b.z = dets[i].z;
        b.w = dets[i].w; b.l = dets[i].l; b.h = dets[i].h;
        b.yaw = dets[i].yaw; b.vx = dets[i].vx; b.vy = dets[i].vy;
        b.id = dets[i].id;
      }
      for (int i = 0; i < nm; ++i) {
        res::MapVec& b = msg.map[i];
        b.score = maps[i].score;
        b.label = maps[i].label;
        memcpy(b.pts, maps[i].pts, sizeof(b.pts));
      }
      if (use_mp) {
        // M6b: v2 motion/plan 段 = e_mp 原始输出直拷 + final_plan 便捷解码
        for (auto* b : outs_mp) {
          if (b->name == "motion_cls")
            memcpy(msg.motion_cls, b->host[p], res::kMotClsLen * 4);
          else if (b->name == "motion_reg")
            memcpy(msg.motion_reg, b->host[p], res::kMotRegLen * 4);
          else if (b->name == "plan_cls")
            memcpy(msg.plan_cls, b->host[p], res::kPlanClsLen * 4);
          else if (b->name == "plan_reg")
            memcpy(msg.plan_reg, b->host[p], res::kPlanRegLen * 4);
          else if (b->name == "plan_status")
            memcpy(msg.plan_status, b->host[p], res::kPlanStatusLen * 4);
        }
        int best = plan_cmd * 6;
        float bv = msg.plan_cls[best];
        for (int m = 1; m < 6; ++m)
          if (msg.plan_cls[plan_cmd * 6 + m] > bv) {
            bv = msg.plan_cls[plan_cmd * 6 + m];
            best = plan_cmd * 6 + m;
          }
        memcpy(msg.final_plan, msg.plan_reg + best * res::kPlanPts * 2,
               res::kPlanPts * 2 * 4);
        msg.t_mp = mp_ms;
        msg.cmd = (uint32_t)plan_cmd;
      }
      msg.crc = res::msg_crc(msg, crc32);
      // ---- v3 fail-visible (M-PROD A4): Phase A 恒 NOMINAL, 年龄照实填.
      // age = 发布时刻 - 捕获时刻 (CLOCK_REALTIME, group_ts_ns 同源).
      msg.status = res::kStatusNominal;
      msg.reason = res::kReasonNone;
      msg.wd_stage = (uint8_t)watch_cur_stage();
      msg.last_valid_seq = (uint32_t)f_seq[j];
      int64_t age_ms = (now_real_ns() - msg.ts_capture_ns) / 1000000;
      msg.frame_age_ms = (uint16_t)(age_ms < 0 ? 0
                                    : (age_ms > 65535 ? 65535 : age_ms));
      mailbox->publish(msg);
      double t_d1 = now_ns();
      if (fjson) {
        fprintf(fjson,
                "{\"seq\":%lu,\"frame_id\":%ld,\"ts_capture_ns\":%lld,"
                "\"scene\":%u,\"flags\":%u,\"config_hash\":\"%016llx\","
                "\"plugin\":\"%s\",\"pre_ms\":%.3f,\"infer_ms\":%.3f,"
                "\"post_ms\":%.3f,\"e2e_ms\":%.3f,\"n_det\":%d,\"n_map\":%d,"
                "\"det\":[",
                (unsigned long)msg.seq, j, (long long)msg.ts_capture_ns,
                f_scene[j], f_flags[j], (unsigned long long)cfg_hash,
                plugin_so, pre_ms, inf_ms, post_ms, gpu_ms, nd, nm);
        for (int i = 0; i < nd; ++i)
          fprintf(fjson,
                  "%s[%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                  "%d]", i ? "," : "", dets[i].score, dets[i].label,
                  dets[i].x, dets[i].y, dets[i].z, dets[i].w, dets[i].l,
                  dets[i].h, dets[i].yaw, dets[i].vx, dets[i].vy, dets[i].id);
        fprintf(fjson, "],\"map\":[");
        for (int i = 0; i < nm; ++i) {
          fprintf(fjson, "%s[%.6f,%d,[", i ? "," : "", maps[i].score,
                  maps[i].label);
          for (int q = 0; q < post::kMapPts; ++q)
            fprintf(fjson, "%s[%.6f,%.6f]", q ? "," : "", maps[i].pts[q][0],
                    maps[i].pts[q][1]);
          fprintf(fjson, "]]");
        }
        fprintf(fjson, "]}\n");
      }
      s_dec.push_back((t_d1 - t_d0) / 1e6);
      if (fjson) s_json.push_back((now_ns() - t_d1) / 1e6);
    }
    double svc_ms = t_acq[j] > 0 ? (ready - t_acq[j]) / 1e6 : 0;
    if (flog)
      fprintf(flog,
              "%ld\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.0f\t%.3f\t%.3f\t"
              "%.3f\n",
              j, pre_ms, inf_ms, post_ms, gpu_ms, svc_ms, acq_ms_v[j], ready,
              b2_ms, hd_ms, mp_ms);
    s_pre.push_back(pre_ms);
    s_inf.push_back(inf_ms);
    s_post.push_back(post_ms);
    s_svc.push_back(gpu_ms);  // 事件链 e2e = 真实服务延迟 (不受收割节奏影响)
    if (use_hd) s_b2.push_back(b2_ms);
    s_hdms.push_back(hd_ms);
    if (use_mp) s_mp.push_back(mp_ms);
    done = j + 1;
  };

  bool abort_run = false;
  while (!g_stop && submitted < n_frames && !abort_run) {
    long k = submitted;
    // 完成帧 k-2 + fence 归还放在 acquire 之前: 不让源节拍污染 svc 延迟,
    // 也让槽位尽早归还 (正常事件早已完成, 这里只做收割)
    if (k >= 2 && done <= k - 2) complete_frame(k - 2);
    while (!relq.empty() &&
           cudaEventQuery(relq.front().second) == cudaSuccess) {
      bus->release(&relq.front().first);
      relq.pop_front();
    }
    double t0 = now_ns();
    FrameView v;
    int nto = 0;
    bool got = false;
    watch_touch(kWsAcq);
    while (!g_stop) {
      if (bus->acquire(cid, last_seq, &v, 2000) == 0) { got = true; break; }
      if (++nto > 30) {
        fprintf(stderr, "modelnode: source exhausted at n=%ld\n", submitted);
        abort_run = true;
        break;
      }
      if (nto % 10 == 1)
        fprintf(stderr, "modelnode: acquire timeout (n=%ld)\n", submitted);
    }
    if (!got) break;
    double acq_ms = (now_ns() - t0) / 1e6;
    last_seq = v.meta.seq;
    watch_seq(v.meta.seq);
    if (v.meta.seq != seq_base + (uint64_t)(submitted + 1)) {
      if (submitted == 0 && v.meta.seq > 1) {
        // 服务重启重同步: unit 不带 --fresh, 环持久化 seq 单调, 中途接入
        // 从当前 seq 续跑 (精度在下一场景边界自然刷新, 见 spec §5 A3)
        printf("modelnode: resync at seq %lu (mid-stream attach)\n",
               (unsigned long)v.meta.seq);
        seq_base = v.meta.seq - 1;
      } else {
        fatal_exit(12, "run", "seq misalign got %lu expect %lu",
                   (unsigned long)v.meta.seq,
                   (unsigned long)(seq_base + submitted + 1));
      }
    }
    t_acq[k] = t0;
    acq_ms_v[k] = acq_ms;
    f_seq[k] = v.meta.seq;
    f_ts[k] = v.meta.group_ts_ns;
    f_acq_ns[k] = now_real_ns();  // 场景进入节点时刻 (age 语义: 消费方
                                  // now-ts_capture 覆盖处理+重启间隙旧化)
    f_scene[k] = v.meta.scene_id;
    f_flags[k] = v.meta.flags;
    int p = (int)(k & 1);
    // manifest 配对跟 seq 走: filesrc 从 seq=1 起按 seq s 携带 manifest[(s-1)%nman]
    // (含 --loop 回绕)。中途接入 (resync) 时 k 与 seq 不再同步, k%nman 会
    // 配错帧, seq 配对在任何接入时机都精确。
    long mki = (long)((v.meta.seq - 1) % (uint64_t)nman);
    const FrameMetaV2& fm = man.frames[mki];

    // 场景边界/首帧: 首帧零状态已在循环前完成; 边界与链路二口径一致 ——
    // 只重置 t_matrix(identity)+dt(0.5), 实例状态跨场景保留
    // (cur_scene 首帧也要落账, 否则下一帧误判成边界)
    bool reset = !have_prev || fm.scene != cur_scene;
    if (fm.scene != cur_scene) {
      if (have_prev)
        printf("modelnode: scene %u -> %u at frame %ld (tmat=identity,"
               " dt=0.5, 状态保留)\n", cur_scene, fm.scene, k);
      cur_scene = fm.scene;
    }

    // P2 + t_matrix + dt (host, double)
    float proj[96];
    make_projection(man, (int)mki, proj);
    float tmat[16];
    if (reset) {
      for (int i = 0; i < 16; ++i) tmat[i] = i % 5 == 0 ? 1.f : 0.f;
    } else {
      double inv[16], t[16];
      mat4_inv(&fm.l2g[0], inv);
      mat4_mul(inv, prev_l2g, t);
      for (int i = 0; i < 16; ++i) tmat[i] = (float)t[i];
    }
    float dt = reset ? 0.5f : (float)((double)(fm.ts_ns - prev_ts) / 1e9);
    memcpy(prev_l2g, &fm.l2g[0], sizeof(prev_l2g));
    prev_ts = fm.ts_ns;
    have_prev = true;

    // ---- pre submit (异步) ----
    // 该 parity 的 host 暂存上一次使用是 k-2 帧: 等那次 H2D 真正执行完
    // 才允许覆写 pinned 源 (正常远早于此就完成, 这里只是正确性兜底)
    watch_touch(kWsPre);
    watch_maybe_stall(kWsPre, k);
    int64_t wpre0 = watch_now_ms();
    if (k >= 2) cudaEventSynchronize(ev_preB[p]);
    // 设防段只盖 GPU 同步; 之后的 pinned 拷贝/读图是 IO (不设防)
    watch_record(kWsPre, (double)(watch_now_ms() - wpre0));
    watch_touch(kWsIo);
    memcpy(h_proj[p], proj, 96 * 4);
    memcpy(h_tmat[p], tmat, 16 * 4);
    memcpy(h_dt[p], &dt, 4);
    cudaEventRecord(ev_preA[p], pre_stream);
    cudaMemcpyAsync(d_proj[p], h_proj[p], 96 * 4, cudaMemcpyHostToDevice,
                    pre_stream);
    cudaMemcpyAsync(d_tmat[p], h_tmat[p], 16 * 4, cudaMemcpyHostToDevice,
                    pre_stream);
    cudaMemcpyAsync(d_dt[p], h_dt[p], 4, cudaMemcpyHostToDevice, pre_stream);
    if (img_from) {
      // 隔离模式: 跳过前处理直接 H2D 链路二 img. 参数形态:
      //  a) 目录根 → <root>/in_XX/img.bin 逐帧取图 (推荐, 命令行无 %);
      //  b) 含 '%' 的逐帧模板; c) 具体文件 → 每帧复用同一 img.
      static std::vector<char> refimg[kPar];
      char pfb[512];
      struct stat st0;
      if (strchr(img_from, '%')) {
        snprintf(pfb, sizeof(pfb), img_from, (int)(k % nman));
      } else if (stat(img_from, &st0) == 0 && S_ISDIR(st0.st_mode)) {
        snprintf(pfb, sizeof(pfb), "%s/in_%02ld/img.bin", img_from,
                 k % nman);
      } else {
        snprintf(pfb, sizeof(pfb), "%s", img_from);
      }
      FILE* g = fopen(pfb, "rb");
      if (!g) fatal_exit(10, "run", "open %s", pfb);
      fseek(g, 0, SEEK_END);
      long n = ftell(g);
      fseek(g, 0, SEEK_SET);
      refimg[p].resize(n);
      if (fread(refimg[p].data(), 1, n, g) != (size_t)n)
        fatal_exit(10, "run", "short read %s", pfb);
      fclose(g);
      if (k == 0)
        printf("modelnode: img-from %s (%ld bytes)\n", pfb, n);
      cudaMemcpyAsync(d_img[p], refimg[p].data(), refimg[p].size(),
                      cudaMemcpyHostToDevice, pre_stream);
    } else {
      // M8: --dma 时读设备池槽 (fd 导入的设备指针), 否则读 mapped-shm
      const uint8_t* slot_dev =
          use_dma ? dma_dev[v.slot_idx]
                  : dev_base + (v.cam[0] - (const uint8_t*)bus->base());
      pre.run(slot_dev, rm->cam_bytes, d_img[p], pre_stream);
    }
    cudaEventRecord(ev_preB[p], pre_stream);
    relq.push_back({v, ev_preB[p]});  // fence: 前处理完成才还槽
    bus->heartbeat(cid);

    if (dump_img_n > 0 && k < dump_img_n) {
      cudaStreamSynchronize(pre_stream);
      std::vector<float> ih(pre.out_bytes() / 4);
      cudaMemcpy(ih.data(), d_img[p], pre.out_bytes(), cudaMemcpyDeviceToHost);
      char pf[512];
      snprintf(pf, sizeof(pf), "%s/img_%02ld.bin", out_dir.c_str(), k);
      FILE* g = fopen(pf, "wb");
      fwrite(ih.data(), 4, ih.size(), g);
      fclose(g);
      char xf[512];
      snprintf(xf, sizeof(xf), "%s/in_%02ld.bin", out_dir.c_str(), k);
      g = fopen(xf, "wb");
      fwrite(proj, 4, 96, g);
      fwrite(tmat, 4, 16, g);
      fwrite(&dt, 4, 1, g);
      fclose(g);
    }

    // ---- infer submit (异步; 等前处理 + 等 parity 的输出 D2H 完成) ----
    // M7a 双流 (--dual): bb2 走 B 流 (eng_streamB), hd+mp 留 H 流
    // (eng_stream). 事件定序:
    //   B 流: 等 preproc 完成 (ev_preB) + 边界槽 p 上一读方 (k-2 帧) 输出
    //         D2H 完成 (ev_post, 防覆写仍在读的 col_feats) → bb2 → ev_b2.
    //   H 流: 等输出槽复用守卫 (ev_post) + 跨流 ev_b2 (col_feats 就绪) →
    //         hd → 感知状态反馈 → mp → ev_inf1.
    // 逐帧交错后 GPU 侧自然重叠: bb2(k+1) ∥ hd(k)+mp(k), 帧周期
    // ≈ max(bb2, hd+mp). 非双流路径与 M5 单流事件序完全一致.
    cudaStream_t sB = dual ? eng_streamB : eng_stream;
    watch_touch(kWsBB2);
    watch_maybe_stall(kWsBB2, k);
    cudaStreamWaitEvent(sB, ev_preB[p], 0);
    cudaStreamWaitEvent(sB, ev_post[p], 0);  // 槽 p 复用守卫 (B=边界, H=输出)
    cudaEventRecord(ev_inf0[p], sB);
    bool ok = true;
    if (use_hd) {
      // M5: bb2 → (事件) → hd+感知反馈, 分段计时
      if (use_graph) ok = cudaGraphLaunch(gB[p], sB) == cudaSuccess;
      else ok = ctxBB[p]->enqueueV3(sB);
      cudaEventRecord(ev_b2[p], sB);
      watch_touch(kWsHD);
      watch_maybe_stall(kWsHD, k);
      if (ok) {
        cudaStreamWaitEvent(eng_stream, ev_post[p], 0);  // out[p] 复用守卫
        cudaStreamWaitEvent(eng_stream, dual ? ev_b2[p] : ev_preB[p], 0);
        if (use_graph) ok = cudaGraphLaunch(gH[p], eng_stream) == cudaSuccess;
        else {
          ok = ctx[p]->enqueueV3(eng_stream);
          if (ok)
            for (auto& st : states)
              cudaMemcpyAsync(st.in->dev[0], st.out->dev[p], st.in->bytes,
                              cudaMemcpyDeviceToDevice, eng_stream);
        }
      }
    } else {
      cudaStreamWaitEvent(eng_stream, ev_post[p], 0);
      if (use_graph) ok = cudaGraphLaunch(gB[p], eng_stream) == cudaSuccess;
      else {
        ok = ctx[p]->enqueueV3(eng_stream);
        if (ok)
          for (auto& st : states)
            cudaMemcpyAsync(st.in->dev[0], st.out->dev[p], st.in->bytes,
                            cudaMemcpyDeviceToDevice, eng_stream);
      }
      cudaEventRecord(ev_b2[p], eng_stream);  // 单引擎: b2 段记 0, 段时在 hd
    }
    cudaEventRecord(ev_hd[p], eng_stream);
    if (ok && use_mp) {
      // M6a: 场景首帧 mp 状态全清零 (模板 D2D, 同流定序), 再逐帧重绑+推理
      watch_touch(kWsMP);
      watch_maybe_stall(kWsMP, k);
      if (reset) {
        for (size_t si = 0; si < mstates.size(); ++si)
          cudaMemcpyAsync(mstates[si].in->dev[0], d_mrst[si],
                          mstates[si].in->bytes, cudaMemcpyDeviceToDevice,
                          eng_stream);
      }
      ok = mp_bind_all(p) && ctxM->enqueueV3(eng_stream);
      if (ok)
        for (auto& st : mstates)
          cudaMemcpyAsync(st.in->dev[0], st.out->dev[p], st.in->bytes,
                          cudaMemcpyDeviceToDevice, eng_stream);
    }
    if (!ok) {
      fatal_exit(12, "run", "enqueue FAILED at frame %ld", k);
    }
    cudaEventRecord(ev_inf1[p], eng_stream);

    // ---- post submit (异步): 19 路 D2H → pinned host[p] ----
    watch_touch(kWsPost);
    if (!no_dump) {
      cudaStreamWaitEvent(post_stream, ev_inf1[p], 0);
      for (auto* b : outs)
        cudaMemcpyAsync(b->host[p], b->dev[p], b->bytes,
                        cudaMemcpyDeviceToHost, post_stream);
      if (use_mp)
        for (auto* b : outs_mp)
          cudaMemcpyAsync(b->host[p], b->dev[p], b->bytes,
                          cudaMemcpyDeviceToHost, post_stream);
      cudaEventRecord(ev_post[p], post_stream);
    }

    submitted = k + 1;
    if (serial) {
      // M3 串行模式: 提交后立刻逐段同步 (对照基线, 无重叠)
      cudaStreamSynchronize(pre_stream);
      cudaEventSynchronize(ev_inf1[p]);
      complete_frame(k);
    }
    if (submitted % 40 == 0)
      printf("modelnode: %ld/%ld submitted\n", submitted, n_frames);
  }

  // 收尾: 完成所有已提交未完成的帧
  while (done < submitted && !g_stop) complete_frame(done);
  // fence 清理: 等剩余前处理事件, 归还槽位
  while (!relq.empty()) {
    cudaEventSynchronize(relq.front().second);
    bus->release(&relq.front().first);
    relq.pop_front();
  }
  double span = t_last_ready - t_first_ready;
  double fps = (done > 1 && span > 0) ? (done - 1) * 1e9 / span : 0;
  printf("modelnode: done %ld frames fps=%.2f\n", done, fps);
  auto pp = [](const char* nm, std::vector<double>& v) {
    if (v.empty()) return;
    double sum = 0;
    for (double x : v) sum += x;
    printf("  %-6s p50=%6.2f p90=%6.2f p99=%6.2f mean=%6.2f ms\n", nm,
           pct(v, 0.50), pct(v, 0.90), pct(v, 0.99), sum / v.size());
  };
  pp("pre", s_pre);
  pp("infer", s_inf);
  if (use_hd) pp("bb2", s_b2);
  pp("hd+fb", s_hdms);
  if (use_mp) pp("mp", s_mp);
  pp("post", s_post);
  pp("e2e", s_svc);  // 事件链 acquire→ready (preA 事件起算)
  pp("dec", s_dec);
  pp("json", s_json);
  if (flog) fclose(flog);
  if (fjson) fclose(fjson);
  printf("modelnode: mailbox writes=%lu\n",
         (unsigned long)mailbox->writes());
  if (use_hd)
    for (int p = 0; p < kPar; ++p) ctxBB[p]->destroy();
  if (use_mp) ctxM->destroy();
  ctx[0]->destroy();
  ctx[1]->destroy();
  printf("MODELNODE_DONE\n");
  if (abort_run && loop) {
    // --loop 常驻下源枯竭 = 发布端死亡: 退 20 让 systemd 重启本节点等待
    // 发布端恢复 (Requires 不反向拉活, node 必须自己保持 restart 循环)。
    // 评估/隔离跑是有限帧, 枯竭即正常收尾, 保持 0。
    fprintf(stderr, "modelnode: upstream exhausted in loop mode, exit 20\n");
    return 20;
  }
  return 0;
}
