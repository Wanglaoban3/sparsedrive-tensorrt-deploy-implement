// sp_sub - model_node stand-in for M1: read-only subscriber.
// Attaches to the ring, registers the whole mapping with CUDA once
// (cudaHostRegister, zero-copy path on Orin unified memory), then consumes
// frames: verifies CRC + pattern, tracks sequence gaps and latency, holds
// each frame for --hold-ms to exercise reclamation backpressure.
#include "sp_bus.h"

#if !defined(SP_NO_CUDA)
#include <cuda_runtime.h>
#include "sp_kernels.h"
#endif

#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <thread>
#include <vector>

using namespace sp;

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }
#if !defined(SP_NO_CUDA)
static uint8_t* dev_base_ = nullptr;  // registered shm 的设备地址(GPU 用)
#endif

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <shm_name> <n_frames|0=inf> [--hold-ms N]"
            " [--no-cuda] [--wait-create] [--pat] [--fence] [--race]\n",
            argv[0]);
    return 1;
  }
  const char* name = argv[1];
  long n_frames = atol(argv[2]);
  int hold_ms = 0;
  bool no_cuda = false, wait_create = false, pat_check = false;
  int fence_mode = 0;  // 1 = fence(事件后释放) 2 = race(发射即释放, 复现覆写)
  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "--hold-ms") && i + 1 < argc)
      hold_ms = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--no-cuda")) no_cuda = true;
    else if (!strcmp(argv[i], "--wait-create")) wait_create = true;
    else if (!strcmp(argv[i], "--pat")) pat_check = true;
    else if (!strcmp(argv[i], "--fence")) fence_mode = 1;
    else if (!strcmp(argv[i], "--race")) fence_mode = 2;
  }
#if defined(SP_NO_CUDA)
  if (fence_mode) {
    fprintf(stderr, "sub: --fence/--race need a CUDA build\n");
    return 1;
  }
#endif

  signal(SIGINT, on_sig);
  signal(SIGTERM, on_sig);

  char err[256];
  Bus* bus = nullptr;
  for (int tries = 0; tries < 50; ++tries) {
    bus = Bus::open(name, 1600, 900, false, false, err, sizeof(err));
    if (bus) break;
    if (!wait_create) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  if (!bus) {
    fprintf(stderr, "sub: open failed: %s\n", err);
    return 1;
  }
  const RingMeta* m = bus->meta();
  printf("sub: ring %ux%u frame=%u latest=%lu\n", m->width, m->height,
         m->frame_bytes, (unsigned long)m->latest_seq.load());

#if !defined(SP_NO_CUDA)
  if (!no_cuda) {
    // One-shot registration of the entire mapping (design: never
    // register/unregister per frame).
    // 板端实测(JetPack 5.1.5): cudaHostRegisterDefault 后 kernel 直接解引用
    // host 指针 → GPU 页表没有该映射, illegal memory access 整进程崩;
    // 必须 Mapped 注册 + cudaHostGetDevicePointer 取设备指针(与 host
    // 指针不同值), kernel 一律走设备指针。
    cudaError_t e = cudaHostRegister(bus->base(), bus->mapped_bytes(),
                                     cudaHostRegisterMapped);
    printf("sub: cudaHostRegisterMapped(%zu bytes) -> %s\n",
           bus->mapped_bytes(), cudaGetErrorString(e));
    if (e == cudaSuccess) {
      e = cudaHostGetDevicePointer(&dev_base_, bus->base(), 0);
      printf("sub: cudaHostGetDevicePointer -> %s dev=%p host=%p\n",
             cudaGetErrorString(e), dev_base_, bus->base());
    }
    if (e != cudaSuccess) {
      no_cuda = true;
      fence_mode = 0;
    }
  }
#endif

  const int32_t cid = bus->register_consumer();
  if (cid < 0) {
    fprintf(stderr, "sub: no free consumer slot\n");
    return 1;
  }
  printf("sub: consumer id=%d pid=%d\n", cid, (int)getpid());

  setvbuf(stdout, nullptr, _IOLBF, 0);
  std::vector<int64_t> lat_us;
  lat_us.reserve(1 << 16);
  uint64_t last_seq = 0;
  long n = 0, crc_err = 0, pat_err = 0, gaps = 0, timeouts = 0;
  // full-plane pattern reference (synthetic source deep check)
  std::vector<uint8_t> pat(m->width * m->height);

#if !defined(SP_NO_CUDA)
  // ---- M1.5 fence 回收:kernel 求和 + 事件后释放 ----
  cudaStream_t fence_stream = nullptr;
  struct Pend {
    cudaEvent_t ev;
    FrameView v;
    uint64_t hsum;
    uint8_t* out;
    int pool_idx;
  };
  std::deque<Pend> pendq;
  std::vector<cudaEvent_t> ev_pool;
  std::vector<uint8_t*> out_pool;
  std::deque<int> free_ev;
  long fence_ok = 0, fence_err = 0;
  double bw_gbps = 0.0;
  bool bw_done = false;  // 探针只跑一次(n 在 fence 分支前已自增)
  if (fence_mode) {
    cudaStreamCreate(&fence_stream);
    for (int i = 0; i < kRingDepth; ++i) {
      cudaEvent_t ev;
      cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
      ev_pool.push_back(ev);
      uint8_t* op = nullptr;
      cudaMalloc(&op, 8);
      out_pool.push_back(op);
      free_ev.push_back(i);
    }
  }
  auto drain_fenced = [&](bool wait_all) {
    while (!pendq.empty()) {
      Pend& f = pendq.front();
      if (!wait_all && cudaEventQuery(f.ev) != cudaSuccess) break;
      if (wait_all) cudaEventSynchronize(f.ev);
      uint64_t ksum = 0;
      cudaMemcpy(&ksum, f.out, 8, cudaMemcpyDeviceToHost);
      if (ksum == f.hsum) fence_ok += 1;
      else fence_err += 1;
      bus->release(&f.v);
      free_ev.push_back(f.pool_idx);
      pendq.pop_front();
    }
  };
#else
  auto drain_fenced = []() {};
#endif
  int64_t t0 = now_ms();

  while (!g_stop && (n_frames <= 0 || n < n_frames)) {
#if !defined(SP_NO_CUDA)
    if (fence_mode) drain_fenced(false);
#endif
    FrameView v;
    // fence 模式用短超时:槽位全被在飞帧占住时也要周期性回到 drain,
    // 否则发布端等释放、本端等新帧 → 互等
    int acq_to = fence_mode ? 100 : 2000;
    if (bus->acquire(cid, last_seq, &v, acq_to) != 0) {
      timeouts += 1;
      if (timeouts % 10 == 1)
        fprintf(stderr, "sub: acquire timeout (n=%ld latest=%lu)\n", n,
                (unsigned long)bus->meta()->latest_seq.load());
      continue;
    }
    const uint8_t* base = v.cam[0];
    if (sampled_checksum(base, m->frame_bytes) != v.meta.checksum)
      crc_err += 1;
    // deep check (--pat, synthetic sp_pub stream only): full Y plane per cam
    // must equal the pattern; real file/camera sources skip this
    if (pat_check) {
      for (int c = 0; c < kMaxCams; ++c) {
        uint8_t want = (uint8_t)((v.meta.seq + c) & 0xFF);
        memset(pat.data(), want, pat.size());
        if (memcmp(v.cam[c], pat.data(), pat.size()) != 0) {
          pat_err += 1;
          break;
        }
      }
    }
    if (last_seq != 0 && v.meta.seq != last_seq + 1) {
      gaps += 1;
      fprintf(stderr, "sub: seq gap %lu -> %lu\n",
              (unsigned long)last_seq, (unsigned long)v.meta.seq);
    }
    last_seq = v.meta.seq;
    n += 1;
    int64_t lat = (now_real_ns() - v.meta.group_ts_ns) / 1000;
    if (lat >= 0 && lat < 1000000) lat_us.push_back(lat);
    if (fence_mode == 0) {
      if (hold_ms > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
      bus->release(&v);
    }
#if !defined(SP_NO_CUDA)
    else {
      // CPU 和:槽位仍受本端引用保护,读到的即提交内容
      uint64_t hsum = 0;
      for (size_t i = 0; i < m->frame_bytes; ++i) hsum += v.cam[0][i];
      if (free_ev.empty() && n > 0) drain_fenced(true);  // 池耗尽兜底
      // kernel 一律走设备指针(host 指针在 GPU 页表无映射, 见注册处注释)
      const uint8_t* dev_cam0 =
          dev_base_ + (v.cam[0] - (const uint8_t*)bus->base());
      int pi = free_ev.front();
      cudaMemsetAsync(out_pool[pi], 0, 8, fence_stream);
      if (!bw_done) {  // 一次性带宽探针:GPU 直读 registered shm
        slot_sum_launch(dev_cam0, m->frame_bytes,
                        (uint64_t*)out_pool[pi], fence_stream);
        int64_t tk0 = now_ms();
        cudaStreamSynchronize(fence_stream);
        int64_t dt = now_ms() - tk0;
        bw_gbps = dt > 0 ? (double)m->frame_bytes / dt / 1e6 : 0.0;
        uint64_t ksum = 0;
        cudaMemcpy(&ksum, out_pool[pi], 8, cudaMemcpyDeviceToHost);
        if (ksum != hsum) fence_err += 1; else fence_ok += 1;
        printf("sub: bw probe %u bytes in %ld ms = %.2f GB/s\n",
               m->frame_bytes, (long)dt, bw_gbps);
        bw_done = true;
        // 探针帧不进队列, 必须就地释放 —— 否则该槽引用泄漏, 发布端
        // claim 死等这一个槽(claim 不轮转), 全链路挂死
        bus->release(&v);
      } else {
        // 受控实验:两模式发同样的 延迟(30ms 真实时间)+求和,只差释放时机;
        // race = 发射即释放(30ms 后求和时槽位已被发布端覆写)
        delay_launch(30000000ull, fence_stream);
        slot_sum_launch(dev_cam0, m->frame_bytes,
                        (uint64_t*)out_pool[pi], fence_stream);
        cudaEventRecord(ev_pool[pi], fence_stream);
        free_ev.pop_front();
        if (fence_mode == 2)
          bus->release(&v);  // race: 发射即释放, 复现覆写危害
        else
          pendq.push_back({ev_pool[pi], v, hsum, out_pool[pi], pi});
      }
    }
#endif
    bus->heartbeat(cid);
    if (n % 100 == 0) {
      printf("sub: n=%ld seq=%lu gaps=%ld crc_err=%ld forced=%lu", n,
             (unsigned long)last_seq, gaps, crc_err,
             (unsigned long)bus->forced_recycles());
#if !defined(SP_NO_CUDA)
      if (fence_mode)
        printf(" fence_ok=%ld fence_err=%ld inflight=%zu", fence_ok,
               fence_err, pendq.size());
#endif
      printf("\n");
    }
  }

#if !defined(SP_NO_CUDA)
  if (fence_mode) drain_fenced(true);
#endif

  bus->unregister_consumer(cid);
  std::sort(lat_us.begin(), lat_us.end());
  int64_t p50 = lat_us.empty() ? 0 : lat_us[lat_us.size() / 2];
  int64_t p99 = lat_us.empty() ? 0 : lat_us[lat_us.size() *
                                            99 / 100];
  printf("sub: done n=%ld crc_err=%ld pat_err=%ld gaps=%ld timeouts=%ld "
         "forced=%lu lat_us p50=%ld p99=%ld elapsed=%ldms", n, crc_err,
         pat_err, gaps, timeouts, (unsigned long)bus->forced_recycles(),
         (long)p50, (long)p99, (long)(now_ms() - t0));
#if !defined(SP_NO_CUDA)
  if (fence_mode)
    printf(" fence_ok=%ld fence_err=%ld bw=%.2fGB/s", fence_ok, fence_err,
           bw_gbps);
#endif
  printf("\n");
  delete bus;
  return crc_err > 0 ? 2 : 0;
}
