// sp_filesrc: manifest 驱动的 NV12 文件回放发布进程 (M1).
// 用法: sp_filesrc <ring> <w> <h> <fps> <manifest.jsonl> <data_root>
//                 [--loop] [--fresh] [--frames N] [--wait-cons N] [--dma]
// IImageSource(FileReplaySource) → 拷贝进 shm ring 槽位 → commit.
// 拷贝 12.9MB/帧 ~2ms, 2Hz 回放下可忽略; --dma 走 M8 设备池零拷贝:
// 文件字节 H2D 进 fd 可共享设备池 (SCM_RIGHTS 分发, 消费端
// cudaExternalMemory 导入), shm payload 不填, 校验和对 host 源自算.
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <chrono>
#include <thread>

#include <cuda_runtime.h>

#include "file_source.h"
#include "sp_bus.h"
#include "sp_dmapool.h"
#include "sp_watch.h"

using namespace sp;

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc < 7) {
    fprintf(stderr,
            "usage: %s <ring> <w> <h> <fps> <manifest.jsonl> <data_root> "
            "[--loop] [--fresh] [--frames N] [--wait-cons N] [--dma]\n",
            argv[0]);
    return 2;
  }
  const char* name = argv[1];
  uint32_t w = (uint32_t)atol(argv[2]);
  uint32_t h = (uint32_t)atol(argv[3]);
  double fps = atof(argv[4]);
  const char* manifest = argv[5];
  const char* root = argv[6];
  bool fresh = false, unlink_exit = false, loop = false, dma = false;
  long n_frames = -1;
  int wait_cons = 0;
  for (int i = 7; i < argc; ++i) {
    if (!strcmp(argv[i], "--fresh")) fresh = true;
    else if (!strcmp(argv[i], "--unlink")) unlink_exit = true;
    else if (!strcmp(argv[i], "--loop")) loop = true;
    else if (!strcmp(argv[i], "--dma")) dma = true;
    else if (!strcmp(argv[i], "--wait-cons") && i + 1 < argc)
      wait_cons = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--frames") && i + 1 < argc)
      n_frames = atol(argv[++i]);
  }
  signal(SIGINT, on_sig);
  signal(SIGTERM, on_sig);

  char err[256];
  Bus* bus = Bus::open(name, w, h, /*create=*/true, fresh, err, sizeof(err));
  if (!bus) {
    fatal_exit(10, "init", "bus open: %s", err);
  }
  SourceConfig cfg;
  cfg.manifest_path = manifest;
  cfg.data_root = root;
  cfg.loop = loop;
  FileReplaySource src;
  if (src.open(cfg, err, sizeof(err)) != 0 || src.start() != 0) {
    fatal_exit(10, "init", "source open: %s", err);
  }
  const RingMeta* m = bus->meta();
  printf("filesrc: ring %ux%u cam=%u frame=%u slots=%u\n", m->width,
         m->height, m->cam_bytes, m->frame_bytes, kRingDepth);
  // M8: --dma 建设备池 + fd 服务, 先于消费者注册注册池几何
  DmaPoolPub pool;
  if (dma) {
    if (!pool.create(kRingDepth, m->frame_bytes, err, sizeof(err)) ||
        !pool.serve(name, err, sizeof(err))) {
      fatal_exit(12, "init", "dma pool: %s", err);
    }
    // RingMeta 携带对齐后槽尺寸, 与 UDS DmaPoolInfo 一致 (消费端比对用)
    bus->set_dma_info(kRingDepth, pool.slot_bytes());
    printf("filesrc: dma pool ON (payload shm 槽不填)\n");
  }
  printf("filesrc: manifest=%s entries=%zu %s\n", manifest,
         src.num_entries(), loop ? "loop" : "once");
  if (wait_cons > 0) {
    // 建环后等消费端注册再开拍 (编排: 慢启动的 node 不丢首帧)
    while (bus->consumer_count() < wait_cons && !g_stop) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    printf("filesrc: consumer present, start publishing\n");
  }

  const int64_t frame_ms =
      fps > 0 ? (int64_t)(1000.0 / fps) : 0;
  long published = 0, blocked = 0;
  int consec_timeouts = 0;  // M-PROD A2: 连续 claim 超时 → exit 20 升级
  bool exhausted = false;
  int64_t t0 = now_ms();
  while (!g_stop && (n_frames < 0 || published < n_frames)) {
    // 先 claim 占槽, 后读盘: claim 失败不消耗清单条目 → seq↔manifest 永不错位
    int64_t ta = now_ms();
    int32_t slot_idx = -1;
    uint8_t* p = bus->claim_of(dma ? &slot_idx : nullptr, 1000);
    if (!p) {
      blocked += 1;
      consec_timeouts += 1;
      // 正常恢复路径靠 lease 强制回收 (M1.5); 连续 30 次 = 回收也失效,
      // 交 systemd 重启 (spec §5 A3: exit 20 是兜底, 不是常规恢复动作)
      if (consec_timeouts >= 30)
        fatal_exit(20, "publish", "claim timeout x30 (recycle stuck?)");
      fprintf(stderr, "filesrc: claim timeout (holders stuck?) %d/30\n",
              consec_timeouts);
      continue;
    }
    consec_timeouts = 0;
    FrameView v;
    int rc = src.acquire(v, 1000);
    if (rc == 2) {
      printf("filesrc: manifest exhausted at n=%ld\n", published);
      exhausted = true;
      break;  // 占住的槽 ref=0, 下次 claim 自动复用, 无需回滚
    }
    if (rc != 0) {
      blocked += 1;
      fprintf(stderr, "filesrc: acquire rc=%d\n", rc);
      continue;
    }
    FrameMeta meta = v.meta;  // seq 由 commit 重排, ts/scene 透传
    if (dma) {
      // 设备池路径: 文件字节 → 池槽 (设备), shm payload 不填;
      // 校验和对 host 源自算 (commit_dma 不重读 shm)
      if (cudaMemcpy(pool.dev(slot_idx), v.cam[0], m->frame_bytes,
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "filesrc: dma h2d slot %d failed\n", slot_idx);
        return 1;
      }
      meta.checksum = sampled_checksum(v.cam[0], m->frame_bytes);
      bus->commit_dma(meta);
    } else {
      memcpy(p, v.cam[0], m->frame_bytes);
      bus->commit(meta);
    }
    src.release(v);
    published += 1;
    if (published % 30 == 0)
      printf("filesrc: seq=%lu n=%ld read+copy=%ldms forced=%lu\n",
             (unsigned long)bus->meta()->latest_seq.load(), published,
             (long)(now_ms() - ta),
             (unsigned long)bus->forced_recycles());
    if (frame_ms > 0) {
      int64_t target = t0 + published * frame_ms;
      int64_t d = target - now_ms();
      if (d > 0) std::this_thread::sleep_for(std::chrono::milliseconds(d));
    }
  }
  printf("filesrc: done n=%ld blocked=%ld forced=%lu elapsed=%ldms\n",
         published, blocked, (unsigned long)bus->forced_recycles(),
         (long)(now_ms() - t0));
  src.stop();
  delete bus;
  if (unlink_exit) {
    char full[kNameMax];
    snprintf(full, sizeof(full), "/sp_%s", name);
    shm_unlink(full);
  }
  // A2 退出码契约: 非 loop 模式清单耗尽 = 21 (编排可见的"正常结束";
  // systemd 服务恒 --loop, 不会触发; 直接跑的门禁脚本不依赖该 rc)
  if (exhausted && !loop) return 21;
  return 0;
}
