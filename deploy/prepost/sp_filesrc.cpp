// sp_filesrc: manifest 驱动的 NV12 文件回放发布进程 (M1).
// 用法: sp_filesrc <ring> <w> <h> <fps> <manifest.jsonl> <data_root>
//                 [--loop] [--fresh] [--frames N]
// IImageSource(FileReplaySource) → 拷贝进 shm ring 槽位 → commit.
// 拷贝 12.9MB/帧 ~2ms, 2Hz 回放下可忽略; 零拷贝路径由 M5 dmabuf 源承担.
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <chrono>
#include <thread>

#include "file_source.h"
#include "sp_bus.h"

using namespace sp;

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc < 7) {
    fprintf(stderr,
            "usage: %s <ring> <w> <h> <fps> <manifest.jsonl> <data_root> "
            "[--loop] [--fresh] [--frames N]\n",
            argv[0]);
    return 2;
  }
  const char* name = argv[1];
  uint32_t w = (uint32_t)atol(argv[2]);
  uint32_t h = (uint32_t)atol(argv[3]);
  double fps = atof(argv[4]);
  const char* manifest = argv[5];
  const char* root = argv[6];
  bool fresh = false, unlink_exit = false, loop = false;
  long n_frames = -1;
  int wait_cons = 0;
  for (int i = 7; i < argc; ++i) {
    if (!strcmp(argv[i], "--fresh")) fresh = true;
    else if (!strcmp(argv[i], "--unlink")) unlink_exit = true;
    else if (!strcmp(argv[i], "--loop")) loop = true;
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
    fprintf(stderr, "filesrc: bus open: %s\n", err);
    return 1;
  }
  SourceConfig cfg;
  cfg.manifest_path = manifest;
  cfg.data_root = root;
  cfg.loop = loop;
  FileReplaySource src;
  if (src.open(cfg, err, sizeof(err)) != 0 || src.start() != 0) {
    fprintf(stderr, "filesrc: source open: %s\n", err);
    return 1;
  }
  const RingMeta* m = bus->meta();
  printf("filesrc: ring %ux%u cam=%u frame=%u slots=%u\n", m->width,
         m->height, m->cam_bytes, m->frame_bytes, kRingDepth);
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
  int64_t t0 = now_ms();
  while (!g_stop && (n_frames < 0 || published < n_frames)) {
    // 先 claim 占槽, 后读盘: claim 失败不消耗清单条目 → seq↔manifest 永不错位
    int64_t ta = now_ms();
    uint8_t* p = bus->claim(1000);
    if (!p) {
      blocked += 1;
      fprintf(stderr, "filesrc: claim timeout (holders stuck?)\n");
      continue;
    }
    FrameView v;
    int rc = src.acquire(v, 1000);
    if (rc == 2) {
      printf("filesrc: manifest exhausted at n=%ld\n", published);
      break;  // 占住的槽 ref=0, 下次 claim 自动复用, 无需回滚
    }
    if (rc != 0) {
      blocked += 1;
      fprintf(stderr, "filesrc: acquire rc=%d\n", rc);
      continue;
    }
    memcpy(p, v.cam[0], m->frame_bytes);
    FrameMeta meta = v.meta;  // seq 由 commit 重排, ts/scene 透传
    bus->commit(meta);
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
  return 0;
}
