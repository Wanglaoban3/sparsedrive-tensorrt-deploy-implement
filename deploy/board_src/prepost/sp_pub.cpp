// sp_pub - capture_pub stand-in for M1: synthetic NV12 pattern source.
// Fills claimed slots with a sequence-dependent pattern (fast memsets) and
// publishes at a fixed rate. Real file replay replaces only the fill step.
#include "sp_bus.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

using namespace sp;

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr,
            "usage: %s <shm_name> <width> <height> <fps> <n_frames|0=inf>"
            " [--fresh] [--unlink]\n", argv[0]);
    return 1;
  }
  const char* name = argv[1];
  uint32_t w = (uint32_t)atoi(argv[2]);
  uint32_t h = (uint32_t)atoi(argv[3]);
  double fps = atof(argv[4]);
  long n_frames = atol(argv[5]);
  bool fresh = false, unlink_exit = false;
  for (int i = 6; i < argc; ++i) {
    if (!strcmp(argv[i], "--fresh")) fresh = true;
    if (!strcmp(argv[i], "--unlink")) unlink_exit = true;
  }

  setvbuf(stdout, nullptr, _IOLBF, 0);
  signal(SIGINT, on_sig);
  signal(SIGTERM, on_sig);

  char err[256];
  Bus* bus = Bus::open(name, w, h, true, fresh, err, sizeof(err));
  if (!bus) {
    fprintf(stderr, "pub: open failed: %s\n", err);
    return 1;
  }
  const RingMeta* m = bus->meta();
  printf("pub: ring %ux%u cam=%u frame=%u slots=%u\n", m->width, m->height,
         m->cam_bytes, m->frame_bytes, kRingDepth);

  const int64_t frame_ms = fps > 0 ? (int64_t)(1000.0 / fps) : 0;
  long published = 0, blocked = 0;
  int64_t t0 = now_ms();
  uint64_t seq = 0;
  while (!g_stop && (n_frames <= 0 || published < n_frames)) {
    int64_t ta = now_ms();
    uint8_t* p = bus->claim(1000);
    if (!p) {
      blocked += 1;
      fprintf(stderr, "pub: claim timeout (holders stuck?)\n");
      continue;
    }
    seq = m->latest_seq + 1;
    for (int c = 0; c < kMaxCams; ++c) {
      uint8_t* y = p + c * m->cam_bytes;
      uint8_t* uv = y + (size_t)w * h;
      memset(y, (int)((seq + c) & 0xFF), (size_t)w * h);
      memset(uv, (int)((seq * 3 + c) & 0xFF), (size_t)w * h / 2);
    }
    FrameMeta meta;
    memset(&meta, 0, sizeof(meta));
    meta.scene_id = (uint32_t)(seq / 40);
    meta.flags = kFlagSourceOk | kFlagAllCamsOk;  // 合成源: 全 OK
    bus->commit(meta);
    published += 1;
    if (published % 60 == 0)
      printf("pub: seq=%lu n=%ld forced=%lu dt=%ldms\n",
             (unsigned long)seq, published,
             (unsigned long)bus->forced_recycles(),
             (long)(now_ms() - ta));
    if (frame_ms > 0) {
      int64_t target = t0 + published * frame_ms;
      int64_t d = target - now_ms();
      if (d > 0) std::this_thread::sleep_for(std::chrono::milliseconds(d));
    }
  }
  printf("pub: done n=%ld blocked_timeouts=%ld forced=%lu elapsed=%ldms\n",
         published, blocked, (unsigned long)bus->forced_recycles(),
         (long)(now_ms() - t0));
  delete bus;
  if (unlink_exit) {
    char full[kNameMax];
    snprintf(full, sizeof(full), "/sp_%s", name);
    shm_unlink(full);
  }
  return 0;
}
