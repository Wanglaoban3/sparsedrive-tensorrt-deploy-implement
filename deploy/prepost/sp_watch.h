// sp_watch.h —— M-PROD A1/A2: 节点内 watchdog + 统一 FATAL 退出
// (spec: docs/superpowers/specs/2026-10-03-mprod-hardening-design.md §5).
//
// watchdog 契约: 主循环每进入一个阶段 watch_touch(stage), 每完成一段
// watch_record(stage, ms) 喂滚动 p50; 独立线程每 20ms 检查"在某阶段停留
// 超过 max(3×p50, 500ms)" → FATAL code=13 → _exit(13). 刻意不走任何
// 清理路径: CUDA 出错后 context 沾毒, 析构/TRT destroy 会死锁(项目实测
// 坑); 状态都在 shm, 进程消失即一致. ACQUIRE 不设防 —— 输入饥饿属于
// 发布端故障域(filesrc exit 20 + systemd 编排), 消费端等待是正确行为.
// SP_WD_MS=<ms> 强制覆盖 deadline; SP_WD_TEST_STALL=<stage> 故障注入钩子
// (第 5 帧进入该阶段时睡 5s, 默认关).
//
// fatal_exit: 全链路统一异常退出 (A2 退出码契约), 打一行可解析的
// "FATAL code= stage= seq= detail=" 后 _exit, 供 systemd/日志采集锚定.
#ifndef SP_WATCH_H_
#define SP_WATCH_H_

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <time.h>
#include <unistd.h>

namespace sp {

enum {
  kWsPre = 0,
  kWsBB2 = 1,
  kWsHD = 2,
  kWsMP = 3,
  kWsPost = 4,
  kWsAcq = 5,  // 不设防
  kWsIo = 6,   // 不设防: 文件读/落盘/解码/发布 (闪存 GC 停顿秒级, 非 hang)
  kWsN
};

inline const char* watch_stage_name(int s) {
  static const char* nm[kWsN] = {"pre", "bb2", "hd", "mp", "post", "acquire",
                                 "io"};
  return (s >= 0 && s < kWsN) ? nm[s] : "?";
}

inline int64_t watch_now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return int64_t(ts.tv_sec) * 1000 + ts.tv_nsec / 1000000;
}

struct WatchState {
  std::atomic<int64_t> beat_ms{0};
  std::atomic<int> stage{kWsAcq};
  std::atomic<uint64_t> seq{0};
  std::mutex mu;
  double ring[kWsN][64] = {{0}};
  int rhead[kWsN] = {0};
  int rlen[kWsN] = {0};
};

inline WatchState& watch() {
  static WatchState s;
  return s;
}

inline void watch_touch(int stage) {
  watch().stage.store(stage, std::memory_order_relaxed);
  watch().beat_ms.store(watch_now_ms(), std::memory_order_relaxed);
}

inline void watch_seq(uint64_t s) { watch().seq.store(s); }
inline uint64_t watch_seq_get() { return watch().seq.load(); }
inline int watch_cur_stage() { return watch().stage.load(); }

inline void watch_record(int stage, double ms) {
  if (stage < 0 || stage >= kWsN) return;
  WatchState& w = watch();
  std::lock_guard<std::mutex> lk(w.mu);
  w.ring[stage][w.rhead[stage]] = ms;
  w.rhead[stage] = (w.rhead[stage] + 1) % 64;
  if (w.rlen[stage] < 64) w.rlen[stage] += 1;
}

inline double watch_p50(int stage) {
  WatchState& w = watch();
  std::lock_guard<std::mutex> lk(w.mu);
  int n = w.rlen[stage];
  if (!n) return 0.0;
  double tmp[64];
  memcpy(tmp, w.ring[stage], sizeof(double) * n);
  std::sort(tmp, tmp + n);
  return tmp[n / 2];
}

inline double watch_p99(int stage) {
  WatchState& w = watch();
  std::lock_guard<std::mutex> lk(w.mu);
  int n = w.rlen[stage];
  if (!n) return 0.0;
  double tmp[64];
  memcpy(tmp, w.ring[stage], sizeof(double) * n);
  std::sort(tmp, tmp + n);
  return tmp[(int)((n - 1) * 0.99)];
}

inline void watch_maybe_stall(int stage, long frame) {
  static int stall_stage = -2;  // -2 = env 未解析
  if (stall_stage == -2) {
    const char* e = getenv("SP_WD_TEST_STALL");
    stall_stage = e ? atoi(e) : -1;
  }
  if (stall_stage >= 0 && stall_stage == stage && frame == 5) {
    printf("watch: TEST_STALL stage=%s sleep 5s\n", watch_stage_name(stage));
    fflush(stdout);
    struct timespec ts = {5, 0};
    nanosleep(&ts, nullptr);
  }
}

inline void watch_start() {
  // init 窗口 (attach 重试/信箱创建等) 不设防: 线程启动即置 IO disarm,
  // 主循环第一个 watch_touch 才进入武装段 (否则 cur_stage 停留 0=armed,
  // attach 慢一次就 500ms 误杀)
  watch().stage.store(kWsIo, std::memory_order_relaxed);
  std::thread([]() {
    static int64_t forced_ms = []() {
      const char* e = getenv("SP_WD_MS");
      return e && atol(e) > 0 ? atol(e) : 0;
    }();
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      int st = watch().stage.load(std::memory_order_relaxed);
      if (st == kWsAcq || st == kWsIo) continue;
      // dl = max(3×p50, 2×p99, 2000ms): 设防段只含 GPU 同步/提交, 但 iGPU
      // 与车厂其他进程共卡, 百毫秒级调度抖动是常态 (实测 post 段一次
      // 503ms 抖动撞 500ms 地板误杀); 2s 停滞 @5fps=丢 10 帧, 仍是"挂死"
      // 量级. 文件 IO 已由 kWsIo 不设防段盖住 (闪存 GC 停顿 0.5-2s).
      int64_t dl = (int64_t)(watch_p50(st) * 3.0);
      int64_t p99 = (int64_t)(watch_p99(st) * 2.0);
      if (p99 > dl) dl = p99;
      if (forced_ms > 0) dl = forced_ms;
      if (dl < 2000) dl = 2000;
      int64_t stalled = watch_now_ms() - watch().beat_ms.load();
      if (stalled > dl) {
        fprintf(stderr,
                "FATAL code=13 stage=%s stalled_ms=%ld dl_ms=%ld seq=%lu "
                "detail=watchdog\n",
                watch_stage_name(st), (long)stalled, (long)dl,
                (unsigned long)watch_seq_get());
        fflush(stderr);
        _exit(13);
      }
    }
  }).detach();
}

// A2: 统一异常退出. 只打结构化日志 + _exit, 绝不清理 (见文件头).
inline void fatal_exit(int code, const char* stage, const char* fmt, ...) {
  char detail[256];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(detail, sizeof(detail), fmt, ap);
  va_end(ap);
  fprintf(stderr, "FATAL code=%d stage=%s seq=%lu detail=%s\n", code, stage,
          (unsigned long)watch_seq_get(), detail);
  fflush(stderr);
  fflush(stdout);
  _exit(code);
}

}  // namespace sp

#endif  // SP_WATCH_H_
