// sp_watch.h —— M-PROD A1/A2: 节点内 watchdog + 统一 FATAL 退出
// (spec: docs/superpowers/specs/2026-10-03-mprod-hardening-design.md §5).
//
// watchdog 契约: 主循环每进入一个阶段 watch_touch(stage), 每完成一段
// watch_record(stage, ms) 喂滚动 p50; 独立线程每 20ms 检查"在某阶段停留
// 超过 max(3×p50, 2×p99, 2000ms)" (SP_WD_MS 强制覆盖). M10 分级:
//   tier1 首次超阈 → 只置弃帧请求 (主循环在设防段边界 watch_abandon_take
//         排干: 释放槽引用/跳过本帧输出/计数), 不退出;
//   tier2 请求挂起且主循环超 max(dl/2, 500ms) 未消费 = 排干失败 (真 hang)
//         → FATAL code=13 (detail=watchdog-tier2) → _exit(13).
//   SP_WD_MS 逃生门不分级 (强制覆盖时直退 13).
// 刻意不走任何清理路径: CUDA 出错后 context 沾毒, 析构/TRT destroy 会
// 死锁(项目实测坑); 状态都在 shm, 进程消失即一致. ACQUIRE/IO 不设防 —
// 输入饥饿/文件停顿属于发布端与闪存故障域, 消费端等待是正确行为.
// SP_WD_TEST_STALL=<stage> 故障注入 (第 5 帧该阶段睡 2.5s 后返回, 测
// tier1); SP_WD_TEST_HANG=<stage> 卡死不返回 (测 tier2). 默认关.
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
  std::atomic<int> abandon_req{0};   // M10 tier1: 弃帧请求挂起 (主循环排干)
  std::atomic<int64_t> abandon_ms{0};
  std::mutex mu;
  // 512 帧窗口 (Phase C 遥测 perf 行与 watchdog 共用一份环形)
  static const int kRing = 512;
  double ring[kWsN][kRing] = {{0}};
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

// M10 tier1 排干点: 主循环在设防段边界调用; true = 本帧应被弃掉
// (watchdog 已请求). 取走即清零, 下帧不受影响.
inline bool watch_abandon_take() {
  return watch().abandon_req.exchange(0) != 0;
}

inline void watch_record(int stage, double ms) {
  if (stage < 0 || stage >= kWsN) return;
  WatchState& w = watch();
  std::lock_guard<std::mutex> lk(w.mu);
  w.ring[stage][w.rhead[stage]] = ms;
  w.rhead[stage] = (w.rhead[stage] + 1) % WatchState::kRing;
  if (w.rlen[stage] < WatchState::kRing) w.rlen[stage] += 1;
}

inline double watch_p50(int stage) {
  WatchState& w = watch();
  std::lock_guard<std::mutex> lk(w.mu);
  int n = w.rlen[stage];
  if (!n) return 0.0;
  double tmp[WatchState::kRing];
  memcpy(tmp, w.ring[stage], sizeof(double) * n);
  std::sort(tmp, tmp + n);
  return tmp[n / 2];
}

inline double watch_p99(int stage) {
  WatchState& w = watch();
  std::lock_guard<std::mutex> lk(w.mu);
  int n = w.rlen[stage];
  if (!n) return 0.0;
  double tmp[WatchState::kRing];
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
  // 2.5s: 落在 tier1(dl≈2s 地板) 与 tier2(dl+dl/2=3s) 之间 — 返回型停顿
  // 必须在宽限窗内返回才走弃帧; 睡过 tier2 线就是"真挂死"输入 (FT3 首版
  // 睡 5s 被 tier2 先杀, 属测试输入错而非实现错)
  if (stall_stage >= 0 && stall_stage == stage && frame == 5) {
    printf("watch: TEST_STALL stage=%s sleep 2.5s\n",
           watch_stage_name(stage));
    fflush(stdout);
    struct timespec ts = {2, 500000000};
    nanosleep(&ts, nullptr);
  }
}

// M10: 与 maybe_stall 成对的 tier2 注入钩子 — 卡死不返回 (测"排干失败
// → exit13"路径; stall 返回型只能测 tier1). SP_WD_TEST_HANG=<stage>.
inline void watch_maybe_hang(int stage, long frame) {
  static int hang_stage = -2;
  if (hang_stage == -2) {
    const char* e = getenv("SP_WD_TEST_HANG");
    hang_stage = e ? atoi(e) : -1;
  }
  if (hang_stage >= 0 && hang_stage == stage && frame == 5) {
    printf("watch: TEST_HANG stage=%s loop forever\n",
           watch_stage_name(stage));
    fflush(stdout);
    for (;;) {
      struct timespec ts = {1, 0};
      nanosleep(&ts, nullptr);
    }
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
        // M10 分级: forced 逃生门不分级 (直退); 否则 tier1 = 首次超阈
        // 只请求弃帧 (主循环在设防段返回后排干), tier2 = 请求挂起且
        // 主循环超宽限 (max(dl/2,500ms)) 未消费 = 排干失败 (真 hang)
        // → exit 13.
        if (forced_ms > 0) {
          fprintf(stderr,
                  "FATAL code=13 stage=%s stalled_ms=%ld dl_ms=%ld seq=%lu "
                  "detail=watchdog\n",
                  watch_stage_name(st), (long)stalled, (long)dl,
                  (unsigned long)watch_seq_get());
          fflush(stderr);
          _exit(13);
        }
        if (watch().abandon_req.exchange(1) == 0) {
          watch().abandon_ms.store(watch_now_ms());
          fprintf(stderr,
                  "watch: ABANDON req stage=%s stalled_ms=%ld dl_ms=%ld "
                  "seq=%lu detail=watchdog-tier1\n",
                  watch_stage_name(st), (long)stalled, (long)dl,
                  (unsigned long)watch_seq_get());
          fflush(stderr);
        } else {
          int64_t grace = dl / 2 > 500 ? dl / 2 : 500;
          if (stalled > dl + grace) {
            fprintf(stderr,
                    "FATAL code=13 stage=%s stalled_ms=%ld dl_ms=%ld "
                    "seq=%lu detail=watchdog-tier2\n",
                    watch_stage_name(st), (long)stalled, (long)dl,
                    (unsigned long)watch_seq_get());
            fflush(stderr);
            _exit(13);
          }
        }
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
