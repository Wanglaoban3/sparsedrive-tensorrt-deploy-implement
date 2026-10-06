// test_bus_lease.cpp — M10 Task 1 RED/GREEN: wrap-lease 定向强抢.
// 断言见 main() 内 PASS 1..4; 退出码 0 = 4/4 全过.
//
// 单进程双角色安全性: BusLock 按 pid 记属主且不可重入 (sp_bus.cpp lock_lk
// owner==me 时自旋), 故发布/消费操作必须串行; 唯一的并发是 case 3 的心跳
// 线程 —— heartbeat() 只写原子不加锁 (sp_bus.cpp:356), 合法.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "sp_bus.h"

using namespace sp;
using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// 场景: 发布 1 帧被消费者持有, 再发布 3 帧让 pub_idx 绕回持有槽.
struct Held {
  Bus* bus;
  int32_t cid;
  FrameView v;
};

static Held hold_one(const char* ring) {
  char err[256];
  Bus* b = Bus::open(ring, 64, 64, true, true, err, sizeof(err));
  if (!b) { fprintf(stderr, "open %s: %s\n", ring, err); exit(2); }
  uint8_t* p = b->claim_of(nullptr, 1000);
  if (!p) { fprintf(stderr, "claim0 failed\n"); exit(2); }
  memset(p, 0xAB, 64 * 64 * 3 / 2 * 6);
  FrameMeta m{};
  m.scene_id = 1;
  b->commit(m);  // seq=1, pub_idx 0->1
  Held h;
  h.bus = b;
  h.cid = b->register_consumer();
  if (h.cid < 0) { fprintf(stderr, "register failed\n"); exit(2); }
  if (b->acquire(h.cid, 0, &h.v, 2000) != 0) {
    fprintf(stderr, "acquire failed\n"); exit(2);
  }
  // 绕回: 再发 3 帧占满槽 1..3, pub_idx 回到消费者持有的槽 0
  for (int i = 0; i < 3; ++i) {
    uint8_t* q = b->claim_of(nullptr, 1000);
    if (!q) { fprintf(stderr, "claim%d failed\n", i + 1); exit(2); }
    b->commit(m);
  }
  return h;
}

int main() {
  int passed = 0;
  // ---- Case 1: 卡死持有者 + WRAP=300 → 定向强抢, claim 快速返回 ----
  // ---- Case 4: 被抢 view 的重复 release 幂等 (ref 已 0 再减会下溢回绕,
  //              环立刻卡死 — 用"还能正常 claim"做行为级断言) ----
  {
    setenv("SP_BUS_WRAP_LEASE_MS", "300", 1);
    Held h = hold_one("wltest1");
    std::this_thread::sleep_for(std::chrono::milliseconds(400));  // hb 老 >300
    clk::time_point t0 = clk::now();
    uint8_t* p = h.bus->claim_of(nullptr, 1500);
    double took = ms_since(t0);
    if (p && took < 600 && h.bus->forced_recycles() == 1) {
      printf("PASS 1 (steal took=%.0fms forced=%lu)\n", took,
             (unsigned long)h.bus->forced_recycles());
      ++passed;
      FrameMeta m{}; h.bus->commit(m);  // 抢到后能正常发布 (pub_idx 前进)
    } else {
      printf("FAIL 1 (p=%p took=%.0fms forced=%lu)\n", (void*)p, took,
             (unsigned long)h.bus->forced_recycles());
    }
    h.bus->release(&h.v);
    h.bus->release(&h.v);  // 第二次必须无害 (release 幂等护栏)
    uint8_t* q = h.bus->claim_of(nullptr, 1000);
    if (q) {
      printf("PASS 4 (idempotent release, ring alive)\n");
      ++passed;
      FrameMeta m{}; h.bus->commit(m);
    } else {
      printf("FAIL 4 (ring stuck after double release)\n");
    }
    delete h.bus;
  }
  // ---- Case 2: WRAP 未设 (=0) → 5s flat lease, 400ms 不抢 ----
  {
    unsetenv("SP_BUS_WRAP_LEASE_MS");
    Held h = hold_one("wltest2");
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    clk::time_point t0 = clk::now();
    uint8_t* p = h.bus->claim_of(nullptr, 1000);
    double took = ms_since(t0);
    if (!p && took >= 950 && h.bus->forced_recycles() == 0) {
      printf("PASS 2 (blocked %.0fms, no steal)\n", took);
      ++passed;
    } else {
      printf("FAIL 2 (p=%p took=%.0fms forced=%lu)\n", (void*)p, took,
             (unsigned long)h.bus->forced_recycles());
    }
    delete h.bus;
  }
  // ---- Case 3: 活持有者 (50ms 心跳) + WRAP=300 → 不误抢 ----
  {
    setenv("SP_BUS_WRAP_LEASE_MS", "300", 1);
    Held h = hold_one("wltest3");
    std::atomic<bool> live{true};
    std::thread hb([&]() {
      while (live.load()) {
        h.bus->heartbeat(h.cid);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    });
    clk::time_point t0 = clk::now();
    uint8_t* p = h.bus->claim_of(nullptr, 1000);
    double took = ms_since(t0);
    live.store(false);
    hb.join();
    if (!p && took >= 950 && h.bus->forced_recycles() == 0) {
      printf("PASS 3 (live holder kept, blocked %.0fms)\n", took);
      ++passed;
    } else {
      printf("FAIL 3 (p=%p took=%.0fms forced=%lu)\n", (void*)p, took,
             (unsigned long)h.bus->forced_recycles());
    }
    delete h.bus;
  }
  printf("== %d/4 ==\n", passed);
  return passed == 4 ? 0 : 1;
}
