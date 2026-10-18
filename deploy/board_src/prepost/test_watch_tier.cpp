// test_watch_tier.cpp — M10 Task 3 RED/GREEN: watchdog 分级.
// 用例 (父进程 fork 子进程逐个驱动):
//   1 tier1: 设防段停 2.6s (dl=2000ms 被触发) → 子进程收到弃帧请求
//     (take==true) 且排干后重 touch → 睡过原 tier2 死线 (3s) 仍活着,
//     正常 exit 0. 旧代码 (无分级) 会在 2s 直接 _exit(13) → 抓住.
//   2 tier2: 永不排干 (touch 后死睡) → dl+grace=3s 处 _exit(13).
//   3 逃生门: SP_WD_MS=1000 → ~1s 直退 13, 不分级.
//   4 注入钩子: SP_WD_TEST_HANG=post → watch_maybe_hang 死循环不返回.
// 退出码 0 = 4/4.
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "sp_watch.h"

using namespace sp;
using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

static void sleep_ms(int64_t ms) {
  struct timespec ts = {0, 100 * 1000 * 1000L};
  int64_t left = ms;
  while (left > 0) {
    long chunk = left > 100 ? 100 : (long)left;
    ts.tv_nsec = chunk * 1000 * 1000L;
    nanosleep(&ts, nullptr);
    left -= chunk;
  }
}

// 子进程退出语义: 0= PASS 断言成立; 5=断言败; 13=watchdog FATAL (透传)
static int run_case(int no) {
  pid_t p = fork();
  if (p == 0) {
    if (no == 1) {
      watch_start();
      watch_touch(kWsPre);  // 无 record → dl=2000ms 地板
      sleep_ms(2600);       // 停 2.6s: abandon 在 ~2.0-2.3s 置位
      bool got = watch_abandon_take();
      if (!got) _exit(5);
      watch_touch(kWsPre);  // 排干点: 下一帧重新武装 (beat 刷新)
      sleep_ms(1500);       // 睡过 3.0s 原 tier2 死线: 必须被解除
      _exit(0);
    }
    if (no == 2) {
      watch_start();
      watch_touch(kWsPre);
      for (;;) sleep_ms(1000);  // 永不排干 → tier2
    }
    if (no == 3) {  // 父进程已 setenv SP_WD_MS=1000
      watch_start();
      watch_touch(kWsPre);
      for (;;) sleep_ms(1000);
    }
    if (no == 4) {  // 父进程已 setenv SP_WD_TEST_HANG=4(post)
      watch_maybe_hang(kWsPost, 5);
      _exit(0);  // 到不了
    }
    _exit(9);
  }
  return p;
}

int main() {
  int passed = 0;
  // ---- Case 1 ----
  {
    clk::time_point t0 = clk::now();
    pid_t p = run_case(1);
    int st = 0;
    waitpid(p, &st, 0);
    double took = ms_since(t0);
    bool ok = WIFEXITED(st) && WEXITSTATUS(st) == 0 && took > 3900;
    if (ok) { printf("PASS 1 (tier1 drained, alive %.1fs)\n", took / 1e3); ++passed; }
    else printf("FAIL 1 (exit=%d signaled=%d took=%.1fs)\n",
                WIFEXITED(st) ? WEXITSTATUS(st) : -1, WIFSIGNALED(st),
                took / 1e3);
  }
  // ---- Case 2 ----
  {
    clk::time_point t0 = clk::now();
    pid_t p = run_case(2);
    int st = 0;
    waitpid(p, &st, 0);
    double took = ms_since(t0);
    bool ok = WIFEXITED(st) && WEXITSTATUS(st) == 13 &&
              took > 2800 && took < 4600;
    if (ok) { printf("PASS 2 (tier2 exit13 at %.1fs)\n", took / 1e3); ++passed; }
    else printf("FAIL 2 (exit=%d took=%.1fs)\n",
                WIFEXITED(st) ? WEXITSTATUS(st) : -1, took / 1e3);
  }
  // ---- Case 3 ----
  {
    setenv("SP_WD_MS", "1000", 1);
    clk::time_point t0 = clk::now();
    pid_t p = run_case(3);
    int st = 0;
    waitpid(p, &st, 0);
    double took = ms_since(t0);
    unsetenv("SP_WD_MS");
    bool ok = WIFEXITED(st) && WEXITSTATUS(st) == 13 &&
              took > 900 && took < 2100;
    if (ok) { printf("PASS 3 (forced exit13 at %.1fs)\n", took / 1e3); ++passed; }
    else printf("FAIL 3 (exit=%d took=%.1fs)\n",
                WIFEXITED(st) ? WEXITSTATUS(st) : -1, took / 1e3);
  }
  // ---- Case 4 ----
  {
    setenv("SP_WD_TEST_HANG", "4", 1);
    pid_t p = run_case(4);
    sleep_ms(3000);
    bool hung = (kill(p, 0) == 0);
    kill(p, SIGKILL);
    int st = 0;
    waitpid(p, &st, 0);
    unsetenv("SP_WD_TEST_HANG");
    if (hung && WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL) {
      printf("PASS 4 (maybe_hang never returns)\n");
      ++passed;
    } else {
      printf("FAIL 4 (hung=%d signaled=%d)\n", hung, WIFSIGNALED(st));
    }
  }
  printf("== %d/4 ==\n", passed);
  return passed == 4 ? 0 : 1;
}
