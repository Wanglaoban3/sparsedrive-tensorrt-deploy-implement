// sp_quota.h — M9a 事件配额仲裁 (header-only).
// 纪律 (ad-data-engine 数据回灌详细设计 §5): 同 log ±5s 贪心去重 /
// 事件类型配额 / 全局预算 2-5% (5fps≈300 帧/min → 默认 15 事件/min).
// 窗口一律按 ts_ns (manifest 采集时, 与离线对拍同源).
#ifndef SP_QUOTA_H_
#define SP_QUOTA_H_
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <string>

struct Quota {
  double dedup_window_s = 5.0;   // 同名事件冷却窗 (±5s 贪心去重)
  int per_event_per_min = 12;    // 单事件类型尾窗 (60s) 配额
  int global_per_min = 15;       // 全局尾窗预算 (2-5% @5fps)
  unsigned long long suppressed = 0;

  // [quota] 段原文 (k=v 行); 未知键忽略
  void configure(const char* kv_text) {
    if (!kv_text) return;
    const char* p = kv_text;
    while (*p) {
      const char* eol = strchr(p, '\n');
      size_t len = eol ? (size_t)(eol - p) : strlen(p);
      if (len > 0 && len < 256) {
        char line[256];
        memcpy(line, p, len);
        line[len] = 0;
        char* eq = strchr(line, '=');
        if (eq) {
          *eq = 0;
          if (!strcmp(line, "dedup_window_s")) dedup_window_s = atof(eq + 1);
          else if (!strcmp(line, "per_event_per_min"))
            per_event_per_min = atoi(eq + 1);
          else if (!strcmp(line, "global_per_min"))
            global_per_min = atoi(eq + 1);
        }
      }
      p = eol ? eol + 1 : p + len;
    }
  }

  // true = 放行 (调用方落盘); false = 仲裁吞掉 (计数器可见)
  bool allow(uint64_t /*seq*/, int64_t ts_ns, const char* name) {
    auto it = last_ns.find(name);
    if (it != last_ns.end() &&
        (double)(ts_ns - it->second) < dedup_window_s * 1e9) {
      ++suppressed;
      return false;
    }
    if (!window_allow(per_ev[name], ts_ns, per_event_per_min)) {
      ++suppressed;
      return false;
    }
    if (!window_allow(global, ts_ns, global_per_min)) {
      ++suppressed;
      return false;
    }
    last_ns[name] = ts_ns;
    return true;
  }

 private:
  std::map<std::string, int64_t> last_ns;
  std::map<std::string, std::deque<int64_t>> per_ev;
  std::deque<int64_t> global;

  static bool window_allow(std::deque<int64_t>& dq, int64_t ts_ns,
                           int limit) {
    const int64_t win = (int64_t)60 * 1000000000ll;
    while (!dq.empty() && dq.front() <= ts_ns - win) dq.pop_front();
    if ((int)dq.size() >= limit) return false;
    dq.push_back(ts_ns);
    return true;
  }
};

#endif  // SP_QUOTA_H_
