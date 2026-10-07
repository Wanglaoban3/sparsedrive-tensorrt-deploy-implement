// sp_status - M-PROD Phase C 一行概览工具 (spec §7 C1):
// ring 满度/forced_recycles + 信箱 status/age/writes + 节点 RSS/uptime
// + 温度/时钟. 供人工巡检与 soak 采样; --json 供脚本, --csv 单行逗号分隔
// 供板端 bash 采样 (无引号无内嵌逗号, awk 可切).
//
// 用法: sp_status <ring> [--json|--csv] [--wait-ms M]
// 退出码 0 = 全部读到; 1 = ring/信箱附着失败 (部分信息也尽力打印).
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "sp_bus.h"
#include "sp_result.h"

using namespace sp;
using namespace sp::res;

// ring 头快照: 单次 pread RingMeta (状态工具可容忍发布中的微小撕裂;
// 不走 Bus::open —— attach 有几何校验, 而本工具不该要求调用方传宽高)
static bool read_ring_meta(const char* ring, RingMeta* out, char* err,
                           size_t errlen) {
  char path[96];
  snprintf(path, sizeof(path), "/sp_%s", ring);
  int fd = shm_open(path, O_RDWR, 0600);
  if (fd < 0) {
    snprintf(err, errlen, "shm_open(%s): errno=%d", path, errno);
    return false;
  }
  bool ok = pread(fd, out, sizeof(*out), 0) == (ssize_t)sizeof(*out) &&
            out->magic == sp::kMagic;
  if (!ok) snprintf(err, errlen, "bad ring header %s", path);
  close(fd);
  return ok;
}

struct ProcInfo {
  bool found = false;
  int pid = 0;
  long rss_kb = 0;
  double uptime_s = 0;
};

// /proc 扫描: comm 精确匹配 name 的第一个进程; RSS + 基于 starttime 的
// 进程 uptime (alive 即单调增长, soak 斜率的观测量)
static ProcInfo find_proc(const char* name) {
  ProcInfo p;
  DIR* d = opendir("/proc");
  if (!d) return p;
  struct dirent* e;
  while ((e = readdir(d))) {
    if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
    char path[128], comm[64] = "";
    snprintf(path, sizeof(path), "/proc/%s/comm", e->d_name);
    FILE* f = fopen(path, "r");
    if (!f) continue;
    if (!fgets(comm, sizeof(comm), f)) {
      fclose(f);
      continue;
    }
    fclose(f);
    comm[strcspn(comm, "\n")] = 0;
    if (strcmp(comm, name)) continue;
    p.pid = atoi(e->d_name);
    p.found = true;
    snprintf(path, sizeof(path), "/proc/%s/status", e->d_name);
    f = fopen(path, "r");
    if (f) {
      char ln[256];
      while (fgets(ln, sizeof(ln), f))
        if (sscanf(ln, "VmRSS: %ld kB", &p.rss_kb) == 1) break;
      fclose(f);
    }
    // starttime (field 22, boot 后时钟节拍) → 进程存活时长
    long hz = sysconf(_SC_CLK_TCK);
    snprintf(path, sizeof(path), "/proc/%s/stat", e->d_name);
    f = fopen(path, "r");
    if (f) {
      // comm 可能含空格/括号: 从最后一个 ')' 之后数
      char buf[2048];
      size_t n = fread(buf, 1, sizeof(buf) - 1, f);
      buf[n] = 0;
      fclose(f);
      char* q = strrchr(buf, ')');
      if (q) {
        long long starttime = 0;
        if (sscanf(q + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u "
                          "%*u %*u %*d %*d %*d %*d %*d %*d %lld",
                   &starttime) == 1 && hz > 0) {
          double up = 0;
          f = fopen("/proc/uptime", "r");
          if (f) {
            if (fscanf(f, "%lf", &up) != 1) up = 0;
            fclose(f);
          }
          p.uptime_s = up - (double)starttime / hz;
          if (p.uptime_s < 0) p.uptime_s = 0;
        }
      }
    }
    break;
  }
  closedir(d);
  return p;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <ring> [--json] [--wait-ms M]\n", argv[0]);
    return 2;
  }
  const char* ring = argv[1];
  bool json = false, csv = false;
  int wait_ms = 0;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--json")) json = true;
    else if (!strcmp(argv[i], "--csv")) csv = true;
    else if (!strcmp(argv[i], "--wait-ms") && i + 1 < argc)
      wait_ms = atoi(argv[++i]);
  }

  char err[256];
  RingMeta rm{};  // 0 值初始化 (非平凡类型不能用 memset)
  bool have_ring = false;
  double t0 = now_ms();
  while (!(have_ring = read_ring_meta(ring, &rm, err, sizeof(err)))) {
    if (now_ms() - t0 > wait_ms) break;
    struct timespec ts = {0, 200 * 1000000L};
    nanosleep(&ts, nullptr);
  }
  Mailbox* mb = nullptr;
  t0 = now_ms();
  while (!(mb = Mailbox::attach(std::string("sp_result_").append(ring).c_str(),
                                err, sizeof(err)))) {
    if (now_ms() - t0 > wait_ms) break;
    struct timespec ts = {0, 200 * 1000000L};
    nanosleep(&ts, nullptr);
  }
  ProcInfo node = find_proc("sp_modelnode");
  ProcInfo fs = find_proc("sp_filesrc");

  // 温度/时钟与节点采样同源 (读不到 = -1 → json null / 文本 n/a)
  auto read_thermal = [&](const char* key) {
    DIR* d = opendir("/sys/class/thermal");
    if (!d) return -1;
    int out = -1;
    struct dirent* e;
    while ((e = readdir(d))) {
      if (strncmp(e->d_name, "thermal_zone", 12)) continue;
      char p[160], ty[64] = "";
      snprintf(p, sizeof(p), "/sys/class/thermal/%s/type", e->d_name);
      FILE* f = fopen(p, "r");
      if (!f) continue;
      if (!fgets(ty, sizeof(ty), f)) {
        fclose(f);
        continue;
      }
      fclose(f);
      if (!strstr(ty, key)) continue;
      snprintf(p, sizeof(p), "/sys/class/thermal/%s/temp", e->d_name);
      f = fopen(p, "r");
      if (!f) break;
      int mc = -1;
      if (fscanf(f, "%d", &mc) == 1 && mc > 0) out = mc / 1000;
      fclose(f);
      break;
    }
    closedir(d);
    return out;
  };
  int gt = read_thermal("GPU"), ct = read_thermal("CPU");
  int ck = -1;
  {
    DIR* d = opendir("/sys/class/devfreq");
    if (d) {
      struct dirent* e;
      while ((e = readdir(d))) {
        if (!strstr(e->d_name, "ga10b")) continue;
        char p[192];
        snprintf(p, sizeof(p), "/sys/class/devfreq/%s/cur_freq", e->d_name);
        FILE* f = fopen(p, "r");
        if (!f) break;
        long hz = 0;
        if (fscanf(f, "%ld", &hz) == 1 && hz > 0) ck = (int)(hz / 1000000);
        fclose(f);
        break;
      }
      closedir(d);
    }
  }

  // 信箱最新帧 (读不到 = 上游从未发布或信箱不存在)
  ResultMsg m;
  memset(&m, 0, sizeof(m));
  bool have_msg = mb && mb->read_latest(&m);
  int64_t lage_ms =
      have_msg ? (now_real_ns() - m.ts_capture_ns) / 1000000 : -1;
  if (lage_ms < 0) lage_ms = 0;
  static const char* stn[] = {"NOMINAL", "DEGRADED_RESET", "DEGRADED_LATCH",
                              "SELFTEST_FAIL"};

  uint64_t latest = have_ring ? rm.latest_seq.load() : 0;
  uint32_t pub_idx = have_ring ? rm.pub_idx : 0;
  uint64_t published = have_ring ? rm.published_count : 0;
  // 满度 = 当前被消费者持引用的槽位数 (逐消费者 held_slot 统计)
  int held = 0;
  int nc = have_ring ? (int)rm.n_consumers.load() : 0;
  if (nc > kMaxConsumers) nc = kMaxConsumers;
  for (int i = 0; i < nc; ++i)
    if (rm.cons[i].held_slot.load() >= 0) ++held;
  uint64_t forced = have_ring ? rm.forced_recycles : 0;

  if (csv) {
    // 单行 CSV (无引号无内嵌逗号; 板端 bash 采样: $(sp_status m3 --csv))
    char b1[16], b2t[16], b3[16];
    auto na = [](char* b, int v) {
      if (v < 0) snprintf(b, 16, "n/a");
      else snprintf(b, 16, "%d", v);
    };
    na(b1, gt);
    na(b2t, ct);
    na(b3, ck);
    printf("%lu,%lu,%u,%d,%d,%lu,%d,%s,%u,%lld,%lu,%u,%u,%u,%u,"
           "%d,%ld,%.1f,%d,%ld,%.1f,%s,%s,%s\n",
           (unsigned long)latest, (unsigned long)published, pub_idx, held,
           kRingDepth, (unsigned long)forced, nc,
           have_msg ? (m.status < 4 ? stn[m.status] : "?") : "NO_MSG",
           have_msg ? m.frame_age_ms : 0U, (long long)lage_ms,
           mb ? (unsigned long)mb->writes() : 0UL,
           have_msg ? m.last_valid_seq : 0U, have_msg ? m.resets_60s : 0U,
           have_msg ? m.nan_hits : 0U, have_msg ? m.div_hits : 0U,
           node.pid, node.rss_kb, node.uptime_s, fs.pid, fs.rss_kb,
           fs.uptime_s, b1, b2t, b3);
  } else if (json) {
    printf("{\"ring\":\"%s\",\"latest_seq\":%lu,\"published\":%lu,"
           "\"pub_idx\":%u,\"slots_held\":%d,\"ring_depth\":%d,"
           "\"forced_recycles\":%lu,\"n_consumers\":%d,"
           "\"mbox_status\":\"%s\",\"mbox_seq\":%lu,\"age_ms\":%u,"
           "\"lage_ms\":%lld,\"mbox_writes\":%lu,\"last_valid_seq\":%u,"
           "\"resets_60s\":%u,\"nan_hits\":%u,\"div_hits\":%u,"
           "\"node_pid\":%d,\"node_rss_kb\":%ld,\"node_uptime_s\":%.1f,"
           "\"fs_pid\":%d,\"fs_rss_kb\":%ld,\"fs_uptime_s\":%.1f,"
           "\"gpu_temp_c\":%s,\"cpu_temp_c\":%s,\"sm_clock_mhz\":%s}\n",
           ring, (unsigned long)latest, (unsigned long)published, pub_idx,
           held, kRingDepth, (unsigned long)forced, nc,
           have_msg ? (m.status < 4 ? stn[m.status] : "?") : "NO_MSG",
           have_msg ? (unsigned long)m.seq : 0UL,
           have_msg ? m.frame_age_ms : 0U, (long long)lage_ms,
           mb ? (unsigned long)mb->writes() : 0UL,
           have_msg ? m.last_valid_seq : 0U, have_msg ? m.resets_60s : 0U,
           have_msg ? m.nan_hits : 0U, have_msg ? m.div_hits : 0U,
           node.pid, node.rss_kb, node.uptime_s, fs.pid, fs.rss_kb,
           fs.uptime_s,
           gt < 0 ? "null" : std::to_string(gt).c_str(),
           ct < 0 ? "null" : std::to_string(ct).c_str(),
           ck < 0 ? "null" : std::to_string(ck).c_str());
  } else {
    char b1[16], b2[16], b3[16];
    auto na = [](char* b, int v) {
      if (v < 0) snprintf(b, 16, "n/a");
      else snprintf(b, 16, "%d", v);
    };
    na(b1, gt);
    na(b2, ct);
    na(b3, ck);
    printf("ring=%s seq=%lu pub=%lu slots=%d/%d cons=%d forced=%lu | "
           "mbox=%s seq=%lu age=%ums lage=%lldms writes=%lu last_valid=%u "
           "resets60=%u nan=%u div=%u | node pid=%d rss=%ldkB up=%.0fs | "
           "fs pid=%d rss=%ldkB up=%.0fs | gpu=%sc cpu=%sc clk=%sMHz\n",
           ring, (unsigned long)latest, (unsigned long)published, held,
           kRingDepth, nc, (unsigned long)forced,
           have_msg ? (m.status < 4 ? stn[m.status] : "?") : "NO_MSG",
           have_msg ? (unsigned long)m.seq : 0UL,
           have_msg ? m.frame_age_ms : 0U, (long long)lage_ms,
           mb ? (unsigned long)mb->writes() : 0UL,
           have_msg ? m.last_valid_seq : 0U, have_msg ? m.resets_60s : 0U,
           have_msg ? m.nan_hits : 0U, have_msg ? m.div_hits : 0U, node.pid,
           node.rss_kb, node.uptime_s, fs.pid, fs.rss_kb, fs.uptime_s,
           b1, b2, b3);
  }
  return (have_ring && mb) ? 0 : 1;
}
