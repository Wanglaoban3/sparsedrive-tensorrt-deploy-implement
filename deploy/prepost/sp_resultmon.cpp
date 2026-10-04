// sp_resultmon - 结果信箱读端验证工具: 附着 seqlock 信箱, 轮询最新结果,
// 校验 magic/crc, 打印摘要; 可选把最新一帧落成 JSON (与节点旁路同格式).
//
// 用法: sp_resultmon <name> [--interval-ms N] [--duration-ms M] [--json P]
//        [--quiet] [--wait-ms M]   只统计不逐帧打印 --quiet;
//        --wait-ms: 信箱尚不存在时等待创建 (下游先于感知节点启动的正常形态)
// 退出码 0: 至少读到一帧完整结果; 1: 没读到; 2: 出现 crc 撕裂.
#include <signal.h>
#include <time.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "sp_bus.h"
#include "sp_result.h"

using namespace sp;
using namespace sp::res;

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

static double mono_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <name> [--interval-ms N] [--duration-ms M] "
                    "[--json P] [--quiet]\n", argv[0]);
    return 2;
  }
  const char* name = argv[1];
  int itv = 100, dur = 0, wait_ms = 0;
  const char* json_out = nullptr;
  bool quiet = false;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--interval-ms") && i + 1 < argc) itv = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--duration-ms") && i + 1 < argc)
      dur = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--json") && i + 1 < argc) json_out = argv[++i];
    else if (!strcmp(argv[i], "--quiet")) quiet = true;
    else if (!strcmp(argv[i], "--wait-ms") && i + 1 < argc)
      wait_ms = atoi(argv[++i]);
  }
  signal(SIGINT, on_sig);
  signal(SIGTERM, on_sig);

  char err[256];
  Mailbox* mb = nullptr;
  double t0 = mono_ms();
  while (!mb) {
    mb = Mailbox::attach(name, err, sizeof(err));
    if (mb) break;
    if (mono_ms() - t0 > wait_ms) {
      fprintf(stderr, "resultmon: %s (waited %dms)\n", err, wait_ms);
      return 1;
    }
    struct timespec ts = {0, 200 * 1000000L};
    nanosleep(&ts, nullptr);
  }
  printf("resultmon: attached sp_res_%s (writes=%lu)\n", name,
         (unsigned long)mb->writes());

  ResultMsg m;
  memset(&m, 0, sizeof(m));
  uint64_t seen_seq = 0, n_new = 0, n_torn = 0;
  double t_end = dur > 0 ? mono_ms() + dur : 1e18;
  bool got = false;
  while (!g_stop && mono_ms() < t_end) {
    ResultMsg tmp;
    if (mb->read_latest(&tmp)) {
      uint32_t crc = msg_crc(tmp, crc32);
      if (crc != tmp.crc) {
        ++n_torn;
        fprintf(stderr, "resultmon: CRC MISMATCH seq=%lu got=%08x want=%08x\n",
                (unsigned long)tmp.seq, crc, tmp.crc);
      } else {
        m = tmp;
        got = true;
        if (m.seq != seen_seq) {
          seen_seq = m.seq;
          ++n_new;
          if (!quiet) {
            float top = 0;
            for (uint32_t i = 0; i < m.n_det; ++i)
              if (m.det[i].score > top) top = m.det[i].score;
            printf("seq=%lu fid=%u scene=%u flags=%u det=%u(map=%u) "
                   "top=%.4f pre=%.2f inf=%.2f post=%.2f e2e=%.2f\n",
                   (unsigned long)m.seq, m.frame_id,
                   m.scene_flags >> 8, m.scene_flags & 0xFF,
                   m.n_det, m.n_map, top, m.pre_ms, m.infer_ms, m.post_ms,
                   m.e2e_ms);
            if (m.version >= 2 && m.t_mp > 0.0f) {
              // v2: plan 摘要 (cmd 槽内 argmax) + final_plan 全 6 点
              int cmd = (int)m.cmd, best = cmd * 6;
              float bv = m.plan_cls[best];
              for (int mi = 1; mi < 6; ++mi)
                if (m.plan_cls[cmd * 6 + mi] > bv) {
                  bv = m.plan_cls[cmd * 6 + mi];
                  best = cmd * 6 + mi;
                }
              printf("  v2 seq=%lu cmd=%u mode=%d conf=%.4f t_mp=%.2f "
                     "plan=[[%.3f,%.3f],[%.3f,%.3f],[%.3f,%.3f],"
                     "[%.3f,%.3f],[%.3f,%.3f],[%.3f,%.3f]]\n",
                     (unsigned long)m.seq, m.cmd, best, bv, m.t_mp,
                     m.final_plan[0][0], m.final_plan[0][1],
                     m.final_plan[1][0], m.final_plan[1][1],
                     m.final_plan[2][0], m.final_plan[2][1],
                     m.final_plan[3][0], m.final_plan[3][1],
                     m.final_plan[4][0], m.final_plan[4][1],
                     m.final_plan[5][0], m.final_plan[5][1]);
            }
          }
          if (m.version >= 3) {
            // v3 fail-visible (M-PROD A4): 状态/年龄/健康计数.
            // age = 发布时冻结的处理时延 (死节点信箱上恒定, 不能单读判活);
            // lage = 读时活年龄 now-ts_capture, 死信箱上持续增长 —— 单读
            // 即可判活的 fail-visible 信号, 下游告警应看它.
            static const char* stn[] = {"NOMINAL", "DEGRADED_RESET",
                                        "DEGRADED_LATCH", "SELFTEST_FAIL"};
            int64_t lage = (now_real_ns() - m.ts_capture_ns) / 1000000;
            if (lage < 0) lage = 0;
            if (lage > 65535) lage = 65535;
            printf("  v3 status=%s reason=%u age=%ums lage=%lldms "
                   "last_valid=%u resets60=%u nan=%u div=%u\n",
                   m.status < 4 ? stn[m.status] : "?", m.reason,
                   m.frame_age_ms, (long long)lage, m.last_valid_seq,
                   m.resets_60s, m.nan_hits, m.div_hits);
          }
          if (json_out) {
            FILE* f = fopen(json_out, "wb");
            if (f) {
              // 最新帧摘要 JSON (检查用; 节点侧逐帧旁路格式同源)
              fprintf(f, "{\"seq\":%lu,\"frame_id\":%u,\"scene\":%u,"
                         "\"flags\":%u,\"n_det\":%u,\"n_map\":%u,"
                         "\"pre_ms\":%.3f,\"infer_ms\":%.3f,"
                         "\"post_ms\":%.3f,\"e2e_ms\":%.3f,"
                         "\"det\":[",
                      (unsigned long)m.seq, m.frame_id, m.scene_flags >> 8,
                      m.scene_flags & 0xFF, m.n_det, m.n_map,
                      m.pre_ms, m.infer_ms, m.post_ms, m.e2e_ms);
              for (uint32_t i = 0; i < m.n_det; ++i)
                fprintf(f, "%s[%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                           "%.6f,%.6f,%d]", i ? "," : "",
                        m.det[i].score, m.det[i].label, m.det[i].x, m.det[i].y,
                        m.det[i].z, m.det[i].w, m.det[i].l, m.det[i].h,
                        m.det[i].yaw, m.det[i].vx, m.det[i].vy, m.det[i].id);
              fprintf(f, "],\"map\":[");
              for (uint32_t i = 0; i < m.n_map; ++i) {
                fprintf(f, "%s[%.6f,%d,[", i ? "," : "", m.map[i].score,
                        m.map[i].label);
                for (int p = 0; p < post::kMapPts; ++p)
                  fprintf(f, "%s[%.6f,%.6f]", p ? "," : "",
                          m.map[i].pts[p][0], m.map[i].pts[p][1]);
                fprintf(f, "]]");
              }
              fprintf(f, "]}\n");
              fclose(f);
            }
          }
        }
      }
    }
    struct timespec ts = {0, itv * 1000000L};
    nanosleep(&ts, nullptr);
  }
  printf("resultmon: new=%lu torn=%lu writes=%lu -> rc=%d\n",
         (unsigned long)n_new, (unsigned long)n_torn,
         (unsigned long)mb->writes(), got ? 0 : 1);
  return !got ? 1 : (n_torn ? 2 : 0);
}
