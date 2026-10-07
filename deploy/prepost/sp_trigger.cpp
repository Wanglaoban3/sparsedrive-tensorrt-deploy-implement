// sp_trigger.cpp — M9a: L1 规则触发器宿主 (CPU, 只读结果信箱, 零 GPU/DLA).
// spec: docs/superpowers/specs/2026-10-06-m9-multimodel-dla-design.md §4/§4.2.
// 10Hz tick: 读 sp_result_<ring> 信箱 (seqlock) → 按 seq join manifest 取
// l2g/ts → sp_egoring 差分 ego 运动学 (events.py 口径, 时间窗按 ts 实差) →
// sp_frame_ctx (64 帧历史) → 逐规则 eval → sp_quota 去重/配额 → events.jsonl
// (10MB×5 轮转). ctx-dump 模式落逐 tick 二进制夹具 (§7.5, 布局钉死).
// 故障域 (spec §6): 本进程崩溃只暂停数据飞轮, systemd 拉起, 关键链无感.
#include <errno.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "sp_egoring.h"
#include "sp_quota.h"
#include "sp_result.h"
#include "sp_rule.h"
#include "sp_rule_util.h"
#include "sp_ruleload.h"

static std::atomic<int> g_stop{0};
static std::atomic<int> g_rescan{0};
static void on_term(int) { g_stop.store(1); }
static void on_hup(int) { g_rescan.store(1); }

static std::string env_s(const char* k, const char* dflt) {
  const char* e = getenv(k);
  return e && e[0] ? e : dflt;
}
static double env_d(const char* k, double dflt) {
  const char* e = getenv(k);
  return e && e[0] ? atof(e) : dflt;
}

// ---------- 迷你 manifest 读取 (只取 ts_ns/scene/l2g; 键序与节点同源) ----------
struct TrigFrame {
  uint64_t ts_ns;
  uint32_t scene;
  double l2g[16];
};

static const char* jump_num(const char* p) {  // 跳过 JSON 结构字符取数字
  while (*p && (*p == ' ' || *p == '\t' || *p == ':' || *p == ',' ||
                *p == '"' || *p == '[' || *p == ']'))
    ++p;
  return p;
}

static bool parse_manifest_min(const char* path, std::vector<TrigFrame>* out) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  std::string all;
  char buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), f)) > 0) all.append(buf, n);
  fclose(f);
  size_t pos = all.find('\n');  // 首行是 header
  if (pos == std::string::npos) return false;
  while (pos < all.size()) {
    size_t eol = all.find('\n', pos + 1);
    if (eol == std::string::npos) eol = all.size();
    std::string line = all.substr(pos, eol - pos);
    pos = eol;
    if (line.size() < 3) continue;
    TrigFrame fm;
    memset(&fm, 0, sizeof(fm));
    size_t k = line.find("\"ts_ns\"");
    if (k == std::string::npos) return false;
    fm.ts_ns = strtoull(jump_num(line.c_str() + k + 7), 0, 10);
    k = line.find("\"scene\"");
    if (k == std::string::npos) return false;
    fm.scene = (uint32_t)strtoul(jump_num(line.c_str() + k + 7), 0, 10);
    k = line.find("\"l2g\"");
    if (k == std::string::npos) return false;
    const char* p = jump_num(line.c_str() + k + 5);
    for (int i = 0; i < 16; ++i) {
      char* end = 0;
      fm.l2g[i] = strtod(p, &end);  // endptr 必须拿: 否则 p 不前进全读同值
      p = jump_num(end);
    }
    out->push_back(fm);
  }
  return !out->empty();
}

// ---------- 事件流 (jsonl, 10MB×5 轮转) ----------
static const size_t kEvRotateBytes = 10ull * 1024 * 1024;
static FILE* g_ev = 0;
static std::string g_ev_path;
static unsigned long long g_ev_total = 0;

static void ev_open(const std::string& path) {
  g_ev_path = path;
  g_ev = fopen(path.c_str(), "a");
}

static void ev_rotate() {
  if (!g_ev) return;
  fclose(g_ev);
  g_ev = 0;
  for (int i = 4; i >= 1; --i) {
    std::string src = (i == 1) ? g_ev_path
                               : g_ev_path + "." + std::to_string(i - 1);
    std::string dst = g_ev_path + "." + std::to_string(i);
    rename(src.c_str(), dst.c_str());
  }
  g_ev = fopen(g_ev_path.c_str(), "a");
}

static void ev_write(uint64_t seq, int64_t ts_ns, const char* name,
                     float strength) {
  if (!g_ev) return;
  fprintf(g_ev,
          "{\"ts_ns\":%lld,\"seq\":%llu,\"event\":\"%s\",\"strength\":%.6f}\n",
          (long long)ts_ns, (unsigned long long)seq, name, (double)strength);
  g_ev_total += 1;
  if (ftell(g_ev) > (long)kEvRotateBytes) {
    ev_rotate();
    fprintf(stderr, "trigger: events.jsonl rotated (%llu total)\n",
            g_ev_total);
  }
}

// ---------- 观察模式浮点 json 字段缓冲 (一次 fprintf 多处引用, 轮转池) ----------
static const char* buf_f(double v) {
  static char bufs[8][32];
  static int idx = 0;
  char* b = bufs[idx++ & 7];
  snprintf(b, 32, "%.3f", v);
  return b;
}

int main() {
  umask(0027);
  signal(SIGTERM, on_term);
  signal(SIGINT, on_term);
  signal(SIGHUP, on_hup);
  signal(SIGPIPE, SIG_IGN);

  const std::string ring = env_s("SP_TRIG_RING", "m3");
  const double hz = env_d("SP_TRIG_HZ", 10.0);
  const std::string man_path = env_s(
      "SP_TRIG_MANIFEST", "/opt/m0/trt-dev/nv12_r0/manifest.jsonl");
  const std::string rules_dir =
      env_s("SP_TRIG_RULES_DIR", "/usr/local/share/sp/rules");
  const std::string thr_path = env_s("SP_TRIG_THR", "/etc/sp/thr.conf");
  const std::string out_dir = env_s("SP_TRIG_OUT", "/var/lib/sp/trigger");
  const bool ctx_dump = env_d("SP_TRIG_CTX_DUMP", 0) != 0;
  const bool observe = env_d("SP_TRIG_OBSERVE", 0) != 0;

  std::vector<TrigFrame> man;
  if (!parse_manifest_min(man_path.c_str(), &man)) {
    fprintf(stderr, "trigger: manifest parse failed: %s\n", man_path.c_str());
    return 2;
  }
  const size_t nman = man.size();
  fprintf(stderr, "trigger: manifest %s (%zu frames)\n", man_path.c_str(),
          nman);
  for (int i = 0; i < 3 && i < (int)nman; ++i)
    fprintf(stderr, "trigger: man[%d] x=%.1f y=%.1f ts=%llu\n", i,
            man[i].l2g[3], man[i].l2g[7],
            (unsigned long long)man[i].ts_ns);

  mkdir(out_dir.c_str(), 0750);  // 只管新建; 已存在不动 (Phase D 口径)
  if (ctx_dump) {
    std::string cd = out_dir + "/ctx";
    mkdir(cd.c_str(), 0750);
  }
  ev_open(out_dir + "/events.jsonl");
  if (!g_ev) {
    fprintf(stderr, "trigger: cannot open %s/events.jsonl\n", out_dir.c_str());
    return 2;
  }
  // 观察模式 (阈值重标, spec §9): 逐帧落规则统计量原始值, 不评规则不发事件
  FILE* g_ob = 0;
  if (observe) {
    std::string p = out_dir + "/observe.jsonl";
    g_ob = fopen(p.c_str(), "a");
    if (!g_ob) {
      fprintf(stderr, "trigger: cannot open %s\n", p.c_str());
      return 2;
    }
    fprintf(stderr, "trigger: OBSERVE mode (no rule eval, no events)\n");
  }
  // parity 原始事件流 (仅 ctx_dump 模式, 免配额)
  FILE* g_raw = 0;
  if (ctx_dump) {
    std::string p = out_dir + "/events_raw.jsonl";
    g_raw = fopen(p.c_str(), "a");
    if (!g_raw) {
      fprintf(stderr, "trigger: cannot open %s\n", p.c_str());
      return 2;
    }
  }

  // 信箱附着: 下游先于节点启动是常态, 轮询等
  sp::res::Mailbox* mb = 0;
  while (!mb && !g_stop.load()) {
    char err[256] = {0};
    mb = sp::res::Mailbox::attach(("sp_result_" + ring).c_str(), err,
                                  sizeof(err));
    if (!mb) {
      fprintf(stderr, "trigger: mailbox wait (%s)\n", err);
      for (int i = 0; i < 50 && !g_stop.load(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  if (!mb) return 0;
  fprintf(stderr, "trigger: mailbox sp_result_%s attached\n", ring.c_str());

  char serr[256] = {0};
  int nrules = rule_scan(rules_dir.c_str(), thr_path.c_str(), 0, 256, serr,
                         sizeof(serr));
  fprintf(stderr, "trigger: %d rules, tick=%.1fHz, ctx_dump=%d\n", nrules,
          hz, (int)ctx_dump);

  // [quota] 段 → 仲裁器
  Quota quota;
  {
    FILE* f = fopen(thr_path.c_str(), "rb");
    if (f) {
      std::string all;
      char buf[8192];
      size_t n;
      while ((n = fread(buf, 1, sizeof(buf), f)) > 0) all.append(buf, n);
      fclose(f);
      quota.configure(sprule::thr_section(all, "quota").c_str());
    }
  }
  fprintf(stderr, "trigger: quota dedup=%.1fs per_ev=%d/min global=%d/min\n",
          quota.dedup_window_s, quota.per_event_per_min, quota.global_per_min);

  Egoring er;
  uint64_t last_seq = 0;
  unsigned long long n_ctx = 0, n_ev = 0, n_skip = 0;
  sp_trk cur_det[sp::res::kDetCap];

  auto tick_period =
      std::chrono::microseconds((int)(1e6 / (hz > 0 ? hz : 10)));
  auto next_tick = std::chrono::steady_clock::now();
  unsigned long long hb = 0;

  while (!g_stop.load()) {
    next_tick += tick_period;
    if (g_rescan.exchange(0)) {
      nrules = rule_rescan(rules_dir.c_str(), thr_path.c_str());
      fprintf(stderr, "trigger: rescan -> %d rules\n", nrules);
    }
    sp::res::ResultMsg msg;
    bool got = mb->read_latest(&msg);
    if (got && msg.magic == sp::res::kMagic && msg.seq != last_seq &&
        msg.status == sp::res::kStatusNominal) {
      if (last_seq && msg.seq < last_seq) {
        fprintf(stderr, "trigger: seq rewind %llu->%llu, ring rebuilt\n",
                (unsigned long long)last_seq, (unsigned long long)msg.seq);
      }
      last_seq = msg.seq;
      // manifest join: seq 1-based; filesrc 按 seq s 携带 manifest[(s-1)%nman]
      const TrigFrame& fm = man[(msg.seq - 1) % nman];
      // 检出转换 (DetBox 与 sp_trk 字段序不同, 逐字段拷贝)
      uint32_t nd = msg.n_det > sp::res::kDetCap ? sp::res::kDetCap
                                                 : msg.n_det;
      for (uint32_t i = 0; i < nd; ++i) {
        const sp::res::DetBox& d = msg.det[i];
        cur_det[i].score = d.score;
        cur_det[i].label = d.label;
        cur_det[i].id = d.id;
        cur_det[i].x = d.x;
        cur_det[i].y = d.y;
        cur_det[i].z = d.z;
        cur_det[i].w = d.w;
        cur_det[i].l = d.l;
        cur_det[i].h = d.h;
        cur_det[i].yaw = d.yaw;
        cur_det[i].vx = d.vx;
        cur_det[i].vy = d.vy;
      }
      TrigFrameLite fr{msg.seq, fm.scene, (int64_t)fm.ts_ns, fm.l2g};
      er.push_frame(fr, cur_det, nd);  // 边界/ts 异常内部重建窗口
      sp_frame_ctx ctx;
      er.build_ctx(&ctx, msg.status);
      ctx.final_plan = &msg.final_plan[0][0];
      if (observe) {
        // 规则统计量原始值 (Task 4 重标定输入; 空缺 = null)
        double lead_d = 0, lead_v = 0, vru_lat = 0, vru_cross = 0,
               cone_d = 0;
        int has_lead = 0, has_vru = 0, has_cone = 0;
        int li = -1;
        double bd = 1e18;
        for (uint32_t i = 0; i < nd; ++i) {
          const sp_trk* d = &cur_det[i];
          double dd = sqrt((double)d->x * d->x + (double)d->y * d->y);
          if (ru_is_cone(d->label)) {
            has_cone = 1;
            if (dd < cone_d || cone_d == 0) cone_d = dd;
          }
          if (ru_is_vehicle(d->label) && d->x > 2.0 && d->x < 40.0 &&
              fabs((double)d->y) < 2.5 && dd < bd) {
            bd = dd;
            li = (int)i;
          }
        }
        if (li >= 0) {
          has_lead = 1;
          lead_d = bd;
          lead_v = cur_det[li].vx;
          double svx[SP_RULE_HIST];
          int k = ru_id_series(&ctx, cur_det[li].id, 0, 0, svx, 0, 0);
          if (k >= 3) {
            double s = 0;
            for (int i = 0; i < k; ++i) s += svx[i];
            lead_v = s / k;
          }
        }
        for (uint32_t i = 0; i < nd; ++i) {
          const sp_trk* d = &cur_det[i];
          if (!ru_is_vru(d->label) || d->x <= -5.0 || d->x >= 30.0) continue;
          has_vru = 1;
          if (fabs((double)d->y) < vru_lat || vru_lat == 0)
            vru_lat = fabs((double)d->y);
          if (d->id >= 0) {
            double sy[SP_RULE_HIST], sts[SP_RULE_HIST];
            int k = ru_id_series(&ctx, d->id, 0, sy, 0, 0, sts);
            for (int j = 1; j < k; ++j) {
              double dt = (sts[j] - sts[j - 1]) / 1e9;
              if (dt <= 0) continue;
              double sp = fabs((sy[j] - sy[j - 1]) / dt);
              if (sp > vru_cross) vru_cross = sp;
            }
          }
        }
        fprintf(g_ob,
                "{\"seq\":%llu,\"ts_ns\":%lld,\"acc\":%.4f,\"speed\":%.4f,"
                "\"yaw_abs\":%.4f,\"lat_abs\":%.4f,"
                "\"lead_d\":%s,\"lead_v\":%s,\"vru_lat\":%s,"
                "\"vru_cross\":%s,\"cone_d\":%s}\n",
                (unsigned long long)msg.seq, (long long)fm.ts_ns,
                (double)ctx.ego.acc, (double)ctx.ego.speed,
                fabs((double)ctx.ego.yaw_rate),
                fabs((double)ctx.ego.speed * (double)ctx.ego.yaw_rate),
                has_lead ? buf_f(lead_d) : "null",
                has_lead ? buf_f(lead_v) : "null",
                has_vru ? buf_f(vru_lat) : "null",
                vru_cross > 0 ? buf_f(vru_cross) : "null",
                has_cone ? buf_f(cone_d) : "null");
        ++n_ctx;
        if (++hb % (unsigned long long)(hz * 60) == 0)
          fprintf(stderr, "trigger: hb seq=%llu observe=%llu\n",
                  (unsigned long long)last_seq, n_ctx);
        std::this_thread::sleep_until(next_tick);
        continue;
      }
      // 规则评估 → 仲裁 → 落盘
      for (int i = 0; i < nrules; ++i) {
        sp_rule_desc* r = rule_at(i);
        if (!r) continue;
        sp_rule_event ev;
        memset(&ev, 0, sizeof(ev));
        int ne = r->eval(&ctx, &ev);
        for (int e = 0; e < ne; ++e) {
          // ctx_dump = parity 夹具模式: 原始事件 (免配额) 另落一份,
          // 供 §7.5 与 numpy 参考实现全等比对
          if (ctx_dump) {
            fprintf(g_raw,
                    "{\"ts_ns\":%lld,\"seq\":%llu,\"event\":\"%s\","
                    "\"strength\":%.6f}\n",
                    (long long)fm.ts_ns, (unsigned long long)msg.seq,
                    ev.name, (double)ev.strength);
          }
          if (quota.allow(msg.seq, fm.ts_ns, ev.name)) {
            ev_write(msg.seq, fm.ts_ns, ev.name, ev.strength);
            ++n_ev;
          }
        }
      }
      // ctx-dump (对拍夹具; 布局钉死见 SDD ledger pre-flight)
      if (ctx_dump) {
        char pf[512];
        snprintf(pf, sizeof(pf), "%s/ctx/ctx_%05llu.bin", out_dir.c_str(),
                 (unsigned long long)n_ctx);
        FILE* f = fopen(pf, "wb");
        if (f) {
          uint32_t magic = 0x53505431, flags = 0;
          uint32_t cur = (er.head + SP_RULE_HIST - 1) % SP_RULE_HIST;
          fwrite(&magic, 4, 1, f);
          fwrite(&ctx.abi_ver, 4, 1, f);
          fwrite(&flags, 4, 1, f);
          fwrite(&ctx.seq, 8, 1, f);
          fwrite(&ctx.ts_ns, 8, 1, f);
          fwrite(&ctx.scene, 4, 1, f);
          fwrite(&ctx.n_det, 4, 1, f);
          fwrite(&ctx.n_hist, 4, 1, f);
          fwrite(&ctx.ego, sizeof(sp_ego_state), 1, f);
          fwrite(cur_det, sizeof(sp_trk), nd, f);
          for (uint32_t i = 0; i < ctx.n_hist; ++i) {
            uint32_t ix = (cur + SP_RULE_HIST - i) % SP_RULE_HIST;
            fwrite(&er.ego[ix], sizeof(sp_ego_state), 1, f);
            int64_t t8 = er.ts[ix];
            uint32_t n4 = er.n[ix], pad = 0;
            fwrite(&t8, 8, 1, f);
            fwrite(&n4, 4, 1, f);
            fwrite(&pad, 4, 1, f);
            fwrite(er.det[ix], sizeof(sp_trk), n4, f);
          }
          fclose(f);
        }
      }
      n_ctx += 1;
      if (++hb % (unsigned long long)(hz * 60) == 0)
        fprintf(stderr,
                "trigger: hb seq=%llu ctx=%llu events=%llu supp=%llu "
                "rules=%d\n",
                (unsigned long long)last_seq, n_ctx, n_ev,
                quota.suppressed, nrules);
    } else if (got) {
      ++n_skip;  // seq 未前进 / LATCH 帧 / 撕裂重试耗尽
    }
    if (g_stop.load()) break;
    std::this_thread::sleep_until(next_tick);
    if (std::chrono::steady_clock::now() - next_tick >
        std::chrono::seconds(2))
      next_tick = std::chrono::steady_clock::now();  // 追不上就重置节拍
  }
  fprintf(stderr,
          "trigger: stop (ctx=%llu events=%llu suppressed=%llu skip=%llu)\n",
          n_ctx, n_ev, quota.suppressed, n_skip);
  if (g_ev) fclose(g_ev);
  if (g_raw) fclose(g_raw);
  if (g_ob) fclose(g_ob);
  rule_unload_all();
  return 0;
}
