// sp_result.h - 感知结果消息 + 单槽 latest-wins 信箱 (seqlock).
//
// 设计稿 §4 "结果消息": 版本化 struct + JSON 旁路; 发布走单槽 latest-wins
// 信箱, 下游永远取最新帧.
//
// 信箱实现: POSIX shm 一块, 头部是 seqlock 写序号 (奇 = 写入中), 读方
// 读前读后各取一次, 奇数或不相等即重试 —— 单写多读, 读方永不阻塞写方,
// 写方永不阻塞读方 (适合"下游只要最新帧"的感知输出语义).
//
// 布局契约: ResultMsg 定长 (det/map 各 300 容量), sizeof 由 header_size
// 字段自证; 读方校验 magic/version/header_size 不匹配即拒绝.
#ifndef SP_RESULT_H_
#define SP_RESULT_H_

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "postproc.h"

namespace sp {
namespace res {

constexpr uint32_t kMagic = 0x53505231;   // "SPR1"
constexpr uint16_t kVer = 3;              // v3: 尾部追加 fail-visible 段 (M-PROD A4)
// ---- v3 fail-visible 状态/原因枚举 (M-PROD) ----
enum {
  kStatusNominal = 0,       // 本帧正常
  kStatusDegReset = 1,      // 发散已复位, 本帧结果基于复位态 (Phase B)
  kStatusDegLatch = 2,      // 持续异常, 载荷=last_valid + frame_age_ms 真实旧化
  kStatusSelftestFail = 3   // 自检失败, det/map 清零, last_valid_seq=0
};
enum {
  kReasonNone = 0, kReasonNan = 1, kReasonDivergence = 2,
  kReasonResetCount = 3, kReasonSelftest = 4, kReasonStageFail = 5
};
constexpr int kDetCap = 300;              // 与 det topk 上限一致
constexpr int kMapCap = post::kMapAnchors * post::kMapCls;
constexpr int kPluginPathMax = 192;
// ---- v2 motion/plan 段尺寸 (与 e_mp 引擎 IO 同形, eval_mp_mini SHAPES) --
constexpr int kMotClsLen = 900 * 6;           // motion_cls [900,6]
constexpr int kMotRegLen = 900 * 6 * 12 * 2;  // motion_reg [900,6,12,2]
constexpr int kPlanClsLen = 3 * 6;            // plan_cls   [18] = 3cmd x 6mode
constexpr int kPlanRegLen = 3 * 6 * 6 * 2;    // plan_reg   [18,6,2]
constexpr int kPlanStatusLen = 10;            // plan_status[10]
constexpr int kPlanPts = 6;                   // final_plan [6,2]

struct DetBox {                 // det[]{score,label,x,y,z,w,l,h,yaw,vx,vy,id}
  float score;
  int32_t label;
  float x, y, z, w, l, h, yaw, vx, vy;
  int32_t id;
};

struct MapVec {                 // map[]{score,label,pts[20][2]}
  float score;
  int32_t label;
  float pts[post::kMapPts][2];
};

struct ResultMsg {
  // ---- header (设计稿: seq/ts/frame_id/scene_flags/config_hash/plugin_path) ----
  uint32_t magic;               // kMagic
  uint16_t version;             // kVer
  uint16_t header_size;         // offsetof(ResultMsg, det) 布局自证
  uint64_t seq;                 // 总线帧序 (1-based, 与 FrameMeta.seq 同源)
  int64_t ts_capture_ns;        // CLOCK_REALTIME 节点 acquire 时刻 (非 manifest 采集 ts)
  uint32_t frame_id;            // 节点内帧号 (0-based, 回绕时随 seq 区分)
  uint32_t scene_flags;         // 高 24b scene_id, 低 8b 源状态位 (FrameMeta.flags)
  uint64_t config_hash;         // FNV1a(engine|plugin|det_thr|map_thr|topk)
  char plugin_path[kPluginPathMax];
  // ---- 各级时间戳 (ms, 事件链实测) ----
  float pre_ms, infer_ms, post_ms, e2e_ms;
  // ---- payloads ----
  uint32_t n_det;
  uint32_t n_map;
  uint32_t crc;                 // det+map 区域 crc32 ( torn-read 辅助校验)
  uint32_t reserved;
  DetBox det[kDetCap];
  MapVec map[kMapCap];
  // ---- v2 追加段 (M6b; v1 读端按 version 拒绝, 结构内 append 不动 v1 前缀) --
  // motion/plan 为 e_mp 引擎原始输出 (密集 900 锚/18 槽全量); final_plan 是
  // 节点便捷解码 = plan_reg[cmd*6 + argmax(plan_cls[cmd*6..cmd*6+6])] 行,
  // cmd 由节点 --cmd 给定 (默认 2=直行; 部署上真 cmd 来自车辆接口, 下游
  // 可按 plan_cls/reg 自行重选). 口径 = eval_mp_mini final_planning
  // ( PlanningDecoder.select: 按 cmd 取 6 模式 → argmax → 对应 plan_reg 行).
  float motion_cls[kMotClsLen];
  float motion_reg[kMotRegLen];
  float plan_cls[kPlanClsLen];
  float plan_reg[kPlanRegLen];
  float plan_status[kPlanStatusLen];
  float final_plan[kPlanPts][2];
  float t_mp;                   // mp 段耗时 ms (无 --mp 时 0)
  uint32_t cmd;                 // 节点解码用的 cmd 索引
  uint32_t reserved2;
  // ---- v3 追加段 (M-PROD fail-visible; v2 读端按 header_size 忽略尾部) --
  // 语义: 下游要么拿到新结果 (NOMINAL), 要么明确知道拿到的是 frame_age_ms
  // 前的旧结果 (LATCH), 要么明确知道节点不可用 (SELFTEST_FAIL 心跳,
  // last_valid_seq=0 表示"自本进程启动从未有有效结果"). 不再有静默.
  uint8_t status;               // kStatus* (Phase A 恒 NOMINAL)
  uint8_t reason;               // kReason*
  uint8_t wd_stage;             // 发布时刻所处/最后阶段 (诊断)
  uint8_t reserved3[5];
  uint32_t last_valid_seq;      // status!=0 时有效结果所属帧 seq (低 32b)
  uint16_t frame_age_ms;        // 结果相对输入帧年龄 (饱和 65535)
  uint16_t resets_60s;          // 60s 窗口复位次数 (Phase B, 健康遥测)
  uint32_t nan_hits;            // 累计 NaN/Inf 命中 (Phase B)
  uint32_t div_hits;            // 累计发散命中 (Phase B)
};

constexpr size_t kMsgBytes = sizeof(ResultMsg);

// crc 只盖 det 容量区 + map 有效条数 (发布端与读端必须同式; 定义在此防漂移)
inline uint32_t msg_crc(const ResultMsg& m,
                        uint32_t (*crc32fn)(const uint8_t*, size_t)) {
  return crc32fn((const uint8_t*)m.det,
                 kDetCap * sizeof(DetBox) + m.n_map * sizeof(MapVec));
}

struct Slot {                   // shm 布局: seqlock 前缀 + 消息体
  std::atomic<uint64_t> wseq;   // 偶 = 稳定, 奇 = 写入中
  std::atomic<uint64_t> writes; // 诊断: 发布次数
  ResultMsg msg;
};

class Mailbox {
 public:
  // 写端创建 (O_CREAT, 已存在则复用并重置). 失败返回 nullptr 并填 err.
  static Mailbox* create(const char* name, char* err, size_t errlen) {
    return open_impl(name, true, err, errlen);
  }
  // 读端附着 (不存在返回 nullptr, 不创建).
  static Mailbox* attach(const char* name, char* err, size_t errlen) {
    return open_impl(name, false, err, errlen);
  }
  ~Mailbox() {
    if (map_) {
      munmap(map_, sizeof(Slot));
      close(fd_);
    }
  }
  // 单写者假设: 一个 ring 只应有一个 node 在发布.
  void publish(const ResultMsg& m) {
    uint64_t s = slot_->wseq.load(std::memory_order_relaxed);
    slot_->wseq.store(s + 1, std::memory_order_relaxed);   // → 奇: 写入中
    std::atomic_thread_fence(std::memory_order_release);
    slot_->msg = m;
    std::atomic_thread_fence(std::memory_order_release);
    slot_->wseq.store(s + 2, std::memory_order_relaxed);   // → 偶: 稳定
    slot_->writes.fetch_add(1, std::memory_order_relaxed);
  }
  // 读最新完整帧; 写入中/撕裂最多重试 retry 次. false = 信箱空或持续写入.
  bool read_latest(ResultMsg* out, int retry = 200) const {
    for (int i = 0; i < retry; ++i) {
      uint64_t s1 = slot_->wseq.load(std::memory_order_acquire);
      if (s1 & 1) continue;                                // 写入中
      std::atomic_thread_fence(std::memory_order_acquire);
      memcpy(out, &slot_->msg, sizeof(ResultMsg));
      std::atomic_thread_fence(std::memory_order_acquire);
      uint64_t s2 = slot_->wseq.load(std::memory_order_relaxed);
      if (s1 == s2 && out->magic == kMagic) return true;
    }
    return false;
  }
  uint64_t writes() const {
    return slot_->writes.load(std::memory_order_relaxed);
  }

 private:
  static Mailbox* open_impl(const char* name, bool create, char* err,
                            size_t errlen) {
    char path[128];
    snprintf(path, sizeof(path), "/sp_res_%s", name);
    int fd = shm_open(path, create ? (O_RDWR | O_CREAT) : O_RDWR, 0666);
    if (fd < 0) {
      snprintf(err, errlen, "shm_open %s: errno=%d", path, errno_r());
      return nullptr;
    }
    if (create && ftruncate(fd, sizeof(Slot)) != 0) {
      snprintf(err, errlen, "ftruncate: errno=%d", errno_r());
      close(fd);
      return nullptr;
    }
    void* m = mmap(nullptr, sizeof(Slot), PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd, 0);
    close(fd);   // mmap 持有引用, fd 可关
    if (m == MAP_FAILED) {
      snprintf(err, errlen, "mmap: errno=%d", errno_r());
      return nullptr;
    }
    Mailbox* mb = new Mailbox();
    mb->fd_ = -1;
    mb->map_ = m;
    mb->slot_ = (Slot*)m;
    if (create) {
      mb->slot_->wseq.store(0, std::memory_order_relaxed);
      mb->slot_->writes.store(0, std::memory_order_relaxed);
      memset(&mb->slot_->msg, 0, sizeof(ResultMsg));
      std::atomic_thread_fence(std::memory_order_release);
    }
    return mb;
  }
  static int errno_r() { return errno; }
  Mailbox() = default;
  int fd_ = -1;
  void* map_ = nullptr;
  Slot* slot_ = nullptr;
};

}  // namespace res
}  // namespace sp

#endif  // SP_RESULT_H_
