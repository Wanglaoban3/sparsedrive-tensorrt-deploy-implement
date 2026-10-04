// sp_bus.h - cross-process NV12 frame ring for the pre/post operator library.
//
// Topology: one publisher (capture) writes slots in a POSIX shared-memory
// ring; N consumers mmap it READ-ONLY and hold per-slot references.
// Contract (design: docs/PREPOST_OPS_DESIGN.md "IPC 内存契约"):
//   - consumers get 6 read-only camera plane pointers per frame;
//   - a slot held by any consumer is never recycled: publisher blocks on
//     claim() until refcount hits 0 (or a stale lease is force-recycled);
//   - consumers must heartbeat() while holding frames; a crashed consumer's
//     references are reclaimed after SP_BUS_LEASE_MS (default 5000);
//   - acquire() returns frames in publish order and reports sequence gaps.
#ifndef SP_BUS_H_
#define SP_BUS_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <pthread.h>

namespace sp {

constexpr int kMaxCams = 6;
constexpr int kMaxConsumers = 8;
constexpr int kRingDepth = 4;
constexpr uint32_t kMagic = 0x53504231;  // "SPB1"
constexpr uint32_t kVersion = 4;  // v4: 自研跨进程锁 (robust mutex 不可靠)
constexpr size_t kNameMax = 64;

// FrameMeta.flags 源状态位 (发布端置位, 消费端只读透传到结果消息):
constexpr uint32_t kFlagSourceOk = 1u << 0;   // 整帧源 OK (无丢流/校验错)
constexpr int kFlagCamOkShift = 8;            // bits 8..13 = cam[0..5] 单路 OK
constexpr uint32_t kFlagAllCamsOk =
    0x3Fu << kFlagCamOkShift;                 // 文件回放源恒为全 OK

struct FrameMeta {              // per-slot header, written by publisher only
  uint64_t seq;                 // 1-based; 0 = slot free / being refilled
  int64_t group_ts_ns;          // trigger timestamp of the sync group
  int64_t cam_ts_ns[kMaxCams];
  uint32_t scene_id;
  uint32_t flags;               // bit0: source_ok
  uint32_t width;
  uint32_t height;
  uint32_t cam_bytes;           // per-cam NV12 size (W*H*3/2)
  uint32_t frame_bytes;         // 6 cams packed
  uint32_t checksum;            // crc32 of the frame payload
  uint32_t reserved;
};

struct SlotHdr {
  std::atomic<uint32_t> ref;    // consumers currently holding this slot
  FrameMeta meta;
};

struct ConsumerEntry {
  std::atomic<int64_t> hb_ms;   // last heartbeat, CLOCK_MONOTONIC ms
  std::atomic<int32_t> held_slot;  // slot index or -1
  int32_t pid;
  int32_t active;               // 0 free, 1 registered
};

// 跨进程锁 (自研, 替代 pthread robust mutex): 属主 pid+starttime 记在 shm,
// /proc 证据表明属主已死/已复用时 CAS 强制接管, kill -9 安全不依赖内核
// robust-list (平台实测 glibc 属主死亡移交在长跑 Orin 上偶发失效)。
struct BusLock {
  std::atomic<uint32_t> owner;  // 0=free, else owner pid
  std::atomic<uint64_t> start;  // 属主 /proc/<pid>/stat starttime (防 pid 复用)
};

struct RingMeta {
  uint32_t magic;
  uint32_t version;
  uint32_t width;
  uint32_t height;
  uint32_t cam_bytes;
  uint32_t frame_bytes;
  uint64_t slot_stride;         // sizeof(SlotHdr aligned) + frame_bytes
  BusLock lk;                   // 互斥 (等待一律 lock 外 10ms 轮询, 无 cond)
  std::atomic<uint64_t> latest_seq;
  uint64_t published_count;
  uint32_t pub_idx;             // next slot the publisher rotates into
  std::atomic<int32_t> n_consumers;
  ConsumerEntry cons[kMaxConsumers];
  uint64_t forced_recycles;     // lease-expired references stolen
  // v2 尾段: M8 dmabuf 池注册 (发布端建池+listen 后置位; fd 走 UDS 旁路,
  // 不进 RingMeta). 消费端 --dma 校验三者一致后 OpaqueFd 导入.
  uint32_t dma_present;         // 0/1
  uint32_t dma_n_slots;
  uint64_t dma_slot_bytes;
  uint32_t reserved2[5];
};

// Read-only view handed to a consumer. `cam[i]` points into the shared
// mapping; the consumer must not write through it.
struct FrameView {
  int32_t slot_idx;
  int32_t consumer_id;
  FrameMeta meta;
  const uint8_t* cam[kMaxCams];
};

class Bus {
 public:
  // Attaches to (or, when create=true, initializes) the named ring.
  // Returns nullptr on failure and fills `err`.
  static Bus* open(const char* name, uint32_t width, uint32_t height,
                   bool create, bool fresh, char* err, size_t errlen);
  ~Bus();

  // --- publisher side ---
  // Rotates to the next slot and blocks (bounded by timeout_ms) until its
  // reference count drops to 0; lease-expired holders are force-recycled.
  // Returns the payload pointer to fill, or nullptr on timeout.
  uint8_t* claim(int64_t timeout_ms) { return claim_of(nullptr, timeout_ms); }
  // M8: 同 claim(), 另回槽位号 (--dma 写池槽用).
  uint8_t* claim_of(int32_t* slot_idx, int64_t timeout_ms);
  // Publishes the slot filled after claim(): assigns the sequence number
  // and wakes consumers. `meta` fields except seq are taken as-is.
  void commit(FrameMeta meta);
  // M8: --dma 发布 (payload 在设备池, 校验和由发布端自算, 不读 shm 重算).
  void commit_dma(FrameMeta meta);
  // M8: 建池后注册池几何 (消费端开环即可见; 必须在消费者 attach 前调用).
  void set_dma_info(uint32_t n_slots, uint64_t slot_bytes);

  // --- consumer side ---
  int32_t register_consumer();
  void unregister_consumer(int32_t cid);
  int consumer_count() const { return m_->n_consumers; }
  void heartbeat(int32_t cid);
  // Waits for the next frame after `last_seq` (publish order), takes a
  // reference and fills `out`. Returns 0 on success, 1 on timeout.
  // A sequence gap (out.meta.seq > last_seq + 1) means frames were
  // overwritten before this consumer saw them.
  int acquire(int32_t cid, uint64_t last_seq, FrameView* out,
              int64_t timeout_ms);
  void release(FrameView* v);

  const RingMeta* meta() const { return m_; }
  uint64_t forced_recycles() const { return m_->forced_recycles; }
  // Base/length of the whole shared mapping (for one-shot cudaHostRegister).
  void* base() const { return map_; }
  size_t mapped_bytes() const { return map_len_; }

 private:
  Bus() = default;
  bool init_layout(uint32_t width, uint32_t height);
  // Slots are laid out by slot_stride, NOT sizeof(SlotHdr) - never index a
  // raw SlotHdr array across processes; go through this accessor.
  SlotHdr* slot(int idx) const {
    return (SlotHdr*)((uint8_t*)slots_ + idx * m_->slot_stride);
  }
  uint8_t* payload_of(int idx) const {
    return (uint8_t*)slot(idx) + (m_->slot_stride - m_->frame_bytes);
  }
  void force_stale_locked(int32_t slot_idx, int64_t now_ms, int64_t lease_ms);

  int fd_ = -1;
  void* map_ = nullptr;
  size_t map_len_ = 0;
  bool owner_ = false;
  char name_[kNameMax];
  RingMeta* m_ = nullptr;
  SlotHdr* slots_ = nullptr;
  uint8_t* payloads_ = nullptr;
};

// Milliseconds from CLOCK_MONOTONIC (shared by pub/sub for timestamps).
int64_t now_ms();
// Nanoseconds from CLOCK_REALTIME (wall clock, used in FrameMeta).
int64_t now_real_ns();
uint32_t crc32(const uint8_t* p, size_t n);
// Head+tail window checksum used by commit(); verify with the same.
uint32_t sampled_checksum(const uint8_t* p, size_t n);

}  // namespace sp

#endif  // SP_BUS_H_
