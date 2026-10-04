// sp_bus.cpp - POSIX shm ring implementation. See sp_bus.h for the contract.
//
// Locking discipline: slot references and header visibility are serialized by
// RingMeta::mu; heartbeats are plain atomics outside the lock. Publisher
// payload writes happen outside the lock on a slot that is invalid
// (meta.seq == 0) and unreferenced, so consumers can never observe a
// half-written frame.
#include "sp_bus.h"

#include <cerrno>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace sp {

static constexpr int64_t kLeaseMsDefault = 5000;

static int64_t lease_ms_from_env() {
  const char* e = getenv("SP_BUS_LEASE_MS");
  if (!e) return kLeaseMsDefault;
  long v = atol(e);
  return v > 0 ? v : kLeaseMsDefault;
}

int64_t now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return int64_t(ts.tv_sec) * 1000 + ts.tv_nsec / 1000000;
}

// ---- 跨进程自研锁 (BusLock, sp_bus.h) ----
// 平台实测: glibc robust mutex 的属主死亡移交在长跑 Orin 上偶发失效
// (FT1: 持锁者 kill -9 后双进程永久 futex 等待, EOWNERDEAD 不来)。
// 自管方案不赌内核: 属主 pid+starttime 落 shm, /proc 判死即 CAS 接管。
// 无争用 = 一次 CAS; 争用 = 2ms 轮询 (临界区 µs 级, 帧间隔 ≥40ms)。
// 等待条件 (空槽/新帧) 一律解锁后 10ms 轮询, 不用 condvar。
static uint64_t proc_starttime(uint32_t pid) {
  char p[64];
  snprintf(p, sizeof(p), "/proc/%u/stat", pid);
  FILE* f = fopen(p, "r");
  if (!f) return 0;  // task 不存在
  char buf[1024];
  const char* line = fgets(buf, sizeof(buf), f) ? buf : nullptr;
  fclose(f);
  if (!line) return 0;
  const char* q = strrchr(line, ')');
  if (!q) return 0;
  // ')' 后字段从 state(3) 起, starttime=22 → 跳 19 个读 1 个
  long long st = 0;
  if (sscanf(q + 2,
             "%*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s "
             "%*s %*s %*s %lld",
             &st) != 1)
    return 0;
  return (uint64_t)st;
}

static inline void lock_lk(BusLock* lk) {
  static uint64_t my_start = proc_starttime((uint32_t)getpid());
  const uint32_t me = (uint32_t)getpid();
  for (;;) {
    uint32_t exp = 0;
    if (lk->owner.compare_exchange_strong(exp, me)) {
      lk->start.store(my_start, std::memory_order_release);
      return;
    }
    uint32_t cur = lk->owner.load(std::memory_order_acquire);
    if (cur != 0 && cur != me) {
      uint64_t own_start = lk->start.load(std::memory_order_acquire);
      uint64_t now_st = proc_starttime(cur);
      if (now_st == 0 || (own_start != 0 && now_st != own_start)) {
        // 属主已死或 pid 已被复用: 强制接管 (CAS 保证只有一个胜利者)
        uint32_t exp2 = cur;
        lk->owner.compare_exchange_strong(exp2, 0);
      }
    }
    struct timespec ts = {0, 2 * 1000 * 1000};
    nanosleep(&ts, nullptr);
  }
}

static inline void unlock_lk(BusLock* lk) {
  lk->owner.store(0, std::memory_order_release);
}

static inline void poll_sleep_ms(int ms) {
  struct timespec ts = {ms / 1000, (ms % 1000) * 1000 * 1000L};
  nanosleep(&ts, nullptr);
}

int64_t now_real_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return int64_t(ts.tv_sec) * 1000000000ll + ts.tv_nsec;
}

uint32_t crc32(const uint8_t* p, size_t n) {
  static uint32_t table[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k)
        c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      table[i] = c;
    }
    init = true;
  }
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

// Sampled integrity checksum: a full-frame CRC over a 12.9 MB NV12 frame
// costs hundreds of microseconds-to-ms per call on both sides, so commit()
// uses head+tail windows. sub verifies the same windows every frame and can
// additionally run full crc32() on a low-rate schedule.
uint32_t sampled_checksum(const uint8_t* p, size_t n) {
  const size_t win = 64 * 1024;
  if (n <= 2 * win) return crc32(p, n);
  return crc32(p, win) ^ crc32(p + n - win, win) ^
         (uint32_t)(n & 0xFFFFFFFFu);
}

static uint64_t align_up(uint64_t v, uint64_t a) {
  return (v + a - 1) / a * a;
}

Bus* Bus::open(const char* name, uint32_t width, uint32_t height, bool create,
               bool fresh, char* err, size_t errlen) {
  Bus* b = new Bus();
  snprintf(b->name_, sizeof(b->name_), "/sp_%s", name);
  if (create && fresh) shm_unlink(b->name_);

  int fd = shm_open(b->name_, (create ? O_CREAT : 0) | O_RDWR, 0600);
  if (fd < 0) {
    snprintf(err, errlen, "shm_open(%s) failed: %s", b->name_,
             strerror(errno));
    delete b;
    return nullptr;
  }
  b->fd_ = fd;

  size_t cam_bytes = uint64_t(width) * height * 3 / 2;
  size_t hdr_sz = align_up(sizeof(SlotHdr), 64);
  size_t slot_stride = hdr_sz + cam_bytes * kMaxCams;
  size_t map_len = align_up(sizeof(RingMeta), 64) + slot_stride * kRingDepth;

  struct stat st;
  bool do_init = create;
  if (create && fstat(fd, &st) == 0 && st.st_size == (off_t)map_len) {
    // 同尺寸旧环: magic+version 都匹配才 attach (崩溃的发布端不清消费者
    // 状态); 版本不匹配 (如锁属性升级) 就地重建。
    RingMeta head;
    if (pread(fd, &head, sizeof(head), 0) == (ssize_t)sizeof(head) &&
        head.magic == kMagic && head.version == kVersion) {
      do_init = false;
    }
  }
  if (do_init) {
    if (ftruncate(fd, map_len) != 0) {
      snprintf(err, errlen, "ftruncate failed: %s", strerror(errno));
      delete b;
      return nullptr;
    }
  }
  void* map = mmap(nullptr, map_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                   0);
  if (map == MAP_FAILED) {
    snprintf(err, errlen, "mmap failed: %s", strerror(errno));
    delete b;
    return nullptr;
  }
  b->map_ = map;
  b->map_len_ = map_len;
  b->m_ = (RingMeta*)map;
  b->owner_ = do_init;

  if (do_init) {
    memset(map, 0, map_len);
    RingMeta* m = b->m_;
    m->magic = kMagic;
    m->version = kVersion;
    m->width = width;
    m->height = height;
    m->cam_bytes = cam_bytes;
    m->frame_bytes = cam_bytes * kMaxCams;
    m->slot_stride = slot_stride;
    // BusLock 零值即解锁态 (memset 已保证); 无需 pthread 初始化
    m->pub_idx = 0;
    m->latest_seq = 0;
    m->n_consumers = 0;
    for (int i = 0; i < kMaxConsumers; ++i) {
      m->cons[i].held_slot = -1;
      m->cons[i].hb_ms = 0;
    }
  } else {
    const RingMeta* m = b->m_;
    if (m->magic != kMagic || m->version != kVersion ||
        m->width != width || m->height != height ||
        m->cam_bytes != cam_bytes || m->slot_stride != slot_stride) {
      snprintf(err, errlen,
               "geometry mismatch: magic=%08x ver=%u %ux%u cam=%u stride=%lu",
               m->magic, m->version, m->width, m->height, m->cam_bytes,
               (unsigned long)m->slot_stride);
      delete b;
      return nullptr;
    }
  }
  b->slots_ = (SlotHdr*)((uint8_t*)map + align_up(sizeof(RingMeta), 64));
  b->payloads_ = (uint8_t*)b->slots_;
  return b;
}

Bus::~Bus() {
  if (map_) munmap(map_, map_len_);
  if (fd_ >= 0) close(fd_);
}

uint8_t* Bus::claim_of(int32_t* slot_idx, int64_t timeout_ms) {
  const int64_t lease = lease_ms_from_env();
  const int64_t deadline = now_ms() + timeout_ms;
  for (;;) {
    lock_lk(&m_->lk);
    SlotHdr& s = *slot(m_->pub_idx);
    if (s.ref.load() == 0) {
      s.meta.seq = 0;  // invalidate stale seq before refilling
      const int32_t idx = m_->pub_idx;
      unlock_lk(&m_->lk);
      if (slot_idx) *slot_idx = idx;
      return payload_of(idx);
    }
    force_stale_locked(m_->pub_idx, now_ms(), lease);
    if (s.ref.load() == 0) {
      s.meta.seq = 0;
      const int32_t idx = m_->pub_idx;
      unlock_lk(&m_->lk);
      if (slot_idx) *slot_idx = idx;
      return payload_of(idx);
    }
    unlock_lk(&m_->lk);
    poll_sleep_ms(10);
    if (timeout_ms >= 0 && now_ms() >= deadline) return nullptr;
  }
}

void Bus::commit(FrameMeta meta) {
  const int32_t idx = m_->pub_idx;
  SlotHdr& s = *slot(idx);
  meta.width = m_->width;
  meta.height = m_->height;
  meta.cam_bytes = m_->cam_bytes;
  meta.frame_bytes = m_->frame_bytes;
  if (meta.group_ts_ns == 0) meta.group_ts_ns = now_real_ns();
  meta.checksum = sampled_checksum(payload_of(idx), m_->frame_bytes);
  lock_lk(&m_->lk);
  s.meta = meta;
  s.meta.seq = m_->latest_seq + 1;
  m_->latest_seq = s.meta.seq;
  m_->published_count += 1;
  m_->pub_idx = (idx + 1) % kRingDepth;
  unlock_lk(&m_->lk);
}

void Bus::commit_dma(FrameMeta meta) {
  const int32_t idx = m_->pub_idx;
  SlotHdr& s = *slot(idx);
  meta.width = m_->width;
  meta.height = m_->height;
  meta.cam_bytes = m_->cam_bytes;
  meta.frame_bytes = m_->frame_bytes;
  if (meta.group_ts_ns == 0) meta.group_ts_ns = now_real_ns();
  // 校验和由发布端对 host 侧源数据自算 (payload 不落 shm, 无法重算)
  lock_lk(&m_->lk);
  s.meta = meta;
  s.meta.seq = m_->latest_seq + 1;
  m_->latest_seq = s.meta.seq;
  m_->published_count += 1;
  m_->pub_idx = (idx + 1) % kRingDepth;
  unlock_lk(&m_->lk);
}

void Bus::set_dma_info(uint32_t n_slots, uint64_t slot_bytes) {
  m_->dma_present = 1;
  m_->dma_n_slots = n_slots;
  m_->dma_slot_bytes = slot_bytes;
}

int32_t Bus::register_consumer() {
  // 持久环上 kill -9 的消费者没有 unregister 机会, cid 会被永久占住。
  // 活消费者每帧(~200ms@5fps)及 acquire 等待中都心跳, 10s 无心跳 + 残留
  // 槽引用未动 = 前主已死, 接管该 cid 并释放其引用 (计数不变, 槽位易主)。
  static const int64_t kCidTakeoverMs = 10000;
  lock_lk(&m_->lk);
  int32_t cid = -1;
  bool took_over = false;
  for (int i = 0; i < kMaxConsumers; ++i) {
    if (!m_->cons[i].active) {
      m_->cons[i].active = 1;
      cid = i;
      break;
    }
    if (now_ms() - m_->cons[i].hb_ms.load() > kCidTakeoverMs) {
      if (m_->cons[i].held_slot >= 0) {
        SlotHdr& s = *slot(m_->cons[i].held_slot);
        if (s.ref.load() > 0) s.ref.fetch_sub(1);
        m_->forced_recycles += 1;
      }
      fprintf(stderr,
              "bus: cid=%d taken over from stale consumer pid=%d "
              "(hb %lldms old)\n",
              i, m_->cons[i].pid,
              (long long)(now_ms() - m_->cons[i].hb_ms.load()));
      took_over = true;
      cid = i;
      break;
    }
  }
  if (cid >= 0) {
    m_->cons[cid].pid = (int32_t)getpid();
    m_->cons[cid].held_slot = -1;
    m_->cons[cid].hb_ms = now_ms();
    if (!took_over) m_->n_consumers += 1;
  }
  unlock_lk(&m_->lk);
  return cid;
}

void Bus::unregister_consumer(int32_t cid) {
  if (cid < 0) return;
  lock_lk(&m_->lk);
  ConsumerEntry& c = m_->cons[cid];
  if (c.active) {
    if (c.held_slot >= 0) {
      SlotHdr& s = *slot(c.held_slot);
      if (s.ref.load() > 0) s.ref.fetch_sub(1);
      c.held_slot = -1;
    }
    c.active = 0;
    m_->n_consumers -= 1;
  }
  unlock_lk(&m_->lk);
}

void Bus::heartbeat(int32_t cid) {
  if (cid < 0) return;
  m_->cons[cid].hb_ms.store(now_ms());
}

int Bus::acquire(int32_t cid, uint64_t last_seq, FrameView* out,
                 int64_t timeout_ms) {
  const int64_t deadline =
      timeout_ms < 0 ? INT64_MAX : now_ms() + timeout_ms;
  for (;;) {
    lock_lk(&m_->lk);
    // smallest published seq > last_seq
    uint64_t best = 0;
    int best_idx = -1;
    for (int i = 0; i < kRingDepth; ++i) {
      uint64_t q = slot(i)->meta.seq;
      if (q > last_seq && (best_idx < 0 || q < best)) {
        best = q;
        best_idx = i;
      }
    }
    if (best_idx >= 0) {
      SlotHdr& s = *slot(best_idx);
      s.ref.fetch_add(1);
      m_->cons[cid].held_slot = best_idx;
      m_->cons[cid].hb_ms.store(now_ms());
      out->slot_idx = best_idx;
      out->consumer_id = cid;
      out->meta = s.meta;
      for (int c = 0; c < kMaxCams; ++c) {
        out->cam[c] = payload_of(best_idx) + c * m_->cam_bytes;
      }
      unlock_lk(&m_->lk);
      return 0;
    }
    unlock_lk(&m_->lk);
    poll_sleep_ms(10);  // 新帧轮询 (原 condvar 100ms chunk, 现更及时)
    heartbeat(cid);
    if (now_ms() >= deadline) return 1;
  }
}

void Bus::release(FrameView* v) {
  lock_lk(&m_->lk);
  SlotHdr& s = *slot(v->slot_idx);
  if (s.ref.load() > 0) s.ref.fetch_sub(1);
  if (m_->cons[v->consumer_id].held_slot == v->slot_idx)
    m_->cons[v->consumer_id].held_slot = -1;
  unlock_lk(&m_->lk);
  v->slot_idx = -1;
}

// Caller holds lk. Steals references of consumers whose heartbeat is
// older than the lease and that hold `slot_idx`.
void Bus::force_stale_locked(int32_t slot_idx, int64_t now_ms_v,
                             int64_t lease_ms) {
  for (int i = 0; i < kMaxConsumers; ++i) {
    ConsumerEntry& c = m_->cons[i];
    if (!c.active || c.held_slot != slot_idx) continue;
    if (now_ms_v - c.hb_ms.load() > lease_ms) {
      c.held_slot = -1;
      if ((*slot(slot_idx)).ref.load() > 0) (*slot(slot_idx)).ref.fetch_sub(1);
      m_->forced_recycles += 1;
    }
  }
}

}  // namespace sp
