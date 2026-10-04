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
    // Existing ring with matching geometry: attach instead of re-init, so a
    // crashed publisher does not wipe consumer state mid-run.
    do_init = false;
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
    pthread_mutexattr_t ma;
    pthread_mutexattr_init(&ma);
    pthread_mutexattr_setpshared(&ma, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&m->mu, &ma);
    pthread_condattr_t ca;
    pthread_condattr_init(&ca);
    pthread_condattr_setpshared(&ca, PTHREAD_PROCESS_SHARED);
    pthread_cond_init(&m->cv_free, &ca);
    pthread_cond_init(&m->cv_frame, &ca);
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

uint8_t* Bus::claim(int64_t timeout_ms) {
  const int64_t lease = lease_ms_from_env();
  const int64_t deadline = now_ms() + timeout_ms;
  pthread_mutex_lock(&m_->mu);
  for (;;) {
    SlotHdr& s = *slot(m_->pub_idx);
    if (s.ref.load() == 0) {
      s.meta.seq = 0;  // invalidate stale seq before refilling
      const int32_t idx = m_->pub_idx;
#ifdef SP_BUS_DEBUG
      fprintf(stderr, "dbg claim idx=%d payload=%p meta=%p delta=%ld "
              "stride=%lu frame=%u\n", idx, (void*)payload_of(idx),
              (void*)slot(idx), (long)((uint8_t*)payload_of(idx) -
              (uint8_t*)slot(idx)), (unsigned long)m_->slot_stride,
              m_->frame_bytes);
#endif
      pthread_mutex_unlock(&m_->mu);
      return payload_of(idx);
    }
    force_stale_locked(m_->pub_idx, now_ms(), lease);
    if (s.ref.load() == 0) {
      s.meta.seq = 0;
      const int32_t idx = m_->pub_idx;
      pthread_mutex_unlock(&m_->mu);
      return payload_of(idx);
    }
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_nsec += 100 * 1000000;  // 100 ms chunk
    if (ts.tv_nsec >= 1000000000) {
      ts.tv_sec += 1;
      ts.tv_nsec -= 1000000000;
    }
    pthread_cond_timedwait(&m_->cv_free, &m_->mu, &ts);
    if (timeout_ms >= 0 && now_ms() >= deadline) break;
  }
  pthread_mutex_unlock(&m_->mu);
  return nullptr;
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
  pthread_mutex_lock(&m_->mu);
  s.meta = meta;
  s.meta.seq = m_->latest_seq + 1;
  m_->latest_seq = s.meta.seq;
  m_->published_count += 1;
  m_->pub_idx = (idx + 1) % kRingDepth;
  pthread_cond_broadcast(&m_->cv_frame);
  pthread_mutex_unlock(&m_->mu);
}

int32_t Bus::register_consumer() {
  pthread_mutex_lock(&m_->mu);
  int32_t cid = -1;
  for (int i = 0; i < kMaxConsumers; ++i) {
    if (!m_->cons[i].active) {
      m_->cons[i].active = 1;
      m_->cons[i].pid = (int32_t)getpid();
      m_->cons[i].held_slot = -1;
      m_->cons[i].hb_ms = now_ms();
      cid = i;
      break;
    }
  }
  if (cid >= 0) m_->n_consumers += 1;
  pthread_mutex_unlock(&m_->mu);
  return cid;
}

void Bus::unregister_consumer(int32_t cid) {
  if (cid < 0) return;
  pthread_mutex_lock(&m_->mu);
  ConsumerEntry& c = m_->cons[cid];
  if (c.active) {
    if (c.held_slot >= 0) {
      SlotHdr& s = *slot(c.held_slot);
      if (s.ref.load() > 0) s.ref.fetch_sub(1);
      if (s.ref.load() == 0) pthread_cond_broadcast(&m_->cv_free);
      c.held_slot = -1;
    }
    c.active = 0;
    m_->n_consumers -= 1;
  }
  pthread_mutex_unlock(&m_->mu);
}

void Bus::heartbeat(int32_t cid) {
  if (cid < 0) return;
  m_->cons[cid].hb_ms.store(now_ms());
}

int Bus::acquire(int32_t cid, uint64_t last_seq, FrameView* out,
                 int64_t timeout_ms) {
  const int64_t deadline =
      timeout_ms < 0 ? INT64_MAX : now_ms() + timeout_ms;
  pthread_mutex_lock(&m_->mu);
  for (;;) {
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
      pthread_mutex_unlock(&m_->mu);
      return 0;
    }
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_nsec += 100 * 1000000;
    if (ts.tv_nsec >= 1000000000) {
      ts.tv_sec += 1;
      ts.tv_nsec -= 1000000000;
    }
    pthread_cond_timedwait(&m_->cv_frame, &m_->mu, &ts);
    heartbeat(cid);
    if (now_ms() >= deadline) break;
  }
  pthread_mutex_unlock(&m_->mu);
  return 1;
}

void Bus::release(FrameView* v) {
  pthread_mutex_lock(&m_->mu);
  SlotHdr& s = *slot(v->slot_idx);
  if (s.ref.load() > 0) s.ref.fetch_sub(1);
  if (m_->cons[v->consumer_id].held_slot == v->slot_idx)
    m_->cons[v->consumer_id].held_slot = -1;
  if (s.ref.load() == 0) pthread_cond_broadcast(&m_->cv_free);
  pthread_mutex_unlock(&m_->mu);
  v->slot_idx = -1;
}

// Caller holds m_->mu. Steals references of consumers whose heartbeat is
// older than the lease and that hold `slot_idx`.
void Bus::force_stale_locked(int32_t slot_idx, int64_t now_ms_v,
                             int64_t lease_ms) {
  bool changed = false;
  for (int i = 0; i < kMaxConsumers; ++i) {
    ConsumerEntry& c = m_->cons[i];
    if (!c.active || c.held_slot != slot_idx) continue;
    if (now_ms_v - c.hb_ms.load() > lease_ms) {
      c.held_slot = -1;
      if ((*slot(slot_idx)).ref.load() > 0) (*slot(slot_idx)).ref.fetch_sub(1);
      m_->forced_recycles += 1;
      changed = true;
    }
  }
  if (changed && (*slot(slot_idx)).ref.load() == 0)
    pthread_cond_broadcast(&m_->cv_free);
}

}  // namespace sp
