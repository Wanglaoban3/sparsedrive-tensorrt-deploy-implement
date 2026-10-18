// sp_inspect - dump ring state once (debug helper for M1).
#include "sp_bus.h"

#include <cstdio>

using namespace sp;

int main(int argc, char** argv) {
  const char* name = argc > 1 ? argv[1] : "m1t";
  char err[256];
  Bus* bus = Bus::open(name, 1600, 900, false, false, err, sizeof(err));
  if (!bus) {
    printf("inspect: %s\n", err);
    return 1;
  }
  const RingMeta* m = bus->meta();
  printf("latest_seq=%lu published=%lu pub_idx=%u n_cons=%d forced=%lu\n",
         (unsigned long)m->latest_seq.load(),
         (unsigned long)m->published_count, m->pub_idx,
         m->n_consumers.load(), (unsigned long)m->forced_recycles);
  // Slots are addressed by slot_stride (see sp_bus.h), never by
  // sizeof(SlotHdr).
  const uint8_t* base = (const uint8_t*)m;
  const SlotHdr* slots = (const SlotHdr*)(base +
      ((sizeof(RingMeta) + 63) / 64 * 64));
  for (int i = 0; i < kRingDepth; ++i) {
    const SlotHdr* s = (const SlotHdr*)((const uint8_t*)slots +
                                        i * m->slot_stride);
    printf("slot %d: ref=%u seq=%lu scene=%u\n", i, s->ref.load(),
           (unsigned long)s->meta.seq, s->meta.scene_id);
  }
  for (int i = 0; i < kMaxConsumers; ++i) {
    if (!m->cons[i].active) continue;
    printf("cons %d: pid=%d held=%d hb_age=%ldms\n", i, m->cons[i].pid,
           m->cons[i].held_slot.load(),
           (long)(now_ms() - m->cons[i].hb_ms.load()));
  }
  return 0;
}
