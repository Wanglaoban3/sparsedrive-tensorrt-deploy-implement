#include "file_source.h"

#include <cerrno>
#include <cstring>
#include <thread>
#include <chrono>

using namespace sp;

// ---- 极简 jsonl 解析 (固定 schema, 非通用 JSON) ----

static bool find_key(const std::string& s, const char* key, size_t* pos) {
  std::string pat = "\"" + std::string(key) + "\"";
  size_t p = s.find(pat);
  if (p == std::string::npos) return false;
  p = s.find(':', p + pat.size());
  if (p == std::string::npos) return false;
  *pos = p + 1;
  return true;
}

static bool parse_u64(const std::string& s, size_t pos, uint64_t* v) {
  while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) ++pos;
  if (pos >= s.size() || s[pos] < '0' || s[pos] > '9') return false;
  uint64_t x = 0;
  while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9')
    x = x * 10 + (uint64_t)(s[pos++] - '0');
  *v = x;
  return true;
}

static bool parse_str(const std::string& s, size_t pos, std::string* v,
                      size_t* next) {
  while (pos < s.size() && s[pos] != '"') ++pos;
  if (pos >= s.size()) return false;
  size_t e = s.find('"', pos + 1);
  if (e == std::string::npos) return false;
  *v = s.substr(pos + 1, e - pos - 1);
  *next = e + 1;
  return true;
}

bool sp::parse_manifest_line(const std::string& line, ManifestEntry* e,
                             char* err, size_t errlen) {
  size_t p;
  uint64_t v;
  if (!find_key(line, "frame", &p) || !parse_u64(line, p, &v)) {
    snprintf(err, errlen, "manifest line missing \"frame\"");
    return false;
  }
  if (find_key(line, "ts_ns", &p) && parse_u64(line, p, &v))
    e->ts_ns = v;
  if (find_key(line, "scene", &p) && parse_u64(line, p, &v))
    e->scene_id = (uint32_t)v;
  if (!find_key(line, "cams", &p)) {
    snprintf(err, errlen, "manifest line missing \"cams\"");
    return false;
  }
  size_t at = line.find('[', p);
  if (at == std::string::npos) {
    snprintf(err, errlen, "manifest \"cams\" not an array");
    return false;
  }
  for (int c = 0; c < kMaxCams; ++c) {
    size_t nxt;
    if (!parse_str(line, at, &e->cam_path[c], &nxt)) {
      snprintf(err, errlen, "manifest cams[%d] missing", c);
      return false;
    }
    at = nxt;
  }
  return true;
}

// ---- FileReplaySource ----

int FileReplaySource::open(const SourceConfig& cfg, char* err,
                           size_t errlen) {
  cfg_ = cfg;
  FILE* f = fopen(cfg.manifest_path.c_str(), "rb");
  if (!f) {
    snprintf(err, errlen, "open manifest %s: %s",
             cfg.manifest_path.c_str(), strerror(errno));
    return -1;
  }
  char line[4096];
  uint32_t w = 0, h = 0;
  int lineno = 0;
  while (fgets(line, sizeof(line), f)) {
    lineno += 1;
    std::string s(line);
    if (s.empty() || s[0] == '#' || s[0] == '\n') continue;
    // 可选尺寸头: {"w":1600,"h":900}
    size_t p;
    uint64_t v;
    if (find_key(s, "w", &p) && parse_u64(s, p, &v) && !find_key(s, "cams", &p)) {
      if (!find_key(s, "w", &p) || !parse_u64(s, p, &v)) {
        snprintf(err, errlen, "manifest:%d bad w", lineno);
        fclose(f);
        return -1;
      }
      w = (uint32_t)v;
      if (!find_key(s, "h", &p) || !parse_u64(s, p, &v)) {
        snprintf(err, errlen, "manifest:%d missing h", lineno);
        fclose(f);
        return -1;
      }
      h = (uint32_t)v;
      continue;
    }
    ManifestEntry e;
    e.ts_ns = 0;
    e.scene_id = 0;
    if (!parse_manifest_line(s, &e, err, errlen)) {
      fclose(f);
      return -1;
    }
    entries_.push_back(e);
  }
  fclose(f);
  if (entries_.empty()) {
    snprintf(err, errlen, "manifest has no frames");
    return -1;
  }
  if (w == 0 || h == 0) {
    snprintf(err, errlen, "manifest missing {\"w\",\"h\"} size header");
    return -1;
  }
  width_ = w;
  height_ = h;
  cam_bytes_ = (size_t)w * h * 3 / 2;
  frame_bytes_ = cam_bytes_ * kMaxCams;
  for (int i = 0; i < kPool; ++i) {
    buf_[i] = (uint8_t*)malloc(frame_bytes_);
    if (!buf_[i]) {
      snprintf(err, errlen, "malloc %zu failed", frame_bytes_);
      return -1;
    }
  }
  ref_[0].store(0);
  ref_[1].store(0);
  return 0;
}

int FileReplaySource::start() {
  next_entry_ = 0;
  loop_base_seq_ = 0;
  return 0;
}

int FileReplaySource::load_frame(size_t idx, int b, char* err,
                                 size_t errlen) {
  for (int c = 0; c < kMaxCams; ++c) {
    std::string path = cfg_.data_root.empty()
        ? entries_[idx].cam_path[c]
        : cfg_.data_root + "/" + entries_[idx].cam_path[c];
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
      snprintf(err, errlen, "frame %zu cam %d open %s: %s", idx, c,
               path.c_str(), strerror(errno));
      return -1;
    }
    size_t got = fread(buf_[b] + c * cam_bytes_, 1, cam_bytes_, f);
    fclose(f);
    if (got != cam_bytes_) {
      snprintf(err, errlen, "frame %zu cam %d short read %zu/%zu", idx, c,
               got, cam_bytes_);
      return -1;
    }
  }
  return 0;
}

int FileReplaySource::acquire(FrameView& out, int timeout_ms) {
  if (entries_.empty()) return -1;
  if (next_entry_ >= entries_.size()) {
    if (!cfg_.loop) return 2;
    next_entry_ = 0;
    loop_base_seq_ += entries_.size();
  }
  int64_t deadline = now_ms() + timeout_ms;
  int b = cur_ ^ 1;
  while (ref_[b].load() != 0) {
    if (now_ms() >= deadline) return 1;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  char err[256];
  if (load_frame(next_entry_, b, err, sizeof(err)) != 0) {
    fprintf(stderr, "filesrc: %s\n", err);
    return -1;
  }
  cur_ = b;
  const ManifestEntry& e = entries_[next_entry_];
  memset(&out, 0, sizeof(out));
  out.slot_idx = b;  // 池下标借 slot_idx 携带, release 据此归还
  out.consumer_id = -1;
  out.meta.seq = loop_base_seq_ + next_entry_ + 1;
  out.meta.group_ts_ns = (int64_t)e.ts_ns;
  for (int c = 0; c < kMaxCams; ++c)
    out.meta.cam_ts_ns[c] = (int64_t)e.ts_ns;
  out.meta.scene_id = e.scene_id;
  // 源状态位: 文件回放源恒定整帧 OK + 6 路全 OK (布局不变, 语义见 sp_bus.h)
  out.meta.flags = kFlagSourceOk | kFlagAllCamsOk;
  out.meta.width = width_;
  out.meta.height = height_;
  out.meta.cam_bytes = (uint32_t)cam_bytes_;
  out.meta.frame_bytes = (uint32_t)frame_bytes_;
  for (int c = 0; c < kMaxCams; ++c)
    out.cam[c] = buf_[b] + c * cam_bytes_;
  next_entry_ += 1;
  ref_[b].store(1);
  return 0;
}

void FileReplaySource::release(FrameView& view) {
  if (view.slot_idx >= 0 && view.slot_idx < kPool)
    ref_[view.slot_idx].store(0);
}

int FileReplaySource::stop() {
  for (int i = 0; i < kPool; ++i) {
    free(buf_[i]);
    buf_[i] = nullptr;
  }
  return 0;
}
