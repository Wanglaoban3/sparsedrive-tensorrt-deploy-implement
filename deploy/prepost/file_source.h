// FileReplaySource: manifest jsonl 驱动的 NV12 文件回放源.
// 每行一帧: {"frame":0,"scene":0,"ts_ns":1533151603547590000,
//            "cams":["f0.nv12",...,"f5.nv12"]}   (相对 data_root)
// 双缓冲轮换: acquire 读盘到空闲缓冲, release 归还; 消费端串行
// acquire→release 时零等待. 读盘 78MB/帧, 页缓存命中 ~20-40ms.
#ifndef SP_FILE_SOURCE_H_
#define SP_FILE_SOURCE_H_

#include <atomic>
#include <cstdio>
#include <string>
#include <vector>

#include "image_source.h"

namespace sp {

struct ManifestEntry {
  uint64_t ts_ns;
  uint32_t scene_id;
  std::string cam_path[kMaxCams];
};

class FileReplaySource : public IImageSource {
 public:
  int open(const SourceConfig& cfg, char* err, size_t errlen) override;
  int start() override;
  int acquire(FrameView& out, int timeout_ms) override;
  void release(FrameView& view) override;
  int stop() override;
  size_t num_entries() const { return entries_.size(); }

 private:
  int load_frame(size_t entry_idx, int buf_idx, char* err, size_t errlen);

  SourceConfig cfg_;
  std::vector<ManifestEntry> entries_;
  size_t next_entry_ = 0;       // 下一条 manifest 下标 (loop 时回绕)
  uint64_t loop_base_seq_ = 0;  // 回绕后的 seq 基址, 保证 seq 单调
  uint32_t width_ = 0, height_ = 0;
  size_t cam_bytes_ = 0, frame_bytes_ = 0;
  static const int kPool = 2;
  uint8_t* buf_[kPool] = {nullptr, nullptr};
  std::atomic<int> ref_[kPool];  // 0 = 空闲
  int cur_ = 0;
};

// 极简 jsonl 解析: 只认固定 schema 的四个键, 供回放清单用, 不做通用 JSON.
bool parse_manifest_line(const std::string& line, ManifestEntry* e,
                         char* err, size_t errlen);

}  // namespace sp

#endif  // SP_FILE_SOURCE_H_
