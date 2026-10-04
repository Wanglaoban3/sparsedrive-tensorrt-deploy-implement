// IImageSource: 图像源抽象 = 同步帧组的生产者, 可替换.
// 首个实现 FileReplaySource (manifest 驱动的 NV12 文件回放);
// 目标实现 dmabuf/NvBufSurface 采集源 (M5), model_node 一行不改.
// 帧内存归源所有; acquire 交出只读视图并持引用, release 归还.
#ifndef SP_IMAGE_SOURCE_H_
#define SP_IMAGE_SOURCE_H_

#include <stddef.h>
#include <string>

#include "sp_bus.h"

namespace sp {

struct SourceConfig {
  std::string manifest_path;  // jsonl: 每行一帧 {"frame","scene","ts_ns","cams":[6]}
  std::string data_root;      // 相对路径前缀 (本地/板上各挂各的根)
  bool loop = false;          // 播完从头再来 (seq 连续递增, 不回绕)
};

class IImageSource {
 public:
  virtual ~IImageSource() = default;
  virtual int open(const SourceConfig& cfg, char* err, size_t errlen) = 0;
  virtual int start() = 0;
  // 阻塞取一组同步帧 (6 路只读平面, 每路 cam_bytes = W*H*3/2, Y|UV 连续).
  // 返回 0 成功; 超时 1; 源结束 2; 错误 -1. 视图在 release 前有效.
  virtual int acquire(FrameView& out, int timeout_ms) = 0;
  virtual void release(FrameView& view) = 0;
  virtual int stop() = 0;
};

}  // namespace sp

#endif  // SP_IMAGE_SOURCE_H_
