// M8: 可共享设备内存池 (fd 旁路发布) —— dmabuf/NvBufSurface 零拷贝采集的
// CUDA 模拟器实现. 生产端 cuMemCreate + cuMemExportToShareableHandle 建
// 每槽一块的设备内存池, fd 经 UDS SCM_RIGHTS 一次性分发给消费端; 消费端
// cudaExternalMemoryImportFd(OpaqueFd) 映射出设备指针, 直喂现有 preproc
// kernel (kernel 零改动). 池内容只走设备侧, 绕开 mapped-shm 的 C2C 读.
// 平台注记 (2026-10-03 板上探针实测, probe_vmm): Orin iGPU 支持 cuMemCreate
// VMM + POSIX fd 导出, 但 size 必须按 granularity 对齐 (本板 = 2MB), 否则
// cuMemCreate 报 invalid argument —— 故槽尺寸向上取整到 granularity,
// DmaPoolInfo.slot_bytes 携带对齐后尺寸, H2D/preproc 仍按 frame_bytes 用.
// 换真相机 (NvBufSurface/V4L2 dmabuf) 时: 池由 ISP 分配, 消费端仅把
// handle 类型 OpaqueFd 换成 DmaBufFd (pool/fd/fence 机制完全复用);
// 槽位回收纪律复用 sp_bus 的 claim/release+fence (M1.5 硬要求不变).
#ifndef SP_DMAPOOL_H_
#define SP_DMAPOOL_H_

#include <stddef.h>
#include <stdint.h>

namespace sp {

struct DmaPoolInfo {          // UDS 首包 + RingMeta 尾字段同构
  uint32_t magic;             // 'SDP1' = 0x53445031
  uint32_t n_slots;
  uint64_t slot_bytes;        // 对齐后的槽尺寸 (>= frame_bytes)
  uint32_t width;
  uint32_t height;
};

// 路径约定: /run/sp/sp_dma_<ring>.sock (Phase D 自 /tmp 迁入; 生产端监听,
// 消费端连接取 fd).
int dma_sock_path(const char* ring, char* out, size_t outlen);

class DmaPoolPub {            // --- 生产端 (sp_filesrc --dma) ---
 public:
  ~DmaPoolPub();
  // cudaFree(0) + cuInit + 逐槽 cuMemCreate(对齐尺寸)/导出 fd/map/RW-access.
  bool create(uint32_t n_slots, uint64_t slot_bytes, char* err, size_t n);
  // 后台线程: accept 每个连入的消费端, 发 DmaPoolInfo + 全部 fd.
  bool serve(const char* ring, char* err, size_t n);
  void stop_serve();
  void* dev(int slot) const { return dev_[slot]; }   // CUdeviceptr as void*
  uint64_t slot_bytes() const { return slot_bytes_; }  // 对齐后

 private:
  uint32_t n_slots_ = 0;
  uint64_t slot_bytes_ = 0;   // 对齐后
  void* dev_[16] = {};        // CUdeviceptr per slot
  int fd_[16] = {};           // shareable fd (生命周期 = 进程)
  unsigned long long mem_[16] = {};  // CUmemGenericAllocationHandle
  int listen_fd_ = -1;
  bool serving_ = false;
};

class DmaPoolSub {            // --- 消费端 (sp_modelnode --dma) ---
 public:
  ~DmaPoolSub();
  // 连接 (重试至多 10s) → 收 DmaPoolInfo+n_slots 个 fd → 逐 fd 导入
  // cudaExternalMemory(OpaqueFd) → 映射设备指针. 校验 info 与 want 一致.
  bool attach(const char* ring, const DmaPoolInfo& want, char* err,
              size_t n);
  const uint8_t* dev(int slot) const { return dev_[slot]; }

 private:
  uint32_t n_slots_ = 0;
  const uint8_t* dev_[16] = {};  // external-memory 映射设备指针 (只读约定)
};

}  // namespace sp

#endif  // SP_DMAPOOL_H_
