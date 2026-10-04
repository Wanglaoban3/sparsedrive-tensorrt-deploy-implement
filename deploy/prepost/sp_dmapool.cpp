// sp_dmapool.cpp —— 见 sp_dmapool.h. 设备池 + fd 分发的 CUDA/driver 实现.
// 平台硬约束 (2026-10-03 板上 probe_vmm 实测): cuMemCreate 的 size 必须
// 按 cuMemGetAllocationGranularity(CU_MEM_ALLOC_GRANULARITY_MINIMUM) 对齐
// (本板 2MB), 否则 invalid argument; iGPU 上带 POSIX fd 句柄类型与导出均
// 可用. cudaIpcMemHandle 跨进程 open 在本板不可用 (invalid argument),
// fd + cudaExternalMemory 是唯一设备池共享路径.
#include "sp_dmapool.h"

#include <errno.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace sp {

static const uint32_t kDmaMagic = 0x53445031;  // "SDP1"
static const int kMaxSlots = 16;

int dma_sock_path(const char* ring, char* out, size_t outlen) {
  return snprintf(out, outlen, "/tmp/sp_dma_%s.sock", ring) < (int)outlen
             ? 0
             : -1;
}

// --- 生产端 ---

DmaPoolPub::~DmaPoolPub() {
  stop_serve();
  for (int i = 0; i < (int)n_slots_ && i < kMaxSlots; ++i) {
    if (dev_[i]) cuMemUnmap((CUdeviceptr)dev_[i], slot_bytes_);
    if (mem_[i]) cuMemRelease((CUmemGenericAllocationHandle)mem_[i]);
  }
}

bool DmaPoolPub::create(uint32_t n_slots, uint64_t slot_bytes, char* err,
                        size_t n) {
  if (n_slots == 0 || n_slots > kMaxSlots || slot_bytes == 0) {
    snprintf(err, n, "dmapool: bad geometry %u x %llu", n_slots,
             (unsigned long long)slot_bytes);
    return false;
  }
  if (cudaFree(0) != cudaSuccess) {  // runtime primary ctx 先建
    snprintf(err, n, "dmapool: ctx: %s",
             cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  if (cuInit(0) != CUDA_SUCCESS) {
    snprintf(err, n, "dmapool: cuInit");
    return false;
  }
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  size_t gran = 0;
  if (cuMemGetAllocationGranularity(&gran, &prop,
                                    CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
      CUDA_SUCCESS) {
    snprintf(err, n, "dmapool: granularity query");
    return false;
  }
  // 槽尺寸向上取整到 granularity —— cuMemCreate 对未对齐 size 直接
  // invalid argument (M8 实测); H2D/preproc 仍按 frame_bytes 用前段.
  slot_bytes_ = ((slot_bytes + gran - 1) / gran) * gran;
  n_slots_ = n_slots;
  for (uint32_t i = 0; i < n_slots; ++i) {
    CUmemGenericAllocationHandle h = {};
    if (cuMemCreate(&h, slot_bytes_, &prop, 0) != CUDA_SUCCESS) {
      snprintf(err, n, "dmapool: cuMemCreate slot %u (%llu B): %s", i,
               (unsigned long long)slot_bytes_, cudaGetErrorString(
                   cudaGetLastError()));
      return false;
    }
    mem_[i] = (unsigned long long)h;
    int fd = -1;
    if (cuMemExportToShareableHandle(&fd, h,
                                     CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                                     0) != CUDA_SUCCESS) {
      snprintf(err, n, "dmapool: export slot %u", i);
      return false;
    }
    fd_[i] = fd;
    CUdeviceptr dptr = 0;
    if (cuMemMap(dptr, slot_bytes_, 0, h, 0) != CUDA_SUCCESS) {
      snprintf(err, n, "dmapool: cuMemMap slot %u", i);
      return false;
    }
    dev_[i] = (void*)dptr;
    CUmemAccessDesc acc = {};
    acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc.location.id = 0;
    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemSetAccess(dptr, slot_bytes_, &acc, 1) != CUDA_SUCCESS) {
      snprintf(err, n, "dmapool: cuMemSetAccess slot %u", i);
      return false;
    }
  }
  printf("dmapool: %u slots x %llu B (frame %llu, gran %llu, fd-exportable)\n",
         n_slots, (unsigned long long)slot_bytes_,
         (unsigned long long)slot_bytes, (unsigned long long)gran);
  return true;
}

bool DmaPoolPub::serve(const char* ring, char* err, size_t n) {
  char path[128];
  if (dma_sock_path(ring, path, sizeof(path)) != 0) {
    snprintf(err, n, "dmapool: sock path");
    return false;
  }
  unlink(path);
  listen_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    snprintf(err, n, "dmapool: socket: %s", strerror(errno));
    return false;
  }
  struct sockaddr_un ad = {};
  ad.sun_family = AF_UNIX;
  snprintf(ad.sun_path, sizeof(ad.sun_path), "%s", path);
  if (bind(listen_fd_, (struct sockaddr*)&ad, sizeof(ad)) < 0 ||
      listen(listen_fd_, 4) < 0) {
    snprintf(err, n, "dmapool: bind/listen: %s", strerror(errno));
    return false;
  }
  serving_ = true;
  std::thread th([this, path]() {
    while (serving_) {
      int cfd = accept(listen_fd_, nullptr, nullptr);
      if (cfd < 0) break;
      DmaPoolInfo info = {};
      info.magic = kDmaMagic;
      info.n_slots = n_slots_;
      info.slot_bytes = slot_bytes_;
      struct iovec iov = {&info, sizeof(info)};
      char cbuf[CMSG_SPACE(sizeof(int) * kMaxSlots)] = {};
      struct msghdr msg = {};
      msg.msg_iov = &iov;
      msg.msg_iovlen = 1;
      msg.msg_control = cbuf;
      msg.msg_controllen = sizeof(cbuf);
      struct cmsghdr* cm = CMSG_FIRSTHDR(&msg);
      cm->cmsg_level = SOL_SOCKET;
      cm->cmsg_type = SCM_RIGHTS;
      cm->cmsg_len = CMSG_LEN(sizeof(int) * n_slots_);
      memcpy(CMSG_DATA(cm), fd_, sizeof(int) * n_slots_);
      msg.msg_controllen = cm->cmsg_len;
      ssize_t w = sendmsg(cfd, &msg, 0);
      if (w < 0) w = 0;  // 消费端读失败不影响生产
      (void)w;
      close(cfd);
    }
    unlink(path);
  });
  th.detach();
  printf("dmapool: serving fds at %s\n", path);
  return true;
}

void DmaPoolPub::stop_serve() {
  serving_ = false;
  if (listen_fd_ >= 0) {
    shutdown(listen_fd_, SHUT_RDWR);
    close(listen_fd_);
    listen_fd_ = -1;
  }
}

// --- 消费端 ---

DmaPoolSub::~DmaPoolSub() {
  // 映射随进程结束回收; cudaDestroyExternalMemory 已在 attach 后调用
  // (GetMappedBuffer 的映射独立于 external memory 对象生命周期).
}

bool DmaPoolSub::attach(const char* ring, const DmaPoolInfo& want, char* err,
                        size_t n) {
  char path[128];
  if (dma_sock_path(ring, path, sizeof(path)) != 0) {
    snprintf(err, n, "dmapool: sock path");
    return false;
  }
  int cfd = -1;
  for (int t = 0; t < 100 && cfd < 0; ++t) {  // 最多 ~10s (发布端建池时序)
    cfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (cfd < 0) break;
    struct sockaddr_un ad = {};
    ad.sun_family = AF_UNIX;
    snprintf(ad.sun_path, sizeof(ad.sun_path), "%s", path);
    socklen_t alen = (socklen_t)(offsetof(struct sockaddr_un, sun_path) +
                                 strlen(path));
    if (connect(cfd, (struct sockaddr*)&ad, alen) < 0) {
      close(cfd);
      cfd = -1;
      usleep(100000);
    }
  }
  if (cfd < 0) {
    snprintf(err, n, "dmapool: connect %s: %s", path, strerror(errno));
    return false;
  }
  DmaPoolInfo info = {};
  struct iovec iov = {&info, sizeof(info)};
  char cbuf[CMSG_SPACE(sizeof(int) * kMaxSlots)] = {};
  struct msghdr msg = {};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = cbuf;
  msg.msg_controllen = sizeof(cbuf);
  ssize_t r = recvmsg(cfd, &msg, 0);
  close(cfd);
  if (r < (ssize_t)sizeof(info)) {
    snprintf(err, n, "dmapool: bad handshake (%zd B)", r);
    return false;
  }
  if (info.magic != kDmaMagic || info.n_slots != want.n_slots ||
      info.slot_bytes != want.slot_bytes) {
    snprintf(err, n, "dmapool: geometry mismatch uds %u/%llu vs ring %u/%llu",
             info.n_slots, (unsigned long long)info.slot_bytes, want.n_slots,
             (unsigned long long)want.slot_bytes);
    return false;
  }
  struct cmsghdr* cm = CMSG_FIRSTHDR(&msg);
  if (!cm || cm->cmsg_len < CMSG_LEN(sizeof(int) * info.n_slots)) {
    snprintf(err, n, "dmapool: no fds in cmsg");
    return false;
  }
  int fds[kMaxSlots] = {};
  memcpy(fds, CMSG_DATA(cm), sizeof(int) * info.n_slots);
  n_slots_ = info.n_slots;
  for (uint32_t i = 0; i < n_slots_; ++i) {
    // fd 所有权随导入转移给内核 —— 成功后不 close
    cudaExternalMemoryHandleDesc hd = {};
    hd.type = cudaExternalMemoryHandleTypeOpaqueFd;
    hd.size = info.slot_bytes;
    hd.handle.fd = fds[i];
    cudaExternalMemory_t ext = nullptr;
    if (cudaImportExternalMemory(&ext, &hd) != cudaSuccess) {
      snprintf(err, n, "dmapool: import slot %u: %s", i,
               cudaGetErrorString(cudaGetLastError()));
      return false;
    }
    cudaExternalMemoryBufferDesc bd = {};
    bd.size = info.slot_bytes;
    void* p = nullptr;
    if (cudaExternalMemoryGetMappedBuffer(&p, ext, &bd) != cudaSuccess) {
      snprintf(err, n, "dmapool: map slot %u: %s", i,
               cudaGetErrorString(cudaGetLastError()));
      return false;
    }
    cudaDestroyExternalMemory(ext);  // 映射在 destroy 后仍有效
    dev_[i] = (const uint8_t*)p;
  }
  printf("dmapool: attached %u slots x %llu B (device ptrs)\n",
         n_slots_, (unsigned long long)info.slot_bytes);
  return true;
}

}  // namespace sp
