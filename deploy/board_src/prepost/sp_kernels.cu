// sp_kernels.cu — M1.5 fence 测试 kernel(模块级编译,见 AGENTS.md 编译纪律)
#include "sp_kernels.h"

#include <cuda_runtime.h>

namespace {

__global__ void slot_sum_kernel(const uint8_t* __restrict__ data, size_t n,
                                uint64_t* __restrict__ out) {
  // 网格跨步逐字节求和;12.9MB @ Orin 统一内存 ≈ 亚毫秒, 足够探测覆写
  uint64_t s = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    s += data[i];
  }
  // 原子加进 out(启动前由 host 清零);整数加精确且可结合
  atomicAdd((unsigned long long*)out, s);
}

__global__ void delay_kernel(unsigned long long ns) {
  // __nanosleep 单次实现有 ~1ms 级上限, 循环凑足真实时间;
  // clock64 在空闲/门控时钟下走得极慢, 不能用
  while (ns > 0) {
    unsigned int step = ns > 1000000ull ? 1000000u : (unsigned int)ns;
    __nanosleep(step);
    ns -= step;
  }
}

}  // namespace

void slot_sum_launch(const uint8_t* data, size_t n, uint64_t* out,
                     void* stream) {
  const int blocks = 128, threads = 256;
  slot_sum_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(data, n,
                                                                out);
}

void delay_launch(unsigned long long cycles, void* stream) {
  delay_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(cycles);
}
