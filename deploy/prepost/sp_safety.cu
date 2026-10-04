// sp_safety.cu —— M-PROD B1: 设备侧状态发散探测 kernel + NaN 注入钩子.
// spec: docs/superpowers/specs/2026-10-03-mprod-hardening-design.md §6 B1.
//
// absmax: 对单个状态缓冲求 max|x|, NaN 按无穷大计 (isnan→inf, 下游统一
// "非有限或超界=发散"). 逐块 shared 归约 + atomicMax(int): 非负 float 的
// 位型与 int 同序, 单标量 D2H 即得结果 —— 满足 "设备侧 kernel → D2H
// 几十字节" 的带宽预算, 不做整状态 D2H.
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sp {
namespace saf {

__global__ void absmax_kernel(const float* p, size_t n, int* out) {
  __shared__ float sh[256];
  float m = 0.f;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    float a = fabsf(p[i]);
    if (isnan(a)) a = CUDART_INF_F;  // NaN → inf, 非有限值必现形
    m = fmaxf(m, a);
  }
  sh[threadIdx.x] = m;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicMax(out, __float_as_int(sh[0]));  // 非负 float 与 int 同序
}

// 结果标量清零 → 逐缓冲归约 (eng_stream 上与推理定序, 谁先谁后都正确:
// 读的是"本帧反馈后的状态", 在反馈 memcpy 之后 launch 即可)
void absmax_zero_async(int* dev_out, cudaStream_t s) {
  cudaMemsetAsync(dev_out, 0, sizeof(int), s);
}

void absmax_async(const float* p, size_t n, int* dev_out, cudaStream_t s) {
  if (!p || !n) return;
  int blocks = (int)((n + 255) / 256);
  if (blocks < 1) blocks = 1;
  if (blocks > 2048) blocks = 2048;
  absmax_kernel<<<blocks, 256, 0, s>>>(p, n, dev_out);
}

// 测试钩子 SP_INJECT_NAN: 往状态缓冲前 n 个元素写 NaN (默认关).
__global__ void inject_nan_kernel(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = CUDART_NAN_F;
}

void inject_nan_async(float* p, int n, cudaStream_t s) {
  if (!p || !n) return;
  if (n > 1024) n = 1024;
  inject_nan_kernel<<<1, 1024, 0, s>>>(p, n);
}

}  // namespace saf
}  // namespace sp
