// preproc.cu — M2 CUDA 前处理 (模块级编译纪律: nvcc -arch=sm_87 -c)
// 数值口径 = deploy/preproc_ref.py (Pillow 12.3 Resample.c 8bpc 定点移植,
// 本地 selftest 已对 1600x900→704x396 位级对齐 diff px=0).
#include "preproc.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace sp {

namespace {

// Pillow Resample.c: kernel_filter cubic_conv (a = -0.5), double 精度
__host__ __device__ double pil_cubic(double x) {
  const double a = -0.5;
  x = fabs(x);
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
  return 0.0;
}

__host__ int ctrunc(double v) { return (int)v; }  // C 截断 (向零)

}  // namespace

// ---- host: 造表 (与 deploy/preproc_ref.py pil_coeffs_int 逐位一致) ----

bool build_axis_lut(int in_size, int out_size, AxisLUT* lut) {
  const double scale = (double)in_size / out_size;
  const double filterscale = scale >= 1.0 ? scale : 1.0;
  const double support = 2.0 * filterscale;
  const int ksize = (int)ceil(support) * 2 + 1;
  lut->ksize = ksize;
  lut->out_n = out_size;
  lut->xmin = (int32_t*)malloc(sizeof(int32_t) * out_size);
  lut->cnt = (int32_t*)malloc(sizeof(int32_t) * out_size);
  lut->kint = (int32_t*)malloc(sizeof(int32_t) * out_size * ksize);
  if (!lut->xmin || !lut->cnt || !lut->kint) return false;
  memset(lut->kint, 0, sizeof(int32_t) * out_size * ksize);
  for (int xx = 0; xx < out_size; ++xx) {
    const double center = (xx + 0.5) * scale;
    int xmin = ctrunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = ctrunc(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    const int cnt = xmax - xmin;
    double sum = 0.0;
    for (int j = 0; j < cnt; ++j)
      sum += pil_cubic(((j + xmin) - center + 0.5) / filterscale);
    lut->xmin[xx] = xmin;
    lut->cnt[xx] = cnt;
    for (int j = 0; j < cnt; ++j) {
      double w = pil_cubic(((j + xmin) - center + 0.5) / filterscale);
      if (sum != 0.0) w /= sum;
      // 8bpc: round half away from zero into 2^22 fixed point
      const double kf = w * (double)(1 << 22);
      const double r = kf < 0.0 ? kf - 0.5 : kf + 0.5;
      lut->kint[xx * ksize + j] = (int32_t)r;
    }
  }
  return true;
}

// ---- kernels ----

namespace {

// NV12 → RGB u8 (BT.601 full-range), 与 numpy 参考同序同精度 (float32)
__global__ void nv12_to_rgb_kernel(const uint8_t* __restrict__ nv,
                                   size_t cam_bytes, int w, int h,
                                   uint8_t* __restrict__ rgb) {
  const int c = blockIdx.z;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const uint8_t* cam = nv + (size_t)c * cam_bytes;
  const float yy = (float)cam[(size_t)y * w + x];
  const uint8_t* uv = cam + (size_t)w * h;
  const size_t ui = (size_t)(y / 2) * (w / 2) + (x / 2);
  const float u = (float)uv[ui * 2] - 128.0f;
  const float v = (float)uv[ui * 2 + 1] - 128.0f;
  float r = yy + 1.402f * v;
  float g = yy - 0.344136f * u - 0.714136f * v;
  float b = yy + 1.772f * u;
  uint8_t* o = rgb + ((size_t)c * h * w + (size_t)y * w + x) * 3;
  o[0] = (uint8_t)fminf(255.f, fmaxf(0.f, r + 0.5f));
  o[1] = (uint8_t)fminf(255.f, fmaxf(0.f, g + 0.5f));
  o[2] = (uint8_t)fminf(255.f, fmaxf(0.f, b + 0.5f));
}

// 水平通带: RGB u8 (HWC) → temp u8, 固定点整数累加 (Pillow 8bpc 位级同构)
// temp: [c][h][ow][3]
__global__ void horiz_pass_kernel(const uint8_t* __restrict__ rgb,
                                  int w, int h, int ow,
                                  const int32_t* __restrict__ lut,  // xmin|cnt|kint
                                  int ksize, uint8_t* __restrict__ tmp) {
  const int c = blockIdx.z;
  const int xx = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (xx >= ow || y >= h) return;
  // LUT 打包: [xmin 块 | cnt 块 | kint 块]
  const int xmin = lut[xx];
  const int cnt = lut[ow + xx];
  const int32_t* kint = lut + (size_t)ow * 2 + (size_t)xx * ksize;
  int acc[3] = {0, 0, 0};
  const uint8_t* row = rgb + ((size_t)c * h + y) * (size_t)w * 3;
  for (int j = 0; j < cnt; ++j) {
    const uint8_t* px = row + (size_t)(xmin + j) * 3;
    const int32_t k = kint[j];
    acc[0] += k * px[0];
    acc[1] += k * px[1];
    acc[2] += k * px[2];
  }
  uint8_t* o = tmp + ((size_t)c * h + y) * (size_t)ow * 3 + (size_t)xx * 3;
  o[0] = (uint8_t)min(255, max(0, (acc[0] + (1 << 21)) >> 22));
  o[1] = (uint8_t)min(255, max(0, (acc[1] + (1 << 21)) >> 22));
  o[2] = (uint8_t)min(255, max(0, (acc[2] + (1 << 21)) >> 22));
}

// 垂直通带 + crop 行 + 归一化 → f32 [c][3][oh][ow] (引擎布局)
// lut = 竖轴系数表 (按 resized_h 全高建表, out 行只用 [crop_y, crop_y+oh))
__global__ void vert_norm_kernel(const uint8_t* __restrict__ tmp,
                                 int src_h, int resized_h, int ow, int oh,
                                 int crop_y,
                                 const int32_t* __restrict__ lut, int ksize,
                                 const float* __restrict__ mean,
                                 const float* __restrict__ stdv,
                                 float* __restrict__ out) {
  const int c = blockIdx.z;
  const int xx = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = blockIdx.y * blockDim.y + threadIdx.y;  // 输出行 0..oh-1
  if (xx >= ow || v >= oh) return;
  const int yy = v + crop_y;  // 缩放后坐标系的行
  // LUT 打包: [xmin 块 | cnt 块 | kint 块] (out_n = resized_h)
  const int ymin = lut[yy];
  const int cnt = lut[resized_h + yy];
  const int32_t* kint = lut + (size_t)resized_h * 2 + (size_t)yy * ksize;
  int acc[3] = {0, 0, 0};
  for (int j = 0; j < cnt; ++j) {
    const uint8_t* px = tmp + ((size_t)c * src_h + (ymin + j)) *
                                    (size_t)ow * 3 + (size_t)xx * 3;
    const int32_t k = kint[j];
    acc[0] += k * px[0];
    acc[1] += k * px[1];
    acc[2] += k * px[2];
  }
  float* o = out + (((size_t)c * 3) * oh + v) * ow + xx;
  const size_t plane = (size_t)oh * ow;
  o[0] = ((uint8_t)min(255, max(0, (acc[0] + (1 << 21)) >> 22)) -
          mean[0]) / stdv[0];
  o[plane] = ((uint8_t)min(255, max(0, (acc[1] + (1 << 21)) >> 22)) -
              mean[1]) / stdv[1];
  o[plane * 2] = ((uint8_t)min(255, max(0, (acc[2] + (1 << 21)) >> 22)) -
                  mean[2]) / stdv[2];
}

}  // namespace

Preproc::~Preproc() {
  cudaFree(d_rgb_);
  cudaFree(d_tmp_);
  cudaFree(d_lutx_);
  cudaFree(d_luty_);
  cudaFree(d_mean_);
  cudaFree(d_std_);
  free(lut_x_.xmin);
  free(lut_x_.cnt);
  free(lut_x_.kint);
  free(lut_y_.xmin);
  free(lut_y_.cnt);
  free(lut_y_.kint);
}

bool Preproc::init(const PreprocParams& p, char* err, size_t errlen) {
  p_ = p;
  if (!build_axis_lut(p.src_w, p.resized_w, &lut_x_) ||
      !build_axis_lut(p.src_h, p.resized_h, &lut_y_)) {
    snprintf(err, errlen, "build_axis_lut failed");
    return false;
  }
  cudaError_t e;
  e = cudaMalloc(&d_rgb_, (size_t)6 * p.src_h * p.src_w * 3);
  if (e != cudaSuccess) {
    snprintf(err, errlen, "cudaMalloc rgb: %s", cudaGetErrorString(e));
    return false;
  }
  e = cudaMalloc(&d_tmp_, (size_t)6 * p.src_h * p.resized_w * 3);
  if (e != cudaSuccess) {
    snprintf(err, errlen, "cudaMalloc tmp: %s", cudaGetErrorString(e));
    return false;
  }
  e = cudaMalloc(&d_mean_, 3 * 4);
  if (e != cudaSuccess) {
    snprintf(err, errlen, "cudaMalloc mean: %s", cudaGetErrorString(e));
    return false;
  }
  e = cudaMalloc(&d_std_, 3 * 4);
  if (e != cudaSuccess) {
    snprintf(err, errlen, "cudaMalloc std: %s", cudaGetErrorString(e));
    return false;
  }
  cudaMemcpy(d_mean_, p.mean, 3 * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_std_, p.std, 3 * 4, cudaMemcpyHostToDevice);
  // LUT 打包: [out_n*2 (xmin,cnt)] + [out_n*ksize]
  auto upload = [&](const AxisLUT& l, int32_t** d) {
    size_t n = (size_t)l.out_n * 2 + (size_t)l.out_n * l.ksize;
    std::vector<int32_t> host(n);
    memcpy(host.data(), l.xmin, sizeof(int32_t) * l.out_n);
    memcpy(host.data() + l.out_n, l.cnt, sizeof(int32_t) * l.out_n);
    memcpy(host.data() + (size_t)l.out_n * 2, l.kint,
           sizeof(int32_t) * l.out_n * l.ksize);
    cudaMalloc(d, n * 4);
    cudaMemcpy(*d, host.data(), n * 4, cudaMemcpyHostToDevice);
  };
  upload(lut_x_, &d_lutx_);
  upload(lut_y_, &d_luty_);
  return true;
}

int Preproc::run(const uint8_t* slot_payload_dev, size_t cam_bytes,
                 float* out_img_dev, void* stream) {
  cudaStream_t s = (cudaStream_t)stream;
  const int w = p_.src_w, h = p_.src_h;
  // 1) NV12 → RGB u8
  dim3 b1(32, 8);
  dim3 g1((w + b1.x - 1) / b1.x, (h + b1.y - 1) / b1.y, 6);
  nv12_to_rgb_kernel<<<g1, b1, 0, s>>>(slot_payload_dev, cam_bytes, w, h,
                                       d_rgb_);
  // 2) 水平通带 (全高, resized_w 列)
  dim3 b2(64, 8);
  dim3 g2((p_.resized_w + b2.x - 1) / b2.x, (h + b2.y - 1) / b2.y, 6);
  horiz_pass_kernel<<<g2, b2, 0, s>>>(d_rgb_, w, h, p_.resized_w, d_lutx_,
                                      lut_x_.ksize, d_tmp_);
  // 3) 垂直通带 + crop + 归一化
  dim3 b3(64, 8);
  dim3 g3((p_.out_w + b3.x - 1) / b3.x, (p_.out_h + b3.y - 1) / b3.y, 6);
  vert_norm_kernel<<<g3, b3, 0, s>>>(d_tmp_, h, p_.resized_h, p_.out_w,
                                     p_.out_h, p_.crop_y, d_luty_,
                                     lut_y_.ksize, d_mean_, d_std_,
                                     out_img_dev);
  return (int)cudaGetLastError();
}

size_t Preproc::out_bytes() const {
  return (size_t)6 * 3 * p_.out_h * p_.out_w * 4;
}

}  // namespace sp
