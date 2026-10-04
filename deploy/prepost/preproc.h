// M2 前处理: NV12(x6, 只读设备指针) → RGB u8 → PIL 位级同构 BICUBIC 缩放
// (定点整数管线, 见 deploy/preproc_ref.py 的 Pillow 12.3 Resample.c 移植)
// → crop → 归一化 → 连续 [1,6,3,256,704] f32 (引擎 img binding 直绑).
// 布局/数值全部对齐链路二: BT.601 full-range; mean/std RGB 序; 无 /255.
#ifndef SP_PREPROC_H_
#define SP_PREPROC_H_

#include <stddef.h>
#include <stdint.h>

namespace sp {

struct PreprocParams {
  uint32_t src_w, src_h;      // 1600 x 900
  uint32_t resized_w, resized_h;  // PIL resize 输出 (704 x 396)
  uint32_t out_w, out_h;      // 704 x 256 (final_dim = crop 尺寸)
  uint32_t crop_x, crop_y;    // crop origin in resized coords (0,140)
  float mean[3], std[3];      // RGB order (123.675, 116.28, 103.53)...
};

// 每轴系数表 = Pillow precompute_coeffs 8bpc 定点版 (kint, scale 2^22)
struct AxisLUT {
  int32_t* xmin;   // [out_n]
  int32_t* cnt;    // [out_n]
  int32_t* kint;   // [out_n * ksize]
  int32_t ksize;
  int32_t out_n;
};

class Preproc {
 public:
  ~Preproc();
  // params 全部来自 manifest v2 header; 在 host 上按 Pillow 公式造表并上传
  bool init(const PreprocParams& p, char* err, size_t errlen);
  // slot_payload = 6 路 NV12 连续 (cam c at c*cam_bytes) 的**设备指针**
  // (registered shm 的 devptr); out = [1,6,3,256,704] f32 设备缓冲.
  // 三个 kernel 依次进 stream; 返回 0 成功.
  int run(const uint8_t* slot_payload_dev, size_t cam_bytes,
          float* out_img_dev, void* stream);

  size_t out_bytes() const;  // 1*6*3*256*704*4

 private:
  PreprocParams p_{};
  AxisLUT lut_x_{}, lut_y_{};
  uint8_t* d_rgb_ = nullptr;   // 6 * src_h * src_w * 3
  uint8_t* d_tmp_ = nullptr;   // 6 * src_h * resized_w * 3 (水平通带后, 全高)
  int32_t* d_lutx_ = nullptr;  // xmin|cnt|kint 打包上传
  int32_t* d_luty_ = nullptr;
  float* d_mean_ = nullptr;
  float* d_std_ = nullptr;
};

// host 侧造表 (公开给测试复用/校验): 与 deploy/preproc_ref.py pil_coeffs_int
// 逐位一致. 失败返回 false.
bool build_axis_lut(int in_size, int out_size, AxisLUT* lut);

}  // namespace sp

#endif  // SP_PREPROC_H_
