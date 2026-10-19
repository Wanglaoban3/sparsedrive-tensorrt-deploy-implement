// sp_postproc.h - GPU 后处理 (Step2): det topk/centerness 融合/几何变换 +
// map 类别优先过滤, 全部核可进 gH graph 捕获, 解码结果小 D2H.
//
// 语义逐位对齐 postproc.h CPU 口径 (唯一豁免 = expf/atan2f ULP 级差,
// 验收标准沿用 "集全等 + 分数 <=1e-6 / 几何 <=1e-5"):
//   det: 候选 n = anchor*kDetCls+cls_id (与 CPU cand 构造序一致);
//     第一轮键 = (~bits(s))<<32 | n, u64 升序 == 融合前分数稳定降序
//     (s 同比特时低 32b=n 升序, 即 numpy stable argsort 语义);
//     CUB radix sort (全局内存, temp 预分配, 地址固定可捕获) 取前 topk;
//     第二轮键 = (~bits(fused))<<32 | 第一轮 rank → 融合分稳定重排
//     (tie 保持第一轮序); 阈值过滤 + exp/atan2 几何变换单线程定序压缩.
//   map: 类别优先 (c 外层 a 内层) sigmoid > floor 保留, smem 前缀和压缩
//     保输出序 == CPU 遍历序.
#ifndef SP_POSTPROC_GPU_H_
#define SP_POSTPROC_GPU_H_

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "postproc.h"

namespace sp {
namespace pp {

constexpr int kPar = 2;

// 容量: det 输出 ≤ topk, 节点 clamp det_topk ≤ res::kDetCap=300;
// map 全保留上界 = 100*3 (与 res::kMapCap 同值, 节点侧 static_assert).
constexpr int kTopkCap = 300;
constexpr int kMapCap = post::kMapAnchors * post::kMapCls;

struct PPWs {
  // ---- device (per parity, 地址固定 = graph 捕获前提) ----
  uint64_t* d_keys1[kPar];  // [9000] 第一轮键
  uint64_t* d_keys2[kPar];  // [9000] 排序输出 (前 topk 个 = top 候选)
  float* d_fused[kPar];     // [9000] 融合分 (乘 centerness 后)
  uint64_t* d_keys3[kPar];  // [300] 第二轮键 (tie=第一轮 rank)
  uint64_t* d_keys4[kPar];  // [300] 重排输出
  post::DetOut* d_det[kPar];  // [300] 最终解码
  int32_t* d_nout[kPar];     // [300] 输出槽 -> 候选 n (parity 对拍配对用)
  post::MapOut* d_map[kPar];  // [300]
  int32_t* d_cnt[kPar];      // [2] = {ndet, nmap}
  // ---- host pinned (per parity, post_stream 小 D2H 目标) ----
  post::DetOut* h_det[kPar];
  post::MapOut* h_map[kPar];
  int32_t* h_cnt[kPar];  // [2]
  int32_t* h_nout[kPar];  // [300] 仅 parity 模式 D2H
  // ---- 共享 (两 parity 串行同流, CUB temp 单份安全) ----
  void* d_tmp;
  size_t tmp_bytes;
};

// cudaMalloc/pinned + CUB temp 尺寸查询. 失败返回 false (调用方回退 CPU).
bool pp_init(PPWs* w, char* err, size_t errlen);
void pp_free(PPWs* w);

// 每帧解码核 (graph 捕获内 / eager 等价): 全异步进 s, 固定地址, 无同步.
void pp_decode(const PPWs& w, int p, const float* cls_dev,
               const float* q_dev, const float* bbox_dev,
               const int32_t* id_dev, const float* map_cls_dev,
               const float* map_pts_dev, int topk, float det_thr,
               float map_thr, cudaStream_t s);

}  // namespace pp
}  // namespace sp

#endif  // SP_POSTPROC_GPU_H_
