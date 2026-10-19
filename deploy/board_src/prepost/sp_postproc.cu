// sp_postproc.cu - GPU 后处理核 (语义见 sp_postproc.h).
// 排序用 CUB DeviceRadixSort (u64 键 = 分数比特编码 + 候选序, 全局内存):
// 9000 元素单 block bitonic 需 ~128KB dynamic smem, 超 sm_87 单 block 上限;
// CUB temp 预分配 + 固定地址, 流内纯核发射, 可被 graph 捕获.
#include "sp_postproc.h"

#include <cstdio>
#include <cstring>

#include <cub/device/device_radix_sort.cuh>

namespace sp {
namespace pp {
namespace {

// 升序键: s ∈ (0,1) 恒正 → bits(s) 单调 → ~bits 越小键越小 = 分数降序;
// 同分数比特 tie 由低 32b (候选平坦序 n) 升序决定 == stable 语义.
__device__ __inline__ uint64_t score_key(float s, uint32_t n) {
  return (uint64_t)(~__float_as_uint(s)) << 32 | (uint64_t)n;
}

__global__ void build_det_keys(const float* __restrict__ cls,
                               const float* __restrict__ q,
                               float* __restrict__ fused,
                               uint64_t* __restrict__ keys) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= post::kDetAnchors * post::kDetCls) return;
  const int a = n / post::kDetCls;
  const float s = post::sigmoidf(cls[n]);
  fused[n] = s * post::sigmoidf(q[(size_t)a * 2]);  // CNS = q[a][0] (平坦 2a)
  keys[n] = score_key(s, (uint32_t)n);
}

__global__ void build_rerank_keys(const uint64_t* __restrict__ top,
                                  const float* __restrict__ fused,
                                  uint64_t* __restrict__ keys3, int topk) {
  const int i = threadIdx.x;
  if (i >= topk) return;
  const int n = (int)(top[i] & 0xffffffffu);
  keys3[i] = score_key(fused[n], (uint32_t)i);  // tie=第一轮 rank 升序
}

// 单线程定序压缩 (topk ≤ 300 次 ~15 flop, 微秒级): 输出序 = 第二轮序, 与
// CPU stable_sort 语义一致, 无原子无竞态.
__global__ void det_finalize(const uint64_t* __restrict__ order,
                             const uint64_t* __restrict__ top,
                             const float* __restrict__ fused,
                             const float* __restrict__ bbox,
                             const int32_t* __restrict__ inst_id,
                             post::DetOut* __restrict__ out,
                             int32_t* __restrict__ nout, int* __restrict__ cnt,
                             int topk, float thr) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int m = 0;
  for (int j = 0; j < topk; ++j) {
    const int rank1 = (int)(order[j] & 0xffffffffu);
    const int n = (int)(top[rank1] & 0xffffffffu);
    if (fused[n] < thr) continue;  // ref: mask = cls_scores >= threshold
    const int a = n / post::kDetCls;
    const float* b = bbox + (size_t)a * 11;
    post::DetOut& o = out[m];
    o.score = fused[n];
    o.label = n - a * post::kDetCls;
    o.x = b[0];
    o.y = b[1];
    o.z = b[2];
    o.w = expf(b[3]);
    o.l = expf(b[4]);
    o.h = expf(b[5]);
    o.yaw = atan2f(b[6], b[7]);
    o.vx = b[8];
    o.vy = b[9];  // b[10]=vz 不发布
    o.id = inst_id[a];
    nout[m] = n;
    ++m;
  }
  cnt[0] = m;
}

// 类别优先 (c 外层 a 内层) sigmoid 过滤 + 前缀和压缩保输出序 == CPU 遍历序;
// 单 block 两相, smem ~2.4KB.
__global__ void map_decode(const float* __restrict__ cls,
                           const float* __restrict__ pts,
                           post::MapOut* __restrict__ out, int* __restrict__ cnt,
                           float thr) {
  constexpr int M = post::kMapAnchors * post::kMapCls;  // 300
  __shared__ float ss[M];
  __shared__ int32_t off[M + 1];
  const int t = threadIdx.x, nt = blockDim.x;
  for (int i = t; i < M; i += nt) {
    const int c = i / post::kMapAnchors, a = i - c * post::kMapAnchors;
    ss[i] = post::sigmoidf(cls[a * post::kMapCls + c]);
    off[i] = (ss[i] > thr) ? 1 : 0;  // eval: if s <= floor: continue
  }
  __syncthreads();
  if (t == 0) {
    int m = 0;
    for (int i = 0; i < M; ++i) {
      const int p = off[i];
      off[i] = m;
      m += p;
    }
    off[M] = m;
    cnt[1] = m;
  }
  __syncthreads();
  for (int i = t; i < M; i += nt) {
    if (off[i + 1] == off[i]) continue;
    const int c = i / post::kMapAnchors, a = i - c * post::kMapAnchors;
    post::MapOut& o = out[off[i]];
    o.score = ss[i];
    o.label = c;
    const float* src = pts + (size_t)a * (post::kMapPts * 2);
    for (int k = 0; k < post::kMapPts; ++k) {
      o.pts[k][0] = src[k * 2];
      o.pts[k][1] = src[k * 2 + 1];
    }
  }
}

}  // namespace

bool pp_init(PPWs* w, char* err, size_t errlen) {
  memset(w, 0, sizeof(*w));
  const int N = post::kDetAnchors * post::kDetCls;
  size_t tmp = 0;
  if (cub::DeviceRadixSort::SortKeys(nullptr, tmp, (uint64_t*)nullptr,
                                     (uint64_t*)nullptr, N) != cudaSuccess) {
    snprintf(err, errlen, "cub temp query failed");
    return false;
  }
  w->tmp_bytes = tmp;
  if (cudaMalloc(&w->d_tmp, tmp) != cudaSuccess) {
    snprintf(err, errlen, "cudaMalloc cub tmp %zu", tmp);
    return false;
  }
  const size_t det_b = (size_t)kTopkCap * sizeof(post::DetOut);
  const size_t map_b = (size_t)kMapCap * sizeof(post::MapOut);
  for (int p = 0; p < kPar; ++p) {
    auto dalloc = [&](void** v, size_t b) {
      if (cudaMalloc(v, b) != cudaSuccess) {
        snprintf(err, errlen, "cudaMalloc %zu (p=%d)", b, p);
        return false;
      }
      return true;
    };
    if (!dalloc((void**)&w->d_keys1[p], N * 8) ||
        !dalloc((void**)&w->d_keys2[p], N * 8) ||
        !dalloc((void**)&w->d_fused[p], N * 4) ||
        !dalloc((void**)&w->d_keys3[p], (size_t)kTopkCap * 8) ||
        !dalloc((void**)&w->d_keys4[p], (size_t)kTopkCap * 8) ||
        !dalloc((void**)&w->d_det[p], det_b) ||
        !dalloc((void**)&w->d_nout[p], (size_t)kTopkCap * 4) ||
        !dalloc((void**)&w->d_map[p], map_b) ||
        !dalloc((void**)&w->d_cnt[p], 8))
      return false;
    auto halloc = [&](void** v, size_t b) {
      if (cudaMallocHost(v, b) != cudaSuccess) {
        snprintf(err, errlen, "cudaMallocHost %zu (p=%d)", b, p);
        return false;
      }
      return true;
    };
    if (!halloc((void**)&w->h_det[p], det_b) ||
        !halloc((void**)&w->h_map[p], map_b) ||
        !halloc((void**)&w->h_cnt[p], 8) ||
        !halloc((void**)&w->h_nout[p], (size_t)kTopkCap * 4))
      return false;
  }
  return true;
}

void pp_free(PPWs* w) {
  for (int p = 0; p < kPar; ++p) {
    cudaFree(w->d_keys1[p]);
    cudaFree(w->d_keys2[p]);
    cudaFree(w->d_fused[p]);
    cudaFree(w->d_keys3[p]);
    cudaFree(w->d_keys4[p]);
    cudaFree(w->d_det[p]);
    cudaFree(w->d_nout[p]);
    cudaFree(w->d_map[p]);
    cudaFree(w->d_cnt[p]);
    cudaFreeHost(w->h_det[p]);
    cudaFreeHost(w->h_map[p]);
    cudaFreeHost(w->h_cnt[p]);
    cudaFreeHost(w->h_nout[p]);
  }
  cudaFree(w->d_tmp);
}

void pp_decode(const PPWs& w, int p, const float* cls_dev,
               const float* q_dev, const float* bbox_dev,
               const int32_t* id_dev, const float* map_cls_dev,
               const float* map_pts_dev, int topk, float det_thr,
               float map_thr, cudaStream_t s) {
  const int N = post::kDetAnchors * post::kDetCls;
  build_det_keys<<<(N + 255) / 256, 256, 0, s>>>(cls_dev, q_dev, w.d_fused[p],
                                                 w.d_keys1[p]);
  // end_bit 形参是 int (sizeof()*8 的 size_t 模板匹配失败);
  // temp_storage_bytes 是 size_t& 出入参, const PPWs& 成员绑不上, 取副本
  size_t tmp = w.tmp_bytes;
  cub::DeviceRadixSort::SortKeys(w.d_tmp, tmp, w.d_keys1[p],
                                 w.d_keys2[p], N, 0, 64, s);
  build_rerank_keys<<<1, kTopkCap, 0, s>>>(w.d_keys2[p], w.d_fused[p],
                                           w.d_keys3[p], topk);
  cub::DeviceRadixSort::SortKeys(w.d_tmp, tmp, w.d_keys3[p],
                                 w.d_keys4[p], topk, 0, 64, s);
  det_finalize<<<1, 1, 0, s>>>(w.d_keys4[p], w.d_keys2[p], w.d_fused[p],
                               bbox_dev, id_dev, w.d_det[p], w.d_nout[p],
                               w.d_cnt[p], topk, det_thr);
  map_decode<<<1, 256, 0, s>>>(map_cls_dev, map_pts_dev, w.d_map[p],
                               w.d_cnt[p], map_thr);
}

}  // namespace pp
}  // namespace sp
