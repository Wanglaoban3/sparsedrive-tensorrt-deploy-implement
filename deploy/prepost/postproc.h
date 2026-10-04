// postproc.h - Q1/Q2/Q3: det/map decode from engine outputs to results.
//
// 口径逐位对齐离线评测脚本 (不许自行发挥):
//   det: deploy/eval_t6_mini_v8.py decode_sample()
//     - cls = sigmoid(det_cls[900,10]) 摊平 9000;
//     - 稳定降序 topk(300) (tie 保持平坦下标升序, 同 numpy stable argsort);
//     - centerness = sigmoid(det_quality[anchor,0]) 乘进 score 后整体稳定重排;
//     - box[anchor]: [x,y,z] | exp(w,l,h) | atan2(sin,cos) | [vx,vy,vz];
//     - score >= thr 过滤 (ref decoder: cls_scores >= threshold, 默认 0 全保留).
//   map: deploy/eval_t6_mini_map.py decode_preds()
//     - cls = sigmoid(map_cls[100,3]);
//     - 类别优先 (c 外层, anchor 内层) 遍历, s > floor 保留 (默认 0 全保留);
//     - pts[100,20,2] 已是 lidar 系绝对坐标, 直接引用.
#ifndef SP_POSTPROC_H_
#define SP_POSTPROC_H_

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace sp {
namespace post {

constexpr int kDetAnchors = 900;
constexpr int kDetCls = 10;
constexpr int kMapAnchors = 100;
constexpr int kMapCls = 3;
constexpr int kMapPts = 20;

struct DetOut {   // det[]{score,label,x,y,z,w,l,h,yaw,vx,vy,id} (设计稿 §4)
  float score;
  int32_t label;
  float x, y, z, w, l, h, yaw, vx, vy;
  int32_t id;
};

struct MapOut {   // map[]{score,label,pts[20][2]}
  float score;
  int32_t label;
  float pts[kMapPts][2];
};

inline float sigmoidf(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// Q1+Q2: det decode. cls[900*10] q[900*2] bbox[900*11] inst_id[900] 均为 f32/i32
// host 数据 (post D2H 之后). 返回写入 out 的条数 (<= topk).
inline int det_decode(const float* cls, const float* q, const float* bbox,
                      const int32_t* inst_id, int topk, float thr,
                      DetOut* out) {
  struct Cand {
    int anchor, cls_id;
    float score;  // 融合前 (与 ref 的 cls_scores_origin 一致语义)
    float fused;  // × centerness 后
  };
  static Cand cand[kDetAnchors * kDetCls];
  int n = 0;
  for (int a = 0; a < kDetAnchors; ++a) {
    for (int c = 0; c < kDetCls; ++c) {
      cand[n].anchor = a;
      cand[n].cls_id = c;
      float s = sigmoidf(cls[a * kDetCls + c]);
      cand[n].score = s;
      // CNS = quality[a][0]; quality 是 [900,2] 行主序 → 平坦下标 2a
      // (q[a] 只会读到第 a 个 float 而非第 a 行 —— 已踩过, 对比即现形:
      //  候选集一致但融合分全错)
      cand[n].fused = s * sigmoidf(q[(size_t)a * 2]);
      ++n;
    }
  }
  // 稳定降序 topk: numpy argsort(-flat, kind='stable') 等价 — tie 保持原序
  std::stable_sort(cand, cand + n, [](const Cand& a, const Cand& b) {
    return a.score > b.score;
  });
  if (topk > n) topk = n;
  // 融合分稳定重排 (tie 保持 topk 序)
  std::stable_sort(cand, cand + topk, [](const Cand& a, const Cand& b) {
    return a.fused > b.fused;
  });
  int m = 0;
  for (int i = 0; i < topk; ++i) {
    const Cand& cd = cand[i];
    if (cd.fused < thr) continue;  // ref: mask = cls_scores >= threshold
    const float* b = bbox + cd.anchor * 11;
    DetOut& o = out[m++];
    o.score = cd.fused;
    o.label = cd.cls_id;
    o.x = b[0]; o.y = b[1]; o.z = b[2];
    o.w = expf(b[3]); o.l = expf(b[4]); o.h = expf(b[5]);
    o.yaw = atan2f(b[6], b[7]);
    o.vx = b[8]; o.vy = b[9];  // b[10]=vz 留在张量侧, 结果消息不带
    o.id = inst_id[cd.anchor];
  }
  return m;
}

// Q3: map decode. cls[100*3] pts[100*20*2]. 类别优先序 (与 eval 一致).
inline int map_decode(const float* cls, const float* pts, float floor_thr,
                      MapOut* out) {
  int m = 0;
  for (int c = 0; c < kMapCls; ++c) {
    for (int a = 0; a < kMapAnchors; ++a) {
      float s = sigmoidf(cls[a * kMapCls + c]);
      if (s <= floor_thr) continue;  // eval: if s <= score_floor: continue
      MapOut& o = out[m++];
      o.score = s;
      o.label = c;
      const float* p = pts + a * (kMapPts * 2);
      memcpy(o.pts, p, sizeof(o.pts));
    }
  }
  return m;
}

}  // namespace post
}  // namespace sp

#endif  // SP_POSTPROC_H_
