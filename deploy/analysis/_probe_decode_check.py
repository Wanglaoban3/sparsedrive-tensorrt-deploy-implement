# -*- coding: utf-8 -*-
"""Q1-Q3 验证: 节点 C++ decode (result.jsonl) vs numpy 参考口径 (eval 脚本同源)
逐帧逐项对比 — det: 数量/label/id 全等, score/box max|d|; map: 数量/label 全等,
score max|d|, pts max|d|. 参考实现直接抄 eval_t6_mini_v8.decode_sample 与
eval_t6_mini_map.decode_preds (口径同源是验证的全部意义)."""
import json
import os
import sys

import numpy as np

M3 = os.path.join("work_dirs", "preproc_ref", "m3")
NUM_OUT = 300


def ref_det(k):
    ld = lambda nm, sh: np.fromfile(
        os.path.join(M3, "out_%02d_%s.bin" % (k, nm)), np.float32).reshape(sh)
    cls = 1.0 / (1.0 + np.exp(-ld("det_cls", (900, 10))))
    q = ld("det_quality", (900, 2))
    box = ld("det_bbox", (900, 11))
    ids = np.fromfile(os.path.join(M3, "out_%02d_det_instance_id.bin" % k),
                      np.int32)
    flat = cls.reshape(-1)
    top = np.argsort(-flat, kind="stable")[:NUM_OUT]
    scores = flat[top]
    cls_ids = top % 10
    anchor_idx = top // 10
    cent = 1.0 / (1.0 + np.exp(-q[anchor_idx, 0]))
    scores = scores * cent
    order = np.argsort(-scores, kind="stable")
    scores = scores[order]
    cls_ids = cls_ids[order]
    sel = box[anchor_idx][order]
    xy, whl = sel[:, :3], np.exp(sel[:, 3:6])
    yaw = np.arctan2(sel[:, 6], sel[:, 7])
    vel = sel[:, 8:11]
    boxes = np.concatenate([xy, whl, yaw[:, None], vel], axis=1)
    return {"score": scores, "label": cls_ids, "box": boxes,
            "id": ids[anchor_idx][order]}


def ref_map(k):
    cls = 1.0 / (1.0 + np.exp(-np.fromfile(
        os.path.join(M3, "out_%02d_map_cls.bin" % k), np.float32)
        .reshape(100, 3)))
    pts = np.fromfile(os.path.join(M3, "out_%02d_map_pts.bin" % k),
                      np.float32).reshape(100, 20, 2)
    vectors, scores, labels = [], [], []
    for c in range(3):
        for a in range(100):
            s = float(cls[a, c])
            if s <= 0.0:
                continue
            vectors.append(pts[a])
            scores.append(s)
            labels.append(c)
    return (np.array(scores, np.float32), np.array(labels, np.int64),
            np.array(vectors, np.float32))


def main():
    lines = [json.loads(l) for l in
             open(os.path.join(M3, "result.jsonl"), encoding="utf-8")
             if l.strip()]
    print("jsonl frames:", len(lines))
    n = len(lines)
    ds_max = db_max = dm_max = 0.0
    for ln in lines:
        k = ln["frame_id"]
        d = ref_det(k)
        jd = np.array(ln["det"], np.float32).reshape(-1, 12)
        assert len(jd) == len(d["score"]), \
            "f%d det count %d vs %d" % (k, len(jd), len(d["score"]))
        ds_max = max(ds_max, float(np.abs(jd[:, 0] - d["score"]).max()))
        # 节点消息口径 9 个框分量 (x,y,z,w,l,h,yaw,vx,vy), 参考第 10 列是 vz
        db_max = max(db_max, float(np.abs(jd[:, 2:11] - d["box"][:, :9]).max()))
        assert (jd[:, 1].astype(np.int64) == d["label"]).all(), \
            "f%d det label mismatch" % k
        assert (jd[:, 11].astype(np.int64) == d["id"]).all(), \
            "f%d det id mismatch" % k
        ms, ml, mv = ref_map(k)
        rows = [[r[0], r[1]] + [v for xy in r[2] for v in xy]
                for r in ln["map"]]
        jm = np.array(rows, np.float32).reshape(-1, 2 + 40)
        assert len(jm) == len(ms), \
            "f%d map count %d vs %d" % (k, len(jm), len(ms))
        dm_max = max(dm_max, float(np.abs(jm[:, 0] - ms).max()))
        assert (jm[:, 1].astype(np.int64) == ml).all(), \
            "f%d map label mismatch" % k
        dm_max = max(dm_max, float(np.abs(jm[:, 2:] - mv.reshape(
            len(ms), 40)).max()))
    print("frames=%d  det: count/label/id OK  score_max|d|=%.3g "
          "box_max|d|=%.3g" % (n, ds_max, db_max))
    print("          map: count/label OK  score+pts max|d|=%.3g" % dm_max)
    ok = ds_max < 2e-6 and db_max < 2e-6 and dm_max < 2e-6
    print("DECODE_CHECK_%s (阈值 2e-6, f32 舍入量级)" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
