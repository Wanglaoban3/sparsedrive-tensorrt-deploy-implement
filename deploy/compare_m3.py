# -*- coding: utf-8 -*-
"""compare_m3.py — M3 记录: 板端闭环 (sp_modelnode dumps) vs 链路二 outv8.

参考: work_dirs/sparsedrive_small_stage2/evaldata/mini_eng_v8/outv8_XX/*.bin
板端: work_dirs/preproc_ref/m3/out_XX_<name>.bin (board_m1.py stage m3 拉回)

口径: 每帧每张量 rel_l2 / max_abs / mean_abs; i32 张量比对一致率.
预期(M3 定稿结论): f0 全输出 rel_l2 ~1e-3~1e-2; next_id_count 应 100% 一致
(引擎自驱动 +300/帧); f1 起元素级发散是**引擎状态输出行序非确定性**
(同输入两进程自身 ~1e-1 抖动) + NV12 源地板经 top-k 选择重排放大,
集合级无害 —— det 状态对比须按 id join (逐 id 内容 ~1e-2, 见
AGENTS.md 板端 TRT/闭环坑)。
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2",
                   "evaldata", "mini_eng_v8")
GOT = os.path.join(ROOT, "work_dirs", "preproc_ref", "m3")

KEY = ["det_cls", "det_bbox", "det_quality", "map_cls", "map_pts"]
DEEP = ["next_det_feat", "next_det_anchor", "next_det_conf",
        "next_map_feat", "next_map_anchor", "next_map_conf",
        "det_instance_id", "next_det_instance_id", "next_id_count"]
DEEP_FRAMES = {0, 1, 40, 80}


def load_ref(k, name):
    p = os.path.join(REF, "outv8_%02d" % k, name + ".bin")
    return np.fromfile(p, dtype=np.float32)


def stats(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    rel = np.sqrt((d ** 2).sum()) / (np.sqrt((b ** 2).sum()) + 1e-12)
    return rel, d.max(), d.mean()


def main():
    import re
    frames = sorted(int(m.group(1)) for f in os.listdir(GOT)
                    if (m := re.match(r"out_(\d+)_", f)))
    if not frames:
        sys.exit("no fetched tensors under " + GOT)
    n = frames[-1] + 1
    print("frames: %d (0..%d)" % (n, frames[-1]))
    agg = {}
    worst = {}
    for k in range(n):
        names = list(KEY) + (DEEP if k in DEEP_FRAMES else [])
        for nm in names:
            g = os.path.join(GOT, "out_%02d_%s.bin" % (k, nm))
            if not os.path.exists(g):
                continue
            a = np.fromfile(g, dtype=np.float32)
            b = load_ref(k, nm)
            if a.size != b.size:
                print("frame %d %s: SIZE MISMATCH %d vs %d"
                      % (k, nm, a.size, b.size))
                continue
            if nm.endswith(("instance_id", "id_count")):
                same = (a == b).mean() * 100.0
                agg.setdefault(nm, []).append(100.0 - same)
                continue
            rel, mx, mn = stats(a, b)
            agg.setdefault(nm, []).append(rel)
            worst.setdefault(nm, []).append((mx, k))
    print("\n%-24s %10s %10s %10s" % ("tensor", "rel_l2_avg", "rel_l2_max",
                                      "max_abs_max"))
    for nm, v in agg.items():
        v = np.array(v)
        w = max(worst.get(nm, [(0, 0)]))
        print("%-24s %10.2e %10.2e %10.3e (frame %d)" %
              (nm, v.mean(), v.max(), w[0], w[1]))
    print("\nM3_COMPARE_DONE")


if __name__ == "__main__":
    main()
