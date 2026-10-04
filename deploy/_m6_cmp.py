# -*- coding: utf-8 -*-
"""节点 mp 输出 (m6fix/outm_XX) vs 同源离线链 (mpref/outm_XX) 逐张量对拍.
f0/复位帧期望近位相等; f1+ 状态递归分叉属混沌地板, 看趋势不当门禁."""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "work_dirs", "preproc_ref", "m6fix")
B = os.path.join(ROOT, "work_dirs", "preproc_ref", "mpref")
KEYS = ["motion_cls", "motion_reg", "plan_cls", "plan_reg", "plan_status",
        "next_history_instance_feature", "next_history_anchor",
        "next_history_period", "next_prev_instance_id",
        "next_prev_confidence", "next_history_ego_feature",
        "next_history_ego_anchor", "next_history_ego_period",
        "next_prev_ego_status"]
FRAMES = [0, 1, 2, 39, 40, 41, 79, 80]
np.set_printoptions(precision=4, suppress=True, linewidth=160)

for k in FRAMES:
    print("== frame %02d" % k)
    for nm in KEYS:
        pa = os.path.join(A, "outm_%02d_%s.bin" % (k, nm))
        pb = os.path.join(B, "outm_%02d_%s.bin" % (k, nm))
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print("  %-32s MISSING" % nm)
            continue
        a = np.fromfile(pa, np.int32 if nm.endswith(("id", "period"))
                        else np.float32).astype(np.float64)
        b = np.fromfile(pb, np.int32 if nm.endswith(("id", "period"))
                        else np.float32).astype(np.float64)
        if nm.endswith(("id", "period")):
            same = (a == b).mean() * 100.0
            print("  %-32s int match=%.2f%%" % (nm, same))
            continue
        d = np.abs(a - b)
        rel = np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)
        print("  %-32s rel=%.3g maxabs=%.3g (|a|p99=%.3g)"
              % (nm, rel, d.max(), np.percentile(np.abs(b), 99)))
print("M6_CMP_DONE")
