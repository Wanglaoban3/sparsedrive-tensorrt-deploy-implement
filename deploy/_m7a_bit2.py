# -*- coding: utf-8 -*-
"""判别: 当前二进制串行重跑 (m7srl) vs M6a 基线 (m6fix) bit 比对;
以及 m7srl vs m7fix (串行 vs 双流) bit 比对."""
import codecs
import os
import sys

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "line_buffering")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PR = os.path.join(ROOT, "work_dirs", "preproc_ref")
KEYS = ["det_cls", "det_bbox", "det_quality", "det_instance_id",
        "map_cls", "map_pts"]
MP_KEYS = ["motion_cls", "motion_reg", "plan_cls", "plan_reg", "plan_status",
           "next_history_instance_feature", "next_history_anchor",
           "next_history_period", "next_prev_instance_id",
           "next_prev_confidence", "next_history_ego_feature",
           "next_history_ego_anchor", "next_history_ego_period",
           "next_prev_ego_status"]

for la, lb, nm in (("m7srl", "m6fix", "串行重跑 vs M6a基线"),
                   ("m7srl", "m7fix", "串行 vs 双流")):
    A = os.path.join(PR, la)
    B = os.path.join(PR, lb)
    n_ok = n_bad = 0
    diffs = []
    for k in range(81):
        for pre, keys in (("out", KEYS), ("outm", MP_KEYS)):
            for key in keys:
                fa = os.path.join(A, "%s_%02d_%s.bin" % (pre, k, key))
                fb = os.path.join(B, "%s_%02d_%s.bin" % (pre, k, key))
                if not (os.path.exists(fa) and os.path.exists(fb)):
                    continue
                if open(fa, "rb").read() == open(fb, "rb").read():
                    n_ok += 1
                else:
                    n_bad += 1
                    if len(diffs) < 8:
                        diffs.append("%s_%02d_%s" % (pre, k, key))
    print("%s: ok=%d diff=%d  例: %s" % (nm, n_ok, n_bad, ",".join(diffs)))
