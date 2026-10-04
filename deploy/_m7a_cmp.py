# -*- coding: utf-8 -*-
"""M7a 精度门禁: --dual 81 帧 dump 与 M6 串行基线逐字节比对 (m7fix vs m6fix),
双流只改事件定序不改数学, 应当 bit-identical; 同时解析各 run 的 frame_log."""
import codecs
import os
import sys

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "line_buffering")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PR = os.path.join(ROOT, "work_dirs", "preproc_ref")
A = os.path.join(PR, "m7fix")
B = os.path.join(PR, "m6fix")
KEYS = ["det_cls", "det_bbox", "det_quality", "det_instance_id",
        "map_cls", "map_pts"]
MP_KEYS = ["motion_cls", "motion_reg", "plan_cls", "plan_reg", "plan_status",
           "next_history_instance_feature", "next_history_anchor",
           "next_history_period", "next_prev_instance_id",
           "next_prev_confidence", "next_history_ego_feature",
           "next_history_ego_anchor", "next_history_ego_period",
           "next_prev_ego_status"]

n_ok = n_bad = n_miss = 0
bad = []
for k in range(81):
    for pre, keys in (("out", KEYS), ("outm", MP_KEYS)):
        for nm in keys:
            fa = os.path.join(A, "%s_%02d_%s.bin" % (pre, k, nm))
            fb = os.path.join(B, "%s_%02d_%s.bin" % (pre, k, nm))
            if not (os.path.exists(fa) and os.path.exists(fb)):
                n_miss += 1
                bad.append("MISS %s_%02d_%s" % (pre, k, nm))
                continue
            da = open(fa, "rb").read()
            db = open(fb, "rb").read()
            if da == db:
                n_ok += 1
            else:
                n_bad += 1
                bad.append("DIFF %s_%02d_%s (%d vs %d B)"
                           % (pre, k, nm, len(da), len(db)))
print("bit-compare: ok=%d diff=%d miss=%d" % (n_ok, n_bad, n_miss))
for ln in bad[:20]:
    print(" ", ln)

for tag in ("m7fix", "m7thr", "m7pipe"):
    fp = os.path.join(PR, tag, "frame_log.tsv")
    if not os.path.exists(fp):
        continue
    rows = [ln.split("\t") for ln in open(fp).read().splitlines()[1:]
            if ln.strip()]
    ready = [float(r[7]) for r in rows if len(r) > 7 and r[7] not in ("0", "")]
    if len(ready) > 2:
        fps = (len(ready) - 1) / ((ready[-1] - ready[0]) / 1e9)
        print("%s: fps=%.2f (%d frames, span %.1f ms/frame)"
              % (tag, fps, len(ready), (ready[-1] - ready[0]) / 1e6
                 / max(1, len(ready) - 1)))
