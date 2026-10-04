# -*- coding: utf-8 -*-
"""把节点 mp 跑拉回的 m6fix 组装成 eval_mp_mini 目录:
out_XX/{det_cls,det_bbox,det_quality,det_id,motion_cls,motion_reg,
plan_cls,plan_reg,plan_status,history_anchor,history_period,
history_ego_anchor,history_ego_period} + mini_meta.npz."""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = sys.argv[1] if len(sys.argv) > 1 else "m6fix"
SRC = os.path.join(ROOT, "work_dirs", "preproc_ref", TAG)
DST = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2",
                   "evaldata", "mini_%s_mp" % TAG)
META = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2",
                    "evaldata", "mini_sp2_81", "mini_meta.npz")
DIRECT = ["det_cls", "det_bbox", "det_quality", "motion_cls", "motion_reg",
          "plan_cls", "plan_reg", "plan_status"]
REN = {"det_instance_id": "det_id",
       "next_history_anchor": "history_anchor",
       "next_history_period": "history_period",
       "next_history_ego_anchor": "history_ego_anchor",
       "next_history_ego_period": "history_ego_period"}

n = 0
miss = []
for k in range(81):
    d = os.path.join(DST, "out_%02d" % k)
    os.makedirs(d, exist_ok=True)
    for nm in DIRECT:
        s = os.path.join(SRC, "out_%02d_%s.bin" % (k, nm)) \
            if not nm.startswith("motion") and not nm.startswith("plan") \
            else os.path.join(SRC, "outm_%02d_%s.bin" % (k, nm))
        if os.path.exists(s):
            shutil.copy(s, os.path.join(d, nm + ".bin"))
            n += 1
        else:
            miss.append(os.path.basename(s))
    for src_nm, dst_nm in REN.items():
        s = os.path.join(SRC, "out_%02d_%s.bin" % (k, src_nm)) \
            if src_nm == "det_instance_id" \
            else os.path.join(SRC, "outm_%02d_%s.bin" % (k, src_nm))
        if os.path.exists(s):
            shutil.copy(s, os.path.join(d, dst_nm + ".bin"))
            n += 1
        else:
            miss.append(os.path.basename(s))
shutil.copy(META, os.path.join(DST, "mini_meta.npz"))
print("assembled %d bins -> %s" % (n, DST))
if miss:
    print("MISSING %d e.g. %s" % (len(miss), miss[:5]))
