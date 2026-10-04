# -*- coding: utf-8 -*-
"""preproc_ref/<tag> 扁平 dump -> map + mp 两个评估目录 (Phase A mproda
组装约定的泛化, 保证各阶段门禁 A/B 可比):
  map: mini_<tag>_map/out_XX/{det_cls,det_bbox,det_quality,det_instance_id,
       map_cls,map_pts}.bin + mini_meta.npz (eval_t6_mini_map.py 口径)
  mp:  mini_<tag>_mp/out_XX/{det_id(=det_instance_id), motion_*, plan_*,
       history_*(=outm_XX next_history_*, 同 k 帧 — 与 mproda 基线逐位
       同约定)} + mini_meta.npz (eval_mp_mini.py 口径)
用法: python _mprod_mkeval.py <tag>"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = sys.argv[1] if len(sys.argv) > 1 else "mprodb"
SRC = os.path.join(ROOT, "work_dirs", "preproc_ref", TAG)
ED = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2", "evaldata")
REF = os.path.join(ED, "mini_eng_v8")

MAP_KEYS = ["det_cls", "det_bbox", "det_quality", "det_instance_id",
            "map_cls", "map_pts"]
MP_DIRECT = ["motion_cls", "motion_reg", "plan_cls", "plan_reg",
             "plan_status"]
MP_HIST = {"history_anchor": "next_history_anchor",
           "history_period": "next_history_period",
           "history_ego_anchor": "next_history_ego_anchor",
           "history_ego_period": "next_history_ego_period"}

missing = []


def copy2(dst, src):
    if not os.path.isfile(src):
        missing.append(src)
        return False
    shutil.copyfile(src, dst)
    return True


# ---- map 目录 ----
dmap = os.path.join(ED, "mini_%s_map" % TAG)
os.makedirs(dmap, exist_ok=True)
n = 0
for k in range(81):
    od = os.path.join(dmap, "out_%02d" % k)
    os.makedirs(od, exist_ok=True)
    for nm in MAP_KEYS:
        n += copy2(os.path.join(od, nm + ".bin"),
                   os.path.join(SRC, "out_%02d_%s.bin" % (k, nm)))
shutil.copyfile(os.path.join(REF, "mini_meta.npz"),
                os.path.join(dmap, "mini_meta.npz"))
print("map: %d bins -> %s" % (n, dmap))

# ---- mp 目录 ----
dmp = os.path.join(ED, "mini_%s_mp" % TAG)
os.makedirs(dmp, exist_ok=True)
n = 0
for k in range(81):
    od = os.path.join(dmp, "out_%02d" % k)
    os.makedirs(od, exist_ok=True)
    for nm in ["det_cls", "det_bbox", "det_quality"]:
        n += copy2(os.path.join(od, nm + ".bin"),
                   os.path.join(SRC, "out_%02d_%s.bin" % (k, nm)))
    n += copy2(os.path.join(od, "det_id.bin"),
               os.path.join(SRC, "out_%02d_det_instance_id.bin" % k))
    for nm in MP_DIRECT:
        n += copy2(os.path.join(od, nm + ".bin"),
                   os.path.join(SRC, "outm_%02d_%s.bin" % (k, nm)))
    for dst, src in MP_HIST.items():
        n += copy2(os.path.join(od, dst + ".bin"),
                   os.path.join(SRC, "outm_%02d_%s.bin" % (k, src)))
shutil.copyfile(os.path.join(REF, "mini_meta.npz"),
                os.path.join(dmp, "mini_meta.npz"))
print("mp: %d bins -> %s" % (n, dmp))
if missing:
    print("MISSING %d:" % len(missing))
    for m in missing[:10]:
        print("  " + m)
    raise SystemExit(1)
print("MKEVAL_DONE")
