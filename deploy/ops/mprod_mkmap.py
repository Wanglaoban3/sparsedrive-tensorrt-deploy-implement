# -*- coding: utf-8 -*-
"""mproda 扁平 dump -> map 评估目录 (eval_t6_mini_map.py 口径).
out_XX/{det_cls,det_bbox,det_quality,det_instance_id,map_cls,map_pts}.bin
+ mini_meta.npz (从 mini_eng_v8 拷, GT/元数据与运行无关)."""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "work_dirs", "preproc_ref", "mproda")
REF = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2", "evaldata",
                   "mini_eng_v8")
DST = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2", "evaldata",
                   "mini_mproda_map")
KEYS = ["det_cls", "det_bbox", "det_quality", "det_instance_id",
        "map_cls", "map_pts"]

os.makedirs(DST, exist_ok=True)
n = 0
for k in range(81):
    od = os.path.join(DST, "out_%02d" % k)
    os.makedirs(od, exist_ok=True)
    for nm in KEYS:
        f = os.path.join(SRC, "out_%02d_%s.bin" % (k, nm))
        if not os.path.isfile(f):
            raise SystemExit("MISS %s" % f)
        shutil.copyfile(f, os.path.join(od, nm + ".bin"))
        n += 1
shutil.copyfile(os.path.join(REF, "mini_meta.npz"),
                os.path.join(DST, "mini_meta.npz"))
print("assembled %d bins + mini_meta.npz -> %s" % (n, DST))
