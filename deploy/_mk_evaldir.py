# -*- coding: utf-8 -*-
"""把 m3 扁平 fetch 组装成 eval 目录. 用法: python _mk_evaldir.py <SRC_name> <DST_name> [META_dir]
SRC/DST 相对 work_dirs/preproc_ref 与 evaldata; META 缺省用 mini_sp2_81 的."""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2", "evaldata")
SRC = os.path.join(ROOT, "work_dirs", "preproc_ref", sys.argv[1])
DST = os.path.join(EV, sys.argv[2])
META = os.path.join(EV, sys.argv[3] if len(sys.argv) > 3 else "mini_sp2_81",
                    "mini_meta.npz")
KEY = ["det_cls", "det_bbox", "det_quality", "det_instance_id",
       "map_cls", "map_pts"]

n = 0
for k in range(81):
    d = os.path.join(DST, "outv8_%02d" % k)
    os.makedirs(d, exist_ok=True)
    for nm in KEY:
        s = os.path.join(SRC, "out_%02d_%s.bin" % (k, nm))
        if os.path.exists(s):
            shutil.copy(s, os.path.join(d, nm + ".bin"))
            n += 1
shutil.copy(META, os.path.join(DST, "mini_meta.npz"))
print("assembled %d bins -> %s" % (n, DST))
