# -*- coding: utf-8 -*-
"""compare_m2.py — M2 门禁记录: 板端 CUDA 前处理 vs 本地 numpy/PIL 参考.
参考: work_dirs/preproc_ref/ref_XX.npy ([1,6,3,256,704] f32)
板端: work_dirs/preproc_ref/board_XX.bin
逐像素: max_abs / mean_abs / 位级一致率 (期望位级一致或 ±1e-6)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "work_dirs", "preproc_ref")

ok = True
for k in (0, 1, 40):
    a = np.load(os.path.join(REF, "ref_%02d.npy" % k))
    b = np.fromfile(os.path.join(REF, "board_%02d.bin" % k),
                    dtype=np.float32).reshape(a.shape)
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    bit = (a == b).mean() * 100.0
    print("frame %2d: max=%.3e mean=%.3e bitexact=%.3f%%" %
          (k, d.max(), d.mean(), bit))
    if d.max() > 1e-3:
        ok = False
print("M2_COMPARE", "PASS" if ok else "CHECK")
sys.exit(0 if ok else 1)
