# -*- coding: utf-8 -*-
"""m7fix vs m6fix 差分幅度: 结构性错误(形状/常量链/大偏移) 还是 LSB tactic 噪声."""
import codecs
import os
import struct
import sys

import numpy as np

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "line_buffering")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PR = os.path.join(ROOT, "work_dirs", "preproc_ref")
CASES = [
    ("out_00_det_cls", 900, 10), ("out_00_det_bbox", 900, 11),
    ("out_00_map_pts", 100, 40), ("outm_00_plan_reg", 18, 12),
    ("outm_00_motion_reg", 900, 144),
    ("outm_00_next_history_instance_feature", 900 * 4, 256),
]
for nm, r, c in CASES:
    fa = os.path.join(PR, "m7fix", nm + ".bin")
    fb = os.path.join(PR, "m6fix", nm + ".bin")
    a = np.fromfile(fa, np.float32).reshape(r, c)
    b = np.fromfile(fb, np.float32).reshape(r, c)
    d = np.abs(a - b)
    rel = d / (np.abs(b) + 1e-9)
    print("%-40s maxabs=%.3e  rel_p99=%.3e  bitwise_ne=%d/%d" % (
        nm, d.max(), np.percentile(rel, 99), int((a.view(np.uint32)
                                                  != b.view(np.uint32)).sum()),
        a.size))
    print("   argmax abs: a[%.0f,%d]=%.6f b=%.6f" % (
        np.unravel_index(d.argmax(), d.shape)[0],
        np.unravel_index(d.argmax(), d.shape)[1],
        a.flat[d.argmax()], b.flat[d.argmax()]))
