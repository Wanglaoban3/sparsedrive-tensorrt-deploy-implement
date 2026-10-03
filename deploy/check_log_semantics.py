# -*- coding: utf-8 -*-
import glob
import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
d = r"<REPO>\work_dirs" \
    r"\sparsedrive_small_stage2\evaldata\dump_out"
for k in [0, 1]:
    f = glob.glob(os.path.join(d, f"dump_dfa{k}_log*"))[0]
    a = np.fromfile(f, np.float32)
    n = a.size
    A = n // (6 * 4 * 8)
    a = a.reshape(1, A, 6, 4, -1, 8)
    rs = a.sum(axis=(2, 3, 4))
    print(os.path.basename(f), a.shape,
          "rowsum min/max: %.6f %.6f" % (rs.min(), rs.max()),
          "absmax %.3f" % np.abs(a).max())
    # 是否有负值/大于1的值 (logits 特征; 权重应 in [0,1])
    print("  min %.4f max %.4f  frac>1: %.4g" %
          (a.min(), a.max(), (a > 1).mean()))
