# -*- coding: utf-8 -*-
"""verify c{k}_w3.f16 == raw logits (v3 semantics) vs post-softmax weights."""
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
W = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2\dfa_real_engine")
for k in (0, 1):
    w3 = np.fromfile(W + "\\c%d_w3.f16" % k, np.float16)
    lg = np.fromfile(W + "\\c%d_log.f16" % k, np.float16)
    same = np.array_equal(w3, lg)
    shp = (1, 900, 6, 4, 13, 8) if k == 0 else (1, 100, 6, 4, 300, 8)
    w = w3.reshape(shp).astype(np.float32)
    # if raw logits: softmax over (2,3,4) should normalize to 1
    sm = np.exp(w - w.max(axis=(2, 3, 4), keepdims=True))
    sm = sm / sm.sum(axis=(2, 3, 4), keepdims=True)
    rowsum = w.sum(axis=(2, 3, 4))
    print(f"c{k}: w3==log:{same} |w|max={np.abs(w).max():.3f} "
          f"rowsum[min,max]=[{rowsum.min():.3f},{rowsum.max():.3f}] "
          f"softmaxed-rowsum[min,max]="
          f"[{sm.sum(axis=(2,3,4)).min():.6f},{sm.sum(axis=(2,3,4)).max():.6f}]")
