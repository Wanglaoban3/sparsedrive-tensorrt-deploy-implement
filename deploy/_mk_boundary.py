# -*- coding: utf-8 -*-
"""Make f16 boundary file from P3 f32 dump of e_T6 Reshape_9_output_0."""
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
B = (r"<REPO>\work_dirs"
     r"\sparsedrive_small_stage2\evaldata\dump_out"
     r"\Reshape_9_output_0.__dump.bin")
OUT = (r"<REPO>\work_dirs"
       r"\sparsedrive_small_stage2\evaldata\feat_flat_f16.bin")
a = np.fromfile(B, np.float32)
print("dump elems", a.size, "expect", 89760 * 256)
assert a.size == 89760 * 256
print("stats: min %.4f max %.4f mean %.5f nan=%d" %
      (a.min(), a.max(), a.mean(), np.isnan(a).sum()))
h = a.astype(np.float16)
h.tofile(OUT)
import os
print("wrote", OUT, os.path.getsize(OUT), "bytes")
