# -*- coding: utf-8 -*-
"""preprocess engine dump: logits [1,A,cams,S,P,G] → softmax+transpose
→ weights [1,A,P,cams,S,G] (what v3 kernel expects). Save as _w3.f16."""
import csv
import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
OUT = (r"<REPO>\work_dirs"
       r"\sparsedrive_small_stage2\dfa_real_engine")


def load_manifest(d):
    o = {}
    with open(os.path.join(d, "manifest.tsv")) as f:
        for row in csv.reader(f, delimiter="\t"):
            if not row:
                continue
            name, dt, shape, fn = row[0], row[1], row[2], row[3]
            a = np.fromfile(os.path.join(d, fn),
                            np.float32 if dt == "f32" else np.int32)
            o[name] = a.reshape(tuple(int(x) for x in shape.split(",")))
    return o


d = load_manifest(OUT)

for k in range(12):
    log_k = next(kk for kk in d if kk.startswith(f"dump_dfa{k}_log"))
    log = d[log_k].astype(np.float32)  # [1,A,cams,S,P,G]
    A = log.shape[1]
    # softmax over (cams,S,P) per (a,g) → dims 2,3,4
    axes = (2, 3, 4)
    lmax = log.max(axis=axes, keepdims=True)
    lsum = np.exp(log - lmax).sum(axis=axes, keepdims=True)
    w = np.exp(log - lmax) / lsum  # [1,A,cams,S,P,G] softmaxed
    # transpose to v3 layout [1,A,P,cams,S,G]
    w3 = w.transpose(0, 1, 4, 2, 3, 5).astype(np.float16)
    w3.tofile(os.path.join(OUT, f"c{k}_w3.f16"))
    print(f"c{k}: w3 [{w3.shape}] saved")

print("PREPROCESS_DONE")
