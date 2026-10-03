# -*- coding: utf-8 -*-
"""convert engine dump -> test_dfa_real bins (real engine-bit-exact data)"""
import csv
import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
DUMP = (r"<REPO>\work_dirs"
        r"\sparsedrive_small_stage2\evaldata\dump_out")
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


d = load_manifest(DUMP)
os.makedirs(OUT, exist_ok=True)

# find DFA tensors by name pattern
feat_name = next(k for k in d if k.startswith("dump_dfa_feat"))
feat = d[feat_name]
print("feat:", feat.shape, feat.dtype)
feat.tofile(os.path.join(OUT, "feat.f16"))

shape = np.array([[64, 176], [32, 88], [16, 44], [8, 22]], np.int32)
shape = np.tile(shape[None], (6, 1))  # [6,4,2]
shape.tofile(os.path.join(OUT, "shape.i32"))
ssi_l = [0]
for h, w in [(64, 176), (32, 88), (16, 44), (8, 22)][:-1]:
    ssi_l.append(ssi_l[-1] + h * w)
ssi = np.array([[ssi_l[0], ssi_l[1], ssi_l[2], ssi_l[3]]] * 6, np.int32)
ssi.tofile(os.path.join(OUT, "ssi.i32"))

calls = []
for k in range(12):
    loc_k = next(kk for kk in d if kk.startswith(f"dump_dfa{k}_loc"))
    log_k = next(kk for kk in d if kk.startswith(f"dump_dfa{k}_log"))
    out_k = next(kk for kk in d if kk.startswith(f"dump_dfa{k}_out"))
    loc = d[loc_k]
    A = loc.shape[1]
    P = loc.shape[2]
    tag = "det" if A == 900 else "map"
    calls.append((k, tag, A, P))
    loc.astype(np.float16).tofile(os.path.join(OUT, f"c{k}_loc.f16"))
    d[log_k].astype(np.float16).tofile(os.path.join(OUT, f"c{k}_log.f16"))
    d[out_k].astype(np.float16).tofile(os.path.join(OUT, f"c{k}_out.f16"))
    # save fp32 reference too (engine bit-exact)
    d[out_k].tofile(os.path.join(OUT, f"c{k}_ref32.f32"))

with open(os.path.join(OUT, "manifest.txt"), "w") as f:
    for rank, (k, tag, A, P) in enumerate(
            sorted(calls, key=lambda x: (x[2], x[0]))):
        idx = [c[0] for c in calls].index(k)
        f.write(f"{rank} {tag} {A} c{k}\n")

# print stats
for k, tag, A, P in calls:
    log_k = next(kk for kk in d if kk.startswith(f"dump_dfa{k}_log"))
    w = d[log_k].astype(np.float32)
    loc_k = next(kk for kk in d if kk.startswith(f"dump_dfa{k}_loc"))
    loc = d[loc_k].astype(np.float32)
    # w: [1,A,cams,S,P,G]  loc: [1,A,P,cams,2]
    valid = ((loc[..., 0] > 0) & (loc[..., 0] < 1)
             & (loc[..., 1] > 0) & (loc[..., 1] < 1))  # [1,A,P,cams]
    # reorder valid to [1,A,cams,S,P] to match w dims 2,3,4
    valid_r = valid.transpose(0, 1, 3, 4, 2)  # [1,A,cams,S,P]
    live = (w >= 1e-4) & valid_r[..., None]  # broadcast over G
    print(f"[{tag} c{k}] w max={w.max():.4f} "
          f"live={100*live.mean():.2f}% "
          f"mass_kept={100*w[live].sum()/max(1e-9,w.sum()):.1f}%")
print("CONVERT_ENGINE_DUMP_DONE ->", OUT)
