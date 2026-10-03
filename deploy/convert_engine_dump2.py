# -*- coding: utf-8 -*-
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

feat_name = next(k for k in d if k.startswith("dump_dfa_feat"))
feat = d[feat_name]
# fp32 → f16
feat.astype(np.float16).tofile(os.path.join(OUT, "feat.f16"))

shape = np.tile(np.array([[64, 176], [32, 88], [16, 44], [8, 22]],
                         np.int32)[None], (6, 1))
shape.tofile(os.path.join(OUT, "shape.i32"))
offs = [0, 11264, 14080, 14784]
ssi = np.array([[o for o in offs]] * 6, np.int32)
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
    d[out_k].tofile(os.path.join(OUT, f"c{k}_ref32.f32"))

with open(os.path.join(OUT, "manifest.txt"), "w") as f:
    for rank, (k, tag, A, P) in enumerate(
            sorted(calls, key=lambda x: (0 if x[1] == "map" else 1, x[0]))):
        f.write(f"{rank} {tag} {A} c{k}\n")

# skip stats
for k, tag, A, P in calls:
    w = d[next(kk for kk in d
               if kk.startswith(f"dump_dfa{k}_log"))].astype(np.float32)
    loc = d[next(kk for kk in d
                 if kk.startswith(f"dump_dfa{k}_loc"))].astype(np.float32)
    # w: [bs,A,cams,S,P,G]   loc: [bs,A,P,cams,2]
    valid = ((loc[..., 0] > 0) & (loc[..., 0] < 1)
             & (loc[..., 1] > 0) & (loc[..., 1] < 1))  # [bs,A,P,cams]
    vT = valid.transpose(0, 1, 3, 2)[:, :, :, None, :, None]  # [b,A,cams,1,P,1]
    live = (w >= 1e-4) & vT
    mass = w[live].sum() / max(1e-9, np.abs(w).sum())
    print(f"[{tag} c{k}] |w|max={np.abs(w).max():.4f} "
          f"live={100*live.mean():.2f}% massfrac={100*mass:.1f}%")

print("files:", len(os.listdir(OUT)))
print("CONVERT_ENGINE_DUMP_DONE ->", OUT)
