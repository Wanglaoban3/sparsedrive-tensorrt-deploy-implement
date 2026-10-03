# -*- coding: utf-8 -*-
"""numpy 仿真: v3 语义 vs v8 plan+gather 算法 vs dump ref, 定位 v8 bug。
用 f32 原始 dump (dump_out), call c0 = det (A=900,P=13)."""
import csv
import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
DUMP = (r"<REPO>\work_dirs"
        r"\sparsedrive_small_stage2\evaldata\dump_out")
EPS = 1e-4
CAMS, S, G, C = 6, 4, 8, 256


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
feat = d["dump_dfa_feat"][0]              # [89760,256] f32
feat = feat.reshape(-1)                   # flat 元素偏移索引
k = 0
loc = d[f"dump_dfa{k}_loc"][0]            # [A,P,cams,2]
log = d[f"dump_dfa{k}_log"][0]            # [A,cams,S,P,G]
ref = d[f"dump_dfa{k}_out"][0]            # [A,256]
A, P = loc.shape[0], loc.shape[1]
NCSP = CAMS * S * P
shape = np.tile(np.array([[64, 176], [32, 88], [16, 44], [8, 22]],
                         dtype=np.int32), (CAMS, 1, 1))  # [cams,S,2]
# 真实 ssi (从 v5_P1h.onnx initializer 提取, flat [cams*S], cam c 起点 c*14960)
offs = np.array(
    [0, 11264, 14080, 14784, 14960, 26224, 29040, 29744, 29920, 41184,
     44000, 44704, 44880, 56144, 58960, 59664, 59840, 71104, 73920, 74624,
     74800, 86064, 88880, 89584], np.int64).reshape(CAMS, S)  # [cams,S]
print("A,P,NCSP:", A, P, NCSP, " ref absmax", np.abs(ref).max())

# softmax stats per (a,g)
lmax = log.max(axis=(1, 2, 3), keepdims=True)          # [A,1,1,1,G]
w = np.exp(log - lmax)
lsum = w.sum(axis=(1, 2, 3), keepdims=True)            # [A,1,1,1,G]
wt = (w / lsum).astype(np.float32)                     # [A,cams,S,P,G]

# loc validity [A,P,cams]
valid = ((loc[..., 0] > 0) & (loc[..., 0] < 1)
         & (loc[..., 1] > 0) & (loc[..., 1] < 1))      # [A,P,cams]
vT = valid.transpose(0, 2, 1)[:, :, None, :, None]     # [A,cams,1,P,1]
mass_all = wt.sum()
mass_live = (wt * vT).sum()
mass_eps = (wt * (wt >= EPS) * vT).sum()
live_frac = ((wt >= EPS) & vT).mean()
print(f"mass total={mass_all:.1f} loc-valid={mass_live:.1f} "
      f"live(wt>=eps)={mass_eps:.1f}  dropped={1-mass_eps/mass_all:.4%} "
      f"live-entry-frac={live_frac:.4%}")


def bilinear_wt(a, p, c, s):
    lw_, lh_ = loc[a, p, c]
    h, w_ = shape[c, s]
    start = offs[c, s]
    h_im = lh_ * h - 0.5
    w_im = lw_ * w_ - 0.5
    h_low, w_low = int(np.floor(h_im)), int(np.floor(w_im))
    lh, lw2 = h_im - h_low, w_im - w_low
    hh, hw = 1 - lh, 1 - lw2
    base = start * C
    r0 = h_low * w_ * C
    r1 = r0 + w_ * C
    c0 = w_low * C
    c1 = c0 + C
    o = [base + r0 + c0, base + r0 + c1, base + r1 + c0, base + r1 + c1]
    ww_is_set = [
        h_low >= 0 and w_low >= 0,
        h_low >= 0 and w_low + 1 <= w_ - 1,
        h_low + 1 <= h - 1 and w_low >= 0,
        h_low + 1 <= h - 1 and w_low + 1 <= w_ - 1,
    ]
    for j in range(4):
        if ww_is_set[j] and o[j] >= feat.size:
            print(f"OOB a={a} p={p} c={c} s={s} loc={lw_:.4f},{lh_:.4f} "
                  f"h={h} w={w_} start={start} h_low={h_low} w_low={w_low} "
                  f"o[{j}]={o[j]}")
            raise SystemExit(1)
    ww = [0.0, 0.0, 0.0, 0.0]
    if h_low >= 0 and w_low >= 0:
        ww[0] = hh * hw
    if h_low >= 0 and w_low + 1 <= w_ - 1:
        ww[1] = hh * lw2
    if h_low + 1 <= h - 1 and w_low >= 0:
        ww[2] = lh * hw
    if h_low + 1 <= h - 1 and w_low + 1 <= w_ - 1:
        ww[3] = lh * lw2
    return o, ww


def df_full(wfun, use_valid, eps):
    """wfun(a,c,s,p,g)->float; 通用 DF 仿真"""
    out = np.zeros((A, C), np.float32)
    for a in range(A):
        for g in range(G):
            for p in range(P):
                for c in range(CAMS):
                    if use_valid and not valid[a, p, c]:
                        continue
                    for s in range(S):
                        wtg = wfun(a, c, s, p, g)
                        if eps is not None and wtg < eps:
                            continue
                        o, ww = bilinear_wt(a, p, c, s)
                        v = sum(ww[j] * feat[o[j]] for j in range(4)
                                if ww[j] != 0.0 and o[j] < feat.size)
                        out[a] += wtg * v
    return out


sm_joint = wt  # softmax over (cams,S,P) 已算好
sm_sp = np.exp(log - log.max(axis=(2, 3, 4), keepdims=True))
sm_sp /= sm_sp.sum(axis=(2, 3, 4), keepdims=True)      # over (S,P) per (a,c,g)
sm_p = np.exp(log - log.max(axis=(4,), keepdims=True))
sm_p /= sm_p.sum(axis=(4,), keepdims=True)             # over P per (a,c,s,g)

variants = {
    "a_joint_eps": (lambda a, c, s, p, g: sm_joint[a, c, s, p, g], True, EPS),
    "a_joint_noeps": (lambda a, c, s, p, g: sm_joint[a, c, s, p, g], True, None),
    "b_sp_eps": (lambda a, c, s, p, g: sm_sp[a, c, s, p, g], True, EPS),
    "c_p_eps": (lambda a, c, s, p, g: sm_p[a, c, s, p, g], True, EPS),
    "d_rawlog_noeps": (lambda a, c, s, p, g: log[a, c, s, p, g], True, None),
    "e_joint_eps_novalid": (lambda a, c, s, p, g: sm_joint[a, c, s, p, g],
                            False, EPS),
}
refn = np.linalg.norm(ref)
for name, (fun, uv, eps) in variants.items():
    out = df_full(fun, uv, eps)
    dl = out - ref
    print(f"{name:22s} vs ref: l2rel={np.linalg.norm(dl)/refn:.3e} "
          f"maxabs={np.abs(dl).max():.3e} absmax={np.abs(out).max():.3f}")

# ---- v8 仿真: plan (entries per (a,g)) + gather (每 g 只算自己 32 通道) ----
entries = [[[] for _ in range(G)] for _ in range(A)]
for a in range(A):
    for i in range(NCSP):
        c, rem = divmod(i, S * P)
        s, p = divmod(rem, P)
        if not valid[a, p, c]:
            continue
        o, ww = bilinear_wt(a, p, c, s)
        for g in range(G):
            wtg = sm_joint[a, c, s, p, g]
            if wtg < EPS:
                continue
            entries[a][g].append((o, [ww[j] * wtg for j in range(4)]))
out8 = np.zeros((A, C), np.float32)
for a in range(A):
    for g in range(G):
        chlo, chhi = g * 32, (g + 1) * 32
        for o, w4 in entries[a][g]:
            for j in range(4):
                out8[a, chlo:chhi] += w4[j] * feat[o[j] + chlo: o[j] + chhi]
out3 = df_full(*variants["a_joint_eps"])
d83 = out8 - out3
print("v8sim vs v3sim: l2rel=%.3e maxabs=%.3e  entries=%d" %
      (np.linalg.norm(d83) / max(1e-12, np.linalg.norm(out3)),
       np.abs(d83).max(), sum(len(e) for a_ in entries for e in a_)))
d8 = out8 - ref
print("v8sim vs ref: l2rel=%.3e maxabs=%.3e" %
      (np.linalg.norm(d8) / max(1e-12, np.linalg.norm(ref)),
       np.abs(d8).max()))
print("LOCAL_SIM_DONE")
