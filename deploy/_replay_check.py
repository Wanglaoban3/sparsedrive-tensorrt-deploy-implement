# -*- coding: utf-8 -*-
"""重放误差地图: numpy(Q,K^T,V) 对比 dbg 内核的 S/P/O, 定位错误 tile 模式."""
from pathlib import Path

import numpy as np

W = Path(r"<REPO>"
         r"\work_dirs\sparsedrive_small_stage2")
PD = W / "outs_probe_fa"
RD = W / "outs_replay"
P = "/layers.19/attn/inner_attn/"


def get(d, name, dt, shape):
    a = np.fromfile(d / name, dtype=dt)
    return a.reshape(shape)


Q = get(PD, "_layers.19_attn_inner_attn_Cast_1_output_0.bin",
        np.float16, (8, 900, 64)).astype(np.float32)
KT = get(PD, "_layers.19_attn_inner_attn_Transpose_3_output_0.bin",
         np.float16, (8, 64, 900)).astype(np.float32)
V = get(PD, "_layers.19_attn_inner_attn_Cast_3_output_0.bin",
        np.float16, (8, 900, 64)).astype(np.float32)
S = get(RD, "S_fa2.bin", np.float16, (8, 900, 900)).astype(np.float32)
Pp = get(RD, "P_fa2.bin", np.float16, (8, 900, 900)).astype(np.float32)
O = get(RD, "O_fa2.bin", np.float16, (8, 900, 64)).astype(np.float32)

Sref = np.einsum("hqd,hdk->hqk", Q, KT)
Pn = np.exp((Sref * 0.125 - (Sref * 0.125).max(-1, keepdims=True)))
Pn /= Pn.sum(-1, keepdims=True)
Oref = np.einsum("hqk,hkd->hqd", Pn, V)


def l2(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return np.linalg.norm(d) / (np.linalg.norm(a.astype(np.float64)) + 1e-30)


print("S  l2rel =", f"{l2(Sref, S):.3e}")
print("P  l2rel =", f"{l2(Pn, Pp):.3e}")
print("O  l2rel =", f"{l2(Oref, O):.3e}")

# 误差地图: S 的 |err|>0.5 的位置模式 (前 3 头)
h = 0
err = np.abs(Sref[h] - S[h])
bad = err > 0.5
print(f"\nhead{h}: bad cells {bad.sum()}/{bad.size}")
rows = np.where(bad.any(1))[0]
cols = np.where(bad.any(0))[0]
print("bad rows:", rows[:40], "..." if len(rows) > 40 else "")
print("bad cols:", cols[:40], "..." if len(cols) > 40 else "")
if len(rows) and len(cols):
    rr, cc = np.where(bad)
    print("sample (row,col,ref,got):")
    for i in range(min(10, len(rr))):
        r, c = rr[i], cc[i]
        print(f"  ({r},{c}) ref={Sref[h][r][c]:9.3f} got={S[h][r][c]:9.3f}")
    # tile 归属: 列块 = c//16 (warp 列), 行 = r%16
    print("bad col-tile hist:", np.bincount(cc // 16, minlength=57)[:16])
    print("bad row-mod-16 hist:", np.bincount(rr % 16, minlength=16))

# P 检查
errp = np.abs(Pn - Pp)
print(f"\nP: bad(>1e-2) {(errp > 1e-2).sum()}/{Pn.size}, "
      f"max {errp.max():.4f}")
# O 检查
erro = np.abs(Oref - O)
print(f"O: bad(>0.05) {(erro > 0.05).sum()}/{O.size}, max {erro.max():.4f}")
print("O head0 [0,:8]:", O[0][0][:8].tolist())
print("Oref    [0,:8]:", Oref[0][0][:8].tolist())
