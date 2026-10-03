# -*- coding: utf-8 -*-
"""探针深诊断: S 不匹配的布局/来源穷举."""
from pathlib import Path

import numpy as np

W = Path(r"<REPO>"
         r"\work_dirs\sparsedrive_small_stage2")
DT = {"f32": (np.float32, 4), "f16": (np.float16, 2), "b8": (np.int8, 1),
      "i32": (np.int32, 4), "i64": (np.int64, 8)}


def load_dir(d):
    out = {}
    for line in (d / "manifest.tsv").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        name, dt, dims, fn = parts
        t, _ = DT[dt]
        a = np.frombuffer((d / fn.lstrip("/")).read_bytes(), dtype=t)
        dims = [int(x) for x in dims.split(",") if x]
        out[name] = a.reshape(dims) if dims else a
    return out


base = load_dir(W / "outs_probe_base")
fa = load_dir(W / "outs_probe_fa")
P = "/layers.19/attn/inner_attn/"

Qb = base[P + "Cast_1_output_0"][0].astype(np.float32)
KTb = base[P + "Transpose_3_output_0"][0].astype(np.float32)
Sb = base[P + "MatMul_output_0"][0].astype(np.float32)
Vb = base[P + "Cast_3_output_0"][0].astype(np.float32)
Ob = base[P + "MatMul_1_output_0"][0].astype(np.float32)
Qf = fa[P + "Cast_1_output_0"][0]
KTf = fa[P + "Transpose_3_output_0"][0]
Of = fa[P + "MatMul_1_output_0"][0].astype(np.float32)

print("== 探针间一致性 (上游确定性应逐位同) ==")
print("Q base==fa:", np.array_equal(Qb.astype(np.float16), Qf))
print("KT base==fa:", np.array_equal(KTb.astype(np.float16), KTf))

print("== 量级 ==")
for nm, a in [("Q", Qb), ("KT", KTb), ("S_base", Sb), ("V", Vb),
              ("O_base", Ob), ("O_fa", Of)]:
    print(f"  {nm:8s} min={a.min():9.3f} max={a.max():9.3f} "
          f"absmean={np.abs(a).mean():8.4f}")

def l2(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return np.linalg.norm(d) / (np.linalg.norm(a.astype(np.float64)) + 1e-30)

print("== S 变体穷举 (vs 基线 MatMul 输出) ==")
h = 3  # 单头看更清楚
Sn = np.einsum("qd,dk->qk", Qb[h], KTb[h])
print(f"  A: Q@KT        (h{h}) l2={l2(Sn, Sb[h]):.3e}  "
      f"numpy max={np.abs(Sn).max():.1f}")
Sn2 = np.einsum("qd,kd->qk", Qb[h], KTb[h].T)
print(f"  B: Q@KT.T      (h{h}) l2={l2(Sn2, Sb[h]):.3e}  "
      f"numpy max={np.abs(Sn2).max():.1f}")
Sn3 = np.einsum("dq,dk->qk", Qb[h].T, KTb[h])
print(f"  C: Q.T@KT      (h{h}) l2={l2(Sn3, Sb[h]):.3e}")
Sn4 = np.einsum("qd,dk->kq", Qb[h], KTb[h])
print(f"  D: (Q@KT).T    (h{h}) l2={l2(Sn4, Sb[h]):.3e}")

print("== O 变体穷举 (P基线 vs 基线 V) ==")
Pb = base[P + "Cast_5_output_0"][0].astype(np.float32)
On1 = np.einsum("qk,kd->qd", Pb[h], Vb[h])
print(f"  P@V        (h{h}) l2={l2(On1, Ob[h]):.3e}")
On2 = np.einsum("qk,dk->qd", Pb[h], Vb[h])
print(f"  P@V.T      (h{h}) l2={l2(On2, Ob[h]):.3e}")

print("== O_fa vs O_base 头部分布 ==")
for hh in [0, 3]:
    print(f"  h{hh}: l2={l2(Of[hh], Ob[hh]):.3e}  "
          f"O_fa[:3,:3]={Of[hh][:3,:3].tolist()}  "
          f"O_base[:3,:3]={Ob[hh][:3,:3].tolist()}")
