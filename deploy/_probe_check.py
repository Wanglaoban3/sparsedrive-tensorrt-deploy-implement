# -*- coding: utf-8 -*-
"""SSA5 探针对拍: numpy 从 dump 的 Q/K^T/V 算参考注意力, 对比
基线链输出与 FlashSDPA 输出, 定位内核错误模式."""
from pathlib import Path

import numpy as np

W = Path(r"<REPO>"
         r"\work_dirs\sparsedrive_small_stage2")
PB = W / "outs_probe_base"
PF = W / "outs_probe_fa"
DT = {"f32": (np.float32, 4), "f16": (np.float16, 2), "i32": (np.int32, 4),
      "i64": (np.int64, 8), "b8": (np.int8, 1)}


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


base, fa = load_dir(PB), load_dir(PF)
print("base tensors:")
for k in sorted(base):
    print("   ", k, base[k].shape, base[k].dtype)
print("fa tensors:")
for k in sorted(fa):
    print("   ", k, fa[k].shape, fa[k].dtype)

Q = fa["/layers.19/attn/inner_attn/Cast_1_output_0"][0].astype(np.float32)
KT = fa["/layers.19/attn/inner_attn/Transpose_3_output_0"][0].astype(np.float32)
V = fa["/layers.19/attn/inner_attn/Cast_3_output_0"][0].astype(np.float32)
print("Q", Q.shape, "KT", KT.shape, "V", V.shape)
sc = 0.125

# numpy 参考 (f32 全程)
S = np.einsum("hqd,hdk->hqk", Q, KT)  # KT[h,d,k] → S[h,q,k]
P = np.exp((S * sc - (S * sc).max(-1, keepdims=True)))
P /= P.sum(-1, keepdims=True)
Oref = np.einsum("hqk,hkd->hqd", P, V)


def cmp(a, b, tag):
    d = a.astype(np.float64) - b.astype(np.float64)
    l2 = np.linalg.norm(d) / (np.linalg.norm(a.astype(np.float64)) + 1e-30)
    print(f"  {tag:44s} l2rel={l2:.3e} maxabs={np.abs(d).max():.3e}")
    return l2


# 基线链张量
if "/layers.19/attn/inner_attn/MatMul_output_0" in base:
    b = {k.replace("/layers.19/attn/inner_attn/", ""): v[0]
         for k, v in base.items() if "layers.19" in k}
    print("baseline chain vs numpy:")
    cmp(S.astype(np.float16).astype(np.float32), b["MatMul_output_0"],
        "S_f16(numpy) vs 基线 MatMul(QK^T)")
    sm_out = b["Softmax_output_0"]
    Pm = (b["Mul_output_0"])
    m = Pm.max(-1, keepdims=True)
    Pn = np.exp(Pm - m)
    Pn /= Pn.sum(-1, keepdims=True)
    cmp(Pn.astype(np.float16).astype(np.float32), sm_out,
        "P(numpy 同链舍入) vs 基线 Softmax")
    Obase = b["MatMul_1_output_0"]
    cmp(Obase, Oref.astype(np.float16).astype(np.float32),
        "Obase vs numpy 全程f32参考")
# FA 插件输出
O_fa = fa["/layers.19/attn/inner_attn/MatMul_1_output_0"][0]
print("plugin vs numpy / baseline:")
cmp(O_fa, Oref.astype(np.float16).astype(np.float32), "O_fa vs numpy f32参考")
if "/layers.19/attn/inner_attn/MatMul_output_0" in base:
    cmp(O_fa, Obase, "O_fa vs O_base(同输入同数据)")
    cmp(Obase, Obase, "sanity self")
np.save(W / "_probe_Oref.npy", Oref)
