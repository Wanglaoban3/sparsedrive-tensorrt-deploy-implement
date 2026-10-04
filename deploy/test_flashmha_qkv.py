"""Numerics check for the FlashMHA QKV surgery: packed in_proj vs split
q/k/v Linears must agree bit-exactly.

Run from project root:
    python deploy/test_flashmha_qkv.py
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "deploy"))
os.chdir(ROOT)

import projects.mmdet3d_plugin  # noqa: F401
from projects.mmdet3d_plugin.models.attention import FlashMHA, \
    _in_projection_packed

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

mha = FlashMHA(embed_dim=256, num_heads=8).to(dev).eval()
q = torch.randn(2, 100, 256, device=dev)
k = torch.randn(2, 100, 256, device=dev)
v = torch.randn(2, 100, 256, device=dev)

with torch.no_grad():
    out_ref = mha(q, k, v)

from flashmha_qkv import pack_to_linears
n = pack_to_linears(mha)
assert n == 1, n
with torch.no_grad():
    out_new = mha(q, k, v)

d0 = (out_ref[0] - out_new[0]).abs().max().item()
print(f"packed vs q/k/v Linears: max|diff| = {d0:.3e}")
assert d0 == 0.0, "surgery must be bit-exact"
print("keys:", sorted(p for p, _ in mha.named_parameters()))
print("FLASHMHA_QKV_OK")
