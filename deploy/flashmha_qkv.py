"""FlashMHA QKV surgery: make the packed QKV projection quantizable.

This repo's attention (projects/mmdet3d_plugin/models/attention.py) implements
QKV as a packed raw Parameter (FlashMHA.in_proj_weight [3E, E]) consumed by a
functional F.linear call, so ModelOpt MTQ cannot wrap it - while out_proj is a
regular nn.Linear and IS quantized by INT8_DEFAULT_CFG.  These helpers split
the packed weight into q/k/v nn.Linear modules (numerically identical math)
and patch FlashMHA.forward to use them, so MTQ covers QKV with the same
weight+input+output quantizer scheme as every other Linear.

Order of use:
    build model -> load fp32 ckpt (in_proj_* consumed)
    -> flashmha_qkv.pack_to_linears(model)   [deletes in_proj_*]
    -> MTQ quantize / train / save           [ckpt now carries *_proj keys]

When re-loading a QAT checkpoint: build model -> pack_to_linears(model)
-> load_checkpoint(qat ckpt, strict=False) -> keys line up.
"""

import torch.nn as nn
from einops import rearrange

from projects.mmdet3d_plugin.models.attention import FlashMHA


def _flashmha_forward_linear(self, q, k, v, key_padding_mask=None):
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)
    q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
    k = rearrange(k, "b s (h d) -> b s h d", h=self.num_heads)
    v = rearrange(v, "b s (h d) -> b s h d", h=self.num_heads)
    context, attn_weights = self.inner_attn(
        q, k, v, key_padding_mask=key_padding_mask, causal=self.causal)
    return self.out_proj(
        rearrange(context, "b s h d -> b s (h d)")), attn_weights


def pack_to_linears(model):
    """Split every FlashMHA.in_proj_* into q/k/v Linear submodules.

    Returns the number of converted attention modules.  Idempotent.
    """
    FlashMHA.forward = _flashmha_forward_linear
    n = 0
    for mod in model.modules():
        if mod.__class__.__name__ != "FlashMHA":
            continue
        if not hasattr(mod, "in_proj_weight") or mod.in_proj_weight is None:
            continue
        E = mod.in_proj_weight.shape[1]
        w = mod.in_proj_weight.data
        b = mod.in_proj_bias.data if mod.in_proj_bias is not None else None
        for i, part in enumerate(("q", "k", "v")):
            lin = nn.Linear(E, E, bias=b is not None)
            lin.weight.data = w[i * E:(i + 1) * E].clone()
            if b is not None:
                lin.bias.data = b[i * E:(i + 1) * E].clone()
            lin.to(w.device)
            setattr(mod, f"{part}_proj", lin)
        del mod.in_proj_weight
        if mod.in_proj_bias is not None:
            del mod.in_proj_bias
        n += 1
    return n
