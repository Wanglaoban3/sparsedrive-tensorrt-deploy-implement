"""Pure-PyTorch equivalent of the DeformableAggregation CUDA op.

Numerically mirrors projects/mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu:
  - normalized (w, h) sampling locations; contribution is ZERO when
    loc_w <= 0 or loc_w >= 1 or loc_h <= 0 or loc_h >= 1 (strict exclusion)
  - pixel-center alignment: h_im = loc_h * H - 0.5, w_im = loc_w * W - 0.5
  - bilinear corners with per-corner bounds guards (out-of-range corner -> 0)
  - weights layout [bs, anchors, pts, cams, scales, groups]; each group scalar
    is broadcast over the group's channels (num_embeds / num_groups)

Used automatically when the compiled extension is unavailable (e.g. no nvcc /
MSVC), so PTQ calibration / QAT training / ONNX trace can run on any box.
The ONNX export still emits the SparseDrive::DeformableAggregation custom node
because DeformableAggregationFunction.symbolic overrides tracing.

The intermediates are [bs, A, P, cams, S, C]; with the det head's
A~1800 / P=13 / C=256 a single one is >1 GB, so the anchor axis can be
processed in chunks (SPARSEDRIVE_DFA_CHUNK, e.g. 256) to cap the peak.
"""

import os

import torch


def deformable_aggregation_pure(
    mc_ms_feat,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
):
    chunk = int(os.environ.get("SPARSEDRIVE_DFA_CHUNK", "0"))
    pchunk = int(os.environ.get("SPARSEDRIVE_DFA_PCHUNK", "32"))
    if os.environ.get("SPARSEDRIVE_DFA_DEBUG", "0") == "1":
        print(f"[DAF] shapes={tuple(weights.shape)} chunk={chunk}", flush=True)
    if chunk > 0 and weights.shape[1] > chunk:
        outs = [
            deformable_aggregation_pure(
                mc_ms_feat,
                spatial_shape,
                scale_start_index,
                sampling_location[:, a:b],
                weights[:, a:b],
            )
            for a in range(0, weights.shape[1], chunk)
            for b in [min(a + chunk, weights.shape[1])]
        ]
        return torch.cat(outs, dim=1)
    if pchunk > 0 and weights.shape[2] > pchunk:
        # map head: A=100 anchors but P=300 points -> chunk the point axis.
        # P is summed over inside impl, so slices combine by ADDITION.
        total = None
        for p in range(0, weights.shape[2], pchunk):
            q = min(p + pchunk, weights.shape[2])
            part = deformable_aggregation_pure(
                mc_ms_feat,
                spatial_shape,
                scale_start_index,
                sampling_location[:, :, p:q],
                weights[:, :, p:q],
            )
            total = part if total is None else total + part
        return total
    return _deformable_aggregation_pure_impl(
        mc_ms_feat, spatial_shape, scale_start_index, sampling_location,
        weights)


def _deformable_aggregation_pure_impl(
    mc_ms_feat,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
):
    """Vectorized, fully differentiable re-implementation of the CUDA kernel.

    mc_ms_feat:        [bs, num_feat, C]   float32
    spatial_shape:     [num_cams, scales, 2] int (H, W per cam/scale)
    scale_start_index: [num_cams, scales]  int, flat start row of each block
    sampling_location: [bs, A, P, cams, 2] float, (w, h) in [0, 1]
    weights:           [bs, A, P, cams, scales, G] float
    returns:           [bs, A, C]
    """
    mc_ms_feat = mc_ms_feat.contiguous().float()
    sampling_location = sampling_location.contiguous().float()
    weights = weights.contiguous().float()

    bs, num_feat, C = mc_ms_feat.shape
    A, P = weights.shape[1], weights.shape[2]
    G = weights.shape[-1]
    cg = C // G
    h_sz = spatial_shape[..., 0].long()  # [cams, S]
    w_sz = spatial_shape[..., 1].long()
    cams, S = h_sz.shape
    ssi = scale_start_index.long()  # [cams, S]

    loc_w = sampling_location[..., 0]  # [bs, A, P, cams]
    loc_h = sampling_location[..., 1]
    valid = (loc_w > 0) & (loc_w < 1) & (loc_h > 0) & (loc_h < 1)

    h_im = loc_h[..., None] * h_sz.float() - 0.5  # [bs, A, P, cams, S]
    w_im = loc_w[..., None] * w_sz.float() - 0.5
    h_low = h_im.floor()
    w_low = w_im.floor()
    lh = h_im - h_low
    lw = w_im - w_low
    hh = 1 - lh
    hw = 1 - lw
    h_low_i = h_low.long()
    w_low_i = w_low.long()
    h_high_i = h_low_i + 1
    w_high_i = w_low_i + 1

    m1 = (h_low_i >= 0) & (w_low_i >= 0)
    m2 = (h_low_i >= 0) & (w_high_i <= w_sz - 1)
    m3 = (h_high_i <= h_sz - 1) & (w_low_i >= 0)
    m4 = (h_high_i <= h_sz - 1) & (w_high_i <= w_sz - 1)

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw

    def take(h_i, w_i, mask):
        # h_i, w_i, mask: [bs, A, P, cams, S]
        r = ssi.view(1, 1, cams, S) + (h_i * w_sz + w_i)  # [bs, A, P, cams, S]
        r = r.reshape(bs, -1, 1).clamp(0, num_feat - 1)
        r = r.expand(-1, -1, C)
        vals = flat.gather(1, r)  # [bs, A*P*cams*S, C]
        vals = vals.view(bs, A, P, cams, S, C)
        return vals * mask[..., None]

    flat = mc_ms_feat  # [bs, num_feat, C]
    v1 = take(h_low_i, w_low_i, m1)
    v2 = take(h_low_i, w_high_i, m2)
    v3 = take(h_high_i, w_low_i, m3)
    v4 = take(h_high_i, w_high_i, m4)

    val = (
        w1[..., None] * v1
        + w2[..., None] * v2
        + w3[..., None] * v3
        + w4[..., None] * v4
    )
    val = val * valid[..., None, None]

    w_exp = (
        weights.unsqueeze(-1)
        .expand(-1, -1, -1, -1, -1, -1, cg)
        .reshape(bs, A, P, cams, S, C)
    )
    out = (val * w_exp).sum(dim=(2, 3, 4))  # [bs, A, C]
    return out
