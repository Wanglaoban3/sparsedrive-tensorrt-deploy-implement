"""Self-test for the pure-torch DeformableAggregation port.

Checks (all runnable without the compiled CUDA extension):
  1. vectorized port vs a 1:1 python-loop transcription of
     deformable_aggregation_cuda.cu  (random shapes, incl. boundary locs)
  2. sanity vs F.grid_sample for a single-cam/level, single-group case
  3. backward smoke test (grads flow to feature / sampling_location / weights)

Usage: python deploy/test_dfa_pure.py
"""

import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPS_DIR = os.path.join(ROOT, "projects", "mmdet3d_plugin", "ops")

spec = importlib.util.spec_from_file_location(
    "sdops", os.path.join(OPS_DIR, "__init__.py"),
    submodule_search_locations=[OPS_DIR],
)
sdops = importlib.util.module_from_spec(spec)
sys.modules["sdops"] = sdops
spec.loader.exec_module(sdops)

deformable_aggregation_pure = sdops.deformable_aggregation_pure
deformable_aggregation_function = sdops.deformable_aggregation_function


def naive_kernel_reference(mc_ms_feat, spatial_shape, scale_start_index,
                           sampling_location, weights):
    """1:1 transcription of the CUDA kernel loops (float32, python loops)."""
    mc_ms_feat = mc_ms_feat.contiguous().float()
    bs, num_feat, C = mc_ms_feat.shape
    A = weights.shape[1]
    P = weights.shape[2]
    cams, S = spatial_shape.shape[:2]
    G = weights.shape[-1]
    cg = C // G

    out = torch.zeros(bs, A, C)
    for b in range(bs):
        for a in range(A):
            for ch in range(C):
                g = ch // cg
                acc = 0.0
                for c in range(cams):
                    for s in range(S):
                        for p in range(P):
                            loc_w = float(sampling_location[b, a, p, c, 0])
                            loc_h = float(sampling_location[b, a, p, c, 1])
                            if not (0.0 < loc_w < 1.0 and 0.0 < loc_h < 1.0):
                                continue
                            h = int(spatial_shape[c, s, 0])
                            w = int(spatial_shape[c, s, 1])
                            start = int(scale_start_index[c, s])
                            h_im = loc_h * h - 0.5
                            w_im = loc_w * w - 0.5
                            h_low, w_low = int(h_im // 1), int(w_im // 1)
                            lh, lw = h_im - h_low, w_im - w_low
                            hh, hw = 1 - lh, 1 - lw
                            corners = (
                                (0, 0, hh * hw), (0, 1, hh * lw),
                                (1, 0, lh * hw), (1, 1, lh * lw),
                            )
                            val = 0.0
                            for dh, dw, cw in corners:
                                h_c, w_c = h_low + dh, w_low + dw
                                if 0 <= h_c <= h - 1 and 0 <= w_c <= w - 1:
                                    val += float(
                                        mc_ms_feat[b, start + h_c * w + w_c, ch]
                                    ) * cw
                            acc += val * float(weights[b, a, p, c, s, g])
                out[b, a, ch] = acc
    return out


def make_inputs(bs=2, A=5, P=6, cams=2, S=2, C=8, G=2, seed=0, boundary=True):
    gen = torch.Generator().manual_seed(seed)
    shapes = [(4, 5), (3, 4)] * 2  # per (cam, scale) interleaved
    spatial_shape = torch.tensor(shapes, dtype=torch.int64).reshape(cams, S, 2)
    sizes = [int(spatial_shape[c, s, 0]) * int(spatial_shape[c, s, 1])
             for c in range(cams) for s in range(S)]
    ssi, tot = [], 0
    for n in sizes:
        ssi.append(tot)
        tot += n
    scale_start_index = torch.tensor(ssi, dtype=torch.int64).reshape(cams, S)
    feat = torch.randn(bs, tot, C, generator=gen) * 3
    loc = torch.rand(bs, A, P, cams, 2, generator=gen)
    if boundary:  # sprinkle exactly-0 / exactly-1 / out-of-range locs
        loc.view(-1)[::7] = 0.0
        loc.view(-1)[3::11] = 1.0
        loc.view(-1)[5::13] = -0.3
        loc.view(-1)[9::17] = 1.4
    w = torch.rand(bs, A, P, cams, S, G, generator=gen)
    w = w / w.sum(dim=(2, 3, 4), keepdim=True)  # softmax-like, sums to 1
    return feat, spatial_shape, scale_start_index, loc, w


def test_vs_naive():
    for seed, boundary in [(0, True), (1, False), (2, True)]:
        feat, shape, ssi, loc, w = make_inputs(seed=seed, boundary=boundary)
        got = deformable_aggregation_pure(feat, shape, ssi, loc, w)
        ref = naive_kernel_reference(feat, shape, ssi, loc, w)
        diff = (got - ref).abs().max().item()
        scale = ref.abs().max().item()
        assert diff < 1e-4, f"seed={seed} boundary={boundary}: max diff {diff} (scale {scale})"
        print(f"  vs-naive seed={seed} boundary={boundary}: max|diff|={diff:.3e} ok")


def test_vs_grid_sample():
    # single cam, single level, G=1, unit weights: DFA == grid_sample
    # bilinear align_corners=False with grid = loc*2-1
    H, W, C = 6, 7, 4
    feat = torch.randn(1, H * W, C)
    fm = feat.view(1, H, W, C).permute(0, 3, 1, 2)  # [1,C,H,W]
    A, P = 9, 5
    loc = torch.rand(1, A, P, 1, 2) * 0.9 + 0.05  # strictly interior
    weights = torch.ones(1, A, P, 1, 1, 1)
    shape = torch.tensor([[[H, W]]])
    ssi = torch.tensor([[0]])
    got = deformable_aggregation_pure(feat, shape, ssi, loc, weights)
    grid = (loc * 2 - 1).reshape(1, A * P, 1, 2)
    ref = torch.nn.functional.grid_sample(
        fm, grid, align_corners=False, padding_mode="zeros")
    ref = ref[..., 0].permute(0, 2, 1).reshape(1, A, P, C).sum(dim=2)
    diff = (got - ref).abs().max().item()
    assert diff < 1e-4, f"vs-grid_sample max diff {diff}"
    print(f"  vs-grid_sample: max|diff|={diff:.3e} ok")


def test_backward():
    feat, shape, ssi, loc, w = make_inputs(seed=3, boundary=True)
    feat = feat.requires_grad_(True)
    loc = loc.requires_grad_(True)
    w = w.requires_grad_(True)
    # pure impl is called directly under grad-enabled through the dispatcher
    out = deformable_aggregation_function(feat, shape, ssi, loc, w)
    assert out.shape == (2, 5, 8)
    out.sum().backward()
    for name, t in [("feat", feat), ("loc", loc), ("weights", w)]:
        assert t.grad is not None and torch.isfinite(t.grad).all(), name
    print("  backward: grads finite, shapes ok")


if __name__ == "__main__":
    print("use_pure =", sdops._use_pure())
    test_vs_naive()
    test_vs_grid_sample()
    test_backward()
    print("DFA_PURE_ALL_OK")
