# -*- coding: utf-8 -*-
"""preproc_ref.py — numpy/PIL reference for the M2 CUDA preproc.

Reference numerics (= the dataset pipeline that produced 链路二 in_XX):
  cv2 jpg decode (BGR u8) → PIL resize((704,396)) [default BICUBIC]
  → crop((0,140,704,396)) → float32 → bgr2rgb → (x-mean)/std
  mean=[123.675,116.28,103.53] std=[58.395,57.12,57.375]  (RGB order)

Our source is NV12 (BT.601 full-range) instead of jpg: nv12→rgb u8 replaces
the jpg decode; everything after is identical.

Modes:
  --selftest        bit-exact check: my Pillow-coefficient replication vs
                    PIL.resize on random u8 (must be 100% identical)
  --frame K         build ref for R0 frame K from NV12 → save
                    work_dirs/preproc_ref/ref_K.npy  ([1,6,3,256,704] f32)
  --vs-pipeline K   additionally run the REAL dataset pipeline for frame K
                    and report pixel diff vs the NV12-based ref (source floor)
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R0 = os.path.join(ROOT, "work_dirs", "nv12_r0")
OUT = os.path.join(ROOT, "work_dirs", "preproc_ref")

MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
RESIZE_DIMS = (704, 396)   # PIL (w,h)
CROP = (0, 140, 704, 396)  # PIL (l,t,r,b)
FW, FH = 704, 256


# ---------- Pillow resize coefficient replication (for the CUDA LUT) ----------
# Exact port of Pillow 12.3 Resample.c precompute_coeffs (8bpc fixed-point
# path): filterscale = in/out (>=1); support = 2*filterscale (bicubic);
# center = (xx+0.5)*scale (scale = in/out even when <1!);
# xmin = (int)(center-support+0.5)  [C trunc toward 0!], clipped >=0;
# xmax = (int)(center+support+0.5), clipped <=in; taps = [xmin, xmax);
# w = cubic((x - center + 0.5) / filterscale), normalized by sum.
# 8bpc: kint = round_half_away(k * 2^22); ss = (2^21 + sum kint*u8) >> 22;
# clip [0,255]. PRECISION_BITS = 22.

_PREC = 22


def _pil_cubic(x):
    a = -0.5
    x = abs(x)
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return 0.0


def _trunc(v):
    return int(v)  # C (int) cast: trunc toward zero (v is float)


def pil_coeffs_int(in_size, out_size):
    """Per-output (xmin, kint[]) exactly as Pillow 8bpc.
    kint: int32 fixed-point coefficients (scale 2^22)."""
    scale = in_size / out_size
    filterscale = scale if scale >= 1.0 else 1.0
    support = 2.0 * filterscale
    ksize = int(np.ceil(support)) * 2 + 1
    inv_fs = 1.0 / filterscale
    out = []
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = _trunc(center - support + 0.5)
        if xmin < 0:
            xmin = 0
        xmax = _trunc(center + support + 0.5)
        if xmax > in_size:
            xmax = in_size
        cnt = xmax - xmin
        ws = np.array([_pil_cubic((x + xmin - center + 0.5) * inv_fs)
                       for x in range(cnt)], np.float64)
        ww = ws.sum()
        if ww != 0.0:
            ws = ws / ww
        kf = ws * (1 << _PREC)
        kint = np.where(kf < 0, np.trunc(kf - 0.5), np.trunc(kf + 0.5))
        kint = np.concatenate([kint, np.zeros(ksize - cnt)])
        out.append((xmin, cnt, kint.astype(np.int64)))
    return out, ksize


def my_resize_u8(img_u8, dims):
    """Bit-exact Pillow BICUBIC u8 resize (fixed-point integer pipeline).
    img: HxWxC u8 → dims (w,h)."""
    h, w, c = img_u8.shape
    ow, oh = dims
    co_x, _ = pil_coeffs_int(w, ow)
    co_y, _ = pil_coeffs_int(h, oh)
    tmp = np.empty((h, ow, c), np.uint8)
    for xx in range(ow):
        xmin, cnt, kint = co_x[xx]
        acc = np.zeros((h, c), np.int64)
        for x in range(cnt):
            acc += np.int64(kint[x]) * img_u8[:, xmin + x, :].astype(np.int64)
        ss = (acc + (1 << (_PREC - 1))) >> _PREC
        tmp[:, xx, :] = np.clip(ss, 0, 255).astype(np.uint8)
    out = np.empty((oh, ow, c), np.uint8)
    for yy in range(oh):
        ymin, cnt, kint = co_y[yy]
        acc = np.zeros((ow, c), np.int64)
        for y in range(cnt):
            acc += np.int64(kint[y]) * tmp[ymin + y, :, :].astype(np.int64)
        ss = (acc + (1 << (_PREC - 1))) >> _PREC
        out[yy, :, :] = np.clip(ss, 0, 255).astype(np.uint8)
    return out


def selftest():
    rng = np.random.RandomState(7)
    ok_all = True
    for (h, w, ow, oh) in [(900, 1600, 704, 396), (100, 150, 37, 51),
                           (64, 64, 128, 128)]:
        img = rng.randint(0, 256, (h, w, 3)).astype(np.uint8)
        ref = np.asarray(Image.fromarray(img).resize((ow, oh)))
        mine = my_resize_u8(img, (ow, oh))
        same = (ref == mine).all()
        ok_all &= bool(same)
        print("selftest %dx%d -> %dx%d: %s (diff px=%d)"
              % (w, h, ow, oh, "OK" if same else "FAIL",
                 int((ref != mine).sum())))
    return 0 if ok_all else 1


# ---------- NV12 → normalized [1,6,3,256,704] ----------

def nv12_to_rgb_u8(nv12, w, h):
    y = nv12[:w * h].reshape(h, w).astype(np.float32)
    uv = nv12[w * h:].reshape(h // 2, w // 2, 2).astype(np.float32)
    u = np.repeat(np.repeat(uv[:, :, 0], 2, axis=0), 2, axis=1) - 128.0
    v = np.repeat(np.repeat(uv[:, :, 1], 2, axis=0), 2, axis=1) - 128.0
    r = y + 1.402 * v
    g = y - 0.344136 * u - 0.714136 * v
    b = y + 1.772 * u
    return np.clip(np.stack([r, g, b], axis=-1) + 0.5, 0,
                   255).astype(np.uint8)  # HWC RGB


def frame_ref(k):
    with open(os.path.join(R0, "manifest.jsonl")) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    hdr = json.loads(lines[0])
    rec = json.loads(lines[1 + k])
    w, h = hdr["w"], hdr["h"]
    cams = []
    for c, rel in enumerate(rec["cams"]):
        nv = np.fromfile(os.path.join(R0, rel.replace("/", os.sep)),
                         dtype=np.uint8)
        rgb = nv12_to_rgb_u8(nv, w, h)
        img = Image.fromarray(rgb).resize(RESIZE_DIMS).crop(CROP)
        x = np.asarray(img).astype(np.float32)  # HWC RGB (source was RGB)
        x = (x - MEAN) / STD
        cams.append(x.transpose(2, 0, 1))      # CHW
    return np.ascontiguousarray(np.stack(cams)[None])  # [1,6,3,256,704]


def vs_pipeline(k):
    """run the real dataset pipeline for frame k; compare vs NV12-based ref."""
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "deploy"))
    import copy
    from mmcv import Config
    import projects.mmdet3d_plugin  # noqa
    from projects.mmdet3d_plugin.datasets.nuscenes_3d_dataset import (
        NuScenes3DDataset)
    cfg = Config.fromfile("projects/configs/sparsedrive_small_stage2.py")
    ds_cfg = copy.deepcopy(cfg.data.val)
    ds_cfg["test_mode"] = True
    ds_cfg.pop("type", None)
    ds = NuScenes3DDataset(**ds_cfg)
    item = ds[k]
    img = item["img"]
    if hasattr(img, "data"):
        img = img.data
    if isinstance(img, (list, tuple)):
        img = img[0]
    ref_pipe = np.asarray(img, np.float32).reshape(1, 6, 3, 256, 704)
    ref_nv = frame_ref(k)
    d = np.abs(ref_pipe - ref_nv)
    print("frame %d: pipeline-vs-nv12ref MAE=%.4f max=%.3f (source floor)"
          % (k, d.mean(), d.max()))
    return ref_pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--frame", type=int, nargs="*", default=[])
    ap.add_argument("--vs-pipeline", type=int, nargs="*", default=[])
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    os.makedirs(OUT, exist_ok=True)
    for k in args.frame:
        a = frame_ref(k)
        np.save(os.path.join(OUT, "ref_%02d.npy" % k), a)
        print("frame %d: ref saved, mean=%.4f std=%.4f" % (k, a.mean(),
                                                           a.std()))
    for k in args.vs_pipeline:
        vs_pipeline(k)


if __name__ == "__main__":
    main()
