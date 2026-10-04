# -*- coding: utf-8 -*-
"""make_nv12_manifest.py — build the R0 replay set: mini-val 81 frames x 6 cams
JPG -> NV12 (BT.601 full-range, 4:2:0) + manifest.jsonl for sp_filesrc.

Manifest layout (root = out dir):
  {"w":1600,"h":900}                      header line
  {"frame":k,"scene":s,"ts_ns":...,
   "cams":["frame_00/cam_0.nv12", ...]}   cam order == infos pkl dict order
  (== dataset order: FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT,
   BACK_RIGHT)

Scene id: dt > 2s (or first frame) starts a new scene — same rule as the
inference chain. ts_ns = infos timestamp (us) * 1000.

Usage:
  H:/miniconda3/envs/sparsedrive_deploy/python.exe tools/make_nv12_manifest.py \
      [--frames N] [--out work_dirs/nv12_r0]
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFO_PKL = os.path.join(ROOT, "data", "infos", "mini",
                        "nuscenes_infos_val.pkl")

# BT.601 full-range RGB -> YCbCr, then 4:2:0 subsample by 2x2 mean.
_MY = np.array([0.299, 0.587, 0.114], dtype=np.float32)
_MU = np.array([-0.168736, -0.331264, 0.5], dtype=np.float32)
_MV = np.array([0.5, -0.418688, -0.081312], dtype=np.float32)


def rgb_to_nv12(rgb):
    """rgb: (H,W,3) uint8 -> NV12 bytes (Y plane then interleaved UV)."""
    f = rgb.astype(np.float32)
    y = f @ _MY
    u = f @ _MU + 128.0
    v = f @ _MV + 128.0
    h, w = y.shape
    if h % 2 or w % 2:
        raise ValueError("odd dims %dx%d" % (h, w))
    u2 = u.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    v2 = v.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    y8 = np.clip(y + 0.5, 0, 255).astype(np.uint8)
    uv = np.stack([u2, v2], axis=-1)
    uv8 = np.clip(uv + 0.5, 0, 255).astype(np.uint8)
    return y8.tobytes() + uv8.tobytes()


def norm_path(p):
    """infos data_path embeds an absolute repo root with mixed separators;
    normalize to a path relative to the repo root with forward slashes."""
    p = os.path.normpath(p)
    try:
        rel = os.path.relpath(p, ROOT)
    except ValueError:
        rel = p
    return rel.replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("work_dirs", "nv12_r0"))
    ap.add_argument("--frames", type=int, default=0, help="0 = all 81")
    args = ap.parse_args()

    with open(INFO_PKL, "rb") as f:
        infos = pickle.load(f)
    lst = infos["infos"] if isinstance(infos, dict) else infos
    n = len(lst) if args.frames <= 0 else min(args.frames, len(lst))
    out_root = args.out if os.path.isabs(args.out) else \
        os.path.join(ROOT, args.out)
    os.makedirs(out_root, exist_ok=True)

    lines = []
    prev_ts = None
    scene = 0
    for k in range(n):
        info = lst[k]
        ts_us = info["timestamp"]
        if prev_ts is None:
            pass
        elif ts_us - prev_ts < 0 or (ts_us - prev_ts) / 1e6 > 2.0:
            scene += 1
        prev_ts = ts_us
        cam_types = list(info["cams"].keys())
        if k == 0:
            print("cam order:", cam_types)
        cams_rel = []
        for c, ct in enumerate(cam_types):
            jpg = os.path.join(ROOT, norm_path(info["cams"][ct]["data_path"]))
            img = Image.open(jpg).convert("RGB")
            if img.size != (1600, 900):
                raise ValueError("%s is %s, expected 1600x900"
                                 % (jpg, img.size))
            d = os.path.join(out_root, "frame_%02d" % k)
            os.makedirs(d, exist_ok=True)
            nv12 = os.path.join(d, "cam_%d.nv12" % c)
            with open(nv12, "wb") as fp:
                fp.write(rgb_to_nv12(np.asarray(img)))
            cams_rel.append("frame_%02d/cam_%d.nv12" % (k, c))
        lines.append(json.dumps({
            "frame": k, "scene": scene,
            "ts_ns": int(ts_us) * 1000, "cams": cams_rel}))
        if (k + 1) % 10 == 0 or k + 1 == n:
            print("converted %d/%d (scene %d)" % (k + 1, n, scene),
                  flush=True)

    hdr = {"w": 1600, "h": 900}
    with open(os.path.join(out_root, "manifest.jsonl"), "w") as fp:
        fp.write(json.dumps(hdr) + "\n")
        fp.write("\n".join(lines) + "\n")
    print("wrote manifest +%d frames -> %s" % (n, out_root))


if __name__ == "__main__":
    sys.exit(main())
