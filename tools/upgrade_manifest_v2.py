# -*- coding: utf-8 -*-
"""upgrade_manifest_v2.py — upgrade the R0 replay manifest to v2 in place.

v2 adds (all float64, computed exactly like the dataset pipeline):
  header: {"w":1600,"h":900,"aug":{"resize":r,"crop":[x0,y0,x1,y1]},
           "norm":{"mean":[...],"std":[...]}}           RGB-order mean/std
  frame:  "l2i":[6][4][4] lidar2img (pre-ida, row-major, float64)
          "l2g":[4][4]     lidar2global (float64)

NV12 files are untouched. The file source ignores unknown keys, so v2 stays
compatible with sp_filesrc; sp_modelnode uses the new fields for P2 + t_matrix.

Usage:
  H:/miniconda3/envs/sparsedrive_deploy/python.exe tools/upgrade_manifest_v2.py \
      [--manifest work_dirs/nv12_r0/manifest.jsonl]
"""
import argparse
import copy
import json
import os
import pickle
import sys

import numpy as np
import pyquaternion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFO_PKL = os.path.join(ROOT, "data", "infos", "mini",
                        "nuscenes_infos_val.pkl")

# test-mode aug (nuscenes_3d_dataset.get_augmentation, else-branch):
#   resize = max(fH/H, fW/W); crop_h = int(newH) - fH; crop_w = 0
FW, FH = 704, 256
SRC_W, SRC_H = 1600, 900
MEAN = [123.675, 116.28, 103.53]   # RGB order (mmcv imnormalize to_rgb=True)
STD = [58.395, 57.12, 57.375]


def compute_aug():
    resize = max(FH / SRC_H, FW / SRC_W)
    rw, rh = int(SRC_W * resize), int(SRC_H * resize)
    crop_h = int((1 - 0.0) * rh) - FH
    crop_w = int(max(0, rw - FW) / 2)
    return resize, [crop_w, crop_h, crop_w + FW, crop_h + FH]


def lidar2img_of(cam_info):
    """replica of nuscenes_3d_dataset.get_data_info camera block (float64)."""
    r = np.asarray(cam_info["sensor2lidar_rotation"], dtype=np.float64)
    lidar2cam_r = np.linalg.inv(r)
    t = np.asarray(cam_info["sensor2lidar_translation"], dtype=np.float64)
    lidar2cam_t = t @ lidar2cam_r.T
    rt = np.eye(4, dtype=np.float64)
    rt[:3, :3] = lidar2cam_r.T
    rt[3, :3] = -lidar2cam_t
    intrinsic = copy.deepcopy(np.asarray(cam_info["cam_intrinsic"],
                                         dtype=np.float64))
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    return (viewpad @ rt.T).tolist()


def lidar2global_of(info):
    l2e = np.eye(4, dtype=np.float64)
    l2e[:3, :3] = pyquaternion.Quaternion(
        info["lidar2ego_rotation"]).rotation_matrix
    l2e[:3, 3] = np.array(info["lidar2ego_translation"])
    e2g = np.eye(4, dtype=np.float64)
    e2g[:3, :3] = pyquaternion.Quaternion(
        info["ego2global_rotation"]).rotation_matrix
    e2g[:3, 3] = np.array(info["ego2global_translation"])
    return (e2g @ l2e).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default=os.path.join("work_dirs", "nv12_r0",
                                         "manifest.jsonl"))
    args = ap.parse_args()
    mpath = args.manifest if os.path.isabs(args.manifest) else \
        os.path.join(ROOT, args.manifest)

    with open(INFO_PKL, "rb") as f:
        infos = pickle.load(f)
    lst = infos["infos"] if isinstance(infos, dict) else infos

    with open(mpath) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    header = json.loads(lines[0])
    frames = [json.loads(l) for l in lines[1:]]

    resize, crop = compute_aug()
    header["aug"] = {"resize": resize, "crop": crop}
    header["norm"] = {"mean": MEAN, "std": STD}
    print("aug:", header["aug"])

    out = [json.dumps(header)]
    for rec in frames:
        info = lst[rec["frame"]]
        cams = list(info["cams"].keys())
        assert len(cams) == 6
        rec["l2i"] = [lidar2img_of(info["cams"][ct]) for ct in cams]
        rec["l2g"] = lidar2global_of(info)
        out.append(json.dumps(rec))

    tmp = mpath + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(out) + "\n")
    os.replace(tmp, mpath)
    print("upgraded %d frames -> %s" % (len(frames), mpath))


if __name__ == "__main__":
    sys.exit(main())
