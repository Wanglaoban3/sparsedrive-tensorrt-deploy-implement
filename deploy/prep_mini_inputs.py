# -*- coding: utf-8 -*-
"""nuScenes mini -> per-sample e_T3 engine inputs (real images, chained
temporal metadata), saved in run_engine vec format + uploaded to board.

Outputs work_dirs/sparsedrive_small_stage2/evaldata/mini_eng/in_XX/{12 bins,
manifest.tsv} + mini_meta.npz (tokens/timestamps/l2g for the decode step).
Only sample 0 gets zero state files; later samples get prev_* on board via cp.
"""
import copy
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "deploy"))
os.chdir(ROOT)

from mmcv import Config  # noqa: E402
import projects.mmdet3d_plugin  # noqa: E402,F401
from projects.mmdet3d_plugin.datasets.nuscenes_3d_dataset import (  # noqa
    NuScenes3DDataset,
)
import pyquaternion  # noqa: E402

CFG = "projects/configs/sparsedrive_small_stage2.py"
BASE = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2")
OUT = os.path.join(BASE, "evaldata", "mini_eng")
N_DET, N_MAP, EMB = 600, 33, 256

cfg = Config.fromfile(CFG)
if cfg.get("custom_imports", None):
    from mmcv.utils import import_modules_from_strings
    import_modules_from_strings(**cfg["custom_imports"])
ds_cfg = copy.deepcopy(cfg.data.val)
ds_cfg["test_mode"] = True
ds_cfg.pop("type", None)
ds = NuScenes3DDataset(**ds_cfg)
N = len(ds)
print("dataset:", N, "samples", "version", ds.version)


def l2g_of(info):
    l2e = np.eye(4, dtype=np.float64)
    l2e[:3, :3] = pyquaternion.Quaternion(
        info["lidar2ego_rotation"]).rotation_matrix
    l2e[:3, 3] = np.array(info["lidar2ego_translation"])
    e2g = np.eye(4, dtype=np.float64)
    e2g[:3, :3] = pyquaternion.Quaternion(
        info["ego2global_rotation"]).rotation_matrix
    e2g[:3, 3] = np.array(info["ego2global_translation"])
    return e2g @ l2e


def save_dir(idx, tensors):
    d = os.path.join(OUT, "in_%02d" % idx)
    os.makedirs(d, exist_ok=True)
    lines = []
    for nm, a in tensors.items():
        dt = "i32" if a.dtype == np.int32 else "f32"
        fn = nm + ".bin"
        a.tofile(os.path.join(d, fn))
        lines.append("\t".join([nm, dt, ",".join(map(str, a.shape)), fn]))
    with open(os.path.join(d, "manifest.tsv"), "w", newline="") as f:
        f.write("\n".join(lines) + "\n")


NAMES = ["img", "projection_mat", "prev_det_feat", "prev_det_anchor",
         "prev_det_conf", "prev_det_id", "prev_id_count", "prev_map_feat",
         "prev_map_anchor", "prev_map_conf", "instance_t_matrix",
         "time_interval"]

tokens, tss, boundary = [], [], []
prev_Tg = None
zero_state = {
    "prev_det_feat": np.zeros((1, N_DET, EMB), np.float32),
    "prev_det_anchor": np.zeros((1, N_DET, 11), np.float32),
    "prev_det_conf": np.zeros((1, N_DET), np.float32),
    "prev_det_id": np.full((1, N_DET), -1, np.int32),
    "prev_id_count": np.zeros((1, 1), np.int32),
    "prev_map_feat": np.zeros((1, N_MAP, EMB), np.float32),
    "prev_map_anchor": np.zeros((1, N_MAP, 40), np.float32),
    "prev_map_conf": np.zeros((1, N_MAP), np.float32),
}

for i in range(N):
    item = ds[i]
    img = item["img"]
    if hasattr(img, "data"):
        img = img.data
    if isinstance(img, (list, tuple)):
        img = img[0]
    img = torch.as_tensor(img).float().contiguous()
    if img.dim() == 4:
        img = img[None]
    metas = item["img_metas"]
    if hasattr(metas, "data"):
        metas = metas.data
    if isinstance(metas, (list, tuple)):
        metas = metas[0]
    pmat = np.asarray(item["projection_mat"], np.float32).reshape(6, 4, 4)

    ts = float(item["timestamp"]) if "timestamp" in item \
        else float(metas.get("timestamp", ds.data_infos[i]["timestamp"] / 1e6))
    # scene boundary (pytorch mask = |dt| <= max_time_interval=2 -> False):
    # reset state + default dt, exactly the semantics pytorch applies
    if i == 0 or abs(ts - tss[-1]) > 5.0:
        tm = np.eye(4, dtype=np.float32)[None]
        dt = np.float32(0.5)
        tensors = {
            "img": img.numpy().astype(np.float32),
            "projection_mat": pmat[None],
            "instance_t_matrix": tm,
            "time_interval": np.array([dt], np.float32),
        }
        tensors.update(zero_state)
        boundary.append(i)
    else:
        tm = (metas["T_global_inv"] @ prev_Tg).astype(np.float32)[None]
        dt = np.float32(ts - tss[-1])
        tensors = {
            "img": img.numpy().astype(np.float32),
            "projection_mat": pmat[None],
            "instance_t_matrix": tm,
            "time_interval": np.array([dt], np.float32),
        }
    save_dir(i, tensors)
    tokens.append(ds.data_infos[i]["token"])
    tss.append(ts)
    prev_Tg = np.asarray(metas["T_global"], np.float64)
    if (i + 1) % 10 == 0 or i == N - 1:
        print(f"prep {i+1}/{N} dt={dt:.3f}s", flush=True)

print("boundaries:", boundary)
np.savez(os.path.join(OUT, "mini_meta.npz"),
         tokens=np.array(tokens), ts=np.array(tss),
         boundary=np.array(boundary))
print("MINI_PREP_DONE", N)
