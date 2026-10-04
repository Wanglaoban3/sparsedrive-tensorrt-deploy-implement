# -*- coding: utf-8 -*-
"""FP32 PyTorch (ckpt/sparsedrive_stage2.pth) over the 81 mini frames, dumped
in the ENGINE dump format (map_cls/map_pts/det_cls/det_bbox/det_quality), so
the standalone map eval and the det decode can consume FP32 and engine dumps
interchangeably.

Feeding contract mirrors tools/test_trt.py: scene-boundary reset
(dt > 2.0 or dt < 0), t_matrix = curr_global_inv @ prev_global_mat,
first-frame zero history + dt = 0.5.

Run with the sparsedrive_deploy env python (torch cu118 + mmcv 1.x):
    H:/miniconda3/envs/sparsedrive_deploy/python.exe deploy/infer_sp32_mini.py
"""
import copy
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "deploy"))
sys.path.append(os.path.join(ROOT, "tools"))
os.chdir(ROOT)

from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmcv.parallel.scatter_gather import scatter  # noqa: E402
from mmdet.models import build_detector  # noqa: E402
from mmdet.datasets import build_dataloader  # noqa: E402

import projects.mmdet3d_plugin  # noqa: E402,F401
from projects.mmdet3d_plugin.datasets.nuscenes_3d_dataset import (  # noqa
    NuScenes3DDataset,
)
from export_onnx_det_map import SparseDriveONNXWrapper  # noqa: E402

OUT = os.path.join(ROOT, "work_dirs", "sparsedrive_small_stage2",
                   "evaldata", "mini_sp32")
CKPT = os.path.join(ROOT, "ckpt", "sparsedrive_stage2.pth")


def zero_det(nh, dim):
    return {
        "prev_det_feat": torch.zeros(1, nh, 256, device="cuda"),
        "prev_det_anchor": torch.zeros(1, nh, dim, device="cuda"),
        "prev_det_conf": torch.zeros(1, nh, device="cuda"),
        "prev_det_id": torch.full((1, nh), -1, dtype=torch.int32,
                                  device="cuda"),
        "prev_id_count": torch.zeros(1, 1, dtype=torch.int32, device="cuda"),
    }


def zero_map(nh, dim):
    return {
        "prev_map_feat": torch.zeros(1, nh, 256, device="cuda"),
        "prev_map_anchor": torch.zeros(1, nh, dim, device="cuda"),
        "prev_map_conf": torch.zeros(1, nh, device="cuda"),
    }


def main():
    cfg = Config.fromfile("projects/configs/sparsedrive_small_stage2.py")
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    cfg.task_config["with_det"] = True
    cfg.task_config["with_map"] = True
    cfg.task_config["with_motion_plan"] = False
    cfg.model.head.task_config = cfg.task_config

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, CKPT, map_location="cpu")
    model.cuda().eval()
    wrapper = SparseDriveONNXWrapper(model)

    ds_cfg = copy.deepcopy(cfg.data.val)
    ds_cfg["test_mode"] = True
    ds_cfg.pop("type", None)
    ds = NuScenes3DDataset(**ds_cfg)
    loader = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=0,
                              dist=False, shuffle=False)
    n = len(ds)
    print("dataset:", n, "samples", flush=True)

    det_head = model.head.det_head
    map_head = model.head.map_head
    nh_det = det_head.instance_bank.num_temp_instances
    dim_det = det_head.instance_bank.anchor.shape[-1]
    nh_map = map_head.instance_bank.num_temp_instances
    dim_map = map_head.instance_bank.anchor.shape[-1]
    print("nh_det", nh_det, "dim_det", dim_det,
          "nh_map", nh_map, "dim_map", dim_map, flush=True)

    hist_det = zero_det(nh_det, dim_det)
    hist_map = zero_map(nh_map, dim_map)
    prev_global, prev_time = None, None

    tokens = []
    os.makedirs(OUT, exist_ok=True)
    for k, data in enumerate(loader):
        with torch.no_grad():
            d = scatter(data, [torch.cuda.current_device()])[0]
            img = d["img"]
            proj_mat = d["projection_mat"]
            metas = d["img_metas"][0]
            curr_time = metas["timestamp"]
            if prev_time is None:
                dt, scene_start = 0.5, True
            else:
                dt = curr_time - prev_time
                scene_start = (dt > 2.0 or dt < 0)
            if scene_start:
                dt = 0.5
                hist_det = zero_det(nh_det, dim_det)
                hist_map = zero_map(nh_map, dim_map)
                prev_global = None
            dt_t = torch.tensor([dt], device="cuda", dtype=torch.float32)
            g = metas["T_global"]
            g_inv = metas["T_global_inv"]
            if prev_global is None:
                t_mat = torch.eye(4, device="cuda").unsqueeze(0)
            else:
                t_mat = torch.from_numpy(
                    np.asarray(g_inv) @ np.asarray(prev_global)
                ).float().cuda().unsqueeze(0)
            prev_global = g
            prev_time = curr_time

            outs = wrapper(img, proj_mat,
                           hist_det["prev_det_feat"], hist_det["prev_det_anchor"],
                           hist_det["prev_det_conf"], hist_det["prev_det_id"],
                           hist_det["prev_id_count"],
                           hist_map["prev_map_feat"], hist_map["prev_map_anchor"],
                           hist_map["prev_map_conf"], t_mat, dt_t)
            hist_det = {
                "prev_det_feat": outs[6], "prev_det_anchor": outs[7],
                "prev_det_conf": outs[8], "prev_det_id": outs[9],
                "prev_id_count": outs[10],
            }
            hist_map = {
                "prev_map_feat": outs[15], "prev_map_anchor": outs[16],
                "prev_map_conf": outs[17],
            }
            map_cls, map_pts = outs[11], outs[12]
            assert torch.isfinite(map_cls).all() and \
                torch.isfinite(map_pts).all(), "non-finite map outputs"

            od = os.path.join(OUT, "out_%02d" % k)
            os.makedirs(od, exist_ok=True)
            for name, t in (("map_cls", outs[11]), ("map_pts", outs[12]),
                            ("det_cls", outs[0]), ("det_bbox", outs[1]),
                            ("det_quality", outs[2])):
                t.detach().cpu().numpy().astype(np.float32).tofile(
                    os.path.join(od, name + ".bin"))
            tokens.append(ds.data_infos[k]["token"])
            if (k + 1) % 10 == 0 or k == n - 1:
                smax = map_cls.sigmoid().max().item()
                print("frame %2d/%d scene_start=%d map_sig_max=%.4f"
                      % (k + 1, n, scene_start, smax), flush=True)

    np.savez(os.path.join(OUT, "mini_meta.npz"), tokens=np.array(tokens))
    print("dumped", n, "frames ->", OUT, flush=True)


if __name__ == "__main__":
    main()
